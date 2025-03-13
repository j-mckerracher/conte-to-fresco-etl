import signal
import sys
import pandas as pd
import numpy as np
import re
import json
import os
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import time
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import gc
from tqdm import tqdm
from queue import Queue
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag to indicate termination
terminate_requested = False

# Define file paths
CACHE_DIR = Path("./cache")
JOB_ACCOUNTING_PATH = Path(CACHE_DIR / 'accounting')  # FIXED: Removed 'input/' prefix
PROC_METRIC_PATH = Path(CACHE_DIR / 'input/metrics')
OUTPUT_PATH = Path(CACHE_DIR / 'output')

# Ensure directories exists
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
# We don't create input directories as they should already exist with data

# Configuration
MAX_WORKERS = 1  # Keep at 1 for better memory management
MIN_FREE_MEMORY_GB = 2.0  # Lower this to be more aggressive with memory management
MIN_FREE_DISK_GB = 5.0  # Keep as is
BASE_CHUNK_SIZE = 50_000  # Reduce from 100,000 to 50,000 for smaller chunks
MAX_MEMORY_USAGE_GB = 25.0  # Lower the limit from 30GB to 25GB for more headroom
MEMORY_CHECK_INTERVAL = 0.1  # Check memory usage frequently

# Force pandas to use specific memory optimization settings
pd.options.mode.use_inf_as_na = True  # Convert inf to NaN (saves memory in some cases)
pd.options.mode.string_storage = 'python'  # Use Python's memory-efficient string storage

# Disable the SettingWithCopyWarning which can slow things down
pd.options.mode.chained_assignment = None


# Register the signal handler for various signals
def setup_signal_handlers():
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
    return memory_gb


def get_available_memory():
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024 * 1024 * 1024)


def get_free_disk_space():
    """Get free disk space in GB"""
    _, _, free = shutil.disk_usage("/")
    return free / (1024 * 1024 * 1024)


def calculate_chunk_size(current_size=BASE_CHUNK_SIZE):
    """More aggressive dynamic chunk size calculation based on available memory"""
    available_memory_gb = get_available_memory()
    memory_usage_gb = get_memory_usage()

    # Add more aggressive memory-based adjustment
    if available_memory_gb < MIN_FREE_MEMORY_GB * 0.5:  # Critical memory situation
        # Drastically reduce chunk size
        return max(5_000, current_size // 4)
    elif available_memory_gb < MIN_FREE_MEMORY_GB:
        # Significantly reduce chunk size
        return max(10_000, current_size // 2)
    elif available_memory_gb > MIN_FREE_MEMORY_GB * 4:
        # Cautiously increase chunk size
        return min(300_000, current_size * 1.5)  # Less aggressive increase

    # If memory usage is already high, be more conservative
    if memory_usage_gb > 40.0:  # If using more than 40GB
        return max(10_000, current_size // 2)

    return current_size


def convert_walltime_to_seconds(walltime_series):
    """Vectorized conversion of HH:MM:SS format to seconds with improved error handling"""
    if walltime_series.empty:
        return walltime_series

    # Handle NaN values
    mask_na = walltime_series.isna()
    result = pd.Series(index=walltime_series.index, dtype=float)
    result[mask_na] = np.nan

    # Only process non-NaN values
    to_process = walltime_series[~mask_na]

    # Check if we have any values to process
    if to_process.empty:
        return result

    # Convert to string if not already
    if not pd.api.types.is_string_dtype(to_process):
        try:
            to_process = to_process.astype(str)
        except Exception as e:
            logger.warning(f"Failed to convert walltime series to string: {e}")
            return result

    # Handle numeric values
    try:
        mask_numeric = pd.to_numeric(to_process, errors='coerce').notna()
        result.loc[to_process[mask_numeric].index] = pd.to_numeric(to_process[mask_numeric])
    except Exception as e:
        logger.warning(f"Error processing numeric walltime values: {e}")
        mask_numeric = pd.Series(False, index=to_process.index)

    # Handle string time formats - only process if we have some non-numeric strings
    if (~mask_numeric).any():
        str_times = to_process[~mask_numeric]

        try:
            # Process HH:MM:SS format
            mask_hhmmss = str_times.str.count(':') == 2
            if mask_hhmmss.any():
                hhmmss = str_times[mask_hhmmss].str.split(':', expand=True).astype(float)
                result.loc[hhmmss.index] = hhmmss[0] * 3600 + hhmmss[1] * 60 + hhmmss[2]

            # Process MM:SS format
            mask_mmss = (str_times.str.count(':') == 1) & (~mask_hhmmss)
            if mask_mmss.any():
                mmss = str_times[mask_mmss].str.split(':', expand=True).astype(float)
                result.loc[mmss.index] = mmss[0] * 60 + mmss[1]
        except Exception as e:
            logger.warning(f"Error processing string walltime formats: {e}")

    return result


def get_exit_status_description(df):
    """Vectorized conversion of exit status to descriptive text with better error handling"""
    # Check if required columns exist
    if 'jobevent' not in df.columns or 'Exit_status' not in df.columns:
        return pd.Series([None] * len(df), index=df.index)

    # Initialize with empty strings
    try:
        # Convert categorical to string type first if needed
        jobevent = df['jobevent'].copy()
        exit_status = df['Exit_status'].copy()

        # Check if columns are categorical and convert to string if needed
        if isinstance(jobevent.dtype, pd.CategoricalDtype):
            jobevent = jobevent.astype(str)
        if isinstance(exit_status.dtype, pd.CategoricalDtype):
            exit_status = exit_status.astype(str)

        # Fill NA values with empty strings after conversion
        jobevent = jobevent.fillna('')
        exit_status = exit_status.fillna('')
    except Exception as e:
        logger.warning(f"Error preparing jobevent or exit_status columns: {e}")
        return pd.Series([None] * len(df), index=df.index)

    # Create result series
    result = pd.Series(index=df.index, dtype='object')

    try:
        # Apply conditions using vectorized operations
        result.loc[(jobevent == 'E') & (exit_status == '0')] = 'COMPLETED'
        result.loc[(jobevent == 'E') & (exit_status != '0')] = 'FAILED:' + exit_status[
            (jobevent == 'E') & (exit_status != '0')]
        result.loc[jobevent == 'A'] = 'ABORTED'
        result.loc[jobevent == 'S'] = 'STARTED'
        result.loc[jobevent == 'Q'] = 'QUEUED'

        # Handle remaining cases
        mask_other = ~result.notna()
        if mask_other.any():
            # Convert to string if needed before concatenation
            if not pd.api.types.is_string_dtype(jobevent):
                jobevent = jobevent.astype(str)
            if not pd.api.types.is_string_dtype(exit_status):
                exit_status = exit_status.astype(str)

            result.loc[mask_other] = jobevent[mask_other] + ':' + exit_status[mask_other]
    except Exception as e:
        logger.warning(f"Error generating exit status descriptions: {e}")
        result[:] = None

    return result


def optimize_dataframe_dtypes(df):
    """Optimize DataFrame memory usage by changing data types, with improved error handling"""
    # Get list of datetime columns to be careful with
    datetime_cols = []
    try:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            # Also identify columns with datetime-related names
            elif col in ["time", "date", "timestamp", "start_time", "end_time", "submit_time"] or any(
                    dt_term in col.lower() for dt_term in ["time", "date", "timestamp"]):
                datetime_cols.append(col)
    except Exception as e:
        logger.warning(f"Error identifying datetime columns: {e}")
        # If we can't identify datetime columns, use a conservative approach
        datetime_cols = ["time", "date", "timestamp", "start_time", "end_time", "submit_time"]

    # Convert object columns that are mostly numeric to appropriate numeric types
    for col in df.select_dtypes(include=['object']).columns:
        # Skip datetime columns
        if col in datetime_cols:
            continue

        # Try to convert to numeric if appropriate
        try:
            # Only convert if most values are numeric to avoid unnecessary work
            sample = df[col].dropna().head(100)
            if len(sample) > 0 and pd.to_numeric(sample, errors='coerce').notna().mean() > 0.8:
                # Convert just numeric values
                num_values = pd.to_numeric(df[col], errors='coerce')
                # If most values can be converted to numeric, convert the column
                if num_values.notna().sum() / len(df) > 0.8:
                    df[col] = num_values
                    # Downcast to the smallest possible numeric type
                    if df[col].notna().any():
                        if (df[col].dropna() % 1 == 0).all():
                            # It's an integer
                            df[col] = pd.to_numeric(df[col], downcast='integer')
                        else:
                            # It's a float
                            df[col] = pd.to_numeric(df[col], downcast='float')
        except Exception as e:
            # Just continue if there's an error
            logger.debug(f"Error optimizing column {col}: {e}")
            continue

    # Convert string columns with low cardinality to category type - but be careful
    for col in df.select_dtypes(include=['object']).columns:
        # Skip datetime columns
        if col in datetime_cols:
            continue

        # Check for string columns, but do it safely
        try:
            # Count strings in a sample
            sample = df[col].dropna().head(1000)
            if len(sample) > 0:
                string_count = sum(isinstance(x, str) for x in sample)
                # If most values are strings
                if string_count / len(sample) > 0.9:
                    # Only convert if there's a reasonable number of unique values
                    nunique = df[col].nunique()
                    if nunique is not None and nunique < len(df) * 0.5 and nunique < 10000:
                        df[col] = df[col].astype('category')
        except Exception as e:
            # Just continue if there's an error
            logger.debug(f"Error categorizing column {col}: {e}")
            continue

    # Downcast numeric columns to save memory
    for col in df.select_dtypes(include=['number']).columns:
        # Skip datetime columns
        if col in datetime_cols:
            continue

        try:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='integer')
        except Exception as e:
            # Just continue if there's an error
            logger.debug(f"Error downcasting column {col}: {e}")
            continue

    return df


def standardize_job_id(job_id_series):
    """Convert jobIDxxxxx to JOBxxxxx with better type handling"""
    # First check if we have a datetime series (which would cause .str accessor to fail)
    if pd.api.types.is_datetime64_any_dtype(job_id_series):
        logger.warning("Cannot standardize job IDs: series is datetime type")
        return job_id_series

    # Handle non-string types by converting to string first
    if not pd.api.types.is_string_dtype(job_id_series):
        # Convert to string type first, but handle non-null values only
        mask = job_id_series.notna()
        if mask.any():
            # Convert safely
            try:
                job_id_series = job_id_series.copy()
                job_id_series[mask] = job_id_series[mask].astype(str)
            except Exception as e:
                logger.warning(f"Error converting job IDs to string: {e}")
                return job_id_series

    # Now we should be able to safely use the .str accessor
    try:
        return job_id_series.str.replace(r'^jobID', 'JOB', regex=True)
    except AttributeError:
        # If we still can't use .str accessor, return original
        logger.warning("Could not use string accessor on job_id_series after conversion")
        return job_id_series


def enforce_schema_types(df, schema_columns):
    """
    Enforce schema data types to exactly match our parquet schema.

    Args:
        df: DataFrame to convert
        schema_columns: Dictionary mapping column names to desired types

    Returns:
        DataFrame with corrected types
    """
    # Define mapping from schema to pandas data types
    type_mapping = {
        'string': 'str',
        'double': 'float64',  # Use float64 to match 'double' in parquet schema
        'timestamp[ns, tz=UTC]': 'datetime64[ns, UTC]'
    }

    # Convert each column to the required type
    for col, dtype in schema_columns.items():
        if col in df.columns:
            try:
                # Check if it's a dictionary type that needs to be converted to string
                if pd.api.types.is_categorical_dtype(df[col]) or 'dictionary' in str(df[col].dtype):
                    df[col] = df[col].astype(str)

                # For string columns, convert objects, dictionaries, etc to plain strings
                if dtype == 'string':
                    df[col] = df[col].astype(str)

                # For double columns, ensure they are float64 not float32
                elif dtype == 'double':
                    df[col] = df[col].astype('float64')

                # Handle timestamp columns
                elif 'timestamp' in dtype:
                    if not pd.api.types.is_datetime64_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    # Ensure timezone is set
                    if hasattr(df[col], 'dt') and df[col].dt.tz is None:
                        df[col] = df[col].dt.tz_localize('UTC')
            except Exception as e:
                logger.warning(f"Error converting column {col} to {dtype}: {e}")

    # Remove any index columns that might cause problems
    if '__index_level_0__' in df.columns:
        df = df.drop(columns=['__index_level_0__'])

    return df


def parse_host_list(exec_host_series):
    """Parse exec_host into a list of hosts in JSON format with better type handling"""
    # Handle empty series
    if exec_host_series.empty:
        return exec_host_series

    # Initialize result series with same index and None values
    result = pd.Series([None] * len(exec_host_series), index=exec_host_series.index)

    # Check for datetime type (which would cause .str accessor to fail)
    if pd.api.types.is_datetime64_any_dtype(exec_host_series):
        logger.warning("Cannot parse host list: series is datetime type")
        return result

    # Force conversion to string if needed
    if not pd.api.types.is_string_dtype(exec_host_series):
        # Convert non-null values to string
        mask = exec_host_series.notna()
        if mask.any():
            # Convert carefully
            try:
                string_series = exec_host_series[mask].astype(str)
                exec_host_series = exec_host_series.copy()
                exec_host_series[mask] = string_series
            except Exception as e:
                logger.warning(f"Error converting exec_host series to string: {e}")
                return result

    # Create a mask for non-null string values
    mask = exec_host_series.notna()

    if mask.any():
        # Extract node names using regular expression
        node_pattern = re.compile(r'([^/+]+)/')

        # Apply the extraction only on valid strings
        valid_hosts = exec_host_series[mask]

        try:
            # Extract all matches for each string
            extracted = valid_hosts.apply(lambda x: node_pattern.findall(x) if isinstance(x, str) else [])

            # Get unique nodes and convert to JSON format
            unique_nodes = extracted.apply(lambda x: json.dumps(list(set(x))).replace('"', '') if x else None)

            # Update only the processed values
            result[mask] = unique_nodes
        except Exception as e:
            logger.warning(f"Error extracting host names: {e}")

    return result


def validate_dataframe_for_schema(df, expected_columns):
    """Ensure the dataframe has the correct columns for the schema"""
    # Check for duplicate column names
    duplicates = df.columns.duplicated()
    if any(duplicates):
        duplicate_cols = df.columns[duplicates].tolist()
        logger.warning(f"Found duplicate columns: {duplicate_cols}")

        # Create a safe copy with renamed duplicates
        safe_df = df.copy()
        for i, col in enumerate(safe_df.columns):
            if duplicates[i]:
                new_name = f"{col}_{i}"
                logger.info(f"Renaming duplicate column {col} to {new_name}")
                safe_df.columns.values[i] = new_name

        df = safe_df

    # Ensure all expected columns exist in the output
    missing_columns = [col for col in expected_columns if col not in df.columns]
    for col in missing_columns:
        df[col] = np.nan

    # Ensure only expected columns are present and in the right order
    return df[expected_columns]


def process_chunk(jobs_df, ts_chunk, chunk_id):
    """Process a chunk of time series data against all jobs with reduced memory usage"""
    thread_id = threading.get_ident()
    logger.debug(f"Thread {thread_id}: Processing chunk {chunk_id} with {len(ts_chunk)} rows")

    # Ensure we're only keeping the columns we'll actually need from ts_chunk
    required_ts_columns = ["Timestamp", "Job Id", "Event", "Value", "Host", "Units"]
    ts_columns_to_keep = [col for col in required_ts_columns if col in ts_chunk.columns]

    # Check if we have the minimum required columns
    if not all(col in ts_chunk.columns for col in ["Timestamp", "Job Id", "Event", "Value"]):
        logger.error(f"Chunk is missing required columns. Available columns: {ts_chunk.columns.tolist()}")
        return None

    # Keep only needed columns
    ts_chunk = ts_chunk[ts_columns_to_keep]

    # CRITICAL FIX: Ensure Timestamp is datetime type and not categorical
    if "Timestamp" in ts_chunk.columns:
        # Check if it's categorical first (using the non-deprecated method)
        if isinstance(ts_chunk["Timestamp"].dtype, pd.CategoricalDtype):
            logger.info("Converting Timestamp from categorical to datetime")
            # Convert to objects first to avoid .str accessor errors with non-string values
            ts_chunk["Timestamp"] = ts_chunk["Timestamp"].astype(object)
            # Then convert to datetime
            ts_chunk["Timestamp"] = pd.to_datetime(ts_chunk["Timestamp"], errors='coerce')
        # Make sure it's datetime regardless
        elif not pd.api.types.is_datetime64_any_dtype(ts_chunk["Timestamp"]):
            ts_chunk["Timestamp"] = pd.to_datetime(ts_chunk["Timestamp"], errors='coerce')

    # Keep only necessary columns from jobs_df for this join
    required_job_columns = ["jobID", "start", "end", "qtime", "Resource_List.walltime",
                            "Resource_List.nodect", "Resource_List.ncpus", "account",
                            "queue", "jobname", "user", "group", "exec_host",
                            "jobevent", "Exit_status"]

    # Only keep columns that actually exist in jobs_df
    job_columns_to_use = [col for col in required_job_columns if col in jobs_df.columns]
    jobs_subset = jobs_df[job_columns_to_use]

    # CRITICAL FIX: Ensure jobID in jobs_subset is string type for proper joining
    if "jobID" in jobs_subset.columns:
        # Ensure jobID is string type, not datetime or other type
        if not pd.api.types.is_string_dtype(jobs_subset["jobID"]):
            try:
                mask = jobs_subset["jobID"].notna()
                if mask.any():
                    jobs_subset = jobs_subset.copy()
                    jobs_subset.loc[mask, "jobID"] = jobs_subset.loc[mask, "jobID"].astype(str)
            except Exception as e:
                logger.warning(f"Error converting jobID to string: {e}")

    # CRITICAL FIX: Ensure Job Id in ts_chunk is string type for proper joining
    if "Job Id" in ts_chunk.columns:
        # Ensure Job Id is string type, not datetime or other type
        if not pd.api.types.is_string_dtype(ts_chunk["Job Id"]):
            try:
                mask = ts_chunk["Job Id"].notna()
                if mask.any():
                    ts_chunk = ts_chunk.copy()
                    ts_chunk.loc[mask, "Job Id"] = ts_chunk.loc[mask, "Job Id"].astype(str)
            except Exception as e:
                logger.warning(f"Error converting Job Id to string: {e}")

    # CRITICAL FIX: Ensure datetime columns in jobs_df are proper datetime
    datetime_cols = ["start", "end", "qtime"]
    for col in datetime_cols:
        if col in jobs_subset.columns:
            if isinstance(jobs_subset[col].dtype, pd.CategoricalDtype):
                logger.info(f"Converting {col} from categorical to datetime")
                # Convert to objects first
                jobs_subset[col] = jobs_subset[col].astype(object)
                # Then convert to datetime
                jobs_subset[col] = pd.to_datetime(jobs_subset[col], errors='coerce')
            elif not pd.api.types.is_datetime64_any_dtype(jobs_subset[col]):
                jobs_subset[col] = pd.to_datetime(jobs_subset[col], errors='coerce')

    # Join time series chunk with jobs data on jobID
    try:
        joined = pd.merge(
            ts_chunk,
            jobs_subset,
            left_on="Job Id",
            right_on="jobID",
            how="inner"
        )

        # Log join results
        logger.debug(f"Thread {thread_id}: Joined dataframe has {len(joined)} rows")

        if joined.empty:
            return None

    except Exception as e:
        logger.error(f"Error joining dataframes: {e}")
        # Log column types to help diagnose issues
        logger.error(f"ts_chunk Job Id dtype: {ts_chunk['Job Id'].dtype}")
        logger.error(f"jobs_subset jobID dtype: {jobs_subset['jobID'].dtype}")
        return None

    # Release memory from the input chunks since they're no longer needed
    del ts_chunk
    del jobs_subset
    gc.collect()  # Force garbage collection

    # Double-check datetime types before filtering to avoid the categorical error
    for col in ["Timestamp", "start", "end"]:
        if col in joined.columns:
            # Ensure it's not categorical (using non-deprecated method)
            if isinstance(joined[col].dtype, pd.CategoricalDtype):
                # Convert to objects first
                logger.info(f"Converting {col} from categorical to datetime for comparison")
                joined[col] = joined[col].astype(object)
                # Then convert to datetime
                joined[col] = pd.to_datetime(joined[col], errors='coerce')
            elif not pd.api.types.is_datetime64_any_dtype(joined[col]):
                joined[col] = pd.to_datetime(joined[col], errors='coerce')

    try:
        # Filter timestamps that fall between job start and end times
        mask = (joined["Timestamp"] >= joined["start"]) & (joined["Timestamp"] <= joined["end"])
        # Apply the filter in-place to reduce memory usage
        filtered = joined.loc[mask]
    except Exception as e:
        logger.error(f"Comparison failed with error: {str(e)}")
        # Log detailed information about column types
        for col in ["Timestamp", "start", "end"]:
            if col in joined.columns:
                logger.error(f"Column {col} dtype: {joined[col].dtype}")
                logger.error(f"Sample values: {joined[col].head(3)}")
        return None

    # Release memory from joined dataframe
    del joined
    gc.collect()

    logger.debug(f"Thread {thread_id}: Filtered dataframe has {len(filtered)} rows")

    if filtered.empty:
        return None

    # Process events more efficiently to reduce memory usage
    try:
        events = filtered['Event'].unique()
    except Exception as e:
        logger.error(f"Error getting unique events: {e}")
        return None

    # Create only the necessary columns with NaN values
    for event in events:
        try:
            if event in ["cpuuser", "gpu_usage", "memused", "memused_minus_diskcache", "nfs", "block"]:
                col_name = f'value_{event}'
            else:
                col_name = event

            # Create the column with NaN values
            filtered[col_name] = np.nan

            # Fill values only for matching event rows
            event_mask = filtered['Event'] == event
            filtered.loc[event_mask, col_name] = filtered.loc[event_mask, 'Value']
        except Exception as e:
            logger.warning(f"Error processing event {event}: {e}")
            continue

    # Drop Event and Value columns to save memory
    try:
        filtered.drop(columns=['Event', 'Value'], inplace=True)
    except Exception as e:
        logger.warning(f"Error dropping columns: {e}")

    # FIXED: Fixed column mapping to avoid duplicates
    column_mapping = {
        # Time fields
        "Timestamp": "time",
        "qtime": "submit_time",
        "start": "start_time",
        "end": "end_time",
        "Resource_List.walltime": "timelimit",

        # Resource allocation
        "Resource_List.nodect": "nhosts",
        "Resource_List.ncpus": "ncores",
        "exec_host": "host_list",

        # Job identification
        "account": "account",
        "queue": "queue",
        "jobname": "jobname",
        "user": "username",
        "group": "group",  # FIXED: Changed from "account" to "group"

        # Core identifiers
        "jobID": "jid",
        "Host": "host",
        "Units": "unit"
    }

    # Only rename columns that exist
    existing_cols = {col: mapping for col, mapping in column_mapping.items() if col in filtered.columns}
    try:
        filtered.rename(columns=existing_cols, inplace=True)  # Rename in-place

        # Check for and handle duplicate columns
        duplicate_check = filtered.columns.value_counts()
        duplicate_cols = duplicate_check[duplicate_check > 1].index.tolist()

        for col in duplicate_cols:
            # Find the duplicate columns and rename them
            dup_positions = [i for i, x in enumerate(filtered.columns) if x == col]
            # Keep the first occurrence, rename the others
            for pos in dup_positions[1:]:
                new_col_name = f"{col}_dup"
                filtered.columns.values[pos] = new_col_name
                logger.info(f"Renamed duplicate column {col} to {new_col_name}")
    except Exception as e:
        logger.warning(f"Error renaming columns: {e}")

    # Process special cases with better error handling
    if "timelimit" in filtered.columns:
        try:
            filtered["timelimit"] = convert_walltime_to_seconds(filtered["timelimit"])
        except Exception as e:
            logger.warning(f"Error converting walltime: {e}")
            filtered["timelimit"] = np.nan

    if "host_list" in filtered.columns:
        try:
            filtered["host_list"] = parse_host_list(filtered["host_list"])
        except Exception as e:
            logger.warning(f"Error parsing host list: {e}")
            filtered["host_list"] = None

    # Generate exitcode from jobevent and Exit_status using vectorized approach
    if "jobevent" in filtered.columns:
        try:
            filtered["exitcode"] = get_exit_status_description(filtered)

            # Remove the source columns after processing
            columns_to_drop = [col for col in ["jobevent", "Exit_status"] if col in filtered.columns]
            if columns_to_drop:
                filtered.drop(columns=columns_to_drop, inplace=True)
        except Exception as e:
            logger.warning(f"Error generating exitcode: {e}")
            filtered["exitcode"] = None

    # Ensure all required columns exist in the output
    set3_columns = ["time", "submit_time", "start_time", "end_time", "timelimit",
                    "nhosts", "ncores", "account", "queue", "host", "jid", "unit",
                    "jobname", "exitcode", "host_list", "username",
                    "value_cpuuser", "value_gpu_usage", "value_memused",
                    "value_memused_minus_diskcache", "value_nfs", "value_block"]

    # Add missing columns
    for col in set3_columns:
        if col not in filtered.columns:
            filtered[col] = np.nan

    # Convert datetime columns to UTC timezone - doing this carefully to prevent errors
    datetime_cols = ["time", "submit_time", "start_time", "end_time"]
    for col in datetime_cols:
        if col in filtered.columns and filtered[col].notna().any():
            try:
                # Get the first valid index carefully
                valid_idx = filtered[filtered[col].notna()].index
                if len(valid_idx) > 0:
                    first_valid = valid_idx[0]
                    sample_dt = filtered.loc[first_valid, col]
                    if hasattr(sample_dt, 'tzinfo') and sample_dt.tzinfo is None:
                        filtered[col] = filtered[col].dt.tz_localize('UTC')
            except (TypeError, AttributeError) as e:
                logger.warning(f"Failed to localize timezone for {col}: {e}. Skipping.")

    # Validate dataframe schema before returning
    filtered = validate_dataframe_for_schema(filtered, set3_columns)

    # Optimize the result dataframe, being careful not to convert datetime columns to categorical
    try:
        filtered = optimize_dataframe_dtypes(filtered)
    except Exception as e:
        logger.warning(f"Error optimizing datatypes: {e}")

    # Add memory usage logging
    mem_usage = filtered.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(
        f"Thread {thread_id}: Completed processing chunk {chunk_id} - produced {len(filtered)} rows - Memory usage: {mem_usage:.2f} MB")

    return filtered


def read_parquet_chunk(file_path, start_row, chunk_size):
    """Read a chunk from a parquet file with better error handling"""
    logger.debug(f"Reading chunk from {start_row} to {start_row + chunk_size}")

    try:
        # First try the simplest approach
        chunk_df = pd.read_parquet(
            file_path,
            engine='pyarrow',
            use_threads=False,
            offset=start_row,
            rows=chunk_size
        )
        return chunk_df
    except Exception as e1:
        logger.warning(f"Offset/rows approach failed: {e1}, trying normal read with skip/nrows")
        try:
            # Try pandas read_parquet with skip/nrows
            chunk_df = pd.read_parquet(
                file_path,
                engine='pyarrow',
                use_threads=False,
                skiprows=range(start_row),
                nrows=chunk_size
            )
            return chunk_df
        except Exception as e2:
            logger.warning(f"Skip/nrows approach failed: {e2}, trying row groups")
            try:
                # Try reading by row groups
                pf = pq.ParquetFile(file_path)

                # Calculate which row groups to read
                row_offsets = [0]
                for i in range(pf.metadata.num_row_groups):
                    row_offsets.append(row_offsets[-1] + pf.metadata.row_group(i).num_rows)

                # Find row groups that overlap with our range
                start_rg = None
                end_rg = None
                for i in range(len(row_offsets) - 1):
                    if start_row < row_offsets[i + 1] and start_rg is None:
                        start_rg = i
                    if start_row + chunk_size <= row_offsets[i]:
                        end_rg = i
                        break

                if start_rg is None:
                    start_rg = 0
                if end_rg is None:
                    end_rg = pf.metadata.num_row_groups - 1

                # Read the row groups
                tables = []
                for i in range(start_rg, end_rg + 1):
                    tables.append(pf.read_row_group(i))

                if tables:
                    table = pa.concat_tables(tables)
                    chunk_df = table.to_pandas()

                    # Slice to get the exact rows we want
                    local_start = start_row - row_offsets[start_rg]
                    local_end = min(local_start + chunk_size, len(chunk_df))
                    return chunk_df.iloc[local_start:local_end]
                else:
                    raise ValueError("No row groups to read")

            except Exception as e3:
                logger.error(f"All approaches failed: {e3}")
                # Final fallback - read the whole file and slice
                try:
                    logger.warning("Falling back to reading entire file - this may use a lot of memory")
                    table = pq.read_table(file_path)
                    chunk_df = table.to_pandas()

                    if start_row < len(chunk_df):
                        end_row = min(start_row + chunk_size, len(chunk_df))
                        return chunk_df.iloc[start_row:end_row]
                    else:
                        logger.error(f"Start row {start_row} exceeds dataframe length {len(chunk_df)}")
                        return pd.DataFrame()
                except Exception as e4:
                    logger.error(f"Emergency fallback failed: {e4}")
                    return pd.DataFrame()


def clean_up_cache():
    """Clean up the cache directory"""
    try:
        # Don't clean up input directories - that's where our data is!
        if OUTPUT_PATH.exists():
            logger.info(f"Cleaning up {OUTPUT_PATH} directory...")
            for file_path in OUTPUT_PATH.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
    except Exception as e:
        logger.error(f"Error cleaning cache: {e}")


def check_and_optimize_resources():
    """Check resources and optimize if needed"""
    # Check disk space
    free_disk_gb = get_free_disk_space()
    if free_disk_gb < MIN_FREE_DISK_GB:
        logger.warning(f"Low disk space: {free_disk_gb:.2f}GB free. Cleaning up cache...")
        clean_up_cache()

    # Check memory
    free_memory_gb = get_available_memory()
    memory_usage_gb = get_memory_usage()

    # Log current memory status
    logger.debug(f"Memory check: Usage={memory_usage_gb:.2f}GB, Available={free_memory_gb:.2f}GB")

    # Take progressively more aggressive actions based on memory pressure
    if free_memory_gb < MIN_FREE_MEMORY_GB * 0.5 or memory_usage_gb > MAX_MEMORY_USAGE_GB * 0.9:
        # Critical situation - force garbage collection and sleep
        logger.warning(f"Critical memory situation: Usage={memory_usage_gb:.2f}GB, Available={free_memory_gb:.2f}GB")
        # Run garbage collection multiple times with pauses
        for _ in range(3):
            gc.collect()
            time.sleep(2)  # Give OS time to reclaim memory

    elif free_memory_gb < MIN_FREE_MEMORY_GB or memory_usage_gb > MAX_MEMORY_USAGE_GB * 0.8:
        # Concerning situation - force garbage collection
        logger.warning(
            f"Low memory: {free_memory_gb:.2f}GB free / Usage: {memory_usage_gb:.2f}GB. Forcing garbage collection...")
        gc.collect()

    # After optimizations, check if we're still in a bad state
    free_memory_gb = get_available_memory()
    memory_usage_gb = get_memory_usage()

    if free_memory_gb < MIN_FREE_MEMORY_GB * 0.5 or memory_usage_gb > MAX_MEMORY_USAGE_GB * 0.95:
        logger.warning(
            f"Memory situation still critical after cleanup: Usage={memory_usage_gb:.2f}GB, Available={free_memory_gb:.2f}GB")
        return False

    return True


def get_year_month_combinations():
    """Get all year-month combinations from both local directories with support for chunked files"""
    logger.info("Getting list of files from local directories...")

    # Print the paths we're checking to help with debugging
    logger.info(f"Job accounting path: {JOB_ACCOUNTING_PATH}")
    logger.info(f"Process metric path: {PROC_METRIC_PATH}")

    # Check if directories exist
    if not JOB_ACCOUNTING_PATH.exists():
        logger.warning(f"Job accounting directory does not exist: {JOB_ACCOUNTING_PATH}")
    if not PROC_METRIC_PATH.exists():
        logger.warning(f"Process metric directory does not exist: {PROC_METRIC_PATH}")

    # Get files from proc metric directory (parquet files)
    proc_metric_files = list(PROC_METRIC_PATH.glob("*.parquet"))
    logger.info(f"Found {len(proc_metric_files)} files in proc metric directory (parquet)")
    for f in proc_metric_files:
        logger.info(f"  Found parquet file: {f.name}")

    # Get files from job accounting directory (CSV files)
    job_accounting_files = list(JOB_ACCOUNTING_PATH.glob("*.csv"))
    logger.info(f"Found {len(job_accounting_files)} files in job accounting directory (CSV)")
    for f in job_accounting_files:
        logger.info(f"  Found CSV file: {f.name}")

    # Extract year-month from filenames more efficiently
    proc_metrics_years_months = set()

    # Compile regex patterns once for better performance
    fresco_pattern = re.compile(r'FRESCO_Conte_ts_(\d{4})_(\d{2})(?:_v\d+)?(?:_chunk\d+)?\.parquet')
    other_pattern = re.compile(r'_(\d{4})_(\d{2})_')

    for filepath in proc_metric_files:
        filename = filepath.name
        logger.debug(f"Processing parquet filename: {filename}")

        # Try the FRESCO pattern first
        match = fresco_pattern.search(filename)
        if match:
            year, month = match.groups()
            proc_metrics_years_months.add((year, month))
            logger.info(f"Matched FRESCO pattern: {year}-{month} from {filename}")
            continue

        # Try the other pattern
        matches = other_pattern.findall(filename)
        if matches:
            year, month = matches[0]
            proc_metrics_years_months.add((year, month))
            logger.info(f"Matched other pattern: {year}-{month} from {filename}")

    # Compile job accounting pattern
    job_pattern = re.compile(r'(\d{4})-(\d{2})\.csv')

    job_accounting_years_months = set()
    for filepath in job_accounting_files:
        filename = filepath.name
        logger.debug(f"Processing CSV filename: {filename}")

        match = job_pattern.match(filename)
        if match:
            year, month = match.groups()
            job_accounting_years_months.add((year, month))
            logger.info(f"Matched job pattern: {year}-{month} from {filename}")

    # Find common year-month combinations
    common_years_months = proc_metrics_years_months.intersection(job_accounting_years_months)
    logger.info(f"Found {len(common_years_months)} common year-month combinations: {common_years_months}")

    return common_years_months


def read_jobs_df(job_file):
    """Read job data from CSV with optimized memory usage from the start"""
    # First, determine column data types by scanning sample
    sample_size = min(1000, os.path.getsize(job_file) // 2)
    sample_df = pd.read_csv(job_file, nrows=1000)

    # Analyze column types in sample
    dtypes = {}
    categorical_columns = []
    datetime_columns = ["ctime", "qtime", "etime", "start", "end", "timestamp", "Timestamp"]

    for col in sample_df.columns:
        # Skip datetime columns
        if col in datetime_columns or any(dt_term in col.lower() for dt_term in ["time", "date", "timestamp"]):
            continue

        nunique = sample_df[col].nunique()

        # Handle column type assignment
        if nunique <= 100 and sample_df[col].dtype == 'object':
            # Low cardinality strings should be categorical
            categorical_columns.append(col)
        elif pd.api.types.is_numeric_dtype(sample_df[col]):
            # Optimize numeric columns
            if sample_df[col].isna().sum() == 0 and (sample_df[col] % 1 == 0).all():
                if sample_df[col].max() < 32767 and sample_df[col].min() > -32768:
                    dtypes[col] = 'int16'
                elif sample_df[col].max() < 2147483647 and sample_df[col].min() > -2147483648:
                    dtypes[col] = 'int32'
                else:
                    dtypes[col] = 'int64'
            else:
                # Use float32 for most floats to save memory
                dtypes[col] = 'float32'

    # Now read the actual file with optimized settings
    jobs_df = pd.read_csv(
        job_file,
        dtype=dtypes,
        low_memory=True,
        memory_map=True  # Memory map the file for more efficient I/O
    )

    # Convert categorical columns after reading
    for col in categorical_columns:
        if col in jobs_df.columns:
            jobs_df[col] = jobs_df[col].astype('category')

    # Standardize job IDs
    if "jobID" in jobs_df.columns:
        jobs_df["jobID"] = standardize_job_id(jobs_df["jobID"])
    else:
        logger.warning("jobID column not found in CSV file")

    # Convert datetime columns explicitly
    for col in datetime_columns:
        if col in jobs_df.columns:
            jobs_df[col] = pd.to_datetime(jobs_df[col], format="%m/%d/%Y %H:%M:%S", errors='coerce')

    return jobs_df


def memory_monitor():
    """Thread to monitor memory usage and set terminate flag if needed"""
    global terminate_requested

    while not terminate_requested:
        current_memory = get_memory_usage()
        available_memory = get_available_memory()

        # Log memory status periodically
        logger.debug(f"Memory monitor: Usage {current_memory:.2f}GB, Available {available_memory:.2f}GB")

        # Alert when approaching limit
        if current_memory > MAX_MEMORY_USAGE_GB * 0.8 and current_memory < MAX_MEMORY_USAGE_GB:
            logger.warning(f"Memory usage approaching limit: {current_memory:.2f}GB / {MAX_MEMORY_USAGE_GB}GB")
            # Try to free some memory
            gc.collect()

        if current_memory > MAX_MEMORY_USAGE_GB:
            logger.warning(f"Memory usage exceeded limit: {current_memory:.2f}GB > {MAX_MEMORY_USAGE_GB}GB")
            logger.warning("Requesting graceful termination")
            terminate_requested = True
            break

        if available_memory < 1.0:  # Critical memory situation - less than 1GB free
            logger.warning(f"Critical memory situation: only {available_memory:.2f}GB available")
            logger.warning("Requesting graceful termination")
            terminate_requested = True
            break

        # Sleep to avoid constant CPU usage
        time.sleep(5)


def process_ts_file_in_parallel(ts_file, jobs_df, output_writer):
    """Process a single time series file with reduced memory usage"""
    global terminate_requested
    logger.info(f"Processing TS file: {ts_file}")

    # Define schema for output columns
    schema_column_types = {
        'time': 'timestamp[ns, tz=UTC]',
        'submit_time': 'timestamp[ns, tz=UTC]',
        'start_time': 'timestamp[ns, tz=UTC]',
        'end_time': 'timestamp[ns, tz=UTC]',
        'timelimit': 'double',
        'nhosts': 'double',
        'ncores': 'double',
        'account': 'string',
        'queue': 'string',
        'host': 'string',
        'jid': 'string',
        'unit': 'string',
        'jobname': 'string',
        'exitcode': 'string',
        'host_list': 'string',
        'username': 'string',
        'value_cpuuser': 'double',
        'value_gpu_usage': 'double',
        'value_memused': 'double',
        'value_memused_minus_diskcache': 'double',
        'value_nfs': 'double',
        'value_block': 'double'
    }

    # List of columns in the expected order
    expected_columns = list(schema_column_types.keys())

    try:
        # Get the file size to estimate appropriate chunk count
        file_size = os.path.getsize(ts_file)

        # Try to read just the file metadata without loading everything
        parquet_file = pq.ParquetFile(ts_file)
        total_rows = parquet_file.metadata.num_rows

        # Use more conservative chunk sizing
        available_memory_gb = get_available_memory()
        current_memory_gb = get_memory_usage()

        # Be even more conservative with chunk sizing
        max_allowed_chunk_gb = min(1.0, available_memory_gb / 4)  # More aggressive memory limits
        estimated_bytes_per_row = file_size / total_rows if total_rows > 0 else 1000

        # Calculate how many rows we can safely process in one chunk
        safe_rows = int((max_allowed_chunk_gb * 1024 * 1024 * 1024) / estimated_bytes_per_row / 2)

        # Clamp chunk size between more conservative bounds
        chunk_size = max(5_000, min(safe_rows, 100_000))  # More conservative upper bound

        # Calculate number of chunks
        num_chunks = (total_rows + chunk_size - 1) // chunk_size  # Ceiling division

        # Process a very small file like 10MB in one go
        if file_size / (1024 * 1024) < 20:  # If file is less than 20MB
            logger.info(f"Small file ({file_size / (1024 * 1024 * 1024):.2f} GB), processing in one go")
            return process_entire_file_with_schema(ts_file, jobs_df, output_writer, expected_columns,
                                                   schema_column_types)

        # Use just 1 worker for more predictable memory usage
        num_workers = 1

        logger.info(
            f"File has {total_rows} rows, processing with {num_workers} thread in {num_chunks} chunks of {chunk_size} rows")

        # Process file in chunks sequentially to better control memory
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # For each chunk, process sequentially
            for chunk_idx in range(num_chunks):
                # Check if termination was requested
                if terminate_requested:
                    logger.info("Termination requested, stopping processing of new chunks")
                    break

                # Check memory before processing each chunk - more aggressive memory check
                current_memory = get_memory_usage()
                available_memory = get_available_memory()

                # More aggressive memory management - if memory is high, take stronger action
                while current_memory > MAX_MEMORY_USAGE_GB * 0.8 or available_memory < MIN_FREE_MEMORY_GB * 1.5:
                    if terminate_requested:
                        break

                    logger.warning(
                        f"Memory usage too high: {current_memory:.2f}GB or available too low: {available_memory:.2f}GB. Pausing.")

                    # Force garbage collection
                    gc.collect()

                    # Wait longer for memory to be freed
                    time.sleep(10)

                    # Check again
                    current_memory = get_memory_usage()
                    available_memory = get_available_memory()

                    # Log progress
                    logger.info(f"After cleanup: Usage={current_memory:.2f}GB, Available={available_memory:.2f}GB")

                    # If we've waited and memory is still too high, terminate
                    if current_memory > MAX_MEMORY_USAGE_GB * 0.9 or available_memory < MIN_FREE_MEMORY_GB:
                        logger.warning("Memory still critical after cleanup. Requesting termination.")
                        terminate_requested = True
                        break

                # Skip this chunk if termination was requested during the memory wait
                if terminate_requested:
                    break

                # Calculate row range for this chunk
                start_row = chunk_idx * chunk_size
                end_row = min(start_row + chunk_size, total_rows)

                try:
                    # Use our improved function to read a chunk
                    chunk_df = read_parquet_chunk(ts_file, start_row, chunk_size)

                    # Force garbage collection after reading
                    gc.collect()

                    # Optimize chunk data types immediately to reduce memory
                    if chunk_df is not None and not chunk_df.empty:
                        # Log chunk read success with memory usage
                        mem_usage = chunk_df.memory_usage(deep=True).sum() / (1024 * 1024)
                        logger.info(
                            f"Successfully read chunk {chunk_idx + 1}/{num_chunks} with {len(chunk_df)} rows - Memory: {mem_usage:.2f} MB")

                        # Apply memory optimizations immediately
                        chunk_df = optimize_dataframe_dtypes(chunk_df)

                        # Process the chunk
                        result_df = process_chunk(jobs_df, chunk_df, chunk_idx + 1)

                        # Clean up chunk dataframe to free memory
                        del chunk_df
                        gc.collect()

                        # Write results if we have any
                        if result_df is not None and not result_df.empty:
                            # Validate the dataframe against our schema before writing
                            result_df = validate_dataframe_for_schema(result_df, expected_columns)

                            # IMPORTANT: Enforce schema types to match exactly
                            result_df = enforce_schema_types(result_df, schema_column_types)

                            # Convert to PyArrow table and write to output
                            try:
                                table = pa.Table.from_pandas(result_df)
                                output_writer.write_table(table)
                                logger.info(f"Successfully wrote {len(result_df)} rows to output")
                            except Exception as e:
                                logger.error(f"Error writing to parquet: {e}")
                                # Log detailed schema info
                                logger.error(f"Result DataFrame dtypes: {result_df.dtypes}")
                                for col in result_df.columns:
                                    logger.error(
                                        f"Column {col} dtype: {result_df[col].dtype}, sample: {result_df[col].head(1).values}")

                            # Release memory
                            del result_df
                            gc.collect()
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_idx}: {e}")
                    continue

                # Check and optimize resources after each chunk
                check_and_optimize_resources()

        return not terminate_requested
    except Exception as e:
        logger.error(f"Error processing file {ts_file}: {e}")
        return False


def process_year_month(year, month):
    """Process a specific year-month combination with better support for chunked files"""
    global terminate_requested

    logger.info(f"Processing year: {year}, month: {month}")

    # Check resources before starting
    check_and_optimize_resources()

    # Check if termination was requested
    if terminate_requested:
        logger.info("Termination requested, skipping processing")
        return

    try:
        # Get job accounting file (CSV)
        job_file = JOB_ACCOUNTING_PATH / f"{year}-{month}.csv"

        # Check if job file exists
        if not job_file.exists():
            logger.error(f"Job file not found: {job_file}")
            return

        # Find all time series files (Parquet) for this year/month with support for chunked files
        ts_files = []

        # UPDATED: Pattern to match both original files and chunked files
        fresco_pattern = f"FRESCO_Conte_ts_{year}_{month}"

        # Find all files that match our patterns
        all_files = []

        # Look for files with the basic pattern
        basic_files = list(PROC_METRIC_PATH.glob(f"{fresco_pattern}*.parquet"))
        all_files.extend(basic_files)

        # If we found too few files, check for chunked versions
        if not all_files:
            # Try backup pattern
            backup_files = list(PROC_METRIC_PATH.glob(f"*_{year}_{month}_*.parquet"))
            all_files.extend(backup_files)

        if all_files:
            # Sort files logically - original files first, then chunked files
            for f in sorted(all_files, key=lambda x: (
                    # Sort by original vs chunked (original files first)
                    1 if "_chunk" in x.name else 0,
                    # Then by version number
                    int(re.search(r'v(\d+)', x.name).group(1)) if re.search(r'v(\d+)', x.name) else 0,
                    # Then by chunk number if it exists
                    int(re.search(r'chunk(\d+)', x.name).group(1)) if re.search(r'chunk(\d+)', x.name) else 0,
                    # Finally by name
                    x.name
            )):
                ts_files.append(f)

        # Log what we found
        logger.info(f"Found {len(ts_files)} files for {year}-{month}")
        for ts_file in ts_files:
            logger.info(f"Found time series file: {ts_file}")

        if not ts_files:
            logger.warning(f"No time series files found for {year}-{month}")
            return

        # Create output file path (will be Parquet)
        output_file = OUTPUT_PATH / f"transformed_{year}_{month}.parquet"

        # Read job data from CSV file with improved memory handling
        logger.info(f"Reading job data from file: {job_file}")

        # First scan to determine data types
        sample_df = pd.read_csv(job_file, nrows=1000)
        dtypes = {}
        date_columns = ["ctime", "qtime", "etime", "start", "end", "timestamp"]

        # Determine column types to minimize memory usage
        for col in sample_df.columns:
            if col in date_columns:
                continue  # Handle date columns separately
            elif sample_df[col].dtype == 'int64':
                # Use smaller integer types when possible
                max_val = sample_df[col].max()
                min_val = sample_df[col].min()
                if pd.notna(max_val) and pd.notna(min_val):
                    if max_val < 32767 and min_val > -32768:
                        dtypes[col] = 'int16'
                    elif max_val < 2147483647 and min_val > -2147483648:
                        dtypes[col] = 'int32'
            elif sample_df[col].dtype == 'float64':
                dtypes[col] = 'float32'  # Use float32 to save memory

        # Read with optimized settings
        jobs_df = pd.read_csv(
            job_file,
            dtype=dtypes,
            parse_dates=date_columns,  # More efficient date parsing
            low_memory=True,
            memory_map=True  # Memory map the file for more efficient I/O
        )

        # Log memory usage
        mem_usage = jobs_df.memory_usage(deep=True).sum() / (1024 * 1024 * 1024)
        logger.info(f"Job data loaded into dataframe with {len(jobs_df)} rows - Memory usage: {mem_usage:.2f} GB")

        # Standardize job IDs - make sure it's done before datetime conversion to avoid
        # the str accessor on datetime error
        if "jobID" in jobs_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(jobs_df["jobID"]):
                jobs_df["jobID"] = standardize_job_id(jobs_df["jobID"])
            else:
                logger.warning("jobID column is datetime type, skipping standardization")
        else:
            logger.warning("jobID column not found in CSV file")

        # Define schema for output columns with proper types
        schema_column_types = {
            'time': 'timestamp[ns, tz=UTC]',
            'submit_time': 'timestamp[ns, tz=UTC]',
            'start_time': 'timestamp[ns, tz=UTC]',
            'end_time': 'timestamp[ns, tz=UTC]',
            'timelimit': 'double',
            'nhosts': 'double',
            'ncores': 'double',
            'account': 'string',
            'queue': 'string',
            'host': 'string',
            'jid': 'string',
            'unit': 'string',
            'jobname': 'string',
            'exitcode': 'string',
            'host_list': 'string',
            'username': 'string',
            'value_cpuuser': 'double',
            'value_gpu_usage': 'double',
            'value_memused': 'double',
            'value_memused_minus_diskcache': 'double',
            'value_nfs': 'double',
            'value_block': 'double'
        }

        # Create parquet schema for output file
        schema = pa.schema([
            ('time', pa.timestamp('ns', tz='UTC')),
            ('submit_time', pa.timestamp('ns', tz='UTC')),
            ('start_time', pa.timestamp('ns', tz='UTC')),
            ('end_time', pa.timestamp('ns', tz='UTC')),
            ('timelimit', pa.float64()),
            ('nhosts', pa.float64()),
            ('ncores', pa.float64()),
            ('account', pa.string()),
            ('queue', pa.string()),
            ('host', pa.string()),
            ('jid', pa.string()),
            ('unit', pa.string()),
            ('jobname', pa.string()),
            ('exitcode', pa.string()),
            ('host_list', pa.string()),
            ('username', pa.string()),
            ('value_cpuuser', pa.float64()),
            ('value_gpu_usage', pa.float64()),
            ('value_memused', pa.float64()),
            ('value_memused_minus_diskcache', pa.float64()),
            ('value_nfs', pa.float64()),
            ('value_block', pa.float64())
        ])

        # Create a single writer for the output file
        with pq.ParquetWriter(str(output_file), schema, compression='snappy') as writer:
            # Process each file sequentially
            for i, ts_file in enumerate(ts_files):
                # Check if termination was requested
                if terminate_requested:
                    logger.info("Termination requested, stopping file processing")
                    break

                logger.info(f"Processing file {i + 1}/{len(ts_files)}: {ts_file.name}")

                # Use more aggressive memory management for very large files
                file_size_gb = os.path.getsize(ts_file) / (1024 * 1024 * 1024)

                # For smaller files, we can just process the entire file at once
                # For larger files, we use our chunking approach
                if file_size_gb < 0.1:  # Less than 100MB
                    logger.info(f"Small file ({file_size_gb:.2f} GB), processing in one go")
                    success = process_entire_file_with_schema(
                        ts_file,
                        jobs_df,
                        writer,
                        list(schema_column_types.keys()),
                        schema_column_types
                    )
                else:
                    # Use our chunking approach for larger files
                    success = process_ts_file_in_parallel(ts_file, jobs_df, writer)

                # Force garbage collection between files
                gc.collect()
                check_and_optimize_resources()

                # Break if processing was not successful
                if not success:
                    logger.warning(f"Processing of {ts_file} was not successful, moving to next file")

        # Check if the output file was created properly
        if output_file.exists() and output_file.stat().st_size > 0 and not terminate_requested:
            logger.info(f"Successfully created {output_file}")
        elif terminate_requested:
            logger.info(f"Processing of {year}-{month} was interrupted")
        else:
            logger.warning(f"No output generated for {year}-{month}")

        # Clear memory
        del jobs_df
        gc.collect()

    except Exception as e:
        logger.error(f"Error processing {year}-{month}: {e}", exc_info=True)


def process_entire_file_with_schema(ts_file, jobs_df, output_writer, expected_columns, schema_column_types):
    """Process a small file in its entirety without chunking, with schema enforcement"""
    global terminate_requested

    try:
        logger.info(f"Processing entire file: {ts_file}")

        # Read the entire file
        ts_df = pd.read_parquet(ts_file)

        # Log information about the loaded dataframe
        mem_usage = ts_df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Loaded entire file with {len(ts_df)} rows - Memory usage: {mem_usage:.2f} MB")

        # Process the dataframe
        result_df = process_chunk(jobs_df, ts_df, 1)

        # Free memory
        del ts_df
        gc.collect()

        # Write the result if we have one
        if result_df is not None and not result_df.empty:
            # Validate the dataframe against our schema before writing
            result_df = validate_dataframe_for_schema(result_df, expected_columns)

            # IMPORTANT: Enforce schema types to match exactly
            result_df = enforce_schema_types(result_df, schema_column_types)

            try:
                # Convert to PyArrow table and write to output
                table = pa.Table.from_pandas(result_df)
                output_writer.write_table(table)
                logger.info(f"Successfully wrote {len(result_df)} rows from whole file to output")
            except Exception as e:
                logger.error(f"Error writing to parquet: {e}")
                # Log detailed schema info
                logger.error(f"Result DataFrame dtypes: {result_df.dtypes}")
                for col in result_df.columns:
                    logger.error(f"Column {col} dtype: {result_df[col].dtype}, sample: {result_df[col].head(1).values}")

            # Release memory
            del result_df
            gc.collect()

        return True
    except Exception as e:
        logger.error(f"Error processing file {ts_file}: {e}")
        return False


def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals"""
    global terminate_requested
    print("\nTermination requested. Cleaning up and exiting gracefully...")
    terminate_requested = True

    # In case of second Ctrl+C, exit immediately
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(1))

    # Give the script a bit of time to clean up
    logger.info("Please wait while current operations complete...")


def main():
    """Main function with memory monitoring"""
    global terminate_requested

    # Set up signal handlers for graceful termination
    setup_signal_handlers()

    logger.info("Starting data transformation process")
    start_time = time.time()

    try:
        # Start memory monitoring thread
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()

        # Get all year-month combinations
        year_month_combinations = get_year_month_combinations()
        processed_something = False

        if year_month_combinations and not terminate_requested:
            # Sort combinations for ordered processing
            sorted_combinations = sorted(year_month_combinations)

            # Process one year-month combination at a time
            for idx, (year, month) in enumerate(sorted_combinations):
                # Check if termination was requested
                if terminate_requested:
                    logger.info("Termination requested, stopping processing")
                    break

                logger.info(f"Processing combination {idx + 1}/{len(sorted_combinations)}: {year}-{month}")

                try:
                    process_year_month(year, month)
                    processed_something = True
                except Exception as e:
                    logger.error(f"Error processing {year}-{month}: {e}", exc_info=True)
                    # Continue with next combination instead of terminating the entire process
                    continue

                # Check resources after each combination
                if not check_and_optimize_resources():
                    logger.warning("Resource check failed. Pausing before continuing...")
                    time.sleep(30)  # Longer pause to let system recover

                    # Check again
                    if not check_and_optimize_resources():
                        logger.error("Resource check failed twice. Requesting termination.")
                        terminate_requested = True
                        break

        elif not terminate_requested:
            logger.warning("No common year-month combinations found to process")

            # Check if files exist but patterns don't match
            logger.info("Attempting manual check for 2015-07 data...")

            # Check for specific files mentioned
            job_file = JOB_ACCOUNTING_PATH / "2015-07.csv"
            ts_file1 = PROC_METRIC_PATH / "FRESCO_Conte_ts_2015_07_v5.parquet"
            ts_file2 = PROC_METRIC_PATH / "FRESCO_Conte_ts_2015_07_v6.parquet"

            # Also check for chunked files
            chunk_files = list(PROC_METRIC_PATH.glob("FRESCO_Conte_ts_2015_07_v*_chunk*.parquet"))

            logger.info(f"Found {len(chunk_files)} chunked files")
            if chunk_files:
                for cf in chunk_files[:5]:  # Log first few to avoid flooding log
                    logger.info(f"Found chunked file: {cf}")

            logger.info(f"Checking if job file exists: {job_file}")
            if job_file.exists():
                logger.info(f"Job file exists: {job_file}")
            else:
                logger.warning(f"Job file does not exist: {job_file}")

            logger.info(f"Checking if TS file 1 exists: {ts_file1}")
            if ts_file1.exists():
                logger.info(f"TS file 1 exists: {ts_file1}")
            else:
                logger.warning(f"TS file 1 does not exist: {ts_file1}")

            logger.info(f"Checking if TS file 2 exists: {ts_file2}")
            if ts_file2.exists():
                logger.info(f"TS file 2 exists: {ts_file2}")
            else:
                logger.warning(f"TS file 2 does not exist: {ts_file2}")

            # If files exist but weren't detected by pattern matching, manually add them
            if job_file.exists() and (
                    ts_file1.exists() or ts_file2.exists() or chunk_files) and not terminate_requested:
                logger.info("Files exist but weren't detected by pattern matching. Processing 2015-07 manually.")
                try:
                    process_year_month("2015", "07")
                    processed_something = True
                except Exception as e:
                    logger.error(f"Error processing 2015-07 manually: {e}", exc_info=True)

        if processed_something and not terminate_requested:
            elapsed_time = time.time() - start_time
            logger.info(f"Data transformation process completed successfully in {elapsed_time:.2f} seconds")
        elif terminate_requested:
            elapsed_time = time.time() - start_time
            logger.info(f"Data transformation process was terminated after {elapsed_time:.2f} seconds")
        else:
            logger.warning("No data was processed. Please check your input directories and file patterns.")
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)

    if terminate_requested:
        logger.info("Clean exit after termination request")
        sys.exit(0)


if __name__ == "__main__":
    main()
