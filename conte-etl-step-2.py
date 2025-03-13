import queue
import signal
import sys
import pandas as pd
import numpy as np
import re
import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import time
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import gc
from queue import Queue
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("conte_server_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global flag to indicate termination
terminate_requested = False

# Define file paths - Updated to use absolute paths that match the ETL manager
CACHE_DIR = Path(r"/home/dynamo/a/jmckerra/projects/conte-to-fresco-etl/cache")
JOB_ACCOUNTING_PATH = Path(CACHE_DIR / 'accounting')
PROC_METRIC_PATH = Path(CACHE_DIR / 'input/metrics')
OUTPUT_PATH = Path(CACHE_DIR / 'output')

# Ensure directories exists
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
# We don't create input directories as they should already exist with data

# Configuration
MAX_WORKERS = 8
MIN_FREE_MEMORY_GB = 2.0
MIN_FREE_DISK_GB = 5.0
BASE_CHUNK_SIZE = 50_000
MAX_MEMORY_USAGE_GB = 25.0
MEMORY_CHECK_INTERVAL = 0.1
SMALL_FILE_THRESHOLD_MB = 20

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
        # Use isinstance instead of deprecated is_categorical_dtype
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

        # FIX: Ensure exit_status is converted to string before concatenation
        non_zero_mask = (jobevent == 'E') & (exit_status != '0')
        if non_zero_mask.any():
            # Explicitly convert to string before concatenation
            result.loc[non_zero_mask] = 'FAILED:' + exit_status[non_zero_mask].astype(str)

        result.loc[jobevent == 'A'] = 'ABORTED'
        result.loc[jobevent == 'S'] = 'STARTED'
        result.loc[jobevent == 'Q'] = 'QUEUED'

        # Handle remaining cases
        mask_other = ~result.notna()
        if mask_other.any():
            # Always convert to string before concatenation
            jobevent_str = jobevent[mask_other].astype(str)
            exit_status_str = exit_status[mask_other].astype(str)
            result.loc[mask_other] = jobevent_str + ':' + exit_status_str
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
                # Check if it's a categorical type that needs to be converted to string
                # Use isinstance instead of deprecated is_categorical_dtype
                if isinstance(df[col].dtype, pd.CategoricalDtype) or 'dictionary' in str(df[col].dtype):
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
    """Read a chunk from a parquet file with improved row group handling"""
    logger.debug(f"Reading chunk from {start_row} to {start_row + chunk_size}")

    try:
        # Use PyArrow's ParquetFile API to get metadata
        pf = pq.ParquetFile(file_path)
        total_rows = pf.metadata.num_rows

        # Check if the start_row is valid
        if start_row >= total_rows:
            logger.error(f"Start row {start_row} exceeds file length {total_rows}")
            return pd.DataFrame()

        # Calculate the actual end row
        end_row = min(start_row + chunk_size, total_rows)
        actual_chunk_size = end_row - start_row

        # Calculate row group boundaries
        row_group_offsets = [0]
        for i in range(pf.metadata.num_row_groups):
            row_group_offsets.append(row_group_offsets[-1] + pf.metadata.row_group(i).num_rows)

        # Find row groups that overlap with our range
        start_rg = None
        end_rg = None

        for i in range(len(row_group_offsets) - 1):
            if row_group_offsets[i] <= start_row < row_group_offsets[i + 1]:
                start_rg = i
            if row_group_offsets[i] < end_row <= row_group_offsets[i + 1]:
                end_rg = i
                break

        if start_rg is None:
            # If start row wasn't found, use the first row group
            start_rg = 0
            logger.warning(f"Could not find row group containing start row {start_row}, defaulting to first group")

        if end_rg is None:
            # If end row wasn't found, read until the last row group
            end_rg = pf.metadata.num_row_groups - 1

        logger.info(f"Reading row groups {start_rg} to {end_rg} for rows {start_row} to {end_row}")

        # Read the selected row groups
        tables = []
        for i in range(start_rg, end_rg + 1):
            try:
                tables.append(pf.read_row_group(i))
            except Exception as e:
                logger.error(f"Error reading row group {i}: {e}")

        if not tables:
            logger.error("No row groups could be read")
            return pd.DataFrame()

        # Combine the tables and convert to pandas
        table = pa.concat_tables(tables)
        all_rows_df = table.to_pandas()

        # Calculate the proper slice indices
        # For the start row, subtract the offset of the first row group we read
        local_start = start_row - row_group_offsets[start_rg]
        # Make sure we don't exceed the DataFrame length
        local_end = min(local_start + actual_chunk_size, len(all_rows_df))

        if local_start < 0:
            logger.warning(f"Calculated negative local start index {local_start}, adjusting to 0")
            local_start = 0

        if local_start >= len(all_rows_df):
            logger.error(f"Calculated local start index {local_start} exceeds DataFrame length {len(all_rows_df)}")
            return pd.DataFrame()

        # Extract the slice we want
        result_df = all_rows_df.iloc[local_start:local_end]
        logger.info(f"Successfully read {len(result_df)} rows from chunk ({start_row}:{end_row})")

        return result_df

    except Exception as e:
        logger.error(f"Error processing parquet file: {e}")

        # Last resort fallback - try to read the whole file
        try:
            logger.warning("Falling back to reading the entire file - this may use more memory")
            full_df = pd.read_parquet(file_path)

            if start_row < len(full_df):
                chunk_end = min(start_row + chunk_size, len(full_df))
                result_df = full_df.iloc[start_row:chunk_end].copy()

                # Free memory immediately
                del full_df
                gc.collect()

                return result_df
            else:
                logger.error(f"Start row {start_row} exceeds DataFrame length {len(full_df)}")
                return pd.DataFrame()

        except Exception as fallback_error:
            logger.error(f"All approaches to read parquet chunk failed: {fallback_error}")
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
    for f in proc_metric_files[:10]:  # Log only first 10 to avoid too much output
        logger.info(f"  Found parquet file: {f.name}")

    if len(proc_metric_files) > 10:
        logger.info(f"  ... and {len(proc_metric_files) - 10} more parquet files")

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
            logger.debug(f"Matched FRESCO pattern: {year}-{month} from {filename}")
            continue

        # Try the other pattern
        matches = other_pattern.findall(filename)
        if matches:
            year, month = matches[0]
            proc_metrics_years_months.add((year, month))
            logger.debug(f"Matched other pattern: {year}-{month} from {filename}")

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
            logger.debug(f"Matched job pattern: {year}-{month} from {filename}")

    # Find common year-month combinations
    common_years_months = proc_metrics_years_months.intersection(job_accounting_years_months)
    logger.info(f"Found {len(common_years_months)} common year-month combinations")
    for year, month in sorted(common_years_months):
        logger.info(f"  Found common year-month: {year}-{month}")

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
    """Process a single time series file with parallel processing"""
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

    # Create a proper PyArrow schema
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

    try:
        # Get the file size to estimate appropriate chunk count
        file_size = os.path.getsize(ts_file)
        file_size_mb = file_size / (1024 * 1024)

        # Try to read just the file metadata without loading everything
        parquet_file = pq.ParquetFile(ts_file)
        total_rows = parquet_file.metadata.num_rows

        # Calculate a reasonable chunk size based on available memory and cores
        available_memory_gb = get_available_memory()

        # Use more aggressive parallelism if we have memory available
        num_workers = min(MAX_WORKERS, os.cpu_count() or 4)

        # Adjust workers based on available memory
        memory_per_worker = available_memory_gb / (num_workers * 2)  # Conservative estimate
        if memory_per_worker < 1.0:  # If less than 1GB per worker
            num_workers = max(1, int(available_memory_gb / 2))
            logger.info(f"Reducing worker count to {num_workers} due to memory constraints")

        logger.info(f"Using {num_workers} parallel workers for processing")

        # Calculate chunk size based on file size and worker count
        estimated_bytes_per_row = file_size / total_rows if total_rows > 0 else 1000

        # For very small files, use a smaller chunk size to ensure all workers get used
        if file_size_mb < SMALL_FILE_THRESHOLD_MB:
            # Divide evenly across workers
            chunk_size = max(1000, total_rows // (num_workers * 2))
            logger.info(f"Small file, using smaller chunk size of {chunk_size} rows per chunk")
        else:
            # For larger files, calculate a reasonable chunk size
            safe_rows_per_worker = int((memory_per_worker * 1024 * 1024 * 1024) / estimated_bytes_per_row)
            chunk_size = max(10_000, min(safe_rows_per_worker, 200_000))

        # Calculate number of chunks (at least num_workers chunks to utilize all workers)
        num_chunks = max(num_workers, (total_rows + chunk_size - 1) // chunk_size)

        # Recalculate chunk size to ensure even distribution
        chunk_size = (total_rows + num_chunks - 1) // num_chunks

        logger.info(
            f"File has {total_rows} rows, processing with {num_workers} workers in {num_chunks} chunks of {chunk_size} rows")

        # Create a thread-safe queue for results
        result_queue = Queue()

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            # Submit all chunks for processing
            for chunk_idx in range(num_chunks):
                # Check termination
                if terminate_requested:
                    logger.info("Termination requested, stopping submission of new chunks")
                    break

                # Calculate row range for this chunk
                start_row = chunk_idx * chunk_size
                end_row = min(start_row + chunk_size, total_rows)
                actual_chunk_size = end_row - start_row

                # Submit the chunk for processing
                future = executor.submit(
                    process_chunk_with_queue,
                    ts_file,
                    jobs_df,
                    start_row,
                    actual_chunk_size,
                    chunk_idx + 1,
                    result_queue
                )
                futures.append(future)

                # Small delay between submissions to avoid memory spikes
                time.sleep(0.1)

            # Process results as they complete
            completed = 0
            total_submitted = len(futures)

            while completed < total_submitted and not terminate_requested:
                try:
                    # Get a result with timeout
                    result_df = result_queue.get(timeout=1)
                    completed += 1

                    # Process the result
                    if result_df is not None and not result_df.empty:
                        # Validate and enforce schema
                        result_df = validate_dataframe_for_schema(result_df, expected_columns)
                        result_df = enforce_schema_types(result_df, schema_column_types)

                        try:
                            # Reset index and convert to PyArrow table
                            result_df = result_df.reset_index(drop=True)
                            table = pa.Table.from_pandas(result_df, schema=schema)

                            # Write to output
                            output_writer.write_table(table)
                            logger.info(
                                f"Successfully wrote chunk result with {len(result_df)} rows to output ({completed}/{total_submitted})")
                        except Exception as e:
                            logger.error(f"Error writing to parquet: {e}")
                            logger.error(f"Result DataFrame dtypes: {result_df.dtypes}")

                        # Release memory
                        del result_df
                        gc.collect()

                    # Check resources periodically
                    if completed % max(1, num_workers // 2) == 0:
                        check_and_optimize_resources()

                except queue.Empty:
                    # Check if all futures are done
                    if all(future.done() for future in futures):
                        break
                    continue
                except Exception as e:
                    logger.error(f"Error processing result: {e}")
                    completed += 1
                    continue

            # Wait for all futures to complete
            for future in futures:
                try:
                    future.result(timeout=1)
                except Exception:
                    # Already logged in the worker
                    pass

        return not terminate_requested

    except Exception as e:
        logger.error(f"Error processing file {ts_file}: {e}")
        return False


def process_chunk_with_queue(ts_file, jobs_df, start_row, chunk_size, chunk_id, result_queue):
    """Process a chunk and put the result in a queue"""
    thread_id = threading.get_ident()
    logger.info(f"Thread {thread_id}: Starting chunk {chunk_id} ({start_row}:{start_row + chunk_size})")

    try:
        # Read the chunk
        chunk_df = read_parquet_chunk(ts_file, start_row, chunk_size)

        if chunk_df is None or chunk_df.empty:
            logger.warning(f"Thread {thread_id}: Empty chunk {chunk_id}")
            result_queue.put(None)
            return

        # Optimize immediately
        chunk_df = optimize_dataframe_dtypes(chunk_df)

        # Process the chunk
        result_df = process_chunk(jobs_df, chunk_df, chunk_id)

        # Clean up input chunk
        del chunk_df
        gc.collect()

        # Put result in queue
        if result_df is not None and not result_df.empty:
            result_queue.put(result_df)
            logger.info(f"Thread {thread_id}: Completed chunk {chunk_id} with {len(result_df)} rows")
        else:
            result_queue.put(None)
            logger.info(f"Thread {thread_id}: Completed chunk {chunk_id} with no results")

    except Exception as e:
        logger.error(f"Thread {thread_id}: Error processing chunk {chunk_id}: {e}")
        result_queue.put(None)


def process_year_month(year, month):
    """Process a specific year-month combination with parallel processing for all files"""
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

        # Find all time series files for this year/month (including chunked files)
        ts_files = []
        fresco_pattern = f"FRESCO_Conte_ts_{year}_{month}"

        # Find all matching files (original and chunked)
        all_files = list(PROC_METRIC_PATH.glob(f"{fresco_pattern}*.parquet"))

        # Log found files
        logger.info(f"Found {len(all_files)} time series files for {year}-{month}")

        # Check backup pattern if needed
        if not all_files:
            backup_files = list(PROC_METRIC_PATH.glob(f"*_{year}_{month}_*.parquet"))
            all_files.extend(backup_files)
            if backup_files:
                logger.info(f"Found {len(backup_files)} additional files with backup pattern")

        if all_files:
            # Sort files logically - chunked files after originals, then by version, then by chunk number
            try:
                for f in sorted(all_files, key=lambda x: (
                        1 if "_chunk" in x.name else 0,
                        int(re.search(r'v(\d+)', x.name).group(1)) if re.search(r'v(\d+)', x.name) else 0,
                        int(re.search(r'chunk(\d+)', x.name).group(1)) if re.search(r'chunk(\d+)', x.name) else 0,
                        x.name
                )):
                    ts_files.append(f)
            except Exception as e:
                logger.warning(f"Error sorting files, using unsorted list: {e}")
                ts_files = all_files

        # Log what we found
        logger.info(f"Processing {len(ts_files)} files for {year}-{month}")
        for ts_file in ts_files[:5]:  # Log first 5 to avoid too much output
            logger.info(f"Found time series file: {ts_file.name}")
        if len(ts_files) > 5:
            logger.info(f"... and {len(ts_files) - 5} more files")

        if not ts_files:
            logger.warning(f"No time series files found for {year}-{month}")
            return

        # Create output file path
        output_file = OUTPUT_PATH / f"transformed_{year}_{month}.parquet"

        # Read job data
        logger.info(f"Reading job data from file: {job_file}")
        jobs_df = read_jobs_df(job_file)

        # Log memory usage
        mem_usage = jobs_df.memory_usage(deep=True).sum() / (1024 * 1024 * 1024)
        logger.info(f"Job data loaded into dataframe with {len(jobs_df)} rows - Memory usage: {mem_usage:.2f} GB")

        # Standardize job IDs
        if "jobID" in jobs_df.columns and not pd.api.types.is_datetime64_any_dtype(jobs_df["jobID"]):
            jobs_df["jobID"] = standardize_job_id(jobs_df["jobID"])

        # Create PyArrow schema
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

        # Create output writer
        with pq.ParquetWriter(str(output_file), schema, compression='snappy') as writer:
            # Process each file
            for i, ts_file in enumerate(ts_files):
                if terminate_requested:
                    logger.info("Termination requested, stopping file processing")
                    break

                logger.info(f"Processing file {i + 1}/{len(ts_files)}: {ts_file.name}")

                # Process the file using parallel processing
                success = process_ts_file_in_parallel(ts_file, jobs_df, writer)

                # Force garbage collection between files
                gc.collect()
                check_and_optimize_resources()

                if not success:
                    logger.warning(f"Processing of {ts_file} was not successful, moving to next file")

        # Check output
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

    logger.info("Starting Conte data transformation process")
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

            # Manual check for specific Conte files (similar to fallback logic but for Conte)
            logger.info("Attempting manual check for available Conte data...")

            # We'll check for all possible year-month combinations in the job accounting dir
            for job_file in JOB_ACCOUNTING_PATH.glob("*.csv"):
                if terminate_requested:
                    break

                match = re.match(r'(\d{4})-(\d{2})\.csv', job_file.name)
                if match:
                    year, month = match.groups()

                    # Check for corresponding time series files
                    ts_pattern = f"FRESCO_Conte_ts_{year}_{month}"
                    ts_files = list(PROC_METRIC_PATH.glob(f"{ts_pattern}*.parquet"))

                    if ts_files:
                        logger.info(f"Found data for {year}-{month} during manual check")
                        try:
                            process_year_month(year, month)
                            processed_something = True
                        except Exception as e:
                            logger.error(f"Error processing {year}-{month} during manual check: {e}", exc_info=True)
                            continue

        if processed_something and not terminate_requested:
            elapsed_time = time.time() - start_time
            logger.info(f"Conte data transformation process completed successfully in {elapsed_time:.2f} seconds")
        elif terminate_requested:
            elapsed_time = time.time() - start_time
            logger.info(f"Conte data transformation process was terminated after {elapsed_time:.2f} seconds")
        else:
            logger.warning("No data was processed. Please check your input directories and file patterns.")
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)

    if terminate_requested:
        logger.info("Clean exit after termination request")
        sys.exit(0)


if __name__ == "__main__":
    main()