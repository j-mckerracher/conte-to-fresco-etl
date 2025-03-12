# The key fixes
# 1. Fix the path to job accounting files
# 2. Make sure directories are created
# 3. Fix the file matching pattern
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

# 1. Adjust these global settings to be more memory-conservative
MAX_WORKERS = 1  # Reduce from whatever it is currently set to
MIN_FREE_MEMORY_GB = 2.0  # Lower minimum free memory threshold to be more aggressive
MIN_FREE_DISK_GB = 5.0  # Keep as is
BASE_CHUNK_SIZE = 50_000  # Reduce from 100,000 to 50,000

# Adjust constants and add memory management controls
MAX_MEMORY_USAGE_GB = 30.0  # Set maximum allowed memory usage
MEMORY_CHECK_INTERVAL = 0.1  # Check memory usage every 10% of chunks


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


def memory_monitor():
    """Thread to monitor memory usage and set terminate flag if needed"""
    global terminate_requested

    while not terminate_requested:
        current_memory = get_memory_usage()
        available_memory = get_available_memory()

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


def standardize_job_id(job_id_series):
    """Convert jobIDxxxxx to JOBxxxxx"""
    return job_id_series.str.replace(r'^jobID', 'JOB', regex=True)


def convert_walltime_to_seconds(walltime_series):
    """Vectorized conversion of HH:MM:SS format to seconds"""
    if walltime_series.empty:
        return walltime_series

    # Handle NaN values
    mask_na = walltime_series.isna()
    result = pd.Series(index=walltime_series.index, dtype=float)
    result[mask_na] = np.nan

    # Only process non-NaN values
    to_process = walltime_series[~mask_na]

    # Handle numeric values
    mask_numeric = pd.to_numeric(to_process, errors='coerce').notna()
    result.loc[to_process[mask_numeric].index] = pd.to_numeric(to_process[mask_numeric])

    # Handle string time formats
    str_times = to_process[~mask_numeric]

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

    return result


def parse_host_list(exec_host_series):
    """Parse exec_host into a list of hosts in JSON format"""
    # Handle empty series
    if exec_host_series.empty:
        return exec_host_series

    # Create a mask for non-null string values
    mask = exec_host_series.notna() & exec_host_series.apply(lambda x: isinstance(x, str))

    # Initialize result series with same index and None values
    result = pd.Series([None] * len(exec_host_series), index=exec_host_series.index)

    if mask.any():
        # Extract node names using regular expression
        node_pattern = re.compile(r'([^/+]+)/')

        # Apply the extraction only on valid strings
        valid_hosts = exec_host_series[mask]

        # Extract all matches for each string
        extracted = valid_hosts.apply(lambda x: node_pattern.findall(x))

        # Get unique nodes and convert to JSON format
        unique_nodes = extracted.apply(lambda x: json.dumps(list(set(x))).replace('"', '') if x else None)

        # Update only the processed values
        result[mask] = unique_nodes

    return result


def get_exit_status_description(df):
    """Vectorized conversion of exit status to descriptive text"""
    # Check if required columns exist
    if 'jobevent' not in df.columns or 'Exit_status' not in df.columns:
        return pd.Series([None] * len(df), index=df.index)

    # Initialize with empty strings
    jobevent = df['jobevent'].fillna('')
    exit_status = df['Exit_status'].fillna('')

    # Create result series
    result = pd.Series(index=df.index, dtype='object')

    # Apply conditions using vectorized operations
    result[(jobevent == 'E') & (exit_status == '0')] = 'COMPLETED'
    result[(jobevent == 'E') & (exit_status != '0')] = 'FAILED:' + exit_status[(jobevent == 'E') & (exit_status != '0')]
    result[jobevent == 'A'] = 'ABORTED'
    result[jobevent == 'S'] = 'STARTED'
    result[jobevent == 'Q'] = 'QUEUED'

    # Handle remaining cases
    mask_other = ~result.notna()
    if mask_other.any():
        result[mask_other] = jobevent[mask_other] + ':' + exit_status[mask_other]

    return result


def optimize_dataframe_dtypes(df):
    """More aggressively optimize DataFrame memory usage by changing data types"""
    # First, optimize object columns to categorical for significant memory savings
    object_columns = df.select_dtypes(include=['object']).columns

    for col in object_columns:
        n_unique = df[col].nunique()

        # Check if this column is a good candidate for categorical
        if n_unique is not None:  # Avoid NaN issues
            # More aggressive categorical conversion - use 70% threshold instead of 50%
            if n_unique < 0.7 * len(df):
                df[col] = df[col].astype('category')
                continue

        # For strings, try to convert to numeric when possible
        try:
            num_values = pd.to_numeric(df[col], errors='coerce')
            # If most values can be converted to numeric, convert the column
            if num_values.notna().sum() / len(df) > 0.7:  # More aggressive threshold
                # Attempt integer conversion first as it uses less memory
                if (num_values.dropna() % 1 == 0).all():
                    # It's an integer
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                else:
                    # It's a float
                    df[col] = pd.to_numeric(df[col], downcast='float')
        except (TypeError, ValueError):
            pass

    # Optimize numeric columns
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].notna().all() and (df[col] % 1 == 0).all():
            # For integers, use the smallest possible integer type
            df[col] = pd.to_numeric(df[col], downcast='integer')
        else:
            # For floats, use the smallest possible float type
            df[col] = pd.to_numeric(df[col], downcast='float')

    return df


def process_chunk(jobs_df, ts_chunk, chunk_id):
    """Process a chunk of time series data against all jobs with reduced memory usage"""
    thread_id = threading.get_ident()
    logger.debug(f"Thread {thread_id}: Processing chunk {chunk_id} with {len(ts_chunk)} rows")

    # Ensure we're only keeping the columns we'll actually need from ts_chunk
    required_ts_columns = ["Timestamp", "Job Id", "Event", "Value", "Host", "Units"]
    ts_columns_to_drop = [col for col in ts_chunk.columns if col not in required_ts_columns]
    if ts_columns_to_drop:
        ts_chunk.drop(columns=ts_columns_to_drop, inplace=True)

    # Ensure Timestamp is datetime
    if "Timestamp" in ts_chunk.columns and not pd.api.types.is_datetime64_any_dtype(ts_chunk["Timestamp"]):
        ts_chunk["Timestamp"] = pd.to_datetime(ts_chunk["Timestamp"], errors='coerce')

    # Keep only necessary columns from jobs_df for this join
    required_job_columns = ["jobID", "start", "end", "qtime", "Resource_List.walltime",
                            "Resource_List.nodect", "Resource_List.ncpus", "account",
                            "queue", "jobname", "user", "group", "exec_host",
                            "jobevent", "Exit_status"]

    # Only keep columns that actually exist in jobs_df
    job_columns_to_use = [col for col in required_job_columns if col in jobs_df.columns]
    jobs_subset = jobs_df[job_columns_to_use]

    # Join time series chunk with jobs data on jobID - do this in-place to save memory
    joined = pd.merge(
        ts_chunk,
        jobs_subset,
        left_on="Job Id",
        right_on="jobID",
        how="inner"
    )

    # Release memory from unnecessary dataframes
    del ts_chunk
    del jobs_subset
    gc.collect()  # Force garbage collection

    logger.debug(f"Thread {thread_id}: Joined dataframe has {len(joined)} rows")

    if joined.empty:
        return None

    # Filter timestamps that fall between job start and end times
    mask = (joined["Timestamp"] >= joined["start"]) & (joined["Timestamp"] <= joined["end"])

    # Apply the filter in-place instead of creating a copy
    joined = joined.loc[mask]

    logger.debug(f"Thread {thread_id}: Filtered dataframe has {len(joined)} rows")

    if joined.empty:
        return None

    # Process events more efficiently to reduce memory usage
    events = joined['Event'].unique()

    # Create only the necessary columns with NaN values
    for event in events:
        if event in ["cpuuser", "gpu_usage", "memused", "memused_minus_diskcache", "nfs", "block"]:
            col_name = f'value_{event}'
        else:
            col_name = event

        # Create the column with NaN values
        joined[col_name] = np.nan

        # Fill values only for matching event rows (in-place)
        event_mask = joined['Event'] == event
        joined.loc[event_mask, col_name] = joined.loc[event_mask, 'Value']

    # Drop Event and Value columns to save memory
    joined.drop(columns=['Event', 'Value'], inplace=True)

    # Map columns efficiently
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
        "group": "account",  # Using group as account based on example

        # Core identifiers
        "jobID": "jid",
        "Host": "host",
        "Units": "unit"
    }

    # Only rename columns that exist
    existing_cols = {col: mapping for col, mapping in column_mapping.items() if col in joined.columns}
    joined.rename(columns=existing_cols, inplace=True)  # Rename in-place

    # Process special cases
    if "timelimit" in joined.columns:
        joined["timelimit"] = convert_walltime_to_seconds(joined["timelimit"])

    if "host_list" in joined.columns:
        joined["host_list"] = parse_host_list(joined["host_list"])

    # Generate exitcode from jobevent and Exit_status using vectorized approach
    if "jobevent" in joined.columns:
        joined["exitcode"] = get_exit_status_description(joined)

        # Remove the source columns after processing
        columns_to_drop = [col for col in ["jobevent", "Exit_status"] if col in joined.columns]
        if columns_to_drop:
            joined.drop(columns=columns_to_drop, inplace=True)

    # Ensure all required columns exist in the output
    set3_columns = ["time", "submit_time", "start_time", "end_time", "timelimit",
                    "nhosts", "ncores", "account", "queue", "host", "jid", "unit",
                    "jobname", "exitcode", "host_list", "username",
                    "value_cpuuser", "value_gpu_usage", "value_memused",
                    "value_memused_minus_diskcache", "value_nfs", "value_block"]

    # Add missing columns
    for col in set3_columns:
        if col not in joined.columns:
            joined[col] = np.nan

    # Convert datetime columns to UTC timezone
    datetime_cols = ["time", "submit_time", "start_time", "end_time"]
    for col in datetime_cols:
        if col in joined.columns and joined[col].notna().any():
            try:
                # Check if timezone info already exists
                sample_dt = joined.loc[joined[col].first_valid_index(), col]
                if hasattr(sample_dt, 'tzinfo') and sample_dt.tzinfo is None:
                    joined[col] = joined[col].dt.tz_localize('UTC')
            except (TypeError, AttributeError):
                logger.warning(f"Failed to localize timezone for {col}. Skipping.")

    # Select only the columns needed for output and ensure correct order
    available_columns = [col for col in set3_columns if col in joined.columns]
    result = joined[available_columns]

    # Optimize the result dataframe
    result = optimize_dataframe_dtypes(result)

    logger.info(f"Thread {thread_id}: Completed processing chunk {chunk_id} - produced {len(result)} rows")
    return result


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
    if free_memory_gb < MIN_FREE_MEMORY_GB:
        logger.warning(f"Low memory: {free_memory_gb:.2f}GB free. Forcing garbage collection...")
        gc.collect()


def get_year_month_combinations():
    """Get all year-month combinations from both local directories"""
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
    fresco_pattern = re.compile(r'FRESCO_Conte_ts_(\d{4})_(\d{2})_v\d+\.parquet')
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


def process_ts_file_in_parallel(ts_file, jobs_df, output_writer):
    """Process a single time series file with reduced memory usage"""
    global terminate_requested
    logger.info(f"Processing TS file: {ts_file}")

    try:
        # Get the file size to estimate appropriate chunk count
        file_size = os.path.getsize(ts_file)

        # Try to read just the file metadata without loading everything
        parquet_file = pq.ParquetFile(ts_file)
        total_rows = parquet_file.metadata.num_rows

        # Use more conservative chunk sizing
        available_memory_gb = get_available_memory()
        current_memory_gb = get_memory_usage()

        # Dynamic chunk sizing based on current memory situation
        max_allowed_increase = min(2.0, available_memory_gb / 4)  # At most 2GB increase or 1/4 of available
        estimated_bytes_per_row = file_size / total_rows if total_rows > 0 else 1000

        # Calculate how many rows we can safely process in one chunk
        safe_rows = int((max_allowed_increase * 1024 * 1024 * 1024) / estimated_bytes_per_row / 2)

        # Clamp chunk size between more conservative bounds
        chunk_size = max(10_000, min(safe_rows, 200_000))

        # Calculate number of chunks
        num_chunks = (total_rows + chunk_size - 1) // chunk_size  # Ceiling division

        # Use just 1 worker for more predictable memory usage
        num_workers = 1

        logger.info(
            f"File has {total_rows} rows, processing with {num_workers} thread in {num_chunks} chunks of {chunk_size} rows")

        # Process file in chunks sequentially to better control memory
        # Only submit one chunk at a time to executor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            # For each chunk, process sequentially
            for chunk_idx in range(num_chunks):
                # Check if termination was requested
                if terminate_requested:
                    logger.info("Termination requested, stopping processing of new chunks")
                    break

                # Check memory before processing each chunk
                current_memory = get_memory_usage()
                available_memory = get_available_memory()

                if current_memory > MAX_MEMORY_USAGE_GB:
                    logger.warning(f"Memory usage too high: {current_memory:.2f}GB. Pausing chunk processing.")
                    # Force garbage collection
                    gc.collect()
                    time.sleep(5)  # Wait for memory to be freed

                    # If still too high, terminate
                    if get_memory_usage() > MAX_MEMORY_USAGE_GB:
                        logger.warning("Memory still too high after cleanup. Requesting termination.")
                        terminate_requested = True
                        break

                # Calculate row range for this chunk
                start_row = chunk_idx * chunk_size
                end_row = min(start_row + chunk_size, total_rows)

                try:
                    # Use row groups if possible for more efficient reading
                    chunk_df = None
                    try:
                        if chunk_idx < parquet_file.metadata.num_row_groups:
                            chunk_df = parquet_file.read_row_group(chunk_idx).to_pandas()
                        else:
                            # Read slice using PyArrow for better memory efficiency
                            table_slice = pq.read_table(
                                ts_file,
                                use_threads=False,  # Disable threads for more predictable memory usage
                                columns=None,  # Read all columns
                                filters=[('row_index', '>=', start_row), ('row_index', '<', end_row)]
                            )
                            chunk_df = table_slice.to_pandas()
                    except Exception as e:
                        logger.warning(f"First approach failed: {e}, trying alternative")
                        # Fall back to reading slice from disk in a different way
                        table = pq.read_table(
                            ts_file,
                            use_threads=False,
                            columns=None,
                        )
                        chunk_df = table.slice(start_row, min(chunk_size, end_row - start_row)).to_pandas()
                        del table  # Free memory immediately
                        gc.collect()  # Force garbage collection

                    # Optimize chunk data types immediately to reduce memory
                    if chunk_df is not None and not chunk_df.empty:
                        # Apply memory optimizations immediately
                        chunk_df = optimize_dataframe_dtypes(chunk_df)

                        # Process one chunk at a time and wait for it
                        future = executor.submit(process_chunk, jobs_df, chunk_df, chunk_idx + 1)
                        futures.append(future)

                        # Clean up chunk dataframe to free memory
                        del chunk_df
                        gc.collect()

                        # Wait for this chunk to complete before starting next one
                        result_df = future.result()
                        if result_df is not None and not result_df.empty:
                            # Convert to PyArrow table and write to output
                            table = pa.Table.from_pandas(result_df)
                            output_writer.write_table(table)

                            # Release memory
                            del result_df, table
                            futures = []  # Clear processed future
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


def read_jobs_df(job_file):
    """Read job data from CSV with optimized memory usage from the start"""
    # First, determine column data types by scanning sample
    sample_size = min(1000, os.path.getsize(job_file) // 2)
    sample_df = pd.read_csv(job_file, nrows=1000)

    # Analyze column types in sample
    dtypes = {}
    categorical_columns = []

    for col in sample_df.columns:
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

    # Convert datetime columns - but use optimized approach
    datetime_cols = ["ctime", "qtime", "etime", "start", "end", "timestamp"]
    for col in datetime_cols:
        if col in jobs_df.columns:
            jobs_df[col] = pd.to_datetime(jobs_df[col], format="%m/%d/%Y %H:%M:%S", errors='coerce', cache=True)

    return jobs_df


def process_year_month(year, month):
    """Process a specific year-month combination with intra-file parallelism"""
    logger.info(f"Processing year: {year}, month: {month}")

    # Check resources before starting
    check_and_optimize_resources()

    try:
        # Get job accounting file (CSV)
        job_file = JOB_ACCOUNTING_PATH / f"{year}-{month}.csv"

        # Check if job file exists
        if not job_file.exists():
            logger.error(f"Job file not found: {job_file}")
            return

        # Find all time series files (Parquet) for this year/month
        # Modified pattern to match FRESCO_Conte_ts_2015_07_v5.parquet format
        ts_files = []

        # Try the FRESCO pattern first
        fresco_files = list(PROC_METRIC_PATH.glob(f"FRESCO_Conte_ts_{year}_{month}_*.parquet"))
        if fresco_files:
            ts_files.extend(fresco_files)
            logger.info(f"Found {len(fresco_files)} FRESCO pattern files")

        # Try other pattern as backup
        other_files = list(PROC_METRIC_PATH.glob(f"*_{year}_{month}_*.parquet"))
        new_files = [f for f in other_files if f not in ts_files]
        if new_files:
            ts_files.extend(new_files)
            logger.info(f"Found {len(new_files)} additional files with other pattern")

        if not ts_files:
            logger.warning(f"No time series files found for {year}-{month}")
            # List all files in directory to help debug
            all_files = list(PROC_METRIC_PATH.glob("*.parquet"))
            logger.info(f"Available parquet files in directory: {[f.name for f in all_files]}")
            return

        # Log the files we found
        for ts_file in ts_files:
            logger.info(f"Found time series file: {ts_file}")

        # Create output file path (will be Parquet)
        output_file = OUTPUT_PATH / f"transformed_{year}_{month}.parquet"

        # Read job data from CSV file
        logger.info(f"Reading job data from file: {job_file}")

        # Read with optimized dtypes
        jobs_df = pd.read_csv(job_file, low_memory=False)

        # Optimize datatypes for memory efficiency
        jobs_df = optimize_dataframe_dtypes(jobs_df)

        # Handle problematic large integers
        for col in jobs_df.select_dtypes(include=['int64']).columns:
            # Check if any values exceed C long limit
            try:
                if jobs_df[col].max() > 2147483647 or jobs_df[col].min() < -2147483648:
                    jobs_df[col] = jobs_df[col].astype(str)
            except (TypeError, ValueError, OverflowError):
                jobs_df[col] = jobs_df[col].astype(str)

        # Standardize job IDs
        if "jobID" in jobs_df.columns:
            jobs_df["jobID"] = standardize_job_id(jobs_df["jobID"])
        else:
            logger.warning("jobID column not found in CSV file")

        # Convert datetime columns
        datetime_cols = ["ctime", "qtime", "etime", "start", "end", "timestamp"]
        for col in datetime_cols:
            if col in jobs_df.columns:
                jobs_df[col] = pd.to_datetime(jobs_df[col], format="%m/%d/%Y %H:%M:%S", errors='coerce')

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
            # Process each file sequentially, but with internal parallelism
            for i, ts_file in enumerate(ts_files):
                logger.info(f"Processing file {i + 1}/{len(ts_files)}: {ts_file.name}")
                process_ts_file_in_parallel(ts_file, jobs_df, writer)

                # Force garbage collection between files
                gc.collect()
                check_and_optimize_resources()

        # Check if the output file was created properly
        if output_file.exists() and output_file.stat().st_size > 0:
            logger.info(f"Successfully created {output_file}")
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

    logger.info("Starting data transformation process")
    start_time = time.time()

    try:
        # Start memory monitoring thread
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()
        # Initial cleanup - don't clean input data
        # clean_up_cache()

        # Get all year-month combinations
        year_month_combinations = get_year_month_combinations()
        processed_something = False

        if year_month_combinations and not terminate_requested:
            # Sort combinations for ordered processing
            sorted_combinations = sorted(year_month_combinations)

            # Process one year-month combination at a time (as requested)
            for idx, (year, month) in enumerate(tqdm(sorted_combinations, desc="Processing year-month combinations")):
                # Check if termination was requested
                if terminate_requested:
                    logger.info("Termination requested, stopping processing")
                    break

                logger.info(f"Processing combination {idx + 1}/{len(sorted_combinations)}: {year}-{month}")
                process_year_month(year, month)
                processed_something = True

                # Check resources after each combination
                check_and_optimize_resources()
        elif not terminate_requested:
            logger.warning("No common year-month combinations found to process")

            # Check if files exist but patterns don't match
            logger.info("Attempting manual check for 2015-07 data...")

            # Check for specific files mentioned
            job_file = JOB_ACCOUNTING_PATH / "2015-07.csv"
            ts_file1 = PROC_METRIC_PATH / "FRESCO_Conte_ts_2015_07_v5.parquet"
            ts_file2 = PROC_METRIC_PATH / "FRESCO_Conte_ts_2015_07_v6.parquet"

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
            if job_file.exists() and (ts_file1.exists() or ts_file2.exists()) and not terminate_requested:
                logger.info("Files exist but weren't detected by pattern matching. Processing 2015-07 manually.")
                process_year_month("2015", "07")
                processed_something = True

        # Don't clean up input data
        # clean_up_cache()

        if processed_something and not terminate_requested:
            elapsed_time = time.time() - start_time
            logger.info(f"Data transformation process completed successfully in {elapsed_time:.2f} seconds")
        elif terminate_requested:
            elapsed_time = time.time() - start_time
            logger.info(f"Data transformation process was terminated after {elapsed_time:.2f} seconds")
        else:
            logger.warning("No data was processed. Please check your input directories and file patterns.")
    except Exception as e:
        logger.error(f"Error: {e}")

    if terminate_requested:
        logger.info("Clean exit after termination request")
        sys.exit(0)


def process_year_month(year, month):
    """Process a specific year-month combination with intra-file parallelism"""
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

        # Find all time series files (Parquet) for this year/month
        # Modified pattern to match FRESCO_Conte_ts_2015_07_v5.parquet format
        ts_files = []

        # Try the FRESCO pattern first
        fresco_files = list(PROC_METRIC_PATH.glob(f"FRESCO_Conte_ts_{year}_{month}_*.parquet"))
        if fresco_files:
            ts_files.extend(fresco_files)
            logger.info(f"Found {len(fresco_files)} FRESCO pattern files")

        # Try other pattern as backup
        other_files = list(PROC_METRIC_PATH.glob(f"*_{year}_{month}_*.parquet"))
        new_files = [f for f in other_files if f not in ts_files]
        if new_files:
            ts_files.extend(new_files)
            logger.info(f"Found {len(new_files)} additional files with other pattern")

        if not ts_files:
            logger.warning(f"No time series files found for {year}-{month}")
            # List all files in directory to help debug
            all_files = list(PROC_METRIC_PATH.glob("*.parquet"))
            logger.info(f"Available parquet files in directory: {[f.name for f in all_files]}")
            return

        # Log the files we found
        for ts_file in ts_files:
            logger.info(f"Found time series file: {ts_file}")

        # Create output file path (will be Parquet)
        output_file = OUTPUT_PATH / f"transformed_{year}_{month}.parquet"

        # Read job data from CSV file
        logger.info(f"Reading job data from file: {job_file}")

        # Read with optimized dtypes
        jobs_df = pd.read_csv(job_file, low_memory=False)

        # Optimize datatypes for memory efficiency
        jobs_df = optimize_dataframe_dtypes(jobs_df)

        # Handle problematic large integers
        for col in jobs_df.select_dtypes(include=['int64']).columns:
            # Check if any values exceed C long limit
            try:
                if jobs_df[col].max() > 2147483647 or jobs_df[col].min() < -2147483648:
                    jobs_df[col] = jobs_df[col].astype(str)
            except (TypeError, ValueError, OverflowError):
                jobs_df[col] = jobs_df[col].astype(str)

        # Standardize job IDs
        if "jobID" in jobs_df.columns:
            jobs_df["jobID"] = standardize_job_id(jobs_df["jobID"])
        else:
            logger.warning("jobID column not found in CSV file")

        # Convert datetime columns
        datetime_cols = ["ctime", "qtime", "etime", "start", "end", "timestamp"]
        for col in datetime_cols:
            if col in jobs_df.columns:
                jobs_df[col] = pd.to_datetime(jobs_df[col], format="%m/%d/%Y %H:%M:%S", errors='coerce')

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
            # Process each file sequentially, but with internal parallelism
            for i, ts_file in enumerate(ts_files):
                # Check if termination was requested
                if terminate_requested:
                    logger.info("Termination requested, stopping file processing")
                    break

                logger.info(f"Processing file {i + 1}/{len(ts_files)}: {ts_file.name}")
                process_ts_file_in_parallel(ts_file, jobs_df, writer)

                # Force garbage collection between files
                gc.collect()
                check_and_optimize_resources()

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


if __name__ == "__main__":
    main()
