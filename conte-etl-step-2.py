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

# Define file paths
CACHE_DIR = Path("./cache")
JOB_ACCOUNTING_PATH = Path(CACHE_DIR / 'input/accounting')
PROC_METRIC_PATH = Path(CACHE_DIR / 'input/metrics')
OUTPUT_PATH = Path(CACHE_DIR / 'output')

# Ensure directories exists
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_PATH.mkdir(exist_ok=True)
PROC_METRIC_PATH.mkdir(exist_ok=True)
JOB_ACCOUNTING_PATH.mkdir(exist_ok=True)

# Configuration
MAX_WORKERS = max(1, mp.cpu_count() - 1)  # Leave one CPU core free
MIN_FREE_MEMORY_GB = 2.0  # Minimum free memory to maintain in GB
MIN_FREE_DISK_GB = 5.0  # Minimum free disk space to maintain in GB
BASE_CHUNK_SIZE = 100_000  # Base chunk size for processing


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
    """Dynamically calculate chunk size based on available memory"""
    available_memory_gb = get_available_memory()

    # Adjust chunk size based on available memory
    if available_memory_gb < MIN_FREE_MEMORY_GB:
        # Reduce chunk size if memory is low
        return max(10_000, current_size // 2)
    elif available_memory_gb > MIN_FREE_MEMORY_GB * 3:
        # Increase chunk size if plenty of memory is available
        return min(500_000, current_size * 2)

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


def process_chunk(jobs_df, ts_chunk, chunk_id):
    """Process a chunk of time series data against all jobs"""
    thread_id = threading.get_ident()
    logger.debug(f"Thread {thread_id}: Processing chunk {chunk_id} with {len(ts_chunk)} rows")

    # Ensure Timestamp is datetime
    if "Timestamp" in ts_chunk.columns and not pd.api.types.is_datetime64_any_dtype(ts_chunk["Timestamp"]):
        ts_chunk["Timestamp"] = pd.to_datetime(ts_chunk["Timestamp"], errors='coerce')

    # Join time series chunk with jobs data on jobID
    joined = pd.merge(
        ts_chunk,
        jobs_df,
        left_on="Job Id",
        right_on="jobID",
        how="inner"
    )

    # Release memory from the input chunks since they're no longer needed
    del ts_chunk
    gc.collect()

    logger.debug(f"Thread {thread_id}: Joined dataframe has {len(joined)} rows")

    if joined.empty:
        return None

    # Filter timestamps that fall between job start and end times
    mask = (joined["Timestamp"] >= joined["start"]) & (joined["Timestamp"] <= joined["end"])
    filtered = joined.loc[mask].copy()  # Create explicit copy to avoid SettingWithCopyWarning

    # Release memory from joined dataframe
    del joined
    gc.collect()

    logger.debug(f"Thread {thread_id}: Filtered dataframe has {len(filtered)} rows")

    if filtered.empty:
        return None

    # More efficient event processing
    events = filtered['Event'].unique()

    # Process directly without creating multiple dataframes
    # Create all necessary columns first
    for event in events:
        if event in ["cpuuser", "gpu_usage", "memused", "memused_minus_diskcache", "nfs", "block"]:
            col_name = f'value_{event}'
        else:
            col_name = event

        # Create the column with NaN values
        filtered[col_name] = np.nan

        # Fill values only for matching event rows
        event_mask = filtered['Event'] == event
        filtered.loc[event_mask, col_name] = filtered.loc[event_mask, 'Value']

    # Pivot data by dropping Event and Value columns (no longer needed)
    pivoted = filtered.drop(columns=['Event', 'Value'])

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
    existing_cols = {col: mapping for col, mapping in column_mapping.items() if col in pivoted.columns}
    pivoted = pivoted.rename(columns=existing_cols)

    # Process special cases
    if "timelimit" in pivoted.columns:
        pivoted["timelimit"] = convert_walltime_to_seconds(pivoted["timelimit"])

    if "host_list" in pivoted.columns:
        pivoted["host_list"] = parse_host_list(pivoted["host_list"])

    # Generate exitcode from jobevent and Exit_status using vectorized approach
    if "jobevent" in pivoted.columns:
        pivoted["exitcode"] = get_exit_status_description(pivoted)

        # Remove the source columns after processing
        columns_to_drop = [col for col in ["jobevent", "Exit_status"] if col in pivoted.columns]
        if columns_to_drop:
            pivoted = pivoted.drop(columns=columns_to_drop)

    # Ensure all required columns exist in the output
    set3_columns = ["time", "submit_time", "start_time", "end_time", "timelimit",
                    "nhosts", "ncores", "account", "queue", "host", "jid", "unit",
                    "jobname", "exitcode", "host_list", "username",
                    "value_cpuuser", "value_gpu_usage", "value_memused",
                    "value_memused_minus_diskcache", "value_nfs", "value_block"]

    # Add missing columns
    for col in set3_columns:
        if col not in pivoted.columns:
            pivoted[col] = np.nan

    # Convert datetime columns to UTC timezone
    datetime_cols = ["time", "submit_time", "start_time", "end_time"]
    for col in datetime_cols:
        if col in pivoted.columns and pivoted[col].notna().any():
            try:
                # Check if timezone info already exists
                sample_dt = pivoted.loc[pivoted[col].first_valid_index(), col]
                if hasattr(sample_dt, 'tzinfo') and sample_dt.tzinfo is None:
                    pivoted[col] = pivoted[col].dt.tz_localize('UTC')
            except (TypeError, AttributeError):
                logger.warning(f"Failed to localize timezone for {col}. Skipping.")

    # Select only the columns needed for output and ensure correct order
    # Use a list comprehension to ensure we only include columns that exist
    available_columns = [col for col in set3_columns if col in pivoted.columns]
    result = pivoted[available_columns]

    # Optimize the result dataframe
    result = optimize_dataframe_dtypes(result)

    logger.info(f"Thread {thread_id}: Completed processing chunk {chunk_id} - produced {len(result)} rows")
    return result


def clean_up_cache():
    """Clean up the cache directory"""
    try:
        if CACHE_DIR.exists():
            logger.info("Cleaning up cache directory...")
            for file_path in CACHE_DIR.glob("*"):
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

    # Get files from proc metric directory (parquet files)
    proc_metric_files = list(PROC_METRIC_PATH.glob("*.parquet"))
    logger.info(f"Found {len(proc_metric_files)} files in proc metric directory (parquet)")

    # Get files from job accounting directory (CSV files)
    job_accounting_files = list(JOB_ACCOUNTING_PATH.glob("*.csv"))
    logger.info(f"Found {len(job_accounting_files)} files in job accounting directory (CSV)")

    # Extract year-month from filenames more efficiently
    proc_metrics_years_months = set()

    # Compile regex patterns once for better performance
    fresco_pattern = re.compile(r'FRESCO_Conte_ts_(\d{4})_(\d{2})_v\d+\.parquet')
    other_pattern = re.compile(r'_(\d{4})_(\d{2})_')

    for filepath in proc_metric_files:
        filename = filepath.name

        # Try the FRESCO pattern first
        match = fresco_pattern.search(filename)
        if match:
            year, month = match.groups()
            proc_metrics_years_months.add((year, month))
            continue

        # Try the other pattern
        matches = other_pattern.findall(filename)
        if matches:
            year, month = matches[0]
            proc_metrics_years_months.add((year, month))

    # Compile job accounting pattern
    job_pattern = re.compile(r'(\d{4})-(\d{2})\.csv')

    job_accounting_years_months = set()
    for filepath in job_accounting_files:
        match = job_pattern.match(filepath.name)
        if match:
            year, month = match.groups()
            job_accounting_years_months.add((year, month))

    # Find common year-month combinations
    common_years_months = proc_metrics_years_months.intersection(job_accounting_years_months)
    logger.info(f"Found {len(common_years_months)} common year-month combinations")

    return common_years_months


def optimize_dataframe_dtypes(df):
    """Optimize DataFrame memory usage by changing data types"""
    # Convert object columns that are mostly numeric to appropriate numeric types
    for col in df.select_dtypes(include=['object']).columns:
        # Try to convert to numeric if appropriate
        try:
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
        except (TypeError, ValueError):
            pass

    # Convert string columns with low cardinality to category type
    for col in df.select_dtypes(include=['object']).columns:
        # Check for string columns
        if df[col].apply(lambda x: isinstance(x, str)).all():
            # If less than 50% unique values, convert to category
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')

    return df


def process_ts_file_in_parallel(ts_file, jobs_df, output_writer):
    """Process a single time series file with multiple threads"""
    logger.info(f"Processing TS file in parallel: {ts_file}")

    try:
        # Get file size to estimate appropriate chunk count
        file_size = os.path.getsize(ts_file)

        # Read the time series file metadata to get row count without loading all data
        parquet_file = pq.ParquetFile(ts_file)
        total_rows = parquet_file.metadata.num_rows

        # Determine optimal chunk size based on available memory and cores
        available_memory_gb = get_available_memory()
        max_chunk_size = int(total_rows * (MIN_FREE_MEMORY_GB / available_memory_gb) / MAX_WORKERS)
        chunk_size = min(max(BASE_CHUNK_SIZE, max_chunk_size), 500_000)

        # Calculate number of chunks
        num_chunks = (total_rows + chunk_size - 1) // chunk_size  # Ceiling division

        # Adjust number of workers based on number of chunks
        num_workers = min(MAX_WORKERS, num_chunks)

        logger.info(f"File has {total_rows} rows, processing with {num_workers} threads in {num_chunks} chunks")

        # Create a thread-safe queue for results
        result_queue = Queue()

        # Create a list to keep track of futures
        futures = []

        # Process file in chunks using a thread pool
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # For each chunk, create a pandas DataFrame and submit to the thread pool
            for chunk_idx in range(num_chunks):
                # Calculate row range for this chunk
                start_row = chunk_idx * chunk_size
                end_row = min(start_row + chunk_size, total_rows)

                # Read just this chunk of rows
                chunk_df = parquet_file.read_row_group_slow(
                    chunk_idx).to_pandas() if chunk_idx < parquet_file.metadata.num_row_groups else None

                # If chunk reading failed, try reading with row ranges
                if chunk_df is None or len(chunk_df) == 0:
                    # Alternative approach: read the entire file and slice it
                    # This is less efficient but more robust
                    logger.info(f"Reading chunk {chunk_idx + 1}/{num_chunks} with range {start_row}-{end_row}")

                    # Read the specific row range
                    # Note: This needs to be optimized further for very large files
                    chunk_df = next(pq.read_table(ts_file, use_threads=True).to_batches(chunk_size))
                    chunk_df = chunk_df.slice(start_row % chunk_size, min(chunk_size, end_row - start_row)).to_pandas()

                # Process the chunk in a separate thread
                if chunk_df is not None and not chunk_df.empty:
                    future = executor.submit(process_chunk, jobs_df, chunk_df, chunk_idx + 1)
                    futures.append(future)

                    # Clean up chunk dataframe to free memory
                    del chunk_df
                    gc.collect()

            # Process results as they complete
            for future in tqdm(futures, desc=f"Processing chunks of {ts_file.name}"):
                try:
                    result_df = future.result()
                    if result_df is not None and not result_df.empty:
                        # Convert to PyArrow table and write to output
                        table = pa.Table.from_pandas(result_df)
                        output_writer.write_table(table)

                        # Release memory
                        del result_df, table
                        gc.collect()
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")

        return True
    except Exception as e:
        logger.error(f"Error processing file {ts_file}: {e}")
        return False


def process_year_month(year, month):
    """Process a specific year-month combination with intra-file parallelism"""
    logger.info(f"Processing year: {year}, month: {month}")

    # Check resources before starting
    check_and_optimize_resources()

    try:
        # Get job accounting file (CSV)
        job_file = JOB_ACCOUNTING_PATH / f"{year}-{month}.csv"

        # Find all time series files (Parquet) for this year/month
        ts_pattern = f".*_{year}_{month}.*\.parquet"
        ts_files = list(PROC_METRIC_PATH.glob(ts_pattern))

        if not ts_files:
            logger.warning(f"No time series files found for {year}-{month}")
            return

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
        jobs_df["jobID"] = standardize_job_id(jobs_df["jobID"])

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
        logger.error(f"Error processing {year}-{month}: {e}")


def main():
    """Main function to process all year-month combinations"""
    logger.info("Starting data transformation process")
    start_time = time.time()

    try:
        # Initial cleanup
        clean_up_cache()

        # Get all year-month combinations
        year_month_combinations = get_year_month_combinations()

        if not year_month_combinations:
            logger.warning("No common year-month combinations found to process")
            return

        # Sort combinations for ordered processing
        sorted_combinations = sorted(year_month_combinations)

        # Process one year-month combination at a time (as requested)
        for idx, (year, month) in enumerate(tqdm(sorted_combinations, desc="Processing year-month combinations")):
            logger.info(f"Processing combination {idx + 1}/{len(sorted_combinations)}: {year}-{month}")
            process_year_month(year, month)

            # Check resources after each combination
            check_and_optimize_resources()

        # Final cleanup
        clean_up_cache()

        elapsed_time = time.time() - start_time
        logger.info(f"Data transformation process completed successfully in {elapsed_time:.2f} seconds")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Error in main process: {e}")
    finally:
        # Final cleanup
        clean_up_cache()


if __name__ == "__main__":
    main()