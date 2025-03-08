import pandas as pd
import numpy as np
import re
import json
import os
import logging
from pathlib import Path
import shutil
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define new file paths
JOB_ACCOUNTING_PATH = Path("P:/Conte/conte-job-accounting-1")
PROC_METRIC_PATH = Path("P:/Conte/conte-ts-to-fresco-ts-1")
OUTPUT_PATH = Path("P:/Conte/conte-transformed-2")

# Create cache directory
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

# Ensure output directory exists
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)


def standardize_job_id(job_id_series):
    """Convert jobIDxxxxx to JOBxxxxx"""
    return job_id_series.str.replace(r'^jobID', 'JOB', regex=True)


def convert_walltime_to_seconds(walltime_series):
    """Convert HH:MM:SS format to seconds"""

    def parse_time(time_str):
        if pd.isna(time_str):
            return np.nan
        if not isinstance(time_str, str):
            return float(time_str)

        parts = time_str.split(':')
        if len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        else:
            try:
                return float(time_str)
            except (ValueError, TypeError):
                return np.nan

    return walltime_series.apply(parse_time)


def parse_host_list(exec_host_series):
    """Parse exec_host into a list of hosts in JSON format"""

    def extract_hosts(host_str):
        if pd.isna(host_str):
            return None

        # Example format: "NODE407/0-15+NODE476/0-15"
        if isinstance(host_str, str):
            # Extract unique node names using regex
            nodes = re.findall(r'([^/+]+)/', host_str)
            unique_nodes = list(set(nodes))
            # Return as JSON format string to match example
            return json.dumps(unique_nodes).replace('"', '')
        return None

    return exec_host_series.apply(extract_hosts)


def get_exit_status_description(jobevent_series, exit_status_series):
    """Convert exit status to descriptive text"""

    def get_description(row):
        jobevent = row['jobevent'] if not pd.isna(row['jobevent']) else ''
        exit_status = row['Exit_status'] if not pd.isna(row['Exit_status']) else ''

        if jobevent == 'E' and exit_status == '0':
            return 'COMPLETED'
        elif jobevent == 'E':
            return f'FAILED:{exit_status}'
        elif jobevent == 'A':
            return 'ABORTED'
        elif jobevent == 'S':
            return 'STARTED'
        elif jobevent == 'Q':
            return 'QUEUED'
        else:
            return f'{jobevent}:{exit_status}'

    return pd.DataFrame({
        'jobevent': jobevent_series,
        'Exit_status': exit_status_series
    }).apply(get_description, axis=1)


def process_chunk(jobs_df, ts_chunk):
    """Process a chunk of time series data against all jobs"""
    logger.debug(f"Processing chunk with {len(ts_chunk)} rows against {len(jobs_df)} jobs")

    # Join time series chunk with jobs data on jobID
    joined = pd.merge(
        ts_chunk,
        jobs_df,
        left_on="Job Id",
        right_on="jobID",
        how="inner"
    )

    logger.debug(f"Joined dataframe has {len(joined)} rows")

    # Filter timestamps that fall between job start and end times
    filtered = joined[
        (joined["Timestamp"] >= joined["start"]) &
        (joined["Timestamp"] <= joined["end"])
        ].copy()  # Create explicit copy to avoid SettingWithCopyWarning

    logger.debug(f"Filtered dataframe has {len(filtered)} rows")

    if filtered.empty:
        return None

    # Skip the pivot operation completely and use a different approach
    try:
        # Get unique Events
        events = filtered['Event'].unique()

        # Create a list to hold dataframes for each event
        event_dfs = []

        # Process each event type separately
        for event in events:
            # Filter rows for this event
            event_data = filtered[filtered['Event'] == event].copy()

            # Rename the Value column to include the event name
            if event in ["cpuuser", "gpu_usage", "memused", "memused_minus_diskcache", "nfs", "block"]:
                event_data = event_data.rename(columns={'Value': f'value_{event}'})
            else:
                event_data = event_data.rename(columns={'Value': event})

            # Drop the Event column since it's now encoded in the column name
            event_data = event_data.drop(columns=['Event'])

            event_dfs.append(event_data)

        # If we have any event dataframes
        if event_dfs:
            # Start with the first event dataframe
            result_df = event_dfs[0]

            # Merge in each additional event dataframe
            for i in range(1, len(event_dfs)):
                # Get columns to merge on (all except the value columns)
                value_cols = [col for col in event_dfs[i].columns if col.startswith('value_') or col in events]
                merge_cols = [col for col in event_dfs[i].columns if col not in value_cols]

                # Merge with the result
                result_df = pd.merge(result_df, event_dfs[i], on=merge_cols, how='outer', suffixes=('', '_drop'))

                # Drop any duplicate columns from the merge
                drop_cols = [col for col in result_df.columns if col.endswith('_drop')]
                if drop_cols:
                    result_df = result_df.drop(columns=drop_cols)

            pivoted = result_df
            logger.debug(f"Merged dataframe has {len(pivoted)} rows and {len(pivoted.columns)} columns")
        else:
            logger.warning("No event dataframes created")
            return None

    except Exception as e:
        logger.error(f"Error during dataframe merging: {e}")
        return None

    # Map the rest of the columns
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

    # Apply the column mapping
    pivoted = pivoted.rename(columns={col: mapping for col, mapping in column_mapping.items()
                                      if col in pivoted.columns})

    # Process special cases that need calculation or parsing
    if "timelimit" in pivoted.columns:
        pivoted["timelimit"] = convert_walltime_to_seconds(pivoted["timelimit"])

    if "host_list" in pivoted.columns:
        pivoted["host_list"] = parse_host_list(pivoted["host_list"])

    # Generate exitcode from jobevent and Exit_status
    if "jobevent" in pivoted.columns:
        pivoted["exitcode"] = get_exit_status_description(
            pivoted["jobevent"],
            pivoted["Exit_status"] if "Exit_status" in pivoted.columns else pd.Series([None] * len(pivoted))
        )

        # Remove the source columns after processing
        columns_to_drop = [col for col in ["jobevent", "Exit_status"] if col in pivoted.columns]
        if columns_to_drop:
            pivoted = pivoted.drop(columns=columns_to_drop)

    # Ensure all required columns exist in the output
    set3_columns = ["time", "submit_time", "start_time", "end_time", "timelimit",
                    "nhosts", "ncores", "account", "queue", "host", "jid", "unit",
                    "jobname", "exitcode", "host_list", "username",
                    "value_cpuuser", "value_gpu", "value_memused",
                    "value_memused_minus_diskcache", "value_nfs", "value_block"]

    for col in set3_columns:
        if col not in pivoted.columns:
            pivoted[col] = np.nan

    # Convert datetime columns to the expected timezone format
    datetime_cols = ["time", "submit_time", "start_time", "end_time"]
    for col in datetime_cols:
        if col in pivoted.columns and not pd.isna(pivoted[col]).all():
            try:
                pivoted[col] = pivoted[col].dt.tz_localize('UTC')
            except (TypeError, AttributeError):
                logger.warning(f"Failed to localize timezone for {col}. Skipping.")

    # Select only the columns needed for Set 3 and ensure correct order
    result = pivoted[set3_columns]

    return result


def check_disk_space():
    """Check available disk space and clean up if less than 5GB available"""
    _, _, free = shutil.disk_usage("/")
    free_gb = free / (1024 * 1024 * 1024)

    if free_gb < 5:
        logger.warning(f"Low disk space: {free_gb:.2f}GB free. Cleaning up cache...")
        try:
            # Remove all files from cache directory
            for file_path in CACHE_DIR.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            logger.info("Cache directory cleaned.")
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")


def get_year_month_combinations():
    """Get all year-month combinations from both local directories"""
    logger.info("Getting list of files from local directories...")

    # Get files from proc metric directory (parquet files)
    proc_metric_files = list(PROC_METRIC_PATH.glob("*.parquet"))
    logger.info(f"Found {len(proc_metric_files)} files in proc metric directory")

    # Get files from job accounting directory (CSV files)
    job_accounting_files = list(JOB_ACCOUNTING_PATH.glob("*.csv"))
    logger.info(f"Found {len(job_accounting_files)} files in job accounting directory")

    # Extract year-month from filenames
    proc_metrics_years_months = set()
    for filepath in proc_metric_files:
        filename = filepath.name
        if '_ts_' in filename:
            # Example: FRESCO_Conte_ts_2015_03_v1.parquet
            parts = filename.split('_')
            if len(parts) >= 5 and parts[3].isdigit() and parts[4].isdigit():
                proc_metrics_years_months.add((parts[3], parts[4]))
        elif re.match(r'.*_\d{4}_\d{2}.*\.parquet', filename):
            # Other filename patterns with year-month
            matches = re.findall(r'_(\d{4})_(\d{2})_', filename)
            if matches:
                year, month = matches[0]
                proc_metrics_years_months.add((year, month))

    job_accounting_years_months = set()
    for filepath in job_accounting_files:
        filename = filepath.name
        if match := re.match(r'(\d{4})-(\d{2})\.csv', filename):
            year, month = match.groups()
            job_accounting_years_months.add((year, month))

    # Find common year-month combinations
    common_years_months = proc_metrics_years_months.intersection(job_accounting_years_months)
    logger.info(f"Found {len(common_years_months)} common year-month combinations")

    return common_years_months


def process_year_month(year, month):
    """Process a specific year-month combination"""
    logger.info(f"Processing year: {year}, month: {month}")

    # Create temp directory for this year-month
    temp_dir = CACHE_DIR / f"{year}_{month}"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Get job accounting file
        job_file = JOB_ACCOUNTING_PATH / f"{year}-{month}.csv"

        # Find all time series files for this year/month
        ts_pattern = f".*_{year}_{month}.*\.parquet"
        ts_files = list(PROC_METRIC_PATH.glob(ts_pattern))

        if not ts_files:
            logger.warning(f"No time series files found for {year}-{month}")
            shutil.rmtree(temp_dir)
            return

        # Create output file
        output_file = OUTPUT_PATH / f"transformed_{year}_{month}.parquet"
        output_file_tmp = temp_dir / f"transformed_{year}_{month}_tmp.csv"
        first_ts_file = True

        # Read job data
        logger.info(f"Reading job data from {job_file}")
        jobs_df = pd.read_csv(job_file, low_memory=False)

        # convert problematic large integers to strings
        int_columns = jobs_df.select_dtypes(include=['int64']).columns
        for col in int_columns:
            # Check if any values exceed C long limit
            try:
                if jobs_df[col].max() > 2147483647 or jobs_df[col].min() < -2147483648:
                    logger.info(f"Converting column {col} to string due to large integer values")
                    jobs_df[col] = jobs_df[col].astype(str)
            except (TypeError, ValueError, OverflowError) as e:
                logger.info(f"Converting column {col} to string due to potential overflow: {e}")
                jobs_df[col] = jobs_df[col].astype(str)

        # Standardize job IDs
        jobs_df["jobID"] = standardize_job_id(jobs_df["jobID"])

        # Convert datetime columns using the job data format
        JOB_DATETIME_FMT = "%m/%d/%Y %H:%M:%S"
        datetime_cols = ["ctime", "qtime", "etime", "start", "end", "timestamp"]
        for col in datetime_cols:
            if col in jobs_df.columns:
                jobs_df[col] = pd.to_datetime(jobs_df[col], format=JOB_DATETIME_FMT, errors='coerce')

        # Process each time series file
        for ts_file_idx, ts_file in enumerate(ts_files):
            logger.info(f"Processing TS file {ts_file_idx + 1}/{len(ts_files)}: {ts_file}")

            # Process time series file in chunks
            chunk_size = 100_000
            first_chunk = first_ts_file
            logger.info(f"Reading time series data in chunks from {ts_file}")

            try:
                # Read time series file - using pyarrow for parquet files
                ts_df = pd.read_parquet(ts_file)

                # Process in chunks to maintain memory efficiency
                for i in range(0, len(ts_df), chunk_size):
                    # Extract chunk
                    ts_chunk = ts_df.iloc[i:i + chunk_size].copy()

                    # Check disk space periodically
                    if i % (5 * chunk_size) == 0:
                        check_disk_space()

                    chunk_idx = i // chunk_size
                    logger.info(f"Processing chunk {chunk_idx + 1} with {len(ts_chunk)} rows")

                    # Ensure Timestamp is datetime
                    if "Timestamp" in ts_chunk.columns and not pd.api.types.is_datetime64_any_dtype(
                            ts_chunk["Timestamp"]):
                        ts_chunk["Timestamp"] = pd.to_datetime(ts_chunk["Timestamp"], errors='coerce')

                    # Process the chunk
                    result_df = process_chunk(jobs_df, ts_chunk)

                    if result_df is not None and not result_df.empty:
                        # Write results to the output CSV file
                        if first_chunk:
                            result_df.to_csv(output_file_tmp, index=False)
                            first_chunk = False
                        else:
                            result_df.to_csv(output_file_tmp, mode='a', header=False, index=False)

            except Exception as e:
                logger.error(f"Error processing {ts_file}: {e}")

            first_ts_file = False

        # Convert final CSV to parquet and save to output directory
        if output_file_tmp.exists() and output_file_tmp.stat().st_size > 0:
            logger.info(f"Converting CSV to parquet and saving to {output_file}")
            final_df = pd.read_csv(output_file_tmp)
            final_df.to_parquet(output_file, index=False)
        else:
            logger.warning(f"No output generated for {year}-{month}")

        # Clean up
        logger.info(f"Cleaning up temporary files for {year}-{month}")
        shutil.rmtree(temp_dir)

    except Exception as e:
        logger.error(f"Error processing {year}-{month}: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    """Main function to process all year-month combinations"""
    logger.info("Starting data transformation process")

    try:
        # Get all year-month combinations
        year_month_combinations = get_year_month_combinations()

        if not year_month_combinations:
            logger.warning("No common year-month combinations found to process")
            return

        # Process each year-month combination
        for idx, (year, month) in enumerate(sorted(year_month_combinations)):
            logger.info(f"Processing combination {idx + 1}/{len(year_month_combinations)}: {year}-{month}")
            process_year_month(year, month)

            # Check disk space after each combination
            check_disk_space()

        logger.info("Data transformation process completed successfully")

    except Exception as e:
        logger.error(f"Error in main process: {e}")
    finally:
        # Final cleanup
        if CACHE_DIR.exists():
            logger.info("Performing final cleanup")
            try:
                for file_path in CACHE_DIR.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                logger.info("Final cleanup completed")
            except Exception as e:
                logger.error(f"Error during final cleanup: {e}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")