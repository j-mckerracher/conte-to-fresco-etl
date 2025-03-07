import pandas as pd
import numpy as np
import re
import json
import os
import logging
from utils.s3 import S3_Client
import shutil
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize S3 client
s3_client = S3_Client(
    upload_bucket="conte-transformed",
    download_bucket_job_accounting="conte-job-accounting",
    download_bucket_proc_metric="data-transform-conte"
)

# Create cache directory
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)


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

    # Try different pivoting approaches to handle the data
    try:
        # First approach: Use groupby + agg to create a pivot
        # Add a dummy column to distinguish duplicate rows
        filtered.loc[:, '_row_id'] = range(len(filtered))

        # Group by Event and all other columns except Value
        value_cols = ['Value', '_row_id']
        group_cols = [col for col in filtered.columns if col not in value_cols]

        # First group to find duplicates in the index
        grouped = filtered.groupby(group_cols + ['Event'])

        # If there are no duplicates, use the faster pivot approach
        if grouped.size().max() == 1:
            # Use traditional pivot
            group_cols = [col for col in filtered.columns if col not in ['Event', 'Value']]
            pivoted = filtered.pivot(
                index=group_cols,
                columns='Event',
                values='Value'
            ).reset_index()
        else:
            # There are duplicates, use the more flexible pivot_table with a first aggregation
            logger.info("Using pivot_table with first aggregation due to duplicates")
            pivoted = pd.pivot_table(
                filtered,
                values='Value',
                index=group_cols,
                columns='Event',
                aggfunc='first'
            ).reset_index()

        logger.debug(f"Pivoted dataframe has {len(pivoted)} rows and {len(pivoted.columns)} columns")
    except Exception as e:
        logger.error(f"Error during pivot: {e}")

        # Alternative approach if the above fails
        try:
            logger.info("Trying alternative pivot approach")

            # Create a fresh copy for the alternative approach
            filtered_alt = filtered.copy()

            # Create a unique compound key from all group columns
            # This ensures we have a unique index for the pivot
            key_cols = [col for col in filtered_alt.columns if col not in ['Event', 'Value']]

            # Convert all columns to strings and concatenate to form a unique key
            for col in key_cols:
                if pd.api.types.is_integer_dtype(filtered_alt[col]):
                    filtered_alt.loc[:, col] = filtered_alt[col].astype(str)

            # Create a unique key by concatenating all columns
            filtered_alt.loc[:, '_unique_key'] = filtered_alt[key_cols].apply(
                lambda row: '__'.join([str(v) for v in row.values]), axis=1
            )

            # Now use the unique key for pivoting
            pivoted = pd.pivot_table(
                filtered_alt,
                values='Value',
                index='_unique_key',
                columns='Event',
                aggfunc='first'
            )

            # Recover the original columns by splitting the index
            key_parts = pd.DataFrame(
                pivoted.index.map(lambda x: x.split('__')).tolist(),
                columns=key_cols,
                index=pivoted.index
            )

            # Join the key parts with the pivoted data
            result = pd.concat([key_parts, pivoted.reset_index(drop=True)], axis=1)
            pivoted = result

            logger.info("Alternative pivot approach succeeded")
        except Exception as e2:
            logger.error(f"Both pivot approaches failed: {e2}")
            return None

    # Map columns to the target schema (Set 3)
    # First, handle the event-based metrics
    event_cols = {
        "cpuuser": "value_cpuuser",
        "gpu_usage": "value_gpu",
        "memused": "value_memused",
        "memused_minus_diskcache": "value_memused_minus_diskcache",
        "nfs": "value_nfs",
        "block": "value_block"
    }

    # Rename the event columns to their target names
    for source, target in event_cols.items():
        if source in pivoted.columns:
            pivoted = pivoted.rename(columns={source: target})

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
    """Get all year-month combinations from both buckets"""
    logger.info("Getting list of files from S3 buckets...")

    # Get files from proc metric bucket (data-transform-conte)
    proc_metric_files = []
    try:
        proc_metric_files = s3_client.list_s3_files(bucket="data-transform-conte")
        logger.info(f"Found {len(proc_metric_files)} files in proc metric bucket")
    except Exception as e:
        logger.error(f"Error listing proc metric files: {e}")

    # Get files from job accounting bucket (conte-job-accounting)
    job_accounting_files = []
    try:
        job_accounting_files = s3_client.list_s3_files("conte-job-accounting")
        logger.info(f"Found {len(job_accounting_files)} files in job accounting bucket")
    except Exception as e:
        logger.error(f"Error listing job accounting files: {e}")

    # Extract year-month from filenames
    proc_metrics_years_months = set()
    for filename in proc_metric_files:
        if '_ts_' in filename:
            # Example: FRESCO_Conte_ts_2015_03_v1.csv
            parts = filename.split('_')
            if len(parts) >= 5 and parts[3].isdigit() and parts[4].isdigit():
                proc_metrics_years_months.add((parts[3], parts[4]))
        elif re.match(r'.*_\d{4}_\d{2}.*\.csv', filename):
            # Other filename patterns with year-month
            matches = re.findall(r'_(\d{4})_(\d{2})_', filename)
            if matches:
                year, month = matches[0]
                proc_metrics_years_months.add((year, month))

    job_accounting_years_months = set()
    for filename in job_accounting_files:
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
        # Download job accounting file
        job_file_s3 = f"{year}-{month}.csv"
        job_file_local = temp_dir / job_file_s3
        logger.info(f"Downloading job file: {job_file_s3}")
        s3_client.download_file(job_file_s3, temp_dir, "job")

        # Find all time series files for this year/month
        ts_pattern = f".*_{year}_{month}.*\.csv"
        all_files = s3_client.list_s3_files("data-transform-conte")
        ts_files = [f for f in all_files if re.match(ts_pattern, f)]

        if not ts_files:
            logger.warning(f"No time series files found for {year}-{month}")
            shutil.rmtree(temp_dir)
            return

        # Create output file
        output_file = temp_dir / f"transformed_{year}_{month}.csv"
        output_file_tmp = temp_dir / f"transformed_{year}_{month}_tmp.csv"
        first_ts_file = True

        # Read job data
        logger.info(f"Reading job data from {job_file_local}")
        jobs_df = pd.read_csv(job_file_local, low_memory=False)

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
        for ts_file_idx, ts_file_s3 in enumerate(ts_files):
            logger.info(f"Processing TS file {ts_file_idx + 1}/{len(ts_files)}: {ts_file_s3}")

            # Download time series file
            ts_file_local = temp_dir / os.path.basename(ts_file_s3)
            try:
                s3_client.download_file(ts_file_s3, temp_dir, "proc")
            except Exception as e:
                logger.error(f"Failed to download {ts_file_s3}: {e}")
                continue

            # Process time series file in chunks
            chunk_size = 100_000
            first_chunk = first_ts_file
            logger.info(f"Reading time series data in chunks from {ts_file_local}")

            try:
                # Read time series file in chunks
                TS_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
                chunk_reader = pd.read_csv(ts_file_local, chunksize=chunk_size)

                for chunk_idx, ts_chunk in enumerate(chunk_reader):
                    # Check disk space periodically
                    if chunk_idx % 5 == 0:
                        check_disk_space()

                    logger.info(f"Processing chunk {chunk_idx + 1} with {len(ts_chunk)} rows")

                    # Convert the Timestamp column to datetime
                    ts_chunk["Timestamp"] = pd.to_datetime(ts_chunk["Timestamp"], format=TS_DATETIME_FMT,
                                                           errors='coerce')

                    # Process the chunk
                    result_df = process_chunk(jobs_df, ts_chunk)

                    if result_df is not None and not result_df.empty:
                        # Write results to the output CSV file
                        if first_chunk:
                            result_df.to_csv(output_file_tmp, index=False)
                            first_chunk = False
                        else:
                            result_df.to_csv(output_file_tmp, mode='a', header=False, index=False)

                # Remove the time series file after processing
                ts_file_local.unlink()

            except Exception as e:
                logger.error(f"Error processing {ts_file_local}: {e}")
                if ts_file_local.exists():
                    ts_file_local.unlink()

            first_ts_file = False

        # Upload the final output file to S3
        if output_file_tmp.exists() and output_file_tmp.stat().st_size > 0:
            logger.info(f"Renaming {output_file_tmp} to {output_file}")
            output_file_tmp.rename(output_file)

            logger.info(f"Uploading processed data for {year}-{month} to S3")
            s3_client.upload_file_to_s3(str(output_file), s3_client.upload_bucket)
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