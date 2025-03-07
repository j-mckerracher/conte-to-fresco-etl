import re
from collections import defaultdict
from pathlib import Path
import pandas as pd
import os
import boto3
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from botocore.exceptions import BotoCoreError, ClientError
import logging
from botocore.config import Config

conte_ts_bucket = "data-transform-conte"
conte_job_bucket = "conte-job-accounting"
upload_bucket = "conte-transformed"

# Set up, etc.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def standardize_job_id(id_series):
    """Convert jobIDxxxxx to JOBxxxxx"""
    # Use pandas string methods for the equivalent operation
    return id_series.str.replace(r'^jobID', 'JOB', regex=True)


def process_chunk(jobs_df, ts_chunk):
    """Process a chunk of time series data against all jobs"""
    try:
        # Check if dataframes are valid
        if jobs_df is None or jobs_df.empty or ts_chunk is None or ts_chunk.empty:
            logger.warning("Empty dataframe passed to process_chunk")
            return None

        # Log the initial join attempt
        logger.info(f"Joining {len(ts_chunk)} time series rows with {len(jobs_df)} job rows")

        # Join time series chunk with jobs data on jobID
        joined = pd.merge(
            ts_chunk,
            jobs_df,
            left_on="Job Id",
            right_on="jobID",
            how="inner"
        )

        logger.info(f"Join result has {len(joined)} rows")

        # Filter timestamps that fall between job start and end times
        # Add null handling to prevent filtering issues
        filtered = joined[(joined["Timestamp"].notna()) &
                          (joined["start"].notna()) &
                          (joined["end"].notna()) &
                          (joined["Timestamp"] >= joined["start"]) &
                          (joined["Timestamp"] <= joined["end"])]

        logger.info(f"After time filtering: {len(filtered)} rows")

        if filtered.empty:
            logger.warning("No rows remained after time filtering")
            return None

        # Check for event and value columns
        if "Event" not in filtered.columns or "Value" not in filtered.columns:
            logger.error("Required columns 'Event' or 'Value' missing from filtered data")
            return None

        # Get unique events for debugging
        unique_events = filtered["Event"].unique().tolist()
        logger.info(f"Unique events: {unique_events}")

        # Pivot the metrics into columns based on Event
        try:
            # In pandas, we create the pivot table differently
            group_cols = [col for col in filtered.columns if col not in ["Event", "Value"]]
            result = filtered.pivot_table(
                values="Value",
                index=group_cols,
                columns="Event",
                aggfunc="first"
            ).reset_index()

            # Flatten the column names (pandas creates MultiIndex)
            result.columns = [col[1] if isinstance(col, tuple) and col[0] == 'Value' else col for col in result.columns]

            logger.info(f"After pivot: {len(result)} rows with {len(result.columns)} columns")
        except Exception as e:
            logger.error(f"Error during pivot operation: {e}")
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
            if source in result.columns:
                result = result.rename(columns={source: target})

        # Map the rest of the columns
        column_mapping = {
            # Time fields
            "Timestamp": "time",
            "qtime": "submit_time",
            "start": "start_time",
            "end": "end_time",
            "Resource_List.walltime": "timelimit",  # Will need conversion

            # Resource allocation
            "Resource_List.nodect": "nhosts",
            "Resource_List.ncpus": "ncores",
            "exec_host": "host_list",  # Will need parsing

            # Job identification
            "account": "account",
            "queue": "queue",
            "jobname": "jobname",
            "user": "username",
            "jobevent": "exitcode",
            "Exit_status": "Exit_status",  # Keep temporarily for combined exitcode

            # Core identifiers
            "jobID": "jid",
            "Host": "host",
            "Units": "unit"
        }

        # Apply the column mapping
        columns_to_rename = {col: mapping for col, mapping in column_mapping.items()
                             if col in result.columns}

        result = result.rename(columns=columns_to_rename)

        # Convert walltime to seconds with better error handling
        if "timelimit" in result.columns:
            try:
                # Create a function to convert walltime strings to seconds
                def convert_walltime_to_seconds(x):
                    if pd.isna(x):
                        return None

                    # Handle both numeric and string formats
                    if isinstance(x, (int, float)):
                        return x

                    if not isinstance(x, str):
                        return None

                    if ":" in x:
                        parts = x.split(":")
                        if len(parts) == 3:  # HH:MM:SS
                            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                        elif len(parts) == 2:  # MM:SS
                            return int(parts[0]) * 60 + int(parts[1])
                        else:
                            return int(parts[0])
                    else:
                        try:
                            return int(x)
                        except (ValueError, TypeError):
                            return None

                # Apply the function to the timelimit column
                result["timelimit"] = result["timelimit"].apply(convert_walltime_to_seconds)
            except Exception as e:
                logger.warning(f"Error converting walltime: {e}")
                # Keep original values if conversion fails
                pass

        # Combine jobevent and Exit_status for detailed exitcode if both exist
        if "exitcode" in result.columns and "Exit_status" in result.columns:
            try:
                # Define a function to combine exitcode and Exit_status
                def combine_exitcode(row):
                    if pd.notna(row["exitcode"]):
                        exit_status = row["Exit_status"] if pd.notna(row["Exit_status"]) else ""
                        return f"{row['exitcode']}:{exit_status}"
                    return row["exitcode"]

                result["exitcode"] = result.apply(combine_exitcode, axis=1)

                # Drop the original Exit_status column after combining
                result = result.drop(columns=["Exit_status"])
            except Exception as e:
                logger.warning(f"Error combining exitcode fields: {e}")
                # Keep original columns if combination fails

        # Convert any missing dates to proper format
        date_columns = ["time", "submit_time", "start_time", "end_time"]
        for col in date_columns:
            if col in result.columns:
                try:
                    # Convert to datetime if not already
                    if result[col].dtype != 'datetime64[ns]':
                        result[col] = pd.to_datetime(result[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Error converting date column {col}: {e}")

        # Add any missing columns with null values
        set3_columns = ["time", "submit_time", "start_time", "end_time", "timelimit",
                        "nhosts", "ncores", "account", "queue", "host", "jid", "unit",
                        "jobname", "exitcode", "host_list", "username",
                        "value_cpuuser", "value_gpu", "value_memused",
                        "value_memused_minus_diskcache", "value_nfs", "value_block"]

        # Add missing columns
        existing_columns = set(result.columns)
        missing_columns = [col for col in set3_columns if col not in existing_columns]

        if missing_columns:
            logger.info(f"Adding missing columns: {missing_columns}")
            for col in missing_columns:
                result[col] = None

        # Select only the columns needed for Set 3
        # Handle the case where some columns might still be missing
        available_columns = [col for col in set3_columns if col in result.columns]
        result = result[available_columns]

        logger.info(f"Final result has {len(result)} rows with {len(result.columns)} columns")
        return result

    except Exception as e:
        logger.error(f"Error in process_chunk: {e}", exc_info=True)
        return None


def debug_dataframe(df, name, sample_size=5):
    """Helper function to print debug information about a dataframe"""
    logger.info(f"--- Debug info for {name} ---")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Data types: {df.dtypes}")

    # Sample data
    if not df.empty:
        logger.info(f"Sample data:")
        sample = df.head(sample_size)
        for _, row in sample.iterrows():
            logger.info(f"  {row.to_dict()}")

    # Check for null values
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            logger.info(f"Column '{col}' has {null_count} null values ({null_count / len(df):.2%})")

    logger.info(f"--- End debug info for {name} ---")


def join_job_timeseries(job_file: str, timeseries_file: str, output_file: str, chunk_size: int = 100_000):
    """
    Join job accounting data with time series data, creating a row for each timestamp.
    """
    # Define datetime formats for both data sources
    JOB_DATETIME_FMT = "%m/%d/%Y %H:%M:%S"  # Format for job data: "03/01/2015 01:29:34"
    TS_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"  # Format for timeseries data: "2015-03-01 14:56:51"

    print("Reading job accounting data...")

    try:
        # First approach: Try reading directly with dtype overrides and inference
        jobs_df = None
        try:
            # Use string dtypes for all problematic columns
            dtype_overrides = {
                "Resource_List.neednodes": str,
                "Resource_List.nodes": str,
                "Resource_List.nodect": str,
                "Resource_List.ncpus": str,
                "Resource_List.neednodes.ppn": str,
                "Resource_List.nodes.ppn": str,
                "Resource_List.mem": str,
                "Resource_List.pmem": str,
                "Resource_List.walltime": str,
            }

            jobs_df = pd.read_csv(job_file, dtype=dtype_overrides)
        except Exception as e:
            logger.warning(f"First attempt to read job file failed: {e}")

            # Second approach: Try reading with error_bad_lines=False (changed to on_bad_lines='skip' in pandas 1.3+)
            try:
                jobs_df = pd.read_csv(
                    job_file,
                    on_bad_lines='skip',  # Use this for pandas >= 1.3.0
                    dtype=dtype_overrides
                )
            except Exception as e:
                logger.warning(f"Second attempt with on_bad_lines='skip' failed: {e}")

                # Last resort: Try reading with low-level settings to ensure it works
                try:
                    # Get all column names first
                    with open(job_file, 'r') as f:
                        header_line = f.readline().strip()

                    column_names = header_line.split(',')

                    # Create a schema that forces all columns to strings
                    dtype = {col: str for col in column_names}

                    jobs_df = pd.read_csv(
                        job_file,
                        dtype=dtype,
                        on_bad_lines='skip',
                        na_values=["", "NULL", "null", "NODE390"]  # Add known problematic values
                    )
                except Exception as e:
                    logger.error(f"All attempts to read job file failed: {e}")
                    return

        if jobs_df is None or jobs_df.empty:
            logger.error("Failed to read job file or file is empty")
            return

        logger.info(f"Successfully read job file with {len(jobs_df)} rows")

        # Standardize job IDs
        jobs_df["jobID"] = standardize_job_id(jobs_df["jobID"])

        # Handle datetime columns with explicit null checks
        # Convert "start" and "end" columns to datetime
        jobs_df["start"] = pd.to_datetime(jobs_df["start"], format=JOB_DATETIME_FMT, errors='coerce')
        jobs_df["end"] = pd.to_datetime(jobs_df["end"], format=JOB_DATETIME_FMT, errors='coerce')

        # Convert numeric columns safely
        numeric_cols = [
            "Resource_List.nodect",
            "Resource_List.ncpus",
        ]

        for col in numeric_cols:
            if col in jobs_df.columns:
                # Use pandas numeric conversion with error handling
                jobs_df[col] = pd.to_numeric(jobs_df[col], errors='coerce')

        # Print the job data schema for debugging
        logger.info(f"Job data columns: {jobs_df.columns.tolist()}")
        logger.info(f"Job data types: {jobs_df.dtypes}")

        print("Processing time series data in chunks...")

        # Read the time series data in chunks
        # Initialize the writer for output
        first_chunk = True

        # Use chunksize for efficient processing of large CSV files
        for i, ts_chunk in enumerate(pd.read_csv(timeseries_file, chunksize=chunk_size)):
            chunk_start = i * chunk_size
            chunk_end = chunk_start + len(ts_chunk)
            logger.info(f"Processing chunk {i + 1}: rows {chunk_start}-{chunk_end}")

            # Convert the Timestamp column to datetime using the timeseries format
            ts_chunk["Timestamp"] = pd.to_datetime(ts_chunk["Timestamp"], format=TS_DATETIME_FMT, errors='coerce')

            # Process the chunk by joining and filtering
            result_df = process_chunk(jobs_df, ts_chunk)

            if result_df is not None and not result_df.empty:
                # Write results to the output CSV file
                if first_chunk:
                    # Write with headers for first chunk
                    result_df.to_csv(output_file, index=False)
                    first_chunk = False
                else:
                    # For subsequent chunks, append without headers
                    result_df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                logger.warning(f"No matching data found for chunk {chunk_start}-{chunk_end}")

    except Exception as e:
        logger.error(f"Error in join_job_timeseries: {e}", exc_info=True)


def get_s3_client():
    """Create and return an S3 client with appropriate connection pool settings"""
    # Create a session with a persistent connection pool
    session = boto3.session.Session()
    return session.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1',
        config=Config(
            retries={'max_attempts': 5, 'mode': 'standard'}
        )
    )


def list_s3_files(bucket):
    s3_client = get_s3_client()
    response = s3_client.list_objects_v2(Bucket=bucket)
    files_in_s3 = []

    # Extract files from the initial response
    if 'Contents' in response:
        files_in_s3 = [item['Key'] for item in response['Contents']]

    # Handle pagination if there are more objects
    while response.get('IsTruncated', False):
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            ContinuationToken=response.get('NextContinuationToken')
        )
        if 'Contents' in response:
            files_in_s3.extend([item['Key'] for item in response['Contents']])

    return files_in_s3


@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(4),
    retry=retry_if_exception_type((BotoCoreError, ClientError, Exception))
)
def download_file(s3_client, file, temp_dir, bucket):
    """Download a single file from S3 with retry logic"""
    download_path = temp_dir / os.path.basename(file)
    s3_client.download_file(bucket, file, str(download_path))
    return download_path


@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(4),
    retry=retry_if_exception_type((BotoCoreError, ClientError))
)
def upload_file_to_s3(file_path: str, bucket_name: str) -> None:
    """
    Uploads a single file to an S3 bucket.

    Args:
        file_path (str): Local path to the file.
        bucket_name (str): Name of the S3 bucket.

    Raises:
        BotoCoreError, ClientError: If the upload fails due to a boto3 error.
    """
    # Add content type for CSV files
    s3_client = get_s3_client()
    extra_args = {
        'ContentType': 'text/csv'
    }

    # Use the filename as the S3 key
    s3_key = os.path.basename(file_path)

    logger.info(f"Uploading {file_path} to {bucket_name}/{s3_key}")
    s3_client.upload_file(file_path, bucket_name, s3_key, ExtraArgs=extra_args)


def get_year_month_combos(files_in_s3):
    # Group files by year-month
    files_by_year_month = defaultdict(list)

    for file in files_in_s3:
        # Extract year-month from FRESCO_Conte_ts_YYYY_MM_vX.csv pattern
        match = re.search(r'FRESCO_Conte_ts_(\d{4})_(\d{2})_v\d+\.csv$', file)
        if match:
            year = match.group(1)
            month = match.group(2)
            year_month = f"{year}-{month}"
            files_by_year_month[year_month].append(file)

    return files_by_year_month


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("conte_transform.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting data transformation process")

    try:
        # List files from S3
        logger.info("Listing files from S3 buckets")
        s3_ts_files = list_s3_files(conte_ts_bucket)
        s3_job_files = list_s3_files(conte_job_bucket)

        # Group files by year-month
        combos = get_year_month_combos(s3_ts_files)
        logger.info(f"Found data for {len(combos)} year-month combinations: {list(combos.keys())}")

        s3_client = get_s3_client()

        # Create cache directory
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True, parents=True)

        # Process each year-month combination
        for date, files in combos.items():
            logger.info(f"Processing data for {date}")

            try:
                # Download time series files
                logger.info(f"Downloading {len(files)} time series files for {date}")
                for f in files:
                    download_file(s3_client, f, cache_dir, conte_ts_bucket)

                # Combine time series files
                dfs = []
                for temp_file in os.listdir(cache_dir):
                    try:
                        if not temp_file.startswith("combined_"):  # Skip already combined files
                            df = pd.read_csv(cache_dir / temp_file)
                            dfs.append(df)
                            logger.info(f"Read {temp_file} with {len(df)} rows")
                    except Exception as e:
                        logger.error(f"Error reading {temp_file}: {e}")

                # Skip if no valid data
                if not dfs:
                    logger.error(f"No valid time series data found for {date}, skipping")
                    continue

                # Combine and save time series data
                combined_ts_file = cache_dir / f"combined_{date}.csv"
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df.to_csv(combined_ts_file, index=False)
                logger.info(f"Combined {len(dfs)} files into {combined_ts_file} with {len(combined_df)} rows")
                del combined_df  # Free memory

                # Clean up individual time series files
                for f in files:
                    file_path = Path(cache_dir / os.path.basename(f))
                    if file_path.exists():
                        os.remove(file_path)

                # Download job file
                if f"{date}.csv" in s3_job_files:
                    logger.info(f"Downloading job file for {date}")
                    download_file(s3_client, f"{date}.csv", cache_dir, conte_job_bucket)
                else:
                    logger.error(f"No job accounting data found for {date}! Skipping")
                    os.remove(combined_ts_file)
                    continue

                # Process the data
                job_file = Path(cache_dir / f"{date}.csv")
                timeseries_file = combined_ts_file
                output_file = f"conte_transformed_{date}.csv"

                logger.info(f"Joining job and time series data for {date}")
                join_job_timeseries(str(job_file), str(timeseries_file), output_file)

                # Check if output file was created and has content
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    logger.info(f"Successfully created {output_file}")

                    # Upload to S3
                    logger.info(f"Uploading {output_file} to S3 {upload_bucket}")
                    upload_file_to_s3(output_file, upload_bucket)
                else:
                    logger.error(f"Failed to create valid output file for {date}")

                # Clean up
                if job_file.exists():
                    os.remove(job_file)
                if combined_ts_file.exists():
                    os.remove(combined_ts_file)

            except Exception as e:
                logger.error(f"Error processing data for {date}: {e}", exc_info=True)

        logger.info("Data transformation process completed")

    except Exception as e:
        logger.error(f"Unhandled exception in main process: {e}", exc_info=True)