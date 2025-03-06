import re
from collections import defaultdict
from pathlib import Path
import polars as pl
from tqdm import tqdm
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


def standardize_job_id(id_series: pl.Series) -> pl.Series:
    """Convert jobIDxxxxx to JOBxxxxx"""
    return (
        pl.when(id_series.str.contains('^jobID'))
        .then(pl.concat_str([pl.lit('JOB'), id_series.str.slice(5)]))
        .otherwise(id_series)
    )


def process_chunk(jobs_df: pl.DataFrame, ts_chunk: pl.DataFrame) -> pl.DataFrame:
    """Process a chunk of time series data against all jobs"""
    try:
        # Check if dataframes are valid
        if jobs_df is None or jobs_df.height == 0 or ts_chunk is None or ts_chunk.height == 0:
            logger.warning("Empty dataframe passed to process_chunk")
            return None

        # Log the initial join attempt
        logger.info(f"Joining {ts_chunk.height} time series rows with {jobs_df.height} job rows")

        # Join time series chunk with jobs data on jobID
        joined = ts_chunk.join(
            jobs_df,
            left_on="Job Id",
            right_on="jobID",
            how="inner"
        )

        logger.info(f"Join result has {joined.height} rows")

        # Filter timestamps that fall between job start and end times
        # Add null handling to prevent filtering issues
        filtered = joined.filter(
            pl.col("Timestamp").is_not_null() &
            pl.col("start").is_not_null() &
            pl.col("end").is_not_null() &
            (pl.col("Timestamp") >= pl.col("start")) &
            (pl.col("Timestamp") <= pl.col("end"))
        )

        logger.info(f"After time filtering: {filtered.height} rows")

        if filtered.height == 0:
            logger.warning("No rows remained after time filtering")
            return None

        # Check for event and value columns
        if "Event" not in filtered.columns or "Value" not in filtered.columns:
            logger.error("Required columns 'Event' or 'Value' missing from filtered data")
            return None

        # Get unique events for debugging
        unique_events = filtered.select(pl.col("Event").unique()).to_series().to_list()
        logger.info(f"Unique events: {unique_events}")

        # Pivot the metrics into columns based on Event
        try:
            group_cols = [col for col in filtered.columns if col not in ["Event", "Value"]]

            result = filtered.pivot(
                values="Value",
                index=group_cols,
                on="Event",
                aggregate_function="first"
            )

            logger.info(f"After pivot: {result.height} rows with {len(result.columns)} columns")
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
                result = result.rename({source: target})

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

        result = result.rename(columns_to_rename)

        # Convert walltime to seconds with better error handling
        if "timelimit" in result.columns:
            try:
                result = result.with_columns([
                    pl.when(pl.col("timelimit").is_not_null())
                    .then(convert_walltime_to_seconds(pl.col("timelimit")))
                    .otherwise(None).alias("timelimit")
                ])
            except Exception as e:
                logger.warning(f"Error converting walltime: {e}")
                # Keep original values if conversion fails
                pass

        # Combine jobevent and Exit_status for detailed exitcode if both exist
        if "exitcode" in result.columns and "Exit_status" in result.columns:
            try:
                result = result.with_columns([
                    pl.concat_str([
                        pl.col("exitcode"),
                        pl.lit(":"),
                        pl.when(pl.col("Exit_status").is_not_null())
                        .then(pl.col("Exit_status").cast(pl.Utf8))
                        .otherwise(pl.lit(""))
                    ]).alias("exitcode")
                ])

                # Drop the original Exit_status column after combining
                result = result.drop("Exit_status")
            except Exception as e:
                logger.warning(f"Error combining exitcode fields: {e}")
                # Keep original columns if combination fails

        # Convert any missing dates to proper format
        date_columns = ["time", "submit_time", "start_time", "end_time"]
        for col in date_columns:
            if col in result.columns:
                try:
                    result = result.with_columns([
                        pl.when(
                            pl.col(col).is_not_null() &
                            ~pl.col(col).dtype.is_temporal()
                        )
                        .then(pl.col(col).cast(pl.Datetime))
                        .otherwise(pl.col(col))
                        .alias(col)
                    ])
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
            # Create expressions for each missing column
            exprs = [pl.lit(None).alias(col) for col in missing_columns]

            # Add missing columns to the dataframe
            if exprs:
                result = result.with_columns(exprs)

        # Select only the columns needed for Set 3
        # Handle the case where some columns might still be missing
        available_columns = [col for col in set3_columns if col in result.columns]
        result = result.select(available_columns)

        logger.info(f"Final result has {result.height} rows with {len(result.columns)} columns")
        return result

    except Exception as e:
        logger.error(f"Error in process_chunk: {e}", exc_info=True)
        return None


def convert_walltime_to_seconds(walltime_expr):
    """Convert HH:MM:SS format to seconds"""
    # For expressions, we can't check the dtype directly
    # Instead, use conditional logic to handle numeric and string cases

    return (
        pl.when(walltime_expr.str.contains(":"))
        .then(
            walltime_expr.str.split(":")
            .list.eval(
                pl.element().cast(pl.Int64) *
                pl.when(pl.list.len() == 3).then(pl.Int64([3600, 60, 1]))
                .when(pl.list.len() == 2).then(pl.Int64([60, 1]))
                .otherwise(pl.Int64([1]))
            )
            .list.sum()
        )
        # Try to cast directly to Int64 for numeric values
        .otherwise(walltime_expr.cast(pl.Int64))
    )


def debug_dataframe(df, name, sample_size=5):
    """Helper function to print debug information about a dataframe"""
    logger.info(f"--- Debug info for {name} ---")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns}")
    logger.info(f"Schema: {df.schema}")

    # Sample data
    if df.height > 0:
        logger.info(f"Sample data:")
        sample = df.head(sample_size)
        for row in sample.rows(named=True):
            logger.info(f"  {row}")

    # Check for null values
    for col in df.columns:
        null_count = df.filter(pl.col(col).is_null()).height
        if null_count > 0:
            logger.info(f"Column '{col}' has {null_count} null values ({null_count / df.height:.2%})")

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
            jobs_df = pl.read_csv(
                job_file,
                infer_schema_length=10000,
                schema_overrides={
                    "Resource_List.neednodes": pl.Utf8,
                    "Resource_List.nodes": pl.Utf8,
                    "Resource_List.nodect": pl.Utf8,
                    "Resource_List.ncpus": pl.Utf8,
                    "Resource_List.neednodes.ppn": pl.Utf8,
                    "Resource_List.nodes.ppn": pl.Utf8,
                    "Resource_List.mem": pl.Utf8,
                    "Resource_List.pmem": pl.Utf8,
                    "Resource_List.walltime": pl.Utf8,
                }
            )
        except Exception as e:
            logger.warning(f"First attempt to read job file failed: {e}")

            # Second approach: Try reading with ignore_errors=True
            try:
                jobs_df = pl.read_csv(
                    job_file,
                    ignore_errors=True,
                    infer_schema_length=10000,
                    schema_overrides={
                        "Resource_List.neednodes": pl.Utf8,
                        "Resource_List.nodes": pl.Utf8,
                        "Resource_List.nodect": pl.Utf8,
                        "Resource_List.ncpus": pl.Utf8,
                        "Resource_List.neednodes.ppn": pl.Utf8,
                        "Resource_List.nodes.ppn": pl.Utf8,
                        "Resource_List.mem": pl.Utf8,
                        "Resource_List.pmem": pl.Utf8,
                        "Resource_List.walltime": pl.Utf8,
                    }
                )
            except Exception as e:
                logger.warning(f"Second attempt with ignore_errors failed: {e}")

                # Last resort: Try reading with low-level settings to ensure it works
                try:
                    # Get all column names first
                    with open(job_file, 'r') as f:
                        header_line = f.readline().strip()

                    column_names = header_line.split(',')

                    # Create a schema that forces all columns to strings
                    schema = {col: pl.Utf8 for col in column_names}

                    jobs_df = pl.read_csv(
                        job_file,
                        schema=schema,
                        ignore_errors=True,
                        null_values=["", "NULL", "null", "NODE390"]  # Add known problematic values
                    )
                except Exception as e:
                    logger.error(f"All attempts to read job file failed: {e}")
                    return

        if jobs_df is None or jobs_df.height == 0:
            logger.error("Failed to read job file or file is empty")
            return

        logger.info(f"Successfully read job file with {jobs_df.height} rows")

        # Standardize job IDs
        jobs_df = jobs_df.with_columns([
            standardize_job_id(pl.col("jobID")).alias("jobID")
        ])

        # Handle datetime columns with explicit null checks
        # Use str.lengths() > 0 instead of str.len() > 0 for string length
        jobs_df = jobs_df.with_columns([
            pl.when(pl.col("start").is_not_null() & (pl.col("start").str.length() > 0))
            .then(pl.col("start").str.strptime(pl.Datetime, JOB_DATETIME_FMT))
            .otherwise(None).alias("start"),

            pl.when(pl.col("end").is_not_null() & (pl.col("end").str.length() > 0))
            .then(pl.col("end").str.strptime(pl.Datetime, JOB_DATETIME_FMT))
            .otherwise(None).alias("end")
        ])

        # Convert numeric columns safely
        numeric_cols = [
            "Resource_List.nodect",
            "Resource_List.ncpus",
        ]

        for col in numeric_cols:
            if col in jobs_df.columns:
                jobs_df = jobs_df.with_columns([
                    pl.when(
                        pl.col(col).is_not_null() &
                        pl.col(col).cast(pl.Utf8).str.contains(r'^[0-9]+$')
                    )
                    .then(pl.col(col).cast(pl.Int64))
                    .otherwise(pl.col(col))
                    .alias(col)
                ])

        # Print the job data schema for debugging
        logger.info(f"Job data schema: {jobs_df.schema}")

        print("Processing time series data in chunks...")

        # Read the time series data
        ts_reader = pl.read_csv(timeseries_file)
        total_rows = ts_reader.height
        chunks = range(0, total_rows, chunk_size)
        first_chunk = True

        for chunk_start in tqdm(chunks):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            ts_chunk = ts_reader[chunk_start:chunk_end]

            # Convert the Timestamp column to datetime using the timeseries format
            ts_chunk = ts_chunk.with_columns([
                pl.col("Timestamp").str.strptime(pl.Datetime, TS_DATETIME_FMT)
            ])

            # Process the chunk by joining and filtering
            result_df = process_chunk(jobs_df, ts_chunk)

            if result_df is not None and result_df.height > 0:
                # Write results to the output CSV file
                if first_chunk:
                    # Write with headers for first chunk
                    result_df.write_csv(output_file, include_header=True)
                    first_chunk = False
                else:
                    # For subsequent chunks, append by writing to a temporary file and concatenating
                    temp_file = output_file + '.tmp'
                    result_df.write_csv(temp_file, include_header=False)

                    # Read the temporary file content
                    with open(temp_file, 'r', encoding='utf-8') as temp:
                        content = temp.read()

                    # Append to the main file
                    with open(output_file, 'a', encoding='utf-8') as main:
                        main.write(content)

                    # Clean up temporary file
                    os.remove(temp_file)
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
            Bucket=conte_ts_bucket,
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
                            df = pl.read_csv(Path(cache_dir / temp_file))
                            dfs.append(df)
                            logger.info(f"Read {temp_file} with {df.height} rows")
                    except Exception as e:
                        logger.error(f"Error reading {temp_file}: {e}")

                # Skip if no valid data
                if not dfs:
                    logger.error(f"No valid time series data found for {date}, skipping")
                    continue

                # Combine and save time series data
                combined_ts_file = cache_dir / f"combined_{date}.csv"
                combined_df = pl.concat(dfs)
                combined_df.write_csv(combined_ts_file)
                logger.info(f"Combined {len(dfs)} files into {combined_ts_file} with {combined_df.height} rows")
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