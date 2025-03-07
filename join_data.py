from scipy import sparse
import re
import numpy as np
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


def process_chunk_sparse(jobs_df, ts_chunk):
    """Process a chunk of time series data against all jobs using sparse matrices for memory efficiency"""
    try:
        # Check if dataframes are valid
        if jobs_df is None or jobs_df.empty or ts_chunk is None or ts_chunk.empty:
            logger.warning("Empty dataframe passed to process_chunk_sparse")
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

        try:
            # Convert Event to categorical codes for indexing
            filtered.loc[:, "event_code"] = filtered["Event"].astype('category').cat.codes

            # Make sure Value is float64 to handle large numbers
            filtered.loc[:, "Value"] = filtered["Value"].astype('float64')

            # Define group columns - all columns except Event, Value, and event_code
            # Limit the number of grouping columns to avoid excessive combinations
            essential_group_cols = ["jobID", "Host", "Timestamp"]
            group_cols = [col for col in essential_group_cols if col in filtered.columns]

            # Add debug logging for the group columns
            logger.info(f"Using group columns: {group_cols}")

            # Check for NaN values in group columns before groupby
            for col in group_cols:
                if filtered[col].isna().any():
                    logger.warning(f"Column {col} contains {filtered[col].isna().sum()} NaN values")
                    # Fill NaNs with a placeholder to prevent groupby issues
                    filtered.loc[filtered[col].isna(), col] = f"_MISSING_{col}_"

            # FIX 1: Use a more robust approach for creating row keys
            # Instead of ngroup() which might have integer overflow issues, manually create indices
            try:
                # Create a unique string key for combining group columns
                filtered.loc[:, "group_key"] = filtered[group_cols].astype(str).agg('_'.join, axis=1)

                # Map these string keys to integers starting from 0
                unique_keys = filtered["group_key"].unique()
                key_to_idx = {key: idx for idx, key in enumerate(unique_keys)}
                filtered.loc[:, "row_key"] = filtered["group_key"].map(key_to_idx)

                logger.info(f"Created {len(unique_keys)} unique row combinations")
            except Exception as e:
                logger.error(f"Error creating row keys: {e}")
                # Fallback: Use a simpler approach if the groupby fails
                filtered.loc[:, "row_key"] = range(len(filtered))
                logger.warning("Using row index as fallback for row_key")

            # Get the unique row keys and event codes
            row_keys = filtered["row_key"].unique()
            event_types = filtered["Event"].unique()

            logger.info(
                f"Creating sparse matrix with {len(row_keys)} unique row combinations and {len(event_types)} event types")

            # FIX 2: Validate indices before creating the sparse matrix
            rows = filtered["row_key"].values
            cols = filtered["event_code"].values
            values = filtered["Value"].values

            # Check for invalid indices
            if np.any(np.isnan(rows)) or np.any(np.isnan(cols)):
                logger.error("Found NaN values in row or column indices")
                # Remove rows with NaN indices
                valid_mask = ~(np.isnan(rows) | np.isnan(cols))
                rows = rows[valid_mask]
                cols = cols[valid_mask]
                values = values[valid_mask]

            # Check for negative indices
            if np.any(rows < 0) or np.any(cols < 0):
                logger.error(f"Found negative indices - min row: {np.min(rows)}, min col: {np.min(cols)}")
                # Remove rows with negative indices
                valid_mask = (rows >= 0) & (cols >= 0)
                rows = rows[valid_mask]
                cols = cols[valid_mask]
                values = values[valid_mask]

            # Ensure indices are within bounds
            if len(rows) > 0:  # Only proceed if we have data left
                max_row = np.max(rows)
                max_col = np.max(cols)

                if max_row >= len(row_keys) or max_col >= len(event_types):
                    logger.warning(f"Index out of bounds: max_row={max_row}, row_keys={len(row_keys)}, "
                                   f"max_col={max_col}, event_types={len(event_types)}")
                    # Adjust the shape to accommodate all indices
                    shape = (max(len(row_keys), max_row + 1), max(len(event_types), max_col + 1))
                else:
                    shape = (len(row_keys), len(event_types))

                # FIX 3: Cast indices to int32 to avoid int64 issues
                rows = rows.astype(np.int32)
                cols = cols.astype(np.int32)

                # Create sparse matrix with explicit dtype
                logger.info(f"Creating sparse matrix with shape {shape} and {len(values)} values")
                sparse_matrix = sparse.coo_matrix(
                    (values, (rows, cols)),
                    shape=shape,
                    dtype=np.float64
                )

                # Convert to dense array for easier handling
                dense_values = sparse_matrix.toarray()

                # FIX 4: Create a result dataframe more robustly
                # Create a dataframe with the group columns and row_key
                result = filtered[group_cols + ["row_key"]].drop_duplicates(subset=["row_key"])
                result = result.sort_values("row_key").reset_index(drop=True)

                # Add event columns
                for i, event in enumerate(event_types):
                    # Only add columns for event types that exist in our matrix
                    if i < dense_values.shape[1]:
                        result[event] = dense_values[:, i]

                # Drop the row_key column
                result = result.drop(columns=["row_key"])

                logger.info(f"After sparse pivot: {len(result)} rows with {len(result.columns)} columns")

                # Continue with column mapping and transformation as in the original code
                # ...

                return result
            else:
                logger.error("No valid data left after filtering indices")
                return None

        except Exception as e:
            logger.error(f"Error during sparse matrix operation: {e}", exc_info=True)

            # Add detailed debugging for the Value column
            if "Value" in filtered.columns:
                try:
                    # Log info about the Value column
                    value_min = filtered["Value"].min()
                    value_max = filtered["Value"].max()
                    value_mean = filtered["Value"].mean()
                    value_dtype = filtered["Value"].dtype
                    value_has_nan = filtered["Value"].isna().any()

                    logger.info(f"Value column stats - Min: {value_min}, Max: {value_max}, Mean: {value_mean}, "
                                f"Dtype: {value_dtype}, Has NaN: {value_has_nan}")
                except Exception as ve:
                    logger.warning(f"Error analyzing Value column: {ve}")

            # FIX 5: Fallback to a non-sparse approach if sparse matrix fails
            logger.info("Attempting fallback to non-sparse pivot approach")
            try:
                # Simple pivot approach as fallback
                # Use only essential columns to avoid memory issues
                pivot_cols = ["jobID", "Host", "Timestamp", "Event", "Value"]
                pivot_df = filtered[pivot_cols].copy()

                # Simple pivot
                result = pivot_df.pivot_table(
                    index=["jobID", "Host", "Timestamp"],
                    columns="Event",
                    values="Value",
                    aggfunc='first'
                ).reset_index()

                logger.info(f"Fallback pivot successful with {len(result)} rows")
                return result
            except Exception as pivot_e:
                logger.error(f"Fallback pivot also failed: {pivot_e}")
                return None

    except Exception as e:
        logger.error(f"Error in process_chunk_sparse: {e}", exc_info=True)
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
            result_df = process_chunk_sparse(jobs_df, ts_chunk)

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