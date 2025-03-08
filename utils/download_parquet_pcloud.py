import os
import polars as pl
from pathlib import Path
import tempfile
import concurrent.futures
import logging
from tqdm import tqdm

# Import your existing S3 client
from utils.s3 import S3_Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_csv_to_parquet(csv_path, parquet_output_dir):
    """
    Convert a CSV file to Parquet format using Polars.

    Args:
        csv_path (Path): Path to the CSV file
        parquet_output_dir (Path): Directory to save the Parquet file

    Returns:
        Path: Path to the created Parquet file
    """
    filename = csv_path.name
    parquet_filename = filename.rsplit('.', 1)[0] + '.parquet'
    parquet_path = parquet_output_dir / parquet_filename

    # Ensure output directory exists
    parquet_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Read CSV with Polars and write to Parquet
        df = pl.read_csv(csv_path)
        df.write_parquet(parquet_path)
        logger.info(f"Converted {filename} to {parquet_filename}")
        return parquet_path
    except Exception as e:
        logger.error(f"Error converting {filename}: {str(e)}")
        raise


def process_file(s3_client, file_key, temp_dir, output_dir, source_bucket):
    """
    Process a single file - download, convert, and save.

    Args:
        s3_client: S3 client instance
        file_key: S3 key for the file
        temp_dir: Temporary directory for downloads
        output_dir: Output directory for Parquet files
        source_bucket: Which bucket to download from

    Returns:
        str: Path to the created Parquet file or None if failed
    """
    try:
        # Determine which bucket to use based on source_bucket parameter
        if source_bucket == "proc":
            bucket_type = "proc"
        else:
            bucket_type = "job"

        # Download the file
        local_path = s3_client.download_file(file_key, temp_dir, bucket_type)

        # Convert to Parquet
        parquet_path = convert_csv_to_parquet(local_path, output_dir)

        # Clean up the CSV file
        os.remove(local_path)

        return parquet_path
    except Exception as e:
        logger.error(f"Failed to process {file_key}: {str(e)}")
        return None


def main():
    # Configuration
    upload_bucket = "your-upload-bucket"  # Replace with your bucket name if needed
    download_bucket_proc_metric = "conte-transform-to-fresco-ts"  # Replace with your bucket name
    download_bucket_job_accounting = "your-job-accounting-bucket"  # Replace with your bucket name
    output_dir = Path(r"P:\Stampede\stampede-ts-to-fresco-ts-chunks-1")

    # Source bucket to use (either "proc" or "job")
    source_bucket = "proc"  # Change to "job" if using job accounting bucket

    # Create S3 client
    s3_client = S3_Client(
        upload_bucket=upload_bucket,
        download_bucket_proc_metric=download_bucket_proc_metric,
        download_bucket_job_accounting=download_bucket_job_accounting
    )

    # Get the bucket to use
    bucket = download_bucket_proc_metric if source_bucket == "proc" else download_bucket_job_accounting

    # List all files in the bucket
    logger.info(f"Listing files in bucket: {bucket}")
    files = s3_client.list_s3_files(bucket)

    # Filter for CSV files only
    csv_files = [f for f in files if f.lower().endswith('.csv')]
    logger.info(f"Found {len(csv_files)} CSV files to convert")

    # Create temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Process files with progress bar
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    process_file, s3_client, file_key, temp_path, output_dir, source_bucket
                ): file_key for file_key in csv_files
            }

            # Process as they complete with a progress bar
            with tqdm(total=len(csv_files), desc="Converting files") as pbar:
                for future in concurrent.futures.as_completed(future_to_file):
                    file_key = future_to_file[future]
                    try:
                        parquet_path = future.result()
                        if parquet_path:
                            logger.info(f"Successfully processed: {file_key} -> {parquet_path}")
                    except Exception as e:
                        logger.error(f"Exception processing {file_key}: {str(e)}")
                    finally:
                        pbar.update(1)

    logger.info("Conversion completed")


if __name__ == "__main__":
    main()
