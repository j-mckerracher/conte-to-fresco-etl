import os
import logging
import re
import json
import shutil
import time
import uuid
from collections import defaultdict
import threading
from utils.ready_signal_creator import ReadySignalManager, JobStatus
from utils.split_parquet_files_to_smaller_files import ParquetFileSplitter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("conte_job_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# File paths
source_dir = r"P:\Conte\conte-ts-to-fresco-ts-1"
source_dir_accounting = r"P:\Conte\conte-job-accounting-1"
destination_dir = r"P:\Conte\conte-transformed-2"
server_input_dir = r"U:\projects\conte-to-fresco-etl\cache\input\metrics"
server_accounting_input_dir = r"U:\projects\conte-to-fresco-etl\cache\accounting"
server_complete_dir = r"U:\projects\conte-to-fresco-etl\cache\output"
job_tracking_dir = r"job_tracking"

# Configuration
CHUNK_SIZE = 1000000  # Default chunk size for splitting files (1 million rows)
MAX_ACTIVE_JOBS = 2  # Maximum number of jobs to have active at once
MAX_SERVER_DIR_SIZE = 25 * 1024 * 1024 * 1024  # 25GB max in server directory
CHECK_INTERVAL = 60  # Check for completed jobs every 60 seconds
FILE_SIZE_SPLIT_THRESHOLD = 5  # Files larger than 5GB will be split

# Tracking information
active_jobs = {}  # Track active jobs and their status
processed_files = set()  # Track already processed files
processed_accounting_files = set()  # Track already processed accounting files
month_tracking = {}  # Track month completion status
already_processed_year_months = []  # Will be loaded from tracking file

# Create a global instance of the ReadySignalManager
signal_manager = ReadySignalManager(ready_dir=r"U:\projects\conte-to-fresco-etl\cache\ready", logger=logger)


def setup_directories():
    """Ensure all required directories exist"""
    for dir_path in [server_input_dir, server_accounting_input_dir, server_complete_dir, job_tracking_dir]:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")


def load_processed_files():
    """Load the list of already processed files"""
    global processed_files, processed_accounting_files, already_processed_year_months

    # Load processed time series files
    processed_file_path = os.path.join(job_tracking_dir, "processed_files.json")
    if os.path.exists(processed_file_path):
        try:
            with open(processed_file_path, 'r') as f:
                processed_files = set(json.load(f))
            logger.info(f"Loaded {len(processed_files)} previously processed time series files")
        except Exception as e:
            logger.error(f"Error loading processed files: {str(e)}")

    # Load processed accounting files
    processed_accounting_path = os.path.join(job_tracking_dir, "processed_accounting_files.json")
    if os.path.exists(processed_accounting_path):
        try:
            with open(processed_accounting_path, 'r') as f:
                processed_accounting_files = set(json.load(f))
            logger.info(f"Loaded {len(processed_accounting_files)} previously processed accounting files")
        except Exception as e:
            logger.error(f"Error loading processed accounting files: {str(e)}")

    # Load already processed year-months
    processed_year_months_path = os.path.join(job_tracking_dir, "processed_year_months.json")
    if os.path.exists(processed_year_months_path):
        try:
            with open(processed_year_months_path, 'r') as f:
                already_processed_year_months = json.load(f)
            logger.info(f"Loaded {len(already_processed_year_months)} previously processed year-months")
        except Exception as e:
            logger.error(f"Error loading processed year-months: {str(e)}")


def save_processed_files():
    """Save the tracking information"""
    # Save processed time series files
    processed_file_path = os.path.join(job_tracking_dir, "processed_files.json")
    try:
        with open(processed_file_path, 'w') as f:
            json.dump(list(processed_files), f)
        logger.debug(f"Saved {len(processed_files)} processed time series files")
    except Exception as e:
        logger.error(f"Error saving processed files: {str(e)}")

    # Save processed accounting files
    processed_accounting_path = os.path.join(job_tracking_dir, "processed_accounting_files.json")
    try:
        with open(processed_accounting_path, 'w') as f:
            json.dump(list(processed_accounting_files), f)
        logger.debug(f"Saved {len(processed_accounting_files)} processed accounting files")
    except Exception as e:
        logger.error(f"Error saving processed accounting files: {str(e)}")

    # Save processed year-months
    processed_year_months_path = os.path.join(job_tracking_dir, "processed_year_months.json")
    try:
        with open(processed_year_months_path, 'w') as f:
            json.dump(already_processed_year_months, f)
        logger.debug(f"Saved {len(already_processed_year_months)} processed year-months")
    except Exception as e:
        logger.error(f"Error saving processed year-months: {str(e)}")


def load_month_tracking():
    """Load month tracking information"""
    global month_tracking

    month_tracking_path = os.path.join(job_tracking_dir, "month_tracking.json")
    if os.path.exists(month_tracking_path):
        try:
            with open(month_tracking_path, 'r') as f:
                month_tracking = json.load(f)
            logger.info(f"Loaded tracking for {len(month_tracking)} months")
        except Exception as e:
            logger.error(f"Error loading month tracking: {str(e)}")
            month_tracking = {}


def save_month_tracking():
    """Save month tracking information"""
    month_tracking_path = os.path.join(job_tracking_dir, "month_tracking.json")
    try:
        with open(month_tracking_path, 'w') as f:
            json.dump(month_tracking, f)
        logger.debug(f"Saved tracking for {len(month_tracking)} months")
    except Exception as e:
        logger.error(f"Error saving month tracking: {str(e)}")


def find_source_files():
    """Find all files in the source directory that need processing"""
    # Dictionary to store files by year-month
    year_month_files = defaultdict(list)

    # Pattern to extract year and month from filenames
    ts_pattern = re.compile(r'FRESCO_Conte_ts_(\d{4})_(\d{2})_v\d+\.parquet$')

    # Walk through the source directory
    for root, _, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.endswith('.parquet'):
                match = ts_pattern.search(filename)
                if match:
                    year, month = match.groups()
                    year_month = f"{year}-{month}"

                    # Skip if already processed
                    if year_month in already_processed_year_months:
                        continue

                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, source_dir)

                    if rel_path in processed_files:
                        continue

                    # Add file to the appropriate year-month group
                    file_size = os.path.getsize(full_path)
                    year_month_files[year_month].append((rel_path, full_path, file_size, year, month))

    return year_month_files


def find_accounting_files():
    """Find all accounting files in the accounting source directory"""
    # Dictionary to store accounting files by year-month
    accounting_files = {}

    # Pattern to extract year and month from filenames
    acc_pattern = re.compile(r'(\d{4})-(\d{2})\.csv$')

    # Walk through the accounting source directory
    for root, _, filenames in os.walk(source_dir_accounting):
        for filename in filenames:
            if filename.endswith('.csv'):
                match = acc_pattern.match(filename)
                if match:
                    year, month = match.groups()
                    year_month = f"{year}-{month}"

                    # Skip if already processed
                    if year_month in already_processed_year_months:
                        continue

                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, source_dir_accounting)

                    if rel_path in processed_accounting_files:
                        continue

                    # Add file to the accounting files dictionary
                    accounting_files[year_month] = (rel_path, full_path)

    return accounting_files


def get_server_directory_size(directory):
    """Get the total size of files in a directory"""
    total_size = 0
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            total_size += os.path.getsize(file_path)
    return total_size


def process_accounting_file(accounting_file, year_month):
    """Copy accounting file to the server accounting directory"""
    dest_file = os.path.join(server_accounting_input_dir, os.path.basename(accounting_file))

    try:
        # Copy the accounting file
        shutil.copy2(accounting_file, dest_file)
        logger.info(f"Copied accounting file {accounting_file} to {dest_file}")

        # Add to processed files
        rel_path = os.path.relpath(accounting_file, source_dir_accounting)
        processed_accounting_files.add(rel_path)

        return True
    except Exception as e:
        logger.error(f"Error copying accounting file {accounting_file}: {str(e)}")
        return False


def create_job_for_year_month(year_month, ts_files, accounting_file):
    """Create a job for processing a specific year-month combination"""
    year, month = year_month.split('-')

    # Skip if this year-month is already being processed
    active_jobs_for_month = [job for job_id, job in active_jobs.items()
                             if job.get("year_month") == year_month]

    if active_jobs_for_month:
        logger.info(f"Month {year_month} already has active jobs. Skipping.")
        return None

    # Initialize month tracking if it doesn't exist yet
    if year_month not in month_tracking:
        month_tracking[year_month] = {
            "total_files": len(ts_files),
            "processed_files": 0,
            "started_at": time.time(),
            "is_complete": False,
            "accounting_file": os.path.basename(accounting_file)
        }

    # Generate job ID
    job_id = f"conte_{year}_{month}_{uuid.uuid4().hex[:8]}"

    # Create job data
    job = {
        "job_id": job_id,
        "year_month": year_month,
        "ts_files": [f[1] for f in ts_files],  # Use full paths
        "accounting_file": accounting_file,
        "status": "pending"
    }

    # Update month tracking
    month_tracking[year_month]["job_id"] = job_id
    save_month_tracking()

    return job


def process_job(job):
    """Process a job by splitting files and copying them to the server directories"""
    job_id = job["job_id"]
    year_month = job["year_month"]
    ts_files = job["ts_files"]
    accounting_file = job["accounting_file"]

    year, month = year_month.split('-')

    logger.info(f"Processing job {job_id} for {year_month} with {len(ts_files)} files")

    # Update job status
    job["status"] = "processing"
    active_jobs[job_id] = job

    # Process time series files first
    all_files_success = True
    processed_count = 0
    created_files = []

    # Instantiate the ParquetFileSplitter
    splitter = ParquetFileSplitter(chunk_size=CHUNK_SIZE, logger=logger)

    for ts_file in ts_files:
        # Extract version from filename
        version_match = re.search(r'v(\d+)', os.path.basename(ts_file))
        if not version_match:
            logger.warning(f"Could not extract version from filename {ts_file}. Skipping.")
            continue

        version = version_match.group(1)

        # Check file size to determine if splitting is needed
        file_size = os.path.getsize(ts_file)
        file_size_gb = file_size / (1024 * 1024 * 1024)

        logger.info(f"Processing file {ts_file} (Size: {file_size_gb:.2f} GB)")

        try:
            if file_size_gb > FILE_SIZE_SPLIT_THRESHOLD:
                # For large files, split into chunks
                logger.info(f"File is large ({file_size_gb:.2f} GB), splitting into chunks")

                # Prepare a prefix for the split files
                prefix = f"FRESCO_Conte_ts_{year}_{month}_v{version}"

                # Split the file into the server input directory
                success, chunk_files = splitter.split_file(
                    input_file=ts_file,
                    output_dir=server_input_dir,
                    prefix=prefix
                )

                if success:
                    logger.info(f"Successfully split {ts_file} into {len(chunk_files)} chunks")
                    created_files.extend(chunk_files)
                else:
                    logger.error(f"Failed to split file {ts_file}")
                    all_files_success = False
                    continue
            else:
                # For smaller files, just copy directly
                dest_filename = os.path.basename(ts_file)
                dest_file = os.path.join(server_input_dir, dest_filename)

                # Copy the file
                shutil.copy2(ts_file, dest_file)
                logger.info(f"Copied file {ts_file} to {dest_file}")
                created_files.append(dest_file)

            # Add to processed files
            rel_path = os.path.relpath(ts_file, source_dir)
            processed_files.add(rel_path)
            processed_count += 1

            # Update progress
            job["processed_files"] = processed_count
            month_tracking[year_month]["processed_files"] = processed_count

            logger.info(f"Successfully processed file {processed_count}/{len(ts_files)} for {year_month}")

        except Exception as e:
            logger.error(f"Error processing file {ts_file}: {str(e)}")
            all_files_success = False

    # Process accounting file AFTER all time series files are processed
    if all_files_success and processed_count == len(ts_files):
        if not process_accounting_file(accounting_file, year_month):
            logger.error(f"Failed to process accounting file for {year_month}")
            all_files_success = False
        else:
            logger.info(f"Successfully processed accounting file for {year_month}")

    # Update job status and create ready signal
    if all_files_success and processed_count == len(ts_files):
        job["status"] = "processing_complete"
        month_tracking[year_month]["files_processed"] = True
        logger.info(f"All files for {year_month} have been successfully processed")

        # Create a ready signal to notify the server processor
        signal_manager.create_ready_signal(year, month)
        logger.info(f"Created ready signal for {year_month}")
    else:
        job["status"] = "processing_partial"
        logger.warning(f"Some files for {year_month} could not be processed")

    # Save tracking information
    save_processed_files()
    save_month_tracking()

    return all_files_success


def check_completed_jobs():
    """Check for jobs that have been completed by the server processor"""
    completed_jobs = []

    # Check each job in month_tracking that hasn't been marked as complete
    for year_month, tracking_data in month_tracking.items():
        if tracking_data.get("is_complete", False):
            continue

        year, month = year_month.split('-')
        job_id = tracking_data.get("job_id")

        if not job_id or job_id not in active_jobs:
            continue

        # Check if the server processor has completed this job
        status = signal_manager.check_status(year, month)

        if status == JobStatus.COMPLETE:
            logger.info(f"Job {job_id} for {year_month} has been completed by server processor")
            completed_jobs.append(job_id)
        elif status == JobStatus.FAILED:
            logger.error(f"Job {job_id} for {year_month} failed during server processing")
            # Could implement retry logic here if desired

    return completed_jobs


def process_completed_job(job_id):
    """Process a job that has been completed by the server processor"""
    if job_id not in active_jobs:
        logger.warning(f"Job {job_id} not found in active jobs")
        return False

    job = active_jobs[job_id]
    year_month = job["year_month"]
    year, month = year_month.split('-')

    # Source file path in server complete directory
    source_file = os.path.join(server_complete_dir, f"transformed_{year}_{month}.parquet")

    if not os.path.exists(source_file):
        logger.warning(f"Output file for job {job_id} not found: {source_file}")
        return False

    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Destination file path
    dest_file = os.path.join(destination_dir, f"transformed_{year}_{month}.parquet")

    try:
        # Copy the file to the destination
        shutil.copy2(source_file, dest_file)
        logger.info(f"Copied output file {source_file} to {dest_file}")

        # Update month tracking
        month_tracking[year_month]["is_complete"] = True
        month_tracking[year_month]["completed_at"] = time.time()
        month_tracking[year_month]["destination_file"] = dest_file

        # Add to already processed year-months
        if year_month not in already_processed_year_months:
            already_processed_year_months.append(year_month)

        # Save tracking information
        save_month_tracking()
        save_processed_files()

        # Remove job from active jobs
        del active_jobs[job_id]

        logger.info(f"Successfully completed processing for {year_month}")
        return True

    except Exception as e:
        logger.error(f"Error copying output file for {year_month}: {str(e)}")
        return False


def monitor_completed_jobs():
    """Thread function to continuously monitor for completed jobs"""
    logger.info("Started monitoring thread for completed jobs")

    while True:
        try:
            # Check for completed jobs
            completed_jobs = check_completed_jobs()

            # Process completed jobs
            for job_id in completed_jobs:
                process_completed_job(job_id)

            # Wait before checking again
            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            logger.error(f"Error in monitoring thread: {str(e)}")
            time.sleep(CHECK_INTERVAL * 2)  # Wait longer after an error


def main():
    start_time = time.time()
    logger.info("Starting Conte ETL job manager...")

    # Setup required directories
    setup_directories()

    # Load tracking information
    load_processed_files()
    load_month_tracking()

    # Start monitoring thread for completed jobs
    monitor_thread = threading.Thread(target=monitor_completed_jobs, daemon=True)
    monitor_thread.start()

    try:
        while True:
            # Check current active jobs count
            current_active_jobs = len(active_jobs)

            if current_active_jobs >= MAX_ACTIVE_JOBS:
                logger.info(f"Maximum active jobs reached ({current_active_jobs}/{MAX_ACTIVE_JOBS}). Waiting...")
                time.sleep(60)
                continue

            # Check server directory size
            server_metrics_size = get_server_directory_size(server_input_dir)
            if server_metrics_size >= MAX_SERVER_DIR_SIZE:
                logger.info(
                    f"Server metrics directory size limit reached ({server_metrics_size}/{MAX_SERVER_DIR_SIZE} bytes). Waiting...")
                time.sleep(60)
                continue

            # Find time series files by year-month
            year_month_files = find_source_files()

            # Find accounting files
            accounting_files = find_accounting_files()

            # Find year-months with both time series and accounting files
            valid_year_months = []
            for year_month in year_month_files:
                if year_month in accounting_files:
                    valid_year_months.append(
                        (year_month, year_month_files[year_month], accounting_files[year_month][1]))

            if not valid_year_months:
                logger.info("No new valid year-month combinations to process. Waiting...")
                time.sleep(300)  # Wait 5 minutes before checking again
                continue

            logger.info(f"Found {len(valid_year_months)} valid year-month combinations to process")

            # Create and process jobs
            jobs_to_submit = min(MAX_ACTIVE_JOBS - current_active_jobs, len(valid_year_months))

            if jobs_to_submit <= 0:
                logger.info("No capacity for new jobs. Waiting...")
                time.sleep(60)
                continue

            logger.info(f"Submitting {jobs_to_submit} new jobs")

            for i in range(jobs_to_submit):
                year_month, ts_files, accounting_file = valid_year_months[i]

                # Create job
                job = create_job_for_year_month(year_month, ts_files, accounting_file)

                if job:
                    # Process the job
                    process_success = process_job(job)

                    if process_success:
                        logger.info(f"Job {job['job_id']} for {year_month} successfully submitted")
                    else:
                        logger.warning(f"Job {job['job_id']} for {year_month} had issues during processing")

            # Wait a bit before checking for more jobs
            time.sleep(30)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down gracefully...")

    except Exception as e:
        logger.error(f"Unexpected error in main loop: {str(e)}")

    # Wait for monitor thread to finish cleanly
    logger.info("Waiting for monitoring thread to complete...")
    monitor_thread.join(timeout=10)

    # Final save of tracking information
    save_processed_files()
    save_month_tracking()

    total_runtime = time.time() - start_time
    logger.info(f"Job manager shutting down after running for {total_runtime:.2f} seconds")


if __name__ == "__main__":
    main()