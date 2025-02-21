import os
import glob
import json
import re
from pathlib import Path
import gc
import psutil
import boto3
from botocore.exceptions import ClientError
import shutil
import threading
from queue import Queue
import time
from typing import List, Dict
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import numpy as np
import polars as pl


def parse_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    """Parse timestamp column in MM/DD/YYYY HH:MM:SS format"""
    return df.with_columns([
        pl.col('timestamp').str.strptime(pl.Datetime, '%m/%d/%Y %H:%M:%S').alias('Timestamp')
    ])


def safe_division(numerator, denominator, default: float = 0.0) -> float:
    """Safely perform division with error handling"""
    try:
        return numerator / denominator if denominator != 0 else default
    except Exception:
        return default


def validate_metric(value: float, min_val: float = 0.0, max_val: float = float('inf')) -> float:
    """Ensure metric values are within valid range"""
    return np.clip(value, min_val, max_val)


def calculate_rate(current_value: float, previous_value: float,
                   time_delta_seconds: float) -> float:
    """Calculate rate of change per second"""
    return safe_division(current_value - previous_value, time_delta_seconds)


def process_block_file(file_path: str) -> pl.DataFrame:
    """Process block.csv file with improved error handling"""
    # Define schema with u64 for all numeric columns
    schema = {
        'rd_ios': pl.UInt64,
        'rd_merges': pl.UInt64,
        'rd_sectors': pl.UInt64,
        'rd_ticks': pl.UInt64,
        'wr_ios': pl.UInt64,
        'wr_merges': pl.UInt64,
        'wr_sectors': pl.UInt64,
        'wr_ticks': pl.UInt64,
        'in_flight': pl.UInt64,
        'io_ticks': pl.UInt64,
        'time_in_queue': pl.UInt64
    }

    try:
        # Read CSV with explicit datetime parsing
        df = pl.read_csv(
            file_path,
            schema_overrides=schema,
            try_parse_dates=False  # We'll parse timestamp manually
        )

        # Parse timestamp
        df = parse_timestamp(df)

        # Calculate I/O throughput
        df = df.with_columns([
            (pl.col('rd_sectors') + pl.col('wr_sectors')).alias('total_sectors'),
            (pl.col('rd_ticks') + pl.col('wr_ticks')).alias('total_ticks'),
            pl.col('jobID').str.replace_all('jobID', 'JOB', case_sensitive=False).alias('jobID')
        ])

        # Convert sectors to bytes and calculate throughput
        df = df.with_columns([
            (pl.col('total_sectors') * 512).alias('bytes_processed')
        ])

        # Calculate throughput in GB/s with validation
        df = df.with_columns([
            (pl.when(pl.col('total_ticks') > 0)
             .then(pl.col('bytes_processed') / pl.col('total_ticks') / (1024 * 1024 * 1024))
             .otherwise(0.0))
            .clip(0, None)
            .alias('Value')
        ])

        return pl.DataFrame({
            'Job Id': df['jobID'],
            'Host': df['node'],
            'Event': pl.lit('block').repeat(len(df)),
            'Value': df['Value'],
            'Units': pl.lit('GB/s').repeat(len(df)),
            'Timestamp': df['Timestamp']
        })

    except Exception as e:
        print(f"Error processing block file: {str(e)}")
        return None


def process_cpu_file(file_path: str) -> pl.DataFrame:
    """Process cpu.csv file with support for multi-core CPU percentages"""
    # Define schema with u64 for CPU metrics
    schema = {
        'user': pl.UInt64,
        'nice': pl.UInt64,
        'system': pl.UInt64,
        'idle': pl.UInt64,
        'iowait': pl.UInt64,
        'irq': pl.UInt64,
        'softirq': pl.UInt64
    }

    try:
        # Read CSV with explicit datetime parsing
        df = pl.read_csv(
            file_path,
            schema_overrides=schema,
            try_parse_dates=False  # We'll parse timestamp manually
        )

        # Parse timestamp
        df = parse_timestamp(df)

        # Calculate total CPU time
        df = df.with_columns([
            (pl.col('user') + pl.col('nice') + pl.col('system') +
             pl.col('idle') + pl.col('iowait') + pl.col('irq') +
             pl.col('softirq')).alias('total'),
            (pl.col('user') + pl.col('nice')).alias('user_time'),
            pl.col('jobID').str.replace_all('jobID', 'JOB', case_sensitive=False).alias('jobID')
        ])

        # Calculate CPU percentage with validation
        df = df.with_columns([
            (pl.when(pl.col('total') > 0)
             .then((pl.col('user_time') / pl.col('total')) * 100)
             .otherwise(0.0))
            .clip(0, 100)
            .alias('Value')
        ])

        return pl.DataFrame({
            'Job Id': df['jobID'],
            'Host': df['node'],
            'Event': pl.lit('cpuuser').repeat(len(df)),
            'Value': df['Value'],
            'Units': pl.lit('CPU %').repeat(len(df)),
            'Timestamp': df['Timestamp']
        })

    except Exception as e:
        print(f"Error processing CPU file: {str(e)}")
        return None


def process_mem_file(file_path: str) -> pl.DataFrame:
    """Process mem.csv file with improved validation"""
    # Define schema with u64 for memory values
    schema = {
        'MemTotal': pl.UInt64,
        'MemFree': pl.UInt64,
        'MemUsed': pl.UInt64,
        'Active': pl.UInt64,
        'Inactive': pl.UInt64,
        'Dirty': pl.UInt64,
        'Writeback': pl.UInt64,
        'FilePages': pl.UInt64,
        'Mapped': pl.UInt64,
        'AnonPages': pl.UInt64,
        'PageTables': pl.UInt64,
        'NFS_Unstable': pl.UInt64,
        'Bounce': pl.UInt64,
        'Slab': pl.UInt64,
        'HugePages_Total': pl.UInt64,
        'HugePages_Free': pl.UInt64
    }

    try:
        # Read CSV with explicit datetime parsing
        df = pl.read_csv(
            file_path,
            schema_overrides=schema,
            try_parse_dates=False  # We'll parse timestamp manually
        )

        # Parse timestamp
        df = parse_timestamp(df)

        # Ensure memory values are non-negative and validate
        df = df.with_columns([
            pl.col('MemTotal').clip(0, None).alias('MemTotal'),
            pl.col('MemFree').clip(0, None).alias('MemFree'),
            pl.col('FilePages').clip(0, None).alias('FilePages'),
            pl.col('jobID').str.replace_all('jobID', 'JOB', case_sensitive=False).alias('jobID')
        ])

        # Ensure MemFree doesn't exceed MemTotal
        df = df.with_columns([
            pl.min_horizontal([
                pl.col('MemFree'),
                pl.col('MemTotal')
            ]).alias('MemFree')
        ])

        # Calculate memory metrics in GB
        df = df.with_columns([
            ((pl.col('MemTotal') - pl.col('MemFree')) / (1024 * 1024))
            .clip(0, None)
            .alias('memused_value'),

            ((pl.col('MemTotal') - pl.col('MemFree') - pl.col('FilePages')) / (1024 * 1024))
            .clip(0, None)
            .alias('memused_minus_diskcache_value')
        ])

        # Create separate dataframes for each metric
        memused = pl.DataFrame({
            'Job Id': df['jobID'],
            'Host': df['node'],
            'Event': pl.lit('memused').repeat(len(df)),
            'Value': df['memused_value'],
            'Units': pl.lit('GB').repeat(len(df)),
            'Timestamp': df['Timestamp']
        })

        memused_minus_diskcache = pl.DataFrame({
            'Job Id': df['jobID'],
            'Host': df['node'],
            'Event': pl.lit('memused_minus_diskcache').repeat(len(df)),
            'Value': df['memused_minus_diskcache_value'],
            'Units': pl.lit('GB').repeat(len(df)),
            'Timestamp': df['Timestamp']
        })

        return pl.concat([memused, memused_minus_diskcache])

    except Exception as e:
        print(f"Error processing memory file: {str(e)}")
        return None


def process_nfs_file(file_path: str) -> pl.DataFrame:
    """Process llite.csv file with improved rate calculation"""
    # Define all numeric columns as UInt64
    schema = {col: pl.UInt64 for col in [
        'inode_revalidate', 'dentry_revalidate', 'data_invalidate', 'attr_invalidate',
        'vfs_open', 'vfs_lookup', 'vfs_access', 'vfs_updatepage', 'vfs_readpage',
        'vfs_readpages', 'vfs_writepage', 'vfs_writepages', 'vfs_getdents',
        'vfs_setattr', 'vfs_flush', 'vfs_fsync', 'vfs_lock', 'vfs_release',
        'congestion_wait', 'setattr_trunc', 'extend_write', 'silly_rename',
        'short_read', 'short_write', 'delay', 'normal_read', 'normal_write',
        'direct_read', 'direct_write', 'server_read', 'server_write',
        'read_page', 'write_page', 'xprt_sends', 'xprt_recvs', 'xprt_bad_xids',
        'xprt_req_u', 'xprt_bklog_u'
    ]}

    try:
        # Read CSV with explicit datetime parsing
        df = pl.read_csv(
            file_path,
            schema_overrides=schema,
            try_parse_dates=False  # We'll parse timestamp manually
        )

        # Parse timestamp and prepare data
        df = parse_timestamp(df)

        df = df.with_columns([
            pl.col('jobID').str.replace_all('jobID', 'JOB', case_sensitive=False).alias('jobID')
        ])

        # Calculate total bytes and sort
        df = df.with_columns([
            (pl.col('normal_read') + pl.col('normal_write')).alias('total_bytes')
        ]).sort(['jobID', 'node', 'Timestamp'])

        # Calculate time deltas and byte deltas by group
        df = df.with_columns([
            pl.col('Timestamp')
            .diff()
            .dt.seconds()
            .over(['jobID', 'node'])
            .alias('time_delta'),

            pl.col('total_bytes')
            .diff()
            .over(['jobID', 'node'])
            .alias('byte_delta')
        ])

        # Calculate MB/s with validation
        df = df.with_columns([
            (pl.when(pl.col('time_delta') > 0)
             .then(pl.col('byte_delta') / pl.col('time_delta') / (1024 * 1024))
             .otherwise(0.0))
            .clip(0, None)
            .alias('Value')
        ])

        return pl.DataFrame({
            'Job Id': df['jobID'],
            'Host': df['node'],
            'Event': pl.lit('nfs').repeat(len(df)),
            'Value': df['Value'],
            'Units': pl.lit('MB/s').repeat(len(df)),
            'Timestamp': df['Timestamp']
        })

    except Exception as e:
        print(f"Error processing NFS file: {str(e)}")
        return None


class ThreadedDownloader:
    def __init__(self, num_threads: int = 4, max_retries: int = 3, timeout: int = 300):
        self.num_threads = num_threads
        self.max_retries = max_retries
        self.timeout = timeout
        self.download_queue = Queue()
        self.results = {}
        self.lock = threading.Lock()
        self.completed_downloads = 0
        self.total_downloads = 0
        self.print_lock = threading.Lock()

    def download_worker(self, headers: dict):
        """Worker thread function to process download queue"""
        while True:
            try:
                # Get the next task from queue
                task = self.download_queue.get()
                if task is None:
                    break

                url, local_path = task
                success = False

                # Attempt download with retries
                for attempt in range(self.max_retries):
                    try:
                        response = requests.get(url, headers=headers, timeout=self.timeout)
                        response.raise_for_status()

                        with open(local_path, 'wb') as f:
                            f.write(response.content)

                        if os.path.getsize(local_path) > 0:
                            success = True
                            break

                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            print(f"Failed to download {url}: {str(e)}")
                        else:
                            time.sleep(2 ** attempt)  # Exponential backoff

                # Update results and progress
                with self.lock:
                    self.results[url] = success
                    self.completed_downloads += 1

                    # Print progress
                    with self.print_lock:
                        print_progress(
                            self.completed_downloads,
                            self.total_downloads,
                            prefix='Current Folder:',
                            suffix=f'({self.completed_downloads}/{self.total_downloads} files)'
                        )

            except Exception as e:
                print(f"Worker error: {str(e)}")
            finally:
                self.download_queue.task_done()

    def download_files(self, file_list: List[tuple], headers: dict) -> Dict[str, bool]:
        """
        Download multiple files concurrently

        Args:
            file_list: List of tuples containing (url, local_path)
            headers: Request headers

        Returns:
            Dictionary mapping URLs to download success status
        """
        # Reset tracking variables
        self.results = {}
        self.completed_downloads = 0
        self.total_downloads = len(file_list)

        print(f"\nStarting download of {self.total_downloads} files")

        # Start worker threads
        threads = []
        for _ in range(self.num_threads):
            thread = threading.Thread(
                target=self.download_worker,
                args=(headers,),
                daemon=True
            )
            thread.start()
            threads.append(thread)

        # Add download tasks to queue
        for url, local_path in file_list:
            self.download_queue.put((url, local_path))

        # Add sentinel values to stop workers
        for _ in range(self.num_threads):
            self.download_queue.put(None)

        # Wait for all tasks to complete
        self.download_queue.join()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        return self.results


def download_folder_threaded(
        folder_url: str,
        local_folder: str,
        required_files: List[str],
        headers: dict,
        num_threads: int = 4
) -> bool:
    """Download all required files from a folder using threads"""
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    try:
        # Get folder contents
        response = requests.get(folder_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Build download list
        download_list = []
        files_found = []

        for csv_link in soup.find_all('a'):
            if csv_link.text in required_files:
                files_found.append(csv_link.text)
                csv_url = urljoin(folder_url, csv_link['href'])
                csv_path = os.path.join(local_folder, csv_link.text)
                download_list.append((csv_url, csv_path))

        # Check if all required files were found
        missing_files = set(required_files) - set(files_found)
        if missing_files:
            print(f"Warning: Missing required files: {missing_files}")
            return False

        # Download files using threaded downloader
        downloader = ThreadedDownloader(num_threads=num_threads)
        results = downloader.download_files(download_list, headers)

        # Check if all downloads were successful
        return all(results.values())

    except Exception as e:
        print(f"Error downloading folder {folder_url}: {str(e)}")
        return False


class ProcessingTracker:
    def __init__(self, base_dir, reset=False):
        self.base_dir = base_dir
        self.tracker_file = Path(base_dir) / "processing_status.json"

        if reset and self.tracker_file.exists():
            self.tracker_file.unlink()

        self.load_status()

    def load_status(self):
        if self.tracker_file.exists():
            with open(self.tracker_file, 'r') as f:
                self.status = json.load(f)
        else:
            self.status = {
                'processed_folders': [],
                'failed_folders': [],
                'last_processed_index': -1
            }
            self.save_status()

    def save_status(self):
        with open(self.tracker_file, 'w') as f:
            json.dump(self.status, f)

    def mark_folder_processed(self, folder_name, index):
        if folder_name not in self.status['processed_folders']:
            self.status['processed_folders'].append(folder_name)
            self.status['last_processed_index'] = index
            self.save_status()

    def mark_folder_failed(self, folder_name):
        if folder_name not in self.status['failed_folders']:
            self.status['failed_folders'].append(folder_name)
            self.save_status()

    def is_folder_processed(self, folder_name):
        return folder_name in self.status['processed_folders']

    def get_next_index(self):
        return self.status['last_processed_index'] + 1


class DataVersionManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.version_file = Path(base_dir) / "version_info.json"
        self.load_version_info()

    def load_version_info(self):
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                self.version_info = json.load(f)
        else:
            self.version_info = {
                'current_version': 1,
                'uploaded_versions': []
            }
            self.save_version_info()

    def save_version_info(self):
        with open(self.version_file, 'w') as f:
            json.dump(self.version_info, f)

    def get_current_version(self):
        return f"v{self.version_info['current_version']}"

    def increment_version(self):
        self.version_info['uploaded_versions'].append(self.version_info['current_version'])
        self.version_info['current_version'] += 1
        self.save_version_info()


def get_folder_urls(base_url: str, headers: dict) -> List[tuple]:
    """Get list of all date folder URLs"""
    try:
        response = requests.get(base_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        folders = []
        date_pattern = re.compile(r'^\d{4}-\d{2}/$')

        for link in soup.find_all('a'):
            if date_pattern.match(link.text):
                folder_url = urljoin(base_url, link['href'])
                folder_name = link.text.strip('/')
                folders.append((folder_name, folder_url))

        return sorted(folders)

    except Exception as e:
        print(f"Error accessing {base_url}: {str(e)}")
        return []


def process_folder_data(folder_path: str) -> pl.DataFrame:
    """Process all files in a folder with immediate cleanup"""
    results = []
    file_processors = {
        'block.csv': process_block_file,
        'cpu.csv': process_cpu_file,
        'mem.csv': process_mem_file,
        'llite.csv': process_nfs_file
    }

    for filename, processor in file_processors.items():
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            try:
                # Process in chunks if file is large
                if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB
                    result = pl.scan_csv(file_path).collect()
                    processed_result = processor(result)
                else:
                    result = processor(file_path)

                if result is not None:
                    results.append(result)

                # Immediately remove the file after processing
                os.remove(file_path)
                gc.collect()
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                try:
                    os.remove(file_path)
                except:
                    pass

    if results:
        return pl.concat(results)
    return None


def check_disk_space(warning_gb=20, critical_gb=5) -> tuple:
    """
    Check available disk space considering user quota
    Returns: (is_safe, is_abundant)
    """
    try:
        # Get user's home directory
        home_dir = os.path.expanduser('~')

        # Get total quota (24GB) and used space
        quota_total = 24 * 1024 * 1024 * 1024  # 24GB in bytes
        used_space = sum(os.path.getsize(os.path.join(dirpath, filename))
                         for dirpath, _, filenames in os.walk(home_dir)
                         for filename in filenames)

        # Calculate available space in GB
        available_gb = (quota_total - used_space) / (1024 * 1024 * 1024)

        return available_gb > critical_gb, available_gb > warning_gb
    except Exception as e:
        print(f"Error checking disk space: {str(e)}")
        # If we can't check space, assume we're in a critical state
        return (False, False)


def save_monthly_data(monthly_data: Dict, base_dir: str, version_manager: DataVersionManager) -> List[str]:
    """Save monthly data to files"""
    output_dir = os.path.join(base_dir, "monthly_data")
    os.makedirs(output_dir, exist_ok=True)
    version_suffix = version_manager.get_current_version()
    saved_files = []

    for month, new_df in monthly_data.items():
        file_path = os.path.join(output_dir, f"FRESCO_Conte_ts_{month}_{version_suffix}.csv")
        if os.path.exists(file_path):
            existing_df = pl.read_csv(file_path)
            merged_df = pl.concat([existing_df, new_df]).unique()
            merged_df.write_csv(file_path)  # 'w' mode is explicit here for clarity
        else:
            new_df.write_csv(file_path)
        saved_files.append(file_path)

    return saved_files


def upload_to_s3(file_paths: List[str], bucket_name="data-transform-conte") -> bool:
    """Upload files to S3"""
    print("\nStarting S3 upload...")
    s3_client = boto3.client('s3')

    for i, file_path in enumerate(file_paths, 1):
        file_name = os.path.basename(file_path)
        for attempt in range(3):
            try:
                s3_client.upload_file(file_path, bucket_name, file_name)
                if i % 2 == 0 or i == len(file_paths):
                    print(f"Uploaded {i}/{len(file_paths)} files")
                break
            except ClientError as e:
                if attempt == 2:
                    print(f"Failed to upload {file_name}: {str(e)}")
                    return False
                time.sleep(2 ** attempt)

    return True


def manage_storage(monthly_data: Dict, base_dir: str, version_manager: DataVersionManager) -> bool:
    """Manage storage and upload data when needed"""
    is_safe, is_abundant = check_disk_space(warning_gb=20, critical_gb=5)

    if not is_safe:
        print("\nCritical disk space reached. Uploading data...")
        version_suffix = version_manager.get_current_version()
        local_files = glob.glob(os.path.join(base_dir, "monthly_data", f"*_{version_suffix}.csv"))

        if upload_to_s3(local_files):
            print(f"Successfully uploaded {version_suffix} files")
            # Clean up local files after successful upload
            for file in local_files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Error removing {file}: {str(e)}")

            version_manager.increment_version()
            monthly_data.clear()
            gc.collect()
            return True
        else:
            print("Upload failed. Will retry when critical.")
            return False

    elif not is_abundant:
        print("\nWarning: Disk space running low")
        # Optionally trigger an upload even when not critical
        if len(monthly_data) > 10:  # If we have accumulated significant data
            return manage_storage(monthly_data, base_dir, version_manager)

    return False


def print_progress(current: int, total: int, prefix: str = '', suffix: str = ''):
    """Print progress as a percentage with a progress bar"""
    percent = (current / total) * 100
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)
    if current == total:
        print()


def main():
    base_dir = os.getcwd()
    base_url = "https://www.datadepot.rcac.purdue.edu/sbagchi/fresco/repository/Conte/TACC_Stats/"
    required_files = ['block.csv', 'cpu.csv', 'mem.csv', 'llite.csv']
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    # Initialize managers
    tracker = ProcessingTracker(base_dir, reset=True)
    version_manager = DataVersionManager(base_dir)
    monthly_data = {}

    # Configure number of download threads based on system capabilities
    num_download_threads = min(os.cpu_count() or 4, 8)  # Use at most 8 threads
    print(f"Using {num_download_threads} download threads")

    try:
        # Get list of folders to process
        folder_urls = get_folder_urls(base_url, headers)
        if not folder_urls:
            print("No folders found to process")
            return

        print(f"Found {len(folder_urls)} folders to process")
        start_idx = tracker.get_next_index()

        # Calculate total folders to process
        total_folders = len(folder_urls) - start_idx
        print(f"\nStarting processing of {total_folders} folders")

        # Process one folder at a time
        for i in range(start_idx, len(folder_urls)):
            # Update overall progress
            current_folder = i - start_idx + 1
            print_progress(current_folder, total_folders, prefix='Overall Progress:', suffix='Complete')
            folder_name, folder_url = folder_urls[i]
            print(f"\nProcessing folder {i + 1}/{len(folder_urls)}: {folder_name}")

            # Skip if already processed
            if tracker.is_folder_processed(folder_name):
                print(f"Folder {folder_name} already processed, skipping...")
                continue

            # Check disk space
            if not check_disk_space()[0]:
                print("Critical disk space reached. Uploading current data...")
                if manage_storage(monthly_data, base_dir, version_manager):
                    continue
                else:
                    break

            # Create temporary folder
            temp_folder = os.path.join(base_dir, folder_name)
            try:
                # Download and process folder using threaded downloader
                if download_folder_threaded(
                        folder_url,
                        temp_folder,
                        required_files,
                        headers,
                        num_threads=num_download_threads
                ):
                    result = process_folder_data(temp_folder)

                    if result is not None:
                        # Split and save results
                        batch_monthly = {
                            name: group for name, group in
                            result.group_by(pl.col('Timestamp').dt.strftime('%Y_%m'))
                        }
                        monthly_data.update(batch_monthly)
                        save_monthly_data(batch_monthly, base_dir, version_manager)

                        # Clear memory
                        del result, batch_monthly
                        gc.collect()

                        tracker.mark_folder_processed(folder_name, i)
                    else:
                        tracker.mark_folder_failed(folder_name)
                else:
                    tracker.mark_folder_failed(folder_name)

            except Exception as e:
                print(f"Error processing folder {folder_name}: {str(e)}")
                tracker.mark_folder_failed(folder_name)

            finally:
                # Clean up temporary folder
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder)
                time.sleep(1)  # Small delay between folders

        # Final data upload
        if monthly_data:
            manage_storage(monthly_data, base_dir, version_manager)

    except Exception as e:
        print(f"Error in main processing loop: {str(e)}")
        if monthly_data:
            try:
                save_monthly_data(monthly_data, base_dir, version_manager)
            except Exception as save_error:
                print(f"Error saving final data: {str(save_error)}")

    print("Processing completed!")


if __name__ == "__main__":
    main()
