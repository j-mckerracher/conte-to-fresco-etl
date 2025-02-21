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
import pandas as pd
import numpy as np


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


def process_block_file(file_path: str) -> pd.DataFrame:
    """Process block.csv file with improved error handling"""
    df = pd.read_csv(file_path)

    # Calculate I/O throughput with safety checks
    total_sectors = df['rd_sectors'] + df['wr_sectors']
    total_ticks = df['rd_ticks'] + df['wr_ticks']

    # Convert sectors to bytes and calculate throughput
    bytes_processed = total_sectors * 512
    throughput = [safe_division(b, t) for b, t in zip(bytes_processed, total_ticks)]

    # Convert to GB/s and validate
    df['Value'] = [validate_metric(t / (1024 * 1024 * 1024)) for t in throughput]
    df['jobID'] = df['jobID'].str.replace('jobID', 'JOB', case=False)

    return pd.DataFrame({
        'Job Id': df['jobID'],
        'Host': df['node'],
        'Event': 'block',
        'Value': df['Value'],
        'Units': 'GB/s',
        'Timestamp': pd.to_datetime(df['timestamp'])
    })


def process_cpu_file(file_path: str) -> pd.DataFrame:
    """Process cpu.csv file with support for multi-core CPU percentages"""
    df = pd.read_csv(file_path)

    # Calculate total CPU time with all components
    total = (df['user'] + df['nice'] + df['system'] +
             df['idle'] + df['iowait'] + df['irq'] + df['softirq'])

    # Calculate user CPU percentage without upper bound
    # Including both user and nice time as per original
    user_time = df['user'] + df['nice']
    df['Value'] = [validate_metric(safe_division(u, t) * 100, 0)
                   for u, t in zip(user_time, total)]

    df['jobID'] = df['jobID'].str.replace('jobID', 'JOB', case=False)

    return pd.DataFrame({
        'Job Id': df['jobID'],
        'Host': df['node'],
        'Event': 'cpuuser',
        'Value': df['Value'],
        'Units': 'CPU %',
        'Timestamp': pd.to_datetime(df['timestamp'])
    })


def process_mem_file(file_path: str) -> pd.DataFrame:
    """Process mem.csv file with improved validation"""
    df = pd.read_csv(file_path)

    # Ensure memory values are non-negative
    df['MemTotal'] = df['MemTotal'].clip(lower=0)
    df['MemFree'] = df['MemFree'].clip(lower=0)
    df['FilePages'] = df['FilePages'].clip(lower=0)

    # Ensure MemFree doesn't exceed MemTotal
    df['MemFree'] = df.apply(lambda row: min(row['MemFree'], row['MemTotal']), axis=1)

    # Calculate memory usage in GB with validation
    df['memused_value'] = ((df['MemTotal'] - df['MemFree']) /
                           (1024 * 1024 * 1024)).clip(lower=0)

    # Calculate memory usage minus disk cache
    cache_adjusted = df['MemTotal'] - df['MemFree'] - df['FilePages']
    df['memused_minus_diskcache_value'] = (validate_metric(cache_adjusted) /
                                           (1024 * 1024 * 1024))

    df['jobID'] = df['jobID'].str.replace('jobID', 'JOB', case=False)

    # Create separate dataframes for each metric
    memused = pd.DataFrame({
        'Job Id': df['jobID'],
        'Host': df['node'],
        'Event': 'memused',
        'Value': df['memused_value'],
        'Units': 'GB',
        'Timestamp': pd.to_datetime(df['timestamp'])
    })

    memused_minus_diskcache = pd.DataFrame({
        'Job Id': df['jobID'],
        'Host': df['node'],
        'Event': 'memused_minus_diskcache',
        'Value': df['memused_minus_diskcache_value'],
        'Units': 'GB',
        'Timestamp': pd.to_datetime(df['timestamp'])
    })

    return pd.concat([memused, memused_minus_diskcache])


def process_nfs_file(file_path: str) -> pd.DataFrame:
    """Process llite.csv file with improved rate calculation"""
    df = pd.read_csv(file_path)

    # Sort by timestamp to ensure correct rate calculation
    df = df.sort_values('timestamp')

    # Calculate time deltas in seconds
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    time_deltas = df.groupby(['jobID', 'node'])['timestamp'].diff().dt.total_seconds()

    # Calculate rates with proper time normalization
    total_bytes = df['read_bytes'] + df['write_bytes']
    byte_deltas = total_bytes.groupby([df['jobID'], df['node']]).diff()

    # Calculate MB/s with validation
    df['Value'] = [validate_metric(calculate_rate(bytes, prev_bytes, delta) / (1024 * 1024))
                   for bytes, prev_bytes, delta in
                   zip(total_bytes, byte_deltas, time_deltas)]

    df['jobID'] = df['jobID'].str.replace('jobID', 'JOB', case=False)

    return pd.DataFrame({
        'Job Id': df['jobID'],
        'Host': df['node'],
        'Event': 'nfs',
        'Value': df['Value'],
        'Units': 'MB/s',
        'Timestamp': df['timestamp']
    })

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


def process_folder_data(folder_path: str) -> pd.DataFrame:
    """Process all files in a folder"""
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
                result = processor(file_path)
                if result is not None:
                    results.append(result)
                os.remove(file_path)  # Delete file after processing
                gc.collect()
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    if results:
        return pd.concat(results, ignore_index=True)
    return None


def check_disk_space(warning_gb=50, critical_gb=20) -> tuple:
    """Check available disk space"""
    disk_usage = psutil.disk_usage('/')
    available_gb = 23.0
    return (available_gb > critical_gb, available_gb > warning_gb)


def save_monthly_data(monthly_data: Dict, base_dir: str, version_manager: DataVersionManager) -> List[str]:
    """Save monthly data to files"""
    output_dir = os.path.join(base_dir, "monthly_data")
    os.makedirs(output_dir, exist_ok=True)
    version_suffix = version_manager.get_current_version()
    saved_files = []

    for month, new_df in monthly_data.items():
        file_path = os.path.join(output_dir, f"FRESCO_Conte_ts_{month}_{version_suffix}.csv")
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            merged_df = pd.concat([existing_df, new_df]).drop_duplicates()
            merged_df.to_csv(file_path, index=False)
        else:
            new_df.to_csv(file_path, index=False)
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
    is_safe, is_abundant = check_disk_space()
    if not is_safe:
        print("\nCritical disk space reached. Uploading data...")
        version_suffix = version_manager.get_current_version()
        local_files = glob.glob(os.path.join(base_dir, "monthly_data", f"*_{version_suffix}.csv"))

        if upload_to_s3(local_files):
            print(f"Successfully uploaded {version_suffix} files")
            version_manager.increment_version()
            monthly_data.clear()
            gc.collect()
            return True
        else:
            print("Upload failed. Will retry when critical.")
            return False

    elif not is_abundant:
        print("\nWarning: Disk space running low")

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
                            result.groupby(result['Timestamp'].dt.strftime('%Y_%m'))
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
