import os
import glob
import json
import re
from pathlib import Path
import gc
import psutil
from typing import Dict, List
import pandas as pd
import boto3
from botocore.exceptions import ClientError
import shutil
import time
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess


class DownloadWorker:
    def __init__(self, max_workers=3):
        """
        Initialize download worker with thread pool
        Using 3 workers by default to avoid overloading the system
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_downloads = Queue()
        self.results = []

    def download_file(self, url, local_path, headers, max_retries=3, retry_delay=5):
        """Download file with retry mechanism"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=300)
                response.raise_for_status()

                with open(local_path, 'wb') as f:
                    f.write(response.content)
                return True, local_path

            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))

        return False, local_path

    def download_folder_files(self, folder_url: str, local_folder: str, required_files: List[str], headers: dict):
        """Download all required files from a folder"""
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)

        try:
            response = requests.get(folder_url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            futures = []
            files_found = []

            for csv_link in soup.find_all('a'):
                if csv_link.text in required_files:
                    files_found.append(csv_link.text)
                    csv_url = urljoin(folder_url, csv_link['href'])
                    csv_path = os.path.join(local_folder, csv_link.text)

                    print(f"Queuing download for {csv_link.text} from {csv_url}")
                    future = self.executor.submit(
                        self.download_file,
                        csv_url,
                        csv_path,
                        headers
                    )
                    futures.append(future)

            # Wait for all downloads to complete
            success = True
            for future in as_completed(futures):
                result, path = future.result()
                if not result:
                    success = False
                    print(f"Failed to download {path}")
                    break

            # Check if all required files were found and downloaded
            missing_files = set(required_files) - set(files_found)
            if missing_files:
                print(f"Warning: Could not find the following required files: {missing_files}")
                success = False

            return success

        except Exception as e:
            print(f"Error downloading folder {folder_url}: {str(e)}")
            return False

    def shutdown(self):
        """Shutdown the thread pool executor"""
        self.executor.shutdown(wait=True)


def get_base_dir():
    """Get the current working directory using pwd"""
    try:
        result = subprocess.run(['pwd'], capture_output=True, text=True)
        base_dir = result.stdout.strip()
        return base_dir
    except Exception as e:
        print(f"Error getting base directory: {str(e)}")
        return os.getcwd()  # Fallback to os.getcwd()


def download_file(url, local_path, max_retries=3, retry_delay=5):
    """Download file with retry mechanism"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=300)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                f.write(response.content)
            return True

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

    return False


def download_date_folder(folder_url, local_folder, required_files):
    """
    Downloads all required CSV files for a single date folder.

    Args:
        folder_url: URL of the date folder
        local_folder: Local path to save files
        required_files: List of required CSV files (e.g., ['block.csv', 'cpu.csv'])

    Returns:
        bool: True if all files were downloaded successfully, False otherwise
    """
    try:
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)

        # Add headers to avoid potential 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Get the CSV files in the folder
        response = requests.get(folder_url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        download_success = True
        files_found = []

        for csv_link in soup.find_all('a'):
            if csv_link.text in required_files:
                files_found.append(csv_link.text)
                csv_url = urljoin(folder_url, csv_link['href'])
                csv_path = os.path.join(local_folder, csv_link.text)

                print(f"Downloading {csv_link.text} from {csv_url}")

                # Use the same headers for file download
                response = requests.get(csv_url, headers=headers, timeout=300)
                response.raise_for_status()

                with open(csv_path, 'wb') as f:
                    f.write(response.content)

                if os.path.getsize(csv_path) == 0:
                    print(f"Warning: Downloaded file {csv_link.text} is empty")
                    download_success = False
                    break

                time.sleep(0.5)  # Small delay between files

        # Check if all required files were found
        missing_files = set(required_files) - set(files_found)
        if missing_files:
            print(f"Warning: Could not find the following required files: {missing_files}")
            download_success = False

        return download_success

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error downloading folder {folder_url}: {str(e)}")
        print(f"Response status code: {e.response.status_code}")
        print(f"Response headers: {e.response.headers}")
        return False
    except Exception as e:
        print(f"Error downloading folder {folder_url}: {str(e)}")
        print(f"Response status code: {response.status_code if 'response' in locals() else 'N/A'}")
        print(f"Response content: {response.text[:500] if 'response' in locals() else 'N/A'}")
        return False


def get_date_folders(base_url):
    """
    Get list of all date folder URLs.

    Args:
        base_url: Base URL to fetch the date folders from

    Returns:
        List of tuples containing (folder_name, folder_url)
    """
    try:
        # Add headers to avoid potential 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(base_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all folder links matching YYYY-MM pattern
        date_folders = []
        date_pattern = re.compile(r'^\d{4}-\d{2}/$')

        for link in soup.find_all('a'):
            if date_pattern.match(link.text):
                folder_url = urljoin(base_url, link['href'])
                folder_name = link.text.strip('/')
                date_folders.append((folder_name, folder_url))

        return sorted(date_folders)  # Sort by date

    except Exception as e:
        print(f"Error accessing {base_url}: {str(e)}")
        # Add more detailed error information
        print(f"Response status code: {response.status_code if 'response' in locals() else 'N/A'}")
        print(f"Response content: {response.text[:500] if 'response' in locals() else 'N/A'}")
        return []


def download_conte_data(base_url, base_dir, required_files):
    """
    Download all Conte data folders using thread pool
    Returns: List of successfully downloaded folder names
    """
    downloaded_folders = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Get all date folders
    date_folders = get_date_folders(base_url)
    total_folders = len(date_folders)

    print(f"Found {total_folders} date folders to process")
    if total_folders == 0:
        print("No date folders found. URL content preview:")
        try:
            response = requests.get(base_url, headers=headers)
            print(response.text[:1000])
        except Exception as e:
            print(f"Could not fetch URL content: {str(e)}")
        return downloaded_folders

    # Initialize download worker with 3 threads (conservative for old laptop)
    downloader = DownloadWorker(max_workers=3)

    try:
        for i, (folder_name, folder_url) in enumerate(date_folders, 1):
            print(f"\nProcessing folder {i}/{total_folders}: {folder_name}")
            print(f"Folder URL: {folder_url}")

            local_folder = os.path.join(base_dir, folder_name)

            if downloader.download_folder_files(folder_url, local_folder, required_files, headers):
                print(f"Successfully downloaded {folder_name}")
                downloaded_folders.append(folder_name)
            else:
                print(f"Failed to download {folder_name}")

            # Check disk space after each folder
            if not check_critical_disk_space()[0]:
                print("Critical disk space reached. Stopping downloads.")
                break

            # Small delay between folders to avoid overwhelming the server
            time.sleep(1)

    finally:
        downloader.shutdown()

    return downloaded_folders


class DataVersionManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.version_file = Path(base_dir) / "version_info.json"
        self.load_version_info()

    def load_version_info(self):
        """Load or initialize version tracking data"""
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
                'failed_folders': []
            }

    def save_status(self):
        with open(self.tracker_file, 'w') as f:
            json.dump(self.status, f)

    def is_folder_processed(self, folder_name):
        return folder_name in self.status['processed_folders']

    def mark_folder_processed(self, folder_name):
        if folder_name not in self.status['processed_folders']:
            self.status['processed_folders'].append(folder_name)
            self.save_status()

    def mark_folder_failed(self, folder_name):
        if folder_name not in self.status['failed_folders']:
            self.status['failed_folders'].append(folder_name)
            self.save_status()


def process_block_file(file_path):
    """Process block.csv file and transform to FRESCO format"""
    try:
        df = pd.read_csv(file_path)

        # Calculate value in GB/s (sectors are 512 bytes)
        # Calculate bytes per second from sectors and ticks
        df['Value'] = ((df['rd_sectors'] + df['wr_sectors']) * 512) / (df['rd_ticks'] + df['wr_ticks'])
        # Convert to GB/s
        df['Value'] = df['Value'] / (1024 * 1024 * 1024)

        # Ensure Job Id uses uppercase "JOB" and remove "ID" if present
        df['jobID'] = df['jobID'].str.replace('jobID', 'JOB', case=False)

        result = pd.DataFrame({
            'Job Id': df['jobID'],
            'Host': df['node'],
            'Event': 'block',
            'Value': df['Value'],
            'Units': 'GB/s',
            'Timestamp': pd.to_datetime(df['timestamp'])
        })

        return result
    except Exception as e:
        print(f"Error processing block file {file_path}: {str(e)}")
        return None


def process_cpu_file(file_path):
    """Process cpu.csv file and transform to FRESCO format"""
    try:
        df = pd.read_csv(file_path)

        # Calculate CPU usage percentage
        total = df['user'] + df['nice'] + df['system'] + df['idle'] + df['iowait'] + df['irq'] + df['softirq']
        df['Value'] = ((df['user'] + df['nice']) / total) * 100

        # Ensure Job Id uses uppercase "JOB" and remove "ID" if present
        df['jobID'] = df['jobID'].str.replace('jobID', 'JOB', case=False)

        result = pd.DataFrame({
            'Job Id': df['jobID'],
            'Host': df['node'],
            'Event': 'cpuuser',
            'Value': df['Value'],
            'Units': 'CPU %',
            'Timestamp': pd.to_datetime(df['timestamp'])
        })

        return result
    except Exception as e:
        print(f"Error processing cpu file {file_path}: {str(e)}")
        return None


def process_mem_file(file_path):
    """Process mem.csv file and transform to FRESCO format"""
    try:
        df = pd.read_csv(file_path)

        # Calculate total memory used in GB
        df['memused_value'] = (df['MemTotal'] - df['MemFree']) / (1024 * 1024 * 1024)

        # Calculate memory used minus disk cache in GB
        df['memused_minus_diskcache_value'] = (df['MemTotal'] - df['MemFree'] - df['FilePages']) / (1024 * 1024 * 1024)

        # Ensure Job Id uses uppercase "JOB" and remove "ID" if present
        df['jobID'] = df['jobID'].str.replace('jobID', 'JOB', case=False)

        # Create separate dataframes for each memory metric
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
    except Exception as e:
        print(f"Error processing memory file {file_path}: {str(e)}")
        return None


def process_nfs_file(file_path):
    """Process llite.csv file (which contains NFS data) and transform to FRESCO format"""
    try:
        df = pd.read_csv(file_path)

        # Calculate NFS throughput in MB/s using read and write bytes
        df['Value'] = (df['read_bytes'] + df['write_bytes']) / (1024 * 1024)  # Convert to MB/s

        # Ensure Job Id uses uppercase "JOB" and remove "ID" if present
        df['jobID'] = df['jobID'].str.replace('jobID', 'JOB', case=False)

        result = pd.DataFrame({
            'Job Id': df['jobID'],
            'Host': df['node'],
            'Event': 'nfs',
            'Value': df['Value'],
            'Units': 'MB/s',
            'Timestamp': pd.to_datetime(df['timestamp'])
        })

        return result
    except Exception as e:
        print(f"Error processing NFS file {file_path}: {str(e)}")
        return None


def check_critical_disk_space(warning_gb=50, critical_gb=20):
    """Check disk space on Mac"""
    disk_usage = psutil.disk_usage('/')  # Use root directory for Mac
    available_gb = disk_usage.free / (1024 ** 3)

    return (
        available_gb > critical_gb,  # is_safe
        available_gb > warning_gb  # is_abundant
    )


def save_monthly_data_locally(monthly_data, base_dir, version_manager):
    """
    Save monthly data to local files, updating existing files if they exist
    Returns list of saved file paths
    """
    output_dir = os.path.join(base_dir, "monthly_data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    version_suffix = version_manager.get_current_version()
    saved_files = []

    for month, new_df in monthly_data.items():
        file_path = os.path.join(
            output_dir,
            f"FRESCO_Conte_ts_{month}_{version_suffix}.csv"
        )

        if os.path.exists(file_path):
            # Read existing file and merge with new data
            existing_df = pd.read_csv(file_path)
            merged_df = pd.concat([existing_df, new_df]).drop_duplicates()
            merged_df.to_csv(file_path, index=False)
        else:
            # Create new file
            new_df.to_csv(file_path, index=False)

        saved_files.append(file_path)

    return saved_files


def upload_to_s3(file_paths, bucket_name="data-transform-conte", max_retries=3):
    """
    Upload files to S3 bucket with retry logic.
    Returns True if all uploads successful, False otherwise.
    """
    print("\nStarting S3 upload...")
    s3_client = boto3.client('s3')

    total_files = len(file_paths)
    for i, file_path in enumerate(file_paths, 1):
        file_name = os.path.basename(file_path)

        for attempt in range(max_retries):
            try:
                s3_client.upload_file(file_path, bucket_name, file_name)
                if i % 2 == 0 or i == total_files:  # Progress update every 2 files
                    print(f"Uploaded {i}/{total_files} files to S3")
                break
            except ClientError as e:
                if attempt == max_retries - 1:
                    print(f"Failed to upload {file_name} after {max_retries} attempts")
                    print(f"Error: {str(e)}")
                    return False
                time.sleep(2 ** attempt)  # Exponential backoff

    return True


def manage_storage_and_upload(monthly_data, base_dir, version_manager):
    is_safe, is_abundant = check_critical_disk_space()

    if not is_safe:
        print("\nCritical disk space reached. Initiating upload process...")
        version_suffix = version_manager.get_current_version()
        local_files = glob.glob(
            os.path.join(base_dir, "monthly_data", f"*_{version_suffix}.csv")
        )

        if upload_to_s3(local_files, bucket_name="data-transform-conte"):
            print(f"Successfully uploaded {version_suffix} files to S3")
            version_manager.increment_version()
            monthly_data.clear()
            gc.collect()
            return True
        else:
            print("Failed to upload to S3. Will retry when disk space is critical again.")
            return False

    elif not is_abundant:
        print("\nWarning: Disk space is running low")

    return False


def process_conte_folder(folder_path):
    """Process all files in a Conte date folder"""
    results = []

    # Process each file type
    file_processors = {
        'block.csv': process_block_file,
        'cpu.csv': process_cpu_file,
        'mem.csv': process_mem_file,
        'llite.csv': process_nfs_file  # NFS data in llite.csv
    }

    for filename, processor in file_processors.items():
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            try:
                result = processor(file_path)
                if result is not None:
                    results.append(result)
                # Clear memory after each file
                gc.collect()
            except Exception as e:
                print(f"Error processing {filename} in {folder_path}: {str(e)}")
                continue

    if results:
        return pd.concat(results, ignore_index=True)
    return None


def split_by_month(df):
    """
    Split DataFrame into monthly groups.
    Returns a dictionary with year_month as key and DataFrame as value.
    """
    monthly_data = {}
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Group by year and month
    for name, group in df.groupby(df['Timestamp'].dt.strftime('%Y_%m')):
        monthly_data[name] = group

    return monthly_data


def update_monthly_data(existing_data, new_data):
    """
    Update existing monthly data with new data.
    Both parameters are dictionaries with year_month keys and DataFrames as values.
    """
    for month, new_df in new_data.items():
        if month in existing_data:
            # Concatenate and remove duplicates
            existing_data[month] = pd.concat([existing_data[month], new_df]).drop_duplicates()
        else:
            existing_data[month] = new_df
    return existing_data


def main():
    base_dir = get_base_dir()
    base_url = "https://www.datadepot.rcac.purdue.edu/sbagchi/fresco/repository/Conte/TACC_Stats/"

    print(f"Base directory: {base_dir}")
    print(f"Base URL: {base_url}")

    # Test URL accessibility
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(base_url, headers=headers, timeout=30)
        print(f"Initial URL test - Status code: {response.status_code}")
        print(f"Response content preview: {response.text[:500]}")
    except Exception as e:
        print(f"Error testing base URL: {str(e)}")
        print("Please verify the URL is accessible and doesn't require authentication")
        return

    # Files we need from each folder
    required_files = ['block.csv', 'cpu.csv', 'mem.csv', 'llite.csv']

    # Initialize trackers with reset option
    tracker = ProcessingTracker(base_dir, reset=True)
    version_manager = DataVersionManager(base_dir)
    monthly_data: Dict[str, pd.DataFrame] = {}

    print("\nStarting Conte data download and processing...")

    # Download data with threading
    downloaded_folders = download_conte_data(base_url, base_dir, required_files)
    print(f"\nDownloaded folders: {downloaded_folders}")

    if not downloaded_folders:
        print("No folders were downloaded. Check the URL and network connection.")
        return

    # Process downloaded folders
    for folder_name in downloaded_folders:
        print(f"\nProcessing folder: {folder_name}")

        if tracker.is_folder_processed(folder_name):
            print(f"Folder {folder_name} already processed, skipping...")
            continue

        # Check storage situation
        manage_storage_and_upload(monthly_data, base_dir, version_manager)

        try:
            folder_path = os.path.join(base_dir, folder_name)
            result = process_conte_folder(folder_path)

            if result is not None:
                # Split by month and update storage
                batch_monthly = split_by_month(result)
                monthly_data = update_monthly_data(monthly_data, batch_monthly)

                # Save to local storage
                save_monthly_data_locally(batch_monthly, base_dir, version_manager)

                # Clear memory
                del result
                del batch_monthly
                gc.collect()

                tracker.mark_folder_processed(folder_name)
            else:
                print(f"No valid data found in folder {folder_name}")
                tracker.mark_folder_failed(folder_name)

        except Exception as e:
            print(f"Error processing folder {folder_name}: {str(e)}")
            tracker.mark_folder_failed(folder_name)
            continue

    # Save any remaining data
    if monthly_data:
        save_monthly_data_locally(monthly_data, base_dir, version_manager)

    print("Processing completed!")


def test_conte_processing():
    """Test function for Conte data processing"""
    test_dir = "/tmp/conte_test"  # Use /tmp for Mac
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Create test folder structure
    test_date_folder = os.path.join(test_dir, "2016-06")
    os.makedirs(test_date_folder, exist_ok=True)

    # Create sample test data
    test_data = {
        'block.csv': pd.DataFrame({
            'jobID': ['job1', 'job2'],
            'node': ['node1', 'node2'],
            'rd_sectors': [1000, 2000],
            'wr_sectors': [500, 1000],
            'rd_ticks': [100, 200],
            'wr_ticks': [50, 100],
            'timestamp': ['2016-06-01 10:00:00', '2016-06-01 11:00:00']
        }),
        'cpu.csv': pd.DataFrame({
            'jobID': ['job1', 'job2'],
            'node': ['node1', 'node2'],
            'user': [80, 70],
            'nice': [0, 0],
            'system': [10, 20],
            'idle': [10, 10],
            'iowait': [0, 0],
            'irq': [0, 0],
            'softirq': [0, 0],
            'timestamp': ['2016-06-01 10:00:00', '2016-06-01 11:00:00']
        })
    }

    # Save test data
    for filename, df in test_data.items():
        df.to_csv(os.path.join(test_date_folder, filename), index=False)

    # Initialize managers
    version_manager = DataVersionManager(test_dir)
    print(f"Initial version: {version_manager.get_current_version()}")

    # Process test data
    result = process_conte_folder(test_date_folder)

    if result is not None:
        monthly_data = split_by_month(result)
        saved_files = save_monthly_data_locally(monthly_data, test_dir, version_manager)

        print("\nVerifying saved files...")
        for file_path in saved_files:
            if os.path.exists(file_path):
                print(f"File exists: {file_path}")
                print(f"File size: {os.path.getsize(file_path)} bytes")

                # Read and verify content
                df = pd.read_csv(file_path)
                print(f"Number of rows: {len(df)}")
                print(f"Columns: {df.columns.tolist()}")

    # Clean up test directory
    shutil.rmtree(test_dir)

    return "Test completed successfully"


if __name__ == "__main__":
    main()
    # test_conte_processing()
