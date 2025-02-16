import os
import glob
import json
from pathlib import Path
import gc
import psutil
from typing import Dict
import pandas as pd
import boto3
from botocore.exceptions import ClientError
import shutil
import time


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
    base_dir = "/path/to/fresco/repository/Conte/TACC_Stats"  # Update with actual path

    # Initialize trackers with reset option
    tracker = ProcessingTracker(base_dir, reset=True)
    version_manager = DataVersionManager(base_dir)
    monthly_data: Dict[str, pd.DataFrame] = {}

    # Get all date folders
    date_folders = sorted(glob.glob(os.path.join(base_dir, "????-??")))

    for folder in date_folders:
        folder_name = os.path.basename(folder)
        print(f"\nProcessing folder: {folder_name}")

        if tracker.is_folder_processed(folder_name):
            print(f"Folder {folder_name} already processed, skipping...")
            continue

        # Check storage situation
        manage_storage_and_upload(monthly_data, base_dir, version_manager)

        try:
            result = process_conte_folder(folder)

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
