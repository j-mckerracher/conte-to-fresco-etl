import json
import re
from pathlib import Path
import boto3
import shutil
import threading
from queue import Queue
import time
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import os
import gc
from typing import Optional, Dict, Callable, Union, List


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


def process_block_file(input_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Process block.csv file with improved error handling"""
    # Handle both file path and DataFrame inputs
    if isinstance(input_data, str):
        try:
            df = read_csv_with_fallback_encoding(input_data)
        except Exception as e:
            print(f"Error reading block file: {str(e)}")
            raise
    else:
        df = input_data.copy()

    # Drop rows with missing required columns
    required_columns = ['rd_sectors', 'wr_sectors', 'rd_ticks', 'wr_ticks', 'jobID', 'node', 'timestamp']
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return pd.DataFrame()  # Return empty dataframe

    # Drop rows with NaN in required columns
    df = df.dropna(subset=required_columns)

    if len(df) == 0:
        print("No valid data rows after filtering")
        return pd.DataFrame()

    # Calculate I/O throughput with safety checks
    total_sectors = df['rd_sectors'] + df['wr_sectors']
    total_ticks = df['rd_ticks'] + df['wr_ticks']

    # Convert sectors to bytes and calculate throughput
    bytes_processed = total_sectors * 512
    throughput = [safe_division(b, t) for b, t in zip(bytes_processed, total_ticks)]

    # Convert to GB/s and validate
    df['Value'] = [validate_metric(t / (1024 * 1024 * 1024)) for t in throughput]

    # Handle jobID safely
    if 'jobID' in df.columns and df['jobID'].dtype == object:
        df['jobID'] = df['jobID'].fillna('unknown')
        df['jobID'] = df['jobID'].astype(str).str.replace('jobID', 'JOB', case=False)

    # Parse timestamps safely
    try:
        timestamps = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
        # Drop rows with invalid timestamps
        valid_mask = ~timestamps.isna()
        if valid_mask.sum() < len(df):
            print(f"Dropped {len(df) - valid_mask.sum()} rows with invalid timestamps")
            df = df[valid_mask]
            timestamps = timestamps[valid_mask]
    except Exception as e:
        print(f"Error parsing timestamps: {str(e)}")
        return pd.DataFrame()

    if len(df) == 0:
        print("No valid data rows after timestamp filtering")
        return pd.DataFrame()

    return pd.DataFrame({
        'Job Id': df['jobID'],
        'Host': df['node'],
        'Event': 'block',
        'Value': df['Value'],
        'Units': 'GB/s',
        'Timestamp': timestamps
    })


def process_cpu_file(input_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Process cpu.csv file with support for multi-core CPU percentages"""
    # Handle both file path and DataFrame inputs
    if isinstance(input_data, str):
        try:
            df = read_csv_with_fallback_encoding(input_data)
        except Exception as e:
            print(f"Error reading CPU file: {str(e)}")
            raise
    else:
        df = input_data.copy()

    # Drop rows with missing required columns
    required_columns = ['user', 'nice', 'system', 'idle', 'iowait', 'irq', 'softirq', 'jobID', 'node', 'timestamp']
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return pd.DataFrame()  # Return empty dataframe

    # Drop rows with NaN in required columns
    df = df.dropna(subset=required_columns)

    if len(df) == 0:
        print("No valid data rows after filtering")
        return pd.DataFrame()

    # Calculate total CPU time with all components
    total = (df['user'] + df['nice'] + df['system'] +
             df['idle'] + df['iowait'] + df['irq'] + df['softirq'])

    # Calculate user CPU percentage without upper bound
    user_time = df['user'] + df['nice']
    df['Value'] = [validate_metric(safe_division(u, t) * 100, 0)
                   for u, t in zip(user_time, total)]

    # Handle jobID safely
    if 'jobID' in df.columns and df['jobID'].dtype == object:
        df['jobID'] = df['jobID'].fillna('unknown')
        df['jobID'] = df['jobID'].astype(str).str.replace('jobID', 'JOB', case=False)

    # Parse timestamps safely
    try:
        timestamps = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
        # Drop rows with invalid timestamps
        valid_mask = ~timestamps.isna()
        if valid_mask.sum() < len(df):
            print(f"Dropped {len(df) - valid_mask.sum()} rows with invalid timestamps")
            df = df[valid_mask]
            timestamps = timestamps[valid_mask]
    except Exception as e:
        print(f"Error parsing timestamps: {str(e)}")
        return pd.DataFrame()

    if len(df) == 0:
        print("No valid data rows after timestamp filtering")
        return pd.DataFrame()

    return pd.DataFrame({
        'Job Id': df['jobID'],
        'Host': df['node'],
        'Event': 'cpuuser',
        'Value': df['Value'],
        'Units': 'CPU %',
        'Timestamp': timestamps
    })


def process_mem_file(input_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Process mem.csv file with improved validation"""
    # Handle both file path and DataFrame inputs
    if isinstance(input_data, str):
        try:
            df = read_csv_with_fallback_encoding(input_data)
        except Exception as e:
            print(f"Error reading memory file: {str(e)}")
            raise
    else:
        df = input_data.copy()

    # Drop rows with missing required columns
    required_columns = ['MemTotal', 'MemFree', 'FilePages', 'jobID', 'node', 'timestamp']
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return pd.DataFrame()  # Return empty dataframe

    # Drop rows with NaN in required columns
    df = df.dropna(subset=required_columns)

    if len(df) == 0:
        print("No valid data rows after filtering")
        return pd.DataFrame()

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

    # Handle jobID safely
    if 'jobID' in df.columns and df['jobID'].dtype == object:
        df['jobID'] = df['jobID'].fillna('unknown')
        df['jobID'] = df['jobID'].astype(str).str.replace('jobID', 'JOB', case=False)

    # Parse timestamps safely
    try:
        timestamps = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
        # Drop rows with invalid timestamps
        valid_mask = ~timestamps.isna()
        if valid_mask.sum() < len(df):
            print(f"Dropped {len(df) - valid_mask.sum()} rows with invalid timestamps")
            df = df[valid_mask]
            timestamps = timestamps[valid_mask]
    except Exception as e:
        print(f"Error parsing timestamps: {str(e)}")
        return pd.DataFrame()

    if len(df) == 0:
        print("No valid data rows after timestamp filtering")
        return pd.DataFrame()

    # Create separate dataframes for each metric
    memused = pd.DataFrame({
        'Job Id': df['jobID'],
        'Host': df['node'],
        'Event': 'memused',
        'Value': df['memused_value'],
        'Units': 'GB',
        'Timestamp': timestamps
    })

    memused_minus_diskcache = pd.DataFrame({
        'Job Id': df['jobID'],
        'Host': df['node'],
        'Event': 'memused_minus_diskcache',
        'Value': df['memused_minus_diskcache_value'],
        'Units': 'GB',
        'Timestamp': timestamps
    })

    return pd.concat([memused, memused_minus_diskcache])


def process_nfs_file(input_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Process llite.csv file with improved rate calculation"""
    # Handle both file path and DataFrame inputs
    if isinstance(input_data, str):
        try:
            df = read_csv_with_fallback_encoding(input_data)
        except Exception as e:
            print(f"Error reading NFS file: {str(e)}")
            raise
    else:
        df = input_data.copy()

    # Drop rows with missing required columns
    required_columns = ['read_bytes', 'write_bytes', 'jobID', 'node', 'timestamp']
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return pd.DataFrame()  # Return empty dataframe

    # Drop rows with NaN in required columns
    df = df.dropna(subset=required_columns)

    if len(df) == 0:
        print("No valid data rows after filtering")
        return pd.DataFrame()

    # Parse timestamp safely and sort
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
        # Drop rows with invalid timestamps
        valid_mask = ~df['timestamp'].isna()
        if valid_mask.sum() < len(df):
            print(f"Dropped {len(df) - valid_mask.sum()} rows with invalid timestamps")
            df = df[valid_mask]

        if len(df) == 0:
            print("No valid data rows after timestamp filtering")
            return pd.DataFrame()

        df = df.sort_values('timestamp')
    except Exception as e:
        print(f"Error parsing timestamps: {str(e)}")
        return pd.DataFrame()

    # Handle jobID safely
    if 'jobID' in df.columns and df['jobID'].dtype == object:
        df['jobID'] = df['jobID'].fillna('unknown')
        df['jobID'] = df['jobID'].astype(str).str.replace('jobID', 'JOB', case=False)

    # Calculate time deltas in seconds - handle NaN values
    time_deltas = df.groupby(['jobID', 'node'])['timestamp'].diff().dt.total_seconds()
    time_deltas = time_deltas.fillna(0)

    # Calculate rates with proper time normalization
    total_bytes = df['read_bytes'] + df['write_bytes']
    byte_deltas = total_bytes.groupby([df['jobID'], df['node']]).diff().fillna(0)

    # Calculate MB/s with validation and handle division by zero
    df['Value'] = [validate_metric(calculate_rate(bytes, prev_bytes, max(0.1, delta)) / (1024 * 1024))
                   for bytes, prev_bytes, delta in
                   zip(total_bytes, byte_deltas, time_deltas)]

    return pd.DataFrame({
        'Job Id': df['jobID'],
        'Host': df['node'],
        'Event': 'nfs',
        'Value': df['Value'],
        'Units': 'MB/s',
        'Timestamp': df['timestamp']
    })


def read_csv_with_fallback_encoding(file_path, chunk_size=None, **kwargs):
    """
    Attempt to read a CSV file with multiple encodings, falling back if one fails.

    Args:
        file_path: Path to the CSV file
        chunk_size: If not None, read file in chunks of this size
        **kwargs: Additional arguments to pass to pd.read_csv

    Returns:
        DataFrame or TextFileReader if using chunks
    """
    # Try these encodings in order - latin1 rarely fails as it can read any byte
    encodings = ['latin1', 'ISO-8859-1', 'utf-8']

    # For the last attempt with utf-8, we'll use error handling
    errors = [None, None, 'replace']

    last_exception = None

    for i, encoding in enumerate(encodings):
        try:
            error_param = errors[i]
            encoding_kwargs = {'encoding': encoding}

            # Add error handling if specified
            if error_param:
                encoding_kwargs['encoding_errors'] = error_param

            # Add error_bad_lines=False to skip bad lines (parameter name depends on pandas version)
            try:
                # For newer pandas versions (1.3+)
                combined_kwargs = {**encoding_kwargs, 'on_bad_lines': 'skip', **kwargs}
            except TypeError:
                # For older pandas versions
                combined_kwargs = {**encoding_kwargs, 'error_bad_lines': False, **kwargs}

            if chunk_size is not None:
                return pd.read_csv(file_path, chunksize=chunk_size, **combined_kwargs)
            else:
                return pd.read_csv(file_path, **combined_kwargs)

        except UnicodeDecodeError as e:
            last_exception = e
            print(f"Encoding {encoding} failed, trying next encoding...")
            continue
        except Exception as e:
            # If it's not an encoding error, re-raise
            print(f"Error reading file with {encoding} encoding: {str(e)}")
            raise

    # If we get here, all encodings failed
    raise ValueError(f"Failed to read file with any encoding: {last_exception}")


def safe_parse_timestamp(timestamp_str, default=None):
    """Safely parse timestamp with fallback to default value if parsing fails"""
    try:
        return pd.to_datetime(timestamp_str, format='%m/%d/%Y %H:%M:%S')
    except (ValueError, TypeError):
        print(f"Warning: Couldn't parse timestamp '{timestamp_str}', using default value")
        return default


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

                        # Explicitly close and cleanup response
                        response.close()
                        del response

                        if os.path.getsize(local_path) > 0:
                            success = True
                            break

                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            print(f"Failed to download {url}: {str(e)}")
                        else:
                            time.sleep(2 ** attempt)

                # Update results and progress with proper error handling
                with self.lock:
                    self.results[url] = success
                    self.completed_downloads += 1

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
        self.status = {}

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
            print(f"Processing status: {self.status}")

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


def process_folder_data(folder_path: str) -> Optional[pd.DataFrame]:
    """Process all files in a folder with immediate cleanup and parallel processing"""
    results = []
    file_processors = {
        'block.csv': process_block_file,
        'cpu.csv': process_cpu_file,
        'mem.csv': process_mem_file,
        'llite.csv': process_nfs_file
    }

    def process_file(filename: str, processor: Callable) -> Optional[pd.DataFrame]:
        """Process a single file with proper cleanup"""
        print(f"Processing {filename}")
        file_path = os.path.join(folder_path, filename)
        result = None

        if not os.path.exists(file_path):
            return None

        try:
            # Read and process the file
            if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB
                chunk_size = 10000
                chunks = []

                try:
                    # Use the read_csv_with_fallback_encoding function for chunked reading
                    chunk_reader = read_csv_with_fallback_encoding(file_path, chunk_size=chunk_size)
                    for chunk_idx, chunk in enumerate(chunk_reader):
                        try:
                            processed_chunk = processor(chunk)
                            if processed_chunk is not None and not processed_chunk.empty:
                                chunks.append(processed_chunk)
                            # Print progress for large files
                            if chunk_idx % 50 == 0:
                                print(f"Processed {chunk_idx * chunk_size} rows from {filename}")
                        except Exception as e:
                            print(f"Error processing chunk {chunk_idx} in {filename}: {str(e)}")
                            continue  # Skip this chunk but continue processing
                        finally:
                            del chunk  # Explicitly delete chunk
                            gc.collect()

                    if chunks:
                        print(f"Combining {len(chunks)} processed chunks from {filename}")
                        result = pd.concat(chunks, ignore_index=True)
                        del chunks  # Clean up chunks list
                    else:
                        print(f"No valid data chunks found in {filename}")
                except Exception as e:
                    print(f"Error in chunked processing of {filename}: {str(e)}")
            else:
                # Use processor directly for small files
                result = processor(file_path)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

        finally:
            # Always attempt to clean up the file
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing {filename}: {str(e)}")

            gc.collect()  # Force garbage collection

        return result

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=min(len(file_processors), os.cpu_count() or 4)) as executor:
        future_to_file = {
            executor.submit(process_file, filename, processor): filename
            for filename, processor in file_processors.items()
        }

        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                result = future.result()
                if result is not None and not result.empty:
                    results.append(result)
                    print(f"Successfully processed {filename} with {len(result)} valid rows")
                else:
                    print(f"No valid data extracted from {filename}")
            except Exception as e:
                print(f"Error completing parallel processing for {filename}: {str(e)}")

    # Combine results
    if results:
        try:
            print(f"Combining results from {len(results)} successfully processed files")
            final_result = pd.concat(results, ignore_index=True)
            print(f"Final result contains {len(final_result)} rows")
            del results  # Clean up individual results
            gc.collect()
            return final_result
        except Exception as e:
            print(f"Error combining results: {str(e)}")
            return None
    else:
        print("No valid results to combine")
        return None


def check_disk_space(warning_gb=20, critical_gb=5) -> tuple:
    """
    Check available disk space considering user quota with safety margin
    Returns: (is_safe, is_abundant)
    """
    try:
        # Get user's home directory
        home_dir = os.path.expanduser('~')

        # Get total quota (24GB) and used space
        quota_total = 24 * 1024 * 1024 * 1024  # 24GB in bytes

        # Add 10% safety margin to used space calculation
        used_space = sum(os.path.getsize(os.path.join(dirpath, filename))
                         for dirpath, _, filenames in os.walk(home_dir)
                         for filename in filenames)
        used_space = int(used_space * 1.1)  # Add 10% safety margin

        # Calculate available space in GB
        available_gb = (quota_total - used_space) / (1024 * 1024 * 1024)

        # Add extra margin to thresholds
        safe_threshold = critical_gb * 1.2  # 20% higher than critical
        abundant_threshold = warning_gb * 1.1  # 10% higher than warning

        return (available_gb > safe_threshold, available_gb > abundant_threshold)
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
            existing_df = pd.read_csv(file_path)
            merged_df = pd.concat([existing_df, new_df]).drop_duplicates()
            merged_df.to_csv(file_path, index=False)
        else:
            new_df.to_csv(file_path, index=False)
        saved_files.append(file_path)

    return saved_files


def upload_to_s3(file_paths: List[str], bucket_name="data-transform-conte") -> bool:
    """Upload files to S3 public bucket without requiring credentials"""
    print("\nStarting S3 upload...")

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )

    for i, file_path in enumerate(file_paths, 1):
        file_name = os.path.basename(file_path)
        for attempt in range(3):
            try:
                # Add content type for CSV files
                extra_args = {
                    'ContentType': 'text/csv'
                }

                s3_client.upload_file(
                    file_path,
                    bucket_name,
                    file_name,
                    ExtraArgs=extra_args
                )

                if i % 2 == 0 or i == len(file_paths):
                    print(f"Uploaded {i}/{len(file_paths)} files")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Failed to upload {file_name}: {str(e)}")
                    return False
                time.sleep(2 ** attempt)

    return True


def print_progress(current: int, total: int, prefix: str = '', suffix: str = ''):
    """Print progress as a percentage with a progress bar"""
    percent = (current / total) * 100
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)
    if current == total:
        print()


def save_and_upload_folder_data(result_df: pd.DataFrame, folder_name: str, base_dir: str,
                                version_manager: DataVersionManager) -> bool:
    """Save folder data and immediately upload to S3"""
    saved_files = []
    try:
        # Split results by month
        monthly_data = {
            name: group for name, group in
            result_df.groupby(result_df['Timestamp'].dt.strftime('%Y_%m'))
        }

        # Save files and get paths
        saved_files = save_monthly_data(monthly_data, base_dir, version_manager)

        # Upload immediately to S3
        if upload_to_s3(saved_files):
            print(f"\nSuccessfully uploaded data for folder {folder_name}")

            # Clean up local files after successful upload
            cleanup_success = True
            for file in saved_files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Error removing {file}: {str(e)}")
                    cleanup_success = False

            # Only increment version if upload and cleanup were successful
            if cleanup_success:
                version_manager.increment_version()
                return True
            else:
                print("Warning: Some files could not be cleaned up")
                return False
        else:
            print(f"\nFailed to upload data for folder {folder_name}")
            return False

    except Exception as e:
        print(f"Error in save_and_upload_folder_data: {str(e)}")
        # Attempt to clean up any saved files on error
        for file in saved_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except:
                pass
        return False


def main():
    base_dir = os.getcwd()
    base_url = "https://www.datadepot.rcac.purdue.edu/sbagchi/fresco/repository/Conte/TACC_Stats/"
    required_files = ['block.csv', 'cpu.csv', 'mem.csv', 'llite.csv']
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    # Initialize managers
    tracker = ProcessingTracker(base_dir, reset=False)
    version_manager = DataVersionManager(base_dir)

    # Configure number of download threads
    num_download_threads = min(os.cpu_count() or 4, 8)
    print(f"Using {num_download_threads} download threads")

    try:
        # Get list of folders to process
        folder_urls = get_folder_urls(base_url, headers)
        if not folder_urls:
            print("No folders found to process")
            return

        # Extract just the folder names from folder_urls
        all_folder_names = [name for name, _ in folder_urls]
        # Create a mapping from folder name to its index and URL
        folder_map = {name: (i, url) for i, (name, url) in enumerate(folder_urls)}

        # Identify folders that need processing (not in processed_folders)
        processed_set = set(tracker.status['processed_folders'])
        folders_to_process = [name for name in all_folder_names if name not in processed_set]

        # Special handling for previously failed folders
        failed_folders = tracker.status.get('failed_folders', [])
        print(f"Previously failed folders: {failed_folders}")

        # Calculate total folders to process
        total_folders = len(folders_to_process)
        print(f"\nStarting processing of {total_folders} folders")

        # Process folders one at a time
        for idx, folder_name in enumerate(folders_to_process):
            # Get the original index and URL for this folder
            orig_idx, folder_url = folder_map[folder_name]

            # Update overall progress
            print_progress(idx + 1, total_folders, prefix='Overall Progress:', suffix='Complete')
            print(f"\nProcessing folder {idx + 1}/{total_folders}: {folder_name}")

            # Skip if already processed (double-check)
            if tracker.is_folder_processed(folder_name):
                print(f"Folder {folder_name} already processed, skipping...")
                continue

            # Check disk space
            if not check_disk_space()[0]:
                print("Critical disk space reached. Cannot continue.")
                break

            # Create temporary folder
            temp_folder = os.path.join(base_dir, folder_name)
            try:
                # Download and process folder
                if download_folder_threaded(
                        folder_url,
                        temp_folder,
                        required_files,
                        headers,
                        num_threads=num_download_threads
                ):
                    result = process_folder_data(temp_folder)

                    if result is not None:
                        # Immediately save and upload results
                        if save_and_upload_folder_data(result, folder_name, base_dir, version_manager):
                            print(f"Successfully uploaded {folder_name} to S3")
                            tracker.mark_folder_processed(folder_name, orig_idx)

                            # If this was a previously failed folder, remove it from failed list
                            if folder_name in tracker.status['failed_folders']:
                                tracker.status['failed_folders'].remove(folder_name)
                                tracker.save_status()
                        else:
                            tracker.mark_folder_failed(folder_name)

                        # Clear memory
                        del result
                        gc.collect()
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

    except Exception as e:
        print(f"Error in main processing loop: {str(e)}")

    print("Processing completed!")


if __name__ == "__main__":
    main()
