import os
import time
import pandas as pd
import botocore
import boto3
from typing import List, Union
from datetime import datetime, timedelta


def create_test_data():
    """Create a small test DataFrame with sample metrics"""
    # Generate some sample timestamps
    start_date = datetime(2015, 3, 1)
    dates = [start_date + timedelta(hours=i) for i in range(24)]

    # Create sample data
    data = []
    for timestamp in dates:
        data.append({
            'Job Id': 'JOB123',
            'Host': 'test-host',
            'Event': 'cpuuser',
            'Value': 75.5,
            'Units': 'CPU %',
            'Timestamp': timestamp
        })
        data.append({
            'Job Id': 'JOB123',
            'Host': 'test-host',
            'Event': 'memused',
            'Value': 16.2,
            'Units': 'GB',
            'Timestamp': timestamp
        })

    return pd.DataFrame(data)


def save_test_file(df: pd.DataFrame, base_dir: str) -> str:
    """Save test DataFrame to CSV file"""
    # Create monthly_data directory if it doesn't exist
    output_dir = os.path.join(base_dir, "monthly_data")
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with current timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"FRESCO_Conte_ts_2015_03_test_{timestamp}.csv"
    file_path = os.path.join(output_dir, filename)

    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"Created test file: {file_path}")
    return file_path


def upload_to_s3(file_paths: Union[str, List[str]], bucket_name="data-transform-conte") -> bool:
    """Upload files to S3 public bucket without requiring credentials"""
    print("\nStarting S3 upload...")

    # Handle single file path
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # Configure S3 client for public bucket access
    config = botocore.config.Config(
        signature_version=botocore.UNSIGNED,
        region_name='us-east-1'
    )

    s3_client = boto3.client('s3', config=config)

    for i, file_path in enumerate(file_paths, 1):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False

        file_name = os.path.basename(file_path)
        for attempt in range(3):
            try:
                # Simple content type for CSV files
                extra_args = {
                    'ContentType': 'text/csv'
                }

                print(f"Uploading {file_name}...")
                s3_client.upload_file(
                    file_path,
                    bucket_name,
                    file_name,
                    ExtraArgs=extra_args
                )

                print(f"Successfully uploaded {file_name}")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Failed to upload {file_name} after 3 attempts: {str(e)}")
                    return False
                print(f"Retry {attempt + 1} for {file_name}")
                time.sleep(2 ** attempt)

    return True


def main():
    # Use current directory as base
    base_dir = os.getcwd()

    try:
        # Create and save test data
        print("Creating test data...")
        df = create_test_data()

        # Save to file
        file_path = save_test_file(df, base_dir)

        # Upload to S3
        success = upload_to_s3(file_path)

        # Clean up local file after successful upload
        if success:
            try:
                os.remove(file_path)
                print(f"Cleaned up local file: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error removing local file: {str(e)}")

    except Exception as e:
        print(f"Error in test script: {str(e)}")


if __name__ == "__main__":
    main()