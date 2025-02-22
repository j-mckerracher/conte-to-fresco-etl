import os
import glob
import time
import boto3
import botocore.config
import shutil


def upload_to_s3(file_paths, bucket_name="data-transform-conte"):
    """Upload files to S3 public bucket without requiring credentials"""
    print("\nStarting S3 upload...")

    # Configure S3 client for public bucket access
    s3_client = boto3.client(
        's3',
        config=botocore.config.Config(
            signature_version=botocore.UNSIGNED,
            region_name='us-east-1'
        )
    )

    total_files = len(file_paths)
    uploaded_files = []

    for i, file_path in enumerate(file_paths, 1):
        file_name = os.path.basename(file_path)
        print(f"\nUploading {i}/{total_files}: {file_name}")

        for attempt in range(3):
            try:
                # Add content type for CSV files
                extra_args = {
                    'ContentType': 'text/csv',
                    'ACL': 'public-read'
                }

                s3_client.upload_file(
                    file_path,
                    bucket_name,
                    file_name,
                    ExtraArgs=extra_args
                )

                uploaded_files.append(file_path)
                print(f"Successfully uploaded {file_name}")
                break

            except Exception as e:
                if attempt == 2:
                    print(f"Failed to upload {file_name}: {str(e)}")
                else:
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2 ** attempt)

    return uploaded_files


def main():
    # Directory containing the files
    directory = "/home/dynamo/a/jmckerra/projects/conte-to-fresco-etl/monthly_data"

    # Get all CSV files in the directory
    files = glob.glob(os.path.join(directory, "*.csv"))

    if not files:
        print("No CSV files found in the directory")
        return

    print(f"Found {len(files)} CSV files to upload")

    # Upload files
    uploaded_files = upload_to_s3(files)

    if uploaded_files:
        print("\nDeleting successfully uploaded files...")
        for file_path in uploaded_files:
            try:
                os.remove(file_path)
                print(f"Deleted: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error deleting {os.path.basename(file_path)}: {str(e)}")

    print("\nProcess completed!")
    print(f"Successfully uploaded and deleted: {len(uploaded_files)} files")
    print(f"Failed uploads: {len(files) - len(uploaded_files)} files")


if __name__ == "__main__":
    main()