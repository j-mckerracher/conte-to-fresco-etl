import os
import boto3
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from botocore.exceptions import BotoCoreError, ClientError
from botocore.config import Config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3_Client():
    def __init__(self, upload_bucket, download_bucket_proc_metric, download_bucket_job_accounting):
        self.upload_bucket = upload_bucket
        self.download_bucket_proc_metric = download_bucket_proc_metric
        self.download_bucket_job_accounting = download_bucket_job_accounting
        self.s3_client = self.get_s3_client()

    def get_s3_client(self):
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
            ))

    def list_s3_files(self, bucket):
        response = self.s3_client.list_objects_v2(Bucket=bucket)
        files_in_s3 = []

        # Extract files from the initial response
        if 'Contents' in response:
            files_in_s3 = [item['Key'] for item in response['Contents']]

        # Handle pagination if there are more objects
        while response.get('IsTruncated', False):
            response = self.s3_client.list_objects_v2(
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
    def download_file(self, file, temp_dir, proc_or_job):
        """Download a single file from S3 with retry logic"""
        download_path = temp_dir / os.path.basename(file)
        if proc_or_job == "proc":
            self.s3_client.download_file(self.download_bucket_proc_metric, file, str(download_path))
        else:
            self.s3_client.download_file(self.download_bucket_job_accounting, file, str(download_path))
        return download_path

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(4),
        retry=retry_if_exception_type((BotoCoreError, ClientError))
    )
    def upload_file_to_s3(self, file_path: str, bucket_name: str) -> None:
        """
        Uploads a single file to an S3 bucket.

        Args:
            file_path (str): Local path to the file.
            bucket_name (str): Name of the S3 bucket.

        Raises:
            BotoCoreError, ClientError: If the upload fails due to a boto3 error.
        """
        # Add content type for CSV files
        extra_args = {
            'ContentType': 'text/csv'
        }

        # Use the filename as the S3 key
        s3_key = os.path.basename(file_path)

        logger.info(f"Uploading {file_path} to {bucket_name}/{s3_key}")
        self.s3_client.upload_file(file_path, bucket_name, s3_key, ExtraArgs=extra_args)
