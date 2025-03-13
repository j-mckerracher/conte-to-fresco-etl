"""
Ready signal utility class for ETL job synchronization.
This class handles creating and checking status signals between ETL components.
"""

import os
import sys
import time
import logging
import re  # Added proper import at module level
from pathlib import Path
from enum import Enum


class JobStatus(Enum):
    """Enumeration of possible job status values"""
    UNKNOWN = "unknown"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class ReadySignalManager:
    """
    Class to manage ready signals between ETL components.

    This class handles creating and checking status files that coordinate
    the handoff between ETL Manager and Server Processor.
    """

    def __init__(self, ready_dir=None, logger=None):
        """
        Initialize the ReadySignalManager.

        Args:
            ready_dir (str or Path, optional): Path to the ready directory.
                Defaults to "/home/dynamo/a/jmckerra/projects/conte-to-fresco-etl/cache/ready".
            logger (logging.Logger, optional): Logger to use. If None, creates a new logger.
        """
        # Set default ready directory if not provided
        if ready_dir is None:
            self.ready_dir = Path(r"/home/dynamo/a/jmckerra/projects/conte-to-fresco-etl/cache/ready")
        else:
            self.ready_dir = Path(ready_dir)

        # Set up logging
        self.logger = logger or self._setup_logger()

        # Ensure ready directory exists
        self._ensure_ready_dir()

    def _setup_logger(self):
        """Set up a default logger if none was provided"""
        logger = logging.getLogger(__name__)

        # Only add handlers if none exist yet
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # Add file handler
            try:
                file_handler = logging.FileHandler("ready_signal_manager.log")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not create log file: {e}")

        return logger

    def _ensure_ready_dir(self):
        """Ensure the ready directory exists"""
        try:
            os.makedirs(self.ready_dir, exist_ok=True)
            self.logger.debug(f"Ensured ready directory exists: {self.ready_dir}")
        except Exception as e:
            self.logger.error(f"Error creating ready directory {self.ready_dir}: {e}")

    def get_signal_path(self, year, month, status):
        """
        Get the path to a signal file.

        Args:
            year (str): Year (YYYY)
            month (str): Month (MM)
            status (str or JobStatus): Status for the signal file

        Returns:
            Path: Path to the signal file
        """
        # Convert JobStatus enum to string if needed
        if isinstance(status, JobStatus):
            status = status.value

        # Format year-month
        if len(month) == 1:
            month = f"0{month}"

        year_month = f"{year}-{month}"
        return self.ready_dir / f"{year_month}.{status}"

    def create_signal(self, year, month, status, message=None):
        """
        Create a signal file.

        Args:
            year (str): Year (YYYY)
            month (str): Month (MM)
            status (str or JobStatus): Status for the signal file
            message (str, optional): Custom message to write to the file.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert JobStatus enum to string if needed
            if isinstance(status, JobStatus):
                status_str = status.value
            else:
                status_str = status

            # Get signal file path
            signal_file = self.get_signal_path(year, month, status_str)

            # Default message if none provided
            if message is None:
                message = f"Status {status_str} set at {time.strftime('%Y-%m-%d %H:%M:%S')}"

            # Write the signal file
            with open(signal_file, 'w') as f:
                f.write(message)

            self.logger.info(f"Created {status_str} signal for {year}-{month} at {signal_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating {status} signal for {year}-{month}: {e}")
            return False

    def create_ready_signal(self, year, month, message=None):
        """
        Create a ready signal file.

        Args:
            year (str): Year (YYYY)
            month (str): Month (MM)
            message (str, optional): Custom message to write to the file.

        Returns:
            bool: True if successful, False otherwise
        """
        if message is None:
            message = f"Files ready for processing at {time.strftime('%Y-%m-%d %H:%M:%S')}"

        return self.create_signal(year, month, JobStatus.READY, message)

    def create_processing_signal(self, year, month, message=None):
        """
        Create a processing signal file.

        Args:
            year (str): Year (YYYY)
            month (str): Month (MM)
            message (str, optional): Custom message to write to the file.

        Returns:
            bool: True if successful, False otherwise
        """
        if message is None:
            message = f"Processing started at {time.strftime('%Y-%m-%d %H:%M:%S')}"

        return self.create_signal(year, month, JobStatus.PROCESSING, message)

    def create_complete_signal(self, year, month, message=None):
        """
        Create a complete signal file and remove ready and processing signals.

        Args:
            year (str): Year (YYYY)
            month (str): Month (MM)
            message (str, optional): Custom message to write to the file.

        Returns:
            bool: True if successful, False otherwise
        """
        if message is None:
            message = f"Processing completed successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}"

        # Create complete signal
        result = self.create_signal(year, month, JobStatus.COMPLETE, message)

        # Remove ready and processing signals if they exist
        try:
            ready_file = self.get_signal_path(year, month, JobStatus.READY)
            processing_file = self.get_signal_path(year, month, JobStatus.PROCESSING)

            if ready_file.exists():
                ready_file.unlink()
                self.logger.debug(f"Removed ready signal for {year}-{month}")

            if processing_file.exists():
                processing_file.unlink()
                self.logger.debug(f"Removed processing signal for {year}-{month}")
        except Exception as e:
            self.logger.warning(f"Error removing signals for {year}-{month}: {e}")

        return result

    def create_failed_signal(self, year, month, message=None):
        """
        Create a failed signal file and remove ready and processing signals.

        Args:
            year (str): Year (YYYY)
            month (str): Month (MM)
            message (str, optional): Custom message to write to the file.

        Returns:
            bool: True if successful, False otherwise
        """
        if message is None:
            message = f"Processing failed at {time.strftime('%Y-%m-%d %H:%M:%S')}"

        # Create failed signal
        result = self.create_signal(year, month, JobStatus.FAILED, message)

        # Remove ready and processing signals if they exist
        try:
            ready_file = self.get_signal_path(year, month, JobStatus.READY)
            processing_file = self.get_signal_path(year, month, JobStatus.PROCESSING)

            if ready_file.exists():
                ready_file.unlink()
                self.logger.debug(f"Removed ready signal for {year}-{month}")

            if processing_file.exists():
                processing_file.unlink()
                self.logger.debug(f"Removed processing signal for {year}-{month}")
        except Exception as e:
            self.logger.warning(f"Error removing signals for {year}-{month}: {e}")

        return result

    def check_status(self, year, month):
        """
        Check the current status of a job.

        Args:
            year (str): Year (YYYY)
            month (str): Month (MM)

        Returns:
            JobStatus: Current status of the job
        """
        # Check for complete signal
        complete_file = self.get_signal_path(year, month, JobStatus.COMPLETE)
        failed_file = self.get_signal_path(year, month, JobStatus.FAILED)
        processing_file = self.get_signal_path(year, month, JobStatus.PROCESSING)
        ready_file = self.get_signal_path(year, month, JobStatus.READY)

        if complete_file.exists():
            return JobStatus.COMPLETE
        elif failed_file.exists():
            return JobStatus.FAILED
        elif processing_file.exists():
            return JobStatus.PROCESSING
        elif ready_file.exists():
            return JobStatus.READY
        else:
            return JobStatus.UNKNOWN

    def is_ready(self, year, month):
        """
        Check if a job is ready for processing.

        Args:
            year (str): Year (YYYY)
            month (str): Month (MM)

        Returns:
            bool: True if the job is ready, False otherwise
        """
        return self.check_status(year, month) == JobStatus.READY

    def is_processing(self, year, month):
        """
        Check if a job is currently being processed.

        Args:
            year (str): Year (YYYY)
            month (str): Month (MM)

        Returns:
            bool: True if the job is being processed, False otherwise
        """
        return self.check_status(year, month) == JobStatus.PROCESSING

    def is_complete(self, year, month):
        """
        Check if a job has completed successfully.

        Args:
            year (str): Year (YYYY)
            month (str): Month (MM)

        Returns:
            bool: True if the job is complete, False otherwise
        """
        return self.check_status(year, month) == JobStatus.COMPLETE

    def is_failed(self, year, month):
        """
        Check if a job has failed.

        Args:
            year (str): Year (YYYY)
            month (str): Month (MM)

        Returns:
            bool: True if the job has failed, False otherwise
        """
        return self.check_status(year, month) == JobStatus.FAILED

    def get_ready_jobs(self):
        """
        Get a list of all jobs that are ready for processing.

        Returns:
            list: List of (year, month) tuples for ready jobs
        """
        ready_jobs = []

        # Check for ready job signals
        for signal_file in self.ready_dir.glob("*.ready"):
            try:
                year_month = signal_file.stem  # Filename without extension should be YYYY-MM

                # Validate the year-month format
                if re.match(r'^\d{4}-\d{2}$', year_month):
                    year, month = year_month.split('-')
                    ready_jobs.append((year, month))
            except Exception as e:
                self.logger.warning(f"Error parsing ready file {signal_file}: {e}")

        return ready_jobs


# Command-line interface for manual signal creation
if __name__ == "__main__":
    def print_usage():
        """Print usage instructions"""
        print("Usage: python ready_signal_manager.py <command> <year> <month> [message]")
        print("Commands: ready, processing, complete, failed, status")
        print("Example: python ready_signal_manager.py ready 2015 03 \"Files ready for processing\"")

    # Check command line arguments
    if len(sys.argv) < 4:
        print_usage()
        sys.exit(1)

    try:
        command = sys.argv[1].lower()
        year = sys.argv[2]
        month = sys.argv[3]

        # Get optional message
        message = sys.argv[4] if len(sys.argv) > 4 else None

        # Validate year and month
        if not (year.isdigit() and month.isdigit() and
                len(year) == 4 and 1 <= int(month) <= 12):
            print("Invalid year or month format. Year should be 4 digits, month should be 01-12.")
            print_usage()
            sys.exit(1)

        # Format month to ensure it's 2 digits
        month = month.zfill(2)

        # Create signal manager
        manager = ReadySignalManager()

        # Process command
        if command == "ready":
            success = manager.create_ready_signal(year, month, message)
        elif command == "processing":
            success = manager.create_processing_signal(year, month, message)
        elif command == "complete":
            success = manager.create_complete_signal(year, month, message)
        elif command == "failed":
            success = manager.create_failed_signal(year, month, message)
        elif command == "status":
            status = manager.check_status(year, month)
            print(f"Status for {year}-{month}: {status.value}")
            sys.exit(0)
        else:
            print(f"Unknown command: {command}")
            print_usage()
            sys.exit(1)

        if success:
            print(f"Successfully created {command} signal for {year}-{month}")
            sys.exit(0)
        else:
            print(f"Failed to create {command} signal for {year}-{month}")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)