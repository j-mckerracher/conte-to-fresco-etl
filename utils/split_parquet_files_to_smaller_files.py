"""
Split large parquet files into smaller chunks to improve memory usage during processing.
"""

import os
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ParquetFileSplitter:
    """
    Class to handle splitting large Parquet files into smaller chunks.
    """

    def __init__(self, chunk_size=1000000, logger=None):
        """
        Initialize the splitter with configuration.

        Args:
            chunk_size (int): Number of rows per output chunk file
            logger: Optional logger instance. If None, use this module's logger
        """
        self.chunk_size = chunk_size
        self.logger = logger or logging.getLogger(__name__)

    def log(self, level, message):
        """Helper function for logging"""
        if level == 'info':
            self.logger.info(message)
            print(message)  # Also print to stdout for visibility
        elif level == 'error':
            self.logger.error(message)
            print(f"ERROR: {message}", file=sys.stderr)
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
            print(f"WARNING: {message}")

    def split_file(self, input_file, output_dir, chunk_size=None, prefix=None):
        """
        Split a large parquet file into smaller chunks.

        Args:
            input_file (str): Path to the input parquet file
            output_dir (str): Directory to write the output files
            chunk_size (int): Number of rows per output file (overrides instance setting if provided)
            prefix (str): Prefix for output filenames (defaults to original filename)

        Returns:
            tuple: (success_flag, list_of_created_files)
        """
        # Use provided chunk_size or fall back to instance default
        chunk_size = chunk_size or self.chunk_size
        created_files = []

        input_path = Path(input_file)
        if not input_path.exists():
            self.log('error', f"Input file {input_file} doesn't exist")
            return False, []

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Set prefix to original filename if not specified
        if prefix is None:
            prefix = input_path.stem

        self.log('info', f"Opening {input_file}...")
        try:
            # Open the parquet file using PyArrow (more memory efficient)
            parquet_file = pq.ParquetFile(input_file)
            total_rows = parquet_file.metadata.num_rows

            self.log('info', f"File has {total_rows:,} rows, splitting into chunks of {chunk_size:,} rows")

            # Calculate number of chunks
            num_chunks = (total_rows + chunk_size - 1) // chunk_size

            # Read and write chunks
            for chunk_idx in range(num_chunks):
                start_row = chunk_idx * chunk_size
                # Use min to handle the last chunk which might be smaller
                end_row = min(start_row + chunk_size, total_rows)

                self.log('info', f"Processing chunk {chunk_idx + 1}/{num_chunks} (rows {start_row:,} to {end_row:,})")

                # Read the chunk
                table = None
                try:
                    # Try to read just the specific row range
                    # This approach is faster if the parquet file has an index
                    table = pq.read_table(
                        input_file,
                        use_threads=False,
                        filters=[('row_index', '>=', start_row), ('row_index', '<', end_row)]
                    )
                except Exception:
                    # Fall back to reading the chunk using slicing
                    try:
                        table = parquet_file.read().slice(start_row, end_row - start_row)
                    except Exception:
                        # Last resort: read the full table and slice it (memory intensive)
                        self.log('warning', "Using fallback method to read chunk (may use more memory)")
                        full_table = pq.read_table(input_file)
                        table = full_table.slice(start_row, end_row - start_row)

                if table is None or table.num_rows == 0:
                    self.log('warning', f"Chunk {chunk_idx + 1} came back empty, skipping")
                    continue

                # Write the chunk to a new file
                output_file = output_path / f"{prefix}_chunk{chunk_idx + 1:03d}.parquet"
                output_file_path = str(output_file)
                self.log('info', f"Writing {table.num_rows:,} rows to {output_file}")

                # Write with snappy compression for better performance/size balance
                pq.write_table(table, output_file, compression='snappy')
                created_files.append(output_file_path)

                # Free memory
                del table

                # Report progress
                percent_complete = (end_row / total_rows) * 100
                self.log('info', f"Progress: {percent_complete:.1f}%")

            self.log('info', f"Successfully split {input_file} into {num_chunks} files in {output_dir}")
            return True, created_files

        except Exception as e:
            self.log('error', f"Error splitting file: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False, created_files


# Original function for backward compatibility
def split_parquet_file(input_file, output_dir, chunk_size=1000000, prefix=None, logger=None):
    """
    Legacy function for backward compatibility.
    Uses the ParquetFileSplitter class internally.
    """
    splitter = ParquetFileSplitter(chunk_size=chunk_size, logger=logger)
    success, created_files = splitter.split_file(input_file, output_dir, prefix=prefix)
    return success