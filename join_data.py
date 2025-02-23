import polars as pl
from tqdm import tqdm


def standardize_job_id(id_series: pl.Series) -> pl.Series:
    """Convert jobIDxxxxx to JOBxxxxx"""
    return (
        pl.when(id_series.str.contains('^jobID'))
        .then(pl.concat_str([pl.lit('JOB'), id_series.str.slice(5)]))
        .otherwise(id_series)
    )


def process_chunk(jobs_df: pl.DataFrame, ts_chunk: pl.DataFrame) -> pl.DataFrame:
    """Process a chunk of time series data against all jobs"""
    # Join time series chunk with jobs data on jobID
    joined = ts_chunk.join(
        jobs_df,
        left_on="Job Id",
        right_on="jobID",
        how="inner"
    )

    # Filter timestamps that fall between job start and end times
    filtered = joined.filter(
        (pl.col("Timestamp") >= pl.col("start"))
        & (pl.col("Timestamp") <= pl.col("end"))
    )

    if filtered.height == 0:
        return None

    # Pivot the metrics into columns
    # Group by all columns except Event and Value
    group_cols = [col for col in filtered.columns if col not in ["Event", "Value"]]

    result = filtered.pivot(
        values="Value",
        index=group_cols,
        on="Event",  # Updated from 'columns' to 'on' per deprecation warning
        aggregate_function="first"
    )

    # Rename metric columns to the desired format
    for col in result.columns:
        if col in ["cpuuser", "gpu", "memused", "memused_minus_diskcache", "nfs", "block"]:
            result = result.rename({col: f"value_{col}"})

    return result


def join_job_timeseries(job_file: str, timeseries_file: str, output_file: str, chunk_size: int = 100_000):
    """
    Join job accounting data with time series data, creating a row for each timestamp.
    """
    # Define datetime formats for both data sources
    JOB_DATETIME_FMT = "%m/%d/%Y %H:%M:%S"  # Format for job data: "03/01/2015 01:29:34"
    TS_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"  # Format for timeseries data: "2015-03-01 14:56:51"

    print("Reading job accounting data...")

    # Read jobs data with schema overrides for columns that have mixed types
    jobs_df = pl.scan_csv(
        job_file,
        schema_overrides={
            "Resource_List.neednodes": pl.Utf8,
            "Resource_List.nodes": pl.Utf8,
        }
    ).collect()

    # Standardize job IDs
    jobs_df = jobs_df.with_columns([
        standardize_job_id(pl.col("jobID")).alias("jobID")
    ])

    # Convert the start and end columns to datetime using the job data format
    jobs_df = jobs_df.with_columns([
        pl.col("start").str.strptime(pl.Datetime, JOB_DATETIME_FMT).alias("start"),
        pl.col("end").str.strptime(pl.Datetime, JOB_DATETIME_FMT).alias("end")
    ])

    print("Processing time series data in chunks...")

    # Read the time series data
    ts_reader = pl.scan_csv(timeseries_file).collect()
    total_rows = ts_reader.height
    chunks = range(0, total_rows, chunk_size)
    first_chunk = True

    for chunk_start in tqdm(chunks):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        ts_chunk = ts_reader[chunk_start:chunk_end]

        # Convert the Timestamp column to datetime using the timeseries format
        ts_chunk = ts_chunk.with_columns([
            pl.col("Timestamp").str.strptime(pl.Datetime, TS_DATETIME_FMT)
        ])

        # Process the chunk by joining and filtering
        result_df = process_chunk(jobs_df, ts_chunk)

        if result_df is not None:
            # Write results to the output CSV file
            if first_chunk:
                # Write with headers for first chunk
                result_df.write_csv(output_file, include_header=True)
                first_chunk = False
            else:
                # For subsequent chunks, append by writing to a temporary file and concatenating
                temp_file = output_file + '.tmp'
                result_df.write_csv(temp_file, include_header=False)

                # Read the temporary file content
                with open(temp_file, 'r', encoding='utf-8') as temp:
                    content = temp.read()

                # Append to the main file
                with open(output_file, 'a', encoding='utf-8') as main:
                    main.write(content)

                # Clean up temporary file
                import os
                os.remove(temp_file)


if __name__ == "__main__":
    JOB_FILE = r""
    TIMESERIES_FILE = r""
    OUTPUT_FILE = r""

    join_job_timeseries(JOB_FILE, TIMESERIES_FILE, OUTPUT_FILE)