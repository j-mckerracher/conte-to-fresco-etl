import pandas as pd
from tqdm import tqdm


def standardize_job_id(job_id):
    """Convert jobIDxxxxx to JOBxxxxx"""
    if isinstance(job_id, str):
        if job_id.startswith('jobID'):
            return 'JOB' + job_id[5:]
    return job_id


def analyze_timestamps(job_file, timeseries_file, sample_size=0.01):
    print(f"Analyzing with {sample_size * 100}% sample size...")

    # Read job accounting data
    jobs_df = pd.read_csv(job_file, low_memory=False)

    # Standardize job IDs in accounting data
    jobs_df['jobID'] = jobs_df['jobID'].apply(standardize_job_id)

    # Sample jobs
    jobs_df = jobs_df.sample(frac=sample_size, random_state=42)

    # Convert timestamps for jobs
    if "start" in jobs_df.columns:
        jobs_df['start'] = pd.to_datetime(jobs_df['start'])
        jobs_df['end'] = pd.to_datetime(jobs_df['end'])
    else:
        jobs_df = jobs_df[jobs_df['jobevent'].notna()]
        jobs_df['start'] = pd.to_datetime(jobs_df['start_time'])
        jobs_df['end'] = pd.to_datetime(jobs_df['end_time'])

    # Get sampled job IDs
    sampled_job_ids = jobs_df['jobID'].unique()
    print(f"\nSampled {len(sampled_job_ids)} jobs")
    print("Sample of standardized job IDs we're looking for:")
    print(list(sampled_job_ids[:5]))

    # Read time series data in chunks and filter for sampled jobs
    chunk_size = 100000
    filtered_chunks = []

    print("\nReading time series data in chunks...")
    total_matches = 0

    for chunk in tqdm(pd.read_csv(timeseries_file, chunksize=chunk_size), desc="Reading chunks"):
        # Debug first chunk
        if len(filtered_chunks) == 0:
            print("\nFirst few time series Job Ids:")
            print(chunk['Job Id'].head().tolist())

        filtered_chunk = chunk[chunk['Job Id'].isin(sampled_job_ids)]
        if not filtered_chunk.empty:
            filtered_chunks.append(filtered_chunk)
            total_matches += len(filtered_chunk)

        # Progress update every 5 chunks
        if len(filtered_chunks) % 5 == 0:
            print(f"\nMatches found so far: {total_matches}")

    if filtered_chunks:
        ts_df = pd.concat(filtered_chunks)
        ts_df['Timestamp'] = pd.to_datetime(ts_df['Timestamp'])

        print(f"\nTotal matches found: {len(ts_df)}")

        # Analysis of time series entries outside job durations
        print("\nChecking time series entries outside job durations...")
        outside_entries = 0
        total_entries = 0

        for _, job in tqdm(jobs_df.iterrows(), total=len(jobs_df), desc="Processing jobs"):
            job_ts = ts_df[ts_df['Job Id'] == job['jobID']]
            if len(job_ts) > 0:
                total_entries += len(job_ts)
                outside = job_ts[
                    (job_ts['Timestamp'] < job['start']) |
                    (job_ts['Timestamp'] > job['end'])
                    ]
                outside_entries += len(outside)

        if total_entries > 0:
            print(
                f"Entries outside job duration: {outside_entries}/{total_entries} ({outside_entries / total_entries * 100:.2f}%)")
    else:
        print("\nNo matching time series data found!")
        print("Please check the job ID samples above to verify the matching logic.")


if __name__ == "__main__":
    print("Starting analysis...")
    analyze_timestamps(
        r'C:\Users\jmckerra\OneDrive - purdue.edu\ECNDATA\Desktop\conte-job-accounting\2015-03.csv',
        r'C:\Users\jmckerra\Downloads\FRESCO_Conte_ts_2015_03_v1.csv',
        sample_size=0.01
    )