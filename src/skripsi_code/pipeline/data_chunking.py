from pathlib import Path
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cProfile
import pstats
from memory_profiler import profile
import polars as pl


def write_chunk(output_file: Path, header: list, sample: list) -> None:
    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(sample)  # Write all rows at once


@profile
def split_csv_by_rows(
    input_dir: str, output_dir: str, data_name: str, n_samples: int
) -> None:
    input_file = Path(input_dir) / data_name
    output_dir = Path(output_dir) / data_name[:-4]

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as infile:
        reader = csv.reader(infile)
        header = next(reader)

        chunk_count = 1
        rows = []

        # Thread pool for I/O-bound tasks
        with ThreadPoolExecutor() as executor:
            futures = []
            for row in tqdm(reader, desc=f"Partitioning {input_file.stem}", leave=True):
                rows.append(row)

                if len(rows) >= n_samples:
                    output_file = (
                        output_dir / f"{input_file.stem}_chunk_{chunk_count}.csv"
                    )
                    # Asynchronous thread execution for I/O-bound task
                    future = executor.submit(write_chunk, output_file, header, rows)
                    futures.append(future)

                    rows = []  # Clear the rows after writing
                    chunk_count += 1

            # Write any remaining rows in the last chunk
            if rows:
                output_file = output_dir / f"{input_file.stem}_chunk_{chunk_count}.csv"
                future = executor.submit(write_chunk, output_file, header, rows)
                futures.append(future)

            # Wait for all futures to complete
            for future in as_completed(futures):
                future.result()  # This will raise an exception if any task failed


if __name__ == "__main__":
    INPUT_PATH = "../data/raw/NIDS_DATA/"

    DATA_LIST = [
        # "NF-UNSW-NB15-v2.csv",
        # "NF-ToN-IoT-v2.csv",
        "NF-CSE-CIC-IDS2018-v2.csv",
        "NF-BoT-IoT-v2.csv",
    ]

    OUTPUT_PATH = "../data/interim/"

    profiler = cProfile.Profile()
    profiler.enable()

    for data_file in DATA_LIST:
        split_csv_by_rows(
            input_dir=INPUT_PATH,
            output_dir=OUTPUT_PATH,
            data_name=data_file,
            n_samples=50000,
        )
        print(f"Completed splitting for {data_file}")

    profiler.disable()
    profiler.dump_stats("data_chunk.prof")
    stats = pstats.Stats(profiler).sort_stats("time")
    stats.print_stats(10)
