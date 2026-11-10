"""Pack a folder of per-record JSON files into numbered .tar.gz archives.

This is how the validation reports on Hugging Face (data_part_000.tar.gz, ...) were produced.
It needs the `pigz` command-line tool.

Example:
    python -m src.dataset2tar --source results/validation --output release/validation
"""

import argparse
import os
import subprocess
import tarfile
from functools import partial
from multiprocessing import Pool


def split_files(all_files, chunk_size):
    for i in range(0, len(all_files), chunk_size):
        yield all_files[i:i + chunk_size], i // chunk_size


def create_tar_gz(job, output_folder, archive_prefix, pigz_threads):
    file_chunk, part_index = job
    archive_basename = f"{archive_prefix}_{part_index:03}"
    archive_tar = os.path.join(output_folder, f"{archive_basename}.tar")
    archive_gz = f"{archive_tar}.gz"

    print(f"[PART {part_index}] Creating {archive_tar} with {len(file_chunk)} files")

    # Write .tar
    with tarfile.open(archive_tar, "w") as tar:
        for file_path in file_chunk:
            arcname = os.path.basename(file_path)
            tar.add(file_path, arcname=arcname)

    print(f"[PART {part_index}] Compressing {archive_tar} -> {archive_gz} using pigz...")
    subprocess.run(["pigz", "-p", str(pigz_threads), archive_tar], check=True)

    print(f"[PART {part_index}] Done.")
    return archive_gz


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Folder with the *.json files to pack.")
    parser.add_argument("--output", required=True, help="Folder that receives the archives.")
    parser.add_argument("--files-per-archive", type=int, default=10000)
    parser.add_argument("--prefix", default="data_part", help="Archive name prefix (default: %(default)s).")
    parser.add_argument("--pigz-threads", type=int, default=16, help="Threads per pigz process.")
    parser.add_argument("--jobs", type=int, default=10, help="Archives created in parallel.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    all_files = sorted(
        os.path.join(args.source, f)
        for f in os.listdir(args.source)
        if f.endswith(".json")
    )
    print(f"Total JSON files: {len(all_files)}")

    chunks = list(split_files(all_files, args.files_per_archive))
    print(f"Launching {len(chunks)} archive jobs with {args.jobs} in parallel...")

    worker = partial(
        create_tar_gz,
        output_folder=args.output,
        archive_prefix=args.prefix,
        pigz_threads=args.pigz_threads,
    )
    with Pool(args.jobs) as pool:
        pool.map(worker, chunks)

    print("All archives created.")


if __name__ == "__main__":
    main()
