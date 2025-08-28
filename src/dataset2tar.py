import os
import tarfile
import subprocess
from math import ceil
from multiprocessing import Pool, cpu_count


# ==== CONFIGURATION ====
source_folder = "/mnt/d/data/final_database_republishable/"
output_folder = "/mnt/c/lit2vec/data/"
files_per_archive = 10000
archive_prefix = "data_part"
pigz_threads = 16           # Threads per process (adjust to your CPU)
parallel_jobs = 10          # Number of parallel archives created at once
# ========================


def split_files(all_files, chunk_size):
    for i in range(0, len(all_files), chunk_size):
        yield all_files[i:i + chunk_size], i // chunk_size


def create_tar_gz(args):
    file_chunk, part_index = args
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

    print(f"[PART {part_index}] ✅ Done.")
    return archive_gz


def main():
    os.makedirs(output_folder, exist_ok=True)

    all_files = sorted(
        os.path.join(source_folder, f)
        for f in os.listdir(source_folder)
        if f.endswith(".json")
    )
    total_files = len(all_files)
    print(f"📦 Total JSON files: {total_files}")

    # Prepare argument list
    chunks = list(split_files(all_files, files_per_archive))
    print(f"🚀 Launching {len(chunks)} archive jobs with {parallel_jobs} in parallel...")

    with Pool(parallel_jobs) as pool:
        pool.map(create_tar_gz, chunks)

    print("🎉 All archives created.")


if __name__ == "__main__":
    main()
