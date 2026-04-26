import os
import requests
import json
import gzip
import shutil
from pathlib import Path

# --- CONFIGURATION ---
# Base URL for Common Crawl Index (CC-MAIN-2024-10)
URL = "https://commoncrawl.s3.amazonaws.com/cc-index/collections/CC-MAIN-2024-10/indexes/cdx-00000.gz"
DATA_DIR = Path("DATA")
RAW_GZ = DATA_DIR / "cdx-00000.gz"
RAW_FILE = DATA_DIR / "cdx-00000"
OUTPUT_FILE = DATA_DIR / "common_crawl_FULL.txt"


def setup_environment():
    """Creates the data directory if it doesn't exist."""
    if not DATA_DIR.exists():
        print(f"[SYSTEM] Creating directory: {DATA_DIR}")
        DATA_DIR.mkdir(parents=True)


def download_file(url, dest):
    """Downloads the dataset with a simple progress tracker."""
    if dest.exists() or RAW_FILE.exists():
        print(f"[SKIP] File already exists at {dest} or {RAW_FILE}. Skipping download.")
        return

    print(f"[DOWNLOAD] Fetching dataset from S3... (This might take a while)")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"[SUCCESS] Download completed: {dest}")


def decompress_file(src, dest):
    """Decompresses the .gz file to a raw CDXJ file."""
    if dest.exists():
        print(f"[SKIP] Decompressed file {dest} already exists.")
        return

    print(f"[DECOMPRESS] Unzipping {src} to {dest}...")
    with gzip.open(src, 'rb') as f_in:
        with open(dest, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"[SUCCESS] Decompression complete. Removing temporary .gz file.")
    src.unlink()  # Cleanup to save disk space


def extract_urls(input_file, output_file):
    """Extracts only URLs from the raw metadata file."""
    if output_file.exists():
        print(f"[SKIP] Processed dataset {output_file} already exists.")
        return

    print(f"[EXTRACT] Starting URL extraction from {input_file}...")
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                # CDXJ format: Key Timestamp {JSON_Metadata}
                parts = line.split(' ', 2)
                if len(parts) < 3: continue

                metadata = json.loads(parts[2])
                url = metadata.get('url')
                if url:
                    f_out.write(url + '\n')
                    count += 1
                    if count % 1000000 == 0:
                        print(f"  -> Extracted {count:,} URLs...")
            except (json.JSONDecodeError, IndexError):
                continue

    print(f"[FINISH] Dataset ready: {count:,} URLs saved in {output_file}")


def main():
    print("=" * 60)
    print("      COMMON CRAWL DATASET AUTO-SETUP")
    print("=" * 60)

    try:
        setup_environment()
        download_file(URL, RAW_GZ)
        decompress_file(RAW_GZ, RAW_FILE)
        extract_urls(RAW_FILE, OUTPUT_FILE)
        print("\n[COMPLETE] Your environment is ready for benchmarking.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")


if __name__ == "__main__":
    main()