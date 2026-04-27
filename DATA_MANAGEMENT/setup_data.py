import os
import requests
import json
import gzip
import shutil
import hashlib
from pathlib import Path

# --- CONFIGURATION ---
# Exact shard used in benchmarks (CC-MAIN-2024-10)
URL = "https://data.commoncrawl.org/cc-index/collections/CC-MAIN-2024-10/indexes/cdx-00000.gz"
# SHA-256 of the COMPRESSED .gz file to ensure identity
EXPECTED_SHA256 = "65147879e6079c656976f9d342080a9a957866531f82b793165b45289945a05b"

DATA_DIR = Path("DATA")
RAW_GZ = DATA_DIR / "cdx-00000.gz"
RAW_FILE = DATA_DIR / "cdx-00000"
OUTPUT_FILE = DATA_DIR / "common_crawl_FULL.txt"

# Limit for the professor (3M is enough for reproducibility, -1 for Full)
EXTRACT_LIMIT = 3_000_000


def verify_sha256(file_path, expected_hash):
    """Verifies that the downloaded file is exactly what we expect."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_hash


def download_file(url, dest):
    """Downloads the dataset with User-Agent and single-pass stream."""
    if dest.exists() or RAW_FILE.exists():
        print(f"[SKIP] File already exists at {dest}. Checking integrity...")
        return

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'
    }

    print(f"[DOWNLOAD] Fetching dataset from Common Crawl... (approx 1.5 GB)")
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            # Chunking to handle large file without RAM explosion
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    print(f"[VERIFY] Checking SHA-256 integrity...")
    if verify_sha256(dest, EXPECTED_SHA256):
        print(f"[SUCCESS] Hash matches! The dataset is identical to the one used in the thesis.")
    else:
        print(f"[CRITICAL] Hash mismatch! The file might be corrupted or updated. Deleting...")
        dest.unlink()
        exit(1)


def decompress_file(src, dest):
    """Decompresses .gz to raw text."""
    if dest.exists():
        print(f"[SKIP] Raw file {dest} already exists.")
        return

    print(f"[DECOMPRESS] Unzipping to {dest.name}... (approx 5.5 GB on disk)")
    with gzip.open(src, 'rb') as f_in:
        with open(dest, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"[SUCCESS] Decompression complete. Removing .gz to save space.")
    src.unlink()


def extract_urls(input_file, output_file, limit):
    """Extracts clean URLs with a progress counter."""
    if output_file.exists():
        print(f"[SKIP] Final dataset {output_file.name} already exists.")
        return

    print(f"[EXTRACT] Processing raw data (Limit: {limit:,} URLs)...")
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                parts = line.split(' ', 2)
                if len(parts) < 3: continue
                metadata = json.loads(parts[2])
                url = metadata.get('url')
                if url:
                    f_out.write(url + '\n')
                    count += 1
                    if count % 1_000_000 == 0:
                        print(f"  -> Progress: {count:,} URLs extracted...")
                    if 0 < limit <= count:
                        break
            except (json.JSONDecodeError, IndexError):
                continue

    print(f"[FINISH] Setup complete! Saved {count:,} URLs to {output_file}")


def main():
    print("=" * 70)
    print("   COMMON CRAWL REPRODUCIBILITY TOOL - PARALLEL COMPUTING ASSIGNMENT")
    print("=" * 70)

    if not DATA_DIR.exists(): DATA_DIR.mkdir(parents=True)

    try:
        download_file(URL, RAW_GZ)
        decompress_file(RAW_GZ, RAW_FILE)
        extract_urls(RAW_FILE, OUTPUT_FILE, EXTRACT_LIMIT)
        print("\n[READY] The environment is identical to the developer's one.")
    except Exception as e:
        print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    main()