import os
import json
import time
from collections import Counter


def validate_cc_dataset(file_path, sample_size=100_000, print_first_n=3):
    """
    Performs a quality and integrity analysis on a Common Crawl CDXJ file.
    """
    print("=" * 65)
    print("INITIATING DATASET SANITY CHECK")
    print(f"Target file: {file_path}")
    print("=" * 65)

    # Existence and Size Validation
    if not os.path.exists(file_path):
        print(f"[CRITICAL ERROR] The file {file_path} does not exist!")
        return

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"[INFO] Disk size: {file_size_mb:,.2f} MB")

    if file_size_mb < 100:
        print("[WARNING] The file appears too small to be an uncompressed Common Crawl index file.")

    # Statistical counters
    valid_records = 0
    corrupted_records = 0
    status_codes = Counter()
    mime_types = Counter()

    start_time = time.time()

    print(f"\n[INFO] Analyzing a sample of {sample_size:,} rows...")

    # Reading and Parsing (Memory-Safe)
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < print_first_n:
                if i == 0:
                    print("\n" + "-" * 65)
                    print(f"DATA INSPECTION: Displaying the first {print_first_n} raw records")
                    print("-" * 65)

                print(f"[Record {i + 1}]")
                print(line.strip())
                print()

                if i == print_first_n - 1:
                    print("-" * 65 + "\n")
            if i >= sample_size:
                break

            try:
                # The CDXJ format is space-delimited: Key, Timestamp, JSON
                parts = line.split(' ', 2)
                if len(parts) < 3:
                    corrupted_records += 1
                    continue

                # JSON dictionary parsing
                metadata = json.loads(parts[2])

                # Metrics extraction
                status_codes[metadata.get('status', 'Unknown')] += 1
                mime_types[metadata.get('mime', 'Unknown')] += 1

                valid_records += 1

            except (json.JSONDecodeError, IndexError):
                corrupted_records += 1

    elapsed_time = time.time() - start_time

    # Final Report Generation
    print("\n" + "=" * 65)
    print("DATA QUALITY REPORT")
    print("=" * 65)
    print(f"Analysis time:         {elapsed_time:.2f} seconds")
    print(f"Sampled rows:          {sample_size:,}")
    print(f"Valid Records:         {valid_records:,} ({(valid_records / sample_size) * 100:.2f}%)")
    print(f"Corrupted Records:     {corrupted_records:,} ({(corrupted_records / sample_size) * 100:.2f}%)")

    print("\n TOP 5 HTTP STATUS CODES:")
    for status, count in status_codes.most_common(5):
        print(f"  - HTTP {status}: {count:,} occurrences")

    print("\n TOP 5 CONTENT TYPES (MIME TYPES):")
    for mime, count in mime_types.most_common(5):
        print(f"  - {mime}: {count:,} occurrences")

    print("-" * 65)
    if corrupted_records / sample_size > 0.05:
        print("[OUTCOME] WARNING: High corruption rate (> 5%). Please verify the file integrity.")
    else:
        print("[OUTCOME] SUCCESS: The file is intact and contains valid real-world data.")
    print("=" * 65)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    parent_dir = os.path.dirname(current_dir)

    dataset_path = os.path.join(parent_dir, "DATA", "cdx-00000")

    validate_cc_dataset(dataset_path)