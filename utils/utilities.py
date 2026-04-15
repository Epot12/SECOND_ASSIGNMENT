import sys
import subprocess
import hashlib
import re
from pathlib import Path

# support functions

def verify_dataset_integrity(filepath: Path, expected_hash: str) -> bool:
    print(f"\n[INTEGRITY CHECK] Verifying SHA-256 for {filepath.name}...")
    print("                  (This might take a minute for 5.6 GB...)")

    if not filepath.exists():
        print(f"[ERROR] Dataset not found in: {filepath}")
        return False

    sha256_hash = hashlib.sha256()
    # Reading in 4MB chunks to avoid saturating the RAM
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096 * 1024), b""):
            sha256_hash.update(byte_block)

    calculated_hash = sha256_hash.hexdigest().upper()

    if calculated_hash == expected_hash.upper():
        print("[INTEGRITY CHECK] PASS: Hash matches expected value.")
        return True
    else:
        print(f"[INTEGRITY CHECK] FAIL: Hash mismatch!")
        print(f"                  Expected: {expected_hash.upper()}")
        print(f"                  Got:      {calculated_hash}")
        return False


def run_and_capture(command: list[str], description: str, cwd: Path) -> str:
    print(f"\n[GRAND ORCHESTRATOR] {description}")
    try:
        # capturing standard output in real time
        result = subprocess.run(command, check=True, cwd=cwd, capture_output=True, text=True)
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL ERROR] Execution failed: {' '.join(command)}")
        print(f"Error Output:\n{e.stderr}")
        sys.exit(1)


def parse_io_metrics(stdout: str) -> dict:
    """Using RegEx to extract times and throughputs from I/O script prints."""
    metrics = {}

    # searching sequential throughput
    seq_th_match = re.search(r"\[METRIC\] Throughput:\s+([\d,]+)\s+items/second", stdout)
    # Searching for Asynchronous throughput (using findall to get the second METRIC block)
    all_th_matches = re.findall(r"\[METRIC\] Throughput:\s+([\d,]+)\s+items/second", stdout)

    if len(all_th_matches) >= 2:
        metrics['Sequential_Throughput'] = int(all_th_matches[0].replace(',', ''))
        metrics['Async_Throughput'] = int(all_th_matches[1].replace(',', ''))

    speedup_match = re.search(r"achieved a ([\d.]+)x speedup", stdout)
    if speedup_match:
        metrics['Speedup'] = float(speedup_match.group(1))

    return metrics


def load_data(mode, ins, tst):
    if mode == "synthetic":
        print(f"[SYSTEM] Generating {ins + tst} Synthetic items...")
        present = [f"IN_{i}" for i in range(ins)]
        absent = [f"OUT_{i}" for i in range(tst)]
        return present, absent
    else:
        print(f"[SYSTEM] Pre-loading {ins + tst} REAL items from disk...")
        data_path = os.path.join(parent_dir, "DATA", "common_crawl_FULL.txt")
        all_items = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for _ in range(ins + tst):
                line = f.readline()
                if not line: break
                all_items.append(line.strip())
        return all_items[:ins], all_items[ins:ins + tst]