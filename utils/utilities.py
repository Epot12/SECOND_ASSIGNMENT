import sys
import subprocess
import hashlib
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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



# plotting function

def generate_plots(synth_data, real_data, io_data, plots_dir: Path):
    print("\n[DATA VIZ] Generating High-Fidelity PDF Plots...")

    # PLOT 1: Execution times (Synthetic vs Real)
    plt.figure(figsize=(10, 6))

    labels = list(synth_data.keys())
    synth_totals = [synth_data[k]['ins'] + synth_data[k]['read'] for k in labels]
    real_totals = [real_data.get(k, {'ins': 0, 'read': 0})['ins'] + real_data.get(k, {'ins': 0, 'read': 0})['read'] for
                   k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, synth_totals, width, label='Synthetic Data', color='#4C72B0')
    rects2 = ax.bar(x + width / 2, real_totals, width, label='Real Data (Common Crawl)', color='#55A868')

    ax.set_ylabel('Total Execution Time (Seconds)', fontweight='bold')
    ax.set_title('Architectural Performance Comparison: Synthetic vs Real Workloads', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    fig.tight_layout()
    plt.savefig(plots_dir / 'Fig1_Total_Execution_Time.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    # PLOT 2: I/O Throughput Analysis
    if io_data:
        plt.figure(figsize=(10, 6))

        experiments = list(io_data.keys())  # es. "Synth_GIL", "Real_NOGIL"
        seq_th = [io_data[e].get('Sequential_Throughput', 0) for e in experiments]
        async_th = [io_data[e].get('Async_Throughput', 0) for e in experiments]

        x = np.arange(len(experiments))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, seq_th, width, label='Sequential (Stop-and-Wait)', color='#C44E52')
        ax.bar(x + width / 2, async_th, width, label='Async Parallel (Overlap)', color='#8172B3')

        ax.set_ylabel('Throughput (Items / Second)', fontweight='bold')
        ax.set_title('I/O Pipeline Efficiency: Sequential vs Asynchronous Overlap', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(experiments)
        ax.legend()

        fig.tight_layout()
        plt.savefig(plots_dir / 'Fig2_IO_Throughput.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    print(f"[DATA VIZ] Plots successfully saved in {plots_dir}")
