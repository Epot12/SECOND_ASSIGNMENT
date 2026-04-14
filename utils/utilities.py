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


def plot_amdahl_scaling(amdahl_results: dict, plots_dir: Path):
    """
    Generates the Strong Scaling graph (Amdahl's Law).
    Draws curves calculating speedup vs. time on 1 core.
    """
    print("\n[DATA VIZ] Generating Amdahl's Law Scaling Plot...")

    if not amdahl_results:
        print("[WARNING] No Amdahl data found. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))

    # setting the colors and labels for the 3 architectures
    colors = {
        "Amdahl_NoGIL_Map_Red": "#4C72B0",  # Blue
        "Amdahl_NoGIL_Mul_Thr": "#55A868",  # green
        "Amdahl_IPC": "#C44E52"  # red
    }

    labels = {
        "Amdahl_NoGIL_Map_Red": "No-GIL (Optimized Map-Reduce)",
        "Amdahl_NoGIL_Mul_Thr": "No-GIL (Standard Multi-Threading)",
        "Amdahl_IPC": "Multiprocessing (IPC Shared Memory)"
    }

    max_cores = 1

    # drawing real curves
    for arch_name, times_dict in amdahl_results.items():

        cores = sorted([int(k) for k in times_dict.keys()])
        times = [times_dict[str(c)] for c in cores]

        if max(cores) > max_cores:
            max_cores = max(cores)

        # T(1) / T(N) calculating speedup
        base_time = times[0]  # The time it took with 1 core
        speedups = [base_time / t for t in times]

        plt.plot(cores, speedups, marker='o', linewidth=2.5, markersize=8,
                 label=labels.get(arch_name, arch_name),
                 color=colors.get(arch_name, "black"))

    # drawing the "Ideal Speedup" line (y = x)
    ideal_x = np.arange(1, max_cores + 1)
    plt.plot(ideal_x, ideal_x, '--', color='gray', linewidth=2, label='Ideal Speedup')

    # Vertical line for the 4 physical cores (Hyper-Threading limit)
    plt.axvline(x=4, color='orange', linestyle=':', linewidth=2, label='Physical Core Limit')

    # Graphic aesthetics
    plt.xlabel('Number of Threads / Processes (Cores)', fontweight='bold')
    plt.ylabel('Speedup (x)', fontweight='bold')
    plt.title("Amdahl's Law: Strong Scaling Analysis", fontweight='bold', pad=20)
    plt.xticks(ideal_x)
    plt.yticks(ideal_x)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')

    plt.tight_layout()
    output_file = plots_dir / 'Fig3_Amdahl_Scaling.pdf'
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"[DATA VIZ] Amdahl plot successfully saved to {output_file}")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_gustafson_scaling(gustafson_results: dict, plots_dir: Path):
    """
    Generate a high-fidelity plot for Weak Scaling analysis (Gustafson's Law).

    In Weak Scaling, the workload per processor is kept constant. Theoretically,
    the execution time should remain invariant as the system scales. Deviations
    indicate communication overhead or resource contention.
    """
    print("\n[DATA VIZ] Generating Gustafson's Law Weak Scaling Plot...")

    if not gustafson_results:
        print("[WARNING] No Gustafson data available for plotting.")
        return

    # Academic plotting configuration
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,  # Set to True if a LaTeX distribution is installed
        'pdf.fonttype': 42
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Professional color palette for architectural distinction
    palette = {"Gustafson_NoGIL_Map_Red": "#4C72B0", "Gustafson_IPC": "#C44E52"}

    max_cores = 1
    baseline_time = None

    for arch_name, data in gustafson_results.items():
        # Ensure numeric sorting of keys (cores)
        cores = sorted([int(k) for k in data.keys()])
        times = [data[str(c)]["time"] for c in cores]

        if max(cores) > max_cores:
            max_cores = max(cores)

        if baseline_time is None:
            baseline_time = times[0]  # Execution time at unit scale

        # Plot empirical data
        ax.plot(cores, times, marker='s', markersize=10, linewidth=3,
                label=arch_name.replace("_", " "), color=palette.get(arch_name, "black"))

    # Plot Theoretical Ideal (Constant Time)
    ax.axhline(y=baseline_time, color='gray', linestyle='--', linewidth=2,
               label='Ideal Weak Scaling (Isometric)')

    # Hardware limit annotation (e.g., Physical Core threshold)
    ax.axvline(x=4, color='#E68143', linestyle=':', linewidth=2, label='Physical Cores Boundary')

    # Refining axis aesthetics
    ax.set_xlabel('Computational Units (Cores/Threads)', fontweight='bold')
    ax.set_ylabel('Execution Time (Seconds)', fontweight='bold')
    ax.set_title("Weak Scaling Perspective: Execution Time Invariance", fontweight='bold', pad=20)
    ax.set_xticks(range(1, max_cores + 1))
    ax.legend(frameon=True, loc='upper left', fontsize='small')

    plt.tight_layout()
    output_file = plots_dir / 'Fig4_Gustafson_Weak_Scaling.pdf'
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()
    print(f"[DATA VIZ] Gustafson plot successfully exported to {output_file}")


def plot_chunk_optimization(chunk_results: dict, plots_dir: Path):
    """
    Perform Granularity Analysis to identify the optimal work-distribution threshold.

    The plot visualizes the trade-off between parallelization overhead (small chunks)
    and load imbalance/tail latency (large chunks), typically resulting in a
    U-shaped efficiency curve.
    """
    print("\n[DATA VIZ] Generating Chunk Granularity Optimization Plot...")

    if not chunk_results:
        return

    plt.figure(figsize=(10, 6))

    # Extract data: sizes are keys, times are values
    sizes = sorted([int(k) for k in chunk_results.keys()])
    times = [chunk_results[str(s)] for s in sizes]

    # Use logarithmic scale for X-axis as chunk sizes often span orders of magnitude
    plt.semilogx(sizes, times, marker='D', color='#8172B3', linewidth=3, markersize=8,
                 label='Empirical Performance')

    # Identify and highlight the mathematical optimum
    min_time = min(times)
    optimal_size = sizes[times.index(min_time)]

    plt.annotate(f'Optimal Size: {optimal_size:,}',
                 xy=(optimal_size, min_time),
                 xytext=(optimal_size, min_time + (max(times) * 0.1)),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                 horizontalalignment='center', fontweight='bold')

    plt.xlabel('Task Granularity (Chunk Size)', fontweight='bold')
    plt.ylabel('Total Processing Latency (Seconds)', fontweight='bold')
    plt.title('Granularity Profiling: Identifying the Efficiency "Sweet Spot"', fontweight='bold', pad=20)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    output_file = plots_dir / 'Fig5_Granularity_Analysis.pdf'
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()
    print(f"[DATA VIZ] Granularity plot exported to {output_file}")
