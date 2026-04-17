import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path




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
    print("\n[DATA VIZ] Generating Amdahl's Law Scaling Plot...")
    if not amdahl_results: return

    plt.figure(figsize=(10, 6))

    # Define professional color palette and academic labels for the evaluated architectures
    colors = {"Amdahl_NoGIL_Map_Red": "#4C72B0", "Amdahl_NoGIL_Mul_Thr": "#55A868", "Amdahl_IPC": "#C44E52"}
    labels = {"Amdahl_NoGIL_Map_Red": "No-Old_GIL (Optimized Map-Reduce)",
              "Amdahl_NoGIL_Mul_Thr": "No-Old_GIL (Standard Multi-Threading)",
              "Amdahl_IPC": "Multiprocessing (IPC Shared Memory)"}

    max_cores = 1

    palette = sns.color_palette("husl", n_colors=len(amdahl_results))

    for i, (arch_name, data) in enumerate(amdahl_results.items()):
        if arch_name == "Amdahl_Sequential": continue

        cores = sorted([int(k) for k in data.keys()])
        means = [data[str(c)]["mean"] for c in cores]

        base_time = means[0]
        speedups = [base_time / m for m in means]

        label = labels.get(arch_name, arch_name.replace("Amdahl", "").replace("_", " "))
        color = colors.get(arch_name, palette[i % len(palette)])

        plt.plot(cores, speedups, marker='o', linewidth=2.5, label=label, color=color)

    # Plotting the theoretical ideal speedup (linear scalability)
    ideal_x = np.arange(1, max_cores + 1)
    plt.plot(ideal_x, ideal_x, '--', color='gray', linewidth=2, label='Ideal Speedup')

    # Demarcating the hardware boundary (transition from physical cores to Hyper-Threading)
    plt.axvline(x=4, color='orange', linestyle=':', linewidth=2, label='Physical Core Limit')

    # Refining axis aesthetics and typography
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


def plot_gustafson_scaling(gustafson_results: dict, plots_dir: Path):
    print("\n[DATA VIZ] Generating Gustafson's Law Plot with Confidence Intervals...")
    if not gustafson_results: return

    # Academic plotting configuration via Seaborn
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {"Gustafson_NoGIL_Map_Red": "#4C72B0", "Gustafson_NoGIL_Mul_Thr": "#55A868", "Gustafson_IPC": "#C44E52"}
    labels = {"Gustafson_NoGIL_Map_Red": "No-Old_GIL (Optimized Map-Reduce)",
              "Gustafson_NoGIL_Mul_Thr": "No-Old_GIL (Standard Multi-Threading)",
              "Gustafson_IPC": "Multiprocessing (IPC Shared Memory)"}

    max_cores = 1
    baseline_time = None

    for arch_name, data in gustafson_results.items():
        cores = sorted([int(k) for k in data.keys()])

        # STATISTICAL EXTRACTION: Extracting empirical means and their respective 95% Confidence Intervals
        means = [data[str(c)]["mean"] for c in cores]
        margins = [data[str(c)]["ci_95_margin"] for c in cores]

        if max(cores) > max_cores: max_cores = max(cores)
        if baseline_time is None: baseline_time = means[0]

        # RENDERING ERROR BARS: Displaying statistical uncertainty across multiple runs
        ax.errorbar(cores, means, yerr=margins, fmt='-s', markersize=8, linewidth=2.5, capsize=5, capthick=2,
                    label=labels.get(arch_name, arch_name), color=palette.get(arch_name, "black"))

    # Plotting the theoretical isometric weak scaling (constant execution time)
    ax.axhline(y=baseline_time, color='gray', linestyle='--', linewidth=2, label='Ideal Weak Scaling')
    ax.axvline(x=4, color='#E68143', linestyle=':', linewidth=2, label='Physical Cores Boundary')

    ax.set_xlabel('Computational Units (Cores/Threads)', fontweight='bold')
    ax.set_ylabel('Execution Time (Seconds)', fontweight='bold')
    ax.set_title("Weak Scaling Perspective: 95% Confidence Intervals", fontweight='bold', pad=20)
    ax.set_xticks(range(1, max_cores + 1))
    ax.legend(frameon=True, loc='upper left', fontsize='small')

    plt.tight_layout()
    output_file = plots_dir / 'Fig4_Gustafson_Weak_Scaling.pdf'
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()


def plot_chunk_optimization(chunk_results: dict, plots_dir: Path):
    print("\n[DATA VIZ] Generating Granularity Plot with Error Bands...")
    if not chunk_results: return

    plt.figure(figsize=(10, 6))

    sizes = sorted([int(k) for k in chunk_results.keys()])

    # STATISTICAL EXTRACTION: Converting dictionary values to NumPy arrays for vectorized operations
    means = np.array([chunk_results[str(s)]["mean"] for s in sizes])
    margins = np.array([chunk_results[str(s)]["ci_95_margin"] for s in sizes])

    # Plotting the primary empirical mean trajectory
    plt.semilogx(sizes, means, marker='D', color='#8172B3', linewidth=3, markersize=8, label='Empirical Mean')

    # Rendering a semi-transparent band to visualize the 95% Confidence Interval dispersion
    plt.fill_between(sizes, means - margins, means + margins, color='#8172B3', alpha=0.2,
                     label='95% Confidence Interval')

    # Identifying and annotating the mathematical optimum (efficiency sweet spot)
    min_time = min(means)
    optimal_size = sizes[np.argmin(means)]

    plt.annotate(f'Optimal Size: {optimal_size:,}', xy=(optimal_size, min_time),
                 xytext=(optimal_size, min_time + (max(means) * 0.1)),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                 horizontalalignment='center', fontweight='bold')

    plt.xlabel('Task Granularity (Chunk Size)', fontweight='bold')
    plt.ylabel('Total Processing Latency (Seconds)', fontweight='bold')
    plt.title('Granularity Profiling with Statistical Certainty', fontweight='bold', pad=20)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    output_file = plots_dir / 'Fig5_Granularity_Analysis.pdf'
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()



def generate_execution_time_plot(results_dict, sequential_time, output_path="execution_time_plot.pdf"):
    # configurations
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'pdf.fonttype': 42  # to enforce pdf fonts
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Dynamic palettes and markers
    colors = sns.color_palette("husl", n_colors=len(results_dict))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']

    # 1. Sequential Baseline (Only if value exists)
    if sequential_time is not None:
        ax.axhline(y=sequential_time, color='black', linestyle='--', linewidth=2,
                   label=f"Sequential Baseline ({sequential_time:.2f}s)")
    else:
        print("[WARNING] Plotting without sequential baseline.")

    # 2. Trend plot
    max_x = 1
    for i, (label, times) in enumerate(results_dict.items()):
        x_axis = np.arange(1, len(times) + 1)
        if len(times) > max_x: max_x = len(times)

        clean_label = label.replace("Amdahl_", "").replace("_", " ")

        ax.plot(x_axis, times,
                label=clean_label,
                color=colors[i],
                marker=markers[i % len(markers)],
                markersize=7, linewidth=2, alpha=0.9)

    ax.set_xlabel('Number of Workers (Cores / Threads)', fontweight='bold')
    ax.set_ylabel('Wall Clock Time (Seconds)', fontweight='bold')
    ax.set_title('Strong Scaling: Execution Time Analysis', pad=20, fontweight='bold')

    ax.set_xticks(range(1, max_x + 1))

    ax.legend(loc='upper right', frameon=True, shadow=False, ncol=2)

    ax.grid(True, which='both', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[DATA VIZ] Scientific execution time plot saved to: {output_path}")


