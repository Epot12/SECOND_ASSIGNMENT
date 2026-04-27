# Second Project of Parallel Computing course:
# Bloom Filter

## Overview
This repository contains a high-performance framework implementing a Bloom Filter, developed as Final assignment for the Parallel Computing course. The project is designed to evaluate computational efficiency, strong/weak scaling (Amdahl's and Gustafson's Laws), and granularity optimization through Python frameworks (multiprocessing, multithreading, coroutines) and Python orchestration.

## Technical Requirements

### 1. Operating System
* **Primary Environment:** Microsoft Windows 10 Pro 10.0.19045 N/D build 19045.
* *Note:* The software is natively built for Windows 10 Pro (x86_64) but maintains macOS and Linux compatibility via dynamic path and interpreter resolution. However, performance and scaling profiles for Multiprocessing IPC may vary across platforms due to the fundamental architectural differences between POSIX fork and Windows spawn process generation mechanisms.

### 2. Execution Framework (`uv`)
To guarantee strict scientific reproducibility and deterministic dependency resolution, this project employs **`uv`**, an ultra-fast Python package and project manager.

The benchmarking suite relies on `uv` to autonomously orchestrate distinct, isolated virtual environments (e.g., standard Python 3.12 and Free-Threaded Python 3.13t) without interfering with the host system's global configuration.

If `uv` is not present on your system, install it via the official channels:
* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows (PowerShell):** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

---

## Repository Acquisition & Setup

To reproduce the experimental benchmarking suite locally, the initial step is to clone this repository to your local host machine. Ensure that a modern version of `git` is installed and accessible via your system's PATH.

Execute the following commands in your terminal to clone the repository and navigate into the project's root directory:

```bash
# 1. Clone the repository to your local machine
git clone [https://github.com/Epot12/SECOND_ASSIGNMENT.git](https://github.com/Epot12/SECOND_ASSIGNMENT.git)

# 2. Navigate into the designated project directory
cd SECOND_ASSIGNMENT
```

## Dataset Acquisition & Preparation

This project utilizes real-world web topological data sourced from the **Common Crawl** index (Snapshot: `CC-MAIN-2024-10`). Due to the substantial payload size (~5.5 GB uncompressed) and restrictive AWS S3 bucket policies, the dataset must be downloaded and extracted prior to pipeline execution.

### Quick Start: Automated Data Setup
Execute the following commands from the project root (`SECOND_ASSIGNMENT`) to autonomously fetch and extract the exact data shard required for empirical evaluation:

**1. Create the Local Data Directory**
```bash
mkdir DATA
```

**2. Download the Compressed Index**

```bash
curl -L -o DATA/cdx-00000.gz [https://data.commoncrawl.org/cc-index/collections/CC-MAIN-2024-10/indexes/cdx-00000.gz](https://data.commoncrawl.org/cc-index/collections/CC-MAIN-2024-10/indexes/cdx-00000.gz)
```

**3. Extract the Dataset**
The Python pre-processing engines strictly require standard plaintext (.txt or raw JSONL). You must decompress the archive.

- Linux / macOS / WSL:

```bash
gunzip DATA/cdx-00000.gz
```

- Windows (PowerShell):

```bash
tar -xvzf DATA/cdx-00000.gz -C DATA/
```

(Critical: Ensure the extracted file is named exactly cdx-00000 without any file extension).

**4. Data Processing Pipeline**
Once the raw file is extracted, invoke the Python parsing scripts to validate integrity and isolate the target URLs into a streamlined text file (common_crawl_FULL.txt):

```bash
# Validate CDXJ Format and File Integrity
uv run python DATA_MANAGEMENT/validate_cc_dataset.py

# Extract URL entities for High-Performance Benchmarking
uv run python DATA_MANAGEMENT/prepare_full_dataset.py
```

## Execution Directives
The experimental framework is governed by a central orchestrator (General_Main.py). This executive script dynamically resolves dependencies defined in uv.lock, provisions the required GIL/No-GIL environments, delegates tasks to concurrent subsystems, and aggregates telemetry data.

### Comprehensive Execution
To sequentially execute all experimental phases and automatically generate benchmark plots/PDFs, run:

```bash
uv run python General_Main.py
```
a help menu will be displayed showing all the possible options, that are

### Execution Flags and Module Orchestration

The central orchestrator (`General_Main.py`) employs a modular CLI architecture, allowing researchers to trigger specific experimental subsets or execute the comprehensive benchmarking pipeline. 

The following table details the available execution flags and their underlying methodological implementations:

| Execution Flag | Target Module | Methodological Description |
| :--- | :--- | :--- |
| `--all` | **Comprehensive Suite** | Executes the entirety of the benchmarking pipeline. It sequentially triggers all underlying sub-orchestrators, aggregates the resulting telemetry data into a consolidated master JSON file, and automatically generates the complete set of analytical plots. |
| `--base` | **Baseline Benchmarks** | Evaluates the foundational sequential and parallel processing architectures. This module provides a comparative empirical baseline between standard Python environments (constrained by the Global Interpreter Lock) and free-threaded (No-GIL) environments. |
| `--io` | **I/O Stress Testing** | Orchestrates high-throughput, asynchronous data ingestion simulations. It evaluates the system's capacity to overlap I/O-bound tasks (e.g., network latency or memory-mapped file reading) with CPU-bound hashing operations to measure absolute throughput. |
| `--scaling` | **Scaling Laws Analysis** | Conducts rigorous scalability profiling using dynamically allocated worker pools. It empirically measures Strong Scaling (Amdahl's Law), Weak Scaling (Gustafson's Law), and thread workload granularity (chunk size optimization). |
| `--stripe` | **Memory Strategy Profiling** | Investigates micro-architectural memory access patterns and vectorization efficiencies. It benchmarks contiguous memory alignment strategies, contrasting Array-of-Structures (AoS) against optimized Structure-of-Arrays (SoA) layouts. |

> **Usage Example:** Flags can be combined for targeted empirical evaluation. For instance, executing `uv run python General_Main.py --scaling --io` will bypass the baseline and memory pattern tests, exclusively compiling telemetry for scalability and I/O bottlenecks.


### Micro-Architectural Profiling and Execution Tracing

To rigorously analyze the computational overhead and latency distributions of the different Bloom Filter implementations, the project includes an autonomous profiling orchestrator. 

This module empirically evaluates three distinct architectural paradigms:
1. **Sequential:** Baseline execution constrained by the Global Interpreter Lock (GIL).
2. **Multiprocessing (IPC):** Shared-memory parallelization executed within a standard GIL environment.
3. **Multithreading (SoA):** SIMD-optimized Structure-of-Arrays execution leveraging the Free-Threaded (No-GIL) Python 3.13t interpreter.

#### Orchestrated Execution
To launch the automated profiling suite, execute the showcase script from the project root:

```bash
uv run python profiling_showcase.py
```

## Automated Unit Testing & Theoretical Validation

To guarantee the mathematical soundness and structural integrity of the various Scalable Bloom Filter architectures, this project incorporates an automated testing suite utilizing the `pytest` framework. 

The validation module (`TESTS/test_bloom_filters.py`) employs a parametrized testing architecture to subject every implementation (Sequential, Multi-threaded, and Multi-processed) to the same strict empirical constraints.

### Validation Criteria
The test suite rigorously evaluates three fundamental rules governing Scalable Bloom Filters:
1. **Zero False Negatives (`test_no_false_negatives`):** Ensures absolute retrieval accuracy. If an element has been inserted into the filter, the data structure must definitively return a positive membership query.
2. **Empirical FPR Bounding (`test_empirical_false_positive_rate`):** Validates that the empirical False Positive Rate (FPR) remains strictly bounded and does not statistically deviate beyond the theoretical target limit (accounting for acceptable hash distribution margins) when subjected to dense data saturation.
3. **Dynamic Scaling Architecture (`test_scaling_behavior`):** Confirms the dynamic capacity logic. The test forcefully overflows the initial capacity bounds to verify the autonomous instantiation of secondary filter layers.

### Test Execution
To initiate the full validation suite, execute the following command from the project root directory. We recommend utilizing the `-v` (verbose) flag to monitor the real-time execution of each parametrized architectural variant.

```bash
uv run pytest TESTS/test_bloom_filters.py -v
```


