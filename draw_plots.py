import json
from pathlib import Path
from utils.plot_functions import (
    generate_plots,
    plot_amdahl_scaling,
    plot_gustafson_scaling,
    plot_chunk_optimization,
    generate_execution_time_plot
)


def main():
    print("=" * 60)
    print(" AVVIO GENERATORE GRAFICI DA FILE JSON")
    print("=" * 60)

    # 1. Definisci i percorsi
    root_dir = Path(__file__).parent.resolve()
    json_path = root_dir / "Outputs" / "Tables" / "Aggregated_Benchmark_Tables.json"
    plots_dir = root_dir / "Outputs" / "Plots"

    plots_dir.mkdir(parents=True, exist_ok=True)

    # 2. Carica i dati salvati
    if not json_path.exists():
        print(f"[ERRORE] File JSON non trovato in: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 3. Lancia tutte le funzioni di plotting
    # Fig 1 e Fig 2
    generate_plots(
        data.get("Synthetic_Workloads", {}),
        data.get("Real_Workloads", {}),
        data.get("IO_Stress_Tests", {}),
        plots_dir
    )

    scal_results = data.get("Scaling_Laws", {})
    if scal_results:
        # Fig 3: Amdahl
        if "Amdahl" in scal_results:
            plot_amdahl_scaling(scal_results["Amdahl"], plots_dir)

            # Estrazione dati per il Wall Clock Time di Amdahl
            scientific_data = {}
            base_seq_time = None
            for arch, core_data in scal_results["Amdahl"].items():
                times = [core_data[str(c)]["mean"] for c in sorted(core_data.keys(), key=int)]
                if arch == "Amdahl_Sequential":
                    base_seq_time = times[0]
                else:
                    scientific_data[arch] = times

            if base_seq_time is not None:
                generate_execution_time_plot(
                    scientific_data,
                    base_seq_time,
                    output_path=plots_dir / "Fig3b_Wall_Clock_Time_Scaling.pdf"
                )

        # Fig 4: Gustafson
        if "Gustafson" in scal_results:
            plot_gustafson_scaling(scal_results["Gustafson"], plots_dir)

        # Fig 5: Granularity
        if "Granularity" in scal_results:
            plot_chunk_optimization(scal_results["Granularity"], plots_dir)

    print("\n" + "=" * 60)
    print(" TUTTI I GRAFICI SONO STATI GENERATI CON SUCCESSO!")
    print(f" Li trovi nella cartella: {plots_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()