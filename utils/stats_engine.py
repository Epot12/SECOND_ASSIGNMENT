import math
import stats_engine


# STATISTICAL ENGINE CONSTANTS

WARMUP_RUNS = 2
MEASURED_RUNS = 5
TOTAL_RUNS = WARMUP_RUNS + MEASURED_RUNS

def compute_statistics(times: list) -> dict:
    """
    Calculate Mean and 95% Confidence Interval margin using T-Student for small samples.
    """
    n = len(times)
    mean_val = statistics.mean(times)
    if n > 1:
        stdev = statistics.stdev(times)
        sem = stdev / math.sqrt(n)
        t_critical = 2.776 #value coming from t student table
        margin = t_critical * sem
    else:
        margin = 0.0
    return {
        "mean": round(mean_val, 6),
        "ci_95_margin": round(margin, 6)
    }