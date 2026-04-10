import time
import multiprocessing as mp


def null_task(x):
    return x


if __name__ == "__main__":
    data = ["test"] * 1000  # a typical chunk
    with mp.Pool(1) as pool:
        start = time.perf_counter()
        pool.map(null_task, [data])
        end = time.perf_counter()

    overhead = (end - start) * 1000  # ms
    print(f"Overhead IPC misurato: {overhead:.4f} ms")