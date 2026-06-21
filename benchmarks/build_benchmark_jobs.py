from pathlib import Path
import pandas as pd

def build_benchmarking_jobs(path):
    versions = ["python", "numpy", "numba", "vectorized_numpy",
                "vectorized_pytorch", "vectorized_pytorch_gpu",
                "cupy", "numba_cuda", "julia", "julia_opt", "julia_parallel", "julia_cuda"]
    benchmarking_jobs = pd.DataFrame(columns=["ensumbles", "lattice_size", "memory_length", "synergy_rate"]+versions )
    lattice_sizes = [2**n for n in range(5, 12)]
    memory_length = 15
    synergy_rate = 3
    ensumbles = 10
    n = 0
    for L in lattice_sizes:
        for ens in range(ensumbles):
            benchmarking_jobs.loc[n] = [ens+1, L, memory_length, synergy_rate] + [None for i in range(len(versions))]
            n += 1

    benchmarking_jobs.to_csv(path, index=False)
    return benchmarking_jobs


base_dir = Path(__file__).resolve().parent
target_path = (base_dir / ".." / "configs" / "benchmarking_jobs.csv").resolve()

build_benchmarking_jobs(target_path)
