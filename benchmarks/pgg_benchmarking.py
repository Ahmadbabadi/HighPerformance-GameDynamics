import time
import pandas as pd
from pathlib import Path
import numpy as np
import random
import torch
import sys
import os


def run_pure_python(L, M, Sr, seed):
    from pgg_python import PggPython
    random.seed(seed)
    start = time.perf_counter()
    sim = PggPython(L, M, Sr)
    sim.run_simulation(1000)
    end = time.perf_counter()
    return end-start

def run_numpy(L, M, Sr, seed):
    from pgg_numpy import PggNumpy
    np.random.seed(seed)
    start = time.perf_counter()
    sim = PggNumpy(L, M, Sr)
    sim.run_simulation(1000)
    end = time.perf_counter()
    return end-start

def run_vectorized_numpy(L, M, Sr, seed):
    from pgg_vectorized_numpy import PggVectorizedNumpy
    np.random.seed(seed)
    start = time.perf_counter()
    sim = PggVectorizedNumpy(L, M, Sr)
    sim.run_simulation(1000)
    end = time.perf_counter()
    return end-start

def run_vectorized_pytorch(L, M, Sr, seed):
    from pgg_vectorized_pytorch import PggVectorizedPyTorch
    torch.manual_seed(seed)
    start = time.perf_counter()
    sim = PggVectorizedPyTorch(L, M, Sr)
    sim.run_simulation(1000)
    end = time.perf_counter()
    return end-start

def run_numba(L, M, Sr, seed):
    from pgg_numba import run_simulation, init_simulation
    np.random.seed(seed)
    start = time.perf_counter()
    state = init_simulation(L, M, Sr)
    sim = state = run_simulation(state, 1000)
    end = time.perf_counter()
    return end-start


if __name__ == "__main__":
    pwd = Path(__file__).resolve().parent
    jobs_path = (pwd / ".." / "configs" / "benchmarking_jobs.csv").resolve()
    benchmarking_jobs = pd.read_csv(jobs_path, index_col=False)

    result_path = (pwd / ".." / "data" / "raw" / "benchmarking_results.csv").resolve()

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    # versions = ["pgg_python", "pgg_numpy", "pgg_numba", "pgg_vectorized_numpy",
    #             "pgg_vectorized_pytorch", "pgg_vectorized_pytorch_gpu",
    #             "pgg_cupy", "pgg_numba_cuda", "julia"]
    implementions = ["python", "numpy", "vectorized_numpy",
                     "vectorized_pytorch", "numba"]
    fucntions = [run_pure_python, run_numpy, run_vectorized_numpy, run_vectorized_pytorch, run_numba]
    
    for implt, func in zip(implementions, fucntions):
        for row in benchmarking_jobs.index:
            seed, L, M, Sr = benchmarking_jobs.loc[row, ["ensumbles", "lattice_size", "memory_length", "synergy_rate"]]
            # if L > 1024:
            #     continue
            if benchmarking_jobs.loc[row, implt] != benchmarking_jobs.loc[row, implt]:
                runtime = func(int(L), int(M), float(Sr), int(seed))
                benchmarking_jobs.loc[row, implt] = runtime
                benchmarking_jobs.to_csv(result_path, index=False)
                print(implt, int(L), int(M), float(Sr), int(seed), runtime)
            # print(benchmarking_jobs.loc[row, implt])