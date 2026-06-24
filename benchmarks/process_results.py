import pandas as pd
from pathlib import Path

pwd = Path(__file__).resolve().parent
raw_folder = (pwd / ".." / "data" / "raw").resolve()

data = pd.DataFrame(columns=["version", "lattice_size", "mean_time"])

benchmarking_results = pd.read_csv(raw_folder / "benchmarking_results.csv", index_col=False)
cupy = pd.read_csv(raw_folder / "cupy.csv", index_col=False)
numba_cuda = pd.read_csv(raw_folder / "numba_cuda.csv", index_col=False)
pytorch_gpu = pd.read_csv(raw_folder / "pytorch_gpu.csv", index_col=False)
julia = pd.read_csv(raw_folder / "julia.csv", index_col=False)
julia_opt = pd.read_csv(raw_folder / "julia_opt.csv", index_col=False)
julia_cuda = pd.read_csv(raw_folder / "julia_cuda.csv", index_col=False)
julia_parallel = pd.read_csv(raw_folder / "julia_parallel.csv", index_col=False)



benchmarking_results = benchmarking_results.drop('cupy', axis=1)
benchmarking_results = benchmarking_results.drop('vectorized_pytorch_gpu', axis=1)
benchmarking_results = benchmarking_results.drop('numba_cuda', axis=1)
benchmarking_results = benchmarking_results.drop('julia', axis=1)
benchmarking_results = benchmarking_results.drop('julia_opt', axis=1)
benchmarking_results = benchmarking_results.drop('julia_parallel', axis=1)
benchmarking_results = benchmarking_results.drop('julia_cuda', axis=1)


benchmarking_results = benchmarking_results.loc[:, ~benchmarking_results.columns.str.contains('^Unnamed')]

row = 0
for a, b in benchmarking_results.groupby("lattice_size"):
    for version in b.columns[4:]:
        data.loc[row] = [version, a, b[version].mean()]
        row += 1


cupy = cupy.loc[:, ~cupy.columns.str.contains('^Unnamed')]
for a, b in cupy.groupby("lattice_size"):
    for version in b.columns[4:]:
        data.loc[row] = [version, a, b[version].mean()]
        row += 1

numba_cuda = numba_cuda.loc[:, ~numba_cuda.columns.str.contains('^Unnamed')]
for a, b in numba_cuda.groupby("lattice_size"):
    for version in b.columns[4:]:
        data.loc[row] = [version, a, b[version].mean()]
        row += 1

pytorch_gpu = pytorch_gpu.loc[:, ~pytorch_gpu.columns.str.contains('^Unnamed')]
for a, b in pytorch_gpu.groupby("lattice_size"):
    for version in b.columns[4:]:
        data.loc[row] = [version, a, b[version].mean()]
        row += 1


for i in julia.index:
    data.loc[row] = ["julia", int(julia.loc[i, "L"]), float(julia.loc[i, "mean_ms"])/1000]
    row += 1

for i in julia_opt.index:
    data.loc[row] = ["julia_opt", int(julia_opt.loc[i, "L"]), float(julia_opt.loc[i, "mean_ms"])/1000]
    row += 1

for i in julia_parallel.index:
    data.loc[row] = ["julia_parallel", int(julia_parallel.loc[i, "L"]), float(julia_parallel.loc[i, "mean_ms"])/1000]
    row += 1

for i in julia_cuda.index:
    data.loc[row] = ["julia_cuda", int(julia_cuda.loc[i, "L"]), float(julia_cuda.loc[i, "mean_ms"])/1000]
    row += 1

result_path = (pwd / ".." / "data" / "processed").resolve()

data.to_csv(result_path/"size_time.csv", index=False)

for implt, b in data.groupby("version"):
    if implt == "python":
        t_128_p = b.loc[b["lattice_size"] == 128, "mean_time"].item()
        print(t_128_p)

for implt, b in data.groupby("version"):
    t_128 = b.loc[b["lattice_size"] == 128, "mean_time"].item()
    print(implt, round(t_128, 2), round(t_128_p/t_128, 1))


julia_parallel_thread = pd.DataFrame(columns=["threads", "runtime", "speedup", "efficiency"])
row = 0
for t in [1, 2, 4, 8]:
    temp = pd.read_csv(raw_folder / "pgg_julia_parallel_thread_{}.csv".format(t), index_col=False)
    julia_parallel_thread.loc[row, "threads"] = t
    julia_parallel_thread.loc[row, "runtime"] = temp.loc[0, "mean_ms"].item()/1000
    row += 1
julia_parallel_thread["speedup"] = julia_parallel_thread["runtime"][0]/julia_parallel_thread["runtime"]
julia_parallel_thread["efficiency"] = julia_parallel_thread["speedup"]/julia_parallel_thread["threads"]
print(julia_parallel_thread)
julia_parallel_thread.to_csv(result_path/"julia_parallel_thread.csv", index=False)