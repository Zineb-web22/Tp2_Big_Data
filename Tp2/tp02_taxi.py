import os
import time
import psutil
import pandas as pd
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

FILE = "train.csv"
COL = "fare_amount"   # numeric column used for mean / sum

def mem_mb():
    # memory used by this python process
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

print("Total RAM (GB):", round(psutil.virtual_memory().total / 1024 ** 3, 2))
print("File size (GB):", round(os.path.getsize(FILE) / 1024 ** 3, 2))
results = {}

# Task 1: OOM test
print("\n--- Task 1: standard pandas ---")
print("RAM before:", round(mem_mb(), 1), "MB")
start = time.time()
try:
    df = pd.read_csv(FILE)
    print("loaded, mean =", df[COL].mean())
    results["Pandas"] = (time.time() - start, mem_mb())
except MemoryError as e:
    print("MemoryError:", e)
    print("RAM at crash:", psutil.virtual_memory().percent, "%")
    print("time before crash:", round(time.time() - start, 2), "s")
    results["Pandas"] = (None, None)

# Task 2: pandas chunking
print("\n--- Task 2: pandas chunking ---")
start = time.time()
total, count, peak = 0, 0, 0

# load data in chunks of 100k rows
for chunk in pd.read_csv(FILE, chunksize=100_000, usecols=[COL], dtype={COL: "float64"}):
    total += chunk[COL].sum()
    count += chunk[COL].count()
    peak = max(peak, mem_mb())

t2 = time.time() - start
print("sum =", total)
print("mean =", total / count)
print("time:", round(t2, 2), "s")
print("peak RAM:", round(peak, 1), "MB")
results["Chunking"] = (t2, peak)

# Task 3: dask
print("\n--- Task 3: dask ---")
start = time.time()

# nothing is read here, dask only builds the graph (lazy)
ddf = dd.read_csv(FILE, usecols=[COL], blocksize="64MB", dtype={COL: "float64"})
print("partitions:", ddf.npartitions)

mean_lazy = ddf[COL].mean()
sum_lazy = ddf[COL].sum()
print("type of result before compute:", type(mean_lazy))

# graph size before / after optimization (تعديل آمن لتجنب أخطاء dask-expr الحديثة)
try:
    graph_before = len(mean_lazy.__dask_graph__())
    (mean_opt,) = dask.optimize(mean_lazy)
    graph_after = len(mean_opt.__dask_graph__())
    print("graph tasks before:", graph_before, "| after:", graph_after)
except Exception as e:
    print("Graph optimization skipped (compatibility mode):", e)

try:
    mean_lazy.visualize(filename="dask_graph.png")  # needs graphviz
except Exception:
    print("graphviz not installed, skip visualize")

# compute both together so the file is read only once
with ProgressBar():
    mean_val, sum_val = dask.compute(mean_lazy, sum_lazy)

t3 = time.time() - start
print("sum =", sum_val)
print("mean =", mean_val)
print("time:", round(t3, 2), "s")
print("RAM after compute:", round(mem_mb(), 1), "MB")
results["Dask"] = (t3, mem_mb())

# summary
print("\n--- Summary ---")
print(f"{'Method':<10} {'Time (s)':<12} {'RAM (MB)':<12}")
for name, (t, m) in results.items():
    t_txt = "OOM" if t is None else round(t, 2)
    m_txt = "OOM" if m is None else round(m, 1)
    print(f"{name:<10} {t_txt!s:<12} {m_txt!s:<12}")

ok = {k: v for k, v in results.items() if v[0] is not None}
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(ok.keys(), [v[0] for v in ok.values()])
ax[0].set_title("Execution time (s)")
ax[1].bar(ok.keys(), [v[1] for v in ok.values()], color="orange")
ax[1].set_title("RAM (MB)")
plt.tight_layout()
plt.savefig("benchmark.png")
plt.show()