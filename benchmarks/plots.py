import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

pwd = Path(__file__).resolve().parent
data_path = (pwd / ".." / "data" / "processed" / "size_time.csv").resolve()
plots_path = (pwd / ".." / "results" / "figures").resolve()

data = pd.read_csv(data_path, index_col=False)


def size_time(data):
    plt.figure(figsize=(10, 5))
    for a, b in data.groupby("version"):
        x = b["lattice_size"]
        y = b["mean_time"]
        plt.plot(x, y, "o--", label=a)
    # plt.xlim(0, 2048)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.xlabel("lattic size")
    plt.ylabel("runtime")
    plt.savefig(plots_path / "size_time.png", dpi=600, bbox_inches='tight')
    plt.show()
    return 0

def log_size_time(data):
    plt.figure(figsize=(10, 5))
    for a, b in data.groupby("version"):
        x = np.log(b["lattice_size"])
        y = np.log(b["mean_time"])
        plt.plot(x, y, "o--", label=a)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.xlabel("log lattic size")
    plt.ylabel("log runtime")
    plt.savefig(plots_path / "log_size_time.png", dpi=600, bbox_inches='tight')
    plt.show()
    return 0

size_time(data)
log_size_time(data)