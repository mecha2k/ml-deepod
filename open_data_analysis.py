import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"{device} is available in torch")


def check_graphs_v1(data, labels, interval=10000, img_path=None):
    pieces = int(len(data) // interval)
    for i in range(pieces):
        start = i * interval
        end = min(start + interval, len(data))
        xticks = range(start, end)
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_ylim(-0.2, 1.2)
        ax1.set_xticks(np.arange(start, end, step=1000))
        ax1.grid()
        ax1.set_ylabel("Input Data")
        ax1.plot(xticks, data[start:end])
        ax2 = ax1.twinx()
        ax2.set_ylim(-0.2, 1.2)
        ax2.set_ylabel("Attack Labels")
        ax1.plot(xticks, labels[start:end], color="b", linewidth=6, alpha=0.5)
        fig.tight_layout()
        fig.savefig(img_path / f"swat_raw_data" / f"raw_{i+1:02d}_pages")


def check_graphs_v2(data, preds, threshold=None, name="default", piece=15):
    interval = len(data) // piece
    plt.rcParams["font.size"] = 24
    fig, axes = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        start = i * interval
        end = min(start + interval, len(data))
        xticks = range(start, end)
        axes[i].ticklabel_format(style="scientific", axis="both", scilimits=(0, 0))
        axes[i].set_xticks(np.arange(start, end, step=5000))
        axes[i].set_ylim(0, threshold * 2)
        axes[i].plot(xticks, data[start:end], color="g")
        axes[i].plot(xticks, preds[start:end], color="b")
        if threshold is not None:
            axes[i].axhline(y=threshold, color="r")
        axes[i].grid()
    plt.tight_layout()
    fig.savefig(name)

    plt.rcParams["font.size"] = 16
    fig = plt.figure(figsize=(12, 6))
    plt.ticklabel_format(style="scientific", axis="both", scilimits=(0, 0))
    xticks = range(0, len(data))
    plt.ylabel("Anomaly Score")
    plt.ylim(0, threshold * 2)
    plt.xticks(np.arange(0, len(data), step=50000), rotation=45)
    plt.plot(xticks, data_f, color="g")
    plt.plot(xticks, data, color="cyan", linewidth=2, alpha=0.3)
    plt.plot(xticks, preds, color="b", linestyle="solid", linewidth=2, alpha=0.3)
    if threshold is not None:
        plt.axhline(y=threshold, color="r", linewidth=2)
    plt.grid()
    plt.legend(["Anomaly Score (detrend)", "Anomaly Score", "Prediction", "Threshold"])
    plt.tight_layout()
    fig.savefig(f"{name}_all")


if __name__ == "__main__":
    data_path = Path("datasets/open")
    train_df = pickle.load(open(data_path / "train.pkl", "rb"))
    test_df = pickle.load(open(data_path / "test.pkl", "rb"))
    print(f"Shape of Train : {train_df.shape}, Test : {test_df.shape}")

    # find columns with std values less than 0.01
    train_cols = []
    train_std = train_df.std()
    for col in train_df.columns:
        if train_std[col] < 0.01:
            print(f"Column {col} has {train_std[col]} std values")
            train_cols.append(col)
    test_cols = []
    test_std = test_df.std()
    for col in test_df.columns:
        if test_std[col] < 0.01:
            print(f"Column {col} has {test_std[col]} std values")
            test_cols.append(col)
    plt.hist(train_std.values)
    plt.hist(test_std.values)
    plt.show()

    # check_graphs_v1(X_test, y_test, img_path=Path("images"))
