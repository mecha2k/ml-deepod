import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy import signal


plt.rcParams["figure.max_open_warning"] = 100


def check_graphs_v1(data, preds, threshold=None, name="default", piece=15):
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
    plt.ylim(0, threshold * 2)
    plt.xticks(np.arange(0, len(data), step=50000), rotation=45)
    plt.plot(xticks, data, color="green")
    plt.plot(xticks, preds, color="blue", linestyle="solid", linewidth=2, alpha=0.3)
    if threshold is not None:
        plt.axhline(y=threshold, color="red")
    plt.grid()
    plt.tight_layout()
    fig.savefig(f"{name}_all")


def check_graphs_v2(data, preds, anomaly, interval=10000, img_path=None, mode="train"):
    pieces = int(len(data) // interval)
    scale = 0.2 / anomaly.mean()
    for i in range(pieces):
        start = i * interval
        end = min(start + interval, len(data))
        xticks = range(start, end)
        values = data[start:end]
        plt.figure(figsize=(16, 8))
        plt.ylim(-0.25, 1.25)
        plt.xticks(np.arange(start, end, step=1000), rotation=45)
        plt.grid()
        plt.plot(xticks, values)
        plt.plot(xticks, preds[start:end], color="b", linewidth=6, alpha=0.5)
        plt.plot(xticks, anomaly[start:end] * scale, color="g", linewidth=6)
        plt.axhline(y=threshold * scale, color="r", linewidth=6)
        plt.savefig(img_path / f"{mode}_raw_data" / f"raw_{i+1:02d}_pages")


if __name__ == "__main__":
    data_path = Path("datasets/open")
    with open(data_path / "test.pkl", "rb") as f:
        test = pickle.load(f)
    with open(data_path / "scores.pkl", "rb") as f:
        scores = pickle.load(f)
    scores = scores.values.flatten()
    scores_f = signal.detrend(scores, type="linear")
    scores = scores_f - scores_f.min()

    anomaly_ratio = 12
    threshold = np.percentile(scores, 100 - anomaly_ratio)
    print(f"Anomaly threshold: {threshold}")
    prediction = np.zeros_like(scores)
    prediction[scores > threshold] = 1
    check_graphs_v1(scores, prediction, threshold, name="images/test_anomaly")
    check_graphs_v2(test.values, prediction, scores, img_path=Path("images"), mode="test")

    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    sample_submission["anomaly"] = prediction
    sample_submission.to_csv(data_path / "final_submission.csv", index=False)
    print(sample_submission["anomaly"].value_counts())
