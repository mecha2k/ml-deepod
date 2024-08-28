import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy import signal
from sympy.abc import alpha

plt.rcParams["figure.max_open_warning"] = 100


def check_graphs_v1(data, data_f, preds, threshold=None, name="default", piece=15):
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
    plt.ylim(0, data.mean() * 2)
    plt.xticks(np.arange(0, len(data), step=50000), rotation=45)
    plt.plot(xticks, data_f, color="g")
    plt.plot(xticks, data, color="cyan", linewidth=2, alpha=0.3)
    plt.plot(xticks, preds, color="b", linestyle="solid", linewidth=2, alpha=0.3)
    if threshold is not None:
        plt.axhline(y=threshold, color="r")
    plt.grid()
    plt.legend(["Anomaly Score (detrend)", "Anomaly Score", "Prediction"])
    plt.tight_layout()
    fig.savefig(f"{name}_all")


def check_graphs_v2(data, preds, anomaly, threshold, interval=10000, img_path=None, mode="train"):
    pieces = int(len(data) // interval)
    for i in range(pieces):
        start = i * interval
        end = min(start + interval, len(data))
        xticks = range(start, end)
        values = data[start:end]
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_ylim(-0.2, 1.2)
        ax1.set_xticks(np.arange(start, end, step=1000))
        ax1.grid()
        ax1.set_ylabel("Input Data")
        ax1.plot(xticks, values)
        ax1.plot(xticks, preds[start:end], color="b", linewidth=6, alpha=0.5)
        ax2 = ax1.twinx()
        ax2.set_ylim(0, threshold * 2)
        ax2.set_ylabel("Anomaly Score")
        ax2.plot(xticks, anomaly[start:end], color="g", linewidth=6)
        ax2.axhline(y=threshold, color="r", linewidth=6)
        ax2.legend(["Anomaly Score", "Threshold"])
        fig.tight_layout()
        fig.savefig(img_path / f"{mode}_raw_data" / f"raw_{i+1:02d}_pages")


def find_anomaly(scores, anomaly_ratio=2):
    scores_f = signal.detrend(scores, type="linear")
    scores_f = scores_f - scores_f.min()

    threshold = np.percentile(scores_f, 100 - anomaly_ratio)
    print(f"Anomaly threshold: {threshold}")
    prediction = np.zeros_like(scores_f)
    prediction[scores_f > threshold] = 1
    check_graphs_v1(scores, scores_f, prediction, threshold, name="images/test_anomaly")
    check_graphs_v2(
        test.values, prediction, scores, threshold, img_path=Path("images"), mode="test"
    )
    return prediction


if __name__ == "__main__":
    data_path = Path("datasets/open")
    with open(data_path / "test.pkl", "rb") as f:
        test = pickle.load(f)
    with open(data_path / "test_scores.pkl", "rb") as f:
        scores = pickle.load(f)
    scores = scores.values.flatten()

    prediction = find_anomaly(scores, anomaly_ratio=2)
    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    sample_submission["anomaly"] = prediction
    sample_submission.to_csv(data_path / "final_submission.csv", index=False)
    print(sample_submission["anomaly"].value_counts())
