import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


def check_graphs_v1(data, preds, threshold=None, name="default", piece=1):
    interval = len(data) // piece
    for i in range(piece):
        start = i * interval
        end = min(start + interval, len(data))
        plt.figure(figsize=(12, 6))
        plt.ylim(0, threshold * 2)
        plt.plot(data[start:end], color="blue")
        plt.plot(preds[start:end], color="green")
        if threshold is not None:
            plt.axhline(y=threshold, color="red")
        plt.tight_layout()
        plt.savefig(f"{name}_{i:02d}")


if __name__ == "__main__":
    data_path = Path("datasets")
    with open("datasets/scores.pkl", "rb") as f:
        scores = pickle.load(f)

    anomaly_ratio = 10
    threshold = np.percentile(scores, 100 - anomaly_ratio)
    print(f"Anomaly threshold: {threshold}")
    prediction = np.zeros_like(scores)
    prediction[scores > threshold] = 1
    check_graphs_v1(scores, prediction, threshold, name="images/test_anomaly")

    sample_submission = pd.read_csv("datasets/sample_submission.csv")
    sample_submission["anomaly"] = prediction
    sample_submission.to_csv("datasets/final_submission.csv", index=False)
    print(sample_submission["anomaly"].value_counts())
