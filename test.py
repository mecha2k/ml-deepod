import torch
import numpy as np
import pandas as pd
import pickle
import warnings

from pathlib import Path
from deepod.models.time_series import AnomalyTransformer, TranAD, TimesNet

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"{device} is available in torch")

data_path = Path("datasets/open")
train_df = pickle.load(open(data_path / "train.pkl", "rb"))
test_df = pickle.load(open(data_path / "test.pkl", "rb"))
X_train = train_df.values
X_test = test_df.values
print(X_train.shape, X_test.shape)

clf = AnomalyTransformer(device=device)
clf.fit(X_train)

train_scores = clf.decision_function(X_train)
train_scores = pd.DataFrame(train_scores, index=train_df.index)
train_scores.to_pickle(data_path / "train_scores.pkl")

test_scores = clf.decision_function(X_test)
test_scores = pd.DataFrame(test_scores, index=test_df.index)
test_scores.to_pickle(data_path / "test_scores.pkl")

# # evaluation of time series anomaly detection
# from deepod.metrics import ts_metrics
# from deepod.metrics import (
#     point_adjustment,
# )  # execute point adjustment for time series ad
#
# data_path = Path("datasets/SMD")
# with open(data_path / "SMD_train.npy", "rb") as f:
#     train = np.load(f)
# with open(data_path / "SMD_test.npy", "rb") as f:
#     test = np.load(f)
# with open(data_path / "SMD_test_label.npy", "rb") as f:
#     labels = np.load(f)
# print(train.shape, test.shape, labels.shape)
#
# clf = AnomalyTransformer(device=device)
# clf.fit(train)
#
# scores = clf.decision_function(test)
# # eval_metrics = ts_metrics(labels, scores)
# adj_eval_metrics = ts_metrics(labels, point_adjustment(labels, scores))
# print(adj_eval_metrics)
# # results : roc_auc_score, average_precision_score, best_f1, best_p, best_r
# # (0.76625, 0.26313, 0.30726, 0.42068, 0.24201)
