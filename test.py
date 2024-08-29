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

data_path = Path("datasets/SWaT")
data_name = [
    "SWaT_Dataset_Normal_v1",
    "SWaT_Dataset_Attack_v0",
    "SWaT_train",
    "SWaT_train2",
    "SWaT_raw",
    "swat2",
]

normal_v1 = pd.read_pickle(data_path / "SWaT_Dataset_Normal_v1.pkl")
print(normal_v1.shape)
index = normal_v1["Unnamed: 0"][1:].values
columns = normal_v1.iloc[0].values
normal_v1 = normal_v1.drop("Unnamed: 0", axis=1)
norm1 = normal_v1[1:].values
normal_v1 = pd.DataFrame(normal_v1[1:].values, index=index, columns=columns[1:])
print(normal_v1.shape)
print(normal_v1.head())
print(normal_v1.tail())
print(normal_v1["Normal/Attack"].value_counts())
print(normal_v1.describe())
print("=" * 100)

attack_v0 = pd.read_pickle(data_path / "SWaT_Dataset_Attack_v0.pkl")
print(attack_v0.shape)
index = attack_v0["Unnamed: 0"][1:].values
columns = attack_v0.iloc[0].values
attack_v0 = attack_v0.drop("Unnamed: 0", axis=1)
attack_v0 = pd.DataFrame(attack_v0[1:].values, index=index, columns=columns[1:])
print(attack_v0.shape)
print(attack_v0.head())
print(attack_v0.tail())
print(attack_v0["Normal/Attack"].value_counts())
print(attack_v0.describe())

# for name in data_name:
#     train_df = pd.read_pickle(data_path / f"{name}.pkl")
#     print(train_df.shape)
#     print(train_df.head())
#     break


# index = normal_v1["Unnamed: 0"][1:].values
# columns = normal_v1.iloc[0].values
# print(len(columns), columns)
# normal_v1 = normal_v1.drop("Unnamed: 0", axis=1)
# print(normal_v1.shape)
# print(normal_v1.head())
# normal_v1 = pd.DataFrame(normal_v1.values, index=index, columns=columns)
# print(normal_v1.shape)
# print(normal_v1.head())


# train_df = pd.read_csv(data_path / "SWaT_train.csv")
# print(train_df.shape)
# print(train_df.head())
# train_df.to_pickle(data_path / "SWaT_train.pkl")
# train_df = pd.read_csv(data_path / "SWaT_train2.csv")
# print(train_df.shape)
# print(train_df.head())
# train_df.to_pickle(data_path / "SWaT_train2.pkl")
# train_df = pd.read_csv(data_path / "SWaT_raw.csv")
# print(train_df.shape)
# print(train_df.head())
# train_df.to_pickle(data_path / "SWaT_raw.pkl")
# train_df = pd.read_csv(data_path / "swat2.csv")
# print(train_df.shape)
# print(train_df.head())
# train_df.to_pickle(data_path / "swat2.pkl")
# train_df = pd.read_excel(data_path / "SWaT_Dataset_Normal_v1.xlsx")
# print(train_df.shape)
# print(train_df.tail())
# train_df.to_pickle(data_path / "SWaT_Dataset_Normal_v1.pkl")
# train_df = pd.read_excel(data_path / "SWaT_Dataset_Attack_v0.xlsx")
# print(train_df.shape)
# print(train_df.tail())
# train_df.to_pickle(data_path / "SWaT_Dataset_Attack_v0.pkl")


data_path = Path("datasets/open")
train_df = pickle.load(open(data_path / "train.pkl", "rb"))
test_df = pickle.load(open(data_path / "test.pkl", "rb"))
X_train = train_df.values
X_test = test_df.values
print(X_train.shape, X_test.shape)
print(train_df.describe())
#
# clf = AnomalyTransformer(device=device)
# clf.fit(X_train)
#
# train_scores = clf.decision_function(X_train)
# train_scores = pd.DataFrame(train_scores, index=train_df.index)
# train_scores.to_pickle(data_path / "train_scores.pkl")
#
# test_scores = clf.decision_function(X_test)
# test_scores = pd.DataFrame(test_scores, index=test_df.index)
# test_scores.to_pickle(data_path / "test_scores.pkl")

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
