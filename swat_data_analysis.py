import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings

from pathlib import Path

from alembic.operations.toimpl import drop_column
from sklearn.preprocessing import MinMaxScaler
from deepod.models.time_series import AnomalyTransformer, TranAD, TimesNet

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"{device} is available in torch")


def make_swat_pickle(data_name, data_path):
    for name in data_name:
        if name.split(".")[-1] == "csv":
            data_df = pd.read_csv(data_path / f"{name}")
        else:
            data_df = pd.read_excel(data_path / f"{name}")
        print(data_df.shape)
        print(data_df.head())
        name = name.split(".")[0]
        data_df.to_pickle(data_path / f"{name}.pkl")


def make_train_test_data(data_path):
    normal_v1 = pd.read_pickle(data_path / "SWaT_Dataset_Normal_v1.pkl")
    index = normal_v1["Unnamed: 0"][1:].values
    columns = normal_v1.iloc[0].values
    normal_v1 = normal_v1.drop("Unnamed: 0", axis=1)
    normal_v1 = pd.DataFrame(normal_v1[1:].values, index=index, columns=columns[1:])
    train_df = normal_v1.drop("Normal/Attack", axis=1)
    assert train_df.isna().sum().sum() == 0

    scaler = MinMaxScaler()
    train_df = pd.DataFrame(
        scaler.fit_transform(train_df), index=train_df.index, columns=train_df.columns
    )
    print(train_df.head())

    # find columns with std values less than 0.01
    # drop_columns = []
    # deviation = train_df.std()
    # for col in train_df.columns:
    #     if deviation[col] < 0.01:
    #         print(f"Column {col} has {deviation[col]} std values")
    #         drop_columns.append(col)
    # plt.hist(deviation.values)
    # plt.show()

    with open(data_path / "train.npy", "wb") as f:
        np.save(f, train_df.values)
    print("train data shape : ", train_df.shape)

    attack_v0 = pd.read_pickle(data_path / "SWaT_Dataset_Attack_v0.pkl")
    index = attack_v0["Unnamed: 0"][1:].values
    columns = attack_v0.iloc[0].values
    attack_v0 = attack_v0.drop("Unnamed: 0", axis=1)
    attack_v0 = pd.DataFrame(attack_v0[1:].values, index=index, columns=columns[1:])

    # create new column with name "labels" in attack_v0
    attack_v0["labels"] = 0
    attack_v0["labels"] = attack_v0["Normal/Attack"].apply(lambda x: 0 if x == "Normal" else 1)
    print(attack_v0["labels"].value_counts())
    test_df = attack_v0.drop(["Normal/Attack", "labels"], axis=1)
    assert test_df.isna().sum().sum() == 0
    test_df = pd.DataFrame(test_df, index=test_df.index, columns=train_df.columns)
    test_df = pd.DataFrame(scaler.transform(test_df), index=test_df.index, columns=test_df.columns)
    print(test_df.head())

    with open(data_path / "test.npy", "wb") as f:
        np.save(f, test_df.values)
    with open(data_path / "test_labels.npy", "wb") as f:
        np.save(f, attack_v0["labels"].values)
    print("test data shape : ", test_df.shape)


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


if __name__ == "__main__":
    data_path = Path("datasets/SWaT")
    data_name = [
        "SWaT_Dataset_Normal_v1.xlsx",
        "SWaT_Dataset_Attack_v0.xlsx",
        "SWaT_train.csv",
        "SWaT_train2.csv",
        "SWaT_raw.csv",
        "swat2.csv",
    ]

    # make_swat_pickle(data_name, data_path)
    # make_train_test_data(data_path)

    with open(data_path / "train.npy", "rb") as f:
        X_train = np.load(f, allow_pickle=True)
    with open(data_path / "test.npy", "rb") as f:
        X_test = np.load(f, allow_pickle=True)
    with open(data_path / "test_labels.npy", "rb") as f:
        y_test = np.load(f, allow_pickle=True)
    print(f"Shape of Train : {X_train.shape}, Test : {X_test.shape}, Labels : {y_test.shape}")

    check_graphs_v1(X_test, y_test, img_path=Path("images"))

    # data_path = Path("datasets/open")
    # train_df = pickle.load(open(data_path / "train.pkl", "rb"))
    # test_df = pickle.load(open(data_path / "test.pkl", "rb"))
    # X_train = train_df.values
    # X_test = test_df.values
    # print(X_train.shape, X_test.shape)
    # print(train_df.describe())
