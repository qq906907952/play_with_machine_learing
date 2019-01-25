import pandas as pd
import numpy as nm
import os

pwd = os.path.dirname(__file__)


train_set_num = 300


def load_data(is_train)->(nm.ndarray,nm.ndarray):
    # data set is from https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx
    data = pd.read_excel(pwd + "/data_set.xlsx", )  # type:pd.DataFrame
    data.drop("No", 1)
    data.drop("X1 transaction date", 1)

    if is_train:
        return nm.array(data[:train_set_num].drop("Y house price of unit area", 1)), nm.array(data[:train_set_num][
            "Y house price of unit area"])
    else:
        return nm.array(data[train_set_num:].drop("Y house price of unit area", 1)),(nm.array(data[train_set_num:][
            "Y house price of unit area"]))


if __name__ == "__main__":
    print(load_data(True))
