import pandas as pd
import numpy as nm
import os
from typing import List

pwd = os.path.dirname(__file__)
normalize_max = 1
normalize_min = 0


class uniform_data:
    def __init__(self, data: nm.ndarray, lable):
        self.lable = lable
        self.data = nm.array(data)
        self.mean = nm.array(data.mean(0))
        self.cov = nm.cov(data, rowvar=False)

    def __len__(self):
        return self.data.shape[0]

    def __str__(self):
        return self.__str__()


def _normalize_data(data: nm.ndarray, min, max):
    _max, _min = data.max(0), data.min(0)
    k = (max - min) / (_max - _min)
    return max - k * (_max - data)


def load_train_data() -> (List[uniform_data], nm.ndarray):
    def categroy(data: pd.DataFrame, lable_len):
        uniform_data_list = []
        for i in range(lable_len):
            uniform_data_list.append(uniform_data(data[data.index == i], str(i)))

        return uniform_data_list

    data = _normalize_data(pd.read_csv(pwd + "/iris_training.csv", index_col="lable"), normalize_min,
                           normalize_max)  # type:pd.DataFrame

    return categroy(data, len(data.groupby("lable").count())), nm.array(data.mean())


def load_test_data():
    data = pd.read_csv(pwd + "/iris_test.csv", ) # type:pd.DataFrame
    lable=data["lable"]
    del data["lable"]

    return nm.array(_normalize_data(data,normalize_min,normalize_max)),nm.array(lable)

if __name__ == "__main__":
    a, u= load_train_data()
    test, lable=load_test_data()
    print(a)
    print(u)
    print(test)
    print(lable)

