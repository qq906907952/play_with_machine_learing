import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as nm
from linear_regression import load_data


def main():
    data, y = load_data.load_data(True)
    data = nm.append(data, nm.ones([data.shape[0], 1]), axis=1)
    # loss=(y-data @ w).T @ (y-data @ w)
    # d(loss)/d(w) = 2(y-data@w)@(data) = 0
    # w=(data.T @ data)-1 @ data.T @ y          -1=inverse
    w = nm.linalg.inv(data.T @ data) @ data.T @ y
    test_data, test_y = load_data.load_data(False)
    test_data = nm.append(test_data, nm.ones([test_data.shape[0], 1]), axis=1)
    loss=(test_y - test_data @ w).T@(test_y - test_data @ w)
    print("loss:{}".format(loss))


if __name__ == "__main__":
    main()
