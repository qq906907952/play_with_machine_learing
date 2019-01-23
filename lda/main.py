import sys, os

sys.path.insert(0, os.path.dirname(__file__))
import load_data
import numpy as nm


def main():
    data, u = load_data.load_train_data()  # u is a vector of all data mean
    if len(data) == 0:
        raise RuntimeError("load data fail")

    # 广义锐利商 (w.T @ Sa @ w)/(w.T @ Sb @ w)
    sa, sb = nm.zeros(data[0].cov.shape), nm.zeros(data[0].cov.shape)
    for i in data:
        sa += len(i) * ((i.mean - u).reshape(len(u), 1) @ ((i.mean - u).reshape(1, len(u))))
        sb += i.cov

    # 广义锐利最大值 Sb-1 @ sa 最大特征值对应特征向量（可用拉格郎日乘法证明（周志华著 机器学习 page 61））
    eigvar, eigvect = nm.linalg.eig(nm.linalg.inv(sb) @ sa)
    w = eigvect[:, nm.argmax(eigvar)]  # w is the  eigenvector correspond largest eigenvalue

    distance = []
    for i in data:
        distance.append(i.mean @ w)

    test_data, test_lable = load_data.load_test_data()
    predict = nm.power(
        (test_data @ w).reshape(len(test_data), 1) - nm.array(distance).reshape(1, len(distance)),
        2).argmin(1)

    print("accuration:"+str(nm.sum(nm.equal(predict,test_lable))/len(test_lable)))


if __name__ == "__main__":
    main()
