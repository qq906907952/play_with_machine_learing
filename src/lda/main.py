import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lda import load_data
import numpy as nm

def main():
    data, u = load_data.load_train_data()  # u is a vector of all data mean
    if len(data) == 0:
        raise RuntimeError("load data fail")

    # 广义锐利商 (w.T @ Sa @ w)/(w.T @ Sb @ w)
    # Sa=SUM{Mi*[ (Xi-u) @ (Xi-u).T  ]}     u=所有样本均值向量 Xi=每个类别均值向量 Mi=每个类别样本数
    # Sb=SUM(COVi)   COVi=每个类别协方差矩阵
    sa, sb = nm.zeros(data[0].cov.shape), nm.zeros(data[0].cov.shape)
    for i in data:
        sa += len(i) * ((i.mean - u).reshape(len(u), 1) @ ((i.mean - u).reshape(1, len(u))))
        sb += i.cov

    # 广义锐利取最大值时 w 为 Sb-1 @ sa 最大特征值对应特征向量（可用拉格郎日乘法证明）
    eigvar, eigvect = nm.linalg.eig(nm.linalg.inv(sb) @ sa)
    w = eigvect[:, nm.argmax(eigvar)]  # w is the  eigenvector correspond largest eigenvalue

    distance = []   # 每个类别的均值在w方向的映射
    for i in data:
        distance.append(i.mean @ w)

    test_data, test_lable = load_data.load_test_data()

    # print(distance)
    # print(test_data @ w)

    predict = nm.power(
        # test_data @ w 计算每个样本在w方向的映射 再与distance每个元素相减 平方后取最小对应的索引
        (test_data @ w).reshape(len(test_data), 1) - nm.array(distance).reshape(1, len(distance)),
        2).argmin(1)

    # print(nm.equal(predict,test_lable))
    print("accuration:"+str(nm.sum(nm.equal(predict,test_lable))/len(test_lable)))


if __name__ == "__main__":
    main()
