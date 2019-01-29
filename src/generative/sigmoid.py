import sys, os

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pwd)
sys.path.insert(0, os.path.dirname(pwd))

import numpy as nm
import pandas as pd

# 学习比率
learning_rate = 0.01
# 递归次数
iter_amount = 600000
print_pre_step = 1000


def load_data(is_train):
    path = pwd + "/../lda"
    if is_train:
        path += "/iris_training.csv"
    else:
        path += "/iris_test.csv"
    data = pd.read_csv(path)
    return data


def main():
    data = load_data(True)
    w = []
    classfication_probable = []
    classfication_data = []
    lable = []
    real_lable=data["lable"]
    _data = data.drop("lable", axis=1)
    feature_len=len(data.groupby("lable").count())
    for i in range(feature_len):
        c = data.lable.replace({i: -1})
        c[c != -1] = 0
        c[c == -1] = 1

        classfication_data.append(nm.array(_data))
        lable.append(c)
        classfication_probable.append(len(c[c == 1]) / len(c))
        w.append(nm.random.random([len(_data.columns)]))

    w = nm.array(w)
    w = w.reshape(list(w.shape) + [1])  # regression variables , shape = classfication x feature x 1
    classfication_probable = nm.array(classfication_probable).reshape([len(classfication_probable),1,1])  # classfication weight , shape = classfication x 1 x 1
    classfication_data = nm.array(classfication_data)  # data , shape=classfication x batch x feature
    lable = nm.array(lable)
    lable = lable.reshape(list(lable.shape) + [1])  # lable , shape =  classfication x batch x 1

    for i in range(iter_amount):
        # 逻辑回归梯度下降推导见 src/logistic_regression/main.py
        z = -classfication_data @ w
        gradient = (classfication_data.transpose([0, 2, 1]) @ ((1 - lable) - nm.exp(z) / (1 + nm.exp((z))))) / len(data)
        w -= learning_rate * gradient

        if i % print_pre_step == 0:
            print("==================================")
            print("gradient:\r\n{}".format(str(gradient)))


            """
                bayes_probabilty : P(C|A)=P(AC)/P(A)
                                         =(P(A|C)*P(A))/P(A)
                                         =(P(A|C)*P(A))/sum(P(ACi))
                                         =(P(A|C)*P(A))/sum(P(A|Ci)*P(Ci))
            """


            p=((1 / (1 + nm.exp(z)))*classfication_probable)/nm.sum(1 / (1 + nm.exp(z)),0)
            print()
            print("trainning data accuracy:{}".format(str(nm.sum(nm.equal(nm.argmax(p,0).reshape([len(data)]),real_lable))/len(data))))
            print("============={}/{}===============".format(i,iter_amount))
            print()


    data=load_data(False)
    lable=data["lable"]
    del data["lable"]
    _data=[]
    for i in range(feature_len):
        _data.append(nm.array(data))
    _data=nm.array(_data)
    z = - _data @ w
    p = ((1 / (1 + nm.exp(z))) * classfication_probable) / nm.sum(1 / (1 + nm.exp(z)), 0)
    print("testing data accuracy:{}".format(str(nm.sum(nm.equal(nm.argmax(p, 0).reshape([len(data)]), lable)) / len(data))))

if __name__ == "__main__":
    main()

