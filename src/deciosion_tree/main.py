import os, sys, copy

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pwd)
sys.path.insert(0, os.path.dirname(pwd))
import pandas as pd

from node import *
from entropy import *

del_feature = ["name",
               "home.dest",
               "body",
               "boat",
               # "cabin",
               "ticket",  # 过拟合
               ]

feature_list = {
    "pclass": discrete,
    "sex": discrete,
    "age": continuously,
    "sibsp": discrete,
    "parch": discrete,
    "fare": continuously,
    "embarked": discrete,

    "cabin": discrete,

}

train_set_num = 700
prune_set_num = 300


def load_data():
    data = pd.read_csv(pwd + "/titanic.csv")
    data.columns = data.columns.str.strip()

    for i in del_feature:
        del data[i]
    data.drop_duplicates(inplace=True)
    data = data.sample(frac=1)
    data[weight] = 1
    return data[:train_set_num], data[train_set_num:prune_set_num+train_set_num], data[prune_set_num+train_set_num:]


def create_node(data: pd.DataFrame, feature_list, parent_node, feature_weight, depth):

    # 特征长度为0（只剩lable）    或 没有数据 ，标记为叶节点 ，lable 为父节点lable
    if len(feature_list) == 1 or len(data) == 0:
        return Node("", leaf_node, None, parent_node.node_lable, feature_weight)
    # 所有样本取相同特征 ， 标记为叶节点，lable为当前划分的数据集的特征
    if len(data.groupby(lable).count()) == 1:
        return Node("", leaf_node, None, int(data[lable].unique()[0]), feature_weight)

    feature_list = copy.deepcopy(feature_list)
    feature_info_gain = {}  # 用于记录每个特征的信息增益及连续型数据的划分点 map<string feature_name,[float info_gain,int split_value]>
    totle_info_gain = 0  # 所有特征信息增益总和 用于计算平均信息增益
    # 计算每个特征的信息增益
    for feature, feature_type in feature_list.items():
        _data = data[[feature, weight, lable]]
        _data = _data[_data[feature].astype("str").str.strip() != ""]  # 去掉缺失的数据
        info_gian, split_value = information_gain(_data, feature, feature_type)
        p = len(_data) / len(data)

        totle_info_gain += p * info_gian
        feature_info_gain[feature] = (p * info_gian, split_value)

    average_info_gain = totle_info_gain / len(feature_list)  # 平均信息增益
    if nm.abs(average_info_gain) < zero:  # 即所有属性都无法降低数据纯度，标记为叶节点 lable为数量最多的lable
        return Node("", leaf_node, None, int(data[[lable, weight]].groupby(lable).sum().idxmax()), feature_weight)


    # 过滤超过平均信息增益的属性
    more_than_avg_feature = filter(lambda x: feature_info_gain[x][0] >= average_info_gain, feature_info_gain)

    max_info_gain_rate_feature = ""

    # 信息增益率 info_gain_rate = info_fain/IV
    info_gain_rate = 0

    # 选择最大信息增益率的属性
    for i in more_than_avg_feature:
        iv = IV(data[data[i].astype("str").str.strip() != ""][[i, weight]], feature_info_gain[i][1], i)

        if iv == 0:  # 固有值为0即所有属性取值相同
            continue

        _info_gain_rate = feature_info_gain[i][0] / iv
        if _info_gain_rate > info_gain_rate:
            info_gain_rate, max_info_gain_rate_feature = _info_gain_rate, i

    # create new Node
    node = Node(max_info_gain_rate_feature, non_leaf_node, feature_list[max_info_gain_rate_feature]
                , data[lable].value_counts().idxmax(), feature_weight)

    feature_loss_data = data[data[node.feature_name].astype("str").str.strip() == ""]   #该属性缺失的数据
    __weight = feature_loss_data[weight]

    del feature_list[node.feature_name]

    if node.value_type == discrete: # 离散型
        for i in data[data[node.feature_name].astype("str").str.strip() != ""][node.feature_name].unique(): # 最大信息增益率属性对应所有取值

            _data = data[data[node.feature_name] == i]  # 属性取值为i的数据
            feature_weight = _data[weight].sum() / data[weight].sum()  #节点权重
            if len(feature_loss_data) != 0:
                feature_loss_data[weight] = __weight * feature_weight    #属性缺失数据按权重划分到每个节点
                _data = _data.append(feature_loss_data)
            del _data[node.feature_name]
            node.append_child_node(create_node(_data, feature_list, node, feature_weight, depth + 1), i)  #生成新字节点

    else: # 连续型
        split_value = feature_info_gain[node.feature_name][1]
        not_none_data = data[data[node.feature_name].astype("str").str.strip() != ""]
        data_less_than = not_none_data[not_none_data[node.feature_name].astype("float64") < split_value]
        data_more_than = not_none_data[not_none_data[node.feature_name].astype("float64") >= split_value]
        data_less_than_weight = data_less_than[weight].sum() / data[weight].sum()
        data_more_than_weight = data_more_than[weight].sum() / data[weight].sum()
        if len(feature_loss_data) != 0:
            feature_loss_data[weight] = __weight * data_less_than_weight
            data_less_than = data_less_than.append(feature_loss_data)

            feature_loss_data[weight] = __weight * data_more_than_weight
            data_more_than = data_more_than.append(feature_loss_data)

        del data_less_than[node.feature_name]
        del data_more_than[node.feature_name]
        node.append_child_node(
            [create_node(data_less_than, feature_list, node, data_less_than_weight, depth + 1),
             create_node(data_more_than, copy.deepcopy(feature_list), node, data_more_than_weight, depth + 1)],
            split_value)

    return node


def main():
    data, prune_data, test_data = load_data()
    node = create_node(data, feature_list, None, None, 0)

    # import pickle
    # f=open("node.serize","wb")
    # pickle.dump(node,f)
    # f.close()

    # f=open("node.serize","rb")
    # node=pickle.load(f)

    train_lable = nm.array(data[lable])
    _lable = []
    del data[lable]
    for _, i in data.iterrows():
        l, p = node(i, 1,0)
        _lable.append(l)

    train_accuracy=nm.sum(nm.equal(nm.array(_lable), train_lable)) / len(train_lable)



    test_lable = nm.array(test_data[lable])
    _lable = []

    del test_data[lable]

    for _, i in test_data.iterrows():
        l, p = node(i, 1,0)
        _lable.append(l)
    print(_lable)
    test_accuracy=nm.sum(nm.equal(nm.array(_lable), test_lable)) / len(test_lable)
    print("before prune : train data accuracy:{} | test data accuracy:{}".format(train_accuracy,test_accuracy))


    node.prune(prune_data)



    _lable = []

    for _, i in data.iterrows():
        l, p = node(i, 1,0)
        _lable.append(l)

    train_accuracy=nm.sum(nm.equal(nm.array(_lable), train_lable)) / len(train_lable)

    _lable = []

    for _, i in test_data.iterrows():
        l, p = node(i, 1,0)
        _lable.append(l)
    print(_lable)
    test_accuracy=nm.sum(nm.equal(nm.array(_lable), test_lable)) / len(test_lable)
    print("after prune : train data accuracy:{} | test data accuracy:{}".format(train_accuracy, test_accuracy))


if __name__ == "__main__":
    main()
    # data,_=load_data()
    # print(data[[lable,weight]].groupby(lable).sum())
