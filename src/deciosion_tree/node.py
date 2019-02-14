import pandas as pd
import numpy as nm

leaf_node = -1  # 叶节点
non_leaf_node = 1  # 非叶节点

from entropy import discrete, continuously, lable, weight, zero


class Node():
    def __init__(self, feature_name, node_type, value_type, node_lable, weight):

        """
        :param feature_name: string 该节点划分的属性名称
        :param node_type:    const  节点类型 叶节点？非叶节点？
        :param value_type:   const string  值类型 离散？ 连续？
        :param node_lable:   int   标签
        :param weight:       float 节点权重 用于父节点属性缺失
        """
        self.node_lable = node_lable
        self.node_type = node_type
        self.weight = weight
        if node_type == leaf_node:
            return

        self.feature_name = feature_name
        self.value_type = value_type
        self.split_value = None  # 划分值 只有连续性属性不为none
        self.child_node = {}  # 子节点 字典 map<string feature / bool less_than? more_than? , Node>

    def append_child_node(self, node, value):

        if self.value_type == discrete:
            self.child_node[value] = node
        elif self.value_type == continuously:

            if len(node) != 2:
                raise RuntimeError("child node of continuously value node must a list of len 2")
            self.split_value = value
            self.child_node[True] = node[0]  # 连续型 小于
            self.child_node[False] = node[1]  # 连续型 大于等于

    def __call__(self, data: pd.Series, weight, deep):
        if self.node_type == leaf_node:
            return self.node_lable, weight

        else:

            if str(data[self.feature_name]).strip() == "":

                p = 0
                lable = 0
                for _, v in self.child_node.items():
                    l, _p = v(data, v.weight * weight, deep + 1)
                    if _p > p:
                        p = _p
                        lable = l

                return lable, p

            if self.value_type == discrete:
                if self.child_node.get(data[self.feature_name]) == None:  # 该节点没见过的属性
                    return self.node_lable, weight
                return self.child_node[data[self.feature_name]](data, weight, deep + 1)
            else:
                return self.child_node[float(data[self.feature_name]) < self.split_value](data, weight, deep + 1)

    def prune(self, data: pd.DataFrame):  # 后剪枝
        if len(data) == 0:
            return 0
        if self.node_type == leaf_node:
            _data = data[data[lable] == self.node_lable]
            return _data[weight].sum()

        feature_loss_data = data[data[self.feature_name].astype("str").str.strip() == ""]  # 节点属性缺失的数据
        feature_not_loss_data = data[data[self.feature_name].astype("str").str.strip() != ""]  # 节点属性完整的数据
        __weight = feature_loss_data[weight].astype("float64")

        reduce_data = pd.DataFrame(feature_not_loss_data)  # 没有参与字节点的数据（属性值没见过的数据）

        child_node_totle_accuracy = 0
        if self.value_type == discrete: #离散型
            #计算各子节点准确率
            for k, v in self.child_node.items():

                if len(feature_loss_data) != 0:
                    feature_loss_data[weight] = __weight * v.weight
                _data = feature_not_loss_data[feature_not_loss_data[self.feature_name] == k]
                reduce_data = reduce_data.append(_data).drop_duplicates(keep=False)
                child_node_totle_accuracy += v.prune(_data.append(feature_loss_data))

        else:  # 连续型
            if len(feature_loss_data) != 0:
                feature_loss_data[weight] = __weight * self.child_node[False].weight

            data_more_than = feature_not_loss_data[
                feature_not_loss_data[self.feature_name].astype("float64") >= self.split_value]
            child_node_totle_accuracy += self.child_node[False].prune(data_more_than.append(feature_loss_data)) # 大于等于划分值的节点

            if len(feature_loss_data) != 0:
                feature_loss_data[weight] = __weight * self.child_node[True].weight
            data_less_than = feature_not_loss_data[
                feature_not_loss_data[self.feature_name].astype("float64") < self.split_value]
            child_node_totle_accuracy += self.child_node[True].prune(data_less_than.append(feature_loss_data))  # 小于划分值的节点

            reduce_data = reduce_data.append(data_more_than).append(data_less_than).drop_duplicates(keep=False)

        feature_loss_data[weight] = __weight

        __data = feature_not_loss_data.append(reduce_data).drop_duplicates(keep=False)
        this_accuracy = feature_loss_data[feature_loss_data[lable] == self.node_lable][weight].sum() + \
                        __data[__data[lable] == self.node_lable][weight].sum()

        if nm.abs(this_accuracy - child_node_totle_accuracy) > zero and this_accuracy > child_node_totle_accuracy:
            self.node_type = leaf_node
            self.child_node = None
            return this_accuracy + reduce_data[reduce_data[lable] == self.node_lable][weight].sum()
        return child_node_totle_accuracy + reduce_data[reduce_data[lable] == self.node_lable][weight].sum()

    def __str__(self):
        pass
