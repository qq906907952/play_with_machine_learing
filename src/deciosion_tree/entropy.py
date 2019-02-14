import pandas as pd
import numpy as nm
import math
discrete = "discrete"
continuously = "continuously"
lable = "survived"
weight="weight"
zero=0.001

def __infomation_entropy(data):

    #    信息熵：info_entropy = -sum(pk * log2 pk)  , pk=k类数据数量/数据集总数

    d = nm.array(data[[lable,weight]].groupby(lable).sum())

    return -nm.sum((d / nm.sum(d)) * nm.log2(d / nm.sum(d)))


def information_gain(data: pd.DataFrame, feature, feature_type):
    info_ent = __infomation_entropy(data) # 总熵

    data_len=data[weight].sum()
    if feature_type == discrete:#离散型

        # 信息增益：info_gain=info_ent-sum(wi*info_ent(feature_i))
        # info_ent(feature_i) feature属性取值为i的信息熵
        # wi feature属性取值为i所占比例
        for i in data[feature].unique():
            _data = data[data[feature] == i]
            info_ent -= (_data[weight].sum() / data_len) * __infomation_entropy(_data)

        return info_ent, None

    elif feature_type == continuously: #连续型
        data=data.astype("float64")
        value = list(set(list(data[feature])))
        value.sort()
        i = 0
        split_value,info_gain=0,0
        while i < len(value) - 1:
            _info_gain = info_ent
            #中间值二分
            mid = (value[i] + value[i + 1]) / 2
            data_less_than = data[data[feature] < mid]
            data_more_than = data[data[feature] >= mid]
            _info_gain -= (data_less_than[weight].sum() / data_len) * __infomation_entropy(data_less_than) + (
                        data_more_than[weight].sum() / data_len) * __infomation_entropy(data_more_than)
            if _info_gain>info_gain:
                info_gain,split_value=_info_gain,mid

            i += 1

        return info_gain,split_value

    else:
        raise RuntimeError("unknow feature type")



def IV(data:pd.DataFrame,split_value,feature_name):
    # 固有值： IV=-sum(pi * log2 pi)    , pi=i类特征数据数/数据总数
    data_len=data[weight].sum()
    if split_value==None:#离散型
        iv=0
        for i in data[feature_name].unique():
            _weight=data[data[feature_name]==i][weight].sum()/data_len
            iv+=-(_weight*math.log2(_weight))
        return iv
    else: #连续性

        data=data.astype("float64")
        less_weight=data[data[feature_name]<split_value][weight].sum()/data_len
        more_weight=data[data[feature_name]>=split_value][weight].sum()/data_len

        return -(less_weight*math.log2(less_weight)+more_weight*math.log2(more_weight))