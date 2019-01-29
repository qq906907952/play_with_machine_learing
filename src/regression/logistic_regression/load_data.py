import pandas as pd
import numpy as nm
import os

from lda.load_data import  _normalize_data

train_set_num = 600
pwd = os.path.dirname(os.path.abspath(__file__))

def load_data(is_train)->(nm.ndarray,nm.ndarray):
    data=pd.read_csv(pwd+"/titanic.csv")
    del data["Name"]
    data.loc[:,"Sex"].replace("male",1,inplace=True)
    data.loc[:,"Sex"].replace("female",2,inplace=True)

    lable = data["Survived"]
    del data["Survived"]

    data=_normalize_data(data,-1,1)

    if  is_train:
        data = data[:train_set_num]
        lable = lable[:train_set_num]
    else:
        data=data[train_set_num:]
        lable=lable[train_set_num:]



    return nm.array(data),nm.array(lable,dtype=nm.float)

if __name__=="__main__":
    a,b=load_data(False)
    print(b)