import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as nm
from lda.load_data import load_test_data,load_train_data,uniform_data

#使用共同协方差矩阵 cov=sum(COVi)/n
same_cov=True

def main():

    def gaussian(cov,mean,data):
        #cov.shape=(classfication,row,colunm),
        #mean.shape=(classfication,1,mean_vector),
        #data.shape=(batch,classfication=1,row=1,colunm)

        # 多元高斯分布密度函数：p(x)=( 1/((2*pi)^(k/2)*det(cov)^0.5) )*exp{-0.5*(x-u).T @ cov^-1 @ (x-u)};
        #  cov=协方差矩阵, det() 行列式 , cov^-1 协方差矩阵逆矩阵 ,u 均值向量

        # 多元高斯极大似然：   u=sum(x)/n , cov=sum( (x-u) @ (x-u).T )
        det=nm.linalg.det(cov)
        e=(data-mean)@nm.linalg.inv(cov)@nm.transpose(data-mean,[0,1,3,2])
        return (1/(nm.power(2*nm.pi,feature_dim/2)*nm.power(det.reshape([1,len(det)]),1/2)))*nm.exp(-0.5*e.reshape(list(e.shape[:-2])))

    data,_=load_train_data()

    feature_dim=len(data[0].mean)

    cov_martix,mean_vector,classfication_probable=[],[],[]  # 协方差矩阵，均值向量 ， 类别权重
    for i in data:
        cov_martix.append(i.cov)
        mean_vector.append(i.mean)
        classfication_probable.append(i.probable)
        
    cov_martix,mean_vector,classfication_probable=nm.array(cov_martix),nm.array(mean_vector),nm.array(classfication_probable)
    if same_cov:
        cov_martix=nm.sum(cov_martix,axis=0,keepdims=1)/len(data)

    test_data, test_lalbe = load_test_data()

    # dim = [batch,classfication,row,column]
    gaussian_p=gaussian(cov_martix,mean_vector.reshape([mean_vector.shape[0],1,mean_vector.shape[1]]),test_data.reshape([test_data.shape[0],1,1,test_data.shape[1]]))

    #bayes probability : p(C|A) = P(CA)/P(A) = ( p(A|C)*P(C) )/P(A) ; p(A)=sum(P(ACi))=sum(P(A|Ci)*P(Ci))
    #                    C 类别 , A 样本

    bayes_p=(gaussian_p*classfication_probable)/nm.sum(gaussian_p*classfication_probable,axis=1).reshape(len(gaussian_p),1)
    print("accuracy:{}".format(nm.sum(nm.equal(nm.argmax(bayes_p,axis=1),test_lalbe))/len(test_lalbe)))



if __name__=="__main__":
    main()


