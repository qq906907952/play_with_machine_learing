import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as nm
from logistic_regression.load_data import load_data

#学习比率
learning_rate=0.01
#递归次数
iter_amount=600000

def main():
    global  learning_rate
    data, lable = load_data(True)

    data=nm.append(data,nm.ones([len(data),1]),axis=1)

    w=nm.random.rand(data.shape[1]) #type: nm.ndarray
    w.dtype="float64"
    print_pre_step=1000


    for i in range(iter_amount):
        """
        data.shape=batch x features len
        
        叠乘=mul()
        求和=sum()
        
        ln(y/(1-y))=w.T@x
        
        y=1/(1+e^(-w.T@x))
                    
        maximun likelihood : l=mul( yi*(1/(1+e^(-w.T@xi))) + (1-yi)*(1+(1/e^(-w.T@xi))) )  yi=0 or 1(类别) , w={w1,w2....wn,b}(回归变量) , xi={x1,x2,....xn,1}(特征) , e 自然对数  
        
                             设 z=-w.T@x , sig()=1/(1+e^z)  , 1-sig()=e^w.T@xi/(1+e^z)

                             ln(l)=sum( ln( (yi*sig() + (1-yi)*(1-sig()) ) )
                                  =sum( yi*ln(sig()) + (1-yi)*ln(1-sig()) )
                                  =sum( yi*( ln(1) - ln(1+e^z) ) + (1-yi)*(ln(e^z) - ln(1+e^z)) )
                                  =sum( -yi*ln(1+e^z) + (1-yi)*z - ln(1+e^z) + yi*ln(1+e^z)  )
                                  =sum( (1-yi)*z - ln(1+e^z) )

                             max ln(l)  ---->   min -ln(l)
                             
                             -ln(l)= sum( ln(1+e^z) - (1-yi)*z )
                             
                             gradient = d(-ln(l))/d(w)=(d(-ln(l))/dz) * (dz/dw)
                                      = sum{ (e^z/(1+e^z) - (1-yi))*(-xi) }                            
                                      = sum( (-xi)*(e^(-w.T@xi)/(1+e^(-w.T@xi)) - (1-yi)) )
                                      = sum( x*( (1-yi) - (e^(-w.T@xi)/(1+e(-w.T@xi))) ) )
                                      
                             当x为多批次时：
                                data=[x1.T , x2.T , ......xn.T,[1,1,1,....].T] , y=[y1,y2,....yn].T , w=[w1,w2,....wn,b]
                                gradient = data.T @ ( (1-y) - (exp{-data@w}/(1+exp{data@w})) )
        """
        z=-data@w
        gradient=(data.T@((1-lable)-nm.exp(z)/(1+nm.exp((z)))))/len(data)
        w-=learning_rate*gradient

        if i%print_pre_step==0:
            print("=======================",end="")
            print("{0}/{1}".format(i,iter_amount),end="")
            print("=======================")
            print("gradient:{}".format(gradient))


            # 交叉熵 loss = -sum( yi*ln(sig()) + (1-yi)*(1-sig()) )

            loss = -(lable * nm.log(1 / (1 + nm.exp(z))) + (1-lable) * nm.log(1 - (1 / (1 + nm.exp(z)))))
            print("average loss:{}".format(nm.sum(loss)/len(lable)))


    test_data,test_lable=load_data(False)
    test_data = nm.append(test_data, nm.ones([len(test_data), 1]), axis=1)

    t=nm.equal(test_lable,(1/(1+nm.exp(-(test_data@w))))>0.5)
    print("===============")
    print("accuracy:{}".format(sum(t)/len(t)))




if __name__=="__main__":
    main()
    print()
