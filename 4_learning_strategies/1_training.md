In training procedure, the choose of optimizer and loss function is of great importance, as well as the choose of neural networks.

## Activation functions
激活函数，引入非线性，因为线性函数的组合还是线性的，引入非线性可以增强模型的表达能力，使其变得更复杂。  
sigmoid: 在0-1范围内，非对称，可求导，但是容易梯度消失，使得训练难以进行。  
tanh: [-1, 1]，关于原点对称，但是梯度消失的问题仍没有解决。  
ReLU,解决了部分梯度消失问题，但负的部分仍为0，神经元死亡。  
Leaky RELU, 负的部分很小，但解决了神经元死亡问题。  
Maxout, 参数比较多，相当于就像是又增加了一层神经网络  
ELU，exponential leaky relu, 在Leaky RELU基础上进一步改进，在x<0的地段指数级减小，同时在前面加个系数。具体参见文献：[FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)](https://arxiv.org/pdf/1511.07289.pdf%5cnhttp://arxiv.org/abs/1511.07289%5cnhttp://arxiv.org/abs/1511.07289.pdf)  
![activation_funcs](https://user-images.githubusercontent.com/42667259/91729737-4c45de80-eba5-11ea-9ce5-01ce27896504.png)

除了上述激活函数外，还有一些新的取得很好效果的激活函数。如PReLU, Swish, Mish等。
PReLU: 由式子可见，Leaky RELU只是PRELU的一种特殊情况。具体来源：He Kaiming et al., 2015, [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf)  
![PRELU](https://user-images.githubusercontent.com/42667259/91731243-fa05bd00-eba6-11ea-9792-da910b09b8d5.png)  
Swish:适用于深层次模型，在深层模型上的效果优于 ReLU。可以看做是介于线性函数与ReLU函数之间的平滑函数。  
![swish](https://user-images.githubusercontent.com/42667259/91732403-7b118400-eba8-11ea-9cda-58a2f2589339.png)  
Mish: 类似于swish，Mish几乎在所有测试中都优于RELU激活函数。具体参见论文：Misra, 2019, [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/pdf/1908.08681.pdf)   
![mish_1](https://user-images.githubusercontent.com/42667259/91732401-7b118400-eba8-11ea-8125-6079441456bf.png)  
![mish](https://user-images.githubusercontent.com/42667259/91732398-7a78ed80-eba8-11ea-843f-e4a58768bc84.png)


## Optimization
![优化算法optimization](https://user-images.githubusercontent.com/42667259/91722936-4e0aa480-eb9b-11ea-92df-4d96b2bc09db.png)

Acoording to the previous figure, we know that generally people would like to use one-order itrative method to optimize the loss function. 
以下链接给出了详细的解释，https://zhuanlan.zhihu.com/p/55150256  
随机梯度下降法：sgd，w = w - a \nabla w   
sgd + 动量法：引入了速度项，速度项 v_{t+1} = \rho v_t + (1-\rho) \nabla w,再代入上式，积累了之前w项的梯度作为速度，梯度越大，则速度越大。
https://blog.csdn.net/weixin_36811328/article/details/83451096  
Adagrad, 利用之前积累的\nabla w的平方和在分母内开根号，因此在前期平方和小，参数更新快，后期更新慢，满足条件；但在中后期，由于累积的平方和大，导致参数更新不动，带来新的问题。   
rmsProp方法：在adagrad基础上的更新，将直接用所有的梯度平方和累加改为指数加权的移动平均，设置一个衰减率，默认为0.9，乘以之前的梯度平方和累积，然后加上剩下的部分，总体放到。  
![optimization_rmsprop](https://user-images.githubusercontent.com/42667259/91726163-16eac200-eba0-11ea-9db0-e91723096bb9.png)  
Adam法：相当于是结合之前的速度法中的一阶矩更新，放在分子中，和rmsProp之中的二阶矩方法，放在分母中开根号，相比于缺少修正因子而导致二阶矩估计，可能在训练初期具有很高偏置的RMSProp稳定性更好。  
![optimization_adam](https://user-images.githubusercontent.com/42667259/91726417-7d6fe000-eba0-11ea-8831-5672c3edb37a.png)


## Loss function
For different tasks, there are different settings of loss functions. For instance, in segmentation, loss function are separated into three categories: distribution-based, region-based and compound loss functions. 

从本质上来讲，目标检测和图像分割总是可以分为分类和回归这两个基本任务。例如目标检测中，分类总是需要有分类损失函数，而bounding box的损失函数常看做是回归框的回归损失，可以用基于区域的IOU, GIOU, CIOU来衡量。不过归根结底，都可以说成是分类和回归结合在一起的复合损失函数。图像分割中，也经常需要用到分类（常用的二分类），和基于回归mask，即分割区域的Dice loss等，也可以看成是一种回归损失。

具体的损失函数种类可以查看object detection中和segmentation中的loss function章节，有对应的详细介绍。


