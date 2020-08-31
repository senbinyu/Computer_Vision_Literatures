In training procedure, the choose of optimizer and loss function is of great importance, as well as the choose of neural networks.

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


