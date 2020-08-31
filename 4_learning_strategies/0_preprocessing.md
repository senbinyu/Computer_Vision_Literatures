Learning strategies are also very important to the training and testing. Firstly, preprocessing the data, like image augmentation is sometimes crucial for the prediction of reasonable results.


## 1. 验证集划分
![数据验证集划分](https://user-images.githubusercontent.com/42667259/91734531-5539ae80-ebab-11ea-8d29-71632e3be6a7.png)  
- 直接划分：随机划分的方式使得模型的训练数据可能和测试数据差别很大，导致训练出的模型泛化能力不强。  
- LOOCV: Leave-one-out cross-validation，这相当于是k折交叉验证的一个极端情况，即K=N。每次只用一个数据作为测试，其他均为训练集，重复N次（N为数据集数目）  
- kFold，k折交叉验证，每次的测试集将不再只包含一个数据，而是多个，具体数目将根据K的选取决定。比如，如果K=5，那么我们利用五折交叉验证的步骤就是：1）将所有数据集分成5份；2）不重复地每次取其中一份做测试集，用其他四份做训练集训练模型，之后计算该模型在测试集上的Error_i；3）将5次的Error_i取平均得到最后的Error。

## 2. Image augmentation,图像or数据增强
![data augmentation](https://user-images.githubusercontent.com/42667259/91757069-12d59900-ebce-11ea-96dd-09d0e2102b14.png)

### 2.1 传统方法
- 直方图均衡  
原理是平均分布的图像信息熵最大，也就相当于是说对比度最大。从初始状态转化到直方图均衡状态时，经过一个转移函数，转移函数可由最终想要达到的平均状态作为极限来反推得到。   
具体可参阅：https://zhuanlan.zhihu.com/p/44918476

- 灰度变换  
灰度变换可使图像动态范围增大，对比度得到扩展，使图像清晰、特征明显，是图像增强的重要手段之一。它主要利用图像的点运算来修正像素灰度，由输入像素点的灰度值确定相应输出像素点的灰度值，可以看作是“从像素到像素”的变换操作，不改变图像内的空间关系。像素灰度级的改变是根据输入图像f(x,y)灰度值和输出图像g(x,y)灰度值之间的转换函数g(x，y)=T[f(x，y)]进行的。
灰度变换包含的方法很多，如逆反处理、阈值变换、灰度拉伸、灰度切分、灰度级修正、动态范围调整等。

- 图像平滑   
在空间域中进行平滑滤波技术主要用于消除图像中的噪声，主要有邻域平均法、中值滤波法等等。这种局部平均的方法在削弱噪声的同时，常常会带来图像细节信息的损失。   

- 图像锐化   
采集图像变得模糊的原因往往是图像受到了平均或者积分运算，因此，如果对其进行微分运算，就可以使边缘等细节信息变得清晰。这就是在空间域中的图像锐化处理，其基本方法是对图像进行微分处理，并且将运算结果与原图像叠加。从频域中来看，锐化或微分运算意味着对高频分量的提升。常见的连续变量的微分运算有一阶的梯度运算、二阶的拉普拉斯算子运算，它们分别对应离散变量的一阶差分和二阶差分运算。
https://blog.csdn.net/TheDayIn_CSDN/article/details/86682034  

### 2.2 深度学习中的图像增强（预处理）   
这些手段相当于yolo v4中所说的bags of free，意指用这些数据增强的手段，只会改变训练时的策略或者增加训练的时间，但不会对模型的最终的推断时间产生影响，相当于是免费的技巧。因此，yolo v4中用到了大量的数据增强手段，而结果也令人惊喜，得到了几个点的提升。可使用albumentations包，里面包含各类图像增强技术。

- 几何增强：平移，旋转，剪切等对图像几何改变的方法，可以增强模型的泛化能力。

- 色彩增强：主要是亮度变换，如使用HSV(HueSaturationValue)增强。

- mixup，[mixup: Beyond empirical risk minimization](https://arxiv.org/pdf/1710.09412.pdf)  
上述的通用数据增强方法则是针对同一类做变换，而mixup则是采用对不同类别之间进行建模的方式实现数据增强。不同的类加上不同的权重，而其得到的损失函数也加上不同的权重，最后再进行反向传导求参数。具体可参阅：https://blog.csdn.net/ouyangfushu/article/details/87866579

- 

