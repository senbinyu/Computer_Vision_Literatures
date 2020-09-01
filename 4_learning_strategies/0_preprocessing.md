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
在空间域中进行平滑滤波技术主要用于消除图像中的噪声，主要有邻域平均法、中值滤波法、高斯滤波等等。这种局部平均的方法在削弱噪声的同时，常常会带来图像细节信息的损失。   
上述的滤波方法一般都只是考虑空间域的操作（即相近像素），而忽略了值域空间的相关性，而这正是边缘所在处（值域的剧烈波动）非常重要的因素。因此，有双边滤波（保边滤波）方法，同时考虑了空域和值域，且设置了相应的权重，空域中离关注区域点左边越近，权重系数越大；值域中该值和所关注区域的值接近，其权重系数就大，反之则小，由此可以保住边缘处的信息。  
具体可参阅：https://blog.csdn.net/guyuealian/article/details/82660826   


- 图像锐化   
采集图像变得模糊的原因往往是图像受到了平均或者积分运算，因此，如果对其进行微分运算，就可以使边缘等细节信息变得清晰。这就是在空间域中的图像锐化处理，其基本方法是对图像进行微分处理，并且将运算结果与原图像叠加。从频域中来看，锐化或微分运算意味着对高频分量的提升。常见的连续变量的微分运算有一阶的梯度运算、二阶的拉普拉斯算子运算，它们分别对应离散变量的一阶差分和二阶差分运算。
https://blog.csdn.net/TheDayIn_CSDN/article/details/86682034  

### 2.2 深度学习中的图像增强（预处理）   
这些手段相当于yolo v4中所说的bags of free，意指用这些数据增强的手段，只会改变训练时的策略或者增加训练的时间，但不会对模型的最终的推断时间产生影响，相当于是免费的技巧。因此，yolo v4中用到了大量的数据增强手段，而结果也令人惊喜，得到了几个点的提升。可使用albumentations包，里面包含各类图像增强技术。

- 几何增强：平移，旋转，剪切等对图像几何改变的方法，可以增强模型的泛化能力。

- 色彩增强：主要是亮度变换，如使用HSV(HueSaturationValue)增强。  

- Blurring,模糊，诸如高斯滤波，方框滤波，中值滤波等。可以增强模型对模糊图像的泛化能力。

- mixup，[mixup: Beyond empirical risk minimization](https://arxiv.org/pdf/1710.09412.pdf)   
上述的通用数据增强方法则是针对同一类做变换，而mixup则是采用对不同类别之间进行建模的方式实现数据增强。不同的类加上不同的权重，而其得到的损失函数也加上不同的权重，最后再进行反向传导求参数。具体可参阅：https://blog.csdn.net/ouyangfushu/article/details/87866579

- 随机擦除（Random Erasing, RE）增强, [Random erasing data augmentation](https://arxiv.org/abs/1708.04896)   
随机擦除，提出的目的主要是模拟遮挡，从而提高模型泛化能力，对遮挡有更好的鲁棒性。随机选择一个区域，然后采用随机值进行覆盖，模拟遮挡场景。

- Cutout，DeVries et al., 2017, [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf)   
其的出发点和随机擦除一样，也是模拟遮挡，目的是提高泛化能力，实现上比Random Erasing简单，随机选择一个固定大小的正方形区域，然后采用全0填充就OK了，当然为了避免填充0值对训练的影响，应该要对数据进行中心归一化操作，norm到0。具体也可参阅：https://blog.csdn.net/weixin_41560402/article/details/106036378  

- CutMix, Yun et al., 2019, [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899v2.pdf)  
就是将一部分区域cut掉但不填充0像素而是随机填充训练集中的其他数据的区域像素值，分类结果按一定的比例分配.能让模型更加准确的分类和定位。

上述三种数据增强的区别：cutout和cutmix就是填充区域像素值的区别；mixup和cutmix是混合两种样本方式上的区别：mixup是将两张图按比例进行插值来混合样本，cutmix是采用cut部分区域再补丁的形式去混合图像，不会有图像混合后不自然的情形.

- mosaic   
mosaic数据增强是参考CutMix数据增强，理论上类似.但cutmix使用2张图片，而mosaic则使用4张图片，其优点是丰富检测物体的背景，且在BN计算的时候一下子会计算四张图片的数据，使得mini-batch大小不需要很大，那么一个GPU就可以达到比较好的效果。  
![augmentations](https://user-images.githubusercontent.com/42667259/91769221-3ead4a80-ebdf-11ea-94bd-a18ed709f57c.png)

