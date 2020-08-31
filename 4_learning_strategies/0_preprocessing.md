Learning strategies are also very important to the training and testing. Firstly, preprocessing the data, like image augmentation is sometimes crucial for the prediction of reasonable results.


## 验证集划分
![数据验证集划分](https://user-images.githubusercontent.com/42667259/91734531-5539ae80-ebab-11ea-8d29-71632e3be6a7.png)  
- 直接划分：随机划分的方式使得模型的训练数据可能和测试数据差别很大，导致训练出的模型泛化能力不强。  
- LOOCV: Leave-one-out cross-validation，这相当于是k折交叉验证的一个极端情况，即K=N。每次只用一个数据作为测试，其他均为训练集，重复N次（N为数据集数目）  
- kFold，k折交叉验证，每次的测试集将不再只包含一个数据，而是多个，具体数目将根据K的选取决定。比如，如果K=5，那么我们利用五折交叉验证的步骤就是：1）将所有数据集分成5份；2）不重复地每次取其中一份做测试集，用其他四份做训练集训练模型，之后计算该模型在测试集上的Error_i；3）将5次的Error_i取平均得到最后的Error。

## image augmentation,图像or数据增强
