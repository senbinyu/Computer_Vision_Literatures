Loss function is the representation of the optimal target. But most of the time we can not use the optimal target directly, so we choose the loss function to approach the optimal target as much as possible.
损失函数是最优化目标的一种代表，大多数情况下，我们无法直接用最优化目标，故用损失函数来替代。因此，如何选择一个损失函数，以让他和最优化目标更为接近显得极为重要。

# 1. Review paper

Ma et al., 2020, Nanjing University of Science and Technology, [Segmentation Loss Odyssey](https://arxiv.org/pdf/2005.13449.pdf)

# 2. Loss functions
This figure gives out the relations between all the loss functions.
![loss_relation](https://user-images.githubusercontent.com/42667259/90231756-e96dec80-de1b-11ea-8111-fb57bdf4974f.png)

## 2.1 Distribution-based Loss
Distribution-based loss functions aim to minimize dissimilarity between two distributions. The most fundamental function in this category is cross entropy; all
other functions are derived from cross entropy.
基于分布的损失函数旨在最小化两个分布之间的差异， 此类别中最基本的是交叉熵。 所有其他函数都可以看做是推导自交叉熵。

Cross entropy (CE) is derived from Kullback-Leibler (KL) divergence, which is a measure of dissimilarity between two distributions. 

