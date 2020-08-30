In training procedure, the choose of optimizer and loss function is of great importance, as well as the choose of neural networks.

## Optimizer



## Loss function
For different tasks, there are different settings of loss functions. For instance, in segmentation, loss function are separated into three categories: distribution-based, region-based and compound loss functions. 

从本质上来讲，目标检测和图像分割总是可以分为分类和回归这两个基本任务。例如目标检测中，分类总是需要有分类损失函数，而bounding box的损失函数常看做是回归框的回归损失，可以用基于区域的IOU, GIOU, CIOU来衡量。不过归根结底，都可以说成是分类和回归结合在一起的复合损失函数。图像分割中，也经常需要用到分类（常用的二分类），和基于回归mask，即分割区域的Dice loss等，也可以看成是一种回归损失。

具体的损失函数种类可以查看object detection中和segmentation中的loss function章节，有对应的详细介绍。


