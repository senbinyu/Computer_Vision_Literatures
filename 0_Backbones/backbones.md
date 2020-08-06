Backbones are the most important architecture for the neural networks. 

## 1. Reviews

## 2. History

A brief figure below shows the time line of the nerual network development
![nn_his](https://user-images.githubusercontent.com/42667259/89516735-1b0e0480-d7d9-11ea-8d88-61a3556b0b5f.jpg)

### - Lenet, Yann LeCun et al.
LeNet-5, this is a simple but creative network, originally used for the handwriting recognition. please refer to http://yann.lecun.com/exdb/lenet/
![nn_lenet](https://user-images.githubusercontent.com/42667259/89517382-ec445e00-d7d9-11ea-9b77-cb9493af3d19.png)

### - AlexNet, Alex Krizhevsky et al.
Won the ImageNet Large Scale Visual Recognition Challenge on September 30, 2012. The network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower than that of the runner up. The primary finding was that the depth of the neural network was essential for the high performance of detection. 
网络的深度对于性能的提升具有极其关键的作用，并且伴随着GPU的逐步广泛使用，使用深度大的网络成为现实。
Kernels are relatively large, 11 * 11, 7 * 7, 5 * 5 etc., see from the figure 
refer to paper [ImageNet Classification with Deep Convolutional Neural Networks] (https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
![nn_lenet](https://user-images.githubusercontent.com/42667259/89520673-80182900-d7de-11ea-8e38-c95c03bbb19c.png)


### - VGGNet, Karen Simonyan, Andrew Zisserman
2014 Imagenet competition 2nd. 
1. Validate that increasing the net depth can improve the performance effectively. But this bring a problem: great amount of parameters. 
2. decrease the flame kernel size, two 3 * 3 replace 5 * 5, decrease the parameters amounts
证明网络深度增加有助于检测，但引入更多的参数；于是发现了使用小卷积核能达到和使用大卷积核同样的目的，同时还能减少参数
refer to paper [Imagenet classification with deep convolutional neural networks](https://arxiv.org/abs/1409.1556)
![nn_vgg](https://user-images.githubusercontent.com/42667259/89520223-d46ed900-d7dd-11ea-9554-99f9603fd6e0.png)


