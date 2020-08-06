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

### GoogLeNet, Christian Szegedy, Wei Liu et al.
- v1, 2014 ImageNet competition 1st. 

1. Apart from increasing the network depth, (22 layers for googlenet v1,) increase the width of the network

2. Introduce smaller kernels, 1 * 1 convolution, reduce the dimensions and save parameters. Parameters: AlexNet ~ 12 GoogLeNet, VGG ~ 3 AlexNet

3. Inception module is easy to add or remove, which is to mimic the human brain to create a sparce connection.

Refer to [Going deeper with convolutions](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43022.pdf)

![nn_googlenet_v1](https://user-images.githubusercontent.com/42667259/89532952-e9099c00-d7f2-11ea-9ac3-89cb14c4e1e6.png)

- v2, Christian Szegedy et al. v2, v3 share the same paper.

1. Factorizing Convolutions, use 1 * n and n * 1 to replace 3 * 3, shown as figure below. Theoritically, it can save computational cost dramatically when feature map is large (n is large). But in practice, it can not work well in the early layers, n ranges 12-20 seems to be a reasonable number. 卷积分解不适合早期大的特征层，而适合中期12-20大小的特征层

2. efficient grid size reduction. Use pooling layer (stride 2 in the following figure) and inception(convolution with stride 2 etc.) parallelly. 

3. propose some advices for the design of an efficient network: Avoid representational bottlenecks, especially early in the network; Higher dimensional representations are easier to process locally within a network; Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power (like RGB image to gray); Balance the width and depth of the network (This is further concluded in the recent efficientNet 2019). 早期特征尺寸不能急剧减小，避免出现瓶颈；低维特征时进行空间融合，并不会特别明显的增加损失（这感觉也像是可以进行特征融合的一个体现）

Refer to paper [Rethinking the Inception Architecture for Computer Vision](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)

![nn_googlenet_v2](https://user-images.githubusercontent.com/42667259/89534536-60d8c600-d7f5-11ea-8f99-afc2d13f2985.png#30x15)
![nn_googlenet_v2_2](https://user-images.githubusercontent.com/42667259/89537049-0d687700-d7f9-11ea-8d91-3a204d7e18a6.png#10*10)

- v3, shares the same paper with v2, minor additions.

Model Regularization via Label Smoothing, reduce the model overfitting. Training method: RMSProp to replace SGD. A resolution test was carried out.

Refer to paper [Rethinking the Inception Architecture for Computer Vision](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)

- v4 (pure inception-v4), inception-resnet, together in a paper. So here talk them together, mainly the inception-resnet

1. Combine the inception module with residual module to create a new module: inception-resnet. It increases the net depth and imporves the speed. 

2. Comparison: inception-v3 with inception-resnet-v1; inception-v4 with inception-resnet-v2, have similar accuracies.

Refer to paper [Inception-v4, inception-resnet and the impact of residual connections on learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14806)

### ResNet
1st place in all 5 competitions of ILSVRC and COCO 2015. 

It was found that deeper CNN is extremely useful for almost all the tasks. But they are difficult to train since the existence of degradation problem. A new architecture with residual module was proposed.

1. deeper network performs worse than shallower network. A creative idea: if nothing is learned, not worse than before: thus identity mapping, also known as a shortcut connection is proposed. H(x) = x + F(x), F(x) = H(x) - x, known as a residual.

2. different depth of ResNet, from 18 layers to 34, 50, 101, to extremely deep 152 layers.

3. different ways of shortcut, if dimension is not changed, identity mapping can be used. But in practice, dimension change, therefore, a "bottleneck" is created. (This is popular in the baselines afterwords, since it can save parameters.)

![nn_resnet](https://user-images.githubusercontent.com/42667259/89544188-19a50200-d802-11ea-8bff-88434e5bb831.png)
![nn_resnet_2](https://user-images.githubusercontent.com/42667259/89545365-8bca1680-d803-11ea-8b35-3b824952e96c.png)

Refer to paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
