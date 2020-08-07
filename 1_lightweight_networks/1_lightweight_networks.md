##### collections of some popular light neural networks architectures design, which are widely used in mobiles etc.

## 1. Review papers
Review papers of lightweight  networks usually include the model compression and pruing tricks, which are further detailed described in ../2_model_speedup

- Cheng Yu et al., A survey of model compression and acceleration for deep neural networks, 2017. 

Refer to paper [A survey of model compression and acceleration for deep neural networks](https://arxiv.org/abs/1710.09282)

- Cheng Jian et al., Recent advances in efficient computation of deep convolutional neural networks, 2018.

Refer to paper [Recent advances in efficient computation of deep convolutional neural networks](https://link.springer.com/content/pdf/10.1631/FITEE.1700789.pdf)

- Li Yahui et al., Lightweight Network Research Based on Deep Learning: A Review, 2018

Refer to paper [Lightweight Network Research Based on Deep Learning: A Review](https://ieeexplore.ieee.org/abstract/document/8483963?casa_token=Ro1rJdUIkXoAAAAA:GukzNyQi38qOA4v-B6394PvpFp6R3j0tvvXDNCKnIPkYf5EEA_GtTSfWdvK19WS9Zl4lgP5-mA)

## 2. Collections of varying networks

### SqueezeNet

1. smaller neural networks has many advantages: e.g., require less communication across servers during distributed training; require less bandwidth to export a new model from the cloud to an autonomous car; are more feasible to deploy on FPGAs and other hardware with limited memory. SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters.

2. strategies, replace 3 * 3 kernels with 1 * 1, decreases the input channels for 3 * 3 kernels. These two ways are trying to reduce the parameters amount. Downsample late in the network so that convolution layers have large activation maps. This is trying to maintain the accuracy as high as possible.

3. *fire module*, first layer, squeeze, 1 * 1 kernels, then ReLU second layer, expansion, 1 * 1 and 3 * 3 kernels together, then ReLU. 

Refer to paper [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size](https://arxiv.org/abs/1602.07360)
![nn_squeezeNet](https://user-images.githubusercontent.com/42667259/89582872-a3bc8d00-d839-11ea-9180-5cbc5e55ffa5.png)

### Xception, Chollet, Francois
Based on Inception-v3, since it used depthwise separable convolution, many researchers consider it as a lightweight network. If we do not want to design the inception architecture every time, just use the same structure evenly at one module, it will be easier for the network design. (This idea is popular in the other baselines, i.e., mobileNet.)

1. extreme inception. from equivalent inception structure, now only calculate a part of the channels, (group = xx in pytorch).

2. depthwise separable convolution (originally from a phd thesis: Laurent Sifre, Rigid-Motion Scattering For Image Classification), save parameters greatly. By separating features evenly, the parameters in figure below can be m * k + 3 * 3 * k, m is the features, k is the kernels number.
Details can also see mobileNet below.

Refer to paper [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
![nn_xception](https://user-images.githubusercontent.com/42667259/89560940-17e63900-d818-11ea-95d1-bb73602a132c.png)

### MobileNet, Andrew G. Howard et al.
- v1, efficient models for mobile and embedded vision applications. Compared to squeezeNet, parameters much fewer and accuracy higher.

1. Depthwise separable convolution: two parts, 1) depthwise convolution, for M channels, we have M kernels, each channel corresponds to one kernel; 2) pointwise convolution, we use 1 * 1 kernel to have convolution, thus it is pointwise, see figure below. These two operations can save parameters. Originally, it is M * N * D_K * D_K * imageSize, now it is M * D_K * D_K * imageSize + N * M * imageSize. M is the input channels, N is the output channels

2. thinner models, i.e., model compression. model shrink hyperparameters, $\alpha$, used for input and output channels. $\rho$ for image resolution, but $\rho$ only decreases the FLOPs, but not the parameters amounts.

3. MobileNet can be used in many areas, e.g., object detection, large scale geolocalization, face attributes. In Face attributes, distillation is validated, small networks can behave similar to the large models.

Refer to paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
![nn_mobileNet](https://user-images.githubusercontent.com/42667259/89632812-20d01c80-d8a3-11ea-986a-146a3a132413.png)

- v2, Mark Handler, Andrew Howard et al.
v1 has some problems, depthwise conv has many 0 kernels, which means the training is not good enough. (depthwise训出来的kernel有不少是空的，过小的kernel size加上ReLU的激活影响，0的部分无法恢复)

The innovative parts are inverted residual structures + linear bottleneck, obtain satisfactory results vs mobileNet-v1: 75ms vs 113ms; 3.4M vs 4.2M parameters; 72 vs 70.6 Top1 on ImageNet.

1. inverted residual structure, 1 * 1 expansion the channels + ReLU6, 3 * 3 depthwise conv + ReLU6, and then 1 * 1 with linear func, remove the activation to maintain the manifolds of interest, the *inverted residual with linear bottleneck*. (low dimension, ReLU would destory the manifolds of interests, so use linear func here)

Refer to paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html)
![nn_mobileNet_v2](https://user-images.githubusercontent.com/42667259/89647200-75809100-d8bd-11ea-973c-89528028c09d.png)

- v3, Andrew Howard et al.
We want to reduce the parameters amounts, as well as the MAdds, inference time. MobileNet-v3 is much better than v2 in speed and accuracy. For example, MobileNetV3-Large is 3.2% more accurate on ImageNet classification while reducing latency by 20% compared to MobileNetV2.

1. MobileNet-v2 + SE block (see figure below), SE learns the importance of each channel. 

2. blend automated search with novel architecture advances to build effective models. NAS for block-wise search and NetAdapt for Layer-wise Search, which are platform-aware. In layer-wise search, use smaller expansion size.

3. network improvement, i.e., redesigning expensive layers. Some of the last layers are expensive. Modify them to reduce the latency while maintaining the accuracy. Hard swish activation func is used in the second half (deeper layers) to deal with the nonlinearity, imporving accuracy.

Refer to paper [Searching for mobilenetv3](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html)
![nn_mobileNet_v3](https://user-images.githubusercontent.com/42667259/89651436-399cfa00-d8c4-11ea-8fa8-918e069c17b6.png)
![nn_mobileNet_v3_2](https://user-images.githubusercontent.com/42667259/89652091-3bb38880-d8c5-11ea-9a0c-f0bde0ff9fc7.png)

### shuffleNet
- v1, Zhang Xiangyu
In ResNeXt, group conv (several channels) is used; in mobileNet, it is an extreme case, each channel is a group, is conved separately. In mobileNet-va paper, 1 * 1 conv (pointwise conv) occupies 94% MAdds. ShuffleNet is trying to avoid this problem. Moreover, the relationship between different channels has not been utilized. (组内的卷积不利于通道间信息交流)

1. channel shuffle, together with group conv. use 1 * 1 group conv to replace the 1 * 1 standard conv, to save MAdds. group *g* usually is not too large, e.g., 1,2,3,4,8 etc. Too many groups may lead to the input channels in each group too less, which may cause worse performance.

2. Comparisons between with shuffle and without shuffle, and shuffle with mobilenet. shuffleNet has betetter performance than VGG, ResNet, ResNeXt and mobileNet when they are keeping the similar network size.

Refer to paper [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.html)
![nn_shuffleNet](https://user-images.githubusercontent.com/42667259/89659653-740c9400-d8d0-11ea-8f05-d7627fb2afe2.png)

- v2, Ma Ningning, Zhang Xiangyu
FLOPs is only one of many factors to influence the model speed etc. Here, the authors proposed more practical guidelines for the efficient cnn architecture design. 

1. Use Memory Access Cost (MAC parameter) to measure the memory cost. MAC needs to be smaller to achive higher speed. There are some observations: channel numbers try to be the same (bottleNeck in shuffleNet-v1 counters this); group *g* can not be too large (try more groups in shuffleNet-v1); Network fragmentation reduces degree of parallelism (mobileNet v2 goes against); Element-wise operations are non-negligible (depthwise use ReLU6 also contradicts).

2. Therefore, did some modifications as shown in the following figure. These operations are all following the guidelines above. For instance, repalce the 1 * 1 group conv with 1 * 1 conv again to follow the second rule, channel number try to be kept same. Under these guidelines, shuffleNet-v2 has got satisfactory results both on accuracy and speed.

3. relate to densetNet and condenseNet. half of feature channels (when c′ = c/2) directly go through the block and join the next block, like the *feature reuse* in denseNet and condenseNet. Both of them are supporting that the connections between the adjacent layers are stronger than the others

Refer to paper [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html)
![nn_shuffleNet_v2](https://user-images.githubusercontent.com/42667259/89663337-b5ec0900-d8d5-11ea-8964-074fb5f2693b.png)

### MNASNet, Tan Mingxing et al.
mobile NASNet, 1.8x speed of mobileNet-v2 on ImageNet with 78ms inference latency on a pixel phone. But the search of architecture is computation costly.

1. use a reinforcement learning approach to search the architecture, and the accuracy is mainly dependent on the design of architecture without modifing the parameters a lot. This indicates that the searched architecture is good. But the architecture is similar to MobileNet-v2, which also implies that human design is also impressive, although the search sturcture is mimicing the mobileNet. 说明人工设计的网络架构也是非常精简有效的

Refer to paper [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.html)
![nn_mnasnet](https://user-images.githubusercontent.com/42667259/89627578-64bf2380-d89b-11ea-8cea-bcd923ace03b.png)

