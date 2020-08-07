##### collections of some popular light neural networks architectures design, which are widely used in mobiles etc.

## 1. Review papers
Review papers of lightweight  networks usually include the model compression and pruing tricks, which are detailed described in ../2_model_compression_speedup

Refer to paper [Lightweight Network Research Based on Deep Learning: A Review](https://ieeexplore.ieee.org/abstract/document/8483963?casa_token=Ro1rJdUIkXoAAAAA:GukzNyQi38qOA4v-B6394PvpFp6R3j0tvvXDNCKnIPkYf5EEA_GtTSfWdvK19WS9Zl4lgP5-mA)

## 2. Collections of varying networks

### SqueezeNet

1. smaller neural networks has many advantages: e.g., require less communication across servers during distributed training; require less bandwidth to export a new model from the cloud to an autonomous car; are more feasible to deploy on FPGAs and other hardware with limited memory. SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters.

2. strategies, replace 3 * 3 kernels with 1 * 1, decreases the input channels for 3 * 3 kernels. These two ways are trying to reduce the parameters amount. Downsample late in the network so that convolution layers have large activation maps. This is trying to maintain the accuracy as high as possible.

3. *fire module*, first layer, squeeze, 1 * 1 kernels, then ReLU second layer, expansion, 1 * 1 and 3 * 3 kernels together, then ReLU. 

Refer to paper [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size](https://arxiv.org/abs/1602.07360)
![nn_squeezeNet](https://user-images.githubusercontent.com/42667259/89582872-a3bc8d00-d839-11ea-9180-5cbc5e55ffa5.png)

### MobileNet, Andrew G. Howard et al.
- v1, efficient models for mobile and embedded vision applications.

1. Depthwise separable convolution, 

Refer to paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)


### shuffleNet

Refer to paper []()

### MNASNet, Tan Mingxing et al.
mobile NASNet, 1.8x speed of mobileNet-v2 on ImageNet with 78ms inference latency on a pixel phone. But the search of architecture is computation costly.

1. use a reinforcement learning approach to search the architecture, and the accuracy is mainly dependent on the design of architecture without modifing the parameters a lot. This indicates that the searched architecture is good. But the architecture is similar to MobileNet-v2, which also implies that human design is also impressive, although the search sturcture is mimicing the mobileNet. 说明人工设计的网络架构也是非常精简有效的

Refer to paper [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.html)
![nn_mnasnet](https://user-images.githubusercontent.com/42667259/89627578-64bf2380-d89b-11ea-8cea-bcd923ace03b.png)

