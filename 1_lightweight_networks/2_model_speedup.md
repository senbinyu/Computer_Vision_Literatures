# Model Compression and Acceleration
#### This is a collection of papers focused on model compression and tricks, which are used for the speedup of calculations. If you want to know the area fastly, recommand to read the review papers, which can give you a whole picture. As shown below, model compression has different techniques, here I will mainly list the neural network part. 
![model_compression_techs](https://user-images.githubusercontent.com/42667259/89690882-e187e700-d907-11ea-91bb-f9bc1247e81b.png)

## 1. Review papers
Here are several papers recommanded for the model compression and acceleration. 

- Cheng Yu et al., A survey of model compression and acceleration for deep neural networks, 2017. 

Refer to paper [A survey of model compression and acceleration for deep neural networks](https://arxiv.org/abs/1710.09282)

- Cheng Jian et al., Recent advances in efficient computation of deep convolutional neural networks, 2018.

Refer to paper [Recent advances in efficient computation of deep convolutional neural networks](https://link.springer.com/content/pdf/10.1631/FITEE.1700789.pdf)

- Deng Yunbin, Deep learning on mobile devices: a review, 2019.

Refer to paper [Deep learning on mobile devices: a review](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10993/109930A/Deep-learning-on-mobile-devices-a-review/10.1117/12.2518469.pdf?casa_token=1vnmem4EqK0AAAAA:xqMq8QcEwl66yyIn8hiChVZBu8BbOPHfYmzND2N1732iHPhEVfAKfxPwrUDRoBwDLDW-BMtgBQ)

- Choudhary, Tejalal et al., A comprehensive survey on model compression and acceleration, 2020.

Refer to paper [A comprehensive survey on model compression and acceleration](https://link.springer.com/content/pdf/10.1007/s10462-020-09816-7.pdf)

## 2. Model compression and speedup approaches
Papers above have some overlapping areas, so in this section, I listed the survey into the following categories. Apart from the approaches below, Cheng Jian et al. also tried to speed up through the hardware accelerators. Most of these methods are independently designed and complementary to each other. For example, transferred layers and parameter pruning & quantization can be deployed together. Another example is that, model quantization & binarization can be used together with low-rank approximations to achieve further compression/speedup.

![compression_approaches](https://user-images.githubusercontent.com/42667259/89688580-c8c90280-d902-11ea-82b1-72fdd6006b20.png)

### 2.1 Parameter pruning and quantization
Here I listed some classic papers for different approaches.

Early works [1] showed that network pruning and quantization are effective in reducing the network complexity and addressing the over-fitting problem. After found that pruning can bring regularization to neural networks and hence improve generalization, it has been widely studied to compress DNNs. 早期工作发现模型瘦身能抑制过拟合，之后将其用于压缩DNN. 

[1] Gong et al., 2014, [Compressing deep convolutional networks using vector quantization](https://arxiv.org/abs/1412.6115)
#### 2.1.1 Pruning
1. Fine-grained pruning (细粒度剪枝). Han et al. [2] proposed a deep compression framework to compress DNNs in three steps: pruning, quantization, and Huffman encoding. By using this method, AlexNet could be compressed by 35-fold without drops in accuracy. However, the compressed model has an accuracy drop.  Guo et al. [3] proposed a dynamic network surgery framework consisting of two operations: pruning and splicing. The pruning operation aims to prune those unimportant parameters while the splicing operation aims to recover the incorrectly pruned connections. Their method requires fewer training epochs and achieves a better compression ratio. (这是一种non-structure的压缩行为，因此压缩效率不高，且在硬件上的运行效率也不高)

[2] Han et al., 2015, [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)
[3] Guo et al., 2016, [Dynamic network surgery for efficient dnns](http://papers.nips.cc/paper/6165-dynamic-network-surgery-for-efficient-dnns.pdf)

2. Filter-level pruning.
Fiter is connected to channels number, He et al. [4] introduced a selection weight \beta for each filter and then added sparse constraints on \beta. ()

[4] He et al., 2017, [Channel Pruning for Accelerating Very Deep Neural Networks](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Channel_Pruning_for_ICCV_2017_paper.pdf)





