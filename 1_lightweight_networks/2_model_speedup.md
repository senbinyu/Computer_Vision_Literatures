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
Fiter is connected to channels number, He et al. [4] introduced a selection weight \beta for each filter and then added sparse constraints on \beta. (其实mobileNet-v1中也有类似的行为，将通道数目缩减一些，从而进行网络瘦身)

[4] He et al., 2017, [Channel Pruning for Accelerating Very Deep Neural Networks](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Channel_Pruning_for_ICCV_2017_paper.pdf)

3. Vector-level and kernel-level prunings. 
There is little work. Anwar et al. [5] first explored kernel-level pruning, and then proposed an intra-kernel strided pruning method, which prunes a sub-vector in a fixed stride. Mao et al. [6] explored different granularity levels in pruning, and found that vector-level pruning takes up less storage than fine-grained pruning because vector-level pruning requires fewer indices to indicate the pruned parameters. It is more like a structural pruning than fine-grained pruning and more friendly to memory access and thus more efficient in hardware implementations. 这种方法更像是组织（成批型）的剪枝方法，简单有效，在硬件上更有效。

[5] Anwar et al., 2017, [Structured pruning of deep convolutional neural networks](https://dl.acm.org/doi/pdf/10.1145/3005348)
[6] Mao et al., 2017, [Exploring the regularity of sparse structure in convolutional neural networks](https://arxiv.org/pdf/1705.08922.pdf)

4. Group-level pruning.
Lebedev et al. [7] proposed the group-wise brain damage approach, which prunes the weight matrix in a group-wise fashion (shown in the following figure). Wen et al. [8] added a structured sparsity regularizer on each layer to reduce trivial filters, channels or even layers. 
![pruning_groupwise](https://user-images.githubusercontent.com/42667259/89734073-07250580-da5a-11ea-815a-8e1d544a7409.png)

[7] Lebedev et al., 2016, [Fast ConvNets using groupwise brain damage](https://openaccess.thecvf.com/content_cvpr_2016/papers/Lebedev_Fast_ConvNets_Using_CVPR_2016_paper.pdf)
[8] Wen et al. 2016, [Learning structured sparsity in deep neural networks](https://papers.nips.cc/paper/6504-learning-structured-sparsity-in-deep-neural-networks.pdf)

#### 2.1.2 Quantization
1. Scalar and vector quantization.
By using scalar or vector quantization, the original data can be represented by a codebook and a set of quantization codes with quantization centers. Of course, the number of quantization centers is always less than the original data to achieve compression. Gong et al. [9] used k-means algorithm to compress the parameters. Wu et al. [10] proposed using the PQ algorithm to simultaneously accelerate and compress convolutional neural networks. "During the inference phase, a look-up table was built by precomputing the inner product between feature map patches and codebooks, and then the output feature map can be calculated by simply accessing the lookup table. " (4 to 6 times speed up,huffman编码也有类似的效果，查表法在我博士工作中也经常使用)

[9] Gong et al., 2014 [Compressing deep convolutional networks using vector quantization](https://arxiv.org/pdf/1412.6115.pdf)
[10] Wu et al., 2016 [Quantized convolutional neural networks for mobile devices](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Wu_Quantized_Convolutional_Neural_CVPR_2016_paper.pdf)

2. Fixed-point quantization
Here are many researches studying the bits on the CNN performance. From 8-bit to 16, 32 bit, it seems the convergece becomes better. But recentlly, Dettmers et al. [11] found that 8-bit fixed point quantization to speed up convergence. As an extreme case, the binaryConnect method with weights of +1 and -1 can even outperforms some networks. Rastegari et al. [12] proposed the binary weight network (BWN), which was among the earliest work that achieved good results on the large ImageNet dataset. 点位的减少使得模型在计算时可以占用更少的空间。值得注意的是，二值网络有时甚至能表现得比某些网络更强，这也有可能是因为其相当于加入了正则化，使得模型的泛化能力更强了

[11] Dattmers et al., 2015, [8-bit approximations for parallelism in deep learning](https://arxiv.org/pdf/1511.04561.pdf):)
[12] Rastegari et al., 2016, [XNORNet: ImageNet classification using binary convolutional neural networks](https://arxiv.org/pdf/1603.05279.pdf?source=post_page---------------------------)

### 2.2 Low-rank approximation
Convolution operations contribute the bulk of most computations in deep DNNs, thus reducing the convolution layer would improve the compression rate as well as the overall speedup. For instance, we have w * h * c * n conv kernel, parameters corresponds to kernel width, kernel height, and the numbers of input and output channels. Since a conv layer may have much abundant infomation, we do not need to use conv directly, we can use SVD principle to decompose them into low-ranked matrix. Of course, there are varying methods to decompose from different parameters. For example, Zhang et al. replaced the filter by two filter banks: one consisting of d filters of shape w · h · c and the other composed of n filters of shape 1 × 1 · d, where d represents the rank of the decomposition; i.e., the n filters are linear combinations of the first d filters. (3 times speed up but 1.66% higher top-5 error). But the computation is expensive since decomposition operation; and we have to perform low-rank layer-by layer as different layers have different information; and it also requires extensive model retraining to achieve convergence when compared to
the original model.
如果一个网络中所含有价值的信息量并没有那么多，那么相当于参数矩阵的秩r比较小，所以原矩阵m * n可以分解成m * r和r * n矩阵（类似于SVD分解），因为r<<m or n，所以其参数量大大缩小，这也是下图所示的原理。但此法因为要每层进行分解，计算量大，而且还需要retraining以达到收敛，这些都限制了此法的广泛应用。

[13] Zhang et al., 2015 [Accelerating very deep convolutional networks for classification and detection](https://ieeexplore.ieee.org/abstract/document/7332968)
![low_rank](https://user-images.githubusercontent.com/42667259/89735628-4eb08f00-da64-11ea-8232-4bcb1f96cc84.png)

### 2.3 Compact network design


