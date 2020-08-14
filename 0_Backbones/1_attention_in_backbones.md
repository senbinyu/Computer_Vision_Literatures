## Attention mechanism in backbones

1. channel attention: SENet is a good example. Figure is shown below. U(C * H * W) comes from from X(C' * H' * W*). The features U are first passed through a squeeze operation, which aggregates the feature maps across spatial dimensions H × W to produce a channel descriptor. This descriptor embeds the global distribution of channel-wise feature responses, enabling information from the global receptive field of the network to be leveraged by its lower layers. This is followed by an excitation operation, in which sample-specific activations, learned for each channel by a self-gating mechanism based on channel dependence, govern the excitation of each channel. The feature maps U are then reweighted to generate the output of the SE block which can then be fed directly into subsequent layers.
从X(C' * H' * W*)到U(C * H * W)和原先一致，随后对U做一个全局平均，由此其空间上的信息全部被压缩为1 * 1 * C，C是通道，即为squeeze过程。之后再用Fex对其进行激活，即为文中公式3，采用了sigmoid函数内包ReLU函数进行激活（用sigmoid确保在[0,1]内），至此可以确定哪些是比较重要的channel了。但为了和之前的U结合到一块，再进行一个Fscale（文中公式4）。由此整个Squeeze-excitation模块结束，这里可以让那些重要的通道在激活函数作用下变得更为突出

![nn_senet](https://user-images.githubusercontent.com/42667259/90159719-145f2e80-dd91-11ea-9e68-95dc1cb70dc7.png)

2. spatial attention: in CBAM: Convolutional Block Attention Module, Woo et al., 2018, Korea Advanced Institute of Science and Technology
From the figure below, we can see that for spatial attention, C * H * W becomes 1 * H * W using average and max pooling, which squeeze the channel and obtain the spatial info already containing all the channel info. Through 7 * 7 conv followed by a sigmoid activation func, they get weight matrix Ms, which can be used for the multiplication with feature map C * H * W to finally obtain a new feature. 
与通道注意力相似，对特征F：C * H * W进行通道维度的平均池化和最大池化得到两个 H×W×1 的通道描述，并将这两个描述按照通道拼接在一起。然后，经过一个 7×7 的卷积层，激活函数为 Sigmoid，得到权重系数 Ms。最后，拿权重系数和特征 F 相乘即可得到缩放后的新特征。

see https://zhuanlan.zhihu.com/p/65529934
![nn_cbam_2](https://user-images.githubusercontent.com/42667259/90161966-38703f00-dd94-11ea-91a1-08367238c6fa.png)

3. samples attention
Personal thinking, for the hard negative samples, we use focal loss to focus on them. Utilize the hard examples to train a more robust model.
个人将focal loss等对hard negative samples的侧重选择归类为对样本的attention，通过各种方法将hard negative samples更多的放入到训练池中，以让模型的学习能力增强
