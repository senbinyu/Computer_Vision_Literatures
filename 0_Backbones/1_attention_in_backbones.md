## Attention mechanism in backbones

1. channel attention: SENet is a good example. Figure is shown below. U(C * H * W) comes from from X(C' * H' * W*). The features U are first passed through a squeeze operation, which aggregates the feature maps across spatial dimensions H × W to produce a channel descriptor. This descriptor embeds the global distribution of channel-wise feature responses, enabling information from the global receptive field of the network to be leveraged by its lower layers. This is followed by an excitation operation, in which
sample-specific activations, learned for each channel by a self-gating mechanism based on channel dependence, govern the excitation of each channel. The feature maps U are then reweighted to generate the output of the SE block which can then be fed directly into subsequent layers.
从X(C' * H' * W*)到U(C * H * W)和原先一致，随后对U做一个全局平均，由此其空间上的信息全部被压缩为1 * 1 * C，C是通道，即为squeeze过程。之后再用Fex对其进行激活，即为文中公式3，采用了sigmoid函数内包ReLU,

![nn_senet](https://user-images.githubusercontent.com/42667259/90159719-145f2e80-dd91-11ea-9e68-95dc1cb70dc7.png)

2. spatial attention:
