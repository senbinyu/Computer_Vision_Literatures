Here are some collections of classic and state-of-art papers working on image segmentation by using deep learning.

# 1. Review papers

Guo et al., Leiden University, 2018, [A review of semantic segmentation using deep neural networks](https://link.springer.com/content/pdf/10.1007/s13735-017-0141-z.pdf)

*Recommand* Minaee et al., Snapchat Inc, 2020.01, [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/pdf/2001.05566.pdf)

Here is a figure to show the difference of two different segmentations. 此图承前启后，理解几个任务的不同
![semantic_instance_seg](https://user-images.githubusercontent.com/42667259/89907862-7b43e280-dbed-11ea-9851-0089f68671a6.png)

# 2. Image segmentation
## 2.1 Fully Convolutional Networks
- FCN, Long et al., 2015, Berkeley, [Fully convolutional networks for semantic segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)  
Milestone in image segmentation, demonstrating that deep networks can be trained for semantic segmentation in an end-to-end manner on variablesized images. The authors changed the fully connected layer into fully convolutional layer, so then the network can output spatial segmentation map instead of classification scores. However, it
is not fast enough for real-time inference, it does not take into account the global context information in an efficient way, and it is not easily transferable to 3D images. FCN的工作非常有意义，将一般的CNN最后的FC层改造成全卷积网络，这样就可以直接输出图像的分割，实现了端到端，但效率不高，且无法有效地将全局的语境信息考虑，不易转移到3D图像。

- ParseNet, Liu et al., 2015, UNC Chapel Hill, [Parsenet: Looking wider to see better](https://arxiv.org/pdf/1506.04579.pdf) 
ParseNet adds global context to FCNs by using the average feature for a layer to augment the features at each location. The feature map for a layer is pooled over the whole image resulting in a context vector. This context vector is normalized and unpooled to produce new feature maps of the same size as the initial ones. These feature
maps are then concatenated. 可以参照下图，从feature layer中通过global pooling得到全局语境信息，随后L2 norm，再到最后unpooling回去，可以弥补上述FCN模型的无全局语境信息，提高精确率

![seg_parseNet](https://user-images.githubusercontent.com/42667259/89914256-0b395a80-dbf5-11ea-86db-be9133cc623c.png)

## 2.2 Convolutional Models With Graphical Models
- CNN + CRFs, Chen et al., 2015, UCLA+Google, [Semantic image segmentation with deep convolutional nets and fully connected crfs](https://arxiv.org/pdf/1412.7062.pdf)  
They showed that responses from the final layer of deep CNNs are not sufficiently localized for accurate object segmentation (but it is good at classification). To overcome the poor localization property of deep CNNs, they combined the responses at the final CNN layer with a fully-connected CRF. They showed that their model is able to localize segment boundaries at a higher accuracy rate than it was possible with previous methods. 结合了CNN的语义信息和crf的位置信息，如下图所示，upsample后进行双线性插值结合CRF最后得到分割图。
![seg_cnn_crf](https://user-images.githubusercontent.com/42667259/89916476-eb576600-dbf7-11ea-94e4-89b2b6bd8587.png)

- Liu et al., 2015, The Chinese University of Hong Kong, [Semantic image segmentation via deep parsing network](https://openaccess.thecvf.com/content_iccv_2015/papers/Liu_Semantic_Image_Segmentation_ICCV_2015_paper.pdf)
a semantic segmentation algorithm that incorporates rich information into MRFs, including highorder relations and mixture of label contexts. Unlike previous works that optimized MRFs using iterative algorithms, they proposed a CNN model, namely a Parsing Network, which enables deterministic end-to-end computation in a single forward pass.

## 2.3 Encoder-Decoder Based Models
Most of the DL-based segmentation works use some kind of encoder-decoder models. 
- Noh et al., 2015, POSTECH, Korea, [Learning deconvolution network for semantic segmentation](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Noh_Learning_Deconvolution_Network_ICCV_2015_paper.pdf)  
The network is based on VGG-16 and consists of convolution and deconvolution nets. Between two nets, two fc layers are shown in the figure below. The deconvolution network is composed of deconvolution and unpooling layers. 
![seg_encoder_decoder](https://user-images.githubusercontent.com/42667259/89919589-c238d480-dbfb-11ea-816f-e2a0f547e696.png)

- SegNet, Badrinarayanan, 2017, University of Cambridge, [Segnet: A deep convolutional encoder-decoder architecture for image segmentation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7803544)  
SegNet replace the middle fc layers with fully convolutional layers. The main novelty of SegNet is in the way the decoder upsamples its lower resolution input feature map(s); specifically, it uses pooling indices computed in the max-pooling step of the corresponding encoder to perform non-linear upsampling. This eliminates the need for learning to up-sample. 创新点在于上采样的时候利用下采样（max pooling）记录下的最大值的index，而无需再次进行训练

- U-Net, Ronneberger et al., 2015, University of Freibur, [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)%e5%92%8c%5bTiramisu%5d(https://arxiv.org/abs/1611.09326.pdf)  
This network originally proposed for the segmentaion of medical images, now it is popular in all areas segmentation tasks. The net comprises two parts, a contracting
path to capture context, and a symmetric expanding path that enables precise localization. The detailed process can be seen int the figure below. 对称结构，左右连接，将语义信息和位置信息结合在一起，最后上采样得到图片，再用1 * 1卷积核输出segmentation图
![seg_unet](https://user-images.githubusercontent.com/42667259/89920792-1f815580-dbfd-11ea-85e6-76a0070b90bb.png)


- V-Net, Milletari et al., 2016, Technische Universitat Munchen + Johns Hopkins University, [V-net: Fully convolutional neural networks for volumetric medical image segmentation](https://arxiv.org/pdf/1606.04797.pdf)  
It is used for 3D volumetric image segmentation. The authors introduced a new objective function based on the Dice coefficient, enabling the model to deal with situations in which there is a strong imbalance between the number of voxels in the foreground and background. 
![seg_vnet](https://user-images.githubusercontent.com/42667259/89922572-843daf80-dbff-11ea-8a2c-6c98fd20211f.png)

## 2.4 Multi-Scale and Pyramid Network Based Models
- FPN, Lin et al., 2017, Facebook, [Feature Pyramid Networks for Object Detection](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)  
originally for object detection, but can be extended for image segmentation. Details can be seen in the figure below. 这是文章的一个extend内容，里面简单验证了想法可行
![seg_fpn](https://user-images.githubusercontent.com/42667259/89923638-290cbc80-dc01-11ea-8cae-8898e2b104f6.png)

- Pyramid Scene Parsing Network (PSPNet), Zhao et al., 2017, The Chinese University of Hong Kong, [Pyramid scene parsing network](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf)  
Different patterns are extracted from the input image using a residual network (ResNet) as a feature extractor, with a dilated network. These feature maps are then fed into a pyramid pooling module to distinguish patterns of different scales. They are pooled at four different scales, each one corresponding to a pyramid level and processed by a 1 × 1 convolutional layer to reduce their dimensions. The outputs of the pyramid levels are up-sampled and concatenated with the initial feature maps to capture both local and global context information. Finally, a convolutional layer is used to generate the pixel-wise predictions.结合下图，前面用ResNet提取特征（空洞卷积），随后用4种scale的pool，然后用1 * 1卷积核将其channel减少，然后和之前的feature map拼接输出。
![seg_pspnet](https://user-images.githubusercontent.com/42667259/89925511-d2ed4880-dc03-11ea-916d-e83bc6d2be30.png)

## 2.5 R-CNN Based Models (for Instance Segmentation)
- Mask RCNN, He et al., 2017, Facebook, [Mask R-CNN](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)  
This model efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. Mask R-CNN is essentially a Faster RCNN with 3 output branches: the first computes the bounding box coordinates, the second computes the associated classes, and the third computes the binary mask to segment the object. The Mask R-CNN loss function combines the losses of the bounding box coordinates, the predicted class, and the segmentation mask, and trains all of them jointly. Mask RCNN相当于有3个输出的faster rcnn，分别是目标框的坐标，分类以及segmentation的二值mask。
![seg_mask_rcnn](https://user-images.githubusercontent.com/42667259/89926490-45125d00-dc05-11ea-8612-c54cf1273ebc.png)

- PANet, Liu et al., 2018, The Chinese University of Hong Kong + Peking University + Tecent, [Path Aggregation Network for Instance Segmentation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Path_Aggregation_Network_CVPR_2018_paper.pdf)  
It is based on the Mask R-CNN and FPN models. The feature extractor of the network uses an FPN architecture with a new augmented bottom-up pathway improving the propagation of low-layer features. The output is added to the same stage feature maps of the top-down pathway using a lateral connection and these feature maps feed the next stage. The third processes the RoI with an FCN+fc layer to predict the object mask, as shown in (e) below. 作者发现底层的位置信息要通过太长的路到达最深处，因此缩短信息流通距离；还有就是聚合所有的feature proposals到一起做决策（利用adaptive pooling）；在(e)中多开一个FC层和FCN一块，信息变得有多样性，增强了预测质量。
![seg_panet](https://user-images.githubusercontent.com/42667259/89927270-5f006f80-dc06-11ea-8adf-1fe60c4536ed.png)

- MaskLab, Chen et al., 2018, google, [Masklab: Instance segmentation by refining object detection with semantic and direction features](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_MaskLab_Instance_Segmentation_CVPR_2018_paper.pdf)  
It refines object detection with semantic and direction features based on Faster R-CNN. This model produces three outputs, box detection, semantic segmentation, and direction prediction. Building on the FasterRCNN object detector, the predicted boxes provide accurate localization of object instances. Within each region of interest, MaskLab performs foreground/background segmentation by combining semantic and direction prediction. 与Mask-RCNN不同的是，这个产生的semantic segmentation，其产生的crop logits和direction pooling拼接得到segmentation
![seg_masklab](https://user-images.githubusercontent.com/42667259/89929306-5cebe000-dc09-11ea-9f7e-7b2388ba81f5.png)

## 2.6 Dilated Convolutional Models and DeepLab Family
Dilated conv (also known as atrous conv), 3 * 3 kernel with dilated rate 2 is equivalent to 5 * 5 kernel. 空洞卷积率，可以增大feature extraction时的视野.

DeepLab family: in deepLab v3, the authors begins to call previous deeplab v2. 
- DeepLab v1, Chen et al., 2014, in  ## 2.2, [Semantic image segmentation with deep convolutional nets and fully connected crfs](https://arxiv.org/pdf/1412.7062.pdf)  
- DeepLab v2, Chen et al., 2016, google, [Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7913730)  
1. use of dilated convolution to address the decreasing resolution in the network (caused by max-pooling and striding). 
2. Atrous Spatial Pyramid Pooling (ASPP) probes an incoming convolutional feature layer with filters at multiple sampling rates, thus capturing objects as well as image context at multiple scales to robustly segment objects at multiple scales. 
3. Improved localization of object boundaries is achieved by combining methods from deep CNNs and probabilistic graphical models. 
空洞卷积可以增加感受野，减小分辨率降低（max-pooling）的问题，且空洞金字塔池化能够捕捉多尺度信息，增强信息畅通性。
但为什么在实际的CNN中较少见到？查阅资料后发现是：*空洞卷积在实际中不好优化，速度会大大折扣*
![seg_deepLabv2](https://user-images.githubusercontent.com/42667259/90012174-311d3880-dca3-11ea-8d94-b3ddf033cb25.png)

- DeepLab v3, Chen et al., 2017, google, [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf)  
Combines cascaded and parallel modules of dilated convolutions. The parallel convolution modules are grouped in the ASPP. A 1 × 1 convolution and batch normalisation
are added in the ASPP. All the outputs are concatenated and processed by another 1 × 1 convolution to create the final output with logits for each pixel. 
![seg_deepLabv3](https://user-images.githubusercontent.com/42667259/90012815-65ddbf80-dca4-11ea-8268-20a61818910d.png)

- DeepLab v3+, Chen et al., 2018, google, [Encoder-decoder with atrous separable convolution for semantic image segmentation](https://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf)  
It uses an encoder-decoder architecture, including atrous separable convolution, composed of a depthwise convolution (spatial convolution for each channel of the input) and pointwise convolution (1× 1 convolution with the depthwise convolution as input). They used the DeepLabv3 framework as encoder. It obtains 89.0% mIoU score on the 2012 PASCAL VOC challenge compared with 85.7% with DeepLab v3.
相当于在v3基础上，多加了一个decoder环节，从刚开始的dilated conv处多接了一路，完善了物体边界信息。
![seg_deepLabv3+](https://user-images.githubusercontent.com/42667259/90014316-2369b200-dca7-11ea-805f-f86e3d2b073f.png)

## 2.7 Recurrent Neural Network Based Models
RNNs are useful in modeling the short/long term dependencies among pixels to (potentially) improve the estimation of the segmentation map. Using RNNs, pixels may be linked together and processed sequentially to model global contexts and improve semantic segmentation. RNN在帮助理解周围像素的语义信息对分割也有帮助。

- Byeon et al. 2015, University of Kaiserslautern, [Scene labeling with lstm recurrent neural networks](https://openaccess.thecvf.com/content_cvpr_2015/papers/Byeon_Scene_Labeling_With_2015_CVPR_paper.pdf)  
It is a pixellevel segmentation and classification of scene images using 2D long-short-term-memory (LSTM) network. In this work, classification, segmentation, and
context integration are all carried out by 2D LSTM networks, allowing texture and spatial model parameters to be learned within a single model. 周围方向上的语义信息输入到LSTM中，将分类，分割集合到2D LSTM的网络中
![seg_lstm](https://user-images.githubusercontent.com/42667259/90018427-31223600-dcad-11ea-9b16-66838d55d279.png)

- ReSeg, Visin et al., 2016, Polimi, [Reseg: A recurrent neural network-based model for semantic segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w12/papers/Visin_ReSeg_A_Recurrent_CVPR_2016_paper.pdf)  
Based on ReNet(for classification), four RNNs that sweep the image horizontally and vertically in both directions, encoding patches/activations, and providing relevant global information. The same ReNet layers are followed by up-sampling layers to recover the original image resolution in the final predictions. Gated Recurrent Units (GRUs) are used because they provide a good balance between memory usage and computational power. 骨架和ReNet一致，就是最后的分类阶段变成上采样，从而能输出分割图像。
![seg_reseg](https://user-images.githubusercontent.com/42667259/90016908-059e4c00-dcab-11ea-8d8f-445d0c0aae93.png)

- Liang et al., 2016, National University of Singapore, [Semantic object parsing with graph lstm](https://arxiv.org/pdf/1603.07063.pdf)  
Based on the graph LSTM, LSTM layers built on a super-pixel map are appended on the convolutional layers to enhance visual features with global structure context. The convolutional features pass through 1 × 1 convolutional filters to generate the initial confidence maps for all labels. The node updating sequence for the subsequent Graph LSTM layers is determined by the confidence-drive scheme based on the initial confidence maps, and then the Graph LSTM layers can sequentially update the hidden states of all superpixel nodes. 这其实也像CNN和RNN结合，CNN得到的置信图和卷积结果喂给LSTM。图的LSTM所要判别的pixel是由周围的pixel作为邻居一起决定，因此，可以给出如身体的各个部位信息。
![seg_graph_lstm](https://user-images.githubusercontent.com/42667259/90020617-5feddb80-dcb0-11ea-8bd6-8719acd36edb.png)

- Hu et al., 2016, UC Berkeley, [Segmentation from natural language expressions](https://arxiv.org/pdf/1603.06180.pdf)  
Using a combination of CNN to encode the image and LSTM to encode its natural language description, the authors proposed a semantic segmentation algorithm based on natural language expression. To produce pixel-wise segmentation for language expression, they propose an end-to-end trainable recurrent and convolutional model that jointly learns to process visual and linguistic information. In the considered model, a recurrent LSTM network is used to encode the referential expression into a vector representation, and an FCN is used to extract a spatial feature map from the image and output a spatial response map for the target object.
结合LSTM和FCN，由LSTM学习语言信息，FCN学习图像信息，最后综合给出分割结果
![seg_graph_lstm](https://user-images.githubusercontent.com/42667259/90022326-b0663880-dcb2-11ea-9be4-04d323d76020.png)

## 2.8 Attention-Based Models
In object detection, we already noticed the attention mechanism can improve the task quality significantly. Here we use it in the segmentation task.

- Chen et al., 2016, UCLA, [Attention to scale: Scale-aware semantic image segmentation](https://openaccess.thecvf.com/content_cvpr_2016/papers/Chen_Attention_to_Scale_CVPR_2016_paper.pdf)  
The authors propose an attention mechanism that learns to softly weight the multi-scale features at each pixel location. They adapt a powerful semantic segmentation model and jointly train it with multi-scale images and the attention model. The attention mechanism outperforms average and max pooling, and it enables the model to assess the importance of features at different positions and scales. 
他们将FCN模型和注意力模型一块训练，在大尺度图片上调高小物体的weights，小尺度图片上大物体得到识别，最终得到了超出pooling的有效结果。
![seg_attention](https://user-images.githubusercontent.com/42667259/90023781-6a11d900-dcb4-11ea-835d-5640c4f75dc4.png)

- RAN (Reverse Attention Network), Huang et al., 2017, USC, [Semantic segmentation with reverse attention](https://arxiv.org/pdf/1707.06426.pdf)  
RAN trains the model to capture the opposite concept (i.e., features that are not associated with a target class) as well. The RAN is a three-branch network that performs the direct, and reverse-attention learning processes simultaneously. 
![seg_ran](https://user-images.githubusercontent.com/42667259/90027595-ff16d100-dcb8-11ea-89a3-2082856bd551.png)

- Pyramid Attention Network, Li et al., 2018, Beijing Institute of Technology, [Pyramid attention network for semantic segmentation](https://arxiv.org/pdf/1805.10180.pdf)  
Authors combine attention mechanism and spatial pyramid to extract precise dense features for pixel labeling instead of complicated dilated convolution and artificially designed decoder networks. Specifically, we introduce a Feature Pyramid Attention module to perform spatial pyramid attention structure on high-level output and combine global pooling to learn a better feature representation, and a Global Attention Upsample module on each decoder layer to provide global context as a guidance of low-level features to select category localization details. 
结合FPN和attention机制，形成FPA,进行深层的语义信息提取结合全局pooling，然后和低层的位置信息结合。

- DAnet, Fu et al., 2019, Chinese Academy of Sciences, [Dual attention network for scene segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.pdf)  
Dual attention network can capture rich contextual dependencies based on the self-attention mechanism. Specifically, they append two types of attention modules on top of a dilated FCN which models the semantic interdependencies in spatial and channel dimensions, respectively. The position attention module selectively aggregates the feature at each position by a weighted sum of the features at all positions. Meanwhile, the channel attention module selectively emphasizes interdependent channel maps by integrating associated features among all channel maps. We sum the outputs of the two attention modules to further improve feature representation.
引入两个attention模块并行，一个是spatial(不同位置特征的加权和)，一个是channel（就像SENet对不同channel激活然后归一化），随后将两者结合输出分割图像
![seg_danet](https://user-images.githubusercontent.com/42667259/90029683-6f265680-dcbb-11ea-9cb6-ed26b7e3d22a.png)

## 2.9 Generative Models and Adversarial Training


