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
Dilated conv (also known as atrous conv), 3 * 3 kernel with dilated rate 2 is equivalent to 5 * 5 kernel. 空洞卷积率，可以增大feature extraction时的视野，
- DeepLab Chen et al., 2018, google, [Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7913730)  



