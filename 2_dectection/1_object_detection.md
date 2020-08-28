# Table of Contents
- [1. Review papers](#1-review-papers)
- [2. Detection paradigms](#2-detection-paradigms)
  * [2.1 Two-stage detectors](#21-two-stage-detectors)
  * [2.2 One-stage detectors](#22-one-stage-detectors)
- [3. Feature representing (feature extraction and fusion)](#3-feature-representing--feature-extraction-and-fusion-)
  * [3.1 multi-scale feature learning](#31-multi-scale-feature-learning)
  * [3.2 Region feature encoding](#32-region-feature-encoding)
  * [3.3 Contextual reasoning](#33-contextual-reasoning)
  * [3.4 Deformable feature learning](#34-deformable-feature-learning)
- [4 Applications](#4-applications)
  * [4.1 Face detection](#41-face-detection)
  * [4.2 Pedestrain detection](#42-pedestrain-detection)
  * [4.3 Text detection](#43-text-detection)
  * [4.4 Traffic light detection (autonomous driving)](#44-traffic-light-detection--autonomous-driving-)


![taxonomy](https://user-images.githubusercontent.com/42667259/89758255-a6440e80-dae7-11ea-8ab1-b5cb6b679b17.png)
Image classification aims to recognize semantic categories of objects in a given image. Object detection not only recognizes object categories, but also predicts the location of each object by a bounding box.

Here are some collections of review papers and some classic and state-of-art reseach papers focused on object dectection. The following figure shows the basic time line of classic research outputs.
<!---![timeLine](https://user-images.githubusercontent.com/42667259/89758241-9e846a00-dae7-11ea-9dfe-3487c1ebf90e.png)--->
![roadmap](https://user-images.githubusercontent.com/42667259/89792962-89c6c700-db25-11ea-88ff-b8a01299696a.png)


# 1. Review papers

- Zou et al., 2019, [Object Detection in 20 Years: A Survey](https://arxiv.org/abs/1905.05055) University of Michigan, *Recommand*

- Jiao et al., 2019 [A Survey of Deep Learning-Based Object Detection](https://ieeexplore.ieee.org/abstract/document/8825470/) Xidian University

- Zhao et al. 2019, [Object detection with deep learning: A review](https://ieeexplore.ieee.org/abstract/document/8627998/), Hefei University of Technology

- Dhillon et al. 2020, [Convolutional neural network: a review of models, methodologies and applications to object detection](https://link.springer.com/article/10.1007/s13748-019-00203-0), National Institute of Technology Kurukshetra

- Sultana et al., 2020, [A Review of Object Detection Models based on Convolutional Neural Network](https://arxiv.org/pdf/1905.01614.pdf), University of Gour Banga

- Wu et al., 2020, [Recent Advances in Deep Learning for Object Detection](https://www.sciencedirect.com/science/article/pii/S0925231220301430), Singapore Management University, *Recommand*

# 2. Detection paradigms
Two categories: two-stage vs one stage detectors  

[//]: # (This may be the most platform independent comment)
<!--这些是注释文本，不会显示
| Header Cell | Header Cell |
| ------------- | ------------- |
| Content Cell | Content Cell |
| Content Cell | Content Cell |-->

| Two-stage | one-stage |
| ------- | -------- |
| 1. a sparse set of proposals is generated; 2. the feature vectors of generated proposals are encoded by deep convolutional neural networks followed by making the object class predictions | consider all positions on the image as potential objects, and try to classify each region of interest as either background or a target object |
| lower inference speed | faster inference speed |
| state-of-art results on datasets | relatively poor performance |


## 2.1 Two-stage detectors
Proposal generation + making predictions for these proposals.
![r-cnn_series](https://user-images.githubusercontent.com/42667259/89776594-8540e500-db0a-11ea-88c9-64c854bbb0ab.png)

- R-CNN, Girshick et al. in 2014  
Pioneering two-stage object detector. Compared to previous state-of-art work, SegDPM with 40.4% mAP on Pascal VOC2010, R-CNN significantly improved the detection performance and obtained 53.7% mAP. 
1. Proposal selection, use *selective search* method to generate 1k-2k proposals, region of interest (ROI).一张图像生成1K~2K个候选区域 （采用Selective Search 方法）
2. feature extraction, use CNN. 对每个候选区域，使用深度卷积网络提取特征 （CNN）
3. region classification, use the features obtain above to determine it is background or some objects. 特征送入每一类的SVM 分类器，判别是否属于该类
4. parallel to 3, bounding box regressors are learned using the extracted features as input in order to make the original proposals tightly bound the objects. 在和3平行的另一条支路上，使用回归器精细修正候选框位置  
The weights from R-CNN pretrained from ImageNet, and R-CNN rejects huge number of easy negatives before training, which helps improve learning speed and reduce false positives. However, the Selective Search relied on low-level visual cues and thus struggled to generate high quality proposals in complex contexts; the features of each proposal were extracted by deep convolutional networks separately (i.e., computation was not shared), which led to heavily duplicated computations. Slective search根据低阶视觉信息来获取region of interest，效率和质量都不高，且二阶段的CNN提取特征时需要逐个计算ROI，无法复用计算，导致计算效率低。

Refer to paper [Rich feature hierarchies for accurate object detection and semantic segmentation](https://openaccess.thecvf.com/content_cvpr_2014/html/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.html)

- SPP-Net, He et al., 2015  
Use the idea of spatial pyramid pooling, to concate the features from different levels. Instead of cropping proposal regions and feeding into CNN model separately, SPP-net computes the feature map from the whole image using a deep convolutional network and extracts fixed-length feature vectors on the feature map by a Spatial Pyramid Pooling (SPP) layer. SPP layer did not back-propagate gradients to convolutional kernels and thus all the parameters before the SPP layer were frozen, which limits its learning capability.
SPPNet理论上可以改进任何CNN网络，通过空间金字塔池化，使得CNN的特征不再是单一尺度的。但是SPPNet更适用于处理目标检测问题，首先是网络可以介绍任意大小的输入，也就是说能够很方便地多尺寸训练。其次是空间金字塔池化能够对于任意大小的输入产生固定的输出，这样使得一幅图片的多个region proposal提取一次特征成为可能。虽然能比RCNN取得更好的效果，但是SPP层无法将梯度反向传播给卷积核，参数无法更新，由此也限制了其学习能力。

Refer to paper [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://ieeexplore.ieee.org/document/7005506)  

- Fast R-CNN, Girshick, 2015  
Use ROI pooling layer to extract region features from the whole image. ROI likes a special case of SPP, which only takes one scale N * N and can back-propagates to convolution kernels. After feature extraction, feature vectors were fed into a sequence of fully connected layers followed with classfication and regression branches. In Fast RCNN, the feature extraction, region classification and bounding box regression steps can all be optimized end-to-end, without extra cache space to store features (unlike SPP Net). Fast R-CNN achieved a much better detection accuracy than R-CNN and SPP-net, and had a better training and inference speed. Fast-RCNN 从提取特征到后面fc再到分类和回归是end-to-end的，方便梯度反向传播，这样无需额外的空间去储存特征，因此训练和推断都明显加快，同时还能比之前的模型有更好的精度。

Refer to paper [Fast R-CNN](https://openaccess.thecvf.com/content_iccv_2015/html/Girshick_Fast_R-CNN_ICCV_2015_paper.html)

- Faster-RCNN, Ren et al. 2017  
To avoid the low-lecel visual cues in selective search, Faster-RCNN adopted a novel proposal generator: Region Proposal Network (RPN) to generate proposals. The network slid over the feature map using an n × n sliding window, and generated a feature vector for each position. The feature vector was then fed into two sibling output branches, classification and regression of bbox. These results were then fed into the final layer for the actual object classification and bounding box localization. Faster R-CNN computed feature map of the input image and extracted region features on the feature map, which shared feature extraction computation across different regions. However, the computation was not shared in the region classification step, where each feature vector still needed to go through a sequence of FC
layers separately. Faster R-CNN在计算feature map时，可以在不同region之间共享计算，但在region分类时，仍然无法贡献计算，需要多余的FC层来解决，而且这多余的FC层计算量又很大，使得速度暂时无法提高。

Refer to paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks)

- R-FCN, Dai et al., 2016  
To make up the drawback that in Faster R-CNN, the computations can not be shared in the region classification step, Region-based Fully Convolutional Networks were proposed. R-FCN generated a Position Sensitive Score Map which encoded relative position information of different classes, and used a Position Sensitive ROI Pooling layer (PSROI Pooling) to extract spatial-aware region features by encoding each relative position of the target regions. 在pooling前做卷积，由于ROI pooling会丢失位置信息，故在pooling前加入位置信息，即指定不同score map是负责检测目标的不同位置。pooling后把不同位置得到的score map进行组合就能复现原来的位置信息。

Refer to paper [R-fcn: Object detection via region-based fully convolutional networks](http://papers.nips.cc/paper/6465-r-fcn-object-detection-via-region-based-fully-convolutional-networks)

![R-FCN](https://user-images.githubusercontent.com/42667259/91593796-14e6f000-e961-11ea-9239-135880b8f2d2.png)

- Mask-RCNN, He et al., 2017  
Mask-RCNN is mainly proposed to solve the segmentation tasks, accompanied with the object detection ability. Details of Mask-RCNN and its followers Cascade R-CNN, Mask Scoring R-CNN, are described in ../3_segmentaion.

Refer to paper [Mask R-CNN](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)

## 2.2 One-stage detectors
One-stage detectors typically consider all positions on the image as potential objects, and try to classify each region of interest as either background or a target object.

- OverFeat, Sermanet et al., 2013
1. Object detection can be viewed as a ”multi-region classification” problem. The last FC layer was replaced with 1 * 1 conv to allow arbitrary input.
2. The classification network output a grid of predictions on each region of the input to indicate the presence of an object. After identifying the objects, bounding box regressors were learned to refine the predicted regions based on the same DCNN features of classifierdetect multi-scale objects, the input image was resized into multiple scales which were fed into the network. Finally, the predictions across all the scales were merged together.
end-to-end学习，只需将最后的FC层改为1 * 1 conv层，即可输出想要的，预测速度非常快，将FCN、offset pooling结合了起来，提高分类任务的精度，同时也让读者看到了CNN特征提取的强大用处。

Refer to paper [OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/abs/1312.6229)

The following figures shows several classic one-stage models used for object detection
![one-stage](https://user-images.githubusercontent.com/42667259/89786010-8cbcba00-db1b-11ea-9c8f-be48a067d05a.png)
- YOLO series, Redmon et al., 2016
- v1, image divided into 7 * 7 grids, each grid has several predicting boxes (only 2 in v1). For each cell, a prediction was made which comprised the following information: whether that location had an object, the bounding box coordinates and size (width and height), and the class of the object. Use NMS to remove the overlapping bbox on the same object. YOLO is fast, but the recall is low.
V1中让每个cell预测2个bbox，经过训练后，会发现两个框出现了分工，一个倾向于预测细长的，一个预测宽的，但在最终统计loss时，只取里面IOU大的那个。当然，框越大越好，此处取2个是效率和精度的折中。损失函数是亮点，分类误差 + 置信度误差 + 预测框误差。值得注意的是，w,h取根号，是为了尽量消除大小框大小尺度给loss带来的影响。同时，这种方法因为分为7 * 7个cell，所以最多只能预测49个目标。其中的置信度是p * IOU(gd, pd), p是1或者0，gd是ground truth，pd是predict box

- v2, + a series of strategies, shown in the following figure. Anchor boexs are added in v2, multi-scale detection is also added. Without anchor boxes our intermediate model gets 69.5 mAP with a recall of 81%. With anchor boxes our model gets 69.2 mAP with a recall of 88%. After adding all the strategies, the accuracy was improved significantly. Note, here is a new network, darknet-19. 
YOLOv2 借鉴了很多其它目标检测方法的一些技巧，如 Faster R-CNN 的 anchor boxes, SSD 中的多尺度检测。YOLOv2 可以预测 13x13x5=845 个边界框，模型的召回率由原来的 81% 提升到 88%，mAP 由原来的 69.5% 降低到 69.2%. 召回率提升了 7%，准确率下降了 0.3%。除此之外，YOLOv2 在网络设计上做了很多 tricks, 使它能在保证速度的同时提高检测准确率，Multi-Scale Training 更使得同一个模型适应不同大小的输入，从而可以在速度和精度上进行自由权衡。

v2除了引入anchor外，还进行了如下改进：改变ground truth的编码方式，引入了feature map的融合方式，还引入了联合训练。Wordtree的思想在工业界有很大的启发，因为标注数据少的缘故。
![yolov2](https://user-images.githubusercontent.com/42667259/89789481-effd1b00-db20-11ea-8e56-b2b85ae11831.png)

- v3, + darknet-53. use k-means to cluster 9 anchors. Much faster.   
一种是普通的v3，结合了FPN的多尺度融合。

另一种是SPP的v3, 在预测最大物体的支路上也添加了SPP的结构（用不同大小的kernel进行max pooling，再拼接起来。为了保证大小相同，会用不同的padding），同时整个结构参考了FPN的多尺度融合方式。总共9个anchor，每个尺度下的对应3个anchor，每个尺度下输出的tensor的另一维度为255=3 * (80 + 1 + 4). 损失函数分类和置信度部分变成交叉熵，bbox部分略有改动，和2基本一样。置信度标签是直接有对象就为1，无对象为0.
![yolov3](https://user-images.githubusercontent.com/42667259/89791465-a8c45980-db23-11ea-920a-5f1316958d8c.png)
![yolov3_2](https://user-images.githubusercontent.com/42667259/90336370-b8d4b100-dfdb-11ea-82f0-52fd363a5ca0.png)
![yolov3_spp](https://user-images.githubusercontent.com/42667259/90336391-e883b900-dfdb-11ea-906f-db40ca594f88.png)

- v4 + augmentation strategies, compound many tricks together and obtain efficient and accurate yolov4. Bag of freebies + Bag of specials + (CSPDarkNet53 + SPP + PANet(path-aggregation neck) + YOLOv3-head). *Recommand*

YoloV4 将最近⼏年 CV 界⼤量的trick 集中在⼀套模型中。这篇论⽂不仅仅可以看作是⼀个模型的学习，更能看成是⼀个不错的⽂献总署。更有意思的是作者提出了 backbone,neck,head 的⽬标检测通⽤框架套路。YoloV4 将最近⼏年 CV 界⼤量的trick 集中在⼀套模型中。这篇论⽂不仅仅可以看作是⼀个模型的学习，更能看成是⼀个不错的⽂献总署。更有意思的是作者提出了 backbone,neck,head 的⽬标检测通⽤框架套路。
加了Mish激活函数，19年底出现，效果很不错。dropblock，对feature map的局部丢弃，其实是cutout的推广。
特征融合采用PANet的变种，用的是concatenate.
还有loss函数，作者发现x,y,w,h之间有相互耦合关系，直接用之前的MSE有一定问题，因此提出了用IOU类的方法，先是DIOU，考虑了两边框中心点距离，两边框重合的面积。但是没有考虑到长宽比，因此推广到CIOU. 测试阶段时，DIOU_NMS 代替 NMS，不需要用CIOU，因为测试阶段时，无需考虑长宽比一致。
![yolov4_2](https://user-images.githubusercontent.com/42667259/90336816-a4de7e80-dfde-11ea-9854-5e7dc62eb5a3.png)

Refer to paper v1 [You Only Look Once: Unified, Real-Time Object Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)
v2 [YOLO9000: Better, faster, stronger](https://openaccess.thecvf.com/content_cvpr_2017/html/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.html)
v3 [YOLOv3: An Incremental Improvement](https://sci-hub.st/https://arxiv.org/abs/1804.02767)
v4 Alexey Bochkovskiy et al., [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://sci-hub.st/https://arxiv.org/abs/2004.10934)

最新出来了个v5，只有代码，没有文字，根据代码给出的v4和v5的对比
https://zhuanlan.zhihu.com/p/161083602
v5可以自动check_anchors,若发现目前的数据集和之前收集的anchors差别大，则重新修正。
1. network architecture是一致的，都用了CSPDarknet作为骨架，然后用BiFPN作为neck，而head使用的也是和yolo v3相同的结构；
2. activation函数，v4用的是Mish函数，v5用的是LeakyReLU和sigmoid，但Mish计算更昂贵。
3. 优化函数，optimization function，yolo v5用了Adam和SGD，作者还给出了建议，如果需要训练较小的自定义数据集，Adam是更合适的选择，尽管Adam的学习率通常比SGD低。但是如果训练大型数据集，对于YOLOV5来说SGD效果比Adam好。
4. loss function，YOLO 系列的损失计算是基于 objectness score, class probability score,和 bounding box regression score。YOLO V5使用 GIOU Loss作为bounding box的损失。YOLO V5使用二进制交叉熵和 Logits 损失函数计算类概率和目标得分的损失。同时我们也可以使用fl _ gamma参数来激活Focal loss计算损失函数。
YOLO V4使用 CIOU Loss作为bounding box的损失，与其他提到的方法相比，CIOU带来了更快的收敛和更好的性能。
5. 推理时间，v5比v4快不少。其一，模型大小，v5比v4尺寸小；其二，v5默认可以采用批处理的方式，因此会快一些。而单个图像（批大小为1）上，YOLOV4推断在22毫秒内，YOLOV5s推断在20毫秒内，两者差别不大。

- SSD series, SingleShot Mulibox Detector, Liu et al., 2016  
1. With anchors, to overcome yolov1's problems: can not have accurate localization, low recall.
2. extract different layers feature map for prediction, since shallow layer has detailed location info beneficial for the detection of small objects. several extra convolutional feature maps were added to the original backbone architecture in order to detect large objects and increase receptive fields. The final prediction was made by merging all detection results from different feature maps. 
3. hard negative mining was applied for training in order to avoid huge number of negative proposals dominating training gradients. 简单负样本很多，会对loss贡献很大，导致训练无法很好进行，由此要多引入难负样本。SSD采用了hard negative mining，就是对负样本进行抽样，抽样时按照置信度误差（预测背景的置信度越小，误差越大）进行降序排列，选取误差的较大的top-k作为训练的负样本，以保证正负样本比例接近1:3。

Refer to paper [Ssd: Single shot multibox detector](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2)

- RetinaNet, Lin et al., 2017
1. To overcome the class imbalance problem in one-stage detection, RetinaNet with focal loss FL(pt) = -(1-pt)^\gamma log(pt) is proposed. when \gamma is larger than 0, the easy negative samples (pt close to 1) loss will decreases exponentially. RetinaNet used focal loss which suppressed the gradients of easy negative samples instead of simply discarding them. Their proposed focal loss outperformed naive hard negative mining strategy by large margins. Focal loss可以将那些easy negative的loss呈指数级下降，所以对训练有影响的就是那些hard negative samples了，可以显著提高训练质量.
2. Further, they used feature pyramid networks to detect multi-scale objects at different levels of feature maps. 多尺度的FPN有助于feature融合和检测

Refer to paper [Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)

Previous are anchor-based methods, below I listed two anchor-free, keypoints-based models: CornerNet and centerNet
- CornerNet, Law et al., 2018
1. anchors in other one-stage detectors need a huge number and most of them are useless. The use of anchors also introduces hyperparameters.
2. Therefore, cornerNet is an anchor-free framework, where the goal was to predict keypoints of the bounding box. Use a pair of corners (top-left and bottom-right corners) to replace the anchor.
3. noval design of corner pooling layer, which correctly match keypoints belonging to the same objects, obtaining state-of-the-art results on public benchmarks. Use feature map to corner pooling, can not use the whole iamge. 对某种feature map进行列和行的分别max pooling，不能对整张图片的整行或整列进行max pooling，会导致漏检或误检。

Refer to paper [Cornernet: Detecting objects as paired keypoints](https://openaccess.thecvf.com/content_ECCV_2018/html/Hei_Law_CornerNet_Detecting_Objects_ECCV_2018_paper.html)

- CenterNet, Zhou et al., 2019, University of Texas  
CenterNet achieves the best speed-accuracy trade-off on the MS COCO dataset, with 28.1% AP at 142 FPS, 37.4% AP at 52 FPS, and 45.1% AP with multi-scale testing at 1.4 FPS. 
1. anchor free, predict center (x, y) and offsets (dx, dy)
2. downSample coefficient 4, no use of FPN, compared with SSD 16, which used FPN. 
3. no need of a combinatorial grouping stage after keypoint detection (cornerNet needs), which significantly slows down each algorithm. 
can not deal with some cases like overlapping closely, the centers will collapse together. 无法解决物体靠很近的情况，如果下采样后，中心点几乎重合，则无法预测。

Refer to paper [Objects as points](https://arxiv.org/pdf/1904.07850.pdf)

Here is a net that does not belong to anchor-based or keypoints-based models, AZNet, automatically focused on regions of high interest. 
- AZNet, Lu et al., 2016  
AZnet adopted a search strategy that adaptively directed computation resources to sub-regions which were likely contain objects. For each region, AZnet predicted
two values: zoom indicator and adjacency scores. Zoom indicator determined whether to further divide this region which may contain smaller objects and adjacency scores denoted its objectness. 此模型不常用

Refer to paper [Adaptive object detection using adjacency and zoom prediction](https://openaccess.thecvf.com/content_cvpr_2016/papers/Lu_Adaptive_Object_Detection_CVPR_2016_paper.pdf)


# 3. Feature representing (feature extraction and fusion)
## 3.1 multi-scale feature learning
Detecting objects across large range of scales and aspect ratios is quite challenging on a single feature map. Specifically, shallow layer features with spatial-rich information have higher resolution and smaller receptive fields and thus are more suitable for detecting small objects, while semantic-rich features in deep layers are more robust to illumination, translation and have larger receptive fields (but coarse resolutions), and are more suitable for detecting large objects. 浅层包含更多位置信息，更小的视野，适合小物体检测；深层包含更丰富的语义信息，更大的视野，更适合大物体检测。

There are four categories of multi-scale feature learning
1. Image pyramid: An intuitive idea is to resize input images into a number of different scales (Image Pyramid) and to train multiple detectors, each of which is responsible for a certain range of scales. Singh et. al. [1] argued that single scale-robust detector to handle all scale objects was much more difficult than learning scale-dependent detectors with image pyramids. They proposed a novel framework Scale Normalization for Image Pyramids (SNIP) which trained multiple scale-dependent detectors and each of them was responsible for a certain scale objects. 不同尺寸的图片喂入到模型中进行训练，以来适应检测时需要的不同尺度，Signh认为单个尺度的不如多个尺度的训练
2. Prediction pyramid: This is used in SSD, predictions were made from multiple layers, where each layer was responsible for a certain scale of objects. 这里的就相当于输入一张图，但在CNN过程中得到feature maps分别进行预测，最后再总结统计。
3. Integrated features: Another approach is to construct a single feature map by combining features in multiple layers。By fusing spatially rich shallow layer features and semanticrich deep layer features, the new constructed features contain rich information and thus can detect objects at different scales. Bell et al. [2] proposed Inside-Outside Network (ION) which cropped region features from different layers via ROI Pooling, and combined these multi-scale region features for the final prediction. 和2中不同的是，这是先融合再进行预测，正好反过来
4. Feature pyramid: To combine the advantage of Integrated Features and Prediction Pyramid, Feature Pyramid Network (FPN) [3] integrated different scale features with lateral connections in a top-down fashion to build a set of scale invariant feature maps, and multiple scale-dependent classifiers were learned on these feature pyramids. Specifically, the deep semantic-rich features were used to strengthen the shallow spatially-rich features. These top-down and lateral features were combined by element-wise summation or concatenation, with small convolutions reducing the dimensions. FPN在原先从浅层到深层基础上，反向添加一个从深层到浅层的feature map，由此可以用深层的语义特征来加强浅层的空间特征。

[1] Singh et al., 2018, [An analysis of scale invariance in object detection snip](https://openaccess.thecvf.com/content_cvpr_2018/html/Singh_An_Analysis_of_CVPR_2018_paper.html)  
[2] Bell et al., 2016, [Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks](https://openaccess.thecvf.com/content_cvpr_2016/papers/Bell_Inside-Outside_Net_Detecting_CVPR_2016_paper.pdf)  
[3] Lin et al., 2017, [Feature Pyramid Networks for Object Detection](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)

![feature_pyramid](https://user-images.githubusercontent.com/42667259/89808014-c1d80500-db39-11ea-8be9-cc04e87a100c.png)

## 3.2 Region feature encoding
Previous multi-scale feature learning is appliable for both two and one-stage detectors. Region feature encoding is an important step for two-stage detectors. In Fast RCNN, there is a ROI pooling layer, which is used to encode the region features. ROI Pooling divided each region into n × n cells (e.g. 7 × 7 by default) and only the neuron with the maximum signal would go ahead in the feedforward stage. ROI Pooling extracted features from the down-sampled feature map and as a result struggled to handle small objects. In R-FCN, in order to enhance spatial information of the downsampled region features, Position Sensitive ROI Pooing (PSROI Pooling) was proposed which kept relative spatial information of downsampled features. 在Fast RCNN在的ROI Pooling层就是用于region feature encoding的，R-FCN中则进一步将低层位置信息结合，产生位置敏感ROI pooling层。

## 3.3 Contextual reasoning
Learning the relationship between objects with their surrounding context can improve detector’s ability to understand the scenario. Some works [4] have even shown
that in some cases context information may even harm the detection performance. 结合上下语境有助于推断，但有工作指出在某些情况下甚至会影响推断。
1. Global context reasoning refers to learning from the context in the whole image. DeepIDNet [5] learned a categorical score for each image which is used as contextual features concatenated with the object detection results.
2. Region Context Reasoning encodes contextual information surrounding regions and learns interactions between the objects with their surrounding area. Structure Inference Net (SIN) [6] formulated object detection as a graph inference problem by considering scene contextual information and object relationships. In SIN, each object was treated as a graph node and the relationship between different objects were regarded as graph edges. 

## 3.4 Deformable feature learning
A good detector should be robust to nonrigid deformation of objects. DeepIDNet [5] developed a deformable-aware pooling layer to encode the deformation information across different object categories. 

[4] Cheng et al., 2018, [Revisiting rcnn: On awakening the classification power of faster rcnn](https://openaccess.thecvf.com/content_ECCV_2018/papers/Bowen_Cheng_Revisiting_RCNN_On_ECCV_2018_paper.pdf)  
[5] Ouyang et al., 2015, [Deepid-net: Deformable deep convolutional neural networks for object detection](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ouyang_DeepID-Net_Deformable_Deep_2015_CVPR_paper.pdf)  
[6]  Liu et al., 2018, [Structure inference net: Object detection using scene-level context and instance-level relationships](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Structure_Inference_Net_CVPR_2018_paper.pdf)

# 4 Applications
## 4.1 Face detection
Face detection is a real-world application with human beings, such as face verification, face alignment and face recognition. There are some differences between face detection and generic detection: range of scale for objects in face detection is much larger; Face objects contain strong structural information, and there is only one target category in face detection. 

Before the deep learning era, AdaBoost with Haar features for face detection and obtained excellent performance with high real time prediction speed. (OpenCV has the tutoraials. import haarcascade_frontalface_default.xml) 

Current face detection algorithms based on deep learning are mainly extended from generic detection frameworks such as Fast R-CNN and SSD. 
1. multi-scale features dealing: In order to handle extreme scale variance, multi-scale feature learning methods discussed before have been widely used in face detection. Sun et al. [7] proposed a Fast R-CNN based framework which integrated multi-scale features for prediction and converted the resulting detection bounding boxes into ellipses as the regions of human faces are more elliptical than rectangular. Zhang et al. [8] proposed one-stage S3FD which found faces on different feature maps to detect faces at a large range of scales. They made predictions on larger feature maps to capture small-scale face information. Notably, a set of anchors were carefully designed according to empirical receptive fields and thus provided a better match to the faces. 单阶段多阶段的均利用了多尺度融合的方法进行人脸检测，以减轻人脸检测中尺度跨越大的问题。
2. contextual information: Zhang et al. [9] proposed FDNet based on ResNet with larger deformable convolutional kernels to capture image context. Zhu et al. [10] proposed a Contextual Multi-Scale Region-based Convolution Neural Network (CMS-RCNN) in which multi-scale information was grouped both in region proposal and ROI detection to deal with faces at various range of scale. In addition, contextual information around faces is also considered in training detectors. 语义信息获取，通过deformable的卷积核，和使用CMS-RCNN

[7] Sun et al., 2018, [Face detection using deep learning: An improved faster RCNN approach](https://www.sciencedirect.com/science/article/pii/S0925231218303229?casa_token=D5_Yl1deSnAAAAAA:trmLN3aDhB8UyR_MYeXxw-ZABXI74QpObHs9TX0y1MOvfMvMlnp1ZBDRr00nQfMsE-i9xgPTROE)  
[8] Zhang et al., 2017, [S3fd: Single shot scale-invariant face detector](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_S3FD_Single_Shot_ICCV_2017_paper.pdf)  
[9] Zhang et al., 2018, [Face detection using improved faster rcnn](https://arxiv.org/ftp/arxiv/papers/1802/1802.02142.pdf)  
[10] Zhu et al., 2017, [Cms-rcnn: contextual multi-scale region-based cnn for unconstrained face detection](https://arxiv.org/pdf/1606.05413.pdf)

## 4.2 Pedestrain detection
There are some properties of pedestrian detection different from generic object detection: (i) Pedestrian objects are well structured objects with nearly fixed aspect ratios (about 1.5), but they also lie at a large range of scales; (ii) Pedestrian detection is a real world application, and hence the challenges such as crowding, occlusion and blurring are commonly exhibited; (iii) There are more hard negative samples (such as traffic light, Mailbox etc.) in pedestrian detection due to
complicated contexts. 行人检测有长宽比较为固定，但也存在很多难点，像遮挡，密集，模糊等，同时有很多难负例存在于其复杂语义环境中

In deep learning era, Angelova et al [11] proposed a real-time pedestrian detection framework using a cascade of deep convolutional networks. In their work, a large number of easy negatives were rejected by a tiny model and the remaining hard proposals were then classified by a large deep networks. Further, Yang et al. [12] inserted Scale Dependent Pooling (SDP) and Cascaded Rejection Classifiers (CRC) into Fast RCNN to handle pedestrians at different scales. According to the height
of the instances, SDP extracted region features from a suitable scale feature map, while CRC rejected easy negative samples in shallower layers. Wang et al. [13] proposed a novel Repulsion Loss to detect pedestrians in a crowd. They argued that detecting a pedestrian in a crowd made it very sensitive to the NMS threshold,
which led to more false positives and missing objects. The new proposed repulsion loss pushed the proposals into their target objects but also pulled them away from other objects and their target proposals. 所提出的模型都是为了解决上述提到的尺度跨越大，密集，难负例多这些问题。

To handle occlusion problems, part-based models were proposed which learn a series of part detectors and integrate the results of part detectors to locate and classify objects. Tian et al. [14] proposed DeepParts which consisted of multiple part-based detectors. During training, the important pedestrian parts were automatically selected from a part pool which was composed of parts of the human body (at different scales), and for each selected part, a detector was learned to handle occlusions. To integrate the inaccurate scores of part-based models, Ouyang and Wang [15] proposed a framework which modeled visible parts as hidden variables in training the models. In their work, the visible relationship of overlapping parts were learned by discriminative deep models, instead of being manually defined or even being assumed independent. 遮挡问题是最有挑战的，未解决这个问题，提出了基于部件(将人分为各个part)的detectors，利用部件的检测，可以部分解决遮挡问题。

[11] Angelova et al., 2015, [Real-time pedestrian detection with deep network cascades](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/43850.pdf)  
[12] Yang et al., 2016, [Exploit all the layers: Fast and accurate cnn object detector with scale dependent pooling and cascaded rejection classifiers](https://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_Exploit_All_the_CVPR_2016_paper.pdf)  
[13] Wang et al., 2018, [Face detection using improved faster rcnn](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.pdf)  
[14] Tian et al., 2015, [Deep learning strong parts for pedestrian detection](https://openaccess.thecvf.com/content_iccv_2015/papers/Tian_Deep_Learning_Strong_ICCV_2015_paper.pdf)  
[15] Ouyang and Wang, 2012, [A discriminative deep model for pedestrian detection with occlusion handling](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6248062&casa_token=GwFn0B8cgZkAAAAA:5f7K7VoCPLqpj8hQEQqw-3PLsESactMBi_nWz3zkwg1iQ-WqqiO2wpKzImmCc_oUxWc3Q1-FvQ&tag=1)

## 4.3 Text detection
Text detection has some chanllenges: Different fonts and languages, Text rotation and perspective distortion, Densely arranged text localization, Broken and blurred characters. 

In recent years, researchers have paid more attention to the problem of text localization rather than recognition. Two groups of methods are proposed recently. The first group of methods frame the text detection as a special case of general object detection [16]. These methods have a unified detection framework, but it is less effective for detecting texts with orientation or with large aspect ratio. The second group of methods frame the text detection as an image segmentation problem [17,18]. The advantage of these methods is there are no special restrictions for the shape and orientation of text, but the disadvantage is that it is not easy to distinguish densely arranged text lines from each other based on the segmentation result. 一种方法将text detection当做是object detection的特殊case,但此法不太有效；而另一种方法则是将其看成图像分割问题，好处是对text形状无限制，缺点是无法应对密集分布的text。

For text rotation and perspective changes: The most common solution to this problem is to introduce additional parameters in anchor boxes and RoI pooling layer that are associated with rotation and perspective changes [19]. 

To improve densely arranged text detection: The segmentation-based approach shows more advantages in detecting densely arranged texts. To distinguish the adjacent
text lines, two groups of solutions have been proposed recently. The first one is “segment and linking”, where “segment” refers to the character heatmap, and “linking”
refers to the connection between two adjacent segments indicating that they belong to the same word or line of text [16]. The second group is to introduce an additional corner/border detection task to help separate densely arrange texts, where a group of corners or a closed boundary corresponds to an individual line of text [17].

To improve broken and blurred text detection: A recent idea to deal with broken and blurred texts is to use word level [20] recognition and sentence level recognition [21]. To deal with texts with different fonts, the most effective way is training with synthetic samples [20]. 

[16] He et al., 2017, [Single shot text detector with regional attention](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Single_Shot_Text_ICCV_2017_paper.pdf)  
[17] Liu et al., 2017, [Deep matching prior network: Toward tighter multi-oriented text detection](https://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_Deep_Matching_Prior_CVPR_2017_paper.pdf)  
[18] Wu et al., 2017, [Self-organized text detection with minimal post-processing via border learning](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Self-Organized_Text_Detection_ICCV_2017_paper.pdf)  
[19] Ma et al., 2018, [Arbitrary-oriented scene text detection via rotation proposals](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8323240)  
[20] Jaderberg et al., 2014, [Synthetic data and artificial neural networks for natural scene text recognition](https://arxiv.org/pdf/1406.2227.pdf)  
[21] Wojna et al., 2017, [Attention-based extraction of structured information from street view imagery](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8270074)


## 4.4 Traffic light detection (autonomous driving)
Traffic light detection has a difference from the ordinary object detection: the target, i.e., traffic light is usually very small. Lu et al. [22] presented an attention model based detection framework to tackle the problem of detecting small objects in large high resolution images. Their framework outperforms the baseline faster RCNN, especially when detecting small targets with area less than 322 pixels. Adversarial training have been used to improve detection of small objects under complex traffic environments [23]. Perceptual GAN generates super-resolved representations for small objects to boost detection performance by leveraging the repeatedly updated generator network and the discriminator network. 在Faster-RCNN基础上，加入attention机制用于提取特别小的交通灯（有时候只有几个像素），同样也有用对抗的形式去加强检测能力


[22] [Traffic signal detection and classification in street views using an attention model](https://link.springer.com/content/pdf/10.1007/s41095-018-0116-x.pdf)  
[23] [Perceptual generative adversarial networks for small object detection](https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Perceptual_Generative_Adversarial_CVPR_2017_paper.pdf)

