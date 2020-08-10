# Table of Contents
- [1. Review papers](#1-review-papers)
- [2. Detection paradigms](#2-detection-paradigms)
  * [2.1 Two-stage detectors](#21-two-stage-detectors)
  * [2.2 One-stage detectors](#22-one-stage-detectors)
- [3. Detection proposal generations](#3-detection-proposal-generations)
  * [3.1 Traditional computer vision methods](#31-traditional-computer-vision-methods)
  * [3.2 Anchor-based method](#32-anchor-based-method)
  * [3.3 Keypoints based method](#33-keypoints-based-method)
- [4. Feature representing (feature extraction and fusion)](#4-feature-representing--feature-extraction-and-fusion-)
  * [4.1 multi-scale feature learning](#41-multi-scale-feature-learning)
  * [4.2 Region feature encoding](#42-region-feature-encoding)
  * [4.3 Deformable feature learning](#43-deformable-feature-learning)
- [5 Applications](#6-applications)
  * [5.1 Face detection](#61-face-detection)
  * [5.2 Pedestrain detection](#62-pedestrain-detection)
  * [5.3 Text detection](#63-text-detection)
- [6. Datasets](#7-datasets)

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

Refer to paper [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition]()

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

- v2, + a series of strategies, shown in the following figure. Anchor boexs are added in v2, multi-scale detection is also added. Without anchor boxes our intermediate model gets 69.5 mAP with a recall of 81%. With anchor boxes our model gets 69.2 mAP with a recall of 88%. After adding all the strategies, the accuracy was improved significantly. Note, here is a new network, darknet-19. 
YOLOv2 借鉴了很多其它目标检测方法的一些技巧，如 Faster R-CNN 的 anchor boxes, SSD 中的多尺度检测。YOLOv2 可以预测 13x13x5=845 个边界框，模型的召回率由原来的 81% 提升到 88%，mAP 由原来的 69.5% 降低到 69.2%. 召回率提升了 7%，准确率下降了 0.3%。除此之外，YOLOv2 在网络设计上做了很多 tricks, 使它能在保证速度的同时提高检测准确率，Multi-Scale Training 更使得同一个模型适应不同大小的输入，从而可以在速度和精度上进行自由权衡。
![yolov2](https://user-images.githubusercontent.com/42667259/89789481-effd1b00-db20-11ea-8e56-b2b85ae11831.png)

- v3, + darknet-53. use k-means to cluster 9 anchors. Much faster.
![yolov3](https://user-images.githubusercontent.com/42667259/89791465-a8c45980-db23-11ea-920a-5f1316958d8c.png)

- v4 + augmentation strategies, compound many tricks together and obtain efficient and accurate yolov4. Bag of freebies + Bag of specials + (CSPDarkNet53 + SPP + PANet(path-aggregation neck) + YOLOv3-head). *Recommand*

Refer to paper v1 [You Only Look Once: Unified, Real-Time Object Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)
v2 [YOLO9000: Better, faster, stronger](https://openaccess.thecvf.com/content_cvpr_2017/html/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.html)
v3 [YOLOv3: An Incremental Improvement](https://sci-hub.st/https://arxiv.org/abs/1804.02767)
v4 Alexey Bochkovskiy et al., [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://sci-hub.st/https://arxiv.org/abs/2004.10934)

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

Refer to paper [Objects as points](https://arxiv.org/pdf/1904.07850.pdf)

Here is a net that does not belong to anchor-based or keypoints-based models, AZNet, automatically focused on regions of high interest. 
- AZNet, Lu et al., 2016  
AZnet adopted a search strategy that adaptively directed computation resources to sub-regions which were likely contain objects. For each region, AZnet predicted
two values: zoom indicator and adjacency scores. Zoom indicator determined whether to further divide this region which may contain smaller objects and adjacency scores denoted its objectness. 此模型不常用

Refer to paper [Adaptive object detection using adjacency and zoom prediction](https://openaccess.thecvf.com/content_cvpr_2016/papers/Lu_Adaptive_Object_Detection_CVPR_2016_paper.pdf)


# 3. Feature representing (feature extraction and fusion)
## 3.1 multi-scale feature learning

## 3.2 Region feature encoding

## 3.3 Deformable feature learning


# 5 Applications
## 5.1 Face detection


## 5.2 Pedestrain detection


## 5.3 Text detection


# 6. Datasets

