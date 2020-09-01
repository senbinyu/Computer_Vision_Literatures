和图像分割中将损失函数分为基于分布，基于区域以及基于边界的损失函数不一样，目标检测经常可以认为由2类最基础的损失，分类损失和回归损失而组成。  
![object detection loss](https://user-images.githubusercontent.com/42667259/91607219-d6a6fc00-e973-11ea-9f5e-0ba331b713cf.png)


##### 分类损失
- CE loss，交叉熵损失    
交叉熵损失，二分类损失（binary CE loss）是它的一种极端情况. 在机器学习部分就有介绍它。  
如下图所示,y是真实标签，a是预测标签，一般可通过sigmoid，softmax得到，x是样本，n是样本数目，和对数似然等价。    
![ce_loss](https://user-images.githubusercontent.com/42667259/91491995-2c689f00-e8b5-11ea-8294-e6c122da3476.png)  

- focal loss,   
用改变loss的方式来缓解样本的不平衡，因为改变loss只影响train部分的过程和时间，而对推断时间影响甚小，容易拓展。  
focal loss就是把CE里的p替换为pt，当预测正确的时候，pt接近1，在FL(pt)中，其系数$(1-p_t)^\gamma$越小（只要$\gamma>0$）；简而言之，就是简单的样例比重越小，难的样例比重相对变大   
![loss_focal_pt](https://user-images.githubusercontent.com/42667259/91609496-d0b31a00-e977-11ea-9f2e-be3e883acd90.png)
![loss_focal](https://user-images.githubusercontent.com/42667259/91609497-d14bb080-e977-11ea-9753-4d1edb2d9632.png)

- Rankings类型的损失  
在这有两类，DR(Distributional Ranking) Loss和AP Loss  
- DR Loss, 分布排序损失， Qian et al., 2020, [DR loss: Improving object detection by distributional ranking](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qian_DR_Loss_Improving_Object_Detection_by_Distributional_Ranking_CVPR_2020_paper.pdf)      
DR loss的研究背景和focal loss一样，one-stage方法中样本不平衡。它进行分布的转换以及用ranking作为loss。将分类问题转换为排序问题，从而避免了正负样本不平衡的问题。同时针对排序，提出了排序的损失函数DR loss。具体流程可参考：https://zhuanlan.zhihu.com/p/75896297  
![loss_DR](https://user-images.githubusercontent.com/42667259/91613528-fba16c00-e97f-11ea-964b-ec0896f25d05.png)  

- AP Loss, Chen et al., 2019, [Towards Accurate One-Stage Object Detection with AP-Loss](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Towards_Accurate_One-Stage_Object_Detection_With_AP-Loss_CVPR_2019_paper.pdf)  
AP loss也是解决one-stage方法中样本不平衡问题,同时也和DR loss类似，是一种排序loss。将单级检测器中的分类任务替换为排序任务，并采用平均精度损失(AP-loss)来处理排序问题。由于AP-loss的不可微性和非凸性，使得APloss不能直接优化。因此，本文开发了一种新的优化算法，它将感知器学习中的错误驱动更新方案和深度网络中的反向传播机制无缝地结合在一起。具体可参见：https://blog.csdn.net/jiaoyangwm/article/details/91479594  
![loss_AP](https://user-images.githubusercontent.com/42667259/91614056-12948e00-e981-11ea-86a3-12c83826e493.png)


##### 回归损失
回归损失在这里更多的是对应与bounding box的回归。  
- MSE， RMSE，同样在机器学习中也会用来做回归损失。  
常用在回归任务中，MSE的特点是光滑连续，可导，方便用于梯度下降。因为MSE是模型预测值 f(x) 与样本真实值 y 之间距离平方的平均值，故离得越远，误差越大，即受离群点的影响较大  
![MSE](https://user-images.githubusercontent.com/42667259/91484410-7481c480-e8a9-11ea-851d-a3e69408d395.png)  
![RMSE](https://user-images.githubusercontent.com/42667259/91490109-34730f80-e8b2-11ea-9a97-726b2a25208f.png)

- Huber loss,  
上述MSE对异常值非常敏感，而huber loss则是一种可以对异常值不太敏感的方法，在$\delta$范围外，其采用L1的方式，同时前面固定系数用以控制异常值影响；而在$\delta$范围内，则用L2的方式，同时前面也有系数。  
![loss_huber](https://user-images.githubusercontent.com/42667259/91897431-92c93500-ec9a-11ea-847a-6591fb86e5b2.png)

- Smooth L1 loss，  
特殊的，smoothL1Loss是huber loss中的delta=1时的情况。这个损失函数用在了faster RCNN中，用于定位框的回归损失。    
![smoothL1Loss](https://user-images.githubusercontent.com/42667259/91488847-36d46a00-e8b0-11ea-8197-dfbf551309d5.png)

- balanced L1 Loss,
https://zhuanlan.zhihu.com/p/101303119   
用在了Libra RCNN中，基于smoothL1Loss的改进。作者发现平均每个easy sample对梯度的贡献为hard sample的30%，相当于作者在找一个平衡的点，能让easy和hard的sample所占的梯度贡献差不多，因此引入了这个balancedL1Loss,其在接近于0的时候飞速下降，而在接近于1的时候缓慢上升，而不至于向smoothL1Loss那样只有中间regression error为1的时候有个突变，由此让他变得更加平衡，如下图所示。  
文章：Pang et al., 2019, [Libra R-CNN: Towards Balanced Learning for Object Detection](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pang_Libra_R-CNN_Towards_Balanced_Learning_for_Object_Detection_CVPR_2019_paper.pdf)  
![loss_balancedL1_2](https://user-images.githubusercontent.com/42667259/91611535-ca26a180-e97b-11ea-81ee-9c09cafce4e4.png)  
![loss_balancedL1_3](https://user-images.githubusercontent.com/42667259/91611538-ca26a180-e97b-11ea-9672-ac3b5908fa24.png)  
![loss_balancedL1](https://user-images.githubusercontent.com/42667259/91611532-c98e0b00-e97b-11ea-8f2a-b9c8375c1ed7.png)

- KL Loss, He et al., 2019, [Bounding Box Regression with Uncertainty for Accurate Object Detection](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bounding_Box_Regression_With_Uncertainty_for_Accurate_Object_Detection_CVPR_2019_paper.pdf)  
这篇文章是为了解决边界不确定的box的regression问题(不被模糊样例造成大的loss干扰). 文章预测坐标（x1,y1,x2,y2）的偏移值，对于每个偏移值，假设预测值服从高斯分布，标准值为狄拉克函数（即偏移一直为0），计算这两个分布的距离（这里用KL散度表示距离）作为损失函数。参考smooth L1 loss，也分为|xg-xe|<=1和>1的两段，如下所示：   
![loss_KL](https://user-images.githubusercontent.com/42667259/91612859-9731dd00-e97e-11ea-8bea-f1e74d3323fb.png)

- region-based loss，基于区域的损失函数，IOU类    
以上是针对样本分布的回归损失，后来发现基于区域的损失在回归框的任务中，起到了很好的效果，因此用基于框的回归损失函数来进行回归预测。具体可以看以下提供的实例，详细介绍了IOU的系列发展。


##### 随后，我将基于YOLO系列给出的损失函数作为实例，因为它包括了多数情况。

YOLO系列的损失包括三个部分: 回归框loss, 置信度loss, 分类loss.
1. 从最重要的部分开始: 回归框loss. 
- 从 v1 to v3, 回归框loss更像是MSE，v1是(x-x')^2 + (y-y')^2，而对w和h分别取平方根做差，再求平方，用以消除一些物体大小不均带来的不利。
- v2和v3则利用(2 - w * h)[(x-x')^2 + (y-y')^2 + (w-w')^2 + (h-h')^2], 将框大小的影响放在前面作为系数，连x和y部分也一块考虑了进去。
- v4作者认为v1-v3建立的类MSE损失是不合理的。因为MSE的四者是需要解耦独⽴的，但实际上这四者并不独⽴，应该需要⼀个损失函数能捕捉到它们之间的相互关系。因此引入了IOU系列。经过其验证，GIOU，DIOU, CIOU，最终作者发现CIOU效果最好。注意，使用的时候是他们的loss，应该是1-IOUs，因为IOU越大表示重合越好，而loss是越小越好，因此前面加1-，令其和平常使用规则一致。
- v5作者采用了GIOU，具体还需要等他论文出现。

下面介绍一下IOU系列：
这里有篇博客文章参考：https://zhuanlan.zhihu.com/p/94799295
- IOU, A与B交集 / A与B并集，在这一般是ground truth和predict box之间的相交面积/他们的并面积
![loss_iou](https://user-images.githubusercontent.com/42667259/90417901-ab3f2a00-e0b4-11ea-9606-aa61bda33ba2.png)

- GIOU, Rezatofighi et al., 2019, Stanford University, [Generalized Intersection Over Union: A Metric and a Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)  
GIOU,针对IOU只是一个比值，IoU无法区分两个对象之间不同的对齐方式，因此引入了GIOU。下图中的A_c是两个框的最小闭包区域面积(通俗理解：同时包含了预测框和真实框的最小框的面积)，减去两个框的并集，即通过计算闭包区域中不属于两个框的区域占闭包区域的比重，最后用IoU减去这个比重即可得到GIoU。  
GIoU是IoU的下界，在两个框无限重合的情况下，IoU=GIoU=1；
IoU取值[0,1]，但GIoU有对称区间，取值范围[-1,1]。在两者重合的时候取最大值1，在两者无交集且无限远的时候取最小值-1，因此GIoU是一个非常好的距离度量指标；
与IoU只关注重叠区域不同，GIoU不仅关注重叠区域，还关注其他的非重合区域，能更好的反映两者的重合度。
![loss_giou](https://user-images.githubusercontent.com/42667259/90414506-28b46b80-e0b0-11ea-9857-1347deb18e3f.png)
![loss_giou_2](https://user-images.githubusercontent.com/42667259/90415426-5bab2f00-e0b1-11ea-9b22-e6ca14bb7aab.png)

- DIOU，Zheng et al., 2019, Tianjin University, [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/pdf/1911.08287.pdf)  
DIoU要比GIou更加符合目标框回归的机制，将目标与anchor之间的距离，重叠率以及尺度都考虑进去，使得目标框回归变得更加稳定，不会像IoU和GIoU一样出现训练过程中发散等问题. 如下图所示，b和b^{gt}是预测的中心和ground truth的中心坐标，\rho是指这两点之间的欧氏距离，c是两个框的闭包区域面积的对角线的距离。  
DIoU loss可以直接最小化两个目标框的距离，因此比GIoU loss收敛快得多。
对于包含两个框在水平方向和垂直方向上这种情况，DIoU损失可以使回归非常快，而GIoU损失几乎退化为IoU损失。
DIoU还可以替换普通的IoU评价策略，应用于NMS中，使得NMS得到的结果更加合理和有效。  
![loss_diou_1](https://user-images.githubusercontent.com/42667259/90418168-10931b00-e0b5-11ea-8a21-1ff7f84cffd3.png)
![loss_diou](https://user-images.githubusercontent.com/42667259/90417766-7c28b880-e0b4-11ea-8d7e-7934f016eea2.png)

- CIOU, Zheng et al., 2019, Tianjin University, [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/pdf/1911.08287.pdf)  
虽然DIOU考虑了两中心的距离，但是没有考虑到⻓宽⽐。⼀个好的预测框，应该和 ground truth 的⻓宽⽐尽量保持⼀致。因此有了CIOU，在DIOU基础上加入了惩罚项。如下图是其CIOU loss，前面有了1-。而\nu是衡量长宽比的相似性。  
![loss_ciou](https://user-images.githubusercontent.com/42667259/90419415-cf9c0600-e0b6-11ea-9a82-1b8b228a684d.png)
![loss_ciou_2](https://user-images.githubusercontent.com/42667259/90419536-040fc200-e0b7-11ea-916a-40c2c51f41b2.png)
![loss_ciou_3](https://user-images.githubusercontent.com/42667259/90419540-04a85880-e0b7-11ea-8ba2-23fb92884fee.png)

2. 置信度损失和分类Loss.
这里先给出v1-v3的损失函数，可以看出，v1-v2中置信度误差和分类误差均使用的是MSE；
从v2到v3, 不同的地⽅在于，对于类别和置信度的损失使⽤交叉熵。
![loss_yolov1](https://user-images.githubusercontent.com/42667259/90420638-83ea5c00-e0b8-11ea-8fb2-73239c4bdba3.png)
![loss_yolov2](https://user-images.githubusercontent.com/42667259/90420640-851b8900-e0b8-11ea-823a-4a54374031ab.png)
![loss_yolov3](https://user-images.githubusercontent.com/42667259/90420641-851b8900-e0b8-11ea-96b1-7db01ef28c2c.png)

