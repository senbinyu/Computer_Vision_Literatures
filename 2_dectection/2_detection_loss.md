在这里，我将基于YOLO系列给出损失函数，因为它包括了多数情况。

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

