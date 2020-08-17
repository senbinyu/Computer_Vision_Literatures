Here I will give out the loss based on YOLO series, since it includes most of the cases.

Loss in YOLO series always includes three parts: 回归框loss, 置信度loss, 分类loss.
1. 从最重要的部分开始: 回归框loss. 
- 从 v1 to v3, 回归框loss更像是MSE，v1是(x-x')^2 + (y-y')^2，而对w和h分别取平方根做差，再求平方，用以消除一些物体大小不均带来的不利。
- v2和v3则利用(2 - w * h)[(x-x')^2 + (y-y')^2 + (w-w')^2 + (h-h')^2], 将框大小的影响放在前面作为系数，连x和y部分也一块考虑了进去。
- v4作者认为v1-v3建立的类MSE损失是不合理的。因为MSE的四者是需要解耦独⽴的，但实际上这四者并不独⽴，应该需要⼀个损失函数能捕捉到它们之间的相互关系。因此引入了IOU系列。经过其验证，GIOU，DIOU, CIOU，最终作者发现CIOU效果最好。


objectness score and 
