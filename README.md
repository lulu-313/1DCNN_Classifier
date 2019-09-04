# 1DCNN_Classifier
采用试井压力导数曲线作为训练数据进行油藏模型分类，通过keras构造分类模型   

模型:<br>
---
一维卷积（512 fliter——length=5）——RELU激活层——平均池化层<br>
一维卷积（128 fliter——length=5）——RELU激活层——最大池化层<br>
两个全链接层（2048/1024）<br>
softmax函数进行多分类<br>   
     
实验数据：
---
三种油藏类型的350口井，共44400条数据；分别为均质类型 200个；径向复合类型 100个；双孔拟稳定 50个   


正确率与损失曲线：<br>
![image](https://github.com/lulu-313/1DCNN_Classifier/blob/master/image/acc.png) <br>
![image](https://github.com/lulu-313/1DCNN_Classifier/blob/master/image/loss.png)<br>

交叉检验正确率与损失曲线：<br>
https://github.com/lulu-313/1DCNN_Classifier/blob/master/image/val_acc.png
https://github.com/lulu-313/1DCNN_Classifier/blob/master/image/val_loss.png
