# License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9
works in real-time with detection and recognition accuracy up to 99.8% for Chinese license plates: 100 ms/plate！


  【重要通知1】：将从理论到实践，建立一个微信公众号，把近期的总结以专题形式公布出来，初步目录为：
  欢迎大家关注公众号：ai_portumo。 
  
  【重要通知2】：扫描关注公众号ai_portumo，回复“车牌数据”获取37G的实际车牌数据下载链接。请大家不要随意将下载链接共享，我共享数据的目的是为了给我的公众号引流，大家可以推荐朋友关注公众号ai_portumo，并在公众号输入框回复“车牌数据”，获取下载链接。
----
  共19个文件，每个文件2G左右，覆盖各种数字和英文字体，以及早中晚各种场景，可以用来车牌识别训练和OCR学习。
----
  ![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/%E6%89%AB%E7%A0%81_%E6%90%9C%E7%B4%A2%E8%81%94%E5%90%88%E4%BC%A0%E6%92%AD%E6%A0%B7%E5%BC%8F-%E7%99%BD%E8%89%B2%E7%89%88.png)  


**目录：**

**【第一章 线性代数】**

矩阵、迹、转置

行列式、三角矩阵、行列式的性质、余子式

克拉姆法则、齐次线性方程组

逆矩阵、矩阵的初等变换

矩阵的秩、线性方程组的解

向量

方阵的特征值和特征向量

相似矩阵、矩阵对角化、对阵矩阵对角化

二次型与矩阵正定性

SVD分解的证明、SVD分解、SVD分解的应用

多元线性回归、最小二乘法

**【第二章 微积分】**

中值定理、柯西中值定理、洛必达法则

泰勒公式

函数的凹凸性、函数的极值

不定积分、凑微分、分部积分法、定积分

牛顿莱布尼茨公式

全部积分、偏导数

方向导数与梯度

多元函数的泰勒公式、海森矩阵

多元函数的极值

矩阵的求导

**【第三章 概率论】**

条件概率、全概率公式、贝叶斯公式

独立性

离散随机变量、连续随机变量

分布函数：均匀分布、指数分布、正态分布

随机变量函数的分布

多维随机变量

边缘分布、条件分布

期望（连续型、离散型）

方差

方差的常用性质、协方差

协方差矩阵

参数估计、矩估计

极大似然估计、高斯分布、无偏性

**【第四章 机器学习】**

无约束优化、梯度下降法

牛顿法、两种解释

牛顿法的收敛速度

有约束的优化、拉格朗日乘子的意义、KKT条件

不等式约束

约束问题总结：等式约束、不等式约束

优化的对偶理论、原始问题、对偶问题

**【第五章 深度学习】**

容量、过拟合、欠拟合

逻辑回归、线性回归、二项逻辑回归模型

贝叶斯估计、频率派、贝叶斯派

QA J=‖xa−y‖^2 求导

PCA原理与推导

LDA、PCA与LDA区别

SVM支持向量机

决策树

传递函数、激活函数、损失函数、KL距离、相对熵

前向传播、反向算法

深度学习中的正则化

Bagging、Dropout

深度模型中的优化

动量Momentum、AdaGrad、Adam、BatchNorm

卷积网络

CNN前向算法、反向算法、Pooling

Inception网络：V1、V2、V3、V4

循环神经网络、前向、后向

LSTM

GRU

深度学习中的一些高级技术

神经网络量化

对抗网络


  【重要通知】：将从理论到实践，建立一个微信公众号，把近期的总结以专题形式公布出来，以上为初步目录：
  欢迎大家关注公众号：ai_portumo。


 ![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/%E6%89%AB%E7%A0%81_%E6%90%9C%E7%B4%A2%E8%81%94%E5%90%88%E4%BC%A0%E6%92%AD%E6%A0%B7%E5%BC%8F-%E6%A0%87%E5%87%86%E8%89%B2%E7%89%88.png)  

  
  进来看一定要加star！右上角！
  -----
  github 里面会有一个列表，专门收集了你所有 star 过的项目，点击 github 个人头像，可以看到 your stars 的条目，点击就可以查看你 star 过的所有项目了。

准商业项目：正在整理文档 后面全部开放出来文档和全部资料。
===========================================
本项目采用了多种方式识别车牌，每一种方式各有优缺点，现在统一更新出来！  

| 检测大牌  | 分割单个字符 | 识别车牌 |项目支持 |
| ------------- | ------------- | ------------- |------------- |
| haar+cascade  | haar+cascade  | 切割出单个字符通过cnn识别 |[Y] |
| mtcnn  | 图像处理  | lstm+ctc  |[Y] |
| 图像处理：跳变点  |    | fcn全卷积网络带单个字符定位 |[ ] |
| YOLO  |    |   |[ ] |


|    | 原图 | 目标 |点回归 |
| ------------- | ------------- | ------------- |------------- |
| YOLO  | 归一化  | X |增一层 |
| mtcnn  | X | 归一化  |增一层 |
| haar+cascade  | X   | 归一化  | |


需用用到的第三方库下载[3rdparty 20180726 百度云](https://pan.baidu.com/s/1kZDZ98-EZr90hQt_NU-jjA  "悬停显示")

注意说明：
项目采用vs2015 x64 release cuda9.2编译
--------

##一、整个大车牌检测基于haar+cascade的检测或者mtcnn的检测，
--------------------------------
[车牌识别技术详解六--基于Adaboost+haar训练的车牌检测](https://blog.csdn.net/zhubenfulovepoem/article/details/42474239  "悬停显示")
  
  大牌检测采用车牌比例为90:36的比例，训练基于haar特征的adaboost检测。  

（1）**准备样本：**
**正样本**：样本处理和选择非常有技巧，我的标准是框住整个车牌留出边框，这样既保留了车牌原有的字符特征，字符组特征还有车牌的边框特征。其中双行车牌我只取底下面的一行。并且检测样本最好不要预处理，输入源给出什么图形就用什么图形。*具体的抠图方式可以参考我其他博客车牌识别技术详解三--字符检测的正负样本得取（利用鼠标画框抠图）。*

**负样本**：负样本选择同样非常有技巧性。尽量采集车牌使用环境下的背景图片，并且需要包含一部分车牌字符但是非正样本的取在车牌周围的负样本。



##二、mtcnn检测到车牌之后，通过回归得到四个角点，做透视变换对齐得到水平车牌，实测可以处理角度非常偏的车牌，
-------
![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20180720093811.png)  

##三、单个车牌字符分割是基于haar+cascade加上逻辑筛选，
--------
###1、**图像识别中最核心最难搞的模块之一：目标检测定位分割**
做识别应用最难的部分就是分割了，图像分割好了，后端做识别才更简单。

*检测前需不需要做图像预处理：建议可以根据实际情况简单处理下，常用的比如cvNorm，但是仅在备份图像上做处理，原图尽量不动，原图留做识别抠图。
*训练一个分类器进行目标检测，以haar+adaboost为例，详细参考字符检测的正负样本得取（利用鼠标画框抠图）和准备样本等。 
  
![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/20171121204748663.png)
     
###2.**往往分类器仅仅只能得到以上一个初步的效果，这时候需要根据项目实际图像固有的特征规律进行一些调整。 **



[车牌识别技术详解三--字符检测的正负样本得取（利用鼠标画框抠图）](https://blog.csdn.net/zhubenfulovepoem/article/details/12344639   "悬停显示") 

单个字符的分割可以基于haar或者采用fcn！

[车牌识别技术详解四--二值化找轮廓做分割得样本（车牌分割，验证码分割）](https://blog.csdn.net/zhubenfulovepoem/article/details/12345539   "悬停显示")



##四、识别支持blstm+ctc全图识别、单个字符分割识别和FCN全卷积识别。
-------
| 算法 | 识别车牌的方法 |优缺点 |
| ------------- | ------------- |------------- |
| haar+cascade  | 切割出单个字符通过cnn识别 |由于单个字符样本较多，所以识别率在正面车牌情况下，非常高  |
|   | lstm+ctc  |全图识别，可以处理角度，污迹等等  | 
| fcn+反卷积 | fcn全卷机网络带单个字符定位 |  带定位，但是依赖数据过多 |

（1）**FCN Multilabel Caffe方法综述[FCN的车牌图像识别，end-to-end 目标定位、图像识别](https://blog.csdn.net/zhubenfulovepoem/article/details/78902747   "悬停显示")  **

![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/20171121203935599.png)

![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/20171121203946021.png)

(2)**单个字符识别：**
   识别样本丰富性处理：很多应用你取不到特别多的样本，覆盖不了所有的情况，并且样本之间的均衡性也很难平衡。常见的情况肯定是出现最多的，样本是最多的，还有可能某类样本数是最多的，另一类别下的样本数也是比较少的。 
   实际项目其实时间花的最多的就是在那20%-30%的情况下做边界处理占了项目90%的时间。 
   我们需要尽可能的保证样本的均衡性，采样时候各种情况尽可能包含，每类别下的样本数量尽量均衡。
   某类样本数量不够可以采集图像处理增加样本量，常用的有分割的时候上下左右平移，图像拉伸，滤波等。 



##五、部分结果展示
----------
![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/result_plateCard/QQ%E5%9B%BE%E7%89%8720180529195903.png)

![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/result_plateCard/QQ%E5%9B%BE%E7%89%8720180529195834.png)  

![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/result_plateCard/QQ%E5%9B%BE%E7%89%8720180529195858.png)
  

![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/result_plateCard/QQ%E5%9B%BE%E7%89%8720180529195908.png)  


![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/result_plateCard/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20180530112203.png) 

##六、数据资源共享交流：
主要是促进交流，不为盈利！将准商业的产品开源，欢迎交流，各抒己见，逐步完善成一个通用的目标检测分割识别的OCR开源项目
---
（1）交流加群：加QQ群 图像处理分析机器视觉 109128646
========
感谢群友：E-图像处理_仪山湖  贡献的操作说明！和完整的第三方库，下载地址在群共享里面！  

1: 安装CUDA 9.2， cuda9.2版已经带有显卡驱动，默认安装，保持cuda和显卡驱动一致，否则会报cuda runtime is inefficient 35号错误。  

2：确认各项目的include、lib和第三方库路径的配置，本人在项目中除CUDA使用绝对路径外，其余的include、lib路径均使用相对路径
```
    caffe项目：
	   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include
..\..\opensource\boost_1_57_0
..\..\opensource\opencv\include
..\..\3rdparty\include
..\..\include
..\..\3rdparty\include\lmdb
..\..\3rdparty\include\hdf5
..\..\src\
..\..\3rdparty\include\glog
..\..\3rdparty\include\cudnn
..\..\src\caffe\proto
..\..\3rdparty\include\openblas
   
   ..\..\opensource\boost_1_57_0\lib
..\..\opensource\opencv\lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\lib\x64
..\..\3rdparty\lib
..\..\tools_bin

 
	libClassfication项目：
	..\..\3rdparty\include\openblas
..\..\opensource\opencv\include
..\..\opensource\boost_1_57_0
..\..\3rdparty\include
..\..\include
..\..\3rdparty\include\lmdb
..\..\3rdparty\include\hdf5
..\..\src\
..\..\3rdparty\include\glog
..\..\3rdparty\include\cudnn
..\..\src\caffe\proto
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\lib\x64
..\..\opensource\boost_1_57_0\lib
..\..\opensource\opencv\lib
..\..\3rdparty\lib
..\..\tools_bin
	
	ocr_test:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include
..\..\opensource\boost_1_57_0
..\..\opensource\opencv\include
..\libClassification
..\..\include
..\..\3rdparty\include
..\..\src\
..\..\3rdparty\include\openblas

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\lib\x64
..\..\opensource\opencv\lib
..\..\opensource\boost_1_57_0\lib
..\..\tools_bin
```	
	
3：生成时要选择Release，x64格式  

4：在项目的cuda生成host代码时，支持一些旧版本的cu伪代码，三个项目都设置成：compute_61,sm_61;compute_52,sm_52;compute_50,sm_50;compute_35,sm_35;compute_30,sm_30，否则会报no kernel image 40号错误  

5：整理的3rdparty请使用群文件的3rdparty-20180730.tar  

6：在visual studio 2015打开caffe.sln方案时（项目根目录下），如果出现无法加载工程的错误，打开项目根目录下的caffe-vsproj\caffe.sln  

7：导入工程后，在ocr_test的项目，右键，选择【生成依赖项】-> 【项目依赖项】，勾选caffe, libClassfication两个项目  

8: 项目代码中对路径没有做规范化处理，导致在运行期间报找不到模型文件的错误，为了调试方便，这里把代码中的路径全部修改成绝对路径，如ocr_test.cpp的962行，
     string modelfolder = "E:\\License-Plate-Detect-Recognition\\caffe-vsproj\\ocr_test\\plateCard_test";  其他的类似修改
   但要区分MTCNN模型和ICNNPredict模型的位置	  
   
9：启动一个cmd，切换到工程根目录下的tools_bin目录，启动ocr_test.ext     



（2）**武汉周边的朋友可以加我，周末一起约起，多交流相互学习！**
![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/result_plateCard/%E5%BE%AE%E4%BF%A1%E5%8F%B7%EF%BC%8C%E6%AC%A2%E8%BF%8E%E5%8A%A0%E6%88%91%E4%BA%A4%E6%B5%81.jpg) 
