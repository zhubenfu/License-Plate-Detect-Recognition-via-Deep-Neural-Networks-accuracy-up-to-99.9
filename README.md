# License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9
works in real-time with detection and recognition accuracy up to 99.8% for Chinese license plates: 100 ms/plate！
  
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

需用用到的第三方库下载[3rdparty 20180726 百度云](https://pan.baidu.com/s/1kZDZ98-EZr90hQt_NU-jjA  "悬停显示")

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

（2）**武汉周边的朋友可以加我，周末一起约起，多交流相互学习！**
![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/result_plateCard/%E5%BE%AE%E4%BF%A1%E5%8F%B7%EF%BC%8C%E6%AC%A2%E8%BF%8E%E5%8A%A0%E6%88%91%E4%BA%A4%E6%B5%81.jpg) 
