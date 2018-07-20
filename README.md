# License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9
works in real-time with detection and recognition accuracy up to 99.8% for Chinese license plates: 100 ms/plate！
  
  
正在整理文档 后面全部开放出来文档和全部资料。
===========================================

整个大车牌检测基于haar+cascade的检测或者mtcnn的检测，
--------------------------------
[车牌识别技术详解六--基于Adaboost+haar训练的车牌检测](https://blog.csdn.net/zhubenfulovepoem/article/details/42474239  "悬停显示")

mtcnn检测到车牌之后，通过回归得到四个角点，做透视变换对齐得到水平车牌，实测可以处理角度非常偏的车牌，
-------
![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20180720093811.png)
单个车牌字符分割是基于haar+cascade加上逻辑筛选，
--------

https://blog.csdn.net/zhubenfulovepoem/article/details/12344639 车牌识别技术详解三--字符检测的正负样本得取（利用鼠标画框抠图）
https://blog.csdn.net/zhubenfulovepoem/article/details/12345539 车牌识别技术详解四--二值化找轮廓做分割得样本（车牌分割，验证码分割）


识别支持全图识别和单个字符分割识别：全图识别是基于lstm+ctc。
-------




![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/result_plateCard/QQ%E5%9B%BE%E7%89%8720180529195834.png)
![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/result_plateCard/QQ%E5%9B%BE%E7%89%8720180529195858.png)
  

![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/result_plateCard/QQ%E5%9B%BE%E7%89%8720180529195908.png)  


![image](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9/blob/master/result_plateCard/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20180530112203.png) 
欢迎交流：加QQ群 图像处理分析机器视觉 109128646
