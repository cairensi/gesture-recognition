# 基于TSN网络模型的手语识别系统

## Author:CRS Club，Wizard
## 前言
看很多大佬做的手语识别都是用tensorflow（Google）的，我寻思着整个国产（paddlepaddle）的是吧，于是就有了这个项目
后续还会建个分支做实时识别控制的（和下位机通信，通过串口函数将识别结果传给下位机，实现根据手势动作控制的功能），要开学了，就先鸽着吧
## 直接使用方法

### 转化测试集数据
```shell
pip install wget
python GetTestdata.py
```

### 预测该视频代表的手语信息
```shell
python freeze_infer.py
```

## 数据集的结构

### label.npy的结构

```python
import numpy as np
label = np.load('label.npy', allow_pickle=True) 
print(label)
```

说明：label.npy的结构为一个字典，key为视频对应的标签，如'please'、'walk'等，而value为int值用以区分不同类。

### **pkl的结构**

```python
import pickle
f = open('dataset/train/come1.pkl', 'rb')
pkl_file = pickle.load(f)
f.close()
```

说明：pkl的结构为：(文件名，该文件对应标签的int值，[视频帧1的地址，视频帧2的地址，视频帧3的地址.....])

## **自制数据集**

Tips:自制数据集，请解压自制数据集后运行一次GetDataset.py，不然会生成重复的文件QAQ

1.先建立一个文件夹叫dataset

2.在dataset文件夹里再建立文件夹对应着手语的label

3.运行GetDataset.py

## 开始TSN网络模型训练

### 训练

```python
python train.py --model_name TSN \
                    --epoch 60 \
                    --save_dir 'checkpoints_models' \
                    --use_gpu True \ #看你有没有GPU，有的话先设置下CUDA的变量哦，没有就改成False
                    --pretrain data/ResNet50_pretrained
```

### 固化模型

```shell
python freeze.py --weights 'checkpoints_models'
```

### 预测

```shell
python infer.py --weights 'checkpoints_models' --use_gpu True --save_dir 'infer'
```

### 制作片段

```python
# 制作片段，用作特征以区分不同动作
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

def clip_video(source_file, target_file, start_time, stop_time):

    source_video = VideoFileClip(source_file)
    video = source_video.subclip(int(start_time), int(stop_time))  # 执行剪切操作
    video.write_videofile(target_file)

time = 1
for i in range(5):
    clip_video('test/test/test.mp4', 'test/test/test_'+str(i)+'.mp4', i * time, (i + 1) * time)

```

### 运行

```shell
python infer_video.py
```

#### PS：

潘凯昕和张睿哲十分的气人
有问题提交issues
Thank you,Pablo Gomez.
#### 最后求Star

