### gluon custom数据训练

#### 准备自制的数据

数据文件夹格式应该与pascal voc一样：

```
VOCtemplate
└── VOC2018
    ├── Annotations
    │   └── 000001.xml
    ├── ImageSets
    │   └── Main
    │       └── train.txt
    └── JPEGImages
        └── 000001.jpg
```

xml包含框位置，train.txt包含图片与对应xml索引

将文件放在与VOC2007,VOC2012并列的位置

```
~/.mxnet/datasets/voc/VOC2007
~/.mxnet/datasets/voc/VOC2012
~/.mxnet/datasets/voc/VOC2018
```

####　更改读数据的路径

```
# train_yolov3.py
train_dataset = gdata.VOCDetection(splits=[(2018, 'train'),])             
val_dataset = gdata.VOCDetection(splits=[(2018, 'val')])
```

#### 更改训练类别

```
# gluoncv/data/pascal_voc/detection.py
CLASSES = ['car']
```

#### 推理的时候

推理的时候修改--pretrained这个参数，指向训练好的参数

自制数据：https://gluon-cv.mxnet.io/build/examples_datasets/detection_custom.html

