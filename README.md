# Tensorflow - Starterpack

This repo aims to contain everything required to quickly develop a deep neural network with tensorflow.
It comes with several dataset loaders and network architectures.
If I can find it the models will also contain pretrained weights.
The idea is that if you use existing dataset loaders and networks and only modify them, you will automatically obey best practices and have super fast training speeds.

## Install

Simply create a new python virtual environment (preferably python 3.x) and install the requirements listed in the [requirements.txt](requirements.txt).
Properly install tensorflow-gpu please follow the [official instructions](https://www.tensorflow.org/install/) carefully.


## Running examples

Either use an ide such as pycharm and set your working directory to the keras-starterpack folder.
Or launch your code from the command line like the example bellow.

```bash
# Activate your virtual environment (in my case venv)
# Then do the following
(venv) $ cd /PATH/TO/tensorflow-starterpack
(venv) $ pyhon -m examples.mnist
```

## Datasets

There are handlers for several datasets.
To get you started quickly.

1. [Named Folders (Foldername = Label)](datasets/classification/named_folders.py)
2. [MNIST](datasets/classification/mnist.py)
3. ImageNet [TODO]
4. Coco [TODO]
5. [Cifar10/Cifar100](datasets/classification/cifar.py)
6. [LFW (named folders)](datasets/classification/named_folders.py)
6. PASCAL VOC [TODO]
7. Places [TODO]
8. Kitti [TODO]
9. Tensorbox [TODO]
10. CamVid [TODO]
11. Cityscapes [TODO]
12. ROS-Robot (as data source) [TODO]

## Models

There are also some models implemented to tinker around with.
Most of the implementations are not done by me from scratch but rather refactoring of online found implementations.
Also the common models will come with pre trained weights I found on the internet.
Just check the comment at the top of their source files.

1. Alexnet (single stream version) [TODO]
2. VGG 16 [TODO]
3. GoogLeNet (Inception v3) [TODO]
4. Overfeat/Tensorbox [TODO]
5. ResNet [TODO]
6. SegNet [TODO]
7. Mask RCNN [TODO]
8. monoDepth [TODO]

More non famous models by myself:

1. CNN for MNIST (Digit Recognition) [TODO]
3. CNN for LFW (Person Identification) [TODO]

## Examples

Some samples that should help getting into stuff.

Code:

1. MNIST [TODO]
2. LFW [TODO]
3. Imagenet (Baselines) [TODO]
4. Bounding Box Regression [TODO]
5. Segmentations [TODO]
6. Instance Masks [TODO]
7. Reinforcement Learning [TODO]
