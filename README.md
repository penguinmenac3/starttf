# Tensorflow - Starterpack

This repo aims to contain everything required to quickly develop a deep neural network with tensorflow.
It comes with several dataset loaders and network architectures.
If I can find it the models will also contain pretrained weights.
The idea is that if you use existing dataset loaders and networks and only modify them, you will automatically obey best practices and have super fast training speeds.

## Install

First clone the repository **recursively**!

```bash
git clone --recursive https://github.com/penguinmenac3/tensorflow-starterpack.git
```

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

For dataset support I use my own dataset library including bindings to load many popular datasets in a unified format.

However, to use it you will need to initialize git submodules if you did not do a recursive clone:

```bash
git submodule update --init --recursive
```

The dataset loader basically supports **classification**, **segmentation**, **regression** (including **2d- and 3d-detection**) and some visualization helpers.
For details checkout the readme of the project [**here**](https://github.com/penguinmenac3/datasets/blob/master/README.md).

## Models

Every [model](models/model.py) supports setup, predict, fit and export methods.

There are some models implemented to tinker around with.
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
2. [LFW](examples/lfw_example.py)
3. Imagenet (Baselines) [TODO]
4. Bounding Box Regression [TODO]
5. Segmentations [TODO]
6. Instance Masks [TODO]
7. Reinforcement Learning [TODO]
