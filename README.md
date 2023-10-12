# Multiclass Segmentation 
This repository contains an image segmentation project using segmentation-models, tensorflow and keras. It is based on U-Net and includes code for training and evaluating a segmentation model for multiclass segmentation, as well as examples and guides to get you started.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Evaluation](#evaluation)
- [MIT License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

<a name="introduction"></a>

The project provides an image segmentation model that can be adapted to various segmentation tasks in multiclass segmentation. It is based on deep learning and uses a U-Net from segmentation-models. The initial purpose was for bone fragments and metallic instrument segmentation in lower limb fractures, but it can be used for other medical images, or for other non-medical images.

## Getting Started

To get started with this project, follow the instructions below.

### Prerequisites

Before running the code, you need to have the following prerequisites installed:

- Python 3.10.9
- TensorFlow 2.12.0
- segmentation-models 1.0.1
- Keras 2.12.0
- Numpy 1.23.5
- Albumentations 1.3.0
- glob2 0.7
- OpenCV 4.7.0.72
- PIL 9.5.0
- Matplotlib 3.7.1
- scikit-image 0.20.0
- scipy 1.10.1

### Installation

<a name="installation"></a>

If you are using Google Colab, you need to install firstly segmentation-models by running the following command at each run:

```bash
pip install segmentation-models
```


## Usage

<a name="usage"></a>

To use this segmentation model, follow the guidelines provided in the code. It works for two-class segmentation task (with background three classes), but you can adjust it accordingly for other number of classes. The masks/label images can be previously augmented with **augmentation.py**. 
Hyperparameters are to be adjusted in the training phase to your specific task. 
The datasplit for training/validation/testing is defined in the code 60/15/25. For training/validation 75% of the images and masks are used, and then splitted 80% for training and 20% for validation with:

```bash
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
```

## Folder structure

<a name="folder-structure"></a>

project/\
│\
├── path_to_images/\
│   ├── image1.png\
│   ├── image2.png\
│   └── ...\
├── path_to_masks/\
│   ├── mask1.png\
│   ├── mask2.png\
│   └── ...\
├── augmentation.py\
├── multiclass-segm.py\
└── README.md\

## Evaluation

<a name="evaluation"></a>

The segmentation model is evaluated using the following metrics:

- Dice score
- Jaccard index
- Precision
- Recall
- Confusion matrix

These metrics are calculated separately for the training, validation, and testing phases. Dice and Jaccard are further calculated for each segment.

## MIT License

<a name="license"></a>

Copyright <c> [2023] [Nora Dimitrova]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Acknowledgments


<a name="acknowledgments"></a>

I would like to express our gratitude to the developers and contributors of the libraries and tools used in this project.


