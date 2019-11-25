# Kuzushiji-DropBlock
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![HitCount](http://hits.dwyl.io/sujatasaini/Kuzushiji-DropBlock.svg)](http://hits.dwyl.io/sujatasaini/Kuzushiji-DropBlock)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)
[![Issues](https://img.shields.io/github/issues/sujatasaini/Kuzushiji-DropBlock)](https://github.com/sujatasaini/Kuzushiji-DropBlock)
[![Contributors](https://img.shields.io/github/contributors/sujatasaini/Kuzushiji-DropBlock)](https://github.com/sujatasaini/Kuzushiji-DropBlock)
[![Licence](https://img.shields.io/github/license/sujatasaini/Kuzushiji-DropBlock)](https://github.com/sujatasaini/Kuzushiji-DropBlock)

- [Overview](##overview)
- [Documentation](##documentation)
- [System Requirements](##system-requirements)
- [Installation Guide](##installation-guide)
- [Algorithm](##DCNN-DropBlock-Algorithm's-flow)
- [Data](##Dataset)
- [Results](##Benchmarks-&-Results)
- [Visualization](##Visualization)
- [Credits](##Credits)
- [Citation](##Citation)
- [License](#license)

## Overview

Kuzushiji-Dropblock is a Project based on Recognizition of Japanese Historical Image classification.

## Documnetation

:memo: View our research paper titled "__Japanese Historical Character Recognition using
Deep Convolutional Neural Network (DCNN)
with DropBlock Regularization__" availble at (http://dx.doi.org/10.35940/ijrte.b2923.078219)

## System Requirements 
### Hardware requirements
 :rocket: `Kuzushiji-DropBlock` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
This code is supported for *Windows*, *macOS* and *Linux*. The code has been tested on the following systems:
+ Windows: Professional (10)
+ macOS: Mojave (10.14.1)
+ Linux: Ubuntu 16.04

### ML-Dependencies
`Kuzushiji-DropBlock` mainly depends on the Python scientific stack.

```
Keras
numpy
scipy
pandas
matplotlib
scikit-learn
PyTorch
Tensorflow
DropBlock
```
## Installation-Guide

In Bash, 
```
pip install DropBlock2D
```
In Google Colab,
```
!pip install DropBlock2D
```
GitClone
```
git clone https://github.com/sujatasaini/Kuzushiji-DropBlock
cd Kuzushiji-DropBlock
```

## DCNN-DropBlock Algorithm's flow

![Algorithm](https://raw.githubusercontent.com/sujatasaini/Kuzushiji-DropBlock/master/DropBlock.png)

## Dataset

:file_folder: **Kuzushiji-MNIST** is a drop-in replacement for the MNIST dataset (28x28 grayscale, 70,000 images), provided in the original MNIST format as well as a NumPy format. Since MNIST restricts us to 10 classes, we chose one character to represent each of the 10 rows of Hiragana when creating Kuzushiji-MNIST.

Kuzushiji-MNIST contains 70,000 28x28 grayscale images spanning 10 classes (one from each column of [hiragana](https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Table_hiragana.svg/768px-Table_hiragana.svg.png)), and is perfectly balanced like the original MNIST dataset (6k/1k train/test for each class).

| File            | Examples | Download (MNIST format)    | Download (NumPy format)      |
|-----------------|--------------------|----------------------------|------------------------------|
| Training images | 60,000             | [train-images-idx3-ubyte.gz](http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz) (18MB) | [kmnist-train-imgs.npz](http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz) (18MB)   |
| Training labels | 60,000             | [train-labels-idx1-ubyte.gz](http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz) (30KB) | [kmnist-train-labels.npz](http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz) (30KB)  |
| Testing images  | 10,000             | [t10k-images-idx3-ubyte.gz](http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz) (3MB) | [kmnist-test-imgs.npz](http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz) (3MB)   |
| Testing labels  | 10,000             | [t10k-labels-idx1-ubyte.gz](http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz) (5KB)  | [kmnist-test-labels.npz](http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz) (5KB) |

Mapping from class indices to characters: [kmnist_classmap.csv](http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist_classmap.csv) (1KB)

We recommend using standard top-1 accuracy on the test set for evaluating on Kuzushiji-MNIST.

##### Which format do I download?
If you're looking for a drop-in replacement for the MNIST or Fashion-MNIST dataset (for tools that currently work with these datasets), download the data in MNIST format.

Otherwise, it's recommended to download in NumPy format, which can be loaded into an array as easy as:  
`arr = np.load(filename)['arr_0']`.

**Kuzushiji-49**, as the name suggests, has 49 classes (28x28 grayscale, 270,912 images), is a much larger, but imbalanced dataset containing 48 Hiragana characters and one Hiragana iteration mark.

Kuzushiji-49 contains 270,912 images spanning 49 classes, and is an extension of the Kuzushiji-MNIST dataset.

| File            | Examples |  Download (NumPy format)      |
|-----------------|--------------------|----------------------------|
| Training images | 232,365            | [k49-train-imgs.npz](http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz) (63MB)   |
| Training labels | 232,365            | [k49-train-labels.npz](http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz) (200KB)  |
| Testing images  | 38,547             | [k49-test-imgs.npz](http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz) (11MB)   |
| Testing labels  | 38,547             | [k49-test-labels.npz](http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz) (50KB) |

Mapping from class indices to characters: [k49_classmap.csv](http://codh.rois.ac.jp/kmnist/dataset/k49/k49_classmap.csv) (1KB)

We recommend using balanced accuracy on the test set for evaluating on Kuzushiji-49.

**Kuzushiji-Kanji** is an imbalanced dataset of total 3832 Kanji characters (64x64 grayscale, 140,426 images of both common and rare characters), ranging from 1,766 examples to only a single example per class.

The full dataset is available for download [here](http://codh.rois.ac.jp/kmnist/dataset/kkanji/kkanji.tar) (310MB). 

## Get the data 

💾  You can run [`python download_data.py`](download_data.py) to interactively select and download any of these datasets!
    You can also download the data from [Kaggle](https://www.kaggle.com/anokas/kuzushiji)

## Benchmarks & Results 

📈 Results of our DCNN Model with different regularization methods (DropBlock, Dropout, SpatialDropout) on MNIST, Fashion-MNIST Kuzushiji-MNIST and Kuzushiji-49, trained on Google Colab average over 3 runs.

|Models                           | MNIST | Fashion-MNIST | Kuzushiji-MNIST | Kuzushiji-49 |
|---------------------------------|-------|---------------|-----------------|--------------|
|[DCNN-DropBlock](DropBlock/Kuzushiji-MNIST/train.py)     | **99.47%** | **93.40%** | **97.66%** | **95.67%** |
|[DCNN-Dropout](Dropout/train.py)                         | 97.99% | 85.47% | 86.43% | 95.34% |
|[DCNN-Spatial-Dropout](SpatialDropout/train.py)          | 97.17% | 84.44% |  81.08% | 58.18 |

Have more results to add to the table? Feel free to submit an [issue](https://github.com/sujatasaini/Kuzushiji-DropBlock/issues/new) or [pull request](https://github.com/sujatasaini/Kuzushiji-DropBlock/compare)!

|Models                           | MNIST | Fashion-MNIST | Kuzushiji-MNIST | Kuzushiji-49 |
|---------------------------------|-------|---------------|-----------------|--------------|
|[4-Nearest Neighbor Baseline](Models/4-NearestNeighborBaseline/train.py)| 97.14% | 85.97% | 91.59% | 86.00% |
|[Naive-Bayes](Models/Naive-Bayes/train.py)                              | 98.06% | 86.60% | 92.17% | 88.44% |
|[AlexNet](Models/AlexNet/train.py)                                      | 98.19% | 87.47% | 91.82% | 81.01% |
|[Simple CNN](Models/SimpleCNN/train.py)                                 | 99.08% | 92.54% | 95.02% | 90.42% |
|[Transfer Learning with CNN](Models/Transfer Learning with CNN/train.py)| 99.34% | 97.46% | 97.06% | 83.96% |
|[LeNet-5](Models/LeNet-5/train.py)                                      | 99.13% | 91.33% | 94.66% | 89.64% |
|[MobileNet](Models/MobileNet/train.py)                                  | 99.20% | 93.04% | 95.09% | 91.06% |
|[DCNN-DropBlock](DropBlock/Kuzushiji-MNIST/train.py)    | **99.47%** | **93.40%** | **97.66%** | **95.67%** |

:bar_chart: `* scheduled dropblock with block_size=5 and increasing drop_prob 
from 0.0 to 0.25 over 5000 iterations`

Example available [here](Kuzushiji-DropBlock/DropBlock/Kuzushiji-MNIST/train.py)

## Visualization

![Graph](https://raw.githubusercontent.com/sujatasaini/Kuzushiji-DropBlock/master/accuracy_loss.png)

## Credits 

1. [Keras-DropBlock](https://github.com/CyberZHG/keras-drop-block) :hibiscus:
2. [KMNIST Dataset](https://github.com/rois-codh/kmnist)

## Citation 

Please cite `Kuzushiji-DropBlock` in your publications if it helps your research::100:

@article{2019,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     doi = {10.35940/ijrte.b2923.078219}, <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      url = {https://doi.org/10.35940%2Fijrte.b2923.078219}, <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      year = 2019, <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      month = {jul}, <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      publisher = {Blue Eyes Intelligence Engineering and Sciences Engineering and Sciences Publication - {BEIESP}}, <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      volume = {8}, <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      number = {2}, <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      pages = {3510--3515}, <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      title = {Japanese Historical Character Recognition using Deep Convolutional Neural Network ({DCNN}) with Drop Block Regularization}, <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      journal = {International Journal of Recent Technology and Engineering} <br>
}
## License :scroll:
----
MIT
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
