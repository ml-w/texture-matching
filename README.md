# Introduction

## Purpose of project

The aim of this project is to come up with a system to compare and contrast the textures of two arbitrarily specified regions on an image, without regarding their morphologies.The ultimate goal is to enable comparison between two images of body tissues, and evaluate the textural similarity between them. The assumption is that similar tissues should present similar texture quantifier.

## Existing methodologies

In the field of computer vision and medical imaging, several techniques are employed to match textures between two images, some examples includes:

* **Feature-based methods**
* **Template matching**
* **Histogram-based methods**
* **Neural networks and deep learning**
* **Mutual information**

However, biological tissues can be highly irregular and diverse, presenting a significant challenge for conventional techniques. The inherent non-homogeneity of tissue means that textures can vary greatly within the same image or between images of the same type of tissue. Traditional methods like histogram matching or template matching might fail to capture these variations accurately.

Although deep learning method seems to be the most direct solution of this issue, this project intend to focuses on more general methods that can work on multiple tissues with out requirement of long training time and data collection.


# Data License

Data for python unittest were downloaded as per instruction here: https://torchio.readthedocs.io/datasets.html

See https://nist.mni.mcgill.ca/pediatric-atlases-4-5-18-5y/ for more details.
