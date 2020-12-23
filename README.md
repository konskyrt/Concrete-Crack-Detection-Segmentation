# Concrete-Crack-Detection
This repository contains the code for crack detection in concrete surfaces. It is a PyTorch implementation of the paper by by Young-Jin Cha and Wooram Choi - "Deep Learning-Based Crack Damage Detection Using Convolutional Neural Networks"


The model acheived 98% accuracy on the validation set. A few results are shown below

![Capture](https://user-images.githubusercontent.com/46296774/103016160-edd0b180-4541-11eb-8cfe-3c7680569eb9.PNG)
![Capture2](https://user-images.githubusercontent.com/46296774/103016173-f4f7bf80-4541-11eb-9bb5-933dcd725d9b.PNG)

Dependencies required:

PyTorch
OpenCV
Dataset -The data set can be downloaded from this link: https://data.mendeley.com/datasets/5y9wdsg2zt/2

The dataset file creates the training dataset class to be fed into the Convolutional Neural Network. This class automatically determines the number of classes by the number of folders in 'in_dir' (number of folders=number of classes)
