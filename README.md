# Concrete-Crack-Detection-Segmentation

This repository contains the code for crack detection in concrete surfaces. It is a PyTorch implementation of Deep Learning-Based Crack Damage Detection Using Convolutional Neural Networks with DeepCrack

DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation


Resources: [(Paper: https://github.com/yhlleo/DeepCrack/blob/master/paper/DeepCrack-Neurocomputing-2019.pdf)]
Architecture: based on Holistically-Nested Edge Detection, ICCV 2015.


Dependencies required:

PyTorch,
OpenCV,
Dataset -The data set can be downloaded from this link: https://data.mendeley.com/datasets/5y9wdsg2zt/2

Dataset:
The dataset contains concrete images having cracks. The data is collected from various METU Campus Buildings.
The dataset is divided into two as negative and positive crack images for image classification. 
Each class has 20000 images with a total of 40000 images with 227 x 227 pixels with RGB channels. 
The dataset is generated from 458 high-resolution images (4032x3024 pixel) with the method proposed by Zhang et al (2016). 
High-resolution images have variance in terms of surface finish and illumination conditions. 
No data augmentation in terms of random rotation or flipping is applied. 

The dataset file creates the training dataset class to be fed into the Convolutional Neural Network. This class automatically determines the number of classes by the number of folders in 'in_dir' (number of folders=number of classes)

![Capture](https://github.com/yhlleo/DeepCrack/blob/master/figures/architecture.jpg?raw=true)
![Capture](https://user-images.githubusercontent.com/46296774/103016160-edd0b180-4541-11eb-8cfe-3c7680569eb9.PNG)
![Capture2](https://user-images.githubusercontent.com/46296774/103016173-f4f7bf80-4541-11eb-9bb5-933dcd725d9b.PNG)




2018 – Özgenel, Ç.F., Gönenç Sorguç, A. “Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings”, ISARC 2018, Berlin.
