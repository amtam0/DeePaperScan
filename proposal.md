# DeePaperScan
A Deep Learning model that detects paper corners in pictures / videos

### Domain Backround

supervised learning

keypoints / landmark detection

Convolutional neural networks

### Problem Statement

Image scanner that detects Paper corners in Image / video using a CNN model

The idea is to help a user by using an A.I. algorithm to better detect a Scan corners in image papers.

There are 4 corners positions to detect of a paper document (Top left, Top right, Bottom left, Bottom right) in different backgrounds.

### Datasets and Inputs

Dataset resources are from (link)[https://sites.google.com/site/icdar15smartdoc/challenge-1/challenge1description] [1]

The dataset comes from the SMARTDOC 2015 competition (Smartphone Document Capture and OCR Competition)

The dataset comes with a complex structure containing videos and metadata inside xmls

It contains paper documents that are pictured in different backgrounds and angles.

### Solution Statement

The solution consists on :

- Convert the dataset to csv file to prepare it for training

- Prepare a sample of the Dataset  (around 7000 images), split it to train / validation

- Create a CNN architecture

- Train the model

- Test the model on the test images (outside of training / validation set)

- Discuss improvements and next steps


### Benchmark Model

Similar usecases using Keypoint detection : Facial Keypoint detection

### Evaluation Metrics

The usecase is mainly a Regression problem using images as input.

Evaluation metrics that will be used in the training / validation sets : 
- loss : mean squared error
- metrics : mean absolute error

### Project Design

Steps to develop this project:

1- Explore and prepare the dataset
2- Train and deploy the model on Sagemaker
3- Test the model on new images


#### Reference

[1] Jean-Christophe Burie, Joseph Chazalon, Mickaël Coustaty, Sébastien Eskenazi, Muhammad Muzzamil Luqman, Maroua Mehri, Nibal Nayef, Jean-Marc OGIER, Sophea Prum and Marçal Rusinol: “ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc)”, In 13th International Conference on Document Analysis and Recognition (ICDAR), 2015.
[2] https://fairyonice.github.io/achieving-top-23-in-kaggles-facial-keypoints-detection-with-keras-tensorflow.html
[3] https://gitlab.com/juliensimon/dlnotebooks/blob/master/keras/05-keras-blog-post/Fashion%20MNIST-SageMaker.ipynb
