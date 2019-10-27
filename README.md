# DeePaperScan
A Deep Learning model that detect paper corners in pictures / videos / Smartphone Camera

### Domain Backround

Object keypoints detection and classification



### Problem Statement
Image scanner that detects Paper corners in Image / video

The idea is to help a user by using an A.I. algorithm to better detect a Scan corners before applying image processing

### Datasets and Inputs

Dataset resources are from (link)[https://sites.google.com/site/icdar15smartdoc/challenge-1/challenge1description] [1]
This dataset is SMARTDOC 2015 competition (Smartphone Document Capture and OCR Competition)

The dataset comes with a complex structure containing videos and labels inside xmls (TOADD numbers)

The dataset needs to be processed and prepared for training

### Solution Statement

The solution consists of :

Adapting the dataset architecture to responds to the standards used for training models (ex. using Coco Architecture etc.)

Prepare a small sample of Dataset  (around 2000 frames), split it to train / validation / test sets

This is mainly a CNN problem using Object detection

Testing different Frameworks based on exiting examples using python Frameworks (mainly Pytorch and Tensorflow)


### Benchmark Model

Similar to existing usecases using Keypoint detection : Facial Keypoint detection, Human pose estimation

### Evaluation Metrics

As this is more a Regression problem using images

The main evaluation metric would be the MSELoss

### Project Design
The Workflow consists on restructuring the Dataset for Training



Two main goals to tackle this usecase:

1- Build a model that detect keypoints in images
2- Adapting model to do prediction in Realtime (using videos or a Smartphone Camera)


#### Reference

[1] Jean-Christophe Burie, Joseph Chazalon, Mickaël Coustaty, Sébastien Eskenazi, Muhammad Muzzamil Luqman, Maroua Mehri, Nibal Nayef, Jean-Marc OGIER, Sophea Prum and Marçal Rusinol: “ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc)”, In 13th International Conference on Document Analysis and Recognition (ICDAR), 2015.
