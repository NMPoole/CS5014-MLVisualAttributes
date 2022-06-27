# CS5014 Machine Learning: Visual Attributes

This repository demonstrates work completed as part of the **CS5014 Machine Learning** module for Practical 2. A multi-layer perceptron (i.e., feed-forward neural network) was trained and evaluated for the task of predicting the colour and texture of image objects from a set of features pre-extracted from a subset of the [GQA dataset](https://arxiv.org/abs/1902.09506). Furthermore, a Random Forest Classifier (i.e., ensemble of decision tree classifiers) was also trained for the purpose of comparing the two different approaches.

The aim of this practical was to apply machine learning on a recent, open-ended research problem with the freedom to choose the most appropriate approach (with justification). The submission demonstrates an understanding of:
- How to select and train a suitable classification model;
- How to evaluate and compare the performance of different models; and 
- How to explain and justify the work in a technical report.

The dataset is based on a subset of the GQA dataset for learning attributes and relations. The GQA dataset consists of images where objects are annotated in terms of bounding boxes and relevant attributes and relations. The task was to correctly identify the colour and texture for these objects based on already extracted features (the images were not processed as part of this project). There are two files of concern: 
- *data_train.csv* contains the training set where rows represent individual objects, and columns represent features extracted from that object’s bounding box. The columns labelled ‘colour’ and ‘texture’ are target variables that are predicted from the remaining features.
- *data_test.csv* contains data in very similar format as *data_train.csv* and serves as the test dataset. Note that there are no columns for ‘colour’ and ‘texture’ here so this file cannot be used for training.
