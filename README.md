# Project---CNN-for-Image-Classification
This project uses a Convolutional Neural Network (CNN) implemented with TensorFlow/Keras to classify images of cats and dogs. The model is trained on a dataset of cat and dog images and then used to predict new images.

Table of Contents
1. Project Overview
2. Dataset Structure
3. Dependencies
4. Model Architecture
5. Results
6. Project Structure


1. Project Overview
This project trains a Convolutional Neural Network to perform binary classification on images of cats and dogs. The pipeline includes:

Data preprocessing (rescaling, augmentation).
Building a CNN model with convolution, pooling, and fully connected layers.
Training and evaluating the model.
Predicting new images from a specified folder.

2. Dataset Structure
The project expects a folder structure like this:
Project - CNN for Image Classification/
│
├── dataset/
│   ├── training_set/
│   │   ├── cats/
│   │   ├── dogs/
│   ├── test_set/
│   │   ├── cats/
│   │   ├── dogs/
│
├── single_prediction/
│   ├── image1.jpg
│   ├── image2.jpg

3. Dependencies
Ensure you have the following libraries installed:

Python 3.8 or later
TensorFlow (with Keras integration)
NumPy
Pillow

4. Model Architecture
The CNN consists of:

Convolutional Layers:

First Conv Layer: 32 filters, kernel size (3x3), ReLU activation.
Second Conv Layer: 64 filters, kernel size (3x3), ReLU activation.
Pooling Layers:

MaxPooling applied after each convolution.
Flattening:

Converts the output to a 1D vector.
Fully Connected Layers:

Dense layer with 128 neurons and ReLU activation.
Dropout layer to prevent overfitting.
Output layer with a sigmoid activation function for binary classification.

5. Results
After training for 25 epochs, the model achieves significant accuracy for distinguishing between cats and dogs. Results can be evaluated on the test set.

6. Project Structure
Project - CNN for Image Classification/
│
├── dataset/
│   ├── training_set/
│   ├── test_set/
│
├── single_prediction/
│
├── CNN for Image Classification.ipynb
├── cat_dog_classifier.h5
├── README.md

Link to dataset file: https://drive.google.com/file/d/1YUvSL7px1UTm2Rx6EGCN2bj22fobCdVT/view?usp=drive_link
Since I am currently in China so I cannot post anything to medium.com. Since that website reqiure a foreign phone number to authenticate account. I will copy the post from medium.com and leave it in github link.

