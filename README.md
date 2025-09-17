# MNIST Handwritten Digit Recognition using CNN

## Project Overview
This project focuses on classifying handwritten digits (0–9) using a Convolutional Neural Network (CNN). Handwritten digit recognition is a classic problem in computer vision that helps understand image preprocessing, CNN architecture, model training, evaluation, and predictions.

The MNIST dataset contains 70,000 grayscale images of size 28×28 pixels:
- 60,000 training images
- 10,000 testing images

## Why This Project
- Learn how to preprocess images for neural networks.
- Understand CNN layers such as convolution, pooling, flattening, dense, and dropout.
- Practice training, evaluating, and saving/loading models.
- Apply the trained model to predict real-world handwritten digits.

## Technologies Used
- Python 3
- TensorFlow & Keras
- NumPy
- Matplotlib
- Pillow (PIL)

## Project Steps

### 1. Data Loading and Exploration
The MNIST dataset was loaded from Keras. Training and testing images and labels were inspected to understand the data shape and distribution.

### 2. Data Preprocessing
Images were normalized to a [0,1] range and reshaped to match the CNN input requirements. Labels were one-hot encoded for multi-class classification.

### 3. Building the CNN
The model consists of:
- Convolutional layers to extract spatial features.
- MaxPooling layers to reduce dimensionality and prevent overfitting.
- Flatten layer to convert 2D features to 1D for fully connected layers.
- Dense layers with Dropout for classification and regularization.
- Softmax output for 10 classes (digits 0–9).

### 4. Model Compilation and Training
The model was compiled with RMSProp optimizer and categorical crossentropy loss. It was trained for 5 epochs on the training set and validated on the test set.

### 5. Model Evaluation
The trained model achieved approximately **99% test accuracy** on the MNIST test set, indicating excellent performance.

### 6. Model Saving
The trained CNN was saved to disk, allowing it to be reused without retraining.

### 7. Predicting Custom Images
Custom handwritten digit images (28×28 grayscale) were tested with the trained model. Out of 10 images, the model predicted **6 out of 10 correctly**.  

- Challenges:
  - Variations in handwriting styles affected accuracy.
  - Some images may require preprocessing adjustments for better results.

## Key Learnings
- Understanding and building CNN architectures for image classification.
- Importance of preprocessing steps like normalization, reshaping, and one-hot encoding.
- Training, evaluating, and saving deep learning models.
- Applying the trained model to new, unseen images and interpreting the results.
