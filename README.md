# Handwritten-Digit-Classification-Using-Convolution-Neural-Network

## Project Description

### Handwritten Digit Classification using CNN

This project aims to classify handwritten digits using a Convolutional Neural Network (CNN) model. The dataset used for this project is the MNIST dataset, which is a standard dataset in the machine learning community for benchmarking image classification algorithms.

### Dataset

The MNIST dataset consists of 70,000 images of handwritten digits (0-9) split into 60,000 training images and 10,000 test images. Each image is a 28x28 pixel grayscale image, representing a single handwritten digit.

### Data Preprocessing

1. **Loading the Data**: The dataset is loaded using the `mnist.load_data()` function from TensorFlow.
2. **Normalization**: The pixel values of the images are normalized to the range [0, 1] by dividing by 255.0.
3. **Reshaping**: The images are reshaped to include a single channel (grayscale), resulting in a shape of (28, 28, 1) for each image.

### Model Architecture

The CNN model is built using the Sequential API from TensorFlow's Keras module. The architecture consists of the following layers:

1. **Conv2D Layer**: The first convolutional layer with 32 filters, a kernel size of (3, 3), and ReLU activation.
2. **MaxPooling2D Layer**: The first pooling layer with a pool size of (2, 2).
3. **Conv2D Layer**: The second convolutional layer with 64 filters, a kernel size of (3, 3), and ReLU activation.
4. **MaxPooling2D Layer**: The second pooling layer with a pool size of (2, 2).
5. **Conv2D Layer**: The third convolutional layer with 128 filters, a kernel size of (3, 3), and ReLU activation.
6. **MaxPooling2D Layer**: The third pooling layer with a pool size of (2, 2).
7. **Flatten Layer**: Flattens the output of the previous layer to create a 1D vector.
8. **Dense Layer**: Fully connected layer with 128 units and ReLU activation.
9. **Dropout Layer**: Dropout layer with a dropout rate of 0.5 to prevent overfitting.
10. **Dense Layer**: Fully connected layer with 64 units and ReLU activation.
11. **Dropout Layer**: Dropout layer with a dropout rate of 0.5 to prevent overfitting.
12. **Dense Layer**: Output layer with 10 units (one for each digit) and softmax activation.

### Model Compilation and Training

1. **Compilation**: The model is compiled using the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.
2. **Training**: The model is trained for 30 epochs with a batch size of 64. During training, both training and validation accuracy and loss are tracked.

### Model Evaluation

The trained model is evaluated on the test set to determine its accuracy, which is reported to be 98%.

### Visualization

The training and validation accuracy and loss are plotted to visualize the model's performance over the epochs.

### Predicting Digits

A function is provided to predict the digit from an input image. The user can input an image from the test set or provide their own image, and the model will classify the digit and display the predicted digit along with the image.

### User Interface

A command-line interface (CLI) is provided for users to input the index of an image from the test set to see the model's prediction and the accuracy of the predicted input.

### Additional Feature

A function is included to preprocess and predict digits from user-provided images, allowing for real-world testing of the model's classification capabilities.
