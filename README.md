# Image Classification using Convolutional Neural Network (CNN)

This repository contains code for training a Convolutional Neural Network (CNN) to classify images into two classes: "Happy" and "Sad". The dataset used for training and testing the model is assumed to be stored in the `data` directory.

## Requirements
- Python 3
- TensorFlow
- Keras
- OpenCV (cv2)
- NumPy
- Seaborn
- Matplotlib

Install the dependencies using pip:
```bash
pip install tensorflow keras opencv-python-headless numpy seaborn matplotlib
```

## Description
- The script begins by preprocessing the dataset, ensuring all images have valid extensions and scaling pixel values to the range [0, 1].
- The dataset is then split into training, validation, and testing sets with respective proportions of approximately 70%, 20%, and 10%.
- The CNN model architecture consists of convolutional layers followed by max-pooling layers and fully connected layers.
- The model is trained using the Adam optimizer and binary cross-entropy loss for 20 epochs, with monitoring of accuracy.
- Training progress is visualized using TensorBoard for loss and accuracy metrics.
- After training, the model is evaluated on the testing set, computing precision, recall, and accuracy metrics.
- Finally, an example image (happytest2.jpg) is used to demonstrate model prediction and a confusion matrix is generated to visualize classification performance.

## Results
- The model achieves satisfactory precision, recall, and accuracy on the test set.
- Confusion matrix provides insight into the model's classification performance, indicating true positive, false positive, true negative, and false negative predictions.

## Note
- You can replace the example image (happytest2.jpg) with your own images to test the model's predictions.
- Adjust the model architecture, hyperparameters, and training settings as needed for your specific use case.
- Experiment with different preprocessing techniques or augmentations to improve model performance.