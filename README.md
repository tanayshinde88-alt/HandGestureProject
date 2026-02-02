Hand Gesture Classification using SVM 

This project builds a machine learning system to classify hand gestures from image data using a Support Vector Machine (SVM). The model is trained on grayscale gesture images and learns to recognize different gesture classes.

The aim is to demonstrate how traditional ML techniques can be used for image-based gesture recognition.

+ Repository Structure
hand-gesture-svm-classifier/
│
├── gesture_classification.py     → Main Python script
├── leapGestRecog/               → Dataset folder
│   └── 00/                      → Subset of gesture images
│       ├── 01/ 02/ 03/...       → Gesture class folders
└── README.md                    → Project documentation


Note:
The full dataset is not uploaded due to size limits.
You should place the Leap Motion gesture dataset inside the leapGestRecog/00/ directory.

+ Project Overview

Loads hand gesture images from folders

Converts images to grayscale

Resizes images to 64 × 64 pixels

Flattens image data into numerical vectors

Splits data into training and testing sets

Trains a linear SVM classifier

Evaluates model accuracy

+ Technologies Used

Python

NumPy

Pillow (PIL)

Scikit-learn

+ How to Run the Project

Install required libraries:

pip install numpy pillow scikit-learn


Make sure your folder structure looks like this:

leapGestRecog/
  00/
    01/
    02/
    03/
    ...


Run the script:

python gesture_classification.py

+ Output

Console output showing:

Detected gesture classes

Images loaded successfully

Final model accuracy in percentage

Example:

Model Accuracy: 91.25 %
Program finished successfully

+ Use Case

This project helps with:

Understanding gesture recognition systems

Learning image preprocessing for ML

Applying SVM to multi-class classification

+ Future Improvements

Use feature extraction (HOG / PCA)

Try different SVM kernels

Add real-time gesture detection

Save and load trained models




Note:
The dataset is not included due to size limits.
Download it from: <https://www.kaggle.com/datasets/gti-upm/leapgestrecog>

Place it in the folder:

leapGestRecog/00


Inside that, you should have:

leapGestRecog/00/01
leapGestRecog/00/02
leapGestRecog/00/03
...
