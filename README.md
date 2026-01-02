# Surface Crack Detection using Deep Learning (CNN)

This project implements an end-to-end deep learning pipeline to automatically detect surface cracks in concrete structures using Convolutional Neural Networks (CNNs).  
It solves a real-world civil infrastructure inspection problem using computer vision.

---

## Problem Statement

Manual inspection of concrete structures is time-consuming, subjective, and prone to human error.  
This project aims to automate crack detection from surface images using deep learning, improving accuracy and scalability.

---

## Dataset

- Name: Surface Crack Detection Dataset
- Source: Kaggle  
  https://www.kaggle.com/datasets/arunrk7/surface-crack-detection

Dataset details:
- Total images: 40,000
- Crack images: 20,000
- Non-crack images: 20,000
- Image size: 227 × 227
- Color format: RGB
- Task: Binary image classification

Note: The dataset is not included in this repository due to size constraints.

---

## Model Architecture

A custom Convolutional Neural Network (CNN) was built from scratch.

Architecture components:
- Convolution layers with ReLU activation
- MaxPooling layers
- Dropout layers for regularization
- Fully connected dense layers
- Sigmoid output layer for binary classification

Output classes:
- 0 → Non-Cracked Surface
- 1 → Cracked Surface

---

## Training Configuration

- Loss function: Binary Crossentropy
- Optimizer: Adam
- Metric: Accuracy
- Validation data used during training

---

## Results and Evaluation

The model achieved strong performance with stable training and validation curves.

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Visualizations:
- Training vs Validation Loss
- Training vs Validation Accuracy
- Confusion Matrix Heatmap

---

## Tech Stack

- Python
- TensorFlow and Keras
- NumPy and Pandas
- Matplotlib and Seaborn
- Plotly
- Scikit-learn

---

## Project Structure

surface-crack-detection-cnn/

- detecting-cracks-in-concrete.ipynb
- README.md

---

## How to Run Locally

1. Clone the repository

   git clone https://github.com/Nadipalli-Sandeep/concrete-surface-crack-detection.git
   cd concrete-surface-crack-detection

3. Install dependencies

   pip install -r requirements.txt

3. Download the dataset from Kaggle

   https://www.kaggle.com/datasets/arunrk7/surface-crack-detection

4. Run the notebook

   jupyter notebook detecting-cracks-in-concrete.ipynb

---

## Future Improvements

- Transfer learning with ResNet or EfficientNet
- Grad-CAM visualization for explainability
- Hyperparameter tuning
- Deployment using FastAPI or Streamlit
- Real-time crack detection system

---

## Key Learnings

- End-to-end CNN workflow for image classification
- Dataset preprocessing and training pipelines
- Model evaluation using real-world metrics
- Visual interpretation of deep learning models

---

## Author

Sandeep  
M.Tech – Data Science and Artificial Intelligence  
IIT Tirupati

---

If you find this project useful, feel free to star the repository and explore further improvements.
