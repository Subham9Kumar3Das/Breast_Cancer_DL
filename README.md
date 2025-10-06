🩺 Breast Cancer Detection using Deep Learning
📘 Overview

This project applies Deep Learning techniques to detect breast cancer based on diagnostic data.
The goal is to develop a robust and accurate classification model that can assist in early cancer detection and support healthcare professionals in medical decision-making.

The notebook demonstrates the complete Deep Learning workflow — from data preprocessing to model evaluation — ensuring reproducibility and transparency.

🧠 Key Features

End-to-end Deep Learning pipeline for medical data classification

Data preprocessing, normalization, and feature scaling

Implementation of neural network architectures (ANN / CNN)

Performance evaluation using accuracy, precision, recall, F1-score, and confusion matrix

Visualizations for insights into feature importance and prediction performance

🧩 Dataset

Name: Breast Cancer Wisconsin (Diagnostic) Dataset

Source: Scikit-learn dataset / UCI Machine Learning Repository

Description:
The dataset contains features computed from digitized images of fine-needle aspirates (FNAs) of breast masses.
Each instance is labeled as either Malignant (M) or Benign (B).

Attribute	Description
Mean radius, texture, perimeter, area, smoothness, etc.	Real-valued input features
Diagnosis	Target variable (M = malignant, B = benign)
⚙️ Technologies Used

Programming Language: Python

Libraries & Frameworks:

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

🚀 Model Architecture

A fully connected Deep Neural Network (DNN) was built using Keras Sequential API.

Typical architecture:

Input Layer (30 features)
↓
Dense (20) + ReLU 
↓
Dense (1) + Sigmoid


Loss Function: Binary Cross-Entropy

Optimizer: Adam

Metrics: Accuracy

📊 Results
Metric	Score
Training Accuracy	~98%
Test Accuracy	~97%
F1-Score	High Precision and Recall for both classes

Confusion matrix and ROC curve visualizations are included in the notebook to interpret model performance.

📁 Project Structure
📦 Breast_Cancer_Detection
 ┣ 📜 Breast_Cancer.ipynb      # Main Jupyter Notebook
 ┣ 📜 README.md                # Project documentation
 ┗ 📂 data/                    # (Optional) Dataset folder if using local data

🧪 How to Run

Clone this repository

git clone https://github.com/your-username/Breast_Cancer_Detection.git
cd Breast_Cancer_Detection

