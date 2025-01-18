# Breast Cancer Prediction Using Machine Learning

This project is a web application that predicts whether a breast cancer tumor is malignant or benign based on various features. The application leverages machine learning techniques to train classifiers and provide an intuitive interface for users to select classifiers, evaluate them, and view comparative analysis results.

## Objective

The goal of this project is to develop an AI-powered system that can predict breast cancer diagnosis with high accuracy. Using the Breast Cancer Wisconsin dataset, multiple classifiers such as K-Nearest Neighbors, Support Vector Classifier, Logistic Regression, and more are evaluated for their performance. The system also includes hyperparameter tuning and comparative analysis of classifier performance.

## Features

- **Breast Cancer Prediction**: Predicts if a tumor is malignant or benign based on features like radius, perimeter, area, symmetry, etc.
- **Multiple Classifiers**: Includes classifiers like KNN, SVC, Decision Trees, Logistic Regression, and Ensemble models.
- **Comparative Analysis**: Ranks classifiers based on their accuracy, showing before and after hyperparameter tuning results.
- **Hyperparameter Tuning**: Fine-tunes model parameters to enhance classifier performance.
- **ROC Curve Visualization**: Displays ROC curves for each classifier for performance evaluation.
- **Web Interface**: Simple, intuitive interface to interact with the model and display results.

## Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- HTML/CSS/JavaScript

## Setup

1. Clone this repository:
   ```bash
   git clone https:https://github.com/DhanushGD/Breast-cancer-prediction.git
   ```
2. Install required libraries:
```bash
   pip install -r requirements.txt
```

3. Download the Breast Cancer Wisconsin dataset and place it in the project directory.
4. Run the Flask application:
```bash
   python app.py
```
5. Open the browser and go to http://127.0.0.1:5000/ to interact with the model.

##Usage
- On the homepage, select the classifiers and overview options you want to evaluate.
- The system will display results like accuracy, confusion matrix, classification report, and ROC curve.
- You can also download the trained ensemble model after evaluating the classifiers.
