# Customer Churn Prediction

## 📌 Project Overview

Customer churn prediction is a machine learning project that predicts whether a customer is likely to **leave (churn)** a service or continue using it.

The goal of this project is to analyze customer-related data, identify patterns associated with churn, handle class imbalance, train multiple machine learning models, and determine which model performs best.

This project implements and compares **Decision Tree, Random Forest, and XGBoost** classification models.

---

## 🎯 Objectives

* Predict whether a customer will churn.
* Perform data preprocessing and exploratory data analysis.
* Convert categorical features into numerical form using **Label Encoding**.
* Handle class imbalance using **SMOTE**.
* Train multiple classification algorithms.
* Compare model performance.
* Evaluate models using accuracy, confusion matrix, and classification report.
* Save the trained model using **Pickle** for future use.

---

## 🛠️ Technologies & Libraries Used

* **Python**
* **NumPy** – Numerical computations
* **Pandas** – Data manipulation and analysis
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical data visualization
* **Scikit-learn** – Machine learning algorithms and evaluation
* **Imbalanced-learn** – SMOTE for handling imbalanced data
* **XGBoost** – Gradient boosting classification
* **Pickle** – Model serialization

### Main Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import pickle
```

---

## 📂 Project Workflow

The project follows the following machine learning pipeline:

**Data Collection → Data Preprocessing → Exploratory Data Analysis → Label Encoding → Train-Test Split → SMOTE → Model Training → Cross Validation → Model Evaluation → Model Saving**

---

## 🔍 Project Steps

### 1. Data Loading

The customer churn dataset is loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv("your_dataset.csv")
```

### 2. Data Preprocessing

The dataset is analyzed and prepared for machine learning.

The preprocessing includes:

* Checking missing values
* Removing unnecessary columns
* Identifying categorical and numerical features
* Encoding categorical variables
* Preparing the target variable

### 3. Exploratory Data Analysis

Matplotlib and Seaborn are used to understand the dataset and identify patterns related to customer churn.

Examples of analysis include:

* Churn distribution
* Feature relationships
* Correlation analysis
* Visualization of customer characteristics

### 4. Label Encoding

Categorical variables are converted into numerical values using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
```

### 5. Train-Test Split

The dataset is divided into training and testing sets using `train_test_split`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 6. Handling Class Imbalance

Customer churn datasets can contain an unequal number of churned and non-churned customers.

To address this problem, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to the training data.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train, y_train
)
```

SMOTE generates synthetic samples for the minority class instead of simply duplicating existing observations.

> **Note:** SMOTE should be applied only to the training data to avoid data leakage.

---

## 🤖 Machine Learning Models

Three classification algorithms are used in this project:

### 1. Decision Tree Classifier

A Decision Tree creates a tree-like structure of decision rules to classify customers into churn and non-churn categories.

```python
DecisionTreeClassifier()
```

### 2. Random Forest Classifier

Random Forest combines multiple decision trees to improve prediction performance and reduce overfitting.

```python
RandomForestClassifier()
```

### 3. XGBoost Classifier

XGBoost is a powerful gradient boosting algorithm that builds models sequentially to improve classification performance.

```python
XGBClassifier()
```

---

## 📊 Model Evaluation

The models are evaluated using:

### Accuracy Score

Measures the percentage of correctly classified customers.

```python
accuracy_score(y_test, y_pred)
```

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

```python
confusion_matrix(y_test, y_pred)
```

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

```python
classification_report(y_test, y_pred)
```

### Cross Validation

Cross-validation is used to evaluate model performance across multiple train-validation splits.

```python
cross_val_score(model, X, y, cv=5)
```

---

## 📈 Results

The performance of the models can be compared using their accuracy and classification metrics.

| Model         | Cross-Validation Accuracy |  Accuracy |  Precision |     Recall |   F1-Score |
| ------------- | ------------------------: | --------: | ---------: | ---------: | ---------: |
| Decision Tree |             78            |     72    |     47     |     55     |     51     |
| Random Forest |             84            |     77    |     57     |     58     |     57     |
| XGBoost       |             83            |     78    |     58     |     60     |     59     |

**Best Model:** Random Forest

---

## 💾 Model Saving

The trained model is saved using Python's `pickle` module so that it can be reused without retraining.

Example:

```python
with open("churn_model.pkl", "wb") as file:
    pickle.dump(model, file)
```

The saved model can later be loaded using:

```python
with open("churn_model.pkl", "rb") as file:
    model = pickle.load(file)
```

---

## 📁 Project Structure

```text
Churn-Prediction/
│
├── data/
│   └── churn_dataset.csv
│
├── notebooks/
│   └── churn_prediction.ipynb
│
├── models/
│   └── churn_model.pkl
│
├── README.md
│
└── requirements.txt
```

*The structure can be modified according to the actual files in your repository.*

---

## ⚙️ Clone the Repository

```bash
git clone https://github.com/Gunjan-Kumari/Customer-churn-prediction.git
```

---


## 🔮 Future Improvements

Some possible improvements for this project are:

* Build a web application using **Streamlit or Flask**.
* Add a user interface for real-time churn prediction.
* Perform hyperparameter tuning.
* Add additional machine learning algorithms.
* Improve feature engineering.
* Use ROC-AUC and Precision-Recall curves for evaluation.
* Deploy the model to a cloud platform.
* Add automated model retraining.

---

## 📚 Key Concepts Demonstrated

This project demonstrates practical knowledge of:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Encoding
* Train-Test Split
* Handling Imbalanced Data
* SMOTE
* Decision Trees
* Random Forest
* XGBoost
* Cross Validation
* Model Evaluation
* Confusion Matrix
* Classification Metrics
* Model Serialization with Pickle

---

## 👩‍💻 Author

**Gunjan Kumari**

B.Tech CSE Student | Machine Learning Enthusiast

---

## ⭐ If you found this project useful

Feel free to ⭐ star this repository and explore the project!
