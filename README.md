# Breast_cancer
ML_on Supervised Learning: Classifications 
# 🧠 Supervised Learning: Classification

Welcome to the **Classification Algorithms Repository**!  
This project explores **supervised machine learning techniques** with a focus on **classification** methods, their mathematical foundations, and practical applications using the **Breast Cancer Diagnostic dataset**.

---

## 📌 Agenda
- Introduction to Classification
- Applications of Classification
- Types of Classification
- Binary Classification
- Logistic Regression
- Naive Bayes Classifier
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Performance Metrics & Confusion Matrix

---

## 🔍 What is Classification?
Classification is a **supervised learning technique** where models are trained to predict the **class label** of input data.  
It works by finding a **decision boundary** that separates data into distinct categories.

Example: Spam vs. Not Spam, Malignant vs. Benign.

---

## 🌍 Applications of Classification
- **Healthcare** 🏥: Disease diagnosis, patient risk assessment  
- **Finance** 💳: Credit scoring, fraud detection  
- **Marketing** 📈: Customer segmentation, targeted campaigns  
- **Retail** 🛒: Inventory management, product recommendations  
- **Manufacturing** ⚙️: Quality control, fault detection  

---

## 🧩 Types of Classification
- **Binary Classification** → Two outcomes (Yes/No, Spam/Not Spam)  
- **Multi-class Classification** → More than two categories  
- **Multi-label Classification** → Multiple labels per instance  
- **Imbalanced Classification** → Unequal distribution of classes  

---

## ⚡ Binary Classification Algorithms
Popular algorithms include:
- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Decision Trees
- Support Vector Machines (SVM)

---

## 📊 Logistic Regression
Logistic Regression applies the **sigmoid function** to map predictions into probabilities between 0 and 1.


- **Cost Function**: Binary Cross-Entropy (Log Loss)  
- **Optimization**: Gradient Descent  

---

## 🧪 Example: Breast Cancer Dataset
- **Instances**: 569 samples  
- **Features**: 30 numeric attributes (radius, texture, perimeter, area, etc.)  
- **Target**: Diagnosis → Malignant (M) or Benign (B)  

```python
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Breast_cancer_dataset.csv")

# Preprocess
df = df.drop(["id", "Unnamed: 32"], axis=1)
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"].map({"M":1, "B":0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize + Logistic Regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

📈 Performance Metrics
Confusion Matrix → True Positive, False Positive, True Negative, False Negative

Accuracy, Precision, Recall, F1-Score

🌳 Decision Tree
Splits data based on metrics like Gini Index, Entropy

Pruning prevents overfitting

Hyperparameter tuning improves performance

🧭 Support Vector Machine (SVM)
Finds the optimal hyperplane separating classes

Works well in high-dimensional spaces

Kernel trick allows handling non-linear boundaries
## 🚀 How to Use
Clone the repository: git clone https://github.com/your-username/classification-algorithms.git

## Install dependencies: pip install -r requirements.txt
