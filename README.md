# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries.
2.Load the dataset using pd.read_csv().
3.Display data types, basic statistics, and class distributions.
4.Visualize class distributions with a bar plot.
5.Scale feature columns using MinMaxScaler.
6.Encode target labels with LabelEncoder.
7.Split data into training and testing sets with train_test_split().
8.Train LogisticRegression with specified hyperparameters and evaluate the model using metrics and a confusion matrix plot.
 

## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: mohammad suhael
RegisterNumber:  212224230164
*/


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('food_items.csv')

# Inspect the dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())


x_raw=df.iloc[:,:-1]
y_raw=df.iloc[:,-1:]
x_raw
scaler = MinMaxScaler() 
X = scaler.fit_transform(x_raw)

label_encoder = LabelEncoder()

# Encode the target variable
y = label_encoder.fit_transform(y_raw.values.ravel())  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

penalty = 'l2'

multi_class = 'multinomial'

solver = 'lbfgs'

# Max iteration = 1000
max_iter = 1000

l2_model = LogisticRegression(random_state=123, penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter)

# Fit the model
l2_model.fit(X_train, y_train)
y_pred = l2_model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
```

## Output:

![image](https://github.com/user-attachments/assets/5e692a53-4d58-40b1-88f9-82c44e336dcd)
![image](https://github.com/user-attachments/assets/0e9e6819-8c52-43fb-9369-e559cc96a745)
![image](https://github.com/user-attachments/assets/86f49017-b3c0-4af3-8030-93f32896c5d2)
![image](https://github.com/user-attachments/assets/214704e7-2a8b-4462-9b3f-9cf6eee09518)

![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
