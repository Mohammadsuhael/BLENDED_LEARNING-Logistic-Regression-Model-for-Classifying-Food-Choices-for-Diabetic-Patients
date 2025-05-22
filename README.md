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


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Food_Items.csv")

# Inspect the dataset
print("Dataset Overview:")
print(df.head())
print("\nDATASET INFO")
print(df.info())


df['class'] = df['class'].str.strip("'")
df['Diabetic'] = df['class'].apply(lambda x: 1 if x == 'In Moderation' else 0)

# Define features and target
X = df.drop(columns=['class', 'Diabetic'])  # Features
y = df['Diabetic']                          # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)


# Evaluate the model
print("Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=['Non-Diabetic', 'Diabetic'],yticklabels=['Non-Diabetic', 'Diabetic'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/536111de-7bf2-4912-88e6-bdb0f821635d)
![image](https://github.com/user-attachments/assets/9b438f0f-628e-4194-8482-94aee57cf027)
![image](https://github.com/user-attachments/assets/b4b608a0-b0b2-4fe1-976c-a31b644e6dc9)
![image](https://github.com/user-attachments/assets/37bf283d-42f1-4f92-a003-5a555f4f73a8)





## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
