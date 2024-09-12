# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ABISHEK PV
RegisterNumber: 212222230003

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv("Placement_Data.csv")
print("Placement Data:")
print(data)
if 'salary' in data.columns:
    print("\nSalary Data:")
    print(data['salary'])
else:
    print("\n'Salary' column not found in DataFrame")
data1 = data.drop(["salary"], axis=1, errors='ignore')
print("\nMissing Values Check:")
print(data1.isnull().sum())
print("\nDuplicate Rows Check:")
print(data1.duplicated().sum())

print("\nCleaned Data:")
print(data1)
le = LabelEncoder()

categorical_columns = ['workex', 'status', 'hsc_s']  # List of categorical columns to encode
for column in categorical_columns:
    if column in data1.columns:
        data1[column] = le.fit_transform(data1[column])
    else:
        print(f"'{column}' column not found in DataFrame")

x = data1.drop('status', axis=1, errors='ignore')  # Features
y = data1['status']  # Target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_report1 = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_report1)

print("\nY Prediction Array:")
print(y_pred)

```

## Output:
Placement Data :

![image](https://github.com/user-attachments/assets/3ad03470-567f-4174-8a3f-d72d0ed38fc9)

Salary Data:

![image](https://github.com/user-attachments/assets/76f8f48a-d375-4232-82f4-7f8cf196ee4d)

Missing Values Check and duplicate row data:

![image](https://github.com/user-attachments/assets/a4507747-61e0-4769-8947-abbde1d05d51)

Cleaned Data:

![image](https://github.com/user-attachments/assets/1a0a3ffc-3596-4206-a4f2-457334a94508)

Y-Prediction Array :

![image](https://github.com/user-attachments/assets/703c614a-cc23-45ea-91bf-2791fb025b30)

Acuuracy value:

![image](https://github.com/user-attachments/assets/fd6e3b41-ed00-4f4f-a844-ef4f808dd89e)

Confusion array:

![image](https://github.com/user-attachments/assets/73ed5996-03f7-48b2-aa92-4cc64d7ee4bd)

Classification report:


![image](https://github.com/user-attachments/assets/6128b361-3a77-4b96-b478-00d83f492a12)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
