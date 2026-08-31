# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Employee dataset and select employee attributes such as satisfaction level, evaluation, projects, working hours, and other relevant features.
   
2. Split the dataset into training and testing sets using train_test_split().

3. Create and train the Decision Tree Classifier using the training data.

4. Predict employee churn using the test data, calculate the accuracy, and predict whether a new employee is likely to leave or stay. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Muruga S
RegisterNumber:  212225040265
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("Employee (2).csv")

# Display first 5 rows
print(data.head())

# Select input features
X = data[
    [
        'satisfaction_level',
        'last_evaluation',
        'number_project',
        'average_montly_hours',
        'time_spend_company',
        'Work_accident',
        'promotion_last_5years'
    ]
]

# Target variable
y = data['left']

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree Classifier
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict employee churn
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict for a new employee
new_employee = [[
    0.40,   # satisfaction_level
    0.50,   # last_evaluation
    2,      # number_project
    150,    # average_montly_hours
    3,      # time_spend_company
    0,      # Work_accident
    0       # promotion_last_5years
]]

prediction = model.predict(new_employee)

if prediction[0] == 1:
    print("\nPrediction: Employee is likely to leave.")
else:
    print("\nPrediction: Employee is likely to stay.")

```

## Output:
<img width="857" height="755" alt="image" src="https://github.com/user-attachments/assets/c1bb5147-f1c0-41ed-a971-003294c20ba3" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
