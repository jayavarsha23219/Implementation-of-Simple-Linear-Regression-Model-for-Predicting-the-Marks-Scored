# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students
2. Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored
3. Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis
4. Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b
for each data point calculate the difference between the actual and predicted marks
5. Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error
6. Once the model parameters are optimized, use the final equation to predict marks for any new input data

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JAYAVARSHA T
RegisterNumber: 212223040075

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv('student_scores.csv')
df.head()

df.tail()

X = df.iloc[:, :-1].values
X

Y = df.iloc[:, 1].values
Y

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
Y_pred

Y_test

plt.scatter(X_train, Y_train, color="blue")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, regressor.predict(X_test), color="red")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse = mean_squared_error(Y_test, Y_pred)
print('MSE = ', mse)

mae = mean_absolute_error(Y_test, Y_pred)
print('MAE = ', mae)

rmse = np.sqrt(mse)
print("RMSE = ", rmse)
*/ 
```

## Output:
## HEAD:
<img width="232" height="253" alt="image" src="https://github.com/user-attachments/assets/233f0215-1c01-42a3-8650-1880eb847e0b" />

## TAIL:
<img width="246" height="248" alt="image" src="https://github.com/user-attachments/assets/3de05c2e-3d77-47de-aab5-496154806af2" />


## X:
<img width="203" height="565" alt="image" src="https://github.com/user-attachments/assets/43d4d705-fc22-4e04-a24d-927682da79a7" />


## Y:
<img width="800" height="62" alt="image" src="https://github.com/user-attachments/assets/65ced746-a74e-4789-845e-6fcc5aed0710" />


## Y_PRED:
<img width="775" height="80" alt="image" src="https://github.com/user-attachments/assets/e4365039-4107-4fed-840a-61b427cd1952" />


## Y_TEST:
<img width="632" height="37" alt="image" src="https://github.com/user-attachments/assets/01624602-8f0e-42d3-9c32-01dc2ef97668" />


## TRAINING SET:
<img width="782" height="588" alt="image" src="https://github.com/user-attachments/assets/0435f9fe-f44e-43a1-b9d8-5479430f7884" />


## TEST SET:
<img width="772" height="605" alt="image" src="https://github.com/user-attachments/assets/b41c26a0-4707-4526-a782-6b12d00095e3" />


## VALUES:
<img width="326" height="85" alt="image" src="https://github.com/user-attachments/assets/58f4273f-292f-4f34-ab18-77619aec79ab" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
