# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

1.Import the required libraries.
2.Load the dataset and print the values. 
3.Define X and Y array and display the value.
4.Find the value for cost and gradient. 
5.Plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: A.LAHARI
RegisterNumber:  212223230111
*/
```

```
import pandas as pd
import numpy as np
dataset=pd.read_csv("Placement_Data.csv")
dataset
```

```
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```

```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
```

```
theta = np.random.randn(X.shape[1])
y =Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations = 1000)
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)
```

```
print(y_pred)
```

```
print(Y)
```

```
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```

```
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```


## Output:

## Dataset

![image](https://github.com/user-attachments/assets/ed744821-57c3-4b0d-9ad6-8c7e2b8a8b98)

## dtypes

![image](https://github.com/user-attachments/assets/6de79f0c-cad6-4057-a82d-65c4a0581bad)

## dataset

![image](https://github.com/user-attachments/assets/92b2936b-b76f-4b0e-a85c-408a1befcd81)

## y array

![image](https://github.com/user-attachments/assets/2766e7b8-8548-4158-9782-32979c564ded)

## Accuracy

![image](https://github.com/user-attachments/assets/5837f697-e01e-4f45-8009-3828296a69ce)

## y_pred

![image](https://github.com/user-attachments/assets/d9971619-9c9c-4492-a134-d16051d08d13)

## y

![image](https://github.com/user-attachments/assets/180b42ab-696a-4581-82b4-18b06a175f0c)

## y_prednew

![image](https://github.com/user-attachments/assets/17ef50d8-4817-47aa-9738-20c1849cfb1e)

## y_prednew

![image](https://github.com/user-attachments/assets/f6533576-d624-4c51-82f9-a8b8bcd2ac85)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

