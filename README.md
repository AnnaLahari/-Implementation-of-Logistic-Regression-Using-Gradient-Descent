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

Dataset

![Screenshot 2025-03-29 183116](https://github.com/user-attachments/assets/9513fe3b-7627-4be0-950d-93fb7021a2bc)


![Screenshot 2025-03-29 183336](https://github.com/user-attachments/assets/7a373be2-42d6-46d7-87e6-1a10c390c9ea)



![Screenshot 2025-03-29 183348](https://github.com/user-attachments/assets/218defc0-c787-4bb3-8e61-eccd8da75868)


![Screenshot 2025-03-29 183358](https://github.com/user-attachments/assets/12bd1b85-a4e3-4db1-bcbd-67d38a00d80c)






Accuracy and Predicted value

![Screenshot 2025-03-29 183406](https://github.com/user-attachments/assets/25cc2796-c9c8-4316-b52d-72ccf0ef2367)

Predicted value


![Screenshot 2025-03-29 183425](https://github.com/user-attachments/assets/376b22db-537b-4ec6-b2b4-3113e90ced2d)


![Screenshot 2025-03-29 183433](https://github.com/user-attachments/assets/cf6ff1ea-7537-4af5-ad60-0921364626cc)


![Screenshot 2025-03-29 183445](https://github.com/user-attachments/assets/205e7346-4508-4360-827b-3e50729de7ab)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

