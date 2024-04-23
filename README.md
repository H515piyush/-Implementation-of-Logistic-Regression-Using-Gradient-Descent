# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:piyush kumar 
RegisterNumber:212223220075 
*/
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 
 dataset =pd.read_csv("C:/Users/admin/Downloads/Placement_Data.csv")
 dataset

  dataset =dataset.drop('sl_no',axis=1)

  dataset =dataset.drop('salary',axis=1)

   dataset['gender']=dataset['gender'].astype('category')
 dataset['ssc_b']=dataset['ssc_b'].astype('category')
 dataset['hsc_b']=dataset['hsc_b'].astype('category')
 dataset['degree_t']=dataset['degree_t'].astype('category')
 dataset['workex']=dataset['workex'].astype('category')
 dataset['specialisation']=dataset['specialisation'].astype('category')
 dataset['status']=dataset['status'].astype('category')
 dataset['hsc_s']=dataset['hsc_s'].astype('category')
 dataset.dtypes

 dataset['gender']=dataset['gender'].cat.codes
 dataset['ssc_b']=dataset['ssc_b'].cat.codes
 dataset['hsc_b']=dataset['hsc_b'].cat.codes
 dataset['degree_t']=dataset['degree_t'].cat.codes
 dataset['workex']=dataset['workex'].cat.codes
 dataset['specialisation']=dataset['specialisation'].cat.codes
 dataset['status']=dataset['status'].cat.codes
 dataset['hsc_s']=dataset['hsc_s'].cat.codes
 dataset

 X=dataset.iloc[:,:-1].values
 y=dataset.iloc[:,-1].values
 y

 theta=np.random.randn(X.shape[1])
 Y=y

 def sigmoid(z):
 return 1/(1+np.exp(-z))

 def loss(theta,X,Y):
 h=sigmoid(X.dot(theta))
 return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

 def gradient_descent(theta,X,y,alpha,num_iterations):
 m=len(y)
 for i in range(num_iterations):
 h= sigmoid(X.dot(theta))
 gradient= X.T.dot(h-y)/m
 theta -= alpha*gradient
 return theta 
 
 In [53]:
 In [54]:
 In [55]:
 theta=np.random.randn(X.shape[1])
 Y=y
 def sigmoid(z):
 return 1/(1+np.exp(-z))
 def loss(theta,X,Y):
 h=sigmoid(X.dot(theta))
 return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
 def gradient_descent(theta,X,y,alpha,num_iterations):
 m=len(y)
 for i in range(num_iterations):
 h= sigmoid(X.dot(theta))
 gradient= X.T.dot(h-y)/m
 theta -= alpha*gradient
 return theta    
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
 h = sigmoid(X.dot(theta))
 y_pred=np.where(h>=0.5,1,0)
 return y_pred
 y_pred = predict(theta,X)

 accuracy = np.mean(y_pred.flatten()==y)
 print('Accuracy:',accuracy)

 print(y_pred)

 print(y_pred)

 Xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
 y_prednew = predict(theta,Xnew)
 print(y_prednew)

 Xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
 y_prednew = predict(theta, Xnew)
 print(y_prednew)
```

## Output:
 dataset:
![Screenshot 2024-04-23 093211](https://github.com/H515piyush/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147472999/37c119b6-ecf8-494b-a88e-eac35dc66ddd)

dataset.dtypes:
![Screenshot 2024-04-23 093220](https://github.com/H515piyush/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147472999/26e0e693-d150-4388-8a6d-468b6f8cbe8e)

dataset:

![Screenshot 2024-04-23 093230](https://github.com/H515piyush/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147472999/1ead096f-3416-4e36-b304-4b11c4774b4b)

Y:

![Screenshot 2024-04-23 093239](https://github.com/H515piyush/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147472999/a8366f2a-bff8-4892-b526-719f56e7fa84)

Y_PRED:

![Screenshot 2024-04-23 093249](https://github.com/H515piyush/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147472999/1801ed74-7e64-4062-9d91-fe962a303763)



Y:

![Screenshot 2024-04-23 093256](https://github.com/H515piyush/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147472999/49195f3b-eff6-485c-a061-dbf4d2bb7e98)

Y_PREDNEW:

![Screenshot 2024-04-23 093301](https://github.com/H515piyush/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147472999/4d88025a-5d96-4629-90f0-7b1f5847a385)

y_prednew:

![Screenshot 2024-04-23 093301](https://github.com/H515piyush/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147472999/0a656b5a-2183-4b24-85e4-ffff3348f04d)




   
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

