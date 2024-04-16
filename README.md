# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Masina Sree karsh
RegisterNumber:  212223100033
*/
```
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:

## HEAD() AND INFO():
![image](https://github.com/sreekarsh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139841918/824701fe-fb3c-4bba-8df4-89ac7dd7a0ea)


## NULL & COUNT:
![image](https://github.com/sreekarsh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139841918/079fe853-aeae-41b4-8780-9e26fa49411b)

![image](https://github.com/sreekarsh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139841918/086fefd7-d413-4151-a8f1-27e1e7098c10)


## ACCURACY SCORE:
![image](https://github.com/sreekarsh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139841918/2a814150-d107-430e-916b-f8159115fc37)


## DECISION TREE CLASSIFIER MODEL:
![image](https://github.com/sreekarsh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139841918/819dce32-35a7-4c6c-a113-025b7bcec83a)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
