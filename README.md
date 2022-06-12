# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Gather information and presence of null in the dataset
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy score of the model.
6. Check the trained model.
7. 
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: R.Rajalakshmi
RegisterNumber: 212219040116
*/

import pandas as pd
df=pd.read_csv("Employee.csv")
df.head()
df.info()
df.isnull().sum()
df["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()
x=df[["satisfaction_level","last_evaluation","number_project",
"average_montly_hours","time_spend_company","Work_accident",
"promotion_last_5years","salary"]]
x.head()
y=df["left"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred)
acc
dt.predict([[.5,.8,9,260,6,0,1,2]])
```
## Output:
### Initial Dataset:
![Screenshot (138)](https://user-images.githubusercontent.com/87656716/173228567-d1e54bed-acb5-4576-8293-23fb1dce1283.png)
### Dataset Information:
![Screenshot (140)](https://user-images.githubusercontent.com/87656716/173228661-1a4560ee-2da9-476d-85a5-b4a30113f9dc.png)
### Null dataset:
![Screenshot (142)](https://user-images.githubusercontent.com/87656716/173228801-7459be42-e652-4121-938f-3ce2dc21846d.png)
### Value counts in left column:
![Screenshot (144)](https://user-images.githubusercontent.com/87656716/173228992-bd1a97d6-772e-4899-9525-326182997fcd.png)
### Encoded dataset:
![Screenshot (146)](https://user-images.githubusercontent.com/87656716/173229074-2bcf8051-f4c0-4b6d-b73d-b7e2c8154521.png)
### x set:
![Screenshot (148)](https://user-images.githubusercontent.com/87656716/173229163-17cb84cf-8d4e-44b4-bba7-1a6543c1f75c.png)
### y values:
![Screenshot (150)](https://user-images.githubusercontent.com/87656716/173229263-d6bb186e-fee2-4eb0-92fb-3a7d30247023.png)
### Accuracy Score:
![Screenshot (152)](https://user-images.githubusercontent.com/87656716/173229369-1e5d4308-1c45-4539-af9d-f33329869e71.png)
### Prediction:
![Screenshot (154)](https://user-images.githubusercontent.com/87656716/173229504-1740fc33-6c23-4ed1-90cb-8d2ed07546b5.png)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
