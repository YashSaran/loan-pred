import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('train_loanPrediction.csv')
#for gender
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
d={'Male':1,'Female':0}
data['Gender']=data['Gender'].apply(lambda X:d[X])
#for married
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
e={'Yes':1,'No':0}
data['Married']=data['Married'].apply(lambda X:e[X])
#for dependents
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
f={'0':0,'1':1,'2':2,'3+':3}
data['Dependents']=data['Dependents'].apply(lambda X:f[X])
#for education
g={'Graduate':1,'Not Graduate':0}
data['Education']=data['Education'].apply(lambda X:g[X])
#self employed
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
h={'Yes':1,'No':0}
data['Self_Employed']=data['Self_Employed'].apply(lambda X:h[X])
#for loan amount
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
#for loan amount term
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median())
#for credit history
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].median())
#for property area
i={'Rural':0,'Semiurban':1,'Urban':2}
data['Property_Area']=data['Property_Area'].apply(lambda X:i[X])
#for loan status
j={'N':0,'Y':1}
data['Loan_Status']=data['Loan_Status'].apply(lambda X:j[X])
X=data.iloc[:,1:-1].values
y=data.iloc[:,12].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 3/614, random_state =140)
#logistic reg
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cn=confusion_matrix(y_test,y_pred)
#random forest


data.describe()






print('Enter your gender:')
x1 = input()
if x1=="male":
    x1=1
else:
    x1=0

print('Enter your maritial status:')
x2 = input()
if x2=="yes":
    x2=1
else:
    x2=0

print('Enter your dependents:')
x3 = input()
x3=int(x3)
if x3>=3:
    x3=3

print('Enter your highest education:')
x4 = input()
if x4=="graduate":
    x4=1
else:
    x4=0

print('are you self employed:')
x5 = input()
if x5=="yes":
    x5=1
else:
    x5=0

print('enter your income:')
x6 = input()
x6=int(x6)

print('enter your co applicants income:')
x7 = input()
x7=int(x7)

print('enter your loan amount:')
x8 = input()
x8=int(x8)

print('enter your loan amount term:')
x9 = input()
x9=int(x9)

print('enter credit history:')
x10 = input()
if x10=="yes":
    x10=1
else:
    x10=0
    
print('enter property location:')
x11 = input()
if x11=="urban":
    x11=2
if x11=="semi urban":
    x11=1
if x11=="rural":
    x11=0

df=[[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11]]
print (classifier.predict(df))
