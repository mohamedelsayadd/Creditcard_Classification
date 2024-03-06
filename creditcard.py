# 1 - import packages : 

import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns


# 2 - reed the file : 

df = pd.read_csv(r"C:\Users\moham\Desktop\MY AI\ML Projects\Creditcard Classification\creditcard.csv")
df = pd.DataFrame(df)


# 3 - get some info about the data :

# print(df.shape)
# print(df.sample(5))
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.describe())
# print(df['Class'].unique())


# 3 - Data Preprocessing :

from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()

df[['Amount']]= scaler.fit_transform(df[['Amount']])


# 4 - split the data :

X = df.drop(['Class'],axis=1)
Y =df['Class']


# 5 -  show and plot the ratio of Y values 

# print(Y.value_counts())
fig1, aX_res1 = plt.subplots()
aX_res1.pie(Y.value_counts(), autopct='%.2f')
# plt.show()



# 6 - solve the imbalance problem : 

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(sampling_strategy=1)
X_res, Y_res = ros.fit_resample(X, Y)


# 7 - show and plot the data after sovle imbalance problem : 

# print(Y_res.shape)
fig1, aX_res1 = plt.subplots()
aX_res1.pie(Y_res.value_counts(), autopct='%.2f')
# plt.show()


# 8 - train the data : 

from sklearn.model_selection import train_test_split
X_res_train,X_res_test,Y_res_train,Y_res_test=train_test_split(X_res,Y_res,test_size=0.20 , random_state=42)


# 9 - use Decision Tree model  to predict :

from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier()
DT_model.fit(X_res_train,Y_res_train)
Y_res_pred = DT_model.predict(X_res_test)


# 10 - evaluate the model :

from sklearn.metrics import confusion_matrix
from sklearn import metrics

confusion_matrix = confusion_matrix(Y_res_test, Y_res_pred)
# print(confusion_matrix)


sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
# plt.show()

accuracy  = metrics.accuracy_score(Y_res_test,Y_res_pred)
# print(accuracy)


# 11 - save the model :
import pickle
pickle.dump(DT_model,open('creditcard.pkl','wb'))
model = pickle.load(open( 'creditcard.pkl', 'rb' ))