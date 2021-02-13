import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# #Loading data
df = pd.read_csv('data.csv')
# #Preprocessing Data

# #Drop Unnamed due to missing values
df = df.dropna(axis=1)
df = df.drop(['id'], axis=1)
# #Encode data values

labelencoder_Y = LabelEncoder()
df.iloc[:, 0] = labelencoder_Y.fit_transform(df.iloc[:,0])

# #Split data into independent (X) and dependant (Y) data sets
X = df.iloc[:, 1:30].values
Y = df.iloc[:, 0].values

# #Split data into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# #Scale data (Feature Scaling)
# #Feature Scaling is done during the pre-processing phase of Machine Learning
# #All I'm doing here is normalizing the independent variables

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# #Creating the machine learning models


def models(X_train, Y_train):

    # #Logistic Regression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, Y_train)
  
    # #Decision Tree
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, Y_train)

    # #Random Forest
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, Y_train)

    # #Print models accuracy on training data
    print('[0]Logistic Regression Training Accuracy: ', log.score(X_train, Y_train))
    print('[1]Decision Tree Training Accuracy: ', tree.score(X_train, Y_train))
    print('[2]Random Forest Training Accuracy: ', forest.score(X_train, Y_train))
    return log, tree, forest


# #Getting all the models
model = models(X_train, Y_train)


for i in range(len(model)):
    cm = confusion_matrix(Y_test, model[i].predict(X_test))
    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]
    print('Testing Accuracy = ', (TP+TN)/(TN + TP + FN + FP))


for i in range(len(model)):
    print("CLASSIFICATION REPORT:")
    print(classification_report(Y_test, model[i].predict(X_test)))
    print("ACCURACY REPORT:")
    print(accuracy_score(Y_test, model[i].predict(X_test)))
    print()
