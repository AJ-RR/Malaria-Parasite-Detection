import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
import sklearn
from sklearn import metrics
import pickle
from sklearn.externals import joblib
""" Loading the dataset """
data1 = pd.read_csv("blob_features.csv")
data2 = pd.read_csv("contour_features.csv")
data3 = pd.read_csv("LUCID_features.csv")
data4 = pd.read_csv("num_of_contours.csv")
data1 = data1.iloc[:,1:]
data2 = data2.iloc[:,1:]
data3 = data3.iloc[:,1:]
data4 = data4.iloc[:,1:]
dfT = pd.DataFrame(data = np.concatenate([np.zeros(int(data1.shape[0]/2.)), np.ones(int(data1.shape[0]/2.))]))
data = pd.concat([dfT, data1, data2, data3, data4], axis = 1)

""" Building the train_set and the test_set """

# seperating the training and validation test
X = data.iloc[:,1:]
y = data.iloc[:,0]

i = 0
acc = 0
while i < 10:

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print("------Scaled the features--------")


    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    model1 = RandomForestClassifier(n_estimators=75,max_depth=10)
    model1.fit(X_train,y_train)
    if i==0 :
        pickle.dump(model1,open('traditional_model.sav','wb'))
        joblib.dump(model1, 'traditional_model.pkl')
    y_pred1 = model1.predict(X_test)

    from sklearn.metrics import accuracy_score
    x = accuracy_score(y_pred1, y_test)
    # y_ = accuracy_score(y_pred2, y_test)
    # z = accuracy_score(y_pred3, y_test)
    acc = acc + x

    print(x)

    i = i + 1

print("Accuracy => ", acc*10, "%")
