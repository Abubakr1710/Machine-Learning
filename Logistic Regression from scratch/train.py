import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf = LogisticRegression(lr=0.01) # I changed lr from 0.0001 to 0.01 and accuracy increased from 0.89 to 0.92
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

acc = accuracy(y_pred, y_test)
print("Accuracy:", np.round(acc, 2))