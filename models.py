import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
## models tried
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.linear_model import Perceptron

## Helper functions ##

def getData(needScaled):
    scaler = StandardScaler()
    
    
    
    if (needScaled):
        X_train = np.load('mmscaled_db_train.npy')
        X_test = np.load('mmscaled_db_test.npy')
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = X_train[:,:-1]
        X_test = X_test[:,:-1]
        return X_train, X_test
    else:
        X_train = np.load('mmscaled_db_train.npy')
        X_test = np.load('mmscaled_db_test.npy')
        X_train = X_train[:,:-1]
        X_test = X_test[:,:-1]

        return X_train, X_test

def printResults(clf_final, X_test, y_test):
    y_pred = clf_final.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))



np_loaded = np.load('db.npy')
y = np_loaded[:, -1]
X = np_loaded[:, :-1]


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
X_train_unscaled,X_test_unscaled,y_train,y_test = [],[],[],[]
for train_index, test_index in sss.split(X, y):
    X_train_unscaled, X_test_unscaled = X[train_index], X[test_index]
    y_train, y_test= y[train_index], y[test_index]

## Decision tree 

file = open("Errors.txt", 'w')

try:
    print("")
    print("Decision Tree")
    print("")

    needScaled = False
    X_train, X_test = getData(needScaled)

    clf = DecisionTreeClassifier(random_state=0)
    depth = [i for i in range(5, 16)]
    depth.append(None)
    min_samples_split = [i for i in range(2, 40, 3)]
    min_samples_leaf = [i for i in range(1,10)]
    param_dist = {
        "criterion" : ["gini", "entropy"],
        "min_samples_split" : min_samples_split,
        "min_samples_leaf" : min_samples_leaf,
        "min_impurity_decrease" : [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        "max_depth" : depth
    }
    grid = GridSearchCV(clf, param_grid=param_dist, n_jobs=25, cv = 10)
    grid.fit(X_train, y_train)
    clf_final = grid.best_estimator_
    # print(clf_final)
    f = open('DecisionTreePickle', 'wb') 
    pickle.dump(clf_final, f)                      
    dbfile.close()
except:
    file.write('Decision Tree\n')



## Random Forest ##

try:
    print("")
    print("Random Forest")
    print("")
    needScaled = False
    X_train, X_test = getData(needScaled)

    clf = RandomForestClassifier(random_state=0)
    depth = [i for i in range(5, 16)]
    depth.append(None)
    n_estimators = [i for i in range(50, 200, 5)]
    param_dist = {
        "criterion" : ["gini", "entropy"],
        "n_estimators": n_estimators,
        "bootstrap": [True, False],
        "oob_score": [True, False],
        "max_depth" : depth
    }
    grid = GridSearchCV(clf, param_grid=param_dist, n_jobs=25, cv = 10)
    grid.fit(X_train, y_train)
    clf_final = grid.best_estimator_
    print(clf_final)
    f = open('RandomForestPickle', 'wb') 
    pickle.dump(clf_final, f)                      
    dbfile.close()
except:
    file.write("Random Forest\n")


try:
    ## Logistic Regression ##
    print("")
    print("Logistic Regression")
    print("")
    needScaled = True
    X_train, X_test = getData(needScaled)

    clf = LogisticRegression(random_state=0)
    penalty = ['l1', 'l2', 'elasticnet']
    max_iter = [100, 500, 1000, 5000, 10000]
    C = [i/10 for i in range(1,20)]
    param_dist = {
        "penalty" : penalty,
        "max_iter" : max_iter,
        "C":C,
        "fit_intercept":[True, False],
        
        
    }
    grid = GridSearchCV(clf, param_grid=param_dist, n_jobs=25, cv = 10)
    grid.fit(X_train, y_train)
    clf_final = grid.best_estimator_
    print(clf_final)
    f = open('LogisticRegressionPickle', 'wb') 
    pickle.dump(clf_final, f)                      
    dbfile.close()
except:
    file.write("Logistic Regression\n")


try:
    ## SVM ##
    print("")
    print("SVM")
    print("")
    needScaled = True
    X_train, X_test = getData(needScaled)

    clf = SVC(random_state=0)
    max_iter = [100, 500, 1000, 5000, 10000]

    C = [i/10 for i in range(1,20)]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    gammas = ['scale', 'auto', 0.05, 0.1, 0.15, 0.2]
    param_dist = {
        "kernel" : kernels,
        "gamma" : gammas,
        "C" : C,
        "shrinking" : [True, False],
        "max_iter" : max_iter
    }
    grid = GridSearchCV(clf, param_grid=param_dist, n_jobs=25, cv = 10)
    grid.fit(X_train, y_train)
    clf_final = grid.best_estimator_
    print(clf_final)
    f = open('SVMPickle', 'wb') 
    pickle.dump(clf_final, f)                      
    dbfile.close()
except:
    file.write('SVM\n')

try:
    ## GNB ##
    print("")
    print("Gaussian Naive Bayes")
    print("")
    needScaled = False

    X_train, X_test = getData(needScaled)

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    # X_test = scaler.transform(X_test)
    print(clf.score(X_test, y_test))

    f = open('GNBPickleScaled', 'wb') 
    pickle.dump(clf, f)                      
    f.close()
except:
    file.write('GNBScaled\n')


try:
    ## GNB ##
    print("")
    print("Gaussian Naive Bayes")
    print("")
    needScaled = True

    X_train, X_test = getData(needScaled)

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    # X_test = scaler.transform(X_test)
    print(clf.score(X_test, y_test))


    f = open('GNBPickleUnscaled', 'wb') 
    pickle.dump(clf, f)                      
    f.close()
except:
    file.write('GNBunscaled\n')

try:
    ## KNN ##
    print("")
    print("K Nearest Neighbors")
    print("")
    needScaled = True

    X_train, X_test = getData(needScaled)
    clf = KNeighborsClassifier()
    n_neighbors = [2,4,6,8,10]
    algorithm = ['auto']
    weights = ['uniform', 'distance']
    leaf_size = [i for i in range(10, 51, 10)]

    param_dist = {
        "n_neighbors" : n_neighbors,
        "algorithm" : algorithm,
        "weights" : weights,
        "leaf_size" : leaf_size
    }
    grid = GridSearchCV(clf, param_grid=param_dist, n_jobs=25, cv = 10)
    grid.fit(X_train, y_train)
    clf_final = grid.best_estimator_
    print(clf_final)
    f = open('KNNPickle', 'wb') 
    pickle.dump(clf_final, f)                      
    f.close()
except:
    file.write('KNN\n')


try:
    ## Perceptron ##
    print("")
    print("Perceptron")
    print("")
    needScaled = False
    X_train, X_test = getData(needScaled)
    penalty = ['l2','l1','elasticnet']
    alpha = [0.0001*(10**i) for i in range(0,5)]
    max_iter = [10000]
    param_dist = {
        "penalty" : penalty,
        "alpha" : alpha,
        "max_iter" : max_iter
    }

    clf = Perceptron()
    grid = GridSearchCV(clf, param_grid=param_dist, n_jobs=25, cv = 10)
    grid.fit(X_train, y_train)
    clf_final = grid.best_estimator_
    print(clf_final)
    f = open('PerceptronPickle', 'wb') 
    pickle.dump(clf_final, f)                      
    f.close()

    print("")
    printResults(clf_final, X_test, y_test)

except:
    file.write('Perceptron\n')