from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
import pickle
import numpy as np

# trainX = np.load('scaled_db_train.npy')
# trainY = trainX[:, -1]
# trainX = trainX[:, :-1]

# testX = np.load('scaled_db_test.npy')
# testY = testX[:, -1]
# testX = testX[:, :-1]


clf = SVC(random_state=0)
max_iter = [100, 500, 1000, 5000, 10000]
# max_iter = [5000]
C = [i/10 for i in range(1,20)]  # 19
# C = [0.1]
kernels = ['linear', 'poly', 'rbf', 'sigmoid'] # 4 
# kernels = [ 'sigmoid']
gammas = ['scale', 'auto', 0.05, 0.1, 0.15, 0.2] # 
# gammas = ['auto']

max_acc = -1
clf_final = -1
counter = 0
for c in C:
  for kernel_ in kernels:
    for gamma_ in gammas:
      for shrinking_ in [True, False]:
        if counter == 220:
          print(c, kernel_, gamma_, shrinking_)
        counter += 1
        # try:
        #   svm = SVC(C = c, kernel = kernel_, gamma = gamma_, shrinking=shrinking_)
        #   svm.fit(trainX, trainY)
        #   acc  = svm.score(testX, testY)
        #   if acc >= max_acc:
        #     clf_final = svm
        #   counter += 1
        #   print('Models done', counter, 'Acc', acc)
        # except Exception as e:
        #   print('Error', e)

# preds = clf_final.predict(testX)
# precision_recall_fscore_support(testY, preds, average='macro')
# print(counter)
# f = open('SVM_pickle', 'wb')
# pickle.dump(clf_final, f)
# f.close()