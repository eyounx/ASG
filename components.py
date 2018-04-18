'''
Precision and Recall functions

Author:
    Wei-Yang Qu  quwy@lamda.nju.edu.cn
Date:
    2018.04.15
'''

import numpy as np 
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Deal test label for the binary classification problems
def dealTesty(test_y,plus_set):
    test_y = np.array(test_y)
    res = []
    for i in range(test_y.shape[0]):
        if test_y[i] in plus_set:
            res.append(test_y[i])
        else:
            res.append(-1)
    return np.array(res)

# Get Precision and Recall for seen class
def getPrecisionRecall(label,pred):
    P = precision_score(label, pred, average='macro')
    R = recall_score(label, pred, average='macro')
    return (P,R)

# Get Precision and Recall for novel class
def getNovelPrecisionRecall(label,pred):
    TP,FP,TN,FN = 0,0,0,0
    for i in range(label.shape[0]):
        if label[i] == pred[i]:
            if label[i] ==  1:
                TP += 1
            else:
                TN += 1
        else:
            if label[i] == 1:
                FN += 1
            else:
                FP += 1
    P = (1.0*TN +1e-5) /(TN + FN +1e-5)
    R = (1.0*TN +1e-5) /(TN + FP +1e-5)
    return (P,R)

# Get error rate
def getError(label,pred):
    TP,FP,TN,FN = 0,0,0,0
    for i in range(label.shape[0]):
        if label[i] == pred[i]:
            if label[i] ==  1:
                TP += 1
            else:
                TN += 1
        else:
            if label[i] == 1:
                FN += 1
            else:
                FP += 1

    return (1.0-(1.0*TP/(TP + FN+1e-5)),1.0-(1.0*TN/(TN + FP+1e-5)))


# Run the ASG algorithm, and get Macro-F1 for test data
def get_macroF1(label, pred):
    P = precision_score(label, pred, average='macro')
    R = recall_score(label, pred, average='macro')
    f1 = f1_score(label, pred, average='macro')

    print "precision_score:", P
    print "recall_score:", R
    print "f1_score:", f1

    return f1