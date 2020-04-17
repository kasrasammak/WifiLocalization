#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:14:03 2020

@author: owlthekasra
"""
import methods as mth
from sklearn.neighbors import KNeighborsClassifier

train_test_bf, model_bf = mth.splitNFit(XBuildFloor, yBuildFloor, mth.initializeModel('knn', param_1=4))
model_bf.fit(train_test_bf[0], train_test_bf[2])
pred_bf = model_bf.predict(X_val)
acc_bf = accuracy_score(y_val_bf, pred_bf)
cm_bf = confusion_matrix(y_val_bf, pred_bf)
num = 0
denom = 0
obs = 0
for i in range(0,len(cm_bf)):
    num = num + (sum(cm_bf[i])*sum(cm_bf[:,i]))
    denom = denom+sum(cm_bf[i])
    obs = obs + cm_bf[i,i]
expected = num/denom
kappa = (obs - expected)/(denom - expected)

