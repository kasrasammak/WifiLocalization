#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:55:35 2020

@author: owlthekasra
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV


def add(a, b):
    return a + b

def dropNoVarianceColumns(df):
    for col in df.columns:
        val = df[col][0]
        if (df[col] == val).sum()/len(df) == 1:
            df = df.drop(col,axis=1)
    return df

def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

def dropNthObs(df, n):
    return df.iloc[::n, :]

def createNewTarget(df, df2, name):
        y = df2[name]
        return df, y  

modelnames = ['knn','tree']

def initializeModel(name, neighbors=5):
    if (name == 'knn'):
        model =  KNeighborsClassifier(n_neighbors = neighbors)
    elif (name == 'tree'):
        model = tree.DecisionTreeClassifier()
    elif (name == 'forest'):
        model = RandomForestClassifier()
    elif (name == 'knnr'):
        model = KNeighborsRegressor(n_neighbors=neighbors)
    return model

def splitNFit(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model = model
    model.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test

def predictTest(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    return acc, y_pred, cm, y_test

def fitPredict(X, y, model):
    train_test = splitNFit(X, y, model)
    acc, y_pred, cm, _y_test = predictTest(model, train_test[1], train_test[3])
    return  model, y_pred, acc, cm, train_test

def fitPredictValSet(X, y, X_val, y_val, name, param):
    model = initializeModel(name, param_1 = param)
    model, _, _, _, _ = fitPredict(X, y, model)
    acc, pred, _, y_test = predictTest(model, X_val, y_val)
    return model, acc, pred, y_test

def getBest(model, acc, index, name, finalModel):
    if (index == 0):
        finalModel = (model, acc, name)
    elif (acc > finalModel[1]):
        finalModel = (model, acc, name)
    return finalModel

def checkKCV(X, y, neighbors, cv):
    train_test = train_test_split(X, y, random_state=1)
    model = KNeighborsClassifier(n_neighbors = neighbors, metric='euclidean')
    accuracies = cross_val_score(estimator = model, X = train_test[0], y = train_test[2], cv = cv, n_jobs = -1)
    return accuracies, train_test

def appender(strt, end):
    appenders = range(strt, end)
    mylists = [[], [], [], []]
    for x in appenders:
        for lst in mylists:
            lst.append(x)
    return mylists

def evalHyperParams(X, y, X_val, y_val, name, param):
    mylists = [[],[],[],[]]
    finalModel = ()
    for i in range(1,param+1):
        model = fitPredictValSet(X, y, X_val, y_val, name, i)
        for j in range(0, len(mylists)):   
            mylists[j].append(model[j])
        finalModel = getBest(mylists[0][i-1], mylists[1][i-1], i-1, i, finalModel)
    return finalModel, mylists
    

def evaluateModels(X, y, X_val, y_val, modelnames):
    models = [[],[],[],[]]
    finalModel = ()
    for i in range(0, len(modelnames)):
        model = fitPredictValSet(X, y, X_val, y_val, modelnames[i][0], modelnames[i][1])
        for j in range(0, len(models)):   
            models[j].append(model[j])
#        models[i], _,_,_,_ = fitPredict(X, y , instantiateModel(modelnames[i]))
#        pred[i], acc[i], _, _ = predictTest(models[i], X_val, y_val)

        finalModel = getBest(models[0][i], models[1][i], i, modelnames[i], finalModel)
    return finalModel, models

def kasra_score(y_test, y_pred):
    num = 0
    denom = len(y_test)
    for i in range(0, y_test.shape[0]) :
        if y_test[i] == y_pred[i]:
                num = num +1
    return num/denom
    
#def fitTreePredict(X, y):
#    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#    model = tree.DecisionTreeClassifier()
#    model.fit(X_train, y_train)
#    y_pred = model.predict(X_test)
#    cm = confusion_matrix(y_test, y_pred)
#    train_test = X_train, X_test, y_train, y_test
#    return  y_pred, cm, model, train_test
#
#def fitKNNPredict(X, y, neighbors):
#    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#    model = KNeighborsClassifier(n_neighbors = neighbors, metric='euclidean')
#    model.fit(X_train, y_train)
#    y_pred = model.predict(X_test)
#    cm = confusion_matrix(y_test, y_pred)
#    train_test = X_train, X_test, y_train, y_test
#    return  y_pred, cm, model, train_test]