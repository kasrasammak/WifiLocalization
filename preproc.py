#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:02:53 2020

@author: owlthekasra
"""
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn import tree
import methods as mth

dataset = pd.read_csv('UJIndoorLoc/trainingData.csv')
datasetV = pd.read_csv('UJIndoorLoc/validationData.csv')

df1 = dataset.iloc[:,0:520]
df2 = dataset.iloc[:, 520:]
df1[:].replace({100: -104}, inplace=True)
#mf = pd.concat([df1, df2], axis=1)

df2['LAT-LONG-TIME-USER'] = df2.LATITUDE.astype(str) + "." + df2.LONGITUDE.astype(str) + "." + df2.TIMESTAMP.astype(str) + "." + df2.USERID.astype(str)

dfLatLongTimeUser = df2['LAT-LONG-TIME-USER']

groupDF = pd.concat([df1, df2], axis=1)
groupDF = groupDF.groupby(["LAT-LONG-TIME-USER"]).mean()

groupDF1 = groupDF.iloc[:,0:520]
groupDF1 = groupDF.reset_index()
mover = list(groupDF1.iloc[:, 1:].columns)
mover.append('LAT-LONG-TIME-USER')

groupDF = groupDF1[mover]
df1 = groupDF.iloc[:,0:520]
df2 = groupDF.iloc[:, 520:]

buildint = df2.BUILDINGID.astype(int) 
floorint = df2.FLOOR.astype(int)
bfstring = buildint.astype(str) + floorint.astype(str)
df2['BUILDINGFLOOR'] = bfstring.astype(int)
df2['LATITUDELONGITUDE'] = df2.LATITUDE.astype(str) + "." + df2.LONGITUDE.astype(str)

valid1 = datasetV.iloc[:,0:520]
valid2 = datasetV.iloc[:, 520:]
valid1[:].replace({100: -104}, inplace=True)

buildint2 = valid2.BUILDINGID.astype(int) 
floorint2 = valid2.FLOOR.astype(int)
bfstring2 = buildint2.astype(str) + floorint2.astype(str)
valid2['BUILDINGFLOOR'] = bfstring2.astype(int)
valid2['LATITUDELONGITUDE'] = valid2.LATITUDE.astype(str) + "." + valid2.LONGITUDE.astype(str)

#drop Empty columns from training and validation sets
df1Drop = mth.dropNoVarianceColumns(df1)

dropEmptyList = list(df1Drop.columns)
X_val = valid1[dropEmptyList]
y_val_b = valid2['BUILDINGID']
y_val_f = valid2['FLOOR']
y_val_lat_long = valid2['LATITUDELONGITUDE']
y_val_bf = valid2['BUILDINGFLOOR']
y_val_lat = valid2['LATITUDE']
y_val_long = valid2['LONGITUDE']

#knn
XBuild, yBuild = mth.createNewTarget(df1Drop, df2, 'BUILDINGID')
XFloor, yFloor = mth.createNewTarget(df1Drop, df2, 'FLOOR')
XSpace, ySpace = mth.createNewTarget(df1Drop, df2, 'SPACEID')
XBuildFloor, yBuildFloor = mth.createNewTarget(df1Drop, df2, 'BUILDINGFLOOR')
XLatLong, yLatLong = mth.createNewTarget(df1Drop, df2, 'LATITUDELONGITUDE')
XLat, yLat = mth.createNewTarget(df1Drop, df2, 'LATITUDE')
XLong, yLong = mth.createNewTarget(df1Drop, df2, 'LONGITUDE')