# -*- coding: utf-8 -*-

#import numpy as np
#import matplotlib.pyplot as plt
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
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

sn.set()

trainingUrl = "https://raw.githubusercontent.com/kasrasammak/WifiLocalization/master/UJIndoorLoc/trainingData.csv"
validationUrl = "https://raw.githubusercontent.com/kasrasammak/WifiLocalization/master/UJIndoorLoc/validationData.csv"
df_train = pd.read_csv(trainingUrl)
df_val = pd.read_csv(validationUrl)


modelknnBuild = initializeModel('knn', param_1 = 4)
accuracies_b, train_test_b_cv = mth.checkKCV(XBuild, yBuild,  10, )
accuracies_f, train_test_f_cv = mth.checkKCV(XFloor, yFloor, 10)
accuracies_s, train_test_s_cv = mth.checkKCV(XSpace, ySpace,  10)

val_pred_b, val_cm_b = mth.predictTest(model_b, valSet_X, valSet_y_b)
val_pred_f, val_cm_f = mth.predictTest(model_f, valSet_X, valSet_y_f)

model_bf, acc_bf, pred_bf, y_test_bf = mth.fitPredictValSet(XBuildFloor, yBuildFloor, X_val, y_val_bf, 'knn', 4)

model_bf_tr, acc_bf_tr, pred_bf_tr, ytest_bf_tr =  mth.fitPredictValSet(XBuildFloor, yBuildFloor, X_val, y_val_bf, 'forest')
model_bf_for, acc_bf_for, pred_bf_for, ytest_bf_for =  mth.fitPredictValSet(XBuildFloor, yBuildFloor, X_val, y_val_bf, 'forest')

model_lat, acc_lat, pred_lat, ytest_lat =  mth.fitPredictValSet(XLat, yLat, X_val, y_val_lat, 'knnr', 0, neighbors=4)
model_long, acc_long, pred_long, ytest_long =  mth.fitPredictValSet(XLong, yLong, X_val, y_val_long, 'knnr', 0, neighbors=4)



axes = plt.subplots(1, figsize=(5, 4))
line = np.linspace(0, 1, 1111).reshape(-1, 1)

plt.plot(line, ytest_lat)
plt.plot(line, pred_lat)

error = sqrt(mean_squared_error(ytest_lat,pred_lat))

error
model, lists = mth.evalHyperParams(XFloor, yFloor, X_val, y_val_f, 'knn', 2)
finalmodel_bf, models_bf = mth.evaluateModels(XBuildFloor, yBuildFloor, X_val, y_val_bf, [['knn', 4], ['tree', None], ['forest', None]], 1)
finalmodel_f, models_f = mth.evaluateModels(XFloor, yFloor, X_val, y_val_f, [['knn', 4], ['tree', None], ['forest', None]], 1)

model_lat, lists_lat = mth.evalHyperParams(XLat, yLat, X_val, y_val_lat, 'knnr', 3, 3)
model_long, lists_long = mth.evalHyperParams(XLong, yLong, X_val, y_val_long, 'rnr', 1, 4)



for j in range(1,len(lists)): 
    print(j)
#    lists[j].append(model[j])
   df2['LATITUDE']
   df2['BUILDINGID']
   df2['BUILDINGFLOOR']
mylists = [[],[],[],[]]
#for i in range(0,1):
model = mth.fitPredictValSet(XFloor, yFloor, X_val, y_val_f, 'knn', 2)
#model = (2,3,4,5)

lists[1].append(999)

#only if comparing train_test_split values
#val_test_b = train_test_b[3].reset_index()
#val_test_b = val_test_b.drop(columns=['index'])
val_pred_comp_b = pd.Series(val_pred_b)
val_pred_comp_b = pd.concat([val_pred_comp_b, valSet_y_b], axis=1)

val_pred_comp_f = pd.Series(val_pred_f)
val_pred_comp_f = pd.concat([val_pred_comp_f, valSet_y_f], axis=1)

acc_val_b = accuracy_score(val_pred_b, valSet_y_b)
acc_val_f = accuracy_score(val_pred_f, valSet_y_f)

accuracies_b, train_test_b_cv = mth.checkKCV(XBuild, yBuild, 5, 10)

val_pred_b_tree, val_cm_b_tree = mth.predictTest(model_b_tree, valSet_X, valSet_y_b)
acc_val_b_tree = mth.accuracy_score(val_pred_b_tree, valSet_y_b)

#decision tree
val_pred_b_tree, val_cm_b_tree = mth.predictTest(model_b_tree, valSet_X, valSet_y_b)
acc_val_b_tree = mth.accuracy_score(val_pred_b_tree, valSet_y_b)

val_pred_f_tree, val_cm_f_tree = mth.predictTest(model_f_tree, valSet_X, valSet_y_f)
acc_val_f_tree = mth.accuracy_score(val_pred_f_tree, valSet_y_f )

        

val_pred_comp_f_tree = pd.Series(val_pred_f_tree)
val_pred_comp_f_tree = pd.concat([val_pred_comp_f_tree, valSet_y_f], axis=1)

import plotly.express as px
layout = px.scatter_3d(df2, x='LATITUDE', y='LONGITUDE', z='FLOOR')




#def testTuple(X,y):
#    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#    train_test = X_train, X_test, y_train, y_test
#    name = ['X_train', 'X_test', 'y_train', 'y_test']
#    name2 = (name,)
#    train_test = train_test + name2
#    return train_test

#X_train, X_test, y_train, y_test = train_test_split(XBuild, yBuild, random_state=1)
#knnBuild = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
#
#knnBuild.fit(X_train, y_train)
#
#y_pred = knnBuild.predict(X_test)
#y_pred
#cm = confusion_matrix(y_test, y_pred)

#from keras.wrappers.scikit_learn import KerasClassifier

#accuracies = cross_val_score(estimator = knnBuild, X = X_train, y = y_train, cv = 10, n_jobs = -1)


#preproc get rid of -104
b1f3dr = b1[b1['FLOOR'] == 3]
b1f3dr = dropEmptyColumns(b1f3dr)

#b1f3dr.loc[b1f3dr['WAP090'] >= -67, 'C'] = 1
#b1f3dr.loc[b1f3dr['WAP090'] < -67, 'C'] = 0
#
#X = b1f3dr[['mean area', 'mean compactness']]
#y = b1f3dr['C']

#KNN
nbrsb0f0 = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(b0f0)

distances, indices = nbrsb0f0.kneighbors(b0f0)




        

nbrsb0f0.kneighbors_graph(b0f0).toarray()
b0.SPACEID.unique

countsb0 = b0['SPACEID'].value_counts()
countsb0 = countsb0[:10,]

some_values = b0['SPACEID'].value_counts()[:20].index.tolist()

b0TopVal20 = b0.loc[b0['SPACEID'].isin(some_values)]


b01314 = b0TopVal[['WAP013','WAP014']]

print(b01314.corr())

corrMatrix = b01314.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()