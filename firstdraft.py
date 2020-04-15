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

sn.set()

dataset = pd.read_csv('UJIndoorLoc/trainingData.csv')
datasetV = pd.read_csv('UJIndoorLoc/validationData.csv')


df1 = dataset.iloc[:,0:520]
df2 = dataset.iloc[:, 520:]
df1[:].replace({100: -104}, inplace=True)

df2['BUILDINGFLOOR'] = df2.BUILDINGID.astype(str) + "." + df2.FLOOR.astype(str)
df2['LATITUDELONGITUDE'] = df2.LATITUDE.astype(str) + "." + df2.LONGITUDE.astype(str)
df2['LAT-LONG-TIME-USER'] = df2.LATITUDE.astype(str) + "." + df2.LONGITUDE.astype(str) + "." + df2.TIMESTAMP.astype(str) + "." + df2.USERID.astype(str)

dfLatLongTimeUser = df2['LAT-LONG-TIME-USER']

groupDF = pd.concat([df1, df2], axis=1)
groupDF = groupDF.groupby(["LAT-LONG-TIME-USER"]).mean()

groupDF1 = groupDF.iloc[:,0:520]

groupDF1 = groupDF.reset_index()
groupppp = groupDF1.iloc[:, 1:]
mover = list(groupDF1.iloc[:, 1:].columns)
mover = pd.concat([mover, "LAT-LONG-TIME-USER"])
mover.append('LAT-LONG-TIME-USER')

groupDF = groupDF1[mover]

valid1 = datasetV.iloc[:,0:520]
valid2 = datasetV.iloc[:, 520:]
valid1[:].replace({100: -104}, inplace=True)


mf = pd.concat([df1, df2], axis=1)

dfVT = df1

dfVT = variance_threshold_selector(dfVT, 0)

lisst = mth.appender(0,1)

#drop Empty columns from training and validation sets
df1Drop = mth.dropNoVarianceColumns(df1)


dropEmptyList = list(df1Drop.columns)
X_val = valid1[dropEmptyList]
y_val_b = valid2['BUILDINGID']
y_val_f = valid2['FLOOR']

train_test_b[3]

#knn
XBuild, yBuild = mth.createNewTarget(df1Drop, df2, 'BUILDINGID')
knn1 = KNeighborsClassifier(n_neighbors = 5, metric='euclidean')
pred_b, cm_b, model_b, train_test_b = mth.fitPredict(XBuild, yBuild, knn1)
accuracies_b, train_test_b_cv = mth.checkKCV(XBuild, yBuild, 5, 10)


pred_b, _, _, train_test_b = mth.fitPredict(XBuild, yBuild, knn1)


XFloor, yFloor = mth.createNewTarget(df1Drop, df2, 'FLOOR')
pred_f, cm_f, model_f, train_test_f = mth.fitPredict(XFloor, yFloor, knn1)
accuracies_f, train_test_f_cv = mth.checkKCV(XFloor, yFloor, 5, 10)

XSpace, ySpace = mth.createNewTarget(df1Drop, df2, 'SPACEID')
pred_s, cm_s, model_s, train_test_s = mth.fitKNNPredict(XSpace, ySpace, 5)
accuracies_s, train_test_s_cv = mth.checkKCV(XSpace, ySpace, 5, 10)

tree_floor = mth.fitTreePredict(XFloor, yFloor)

val_pred_b, val_cm_b = mth.predictTest(model_b, valSet_X, valSet_y_b)
val_pred_f, val_cm_f = mth.predictTest(model_f, valSet_X, valSet_y_f)

model_f, acc_f, pred_f, y_test_f = mth.fitPredictValSet(XFloor, yFloor, X_val, y_val_f, 'knn')

def evalHyperParams(X, y, X_val, y_val, name, param):
#    for i in range(0, param):
#        model[i] = [i]
#        acc[i] = i
    appenders = range(0, 4)
    mylists = []
#    finalModel = ()
    for i in appenders:
        mylists.append([])
    for i in range(0,param):
        model = fitPredictValSet(X, y, X_val, y_val, name, param)
        for j, lst in enumerate(mylists):   
            j.append(model[lst])
#        model[i] = instantiateModel(name, param_1 = param)
#        model[i], y_pred[i], _, _ = fitPredict(X, y, model[i])
#        acc[i], pred[i], _, _ = predictTest(model[i], X_val, y_val)
#        finalModel = getBest(model[i], acc[i], i, i, finalModel)
#    return finalModel[0], finalModel[1], finalModel[2]
    return mylists

model, lists = mth.evalHyperParams(XFloor, yFloor, X_val, y_val_f, 'knn', 2)
finalmodel, models = mth.evaluateModels(XFloor, yFloor, X_val, y_val_f, [['knn', 5], ['tree',None], ['forest',None]])
for j in range(1,len(lists)): 
    print(j)
#    lists[j].append(model[j])
    
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

pred_b_tree, cm_b_tree, model_b_tree, train_test_b_tree = mth.fitTreePredict(XBuild, yBuild)
val_pred_b_tree, val_cm_b_tree = mth.predictTest(model_b_tree, valSet_X, valSet_y_b)
acc_val_b_tree = mth.accuracy_score(val_pred_b_tree, valSet_y_b)

#decision tree
pred_b_tree, cm_b_tree, model_b_tree, train_test_b_tree = mth.fitTreePredict(XBuild, yBuild)
val_pred_b_tree, val_cm_b_tree = mth.predictTest(model_b_tree, valSet_X, valSet_y_b)
acc_val_b_tree = mth.accuracy_score(val_pred_b_tree, valSet_y_b)

pred_f_tree, cm_f_tree, model_f_tree, train_test_f_tree = mth.fitTreePredict(XFloor, yFloor)
val_pred_f_tree, val_cm_f_tree = mth.predictTest(model_f_tree, valSet_X, valSet_y_f)
acc_val_f_tree = mth.accuracy_score(val_pred_f_tree, valSet_y_f )

        

val_pred_comp_f_tree = pd.Series(val_pred_f_tree)
val_pred_comp_f_tree = pd.concat([val_pred_comp_f_tree, valSet_y_f], axis=1)

import plotly.express as px
layout = px.scatter_3d(df2, x='LATITUDE', y='LONGITUDE', z='FLOOR')







y = np.array(val_pred_comp_b[0])
yy = np.array(val_pred_comp_b['BUILDINGID'])
acc2 = kasra_score(val_pred_comp_b[0], val_pred_comp_b['BUILDINGID'])



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