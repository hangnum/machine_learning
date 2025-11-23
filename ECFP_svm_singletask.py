# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:18:28 2020

@author: tanzheng
"""
import numpy as np
# import deepchem as DC
# import joblib
import pickle
# from xgboost import XGBRegressor
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
#load dataset
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
import pandas as pd

with open('ECFP_tox21_data.pkl', 'rb') as f:
    data_list = pickle.load(f)
    f.close()
    
tasks, train_X, train_y, train_w, test_X, test_y, test_w, val_X, val_y, val_w = data_list


#training
X = train_X
prop_y = train_y
# y_train = prop_y[:,2]
task = tasks
'''
C=14, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
max_iter=-1, probability=True, random_state=None, shrinking=True,
tol=0.001
'''
C=[2**j for j in range(-5,5,1)]
gamma=[2**j for j in range(-5,5,1)]
degree = 4
tol = 0.0008
a=[]

for i in task:
    all_train_auc = []
    all_val_auc =[]

    start_time = time.time()
    id_=task.index(i)
    w_ = (len(train_y[:,id_]) - sum(train_y[:,id_])) / sum(train_y[:,id_])
    # n_tree = n_esti[id_]
    # depth = max_dep[id_]
    auc_train_all = []
    auc_all = []
    
    for ii in C:
        y_train = prop_y[:,id_]
        y_val = val_y[:,id_]
        
        model = SVC(kernel= 'rbf' ,  ##
                    C = ii ,
                    gamma = 1,
                    degree= 1 ,
                    random_state = True,
                    tol= 1,
                    class_weight={1:w_},
                    probability = True)
        
        model.fit(X, y_train)
        y_train_pred = model.predict(X)
        y_train_prob = model.predict_proba(X)[:,1]
        
        y_val_pred = model.predict(val_X)
        y_val_pred_prob = model.predict_proba(val_X)[:,1]
        
        train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_prob)#sample_weight = test_w[:,i]
        # train AUC
        auc_train = auc(train_fpr, train_tpr)
        all_train_auc.append(auc_train)
        
        
        # val auc
        fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_prob)#sample_weight = test_w[:,i]
        # AUC
        auc_s = auc(fpr, tpr)
        all_val_auc.append(auc_s)
        
    sort_id_ascend=np.argsort(all_val_auc)[0]
    print(id_+1,':最佳模型得分:{}/{}/{}'.format(C[sort_id_ascend], 
                                         all_train_auc[sort_id_ascend], all_val_auc[sort_id_ascend]))
    end_time = time.time()
    print(end_time - start_time)
    a.append(C[sort_id_ascend])
print('1')