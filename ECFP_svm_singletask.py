# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:18:28 2020

@author: tanzheng
"""
import numpy as np

import pickle
# from xgboost import XGBRegressor
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
# 加载数据集
import time
from sklearn.metrics import roc_auc_score, make_scorer

with open('ECFP_tox21_data.pkl', 'rb') as f:
    data_list = pickle.load(f)
    
tasks, train_X, train_y, train_w, test_X, test_y, test_w, val_X, val_y, val_w = data_list


# 训练
X_train = train_X
y_train_all = train_y
X_val = val_X
y_val_all = val_y

# 超参数
C_range = [2**j for j in range(-5, 5, 1)]
param_grid = {'C': C_range}

# 存储最佳参数
best_params = []

print(f"Starting training on {len(tasks)} tasks...")
total_start_time = time.time()

for i, task_name in enumerate(tasks):
    start_time = time.time()
    
    # 准备当前任务的数据
    y_train = y_train_all[:, i]
    y_val = y_val_all[:, i]
    
    # 计算类别权重
    # 注意：原始公式是 (len - sum) / sum，即负/正样本比例。
    # 这实际上是加权正类 (1) 以平衡数量。
    pos_count = np.sum(y_train)
    neg_count = len(y_train) - pos_count
    w_ = neg_count / pos_count if pos_count > 0 else 1.0
    
    # 合并训练集和验证集用于 PredefinedSplit
    # -1 表示训练样本，0 表示验证样本
    test_fold = np.concatenate([np.full(len(y_train), -1), np.full(len(y_val), 0)])
    ps = PredefinedSplit(test_fold)
    
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))
    
    # 初始化 SVC
    # probability=False 为了速度（使用 decision_function）
    # 尽可能保持原始参数，但 probability=False 是关键。
    svc = SVC(kernel='rbf', 
              gamma=1, 
              degree=4, 
              random_state=42, # 固定种子
              tol=0.0008, 
              class_weight={1: w_},
              probability=False) 
    
    # 网格搜索
    # n_jobs=-1 使用所有处理器
    # scoring='roc_auc' 自动为 SVC 使用 decision_function
    grid = GridSearchCV(estimator=svc, 
                        param_grid=param_grid, 
                        cv=ps, 
                        scoring='roc_auc', 
                        n_jobs=-1,
                        return_train_score=True,
                        verbose=0)
    
    grid.fit(X_combined, y_combined)
    
    # 提取结果
    best_idx = grid.best_index_
    best_C = grid.best_params_['C']
    best_train_auc = grid.cv_results_['mean_train_score'][best_idx]
    best_val_auc = grid.cv_results_['mean_test_score'][best_idx]
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"{i+1} : 最佳模型得分: {best_C} / {best_train_auc:.5f} / {best_val_auc:.5f} (Time: {duration:.2f}s)")
    best_params.append(best_C)

print(f"Total time: {time.time() - total_start_time:.2f}s")