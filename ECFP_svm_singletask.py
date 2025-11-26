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
import argparse
import yaml
import os
import csv
import datetime

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ECFP SVM Training Script')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--C_start', type=int, help='Start power for C (2^start)')
    parser.add_argument('--C_end', type=int, help='End power for C (2^end)')
    parser.add_argument('--gamma', type=float, help='Kernel coefficient for rbf')
    parser.add_argument('--degree', type=int, help='Degree of the polynomial kernel function')
    parser.add_argument('--tol', type=float, help='Tolerance for stopping criterion')
    parser.add_argument('--output_dir', type=str, help='Directory to save results')
    return parser.parse_args()

def main():
    # 1. Parse Args and Load Config
    args = parse_args()
    config = load_config(args.config)

    # Override config with CLI args if provided
    if args.C_start is not None: config['C_range']['start'] = args.C_start
    if args.C_end is not None: config['C_range']['end'] = args.C_end
    if args.gamma is not None: config['gamma'] = args.gamma
    if args.degree is not None: config['degree'] = args.degree
    if args.tol is not None: config['tol'] = args.tol
    if args.output_dir is not None: config['output_dir'] = args.output_dir

    # 2. Setup Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(config['output_dir'], f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save current config
    with open(os.path.join(exp_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)

    # 3. Load Data
    print("Loading data...")
    with open('ECFP_tox21_data.pkl', 'rb') as f:
        data_list = pickle.load(f)
        
    tasks, train_X, train_y, train_w, test_X, test_y, test_w, val_X, val_y, val_w = data_list

    # Training Data
    X_train = train_X
    y_train_all = train_y
    X_val = val_X
    y_val_all = val_y

    # Hyperparameters
    C_range = [config['C_range']['base']**j for j in range(config['C_range']['start'], config['C_range']['end'], config['C_range']['step'])]
    param_grid = {'C': C_range}

    # Store best parameters
    results = []

    print(f"Starting training on {len(tasks)} tasks...")
    total_start_time = time.time()

    for i, task_name in enumerate(tasks):
        start_time = time.time()
        
        # Prepare data for this task
        y_train = y_train_all[:, i]
        y_val = y_val_all[:, i]
        
        # Calculate class weight
        pos_count = np.sum(y_train)
        neg_count = len(y_train) - pos_count
        w_ = neg_count / pos_count if pos_count > 0 else 1.0
        
        # Combine Train and Val for PredefinedSplit
        test_fold = np.concatenate([np.full(len(y_train), -1), np.full(len(y_val), 0)])
        ps = PredefinedSplit(test_fold)
        
        X_combined = np.vstack((X_train, X_val))
        y_combined = np.concatenate((y_train, y_val))
        
        # Initialize SVC
        svc = SVC(kernel=config['kernel'], 
                  gamma=config['gamma'], 
                  degree=config['degree'], 
                  random_state=config['random_state'], 
                  tol=config['tol'], 
                  class_weight={1: w_},
                  probability=False) 
        
        # GridSearchCV
        grid = GridSearchCV(estimator=svc, 
                            param_grid=param_grid, 
                            cv=ps, 
                            scoring='roc_auc', 
                            n_jobs=config['n_jobs'],
                            return_train_score=True,
                            verbose=0)
        
        grid.fit(X_combined, y_combined)
        
        # Extract results
        best_idx = grid.best_index_
        best_C = grid.best_params_['C']
        best_train_auc = grid.cv_results_['mean_train_score'][best_idx]
        best_val_auc = grid.cv_results_['mean_test_score'][best_idx]
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"{i+1} : 最佳模型得分: {best_C} / {best_train_auc:.5f} / {best_val_auc:.5f} (Time: {duration:.2f}s)")
        
        results.append({
            'Task': task_name,
            'Best_C': best_C,
            'Train_AUC': best_train_auc,
            'Val_AUC': best_val_auc,
            'Time_Seconds': duration
        })

    total_time = time.time() - total_start_time
    print(f"Total time: {total_time:.2f}s")

    # 4. Save Results to CSV
    csv_path = os.path.join(exp_dir, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Task', 'Best_C', 'Train_AUC', 'Val_AUC', 'Time_Seconds'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {exp_dir}")

if __name__ == '__main__':
    main()