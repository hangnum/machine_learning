#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制超参数调优过程中各任务 AUC 的变化趋势
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 实验配置
experiments = [
    {
        'name': 'Exp 1: γ=1.0, tol=0.0008',
        'path': 'results/exp_20251123_181008/results.csv',
        'color': '#e74c3c',  # 红色 - 过拟合
        'marker': 'o'
    },
    {
        'name': 'Exp 2: γ=0.001, tol=1.0',
        'path': 'results/exp_20251123_192834/results.csv',
        'color': '#f39c12',  # 橙色 - 欠拟合风险
        'marker': 's'
    },
    {
        'name': 'Exp 3: γ=0.001, tol=0.001',
        'path': 'results/exp_20251123_204415/results.csv',
        'color': '#27ae60',  # 绿色 - 最佳
        'marker': '^'
    }
]

def load_results():
    """加载所有实验结果"""
    data = {}
    for exp in experiments:
        if os.path.exists(exp['path']):
            df = pd.read_csv(exp['path'])
            data[exp['name']] = df
            print(f"✓ 载入: {exp['name']} ({len(df)} 个任务)")
        else:
            print(f"✗ 未找到: {exp['path']}")
    return data

def plot_comparison(data):
    """绘制对比折线图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图1: 验证集 AUC 对比
    for exp in experiments:
        exp_name = exp['name']
        if exp_name in data:
            df = data[exp_name]
            ax1.plot(range(len(df)), df['Val_AUC'], 
                    label=exp_name, 
                    color=exp['color'],
                    marker=exp['marker'],
                    linewidth=2,
                    markersize=8,
                    alpha=0.8)
    
    ax1.set_xlabel('任务索引', fontsize=12, fontweight='bold')
    ax1.set_ylabel('验证集 AUC', fontsize=12, fontweight='bold')
    ax1.set_title('不同实验配置下的验证集 AUC 对比', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0.6, 0.9])
    
    # 设置 x 轴刻度为任务名称
    if experiments and experiments[0]['name'] in data:
        tasks = data[experiments[0]['name']]['Task'].tolist()
        ax1.set_xticks(range(len(tasks)))
        ax1.set_xticklabels(tasks, rotation=45, ha='right', fontsize=9)
    
    # 图2: 训练集 AUC 对比 (检测过拟合)
    for exp in experiments:
        exp_name = exp['name']
        if exp_name in data:
            df = data[exp_name]
            ax2.plot(range(len(df)), df['Train_AUC'], 
                    label=exp_name, 
                    color=exp['color'],
                    marker=exp['marker'],
                    linewidth=2,
                    markersize=8,
                    alpha=0.8)
    
    ax2.set_xlabel('任务索引', fontsize=12, fontweight='bold')
    ax2.set_ylabel('训练集 AUC', fontsize=12, fontweight='bold')
    ax2.set_title('不同实验配置下的训练集 AUC 对比', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0.75, 1.0])
    
    if experiments and experiments[0]['name'] in data:
        ax2.set_xticks(range(len(tasks)))
        ax2.set_xticklabels(tasks, rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    output_path = 'tuning_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_path}")
    plt.show()

def plot_individual_tasks(data):
    """为每个任务单独绘制折线图"""
    if not data:
        return
    
    # 获取任务列表
    first_exp = list(data.values())[0]
    tasks = first_exp['Task'].tolist()
    
    # 创建子图
    n_tasks = len(tasks)
    n_cols = 4
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten() if n_tasks > 1 else [axes]
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        
        # 收集该任务在不同实验中的 AUC
        exp_names = []
        val_aucs = []
        train_aucs = []
        
        for exp in experiments:
            exp_name = exp['name']
            if exp_name in data:
                df = data[exp_name]
                task_data = df[df['Task'] == task]
                if not task_data.empty:
                    exp_names.append(exp_name.split(':')[0])  # 简化标签
                    val_aucs.append(task_data['Val_AUC'].values[0])
                    train_aucs.append(task_data['Train_AUC'].values[0])
        
        x = range(len(exp_names))
        ax.plot(x, val_aucs, 'o-', label='验证集 AUC', linewidth=2, markersize=8, color='#3498db')
        ax.plot(x, train_aucs, 's--', label='训练集 AUC', linewidth=2, markersize=8, color='#e67e22', alpha=0.6)
        
        ax.set_title(task, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('AUC', fontsize=9)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.6, 1.0])
    
    # 隐藏多余的子图
    for idx in range(n_tasks, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('各任务在不同实验配置下的 AUC 变化', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = 'tuning_individual_tasks.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 分任务图表已保存: {output_path}")
    plt.show()

def calculate_statistics(data):
    """计算并打印统计信息"""
    print("\n" + "="*60)
    print("各实验统计信息")
    print("="*60)
    
    for exp in experiments:
        exp_name = exp['name']
        if exp_name in data:
            df = data[exp_name]
            print(f"\n{exp_name}:")
            print(f"  平均验证集 AUC: {df['Val_AUC'].mean():.4f}")
            print(f"  平均训练集 AUC: {df['Train_AUC'].mean():.4f}")
            print(f"  过拟合程度 (Train-Val): {df['Train_AUC'].mean() - df['Val_AUC'].mean():.4f}")
            print(f"  最佳任务: {df.loc[df['Val_AUC'].idxmax(), 'Task']} (AUC={df['Val_AUC'].max():.4f})")
            print(f"  平均训练时间: {df['Time_Seconds'].mean():.2f}s")

if __name__ == '__main__':
    print("读取实验结果...")
    results = load_results()
    
    if results:
        print("\n生成对比图表...")
        plot_comparison(results)
        
        print("\n生成分任务图表...")
        plot_individual_tasks(results)
        
        calculate_statistics(results)
        
        print("\n" + "="*60)
        print("✓ 完成！")
        print("="*60)
    else:
        print("\n✗ 错误: 未找到任何实验结果文件")
