import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import ast
from matplotlib.colors import Normalize
import os

# 读取CSV文件
model_list = ['Baseline', 'BCFM', 'CCFM', 'MPESI']

save_dir = 'fig_pca'
os.makedirs(save_dir, exist_ok=True)

for model in model_list:
    model_type = model
    df = pd.read_csv(f'kcat_km_embedding/result_{model_type}_MPEK_kcat_km.csv')

    # 假设feat列是字符串形式的numpy数组，使用ast.literal_eval转换
    df['feat'] = df['feat'].apply(ast.literal_eval)
    features = np.array(df['feat'].tolist())  # 将列表转换为NumPy数组

    # 执行PCA降维，这里降到2维便于可视化
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    # 使用matplotlib内置的bwr色条
    cmap = plt.get_cmap('seismic')

    # 计算kcat的平均值和最大值
    median_kcat = df['Score'].mean()
    max_kcat = df['Score'].max()

    # 归一化
    norm = Normalize(vmin=df['Score'].min(), vmax=max_kcat)

    # 根据Score列的值设置颜色
    colors = cmap(norm(df['Score']))

    # 绘制PCA结果
    fig, ax = plt.subplots(figsize=(4, 3))
    scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1], c=colors, edgecolors='none', s=2)

    # 设置标题和坐标轴标签
    plt.title(f'PCA for {model_type}')

    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    temp_filename = f'{save_dir}/pca_kcat_km_{model_type}.png'
    plt.savefig(temp_filename, dpi=600, format='png')
    plt.close()
