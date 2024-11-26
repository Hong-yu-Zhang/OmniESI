import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

df = pd.read_csv('./mut_embeddings/embedding_ours.csv')

df['features'] = df['feat'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))

df['Y'] = df['Y'].astype(int)

df['colors'] = np.where(df['Y'] > 0, '#00007f', '#db8bf4')

unique_smiles = df['SMILES'].unique()

for index, smiles in enumerate(unique_smiles):
    subset = df[df['SMILES'] == smiles]
    features = np.stack(subset['features'].to_numpy())
    colors = subset['colors'].to_numpy()

    print(f"Features shape for SMILES {smiles}: {features.shape}")

    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    print(f"PCA result shape: {features_pca.shape}")

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(features_pca[:, 0], features_pca[:, 1], c=colors, s=15, edgecolors='white', linewidths=0.3)

    plt.title(f'{smiles[0:10]}')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    plt.savefig(f'./fig_pca/pca_mut_{index}.png', dpi=600)
    plt.close()
