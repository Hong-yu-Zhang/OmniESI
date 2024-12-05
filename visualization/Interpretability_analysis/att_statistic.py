import os
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import glob

model_type = 'models/MESI'
csv_files = glob.glob(f'{model_type}/fold_*.csv')
data = []
for file in csv_files:
    part = pd.read_csv(file)
    data.append(part)
df = pd.concat(data, ignore_index=True)
df['bind'] = df['bind'].apply(ast.literal_eval)
special_points_means = []
non_special_points_means = []

for row_index in range(len(df)):
    att_p_str = df.at[row_index, 'att_p']
    att_p_scores = np.array(ast.literal_eval(att_p_str))

    special_points = df.at[row_index, 'bind']
    p_len = len(df.at[row_index, 'Protein'])
    p_len = p_len if p_len < 1023 else 1023
    special_points = [i if i < 1023 else 1023 for i in special_points]
    special_points_indices = [i - 1 for i in special_points]
    non_special_points_indices = [i for i in range(p_len) if i not in special_points_indices]

    special_points_mean = np.mean(att_p_scores[special_points_indices])
    non_special_points_mean = np.mean(att_p_scores[non_special_points_indices])

    special_points_means.append(special_points_mean)
    non_special_points_means.append(non_special_points_mean)

fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(len(df))


bar1 = ax.bar(index, special_points_means, bar_width, label='Pocket region', color='#2c3a7b')
bar2 = ax.bar(index + bar_width, non_special_points_means, bar_width, label='Non-pocket region', color='#f39689')

ax.set_xlabel('Sample Index')
ax.set_ylabel('Mean Attention Score')
ax.set_yticks(np.arange(0, 0.04, 0.01))
ax.set_ylim(0, 0.035)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

ax.set_xticks([0, len(df) - 1])
ax.set_xticklabels([1, len(df)])
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('fig_att/attention_scores_index.png', dpi=600)
plt.show()
