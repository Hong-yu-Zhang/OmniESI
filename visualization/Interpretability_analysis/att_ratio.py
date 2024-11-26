import os
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import glob

model_type = 'models/MPESI'
csv_files = glob.glob(f'{model_type}/fold_*.csv')
data = []
for file in csv_files:
    part = pd.read_csv(file)
    part = part.dropna()
    data.append(part)
df = pd.concat(data, ignore_index=True)

df['bind'] = df['bind'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)

special_points_means = []
non_special_points_means = []
additional_values = []
ratios = []

for row_index in range(len(df)):
    try:
        att_p_str = df.at[row_index, 'att_p']
        att_p_scores = np.array(pd.eval(att_p_str))

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
        ratios.append(special_points_mean / non_special_points_mean)

        additional_value = 1 / p_len
        additional_values.append(additional_value)

    except Exception as e:
        print(f"Error processing row {row_index}: {e}")

# Modify labels to '>1', '>2', '>3' etc.
thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
labels = [f'>{t}' for t in thresholds]

# Compute the counts for ratios greater than each threshold
counts = [np.sum(np.array(ratios) > t) for t in thresholds]

# Total number of ratios
total_count = len(ratios)

# Create the plot
fig, ax = plt.subplots(figsize=(4, 4))

face_color = '#dce6de'

# Create the bar chart
bars = ax.bar(labels, counts, color=face_color)

# Annotate each bar with the percentage
for bar, count in zip(bars, counts):
    height = bar.get_height()
    percentage = (count / total_count) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{percentage:.2f}%', ha='center', va='bottom', fontsize=6)

ax.set_xlabel('Ratio Threshold')
ax.set_ylabel('Count')
ax.set_yticks(np.arange(0, 500, 100))
ax.set_ylim(0, total_count)  # Set the y-limit to the total count
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.set_title('Distribution of Attention Ratios')

# Tight layout and save the figure
plt.tight_layout()
plt.savefig('fig_att/attention_ratios.png', dpi=600)
plt.show()
