import os
import numpy as np
import pandas as pd
import ast
import pymol
from pymol import cmd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import argparse
import time

parser = argparse.ArgumentParser(description='Show protein-ligand complex with attention scores')
parser.add_argument('--split', required=True, help="Which split to visualize", type=str)
parser.add_argument('--index', required=True, help="Which protein?", type=str)
parser.add_argument('--model', required=True, choices=['Baseline', 'BCFM', 'CCFM', 'MPESI'], help="Which model to visualize")
parser.add_argument('--show_surface', action='store_true', help="Whether to show surface mode for the protein")
args = parser.parse_args()

split = args.split
model = args.model
id_index = int(args.index)

csv_file_path = f'./models/{model}/fold_{split}.csv'    
pdb_folder = f'./AF3_complexs/complex_{split}/'
df = pd.read_csv(csv_file_path)
df['bind'] = df['bind'].apply(ast.literal_eval)

pymol.finish_launching()

def get_color(score, vmin, vmax):
    colormap = cm.get_cmap('Reds')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    color = colormap(norm(score))
    return [float(color[0]), float(color[1]), float(color[2])]

txt_log = []


pdb_file_path = os.path.join(pdb_folder, f'fold_{split}_test_{id_index}_model_0.cif')
cmd.load(pdb_file_path, f'complex_{model}_{split}_{id_index}')

att_p_str = df.at[id_index-1, 'att_p']
att_p_scores = np.array(ast.literal_eval(att_p_str))

special_points = df.at[id_index-1, 'bind']
p_len = len(df.at[id_index-1, 'Protein'])
p_len = p_len if p_len < 1024 else 1024
att_p_scores = att_p_scores[:p_len].squeeze()

attention_scores = {i: att_p_scores[i-1] for i in range(1, len(att_p_scores) + 1)}

vmin, vmax = min(att_p_scores), max(att_p_scores)

cmd.bg_color('white')

cmd.remove('solvent')

cmd.select('protein_chain', f'complex_{model}_{split}_{id_index} and polymer')
cmd.create('pro_cartoon', 'protein_chain')

if args.show_surface:
    cmd.create('pro_surface', 'protein_chain')
    cmd.show('surface', 'pro_surface')
    cmd.color('white', 'pro_surface')
    cmd.set('transparency', 0.8, 'pro_surface')

cmd.show('cartoon', 'pro_cartoon')

for resi, score in attention_scores.items():
    color = get_color(score, vmin, vmax)
    color_name = f'color_{model}_{split}_{id_index}_{resi}'
    cmd.set_color(color_name, color)
    cmd.color(color_name, f'complex_{model}_{split}_{id_index} and polymer and resi {resi}')

top_10_attention = sorted(attention_scores.items(), key=lambda item: item[1], reverse=True)[:10]
p_seq = df.at[id_index-1, 'Protein']

txt_log.append(f'For {id_index}')
for key, att_score in top_10_attention:
    residue_type = p_seq[key-1]
    txt_log.append(f"{key}_{residue_type}: {att_score}")

# save top-10 attention score
os.makedirs("Top10_att", exist_ok=True)
with open(f"Top10_att/Split_{split}_{id_index}_{model}.txt", "w") as file:
    for line in txt_log:
        file.write(line + "\n") 

if 'ligand' in df.columns:
    ligand_name = df.at[id_index - 1, 'ligand']
    cmd.select('ligand_select', f'complex_{model}_{split}_{id_index} and resn {ligand_name}')
    cmd.create('ligand', 'ligand_select')
    cmd.color('red', 'ligand')

cmd.zoom(f'complex_{model}_{split}_{id_index} and polymer')

os.makedirs("complex_images", exist_ok=True)

cmd.viewport(1200, 1200)
image_file_path = f"complex_images/complex_{model}_{split}_{id_index}"
time.sleep(3)
cmd.png(image_file_path, dpi=600, ray=0)

#cmd.quit()