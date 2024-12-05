from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import argparse
import ast
from PIL import Image, ImageDraw, ImageFilter
import io
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import os

# 解析命令行参数
parser = argparse.ArgumentParser(description='Show pocket attention scores')
parser.add_argument('--split', required=True, help="Which split to visualize", type=str)
parser.add_argument('--index', required=True, help="Which substrate?", type=str)
args = parser.parse_args()

split = args.split
model = 'MESI'
csv_file_path = f'./models/{model}/fold_{split}.csv'
save_dirs = './'
df = pd.read_csv(csv_file_path)

atom_featurizer = CanonicalAtomFeaturizer()
bond_featurizer = CanonicalBondFeaturizer(self_loop=True)

id_index = int(args.index)
smiles = df.iloc[id_index-1]['SMILES']
att_d_str = df.at[id_index-1, 'att_d']
attention_scores = np.array(ast.literal_eval(att_d_str)).squeeze()

graph = smiles_to_bigraph(
    smiles=smiles,
    node_featurizer=atom_featurizer,
    edge_featurizer=bond_featurizer,
    add_self_loop=True
)
num_atoms = graph.number_of_nodes()

attention_scores = attention_scores[:num_atoms]

vmin, vmax = min(attention_scores), max(attention_scores)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
print(attention_scores)

def truncate_colormap(cmap, minval=0.0, maxval=0.8, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

original_cmap = cm.get_cmap('cool')
colormap = truncate_colormap(original_cmap, 0.0, 1)
colors = [colormap(norm(score))[:3] for score in attention_scores]

img_size = 600

drawer = rdMolDraw2D.MolDraw2DCairo(img_size, img_size)
options = drawer.drawOptions()
options.bondLineWidth = 4

mol = Chem.MolFromSmiles(smiles)
drawer.DrawMolecule(mol)
drawer.FinishDrawing()

img_data = drawer.GetDrawingText()

img = Image.open(io.BytesIO(img_data))

attention_img = Image.new('RGBA', img.size, (255, 255, 255, 0))
draw = ImageDraw.Draw(attention_img, 'RGBA')

for i, color in enumerate(colors):
    atom_coords = drawer.GetDrawCoords(i)
    if atom_coords:
        x, y = atom_coords
        r, g, b = [int(c * 255) for c in color]
        for offset, alpha in zip([16, 12, 8], [40, 70, 90]):
            draw.ellipse((x-offset, y-offset, x+offset, y+offset), fill=(r, g, b, alpha))

attention_img = attention_img.filter(ImageFilter.GaussianBlur(radius=6))  # 模糊半径为4

img = Image.alpha_composite(img.convert('RGBA'), attention_img)

img.save(os.path.join(save_dirs, f'{args.split}_{args.index}_substrate_attention_{model}.png'))
