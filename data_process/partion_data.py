import pandas as pd
from sklearn.model_selection import train_test_split
import os
dataset_list = ['DLKcat', 'GraphKM', 'MPEK_kcat', 'MPEK_km']
seed_list = [1, 2, 3, 4, 5]

output_folder = "datasets_embeddings" #align with 'embedding_5fold.py'

for dataset in dataset_list:
    for seed in seed_list:
        save_folder = f'../{output_folder}/{dataset}_{seed}'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        csv_file_path = f'../datasets/{dataset}/{dataset}_650m.csv'
        data = pd.read_csv(csv_file_path)

        if dataset == "GraphKM":
            s1, s2 = 1/5, 1/8
        else:
            s1, s2 = 1/10, 1/9

        train_set, test_set = train_test_split(data, test_size=s1, random_state=seed)
        train_set, val_set = train_test_split(train_set, test_size=s2, random_state=seed)
        
        train_set.to_csv(f'{save_folder}/train.csv', index=False)
        val_set.to_csv(f'{save_folder}/val.csv', index=False)
        test_set.to_csv(f'{save_folder}/test.csv', index=False)
        
        print(f"Dataset:{dataset} Fold:{seed} ready")
