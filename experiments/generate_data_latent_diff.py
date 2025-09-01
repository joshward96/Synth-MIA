import os
import sys
import gc
import numpy as np
import pandas as pd
from utils_generate import *
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Define seeds and models
models = {
    #"ddpm": train_sample_ddpm,
    #"arf": train_sample_arf,
    #"tvae": train_sample_tvae,
    #"ctgan": train_sample_ctgan,
    #"nflow": train_sample_nflows,
    #"adsgan": train_sample_adsgan,
    #"pategan": train_sample_pategan,
    "autodiff": train_sample_autodiff,
    "tabsyn": train_sample_tabsyn,
}
seeds = [10,11,12,13,14]

# Get all dataset files from the data/ folder
data_folder = 'data/raw/'
data_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])

for data_file in data_files:
    try:
        # Load and preprocess data
        df = pd.read_csv(os.path.join(data_folder, data_file))
        df = df.dropna()

        # Create a folder for the dataset which is the name without the .csv
        dataset_name = os.path.splitext(data_file)[0]
        dataset_folder = os.path.join('data/processed/', dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)

        for seed in seeds:
            try:
                print(f"Processing dataset {dataset_name} with seed {seed}")
                # Create a subfolder for each seed
                seed_folder = os.path.join(dataset_folder, f'seed_{seed}')
                os.makedirs(seed_folder, exist_ok=True)

                # Randomly split into 80/20 ratio based on seed
                np.random.seed(seed)
                shuffled_indices = np.random.permutation(len(df))
                split_idx = int(0.8 * len(df))
                mem_set = df.iloc[shuffled_indices[:split_idx]]
                holdout_set = df.iloc[shuffled_indices[split_idx:]]

                # Save the mem_set and holdout_set to the seed folder
                mem_set.to_csv(os.path.join(seed_folder, 'mem_set.csv'), index=False)
                holdout_set.to_csv(os.path.join(seed_folder, 'holdout_set.csv'), index=False)
                for model_name, model_func in models.items():
                    try:
                        print(f"Running model: {model_name} with seed {seed}")

                        # Create synth subfolder in the seed folder
                        synth_folder = os.path.join(seed_folder, 'synth', model_name)
                        os.makedirs(synth_folder, exist_ok=True)

                        # Generate synthetic data
                        synth= model_func(mem_set)
                        print(synth.head())

                        # Save the synthetic data to the synth folder labeled with the proportion
                        synth_file = os.path.join(
                            synth_folder, f'synth_{1:.2f}.csv')
                        synth.to_csv(synth_file, index=False)
                        gc.collect()
                    except Exception as e:
                        print(f"Error processing model {model_name} with seed {seed} for {data_file}: {e}")

            except Exception as e:
                print(f"Error processing dataset {dataset_name} with seed {seed}: {e}")

    except Exception as e:
        print(f"Error processing dataset {data_file}: {e}")
