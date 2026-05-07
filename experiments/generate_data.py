import numpy as np
import os
import sys
import gc
import signal
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import pandas as pd
from utils_generate import *

# Define seeds and models
models = {
    "ddpm": train_sample_ddpm,
    "arf": train_sample_arf,
    "tvae": train_sample_tvae,
    "ctgan": train_sample_ctgan,
    "nflow": train_sample_nflows,
    "adsgan": train_sample_adsgan,
    "pategan_e1": train_sample_pategan,
    "tabsyn": train_sample_tabsyn,
    "rtf": train_sample_realtabformer,
}

# Data split seed (fixed for consistent splits)
data_split_seed = 42
# Model training/generation seeds
model_seeds = [1,2,3,4,5]

# Synthetic data generation multipliers
synth_multipliers = [1]  # 1x, 2x, 3x the size of member data

def timeout_model_training(model_func, mem_set, model_seed, timeout_minutes=1000):
    """Train model with timeout using threading"""
    result = {'model': None, 'error': None}
    
    def target_function():
        try:
            synth_model = model_func(mem_set, random_state=model_seed)
            result['model'] = synth_model
        except Exception as e:
            result['error'] = str(e)
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(target_function)
        try:
            future.result(timeout=timeout_minutes * 60)  # Convert minutes to seconds
            if result['error']:
                return None, result['error']
            return result['model'], "success"
        except TimeoutError:
            return None, "timeout"
        except Exception as e:
            return None, str(e)

# Get all dataset files from the data/ folder
data_folder = 'exp_data/'
data_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])

for data_file in data_files:
    try:
        # Load and preprocess data
        print(f"Processing dataset: {data_file}")
        df = pd.read_csv(os.path.join(data_folder, data_file))
        df = df.dropna()
        
        if len(df) < 10:  # Skip datasets that are too small
            print(f"Dataset {data_file} too small ({len(df)} rows), skipping...")
            continue

        # Create a folder for the dataset
        dataset_name = os.path.splitext(data_file)[0]
        dataset_folder = os.path.join('synth_mia_pategan_e10', dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)

        # Split data using fixed seed (80% train, 10% holdout, 10% ref)
        print(f"Splitting data for {dataset_name}...")
        np.random.seed(data_split_seed)
        shuffled_indices = np.random.permutation(len(df))
        
        # Calculate split sizes
        train_size = int(0.8 * len(df))
        holdout_size = int(0.1 * len(df))
        ref_size = len(df) - train_size - holdout_size  # Remaining goes to ref
        
        # Create splits
        mem_set = df.iloc[shuffled_indices[:train_size]]
        holdout_set = df.iloc[shuffled_indices[train_size:train_size + holdout_size]]
        ref_set = df.iloc[shuffled_indices[train_size + holdout_size:train_size + holdout_size + ref_size]]
        
        print(f"Data split - Member: {len(mem_set)}, Holdout: {len(holdout_set)}, Reference: {len(ref_set)}")
        
        # Save the split datasets
        mem_set.to_csv(os.path.join(dataset_folder, 'mem_set.csv'), index=False)
        holdout_set.to_csv(os.path.join(dataset_folder, 'holdout_set.csv'), index=False)
        ref_set.to_csv(os.path.join(dataset_folder, 'ref_set.csv'), index=False)

        # For each model
        for model_name, model_func in models.items():
            print(f"\nProcessing model: {model_name}")
            
            # Create model folder
            model_folder = os.path.join(dataset_folder, model_name)
            os.makedirs(model_folder, exist_ok=True)
            
            # For each model seed
            for model_seed in model_seeds:
                print(f"  Using model seed: {model_seed}")
                
                # Create seed folder
                seed_folder = os.path.join(model_folder, f'seed_{model_seed}')
                
                # Check if this seed is already completed
                if os.path.exists(seed_folder):
                    # Check if all multipliers are completed
                    all_completed = True
                    for multiplier in synth_multipliers:
                        synth_file = os.path.join(seed_folder, f'synth_{multiplier}x.csv')
                        if not os.path.exists(synth_file):
                            all_completed = False
                            break
                    
                    if all_completed:
                        print(f"    All synthetic data already exists for seed {model_seed}, skipping...")
                        continue
                
                os.makedirs(seed_folder, exist_ok=True)
                
                try:
                    # Set seed for model training/generation
                    np.random.seed(model_seed)
                    
                    print(f"    Training model with timeout (30 mins)...")
                    start_time = time.time()
                    
                    # Train model with timeout, passing the model seed
                    synth_model, status = timeout_model_training(model_func, mem_set, model_seed, timeout_minutes=1000)
                    
                    training_time = time.time() - start_time
                    
                    if status == "timeout":
                        print(f"    Model {model_name} timed out after 30 minutes, skipping...")
                        # Create a timeout marker file
                        with open(os.path.join(seed_folder, 'timeout_marker.txt'), 'w') as f:
                            f.write(f"Model training timed out after 30 minutes\nSeed: {model_seed}\n")
                        continue
                    elif status != "success":
                        print(f"    Model {model_name} failed to train: {status}")
                        # Create an error marker file
                        with open(os.path.join(seed_folder, 'error_marker.txt'), 'w') as f:
                            f.write(f"Model training failed: {status}\nSeed: {model_seed}\n")
                        continue
                    
                    print(f"    Model trained successfully in {training_time:.2f} seconds")
                    
                    # Generate synthetic data for each multiplier
                    for multiplier in synth_multipliers:
                        synth_file = os.path.join(seed_folder, f'synth_{multiplier}x.csv')
                        
                        # Skip if already exists
                        if os.path.exists(synth_file):
                            print(f"    Synthetic data {multiplier}x already exists, skipping...")
                            continue
                        
                        try:
                            print(f"    Generating {multiplier}x synthetic data...")
                            synth_size = len(mem_set) * multiplier
                            synth = synth_model.generate(synth_size).dataframe()
                            
                            # Save synthetic data
                            synth.to_csv(synth_file, index=False)
                            print(f"    Saved {multiplier}x synthetic data ({len(synth)} rows) to: {synth_file}")
                            
                        except Exception as e:
                            print(f"    Error generating {multiplier}x synthetic data: {e}")
                            # Create error marker for this specific generation
                            error_file = os.path.join(seed_folder, f'generation_error_{multiplier}x.txt')
                            with open(error_file, 'w') as f:
                                f.write(f"Error generating {multiplier}x data: {str(e)}\nSeed: {model_seed}\n")
                    
                    # Save training info
                    info_file = os.path.join(seed_folder, 'training_info.txt')
                    with open(info_file, 'w') as f:
                        f.write(f"Model: {model_name}\n")
                        f.write(f"Seed: {model_seed}\n")
                        f.write(f"Training time: {training_time:.2f} seconds\n")
                        f.write(f"Member set size: {len(mem_set)}\n")
                        f.write(f"Generated multipliers: {synth_multipliers}\n")
                    
                    # Clean up memory
                    del synth_model
                    gc.collect()
                    
                except Exception as e:
                    print(f"    Error processing model {model_name} with seed {model_seed}: {e}")
                    # Create error marker file
                    error_file = os.path.join(seed_folder, 'model_error.txt')
                    os.makedirs(seed_folder, exist_ok=True)
                    with open(error_file, 'w') as f:
                        f.write(f"Error processing model: {str(e)}\nSeed: {model_seed}\n")

    except Exception as e:
        print(f"Error processing dataset {data_file}: {e}")

print("\nProcessing complete!")
print("\nDirectory structure:")
print("├── dataset_name/")
print("│   ├── mem_set.csv")
print("│   ├── holdout_set.csv")
print("│   ├── ref_set.csv")
print("│   └── model_name/")
print("│       └── seed_X/")
print("│           ├── synth_1x.csv")
print("│           ├── synth_2x.csv")
print("│           ├── synth_3x.csv")
print("│           └── training_info.txt")