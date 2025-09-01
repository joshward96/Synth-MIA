import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_fake_data(n_num, n_cat, n_rows, chunk_size=100000, missing_percent=0):
    """
    Generates 4 separate fake DataFrames (mem, non_mem, ref, synth) with specified parameters.

    Args:
        n_num (int): Number of numerical columns to generate.
        n_cat (int): Number of categorical columns to generate.
        n_rows (int): Total number of rows in the generated DataFrame.
        chunk_size (int, optional): Number of rows per chunk to avoid memory issues. Default is 100000.
        missing_percent (float, optional): Percentage of missing values to introduce in the DataFrame. Default is 0.

    Returns:
        tuple: 4 separate DataFrames (mem, non_mem, ref, synth).
    """
    num_chunks = n_rows // chunk_size + (1 if n_rows % chunk_size != 0 else 0)
    
    def generate_single_copy():
        """Generates a single copy of the fake data."""
        chunks = []
        for chunk_id in tqdm(range(num_chunks)):
            n_rows_chunk = min(chunk_size, n_rows - chunk_id * chunk_size)
            
            data = {}
            for i in range(n_num):
                data[f'num_{i}'] = np.random.rand(n_rows_chunk)
            for i in range(n_cat):
                data[f'cat_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], size=n_rows_chunk)
            
            chunk_df = pd.DataFrame(data)

            if missing_percent > 0:
                # Introduce missing values
                total_cells = n_rows_chunk * (n_num + n_cat)
                num_missing = int(total_cells * missing_percent / 100)
                missing_indices = (
                    np.random.choice(n_rows_chunk, num_missing),
                    np.random.choice(n_num + n_cat, num_missing)
                )
                chunk_df.values[missing_indices] = np.nan

            chunks.append(chunk_df)
        return pd.concat(chunks, ignore_index=True)
    
    # Generate 4 separate copies
    mem = generate_single_copy()
    non_mem = generate_single_copy()
    ref = generate_single_copy()
    synth = generate_single_copy()
    
    return mem, non_mem, ref, synth


def load_data(folder):
    # Extract dataset name from folder name
    mem = pd.read_csv(os.path.join(folder, "mem.csv"))
    non_mem = pd.read_csv(os.path.join(folder, "non_mem.csv"))
    ref = pd.read_csv(os.path.join(folder, "ref.csv"))
    synth = pd.read_csv(os.path.join(folder, "synth.csv"))
    return mem, non_mem, ref, synth

# Helper function to list subfolders in example_data directory that have the required CSV files
def get_subfolders(parent_folder):
    """Get subfolders that contain the required CSV files for testing."""
    valid_folders = []
    required_files = ['mem.csv', 'non_mem.csv', 'ref.csv', 'synth.csv']
    
    for subfolder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(folder_path):
            # Check if all required files exist
            if all(os.path.exists(os.path.join(folder_path, file)) for file in required_files):
                valid_folders.append(folder_path)
    
    return valid_folders
