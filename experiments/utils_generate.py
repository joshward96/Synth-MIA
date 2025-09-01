from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

def train_sample_bn(dataset):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("bayesian_network")
    syn_model.fit(loader)
    return syn_model

def train_sample_ddpm(dataset):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("ddpm",n_iter=4000)
    syn_model.fit(loader)
    return syn_model

def train_sample_great(dataset):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("great")
    syn_model.fit(loader)
    return syn_model

def train_sample_arf(dataset):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("arf")
    syn_model.fit(loader)
    return syn_model

def train_sample_tvae(dataset):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("tvae")
    syn_model.fit(loader)
    return syn_model

def train_sample_ctgan(dataset):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("ctgan")
    syn_model.fit(loader)
    return syn_model

def train_sample_nflows(dataset):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("nflow")
    syn_model.fit(loader)
    return syn_model

def train_sample_adsgan(dataset):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("adsgan")
    syn_model.fit(loader)
    return syn_model

def train_sample_pategan(dataset):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("pategan")
    syn_model.fit(loader)
    return syn_model

import numpy as np
import pandas as pd
import torch
import os
import time

import autodiff.process_GQ as pce
import autodiff.autoencoder as ae
import autodiff.diffusion as diff
import autodiff.TabDDPMdiff as TabDiff
def train_sample_autodiff(dataset: np.ndarray):
    # Convert numpy array to DataFrame for consistency with existing code
    real_df = pd.DataFrame(dataset)

    # Set parameters
    threshold = 0.01  # Threshold for mixed-type variables
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs = 2000
    eps = 1e-5
    weight_decay = 1e-6
    maximum_learning_rate = 1e-2
    lr = 2e-4
    hidden_size = 250
    num_layers = 3
    batch_size = 50
    diff_n_epochs = 2000
    hidden_dims = (256, 512, 1024, 512, 256)
    T = 100
    sigma = 20
    num_batches_per_epoch = 50
    N = len(real_df)
    # Preprocess and parse the data
    parser = pce.DataFrameParser().fit(real_df, threshold)

    # Train the autoencoder and get latent features
    ds = ae.train_autoencoder(real_df, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold)
    latent_features = ds[1].detach()

    # Define converted table dimensions based on latent features
    converted_table_dim = latent_features.shape[1]
    print("Latent feature shape:",latent_features.shape)

    # Train the diffusion model on the latent features
    score = TabDiff.train_diffusion(latent_features, T, eps, sigma, lr,
                                    num_batches_per_epoch, maximum_learning_rate, weight_decay, diff_n_epochs, batch_size)

    # Generate synthetic samples using Euler-Maruyama sampling
    T_sampling = 300  # Sampling time steps
    P = latent_features.shape[1]
    
    start_time = time.time()
    sample = diff.Euler_Maruyama_sampling(score, T_sampling, N, P, device)
    end_time = time.time()
    
    # Convert generated samples back to the original table format
    print("Sample shape:",sample.shape)
    gen_output = ds[0](sample, ds[2], ds[3])
    syn_df = pce.convert_to_table(real_df, gen_output, threshold)

    # Print the sampling duration
    print(f"Sampling duration: {end_time - start_time} seconds")

    return syn_df

def train_autodiff(dataset: np.ndarray):
    # Convert numpy array to DataFrame for consistency with existing code
    real_df = pd.DataFrame(dataset)

    # Set parameters
    threshold = 0.01  # Threshold for mixed-type variables
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs = 20
    eps = 1e-5
    weight_decay = 1e-6
    maximum_learning_rate = 1e-2
    lr = 2e-4
    hidden_size = 250
    num_layers = 3
    batch_size = 50
    diff_n_epochs = 20
    hidden_dims = (256, 512, 1024, 512, 256)
    T = 100
    sigma = 20
    num_batches_per_epoch = 50

    # Preprocess and parse the data
    parser = pce.DataFrameParser().fit(real_df, threshold)

    # Train the autoencoder and get latent features
    ds = ae.train_autoencoder(real_df, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold)
    latent_features = ds[1].detach()

    # Define converted table dimensions based on latent features
    converted_table_dim = latent_features.shape[1]
    print("Latent feature shape:",latent_features.shape)

    # Train the diffusion model on the latent features
    score = TabDiff.train_diffusion(latent_features, T, eps, sigma, lr,
                                    num_batches_per_epoch, maximum_learning_rate, weight_decay, diff_n_epochs, batch_size)


    return ds, score


def sample_autodiff(dataset, ds, score, N):
    threshold = 0.01  # Threshold for mixed-type variables

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    real_df = pd.DataFrame(dataset)

    latent_features = ds[1].detach()
    # Generate synthetic samples using Euler-Maruyama sampling
    T_sampling = 300  # Sampling time steps
    P = latent_features.shape[1]
    
    start_time = time.time()
    sample = diff.Euler_Maruyama_sampling(score, 
                                          T_sampling, 
                                          N, 
                                          P, 
                                          device)
    end_time = time.time()
    
    # Convert generated samples back to the original table format
    print("Sample shape:",sample.shape)
    gen_output = ds[0](sample, ds[2], ds[3])
    syn_df = pce.convert_to_table(real_df, gen_output, threshold)

    # Print the sampling duration
    print(f"Sampling duration: {end_time - start_time} seconds")
    return syn_df

import subprocess
import numpy as np 
import pandas as pd
import os
from TabSyn.utils_gen_mia import create_dataset_with_metadata, infer_task_type
from TabSyn.process_dataset import process_data

def train_sample_tabsyn(dataset, dataset_name="test_dataset", save_path="./data/synthetic_data.csv"):
    """
    Runs the Tabsyn training and synthesis pipeline for a specified dataset.
    
    Args:
    - dataset (np.ndarray): The input dataset to be used.
    - N (int): Unused parameter, could be removed or repurposed as needed.
    - dataset_name (str): The name of the dataset to be used for training and synthesis.
    - save_path (str): The path to save the synthesized data temporarily.

    Returns:
    - pd.DataFrame: The synthetic data as a DataFrame.
    """
    # Save the current working directory
    original_dir = os.getcwd()
    
    # Call the Tabsyn functions to create the dataset with metadata
    
    try:
        # Change to the TabSyn directory
        os.chdir("TabSyn")

        task_type = infer_task_type(dataset)
        print(dataset)
        create_dataset_with_metadata(dataset, dataset_name, task_type)
        process_data(dataset_name)

        print("Current work dir:",os.getcwd())
        import importlib
        import sys
        sys.path.append('~/TabSyn')
        # Step 1: Train the VAE model
        subprocess.run([
            "python", "main.py",
            "--dataname", dataset_name,
            "--method", "vae",
            "--mode", "train"
        ], check=True)
        
        # Step 2: Train the diffusion model
        subprocess.run([
            "python", "main.py",
            "--dataname", dataset_name,
            "--method", "tabsyn",
            "--mode", "train"
        ], check=True)

        print("Diffusion training completed!")
        
        # Step 3: Synthesize data using the trained model
        subprocess.run([
            "python", "main.py",
            "--dataname", dataset_name,
            "--method", "tabsyn",
            "--mode", "sample",
            "--save_path", save_path
        ], check=True)
        
        # Step 4: Load the synthetic data
        synthetic_df = pd.read_csv(save_path)
        
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during subprocess execution: {e}")
        return None
    
    finally:
        
        # Clean up by deleting the synthesized data file after loading
        if os.path.exists(save_path):
            os.remove(save_path)
            
        # Ensure returning to the original working directory
        os.chdir(original_dir)
    
    return synthetic_df

def train_tabsyn(dataset, dataset_name="test_dataset"):
    """
    Trains the Tabsyn models (VAE and Diffusion) for a specified dataset.
   
    Args:
    - dataset (np.ndarray): The input dataset to be used for training.
    - dataset_name (str): The name of the dataset to be used for training.
   
    Returns:
    - bool: True if training was successful, False otherwise.
    """
    # Save the current working directory
    original_dir = os.getcwd()
   
    try:
        # Change to the TabSyn directory
        os.chdir("TabSyn")
        
        # Prepare the dataset
        task_type = infer_task_type(dataset)
        create_dataset_with_metadata(dataset, dataset_name, task_type)
        process_data(dataset_name)
        
        # Step 1: Train the VAE model
        vae_result = subprocess.run([
            "python", "main.py",
            "--dataname", dataset_name,
            "--method", "vae",
            "--mode", "train"
        ], check=True)
       
        # Step 2: Train the diffusion model
        diffusion_result = subprocess.run([
            "python", "main.py",
            "--dataname", dataset_name,
            "--method", "tabsyn",
            "--mode", "train"
        ], check=True)
        
        print("Tabsyn training completed successfully!")
        return True
   
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during training: {e}")
        return False
   
    finally:
        # Ensure returning to the original working directory
        os.chdir(original_dir)

def sample_tabsyn(N, dataset_name="test_dataset", save_path="./data/synthetic_data.csv"):
    """
    Samples synthetic data using a previously trained Tabsyn model.
   
    Args:
    - dataset_name (str): The name of the dataset used during training.
    - save_path (str): The path to save the synthesized data.
   
    Returns:
    - pd.DataFrame: The synthetic data as a DataFrame, or None if sampling fails.
    """
    # Save the current working directory
    original_dir = os.getcwd()
   
    try:
        # Change to the TabSyn directory
        os.chdir("TabSyn")
        
        # Step 3: Synthesize data using the trained model
        subprocess.run([
            "python", "main.py",
            "--dataname", dataset_name,
            "--sample_size", N,
            "--method", "tabsyn",
            "--mode", "sample",
            "--save_path", save_path
        ], check=True)
       
        # Step 4: Load the synthetic data
        synthetic_df = pd.read_csv(save_path)
       
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during data sampling: {e}")
        return None
   
    finally:
        # Clean up by deleting the synthesized data file after loading
        if os.path.exists(save_path):
            os.remove(save_path)
           
        # Ensure returning to the original working directory
        os.chdir(original_dir)
   
    return synthetic_df
if __name__ == "__main__":
    import numpy as np

    # Generate a toy dataset with 100 rows and 3 columns
    toy_data = np.random.rand(100, 3)

    # Desired number of synthetic rows
    N = 100

    # Call the function to generate synthetic data
    synthetic_df = train_sample_tabsyn(toy_data, N)

    # Print the generated synthetic data
    print(synthetic_df)