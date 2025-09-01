import os
import json
import numpy as np
import pandas as pd

def create_dataset_with_metadata(np_array, dataset_name, task_type="binclass"):
    """
    Converts a NumPy array to a Pandas DataFrame, saves it to CSV, and generates
    a metadata JSON file as per the specified format.
    
    Args:
    - np_array (np.ndarray): The input NumPy array with data.
    - dataset_name (str): The name for the dataset and the directory.
    - task_type (str): Task type, "binclass" or "regression". Defaults to "binclass".
    """
    # Step 1: Convert the NumPy array to a DataFrame
    df = pd.DataFrame(np_array)
    
    # Step 2: Infer column types
   
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(exclude=['number']).columns
    
    
    # Define the target column index; you may need to adjust this based on specific requirements
    target_col_idx = [df.shape[1] - 1]  # Assuming the last column as target by default
    target_col = df.columns[-1]
    num_col_idx = [df.columns.get_loc(col) for col in num_cols if col != target_col] 
    cat_col_idx = [df.columns.get_loc(col) for col in cat_cols if col != target_col] 
    print(df.dtypes)
    
    # Step 3: Create the directory structure
    data_dir = f"./data/{dataset_name}"
    os.makedirs(data_dir, exist_ok=True)
    
    # Step 4: Save the DataFrame as a CSV file
    csv_path = f"{data_dir}/{dataset_name}.csv"
    df.to_csv(csv_path, index=False, header=True)
    
    # Step 5: Create metadata dictionary
    metadata = {
        "name": dataset_name,
        "task_type": task_type,
        "header": "infer",
        "column_names": None,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        "data_path": csv_path,
        "test_path": None
    }
    
    # Step 6: Save metadata as a JSON file
    info_dir = f"./data/Info"
    os.makedirs(info_dir, exist_ok=True)
    json_path = f"{info_dir}/{dataset_name}.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Dataset and metadata created successfully in {data_dir} and {info_dir}")

def infer_task_type(dataset):
    if isinstance(dataset, np.ndarray):
        return "regression"
    elif isinstance(dataset, pd.DataFrame):
        # Get the last column
        last_column = dataset.iloc[:, -1]
        
        # Check if the last column is numerical for regression
        if np.issubdtype(last_column.dtype, np.number):
            return "regression"
        
        # Check if the last column is object/string for classification tasks
        elif last_column.dtype == object or isinstance(last_column.iloc[0], str):
            unique_values_count = last_column.nunique()
            
            # If only 2 unique values, it’s binary classification
            if unique_values_count == 2:
                return "binclass"
            # Otherwise, it's multiclass classification
            else:
                return "multiclass"
    else:
        raise ValueError("Dataset type not supported. Please provide a numpy array or pandas DataFrame.")


import numpy as np

from .process_dataset import process_data

def main():
    # Create a sample NumPy array for testing
    # Example: 10 rows, 5 columns (3 numerical and 2 categorical)

    # Create the initial DataFrame
    np_array = np.array([
        [1.0, 2.5, 3.1, 0, 0],
        [2.1, 3.6, 1.5, 0, 1],
        [3.2, 1.4, 2.7, 0, 0],
        [4.5, 2.2, 3.3, 0, 0],
        [5.1, 3.1, 1.8, 1, 1],
        [6.3, 1.9, 2.4, 1, 0],
        [7.0, 2.7, 3.2, 1, 0],
        [8.6, 3.3, 1.6, 1, 1],
        [9.1, 1.8, 2.3, 1, 1],
        [10.5, 2.9, 3.0, 1, 1]
    ])

    #print("\nNumPy array:\n", np_array)
    np_array = pd.read_csv("/home/ubuntu/Gen_MIA/example_data/adult.csv")
    
    # Define the dataset name and task type
    dataset_name = "test_dataset"
    task_type = "binclass"  # You can set "regression" if needed
    
    # Call the function to create the dataset with metadata
    create_dataset_with_metadata(np_array, dataset_name, task_type)
    
    process_data(dataset_name)

# Run the test code only if this script is run directly
if __name__ == "__main__":
    main()
