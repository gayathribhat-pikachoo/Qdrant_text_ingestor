import pandas as pd
import pickle
import os

def load_csv(file_path):
    """
    Load a CSV file and return a list of dictionaries.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")
    
    df = pd.read_csv(file_path)
    return df.to_dict('records')

def load_pickle(file_path):
    """
    Load a pickle file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")
        
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def batch_list(input_list, batch_size):
    """
    Split a list into smaller batches.
    """
    return [input_list[i : i + batch_size] for i in range(0, len(input_list), batch_size)]
