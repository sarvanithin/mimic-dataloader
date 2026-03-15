import numpy as np
import pandas as pd
from typing import Tuple

def train_val_test_split_temporal(
    df: pd.DataFrame, 
    time_column: str,
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame temporally. 
    Crucial for clinical data to avoid predicting the past from the future.
    
    Args:
        df: The dataframe to split.
        time_column: The column name representing the timestamp/date to order by.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        
    Returns:
        Train df, Val df, Test df
    """
    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found in DataFrame.")
        
    # Sort chronologically to prevent future leakage
    df_sorted = df.sort_values(by=time_column).reset_index(drop=True)
    
    n_samples = len(df_sorted)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]
    
    return train_df, val_df, test_df
    

def patient_wise_split(
    df: pd.DataFrame, 
    patient_id_col: str = "subject_id",
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame such that all records for a given patient end up
    in only one of the splits. Prevents leakage where model sees patient A 
    in train and patient A again in test.
    """
    if patient_id_col not in df.columns:
        raise ValueError(f"Column '{patient_id_col}' not found in DataFrame.")
        
    unique_patients = df[patient_id_col].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_patients)
    
    n_patients = len(unique_patients)
    train_end = int(n_patients * train_ratio)
    val_end = train_end + int(n_patients * val_ratio)
    
    train_patients = set(unique_patients[:train_end])
    val_patients = set(unique_patients[train_end:val_end])
    test_patients = set(unique_patients[val_end:])
    
    train_df = df[df[patient_id_col].isin(train_patients)]
    val_df = df[df[patient_id_col].isin(val_patients)]
    test_df = df[df[patient_id_col].isin(test_patients)]
    
    return train_df, val_df, test_df
