import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mimic_dataloader.utils.parsing import load_core_demographics
from mimic_dataloader.utils.splits import patient_wise_split


class MimicDataset(Dataset):
    """
    Abstract Base Class for MIMIC-IV PyTorch Datasets.
    
    This class standardizes data loading, split management, and feature sequence padding 
    to prevent temporal information leakage across train/val/test boundaries.
    """

    def __init__(
        self,
        mimic_dir: Union[str, Path],
        split: str = "train",
        seq_length: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            mimic_dir: Root directory of the MIMIC-IV dataset (e.g. where core/ icu/ are).
            split: Dataset split ('train', 'val', 'test').
            seq_length: Fixed sequence length for temporal data padding/truncation.
                If None, sequences are returned in their original lengths.
            transform: Optional transform to be applied on a sample.
        """
        self.mimic_dir = Path(mimic_dir)
        self.split = split
        self.seq_length = seq_length
        self.transform = transform
        
        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"Split must be 'train', 'val', or 'test'. Got {split}")

        # Subclasses should populate these lists in _load_cohort()
        self.patient_ids: List[int] = []
        self.hadm_ids: List[int] = []
        
        self.data: List[np.ndarray] = []
        self.labels: List[Any] = []
        self._load_cohort()

    def _load_cohort(self):
        """
        Loads the core demographic cohort and applies standard train/val/test splits.
        Calls subclass to extract specific sequences and labels based on the split.
        """
        # Load unified demographic data
        df = load_core_demographics(self.mimic_dir)
        
        # Split it to prevent leakage (patient-wise by default)
        train_df, val_df, test_df = patient_wise_split(
            df, 
            patient_id_col="subject_id", 
            train_ratio=0.7, 
            val_ratio=0.15, 
            random_state=42
        )
        
        if self.split == "train":
            self.cohort_df = train_df
        elif self.split == "val":
            self.cohort_df = val_df
        elif self.split == "test":
            self.cohort_df = test_df
            
        self._extract_features_and_labels()

    def _extract_features_and_labels(self):
        """
        Using self.cohort_df, subclasses must extract the time-series features 
        and define the labels for their specific task.
        Must populate self.data and self.labels.
        """
        raise NotImplementedError("Subclasses must implement _extract_features_and_labels()")

    def _pad_or_truncate(self, sequence: np.ndarray) -> np.ndarray:
        """
        Pads or truncates a sequence to the fixed seq_length.
        Assumes sequence is (time_steps, features).
        Pads with zeros at the end. Truncates from the end.
        """
        if self.seq_length is None:
            return sequence
            
        time_steps, features = sequence.shape
        if time_steps >= self.seq_length:
            return sequence[:self.seq_length, :]
        else:
            pad_len = self.seq_length - time_steps
            padding = np.zeros((pad_len, features), dtype=sequence.dtype)
            return np.vstack((sequence, padding))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.data[idx]
        label = self.labels[idx]

        if self.seq_length is not None:
             features = self._pad_or_truncate(features)

        if self.transform:
            features = self.transform(features)

        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long) # Or float dependending on task

        return features_tensor, label_tensor
