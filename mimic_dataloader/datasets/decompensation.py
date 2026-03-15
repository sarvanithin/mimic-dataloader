from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch

from mimic_dataloader.datasets.base import MimicDataset
from mimic_dataloader.utils.features import extract_time_series_features


class DecompensationDataset(MimicDataset):
    """
    Predicts physiologic decompensation (e.g., mortality) dynamically 
    in the next 24 hours from any given time step.
    This is often framed as a sequence-to-sequence or continuous-prediction problem.
    """

    def __init__(
        self,
        mimic_dir: Union[str, Path],
        split: str = "train",
        window_hours: int = 24, # predicting decomp within the next 24h
        seq_length: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        self.window_hours = window_hours
        super().__init__(
            mimic_dir=mimic_dir, split=split, seq_length=seq_length, transform=transform
        )

    def _extract_features_and_labels(self):
        """
        Unlike static tasks, this loads sequences where every timestep has a label
        indicating if death occurs in the next 24 hours.
        """
        if self.cohort_df.empty:
            return
            
        stay_ids = self.cohort_df['stay_id'].tolist()
        features_df = extract_time_series_features(self.mimic_dir, stay_ids, freq="1H")
        
        for _, row in self.cohort_df.iterrows():
            stay_id = row['stay_id']
            expire_flag = int(row['hospital_expire_flag'])
            
            if stay_id in features_df.index.get_level_values('stay_id'):
                stay_features = features_df.loc[stay_id].values
                simulated_length = stay_features.shape[0]
                
                labels = np.zeros(simulated_length)
                if expire_flag == 1:
                    decomp_start = max(0, simulated_length - self.window_hours)
                    labels[decomp_start:] = 1
                    
                self.data.append(stay_features)
                self.labels.append(labels)
                self.patient_ids.append(row['subject_id'])
                self.hadm_ids.append(row['hadm_id'])

    def _pad_or_truncate_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Decompensation tasks have a label for EVERY timestep, so we must 
        pad labels similarly to how base class pads features.
        """
        if self.seq_length is None:
            return labels
            
        time_steps = len(labels)
        if time_steps >= self.seq_length:
            return labels[:self.seq_length]
        else:
            pad_len = self.seq_length - time_steps
            # Padding continuous labels typically with a non-contributing value (e.g. 0)
            # though in rigorous setups masking should be used in the loss function.
            padding = np.zeros(pad_len, dtype=labels.dtype) 
            return np.concatenate((labels, padding))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.data[idx]
        labels = self.labels[idx]

        if self.seq_length is not None:
             features = self._pad_or_truncate(features)
             labels = self._pad_or_truncate_labels(labels)

        if self.transform:
            features = self.transform(features)

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long) # Sequence of binary targets

        return features_tensor, label_tensor
