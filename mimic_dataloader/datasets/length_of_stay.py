from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch

from mimic_dataloader.datasets.base import MimicDataset
from mimic_dataloader.utils.features import extract_time_series_features


class LengthOfStayDataset(MimicDataset):
    """
    Predicts ICU length of stay.
    Can be configured for regression (predict hours) or classification (bins).
    """

    def __init__(
        self,
        mimic_dir: Union[str, Path],
        split: str = "train",
        task: str = "regression",  # Or 'classification' (10 bins)
        seq_length: Optional[int] = 48,
        transform: Optional[Callable] = None,
    ):
        self.task = task
        if self.task not in ["regression", "classification"]:
             raise ValueError(f"Task must be regression or classification. Got {task}")
             
        super().__init__(
            mimic_dir=mimic_dir, split=split, seq_length=seq_length, transform=transform
        )

    def _extract_features_and_labels(self):
        """
        Loads continuous measurement windows for Los, target is either days 
        or binned classification labels.
        """
        if self.cohort_df.empty:
            return
            
        stay_ids = self.cohort_df['stay_id'].tolist()
        features_df = extract_time_series_features(self.mimic_dir, stay_ids, freq="1H")
        
        for _, row in self.cohort_df.iterrows():
            stay_id = row['stay_id']
            # LOS in mimic icustays table is fractional days
            los_days = float(row['los'])
            
            if self.task == "regression":
                label = los_days
            else:
                # 10 bins: <1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-14, 14+
                bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, float('inf')]
                # digitize is 1-indexed for right edges, so -1 to get 0-index classes 0-9
                label = int(np.digitize(los_days, bins) - 1)
                label = min(max(label, 0), 9)

            if stay_id in features_df.index.get_level_values('stay_id'):
                stay_features = features_df.loc[stay_id].values
                stay_features = stay_features[:self.seq_length or 48, :]
            else:
                stay_features = np.zeros((self.seq_length or 48, len(features_df.columns)))
                
            self.data.append(stay_features)
            self.labels.append(label)
            self.patient_ids.append(row['subject_id'])
            self.hadm_ids.append(row['hadm_id'])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Override to ensure regression tasks yield float labels, not long"""
        features_tensor, label_tensor = super().__getitem__(idx)
        
        if self.task == "regression":
            label_tensor = label_tensor.to(torch.float32)
            
        return features_tensor, label_tensor
