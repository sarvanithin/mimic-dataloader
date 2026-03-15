from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from mimic_dataloader.datasets.base import MimicDataset
from mimic_dataloader.utils.features import extract_time_series_features


class MortalityDataset(MimicDataset):
    """
    Predicts in-hospital mortality based on the first 48 hours of ICU data.
    """

    def __init__(
        self,
        mimic_dir: Union[str, Path],
        split: str = "train",
        seq_length: Optional[int] = 48,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            mimic_dir: Path to the MIMIC-IV dataset directory.
            split: 'train', 'val', or 'test'.
            seq_length: Sequence length (typically 48 for the first 48 hours).
            transform: Transforms to apply.
        """
        super().__init__(
            mimic_dir=mimic_dir, split=split, seq_length=seq_length, transform=transform
        )

    def _extract_features_and_labels(self):
        """
        Loads patient stays, extracts a 48-hour sequence window from chartevents, 
        and defines the target as the binary mortality indicator (hospital_expire_flag).
        """
        if self.cohort_df.empty:
            return
            
        # Get list of qualifying ICUstays
        stay_ids = self.cohort_df['stay_id'].tolist()
        
        # Extract features for these stays (this function returns a MultiIndex df)
        features_df = extract_time_series_features(self.mimic_dir, stay_ids, freq="1H")
        
        # Process each stay into a sequence tensor and label
        for _, row in self.cohort_df.iterrows():
            stay_id = row['stay_id']
            label = int(row['hospital_expire_flag'])
            
            # Extract sequence for this stay
            if stay_id in features_df.index.get_level_values('stay_id'):
                stay_features = features_df.loc[stay_id].values
                # We want up to seq_length hours (typically 48)
                max_len = self.seq_length if self.seq_length else stay_features.shape[0]
                stay_features = stay_features[:max_len, :]
            else:
                # If no features exist, pad with zeros
                stay_features = np.zeros((self.seq_length or 48, len(features_df.columns)))
                
            self.data.append(stay_features)
            self.labels.append(label)
            self.patient_ids.append(row['subject_id'])
            self.hadm_ids.append(row['hadm_id'])
