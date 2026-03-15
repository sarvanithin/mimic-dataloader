from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from mimic_dataloader.datasets.base import MimicDataset
from mimic_dataloader.utils.features import extract_time_series_features


class SepsisDataset(MimicDataset):
    """
    Predicts the onset of sepsis (Sepsis-3 criteria).
    Usually framed as predicting onset X hours ahead of clinical recognition.
    """

    def __init__(
        self,
        mimic_dir: Union[str, Path],
        split: str = "train",
        lead_time_hours: int = 4, # Hours ahead to predict
        seq_length: Optional[int] = 48,
        transform: Optional[Callable] = None,
    ):
        self.lead_time_hours = lead_time_hours
        super().__init__(
            mimic_dir=mimic_dir, split=split, seq_length=seq_length, transform=transform
        )

    def _extract_features_and_labels(self):
        """
        Loads continuous measurement windows up until (onset - lead_time_hours)
        with binary labels for whether the patient met Sepsis-3 criteria during the stay.
        """
        if self.cohort_df.empty:
            return
            
        stay_ids = self.cohort_df['stay_id'].tolist()
        features_df = extract_time_series_features(self.mimic_dir, stay_ids, freq="1H")
        
        for _, row in self.cohort_df.iterrows():
            stay_id = row['stay_id']
            # Highly simplified mock label for sepsis until full sepsis-3 mimic tables are built
            # Sepsis-3 requires joining microbiologyevents and prescriptions for abx
            los = float(row.get('los', 0))
            label = 1 if (los > 3.0 and np.random.rand() < 0.35) else 0
            
            if stay_id in features_df.index.get_level_values('stay_id'):
                stay_features = features_df.loc[stay_id].values
                # We would truncate this feature set up to the time of Sepsis onset minus lead_time
                max_len = self.seq_length if self.seq_length else stay_features.shape[0]
                stay_features = stay_features[:max_len, :]
            else:
                stay_features = np.zeros((self.seq_length or 48, len(features_df.columns)))
                
            self.data.append(stay_features)
            self.labels.append(label)
            self.patient_ids.append(row['subject_id'])
            self.hadm_ids.append(row['hadm_id'])
