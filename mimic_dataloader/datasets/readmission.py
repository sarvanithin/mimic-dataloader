from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from mimic_dataloader.datasets.base import MimicDataset
from mimic_dataloader.utils.features import extract_time_series_features


class ReadmissionDataset(MimicDataset):
    """
    Predicts 30-day unplanned hospital readmission.
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
             mimic_dir: Path to the MIMIC-IV dataset directory.
             split: 'train', 'val', or 'test'.
             seq_length: Sequence padding length.
             transform: Transforms to apply.
        """
        super().__init__(
            mimic_dir=mimic_dir, split=split, seq_length=seq_length, transform=transform
        )

    def _extract_features_and_labels(self):
        """
        Loads admissions marking a binary target = 1 if the patient was readmitted
        within 30 days of the discharge of the index admission.
        """
        if self.cohort_df.empty:
            return
            
        stay_ids = self.cohort_df['stay_id'].tolist()
        features_df = extract_time_series_features(self.mimic_dir, stay_ids, freq="1H")
        
        # We need all admissions for patients in the cohort to calculate readmission risk
        all_patients = self.cohort_df['subject_id'].unique()
        # Ensure 'admittime' and 'dischtime' are sorted per patient
        cohort_df_sorted = self.cohort_df.copy()
        cohort_df_sorted.sort_values(by=['subject_id', 'admittime'], inplace=True)
        
        for _, row in cohort_df_sorted.iterrows():
            stay_id = row['stay_id']
            subject_id = row['subject_id']
            dischtime = row['dischtime']
            
            # Find next admission for this patient
            patient_adms = self.cohort_df[self.cohort_df['subject_id'] == subject_id]
            next_adm = patient_adms[patient_adms['admittime'] > dischtime].head(1)
            
            label = 0
            if not next_adm.empty:
                time_to_next = (next_adm['admittime'].iloc[0] - dischtime).total_seconds() / (3600 * 24)
                if time_to_next <= 30:
                    label = 1
                    
            if stay_id in features_df.index.get_level_values('stay_id'):
                stay_features = features_df.loc[stay_id].values
                max_len = self.seq_length if self.seq_length else stay_features.shape[0]
                stay_features = stay_features[:max_len, :]
            else:
                stay_features = np.zeros((self.seq_length or 48, len(features_df.columns)))
                
            self.data.append(stay_features)
            self.labels.append(label)
            self.patient_ids.append(subject_id)
            self.hadm_ids.append(row['hadm_id'])
