from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch

from mimic_dataloader.datasets.base import MimicDataset
from mimic_dataloader.utils.features import extract_time_series_features


class PhenotypingDataset(MimicDataset):
    """
    Multi-label classification of 25 common clinical conditions (ICD-based phenotypes).
    """

    # Common subset of 25 phenotypes defined in Harutyunyan et al. 2019
    PHENOTYPES = [
        "Acute and unspecified renal failure",
        "Acute cerebrovascular disease",
        "Acute myocardial infarction",
        "Cardiac dysrhythmias",
        "Chronic kidney disease",
        "Chronic obstructive pulmonary disease and bronchiectasis",
        "Complications of surgical procedures or medical care",
        "Conduction disorders",
        "Congestive heart failure; nonhypertensive",
        "Coronary atherosclerosis and other heart disease",
        "Diabetes mellitus with complications",
        "Diabetes mellitus without complication",
        "Disorders of lipid metabolism",
        "Essential hypertension",
        "Fluid and electrolyte disorders",
        "Gastrointestinal hemorrhage",
        "Hypertension with complications and secondary hypertension",
        "Other liver diseases",
        "Other lower respiratory disease",
        "Other upper respiratory disease",
        "Pleurisy; pneumothorax; pulmonary collapse",
        "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
        "Respiratory failure; insufficiency; arrest (adult)",
        "Septicemia (except in labor)",
        "Shock"
    ]

    def __init__(
        self,
        mimic_dir: Union[str, Path],
        split: str = "train",
        seq_length: Optional[int] = 48,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            mimic_dir=mimic_dir, split=split, seq_length=seq_length, transform=transform
        )

    def _extract_features_and_labels(self):
        """
        Loads entire admissions to predict the 25 distinct multi-label morbidities.
        """
        if self.cohort_df.empty:
            return
            
        stay_ids = self.cohort_df['stay_id'].tolist()
        features_df = extract_time_series_features(self.mimic_dir, stay_ids, freq="1H")
        num_classes = len(self.PHENOTYPES)
        
        for _, row in self.cohort_df.iterrows():
            stay_id = row['stay_id']
            # Mock phenotype multi-hot array until diagnoses_icd joins are integrated
            label = np.random.randint(0, 2, size=num_classes)
            
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
        """Override since label is an array of multi-hot values, return float tensor"""
        features_tensor, label_tensor = super().__getitem__(idx)
        return features_tensor, label_tensor.to(torch.float32)

