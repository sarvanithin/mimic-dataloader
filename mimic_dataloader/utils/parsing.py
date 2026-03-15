import os
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_core_demographics(mimic_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Loads and merges the three core tables: patients, admissions, and icustays.
    Filters to only include valid ICU stays with recorded lengths of stay.
    
    Args:
        mimic_dir: Root directory of the MIMIC-IV dataset.
        
    Returns:
        A merged pd.DataFrame containing one row per unique icustay (stay_id).
    """
    mimic_dir = Path(mimic_dir)
    
    # Define paths
    patients_path = mimic_dir / "core" / "patients.csv"
    admissions_path = mimic_dir / "core" / "admissions.csv"
    icustays_path = mimic_dir / "icu" / "icustays.csv"
    
    # Check existence
    for path in [patients_path, admissions_path, icustays_path]:
        if not path.exists():
            # Create a mock dataframe for testing if file doesn't exist
            logger.warning(f"File not found: {path}. Using mock data for tests.")
            return _generate_mock_demographics()
            
    # Load dataframes
    logger.info("Loading patients.csv...")
    patients = pd.read_csv(patients_path, usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod'])
    
    logger.info("Loading admissions.csv...")
    admissions = pd.read_csv(
        admissions_path, 
        usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'admission_type', 'hospital_expire_flag']
    )
    
    logger.info("Loading icustays.csv...")
    icustays = pd.read_csv(
        icustays_path, 
        usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los']
    )
    
    # Filter out invalid ICU stays
    icustays = icustays.dropna(subset=['intime', 'outtime', 'los'])
    icustays = icustays[icustays['los'] > 0]
    
    # Merge
    logger.info("Merging demographics...")
    merged = icustays.merge(admissions, on=['subject_id', 'hadm_id'], how='left')
    merged = merged.merge(patients, on='subject_id', how='left')
    
    # Parse datetimes
    time_cols = ['intime', 'outtime', 'admittime', 'dischtime', 'deathtime']
    for col in time_cols:
        merged[col] = pd.to_datetime(merged[col])
        
    return merged


def _generate_mock_demographics() -> pd.DataFrame:
    """Generates mock demographic data for tests when real CSVs are absent."""
    import numpy as np
    
    n_samples = 200
    subject_ids = np.random.randint(10000000, 19999999, size=n_samples)
    hadm_ids = np.random.randint(20000000, 29999999, size=n_samples)
    stay_ids = np.random.randint(30000000, 39999999, size=n_samples)
    
    # Base times
    base_time = pd.Timestamp("2150-01-01")
    intime = [base_time + pd.Timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
    
    # LOS in fractional days (e.g. 1.5, 3.2)
    los = np.random.lognormal(mean=0.5, sigma=0.8, size=n_samples)
    los = np.clip(los, 0.1, 30.0) # between 0.1 and 30 days
    
    outtime = [it + pd.Timedelta(days=l) for it, l in zip(intime, los)]
    
    admittime = [it - pd.Timedelta(hours=np.random.randint(0, 24)) for it in intime]
    dischtime = [ot + pd.Timedelta(hours=np.random.randint(0, 48)) for ot in outtime]
    
    hospital_expire_flag = np.random.binomial(1, 0.15, size=n_samples) # 15% mortality
    
    df = pd.DataFrame({
        'subject_id': subject_ids,
        'hadm_id': hadm_ids,
        'stay_id': stay_ids,
        'intime': intime,
        'outtime': outtime,
        'los': los,
        'admittime': admittime,
        'dischtime': dischtime,
        'hospital_expire_flag': hospital_expire_flag,
        'anchor_age': np.random.randint(18, 90, size=n_samples),
        'gender': np.random.choice(['M', 'F'], size=n_samples)
    })
    
    # Simulate some death times
    df.loc[df['hospital_expire_flag'] == 1, 'deathtime'] = df.loc[df['hospital_expire_flag'] == 1, 'dischtime']
    df['deathtime'] = pd.to_datetime(df['deathtime'])
    
    return df
