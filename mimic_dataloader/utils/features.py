import os
from pathlib import Path
from typing import Union, List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# A standard set of common vital signs and labs and their MIMIC-IV itemids
# (Heart Rate, O2 Sat, SBP, DBP, MAP, Temp, Resp Rate)
DEFAULT_ITEMIDS = {
    220045: "Heart Rate",
    220210: "Respiratory Rate",
    220277: "O2 saturation pulseoxymetry",
    220179: "Non Invasive Blood Pressure systolic",
    220180: "Non Invasive Blood Pressure diastolic",
    220181: "Non Invasive Blood Pressure mean",
    223761: "Temperature Fahrenheit",
}

def extract_time_series_features(
    mimic_dir: Union[str, Path],
    stay_ids: List[int],
    itemids: Optional[dict] = None,
    freq: str = "1H"
) -> pd.DataFrame:
    """
    Parses chartevents to build a dense continuous feature representation 
    for the requested ICU stays.
    
    Args:
        mimic_dir: Root MIMIC-IV directory.
        stay_ids: List of valid stay_ids to extract for.
        itemids: Dictionary of {itemid: feature_name}. Defaults to common vitals.
        freq: Pandas frequency string (e.g. '1H' for hourly bins).
        
    Returns:
        DataFrame MultiIndexed by (stay_id, time_bucket) pivoted such that 
        columns are the feature names.
    """
    if itemids is None:
        itemids = DEFAULT_ITEMIDS
        
    mimic_dir = Path(mimic_dir)
    chartevents_path = mimic_dir / "icu" / "chartevents.csv"
    
    if not chartevents_path.exists():
        logger.warning(f"File not found: {chartevents_path}. Using mock time-series data.")
        return _generate_mock_chartevents(stay_ids, itemids, freq)
        
    logger.info(f"Loading features from chartevents for {len(stay_ids)} stays...")
    
    # In practice, chartevents is massive (30GB+). A robust implementation 
    # would iterative over chunks or use PySpark/Polars.
    # For dataloaders, it's typical to preprocess this aggressively once.
    # We load a simplified chunk logic.
    chunks = []
    chunk_iter = pd.read_csv(
        chartevents_path, 
        usecols=['stay_id', 'itemid', 'charttime', 'valuenum'],
        chunksize=1000000
    )
    
    allowed_items = list(itemids.keys())
    
    for chunk in chunk_iter:
        # Filter strictly
        chunk = chunk[chunk['stay_id'].isin(stay_ids)]
        chunk = chunk[chunk['itemid'].isin(allowed_items)]
        chunk.dropna(subset=['valuenum'], inplace=True)
        if not chunk.empty:
            chunks.append(chunk)
            
    if not chunks:
        # Return empty MultiIndex struct
        idx = pd.MultiIndex.from_tuples([], names=['stay_id', 'charttime'])
        return pd.DataFrame(columns=list(itemids.values()), index=idx)
        
    events = pd.concat(chunks, ignore_index=True)
    events['charttime'] = pd.to_datetime(events['charttime'])
    
    # Map itemids to readable names
    events['feature'] = events['itemid'].map(itemids)
    
    # Group into time buckets and aggregate
    # Using 'floor' allows binning
    events['time_bucket'] = events['charttime'].dt.floor(freq)
    
    # Pivot
    pivoted = events.pivot_table(
        index=['stay_id', 'time_bucket'],
        columns='feature',
        values='valuenum',
        aggfunc='mean' # Taking mean within the hour bucket
    )
    
    return pivoted

def _generate_mock_chartevents(
    stay_ids: List[int],
    itemids: dict,
    freq: str
) -> pd.DataFrame:
    """Generates a dense pivot table simulating binned chartevents."""
    records = []
    base_time = pd.Timestamp("2150-01-01")
    
    feature_names = list(itemids.values())
    
    for stay_id in stay_ids:
        # Random stay duration 24-120 hours
        duration_hours = np.random.randint(24, 120)
        
        # Generate buckets linearly
        time_buckets = [base_time + pd.Timedelta(hours=i) for i in range(duration_hours)]
        
        # Base vital sign means
        base_vitals = {
            "Heart Rate": np.random.normal(80, 10),
            "Respiratory Rate": np.random.normal(16, 3),
            "O2 saturation pulseoxymetry": np.random.normal(97, 2),
            "Non Invasive Blood Pressure systolic": np.random.normal(120, 15),
            "Non Invasive Blood Pressure diastolic": np.random.normal(80, 10),
            "Non Invasive Blood Pressure mean": np.random.normal(90, 10),
            "Temperature Fahrenheit": np.random.normal(98.6, 1.0),
        }
        
        for t in time_buckets:
            # Random walk variation
            row = {'stay_id': stay_id, 'time_bucket': t}
            for feat in feature_names:
                if feat in base_vitals:
                    val = base_vitals[feat] + np.random.normal(0, 5) # variance
                    # ~20% missing rate in clinical bins
                    if np.random.rand() > 0.2:
                        row[feat] = val
                else:
                    if np.random.rand() > 0.5:
                         row[feat] = np.random.randn()
            records.append(row)
            
    df = pd.DataFrame(records)
    if 'time_bucket' in df.columns:
        df.set_index(['stay_id', 'time_bucket'], inplace=True)
    return df
