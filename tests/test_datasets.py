import pytest
import torch
from pathlib import Path

# Important: ensure pyproject.toml allows importing from mimic_dataloader
from mimic_dataloader.datasets import (
    MortalityDataset,
    ReadmissionDataset,
    LengthOfStayDataset,
    SepsisDataset,
    PhenotypingDataset,
    DecompensationDataset,
)

# Use dummy directory for tests
DUMMY_DIR = Path("/tmp/mimic_dummy")

def test_mortality_dataset():
    dataset = MortalityDataset(mimic_dir=DUMMY_DIR, split="train", seq_length=48)
    assert len(dataset) > 100 # Should be around 140
    
    features, label = dataset[0]
    # Check tensor properties
    assert isinstance(features, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    # Check sequence padding (mock feature dim is 7 now from features.py)
    assert features.shape == (48, 7)
    
    val_dataset = MortalityDataset(mimic_dir=DUMMY_DIR, split="val", seq_length=48)
    assert len(val_dataset) > 10 # Should be around 30

def test_readmission_dataset():
    # Test without padding (seq_length=None)
    dataset = ReadmissionDataset(mimic_dir=DUMMY_DIR, split="test", seq_length=None)
    assert len(dataset) > 10 # Should be around 30
    
    features, label = dataset[0]
    # Sequence length should be variable when seq_length is None
    assert len(features.shape) == 2 
    assert features.shape[1] == 7

def test_length_of_stay_dataset():
    # Test regression typing
    dataset = LengthOfStayDataset(mimic_dir=DUMMY_DIR, task="regression")
    features, label = dataset[0]
    assert label.dtype == torch.float32
    
    # Test classification typing
    class_dataset = LengthOfStayDataset(mimic_dir=DUMMY_DIR, task="classification")
    features_clf, label_clf = class_dataset[0]
    assert label_clf.dtype == torch.int64

def test_phenotyping_dataset():
    dataset = PhenotypingDataset(mimic_dir=DUMMY_DIR, seq_length=48)
    features, label = dataset[0]
    
    # Label should be multi-hot array of 25 items
    assert label.shape == (25,)
    assert label.dtype == torch.float32

def test_decompensation_dataset():
    # Decompensation has labels across the sequence
    dataset = DecompensationDataset(mimic_dir=DUMMY_DIR, seq_length=24, window_hours=6)
    features, label = dataset[0]
    
    assert features.shape == (24, 7)
    assert label.shape == (24,) # One label per timestep
    assert label.dtype == torch.int64
