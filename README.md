# MIMIC-DataLoader

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Drop-in PyTorch Dataset classes for MIMIC-IV clinical prediction tasks (mortality, readmission, LOS, sepsis, phenotyping, decompensation)**

*Topics: pytorch, mimic-iv, clinical-ml, ehr, healthcare-ai, dataset*

`mimic-dataloader` aims to do for structured EHR data what TorchXRayVision did for chest X-rays. Researchers currently spend days writing custom preprocessing scripts to extract features, handle time-series splits securely, and structure the data for PyTorch every time they start a new MIMIC-IV project.

This library provides unified, fully-reusable dataset classes for the 6 standard prediction tasks out-of-the-box (mortality, readmission, length-of-stay, sepsis, phenotyping, decompensation) with robust handling of standard train/val/test splits to avoid temporal information leakage.

## Core Features
* **6 Standard Tasks:** Out of the box datasets for Mortality, Readmission, Length-of-Stay, Sepsis, Phenotyping, and Decompensation.
* **PyTorch Integration:** Datasets subclass `torch.utils.data.Dataset`, making them instantly ready for use with standard DataLoaders.
* **Leakage-proof Splits:** Built-in train, validation, and test splits with rigorous prevention of temporal information leakage.
* **PhysioNet Native:** Helper functions built around the structure of the credentialed PhysioNet MIMIC-IV database.

## Installation

```bash
pip install mimic-dataloader
```

## Quick Start

```python
import torch
from torch.utils.data import DataLoader
from mimic_dataloader.datasets import MortalityDataset, ReadmissionDataset

# Initialize the dataset for predicting in-hospital mortality
train_dataset = MortalityDataset(
    mimic_dir="/path/to/physionet/mimiciv",
    split="train",
    seq_length=48
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Iterate through batches
for batch in train_loader:
    features, labels = batch
    # features shape: (32, 48, num_features)
    # labels shape: (32,)
    # ... train your model ...
```

## Supported Prediction Tasks

1. **Mortality:** Predict patient in-hospital mortality based on the first 48 hours of ICU data.
2. **Readmission:** Predict 30-day unplanned hospital readmission.
3. **Length of Stay:** Predict ICU length of stay (regression or multi-class).
4. **Sepsis:** Predict the onset of sepsis (Sepsis-3 criteria).
5. **Phenotyping:** Multi-label classification of 25 common clinical conditions.
6. **Decompensation:** Predict physiologic decompensation in the next 24 hours.

## Known Limitations & V0.1 Scope

* **Feature Set:** Currently extracts 7 common continuous vital signs (Heart Rate, Respiratory Rate, O2 Saturation, SBP, DBP, MAP, Temperature). Lab values (e.g. creatinine, lactate) and medications are planned for v0.2.
* **Imputation:** Time buckets are forward-filled (`ffill`). If a value is missing at the start of a sequence before a forward-fill can occur, it will be left as NaN (and eventually 0-padded in the tensor), which users should handle downstream.
* **Version Paths:** Supports both MIMIC-IV v1.0 (`core/` demographics directory) and v2.0+ (`hosp/` directory) automatically.

## Requirements
* PyTorch >= 2.0.0
* Pandas >= 2.0.0
* Python >= 3.9

*(Note: Data access requires credentialed access to MIMIC-IV via PhysioNet. This package does not distribute data.)*

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License
