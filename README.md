# EEG Classification Assignment (MECHENG 736)

This repository contains code for **Assignment 1: Measuring and Classifying the Electroencephalogram (EEG)**, part of MECHENG 736 (Biomechatronic Systems).  
The project implements feature extraction and classification of EEG signals to distinguish between eyes-open and eyes-closed states.

## Project structure
```
├── data/               # Raw EEG data files (ignored in Git; add manually)
├── plots/              # Output plots (generated automatically, ignored in Git)
├── scripts/            # Helper functions for features, classifiers, plotting
│   ├── power_features.py
│   ├── linear_classifier.py
│   ├── svm_classifier.py
│   ├── plot_powers.py
│   └── plot_decision_boundary.py
├── main.py             # Main pipeline (runs feature extraction, classifiers, plots)
├── requirements.txt    # Python package dependencies
└── README.md
```

## Features
- **Feature extraction**: Computes mean power in alpha (8–12 Hz) and theta (4–8 Hz) bands for each 5-second EEG epoch, with results expressed in dB.  
- **Linear classifier**: Fisher’s Linear Discriminant (implemented via scikit-learn’s `LinearDiscriminantAnalysis`).  
- **Support Vector Machine (SVM)**: Linear kernel SVM for comparison.  
- **Cross-validation**: Leave-2-out cross-validation used for both classifiers.  
- **Visualisation**: 2D feature space plots with decision boundaries, epoch markers, and state labelling.  
- **Statistical comparison**: McNemar’s test to evaluate whether differences in classifier accuracy are statistically significant.  

## Requirements
This project was developed with **Python 3.10.10**.  
Install required packages with:

```bash
pip install -r requirements.txt
```

## Dependencies
- numpy
- scipy
- matplotlib
- scikit-learn
- statsmodels

## Usage
Run the main script:

```bash
python main.py
```

This will:
- Load EEG data (expects `data/recording2.wav` by default; not tracked in Git, see `.gitignore`).
- Extract alpha and theta band powers.
- Train/test both classifiers with cross-validation.
- Save plots automatically into the `plots/` folder (not tracked in Git, see `.gitignore`).

Authored by **Charlie Brunton**, University of Auckland.
