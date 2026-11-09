# Predicting Damaging Aircraft Wildlife Strikes

CS 6140 Machine Learning, Final Group Project (Fall 2026)

Team: Mohammad Arshan Shaikh, Jay Alexander Rodriguez

## Goal
Predict whether an aircraft-wildlife strike damages the aircraft, using only information available at or before impact, with a cost-sensitive evaluation. Models: logistic regression, random forest, gradient boosting (XGBoost/LightGBM/CatBoost), and a neural network.

## Data
FAA National Wildlife Strike Database: https://wildlife.faa.gov/home
Download the file and place it in `data/raw/`. Raw data is not committed to this repo.

## Structure
- `data/`: raw and processed data (ignored by git)
- `notebooks/`: numbered analysis notebooks (01_..., 02_...)
- `src/`: reusable code (loading, preprocessing, models)
- `reports/`: report drafts and figures

## Setup
```
pip install -r requirements.txt
```

## Status
Work in progress.
