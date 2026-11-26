# Spatiotemporal Analysis of Stubble Burning in Punjab and Haryana (Problem P65)

**Author:** Tanisha Kanojia (University Roll No.: 24570068)  
**Course:** Semester 5 Machine Learning Project  
**Repository:** [https://github.com/TanishaKanojia836/Stubble-Fire-Ml-project](https://github.com/TanishaKanojia836/Stubble-Fire-Ml-project)

## Overview

This repository contains the reproducible data-processing, spatial clustering, and chronological forecasting pipeline for analyzing post-monsoon paddy stubble burning across Punjab and Haryana (`27.6°N–32.5°N`, `73.8°E–77.6°E`) using NASA MODIS Collection 6.1 active fire detections (`MCD14ML`) across five post-monsoon seasons (`2020`, `2021`, `2022`, `2024`, and `2025`; `48,690` verified detections).

## Repository Structure

```text
Stubble-Fire-Ml-project/
├── 1_extract_and_filter.py              # Extracts and filters raw MODIS India CSVs into clean master CSV
├── 2_train.py                           # Runs DBSCAN clustering, chronological forecasting, and generates Figure 1
├── 3_full_analysis.py                   # Executes k-distance quantiles, parameter sweep, CV, ablation, and phase analysis
├── punjab_haryana_stubble_2020_2025.csv # Filtered master dataset (48,690 rows x 15 columns)
├── README.md                            # Setup and execution instructions
├── requirements.txt                     # Python package dependencies
├── metadata.yml                         # Submission metadata
└── CHANGES.md                           # Point-by-point revision log addressing reviewer feedback
```

## Data Source & Licence

- **Dataset:** NASA FIRMS MODIS Collection 6.1 Standard Quality Active Fire Detections (`MCD14ML`)
- **Portal:** [https://firms.modaps.eosdis.nasa.gov/country/](https://firms.modaps.eosdis.nasa.gov/country/)
- **Licence:** NASA Earth Science Data and Information System (ESDIS) Open Data Policy
- **Access Date:** November 2025
- **Note on 2023 Archive:** `modis_2023_India.csv` was omitted in the downloaded annual bundle and is intentionally excluded without synthetic imputation to prevent temporal data leakage.

## Installation & End-to-End Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TanishaKanojia836/Stubble-Fire-Ml-project.git
   cd Stubble-Fire-Ml-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline in sequence:**
   - *(Optional, if raw annual `*India*.csv` files are placed in subdirectories)* Rebuild the filtered dataset:
     ```bash
     python 1_extract_and_filter.py
     ```
   - Train DBSCAN (`eps=0.10`, `min_samples=15`) and the chronological forecasting models (`<=2023` train, `>=2024` test) and generate the spatial/forecasting plots:
     ```bash
     python 2_train.py
     ```
   - Run the complete validation suite (15-NN distance quantiles, 12-configuration DBSCAN sweep, $K$-Means comparison, 5-fold rolling-origin cross-validation, `log1p` target transform, feature ablation, and seasonal phase error breakdown):
     ```bash
     python 3_full_analysis.py
     ```

## Key Verified Results

- **DBSCAN Spatial Clustering:** 6 active clusters (`cluster_id` 0–5) + 0.63% noise (`306` detections); non-noise Silhouette = `0.3782` vs. `0.3584` for $K$-Means ($k=6$). Cluster 0 concentrates `98.90%` (`48,155` detections) across central Punjab and northern Haryana.
- **Chronological Forecasting (`104` Holdout Test Days, 2024–2025):**
  - **Persistence Baseline:** MAE = `94.68`, RMSE = `156.13`, $R^2$ = `-1.0490`
  - **Random Forest Regressor:** MAE = `59.90` (−36.7% vs. persistence), RMSE = `96.78`, $R^2$ = `0.2127`
  - **Support Vector Regressor (RBF):** MAE = `64.38`, RMSE = `96.24`, $R^2$ = `0.2215`
  - **Ridge Regression ($\alpha=10.0$):** MAE = `74.10`, RMSE = `97.10`, $R^2$ = `0.2075`
  - **Lasso Regression ($\alpha=1.0$):** MAE = `94.78`, RMSE = `113.54`, $R^2$ = `-0.0836`