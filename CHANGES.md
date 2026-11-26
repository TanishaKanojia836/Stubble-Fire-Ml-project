# Revision Changelog (Problem P65)

**Name:** Tanisha Kanojia  
**Program:** B.Sc. (Hons.) Computer Science — Semester 5  
**Institution:** Ramanujan College, University of Delhi  
**Submitted to:** Mr. Nikhil  
**Repository:** https://github.com/TanishaKanojia836/Stubble-Fire-Ml-project  

## Summary of Revisions Addressed

1. **Dataset & Summary Reconciliation:** Regenerated all dataset counts directly from `punjab_haryana_stubble_2020_2025.csv` (48,690 verified detections across 2020, 2021, 2022, 2024, and 2025; documented absence of 2023 archive records).
2. **Honest Baseline Comparison:** Corrected all claims regarding persistence comparisons—Random Forest (MAE = 59.90, R² = 0.2127), SVR (MAE = 60.53, R² = 0.2211), and Ridge (MAE = 86.26, R² = 0.1456) outperform the persistence baseline (MAE = 94.68, R² = -1.0490), whereas Lasso (MAE = 94.78) slightly trails persistence on MAE while improving R² (0.0445).
3. **Clustering Verification & Comparative Justification:** Aligned DBSCAN cluster proportions with `3_full_analysis.py` (Cluster 0 = 98.90%, Cluster 1 = 0.25%, Cluster 2 = 0.22%, Noise = 0.63%), added k-distance justification (`eps = 0.1`, `min_samples = 15`), internal validation metrics (Silhouette = 0.4494 on non-noise points), cross-year stability checks, and comparison against K-Means (`k = 3`, Silhouette = 0.4310).
4. **Feature Importance Alignment:** Updated feature importance rankings to match `3_full_analysis.py` (`frp_lag1` = 0.5684, `lag1` = 0.1656, `fire_count_t` = 0.0751) and revised the analytical discussion accordingly.
5. **Expanded Validation, Error Analysis & Operational Thresholds:** Added 5-fold rolling-origin / grouped seasonal cross-validation (mean ± SD), log/sqrt target transformation checks, feature ablation analysis, season-phase error breakdowns (early, peak, late), and a three-tier CPCB advisory decision protocol.
6. **Formal Equations, References & Declarations:** Added numbered mathematical definitions for MAE, RMSE, and R², NASA FIRMS licence and access date, 18 peer-reviewed references with DOIs, an explicit AI-use disclosure, and reproducibility files (`README.md`, `requirements.txt`, `metadata.yml`).
