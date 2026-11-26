import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, silhouette_score
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("punjab_haryana_stubble_2020_2025.csv")
df['acq_date'] = pd.to_datetime(df['acq_date'])
df['year'] = df['acq_date'].dt.year

print("=== 1. DATASET AUDIT ===")
print(f"Total filtered rows: {len(df)}")
print("Year-wise counts:\n", df['year'].value_counts().sort_index())
print(f"Date range: {df['acq_date'].min().date()} to {df['acq_date'].max().date()}")

print("\n=== 2. CLUSTERING & PARAMETER SWEEP ===")
coords = df[['latitude', 'longitude']].values

# K-distance (k=15) percentiles for eps justification
nbrs = NearestNeighbors(n_neighbors=15).fit(coords)
distances, _ = nbrs.kneighbors(coords)
k_dist = np.sort(distances[:, -1])
print("15-NN distance percentiles (50%, 90%, 95%, 99%):",
      np.round(np.percentile(k_dist, [50, 90, 95, 99]), 4))

# Parameter sweep
for eps in [0.05, 0.08, 0.10, 0.15]:
    for ms in [10, 15, 25]:
        labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(coords)
        n_clust = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = round((labels == -1).mean() * 100, 2)
        non_noise = labels != -1
        sil = silhouette_score(coords[non_noise], labels[non_noise]) if n_clust > 1 else np.nan
        print(f"DBSCAN(eps={eps}, ms={ms}) -> clusters={n_clust}, noise={noise_pct}%, sil={sil:.4f}")

# Main DBSCAN (eps=0.1, min_samples=15)
df['cluster_id'] = DBSCAN(eps=0.1, min_samples=15).fit_predict(coords)
print("\nMain DBSCAN Cluster Distribution:")
clust_summary = df.groupby('cluster_id').agg(
    count=('frp', 'count'),
    pct=('frp', lambda x: round(len(x) / len(df) * 100, 2)),
    mean_lat=('latitude', 'mean'),
    mean_lon=('longitude', 'mean'),
    mean_frp=('frp', 'mean')
)
print(clust_summary)

# K-Means comparison (k=6)
km_labels = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(coords)
print(f"\nK-Means (k=6) Silhouette Score: {silhouette_score(coords, km_labels):.4f}")

# Year-over-year DBSCAN stability
print("\nYear-over-Year DBSCAN Stability (eps=0.1, ms=15):")
for yr, grp in df.groupby('year'):
    yr_labels = DBSCAN(eps=0.1, min_samples=15).fit_predict(grp[['latitude', 'longitude']].values)
    n_c = len(set(yr_labels)) - (1 if -1 in yr_labels else 0)
    n_pct = round((yr_labels == -1).mean() * 100, 2)
    c0_pct = round((yr_labels == 0).mean() * 100, 2)
    print(f"Year {yr}: rows={len(grp)}, clusters={n_c}, cluster_0={c0_pct}%, noise={n_pct}%")

print("\n=== 3. DAILY FORECASTING, CV, LOG-TARGET & ABLATION ===")
daily_df = df.groupby('acq_date').agg(
    fire_count_t=('frp', 'count'),
    total_frp_t=('frp', 'sum')
).reset_index().sort_values('acq_date').reset_index(drop=True)

daily_df['fire_count_lag1'] = daily_df['fire_count_t'].shift(1)
daily_df['fire_count_lag2'] = daily_df['fire_count_t'].shift(2)
daily_df['frp_lag1'] = daily_df['total_frp_t'].shift(1)
daily_df['rolling_3d_mean'] = daily_df['fire_count_t'].rolling(3).mean()
doy = daily_df['acq_date'].dt.dayofyear
daily_df['doy_sin'] = np.sin(2 * np.pi * doy / 365.0)
daily_df['doy_cos'] = np.cos(2 * np.pi * doy / 365.0)
daily_df['target_fire_count_t1'] = daily_df['fire_count_t'].shift(-1)
daily_df = daily_df.dropna().reset_index(drop=True)

features = ['fire_count_t', 'fire_count_lag1', 'fire_count_lag2',
            'frp_lag1', 'rolling_3d_mean', 'doy_sin', 'doy_cos']

train_mask = daily_df['acq_date'].dt.year <= 2023
test_mask = daily_df['acq_date'].dt.year >= 2024
print(f"Daily rows -> Total: {len(daily_df)}, Train (<=2022): {train_mask.sum()}, Test (>=2024): {test_mask.sum()}")

X_train, y_train = daily_df.loc[train_mask, features], daily_df.loc[train_mask, 'target_fire_count_t1']
X_test, y_test = daily_df.loc[test_mask, features], daily_df.loc[test_mask, 'target_fire_count_t1']

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Feature Importances (Random Forest)
rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42).fit(X_train_s, y_train)
print("\nRandom Forest Feature Importances:")
for f, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f}: {imp:.4f}")

# Log1p Target Transform Experiment
print("\nLog1p Target Transform Results (Test Set):")
for name, model in [
    ('Ridge (log1p)', Ridge(alpha=10.0)),
    ('Lasso (log1p)', Lasso(alpha=1.0)),
    ('SVR (log1p)', SVR(kernel='rbf', C=10.0, epsilon=0.1)),
    ('RF (log1p)', RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42))
]:
    model.fit(X_train_s, np.log1p(y_train))
    preds = np.expm1(model.predict(X_test_s))
    print(f"  {name}: MAE={mean_absolute_error(y_test, preds):.2f}, "
          f"RMSE={np.sqrt(mean_squared_error(y_test, preds)):.2f}, "
          f"R2={r2_score(y_test, preds):.4f}")

# Rolling-Origin TimeSeriesSplit (5 folds across full series)
print("\n5-Fold Rolling-Origin CV (mean +- sd):")
tscv = TimeSeriesSplit(n_splits=5)
X_all, y_all = daily_df[features], daily_df['target_fire_count_t1']
for name, model in [
    ('Persistence', None),
    ('Ridge', Ridge(alpha=10.0)),
    ('Lasso', Lasso(alpha=1.0)),
    ('SVR', SVR(kernel='rbf', C=10.0, epsilon=1.0)),
    ('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42))
]:
    maes, rmses = [], []
    for tr_idx, va_idx in tscv.split(X_all):
        sc = StandardScaler()
        X_tr, X_va = sc.fit_transform(X_all.iloc[tr_idx]), sc.transform(X_all.iloc[va_idx])
        y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]
        if name == 'Persistence':
            p = X_all.iloc[va_idx]['fire_count_t'].values
        else:
            model.fit(X_tr, y_tr)
            p = model.predict(X_va)
        maes.append(mean_absolute_error(y_va, p))
        rmses.append(np.sqrt(mean_squared_error(y_va, p)))
    print(f"  {name}: MAE = {np.mean(maes):.2f} +- {np.std(maes):.2f}, "
          f"RMSE = {np.mean(rmses):.2f} +- {np.std(rmses):.2f}")

# Feature Ablation (Random Forest on Test Set)
print("\nFeature Ablation (Random Forest Test MAE / RMSE / R2):")
ablation_sets = {
    'All Features': features,
    'Without FRP (no frp_lag1)': [f for f in features if f != 'frp_lag1'],
    'Without Seasonality (no doy)': [f for f in features if not f.startswith('doy')],
    'Only Current Count (fire_count_t)': ['fire_count_t']
}
for label, cols in ablation_sets.items():
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train[cols])
    X_te = sc.transform(X_test[cols])
    m = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42).fit(X_tr, y_train)
    p = m.predict(X_te)
    print(f"  {label}: MAE={mean_absolute_error(y_test, p):.2f}, "
          f"RMSE={np.sqrt(mean_squared_error(y_test, p)):.2f}, "
          f"R2={r2_score(y_test, p):.4f}")

# Phase-wise Error Analysis (Test Set)
print("\nPhase-wise Error Breakdown (Random Forest vs Persistence):")
test_eval = daily_df.loc[test_mask, ['acq_date', 'fire_count_t', 'target_fire_count_t1']].copy()
test_eval['rf_pred'] = rf.predict(X_test_s)
test_eval['doy'] = test_eval['acq_date'].dt.dayofyear
# Early: Oct 1-20 (doy <= 293), Peak: Oct 21-Nov 10 (294-314), Late: Nov 11-30 (>=315)
test_eval['phase'] = pd.cut(test_eval['doy'], bins=[0, 293, 314, 366],
                            labels=['Early (Oct 1-20)', 'Peak (Oct 21-Nov 10)', 'Late (Nov 11-30)'])
for phase, grp in test_eval.groupby('phase', observed=False):
    pers_mae = mean_absolute_error(grp['target_fire_count_t1'], grp['fire_count_t'])
    rf_mae = mean_absolute_error(grp['target_fire_count_t1'], grp['rf_pred'])
    print(f"  {phase} (n={len(grp)}): Persistence MAE={pers_mae:.2f} | RF MAE={rf_mae:.2f}")