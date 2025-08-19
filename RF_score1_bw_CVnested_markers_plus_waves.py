import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from itertools import product
import matplotlib.pyplot as plt
import os

# -----------------------------
# Step 1: Load and Prepare Data
# -----------------------------

# Load marker data
markers = pd.read_csv(
    "Markers_BW_rust_70MISS_1MAF_10HET_90GENO.txt",
    sep="\t",
    index_col=0,
    dtype={0: str}  # Ensures rownames (first column) are read as string
)


markers.index = markers.index.astype(str)

#head markers
print(markers.head())
# Load phenotype data
# load file
pheno = pd.read_csv("All_BLUEs_BW.csv", index_col=0)
#head pheno
print(pheno.head())


# Load files
#markers = pd.read_csv("marker_simulated.txt", sep="\t", index_col=0)
#pheno = pd.read_csv("pheno_simulated.csv", index_col=0)

# Force string index for both
markers.index = markers.index.astype(float).astype(int).astype(str)
pheno.index = pheno.index.astype(float).astype(int).astype(str)

# Find common GIDs (now all string)
common_gids = sorted(set(markers.index).intersection(pheno.index))
print("Common GIDs:", len(common_gids))

duplicates = markers.index[markers.index.duplicated()]
print("Duplicate GIDs in markers:", duplicates.tolist())

markers = markers[~markers.index.duplicated(keep="first")]


# ✅ Use reindex instead of loc (to avoid hidden mismatch)
markers = markers.reindex(common_gids)
pheno = pheno.reindex(common_gids)

# ✅ Confirm dimensions
print("Markers shape:", markers.shape)
print("Pheno shape:", pheno.shape)


#select the columns of interest wavelwngths
#waves = pheno.iloc[:, list(range(1, 6)) + list(range(15, 20)) + list(range(29, 34)) + list(range(43, 48)) + list(range(57, 62))]
#print(waves.head())



waves = pheno.iloc[:, [3, 7, 15, 19, 23]]

X_combined = pd.concat([markers, waves], axis=1)

# Drop NaNs in the response variable (before modeling)
target_col = "score_20240809"
valid_idx = pheno[target_col].notna()

X = X_combined.loc[valid_idx]
y = pheno.loc[valid_idx, target_col].values

print("Markers shape:", X.shape)
print("Pheno shape:", y.shape)


# Hyperparameter Grid
# -----------------------------
n_features = X.shape[1]
param_grid = {
    "mtry": sorted(set([max(1, int(n_features / d)) for d in [5, 3, 2]])),
    "ntree": [500, 1000, 1500],
    "nodesize": [10, 20]
}
grid = list(product(param_grid["mtry"], param_grid["ntree"], param_grid["nodesize"]))

# -----------------------------
# Create output directory
# -----------------------------
output_dir = "nested_cv_rf_results"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 10-Cycle Nested CV
# -----------------------------
all_cycles_results = []

for cycle in range(1, 11):
    print(f"\n===== Cycle {cycle} =====")
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=1000 + cycle)
    
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        inner_cv = KFold(n_splits=4, shuffle=True, random_state=2000 + cycle)
        tuning_results = []

        for mtry, ntree, nodesize in grid:
            fold_PAs, fold_MSPEs, fold_MAPEs = [], [], []
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
                X_inner_train, X_inner_val = X_train_outer.iloc[inner_train_idx], X_train_outer.iloc[inner_val_idx]
                y_inner_train, y_inner_val = y_train_outer[inner_train_idx], y_train_outer[inner_val_idx]

                rf = RandomForestRegressor(
                    n_estimators=ntree,
                    max_features=mtry,
                    min_samples_split=nodesize,
                    min_samples_leaf=5,
                    max_depth=4,
                    max_leaf_nodes=10,
                    n_jobs=-1,
                    random_state=42
                )
                rf.fit(X_inner_train, y_inner_train)
                y_val_pred = rf.predict(X_inner_val)

                pa, _ = pearsonr(y_inner_val, y_val_pred)
                mspe = mean_squared_error(y_inner_val, y_val_pred)
                mape = mean_absolute_error(y_inner_val, y_val_pred)

                fold_PAs.append(pa)
                fold_MSPEs.append(mspe)
                fold_MAPEs.append(mape)

            tuning_results.append({
                "mtry": mtry,
                "ntree": ntree,
                "nodesize": nodesize,
                "PA": np.mean(fold_PAs),
                "MSPE": np.mean(fold_MSPEs),
                "MAPE": np.mean(fold_MAPEs)
            })

        tuning_df = pd.DataFrame(tuning_results)
        best_params = tuning_df.loc[tuning_df["MAPE"].idxmin()]

        rf = RandomForestRegressor(
            n_estimators=int(best_params["ntree"]),
            max_features=int(best_params["mtry"]),
            min_samples_split=int(best_params["nodesize"]),
            min_samples_leaf=5,
            max_depth=4,
            max_leaf_nodes=10,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train_outer, y_train_outer)
        y_train_pred = rf.predict(X_train_outer)
        y_test_pred = rf.predict(X_test_outer)

        pa_train, _ = pearsonr(y_train_outer, y_train_pred)
        pa_test, _ = pearsonr(y_test_outer, y_test_pred)
        mspe = mean_squared_error(y_test_outer, y_test_pred)
        mape = mean_absolute_error(y_test_outer, y_test_pred)

        all_cycles_results.append({
            "Cycle": cycle,
            "OuterFold": outer_fold + 1,
            "PA_train": pa_train,
            "PA_test": pa_test,
            "MSPE": mspe,
            "MAPE": mape,
            "Best_mtry": int(best_params["mtry"]),
            "Best_ntree": int(best_params["ntree"]),
            "Best_nodesize": int(best_params["nodesize"])
        })

# -----------------------------
# Save Results
# -----------------------------
all_results_df = pd.DataFrame(all_cycles_results)
all_results_df["PA_gap"] = all_results_df["PA_train"] - all_results_df["PA_test"]

results_file = os.path.join(output_dir, "RF_score1_bw_CVnested_markers_plus_waves.csv")
all_results_df.to_csv(results_file, index=False)
print(f"\nSaved all results to: {results_file}")

# -----------------------------
# Plot
# -----------------------------
import seaborn as sns
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.lineplot(data=all_results_df, x="OuterFold", y="PA_test", hue="Cycle", marker="o")
plt.title("Test Pearson Correlation Across Cycles and Folds")
plt.xlabel("Outer Fold")
plt.ylabel("PA (Test)")
plt.legend(title="Cycle", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plot_file = os.path.join(output_dir, "RF_score1_bw_CVnested_markers_plus_waves.png")
plt.savefig(plot_file, dpi=300)
print(f"Saved plot to: {plot_file}")
