from matplotlib.colors import cnames
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from itertools import product
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler


# Load phenotype data
# load file
pheno = pd.read_table("input_pheno_cross_crop.txt", index_col=0)
#head pheno
print(pheno.head())

waves = pheno.iloc[:, [3,4,5,6,7]]
print(waves.columns)

scaler = StandardScaler()
waves_scaled = pd.DataFrame(scaler.fit_transform(waves), index=waves.index, columns=waves.columns)

# Drop NaNs in the response variable (before modeling)
target_trait= "score_1"
# Identify valid (non-NA) entries for the target trait
valid_idx = pheno[target_trait].notna()

# Now filter the phenotype data
valid_pheno = pheno[valid_idx]

# Define training and test sets based on crop label
trn = valid_pheno["crop"] == "bread"
tst = valid_pheno["crop"] == "durum"

# Prepare inputs and outputs
X_training = waves_scaled.loc[valid_pheno.index[trn]]
y_training = valid_pheno.loc[trn, target_trait].values

X_test = waves_scaled.loc[valid_pheno.index[tst]]
y_test = valid_pheno.loc[tst, target_trait].values

print("Markers shape for training:", X_training.shape)
print("Pheno shape for training:", y_training.shape)
print("Markers shape for test:", X_test.shape)
print("Pheno shape for test:", y_test.shape)

print(X_training.index[:5])
print(valid_pheno.index[trn][:5])

print(X_test.index[:5])
print(valid_pheno.index[tst][:5])


# Hyperparameter Grid
# -----------------------------
n_features = X_training.shape[1]
param_grid = {
    "mtry": sorted(set([max(1, int(n_features / d)) for d in [5, 3, 2]])),
    "ntree": [500, 1000, 1500],
    "nodesize": [2, 5, 10],
    "max_depth": [20,50],
    "max_leaf_nodes": [10,30],
    "min_samples_leaf": [1,5]
}
grid = list(product(param_grid["mtry"], param_grid["ntree"], param_grid["nodesize"], param_grid["max_depth"], param_grid["max_leaf_nodes"], param_grid["min_samples_leaf"]))

# -----------------------------
# Create output directory
# -----------------------------
output_dir = "nested_cv_rf_results"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 10-Cycle Nested CV
# -----------------------------
all_cycles_results = []

inner_cv = KFold(n_splits=5, shuffle=True, random_state=2000)
tuning_results = []

for mtry, ntree, nodesize, max_depth, max_leaf_nodes, min_samples_leaf in grid:
    print(f"\nTuning with mtry={mtry}, ntree={ntree}, nodesize={nodesize}, max_depth={max_depth}, max_leaf_nodes={max_leaf_nodes}, min_samples_leaf= {min_samples_leaf}")
    fold_PAs, fold_MSPEs, fold_MAPEs = [], [], []
    for inner_train_idx, inner_val_idx in inner_cv.split(X_training):
            X_inner_train, X_inner_val = X_training.iloc[inner_train_idx], X_training.iloc[inner_val_idx]
            y_inner_train, y_inner_val = y_training[inner_train_idx], y_training[inner_val_idx]

            rf = RandomForestRegressor(
                    n_estimators=ntree,
                    max_features=mtry,
                    min_samples_split=nodesize,
                    min_samples_leaf=min_samples_leaf,
                    max_depth=max_depth,
                    max_leaf_nodes=max_leaf_nodes,
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
                "MAPE": np.mean(fold_MAPEs),
                "max_depth": max_depth,
                "max_leaf_nodes": max_leaf_nodes,
                "min_samples_leaf": min_samples_leaf
    })

tuning_df = pd.DataFrame(tuning_results)
best_params = tuning_df.loc[tuning_df["MAPE"].idxmin()]

rf = RandomForestRegressor(
            n_estimators=int(best_params["ntree"]),
            max_features=int(best_params["mtry"]),
            min_samples_split=int(best_params["nodesize"]),
            min_samples_leaf= int(best_params["min_samples_leaf"]),
            max_depth=int(best_params["max_depth"]),
            max_leaf_nodes=int(best_params["max_leaf_nodes"]),
            n_jobs=-1,
            random_state=42
)
rf.fit(X_training, y_training)
y_train_pred = rf.predict(X_training)
y_test_pred = rf.predict(X_test)

pa_train, _ = pearsonr(y_training, y_train_pred)
pa_test, _ = pearsonr(y_test, y_test_pred)
mspe = mean_squared_error(y_test, y_test_pred)
mape = mean_absolute_error(y_test, y_test_pred)

all_cycles_results.append({
            "PA_train": pa_train,
            "PA_test": pa_test,
            "MSPE": mspe,
            "MAPE": mape,
            "Best_mtry": int(best_params["mtry"]),
            "Best_ntree": int(best_params["ntree"]),
            "Best_nodesize": int(best_params["nodesize"]),
            "Best_max_depth": int(best_params["max_depth"]),
            "Best_max_leaf_nodes": int(best_params["max_leaf_nodes"]),
            "Best_min_samples_leaf": int(best_params["min_samples_leaf"])
    })

# -----------------------------
# Save Results
# -----------------------------
all_results_df = pd.DataFrame(all_cycles_results)
all_results_df["PA_gap"] = all_results_df["PA_train"] - all_results_df["PA_test"]

results_file = os.path.join(output_dir, "RF_BW(tst)_score1_cross_crops_waves.csv")
all_results_df.to_csv(results_file, index=False)
print(all_results_df)
