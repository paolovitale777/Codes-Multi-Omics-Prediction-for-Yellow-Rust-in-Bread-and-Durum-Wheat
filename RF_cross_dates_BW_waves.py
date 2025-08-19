import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from itertools import product
import matplotlib.pyplot as plt
import os

# Load phenotype data
# load file
pheno = pd.read_csv("All_BLUEs_BW.csv", index_col=0)
#head pheno
print(pheno.head())


waves = pheno.iloc[:, [3,7,15,19,23,4,8,16,20,24,12]]

# Drop NaNs in the response variable (before modeling)
training_col= "score_20240809"
target_col = "score_20240816"
valid_idx_2 = pheno[training_col].notna()

X_training = waves.loc[valid_idx_2]
y_training = pheno.loc[valid_idx_2, training_col].values

X_test = waves.loc[valid_idx_2]
y_test = pheno.loc[valid_idx_2, target_col].values

print("Markers shape for training:", X_training.shape)
print("Pheno shape for training:", y_training.shape)
print("Markers shape for test:", X_test.shape)
print("Pheno shape for test:", y_test.shape)



# Hyperparameter Grid
# -----------------------------
n_features = X_training.shape[1]
param_grid = {
    "mtry": sorted(set([max(1, int(n_features / d)) for d in [5, 3, 2]])),
    "ntree": [500, 1000, 1500],
    "nodesize": [10, 20],
    "max_depth": [5, 10],
    "max_leaf_nodes": [10, 20]
}
grid = list(product(param_grid["mtry"], param_grid["ntree"], param_grid["nodesize"], param_grid["max_depth"], param_grid["max_leaf_nodes"]))

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

for mtry, ntree, nodesize, max_depth, max_leaf_nodes in grid:
    print(f"\nTuning with mtry={mtry}, ntree={ntree}, nodesize={nodesize}, max_depth={max_depth}, max_leaf_nodes={max_leaf_nodes}")
    fold_PAs, fold_MSPEs, fold_MAPEs = [], [], []
    for inner_train_idx, inner_val_idx in inner_cv.split(X_training):
            X_inner_train, X_inner_val = X_training.iloc[inner_train_idx], X_training.iloc[inner_val_idx]
            y_inner_train, y_inner_val = y_training[inner_train_idx], y_training[inner_val_idx]

            rf = RandomForestRegressor(
                    n_estimators=ntree,
                    max_features=mtry,
                    min_samples_split=nodesize,
                    min_samples_leaf=5,
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
                "max_leaf_nodes": max_leaf_nodes
    })

tuning_df = pd.DataFrame(tuning_results)
best_params = tuning_df.loc[tuning_df["MAPE"].idxmin()]

rf = RandomForestRegressor(
            n_estimators=int(best_params["ntree"]),
            max_features=int(best_params["mtry"]),
            min_samples_split=int(best_params["nodesize"]),
            min_samples_leaf=5,
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
            "Best_max_leaf_nodes": int(best_params["max_leaf_nodes"])
    })

# -----------------------------
# Save Results
# -----------------------------
all_results_df = pd.DataFrame(all_cycles_results)
all_results_df["PA_gap"] = all_results_df["PA_train"] - all_results_df["PA_test"]

results_file = os.path.join(output_dir, "RF_BW_cross_dates_waves.csv")
all_results_df.to_csv(results_file, index=False)
print(all_results_df)
