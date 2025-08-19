import pandas as pd
import numpy as np
from sklearn.svm import SVR
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



waves = pheno.iloc[:, [3,7,15,19,23,4,8,16,20,24,12]]


X_training = pd.concat([markers, waves], axis=1)

X_test = pd.concat([markers, waves], axis=1)



# Drop NaNs in the response variable (before modeling)
training_col= "score_20240809"
target_col = "score_20240816"
valid_idx_2 = pheno[training_col].notna()

X_training = X_training.loc[valid_idx_2]
y_training = pheno.loc[valid_idx_2, training_col].values

X_test = X_test.loc[valid_idx_2]
y_test = pheno.loc[valid_idx_2, target_col].values

print("Markers shape for training:", X_training.shape)
print("Pheno shape for training:", y_training.shape)
print("Markers shape for test:", X_test.shape)
print("Pheno shape for test:", y_test.shape)



# Hyperparameter Grid
# -----------------------------
n_features = X_training.shape[1]
# Hyperparameter Grid for SVM
# -----------------------------
param_grid = {
    "C": [0.01, 0.1, 1, 10,100],
    "epsilon": [0.05, 0.1, 0.2],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["rbf"]
}

grid = list(product(param_grid["C"], param_grid["epsilon"], param_grid["kernel"], param_grid["gamma"]))

# # -----------------------------
# Create output directory

output_dir = "nested_cv_svm_results"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 10-Cycle Nested CV
# -----------------------------
all_cycles_results = []

inner_cv = KFold(n_splits=5, shuffle=True, random_state=2000)
tuning_results = []

for C, epsilon, kernel, gamma in grid:
    print(f"Training with C={C}, epsilon={epsilon}, kernel={kernel}, gamma={gamma}")
    fold_PAs, fold_MSPEs, fold_MAPEs = [], [], []
    for inner_train_idx, inner_val_idx in inner_cv.split(X_training):
            X_inner_train, X_inner_val = X_training.iloc[inner_train_idx], X_training.iloc[inner_val_idx]
            y_inner_train, y_inner_val = y_training[inner_train_idx], y_training[inner_val_idx]

            svr = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)

            svr.fit(X_inner_train, y_inner_train)
            y_val_pred = svr.predict(X_inner_val)

            pa, _ = pearsonr(y_inner_val, y_val_pred)
            mspe = mean_squared_error(y_inner_val, y_val_pred)
            mape = mean_absolute_error(y_inner_val, y_val_pred)

            fold_PAs.append(pa)
            fold_MSPEs.append(mspe)
            fold_MAPEs.append(mape)

            tuning_results.append({
                "C": C,
                "epsilon": epsilon,
                "kernel": kernel,
                "PA": np.mean(fold_PAs),
                "MSPE": np.mean(fold_MSPEs),
                "MAPE": np.mean(fold_MAPEs),
                "gamma": gamma
            })

tuning_df = pd.DataFrame(tuning_results)
best_params = tuning_df.loc[tuning_df["MAPE"].idxmin()]

svr = SVR(C=best_params["C"],
          epsilon=best_params["epsilon"],
          kernel=best_params["kernel"],
          gamma=best_params["gamma"])

svr.fit(X_training, y_training)
y_train_pred = svr.predict(X_training)
y_test_pred = svr.predict(X_test)

pa_train, _ = pearsonr(y_training, y_train_pred)
pa_test, _ = pearsonr(y_test, y_test_pred)
mspe = mean_squared_error(y_test, y_test_pred)
mape = mean_absolute_error(y_test, y_test_pred)


all_cycles_results.append({
            "PA_train": pa_train,
            "PA_test": pa_test,
            "MSPE": mspe,
            "MAPE": mape,
            "Best_C": best_params["C"],
            "Best_epsilon": best_params["epsilon"],
            "Best_kernel": best_params["kernel"],
            "Best_gamma": best_params["gamma"]
        })


# -----------------------------
# Save Results
# -----------------------------
all_results_df = pd.DataFrame(all_cycles_results)
all_results_df["PA_gap"] = all_results_df["PA_train"] - all_results_df["PA_test"]

results_file = os.path.join(output_dir, "SVR_BW_cross_dates_markers_plus_waves.csv")
all_results_df.to_csv(results_file, index=False)
print(all_results_df)
