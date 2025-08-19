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
from sklearn.svm import SVR

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
# Hyperparameter Grid for SVM
# -----------------------------
param_grid = {
    "C": [0.01, 0.1, 1],
    "epsilon": [0.05, 0.1, 0.2, 0.5],
    "gamma": [1, 0.1, 0.01],
    "kernel": ["rbf", "linear", "poly", "sigmoid"]
}

grid = list(product(param_grid["C"], param_grid["epsilon"], param_grid["kernel"], param_grid["gamma"]))

# # -----------------------------
# Create output directory

output_dir = "nested_cv_svm_results"
os.makedirs(output_dir, exist_ok=True)



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

results_file = os.path.join(output_dir, "SVR_BW(training)_cross_crops_waves.csv")
all_results_df.to_csv(results_file, index=False)
print(all_results_df)
