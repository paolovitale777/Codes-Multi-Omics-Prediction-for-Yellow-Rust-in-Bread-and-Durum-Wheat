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
from lightgbm import LGBMRegressor

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
    "num_leaves": [10,20,50],
    "n_estimators": [75,150,300],
    "min_child_samples": [10,20],
    "max_depth": [10,20],
    "reg_alpha": [0,0.1],
    "reg_lambda": [0,0.1],
    "min_data_in_leaf": [50,100],
    "min_child_wight": [0.001],
    'min_split_gain ': [0]
    }

grid = list(product(
    param_grid["num_leaves"],
    param_grid["n_estimators"],
    param_grid["min_child_samples"],
    param_grid["max_depth"],
    param_grid["reg_alpha"],
    param_grid["reg_lambda"],
    param_grid["min_data_in_leaf"],
    param_grid["min_child_wight"],
    param_grid["min_split_gain "]
))

# Output directory
output_dir = "nested_cv_lgbm_results"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 10-Cycle Nested CV
# -----------------------------
all_cycles_results = []

inner_cv = KFold(n_splits=5, shuffle=True, random_state=2000)
tuning_results = []

for num_leaves, n_estimators, min_child_samples, max_depth, reg_alpha, reg_lambda, min_data_in_leaf,min_child_wight,min_split_gain  in grid:
    print(f"Training with num_leaves={num_leaves}, n_estimators={n_estimators}, min_child_samples={min_child_samples}, max_depth={max_depth}")  
    fold_PAs, fold_MSPEs, fold_MAPEs = [], [], []
    for inner_train_idx, inner_val_idx in inner_cv.split(X_training):
            X_inner_train, X_inner_val = X_training.iloc[inner_train_idx], X_training.iloc[inner_val_idx]
            y_inner_train, y_inner_val = y_training[inner_train_idx], y_training[inner_val_idx]

            model = LGBMRegressor(
                    num_leaves=num_leaves,
                    n_estimators=n_estimators,
                    min_child_samples=min_child_samples,
                    max_depth=max_depth,
                    learning_rate=0.05,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    min_data_in_leaf=min_data_in_leaf,
                    min_child_wight=min_child_wight,
                    min_split_gain=min_split_gain,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                )
            model.fit(X_inner_train, y_inner_train)
            y_val_pred = model.predict(X_inner_val)

            pa, _ = pearsonr(y_inner_val, y_val_pred)
            mspe = mean_squared_error(y_inner_val, y_val_pred)
            mape = mean_absolute_error(y_inner_val, y_val_pred)

            fold_PAs.append(pa)
            fold_MSPEs.append(mspe)
            fold_MAPEs.append(mape)

    tuning_results.append({
                "num_leaves": num_leaves,
                "n_estimators": n_estimators,
                "min_child_samples": min_child_samples,
                "max_depth": max_depth,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "min_data_in_leaf": min_data_in_leaf,
                "min_child_wight": min_child_wight,
                "min_split_gain": min_split_gain,
                "PA": np.mean(fold_PAs),
                "MSPE": np.mean(fold_MSPEs),
                "MAPE": np.mean(fold_MAPEs)
            })


tuning_df = pd.DataFrame(tuning_results)
best_params = tuning_df.loc[tuning_df["MAPE"].idxmin()]

model = LGBMRegressor(
            num_leaves=int(best_params["num_leaves"]),
            n_estimators=int(best_params["n_estimators"]),
            min_child_samples=int(best_params["min_child_samples"]),
            max_depth=int(best_params["max_depth"]),
            reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"],
            min_data_in_leaf=int(best_params["min_data_in_leaf"]),
            min_child_wight=int(best_params["min_child_wight"]),
            min_split_gain=best_params["min_split_gain"],
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

model.fit(X_training, y_training)
y_train_pred = model.predict(X_training)
y_test_pred = model.predict(X_test)

pa_train, _ = pearsonr(y_training, y_train_pred)
pa_test, _ = pearsonr(y_test, y_test_pred)
mspe = mean_squared_error(y_test, y_test_pred)
mape = mean_absolute_error(y_test, y_test_pred)

all_cycles_results.append({
            "PA_train": pa_train,
            "PA_test": pa_test,
            "MSPE": mspe,
            "MAPE": mape,
            "Best_num_leaves": int(best_params["num_leaves"]),
            "Best_n_estimators": int(best_params["n_estimators"]),
            "Best_min_child_samples": int(best_params["min_child_samples"]),
            "Best_max_depth": int(best_params["max_depth"]),
            "Best_reg_alpha": best_params["reg_alpha"],
            "Best_reg_lambda": best_params["reg_lambda"],
            "Best_min_data_in_leaf": int(best_params["min_data_in_leaf"]),
            "Best_min_child_wight": int(best_params["min_child_wight"])
        })

# -----------------------------
# Save Results
# -----------------------------
all_results_df = pd.DataFrame(all_cycles_results)
all_results_df["PA_gap"] = all_results_df["PA_train"] - all_results_df["PA_test"]

results_file = os.path.join(output_dir, "LGBM_BW(training)_cross_crops_waves_score1.csv")
all_results_df.to_csv(results_file, index=False)
print(all_results_df)
