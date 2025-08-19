import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Load and Prepare Data
# -----------------------------
pheno = pd.read_csv("All_BLUEs_BW.csv", index_col=0)
waves = pheno.iloc[:, [3, 7, 15, 19, 23]]
target_col = "score_20240809"

valid_idx = pheno[target_col].notna()
X = waves.loc[valid_idx]
y = pheno.loc[valid_idx, target_col].values

# -----------------------------

# Hyperparameter grid with L1 (reg_alpha) and L2 (reg_lambda)
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

# Nested cross-validation
all_cycles_results = []

for cycle in range(1, 11):
    print(f"\n===== Cycle {cycle} =====")
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=1000 + cycle)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        inner_cv = KFold(n_splits=4, shuffle=True, random_state=2000 + cycle)
        tuning_results = []

        for num_leaves, n_estimators, min_child_samples, max_depth, reg_alpha, reg_lambda, min_data_in_leaf,min_child_wight,min_split_gain  in grid:
            fold_PAs, fold_MSPEs, fold_MAPEs = [], [], []

            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
                X_inner_train = X_train_outer.iloc[inner_train_idx]
                X_inner_val = X_train_outer.iloc[inner_val_idx]
                y_inner_train = y_train_outer[inner_train_idx]
                y_inner_val = y_train_outer[inner_val_idx]

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
        model.fit(X_train_outer, y_train_outer)
        y_train_pred = model.predict(X_train_outer)
        y_test_pred = model.predict(X_test_outer)

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
            "Best_num_leaves": int(best_params["num_leaves"]),
            "Best_n_estimators": int(best_params["n_estimators"]),
            "Best_min_child_samples": int(best_params["min_child_samples"]),
            "Best_max_depth": int(best_params["max_depth"]),
            "Best_reg_alpha": best_params["reg_alpha"],
            "Best_reg_lambda": best_params["reg_lambda"],
            "Best_min_data_in_leaf": int(best_params["min_data_in_leaf"]),
            "Best_min_child_wight": int(best_params["min_child_wight"])
        })

# Save results
# -----------------------------
results_df = pd.DataFrame(all_cycles_results)
results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]
results_df.to_csv(os.path.join(output_dir, "LGBM_score1_bw_CVnested_waves.csv"), index=False)

# -----------------------------
