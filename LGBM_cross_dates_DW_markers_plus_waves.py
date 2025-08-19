import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
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
    "Markers_DW_rust_10MISS_1MAF_10HET_10GENO.txt",
    sep="\t",
    index_col=0,
    dtype={0: str}  # Ensures rownames (first column) are read as string
)


markers.index = markers.index.astype(str)

#head markers
print(markers.head())
# Load phenotype data
# load file
pheno = pd.read_csv("All_BLUEs_DW.csv", index_col=0)
#head pheno
print(pheno.head())

markers.index = markers.index.str.replace(r'^X', '', regex=True)

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



waves = pheno.iloc[:, [2, 4, 8, 10, 12,3,5,9,11,13]]


X_training = pd.concat([markers, waves], axis=1)

X_test = pd.concat([markers, waves], axis=1)



# Drop NaNs in the response variable (before modeling)
training_col= "severity_20240829"
target_col = "severity_20240905"
valid_idx_2 = pheno[training_col].notna()

X_training = X_training.loc[valid_idx_2]
y_training = pheno.loc[valid_idx_2, training_col].values

X_test = X_test.loc[valid_idx_2]
y_test = pheno.loc[valid_idx_2, target_col].values

print("Markers shape for training:", X_training.shape)
print("Pheno shape for training:", y_training.shape)
print("Markers shape for test:", X_test.shape)
print("Pheno shape for test:", y_test.shape)



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

results_file = os.path.join(output_dir, "RLGBM_DW_cross_dates_markers_plus_waves.csv")
all_results_df.to_csv(results_file, index=False)
print(all_results_df)
