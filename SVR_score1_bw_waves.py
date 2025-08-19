import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Step 1: Load and Prepare Data
# -----------------------------
pheno = pd.read_csv("All_BLUEs_BW.csv", index_col=0)

print(pheno.head())
waves = pheno.iloc[:, [3, 7, 15, 19, 23]]
print(waves.head())

target_col = "score_20240809"
valid_idx = pheno[target_col].notna()
X = waves.loc[valid_idx]
y = pheno.loc[valid_idx, target_col].values

# -----------------------------
# Hyperparameter Grid for SVM
# -----------------------------
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "epsilon": [0.05, 0.1, 0.2],
    "kernel": ["rbf"]
}

grid = list(product(param_grid["C"], param_grid["epsilon"], param_grid["kernel"]))

# -----------------------------
# Create output directory
# -----------------------------
output_dir = "nested_cv_svm_results"
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

        scaler = StandardScaler()
        X_train_outer_scaled = scaler.fit_transform(X_train_outer)
        X_test_outer_scaled = scaler.transform(X_test_outer)

        inner_cv = KFold(n_splits=4, shuffle=True, random_state=2000 + cycle)
        tuning_results = []

        for C, epsilon, kernel in grid:
            fold_PAs, fold_MSPEs, fold_MAPEs = [], [], []

            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer_scaled):
                X_inner_train = X_train_outer_scaled[inner_train_idx]
                X_inner_val = X_train_outer_scaled[inner_val_idx]
                y_inner_train = y_train_outer[inner_train_idx]
                y_inner_val = y_train_outer[inner_val_idx]

                svr = SVR(C=C, epsilon=epsilon, kernel=kernel)
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
                "MAPE": np.mean(fold_MAPEs)
            })

        tuning_df = pd.DataFrame(tuning_results)
        best_params = tuning_df.loc[tuning_df["MAPE"].idxmin()]

        final_model = SVR(C=best_params["C"], epsilon=best_params["epsilon"], kernel=best_params["kernel"])
        final_model.fit(X_train_outer_scaled, y_train_outer)
        y_train_pred = final_model.predict(X_train_outer_scaled)
        y_test_pred = final_model.predict(X_test_outer_scaled)

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
            "Best_C": best_params["C"],
            "Best_epsilon": best_params["epsilon"],
            "Best_kernel": best_params["kernel"]
        })

# Save results
all_results_df = pd.DataFrame(all_cycles_results)
all_results_df["PA_gap"] = all_results_df["PA_train"] - all_results_df["PA_test"]
results_file = os.path.join(output_dir, "SVM_score1_bw_CVnested_waves.csv")
all_results_df.to_csv(results_file, index=False)

# Plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=all_results_df, x="OuterFold", y="PA_test", hue="Cycle", marker="o")
plt.title("Test Pearson Correlation Across Cycles and Folds")
plt.xlabel("Outer Fold")
plt.ylabel("PA (Test)")
plt.legend(title="Cycle", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plot_file = os.path.join(output_dir, "SVM_score1_bw_CVnested_waves.png")
plt.savefig(plot_file, dpi=300)
plt.show()
