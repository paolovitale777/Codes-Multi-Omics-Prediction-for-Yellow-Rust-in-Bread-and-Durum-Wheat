
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load and Prepare Data
# -----------------------------
pheno = pd.read_csv("All_BLUEs_BW.csv", index_col=0)
waves = pheno.iloc[:, [3, 7, 15, 19, 23]]

target_col = "score_20240809"
valid_idx = pheno[target_col].notna()
X = waves.loc[valid_idx].values.astype(np.float32)
y = pheno.loc[valid_idx, target_col].values.astype(np.float32)

y = np.log1p(y) 
scaler = StandardScaler()
X = scaler.fit_transform(X)
# -----------------------------
# Hyperparameter Grid (Smaller + Expanded)
# -----------------------------


# Hyperparameter Grid (Smaller + Expanded)
# -----------------------------
param_grid = {
    "hidden_sizes": [[64, 32], [128, 64], [32, 16]],
    "learning_rate": [0.0001,0.001,],
    "batch_size": [64, 128, 32],
    "dropout": [0.3, 0.1], 
    "weight_decay": [0, 1e-5], 
    "epochs": [50],
    "L1": [0, 1e-5]
}
grid = list(product(*param_grid.values()))

# -----------------------------
# MLP Model (2 hidden layers)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, x):
        return self.network(x).squeeze()

# -----------------------------
# Output Dir
# -----------------------------
output_dir = "nested_cv_mlp3_results"
os.makedirs(output_dir, exist_ok=True)
all_cycles_results = []

#to store the losses for each cycle
loss_tracking = []

# -----------------------------
# Nested CV + Early Stopping
# -----------------------------
for cycle in range(1, 11):
    print(f"\n===== Cycle {cycle} =====")
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=1000 + cycle)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        inner_cv = KFold(n_splits=4, shuffle=True, random_state=2000 + cycle)
        tuning_results = []

        for hidden_sizes, lr, batch_size, dropout, wd, epochs, L1 in grid:
            fold_PAs, fold_MSPEs, fold_MAPEs = [], [], []

            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
                X_train, X_val = X_train_outer[inner_train_idx], X_train_outer[inner_val_idx]
                y_train, y_val = y_train_outer[inner_train_idx], y_train_outer[inner_val_idx]

                model = MLP(X.shape[1], hidden_sizes, dropout)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                criterion = nn.MSELoss()
                train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)

                best_val_loss = float("inf")
                patience, counter = 10, 0
                min_delta = 1e-4
                lambda_l1=L1


                for epoch in range(epochs):
                    model.train()
                    epoch_train_losses = []

                    for xb, yb in train_loader:
                        optimizer.zero_grad()
                        outputs = model(xb)
                        mse_loss = criterion(outputs, yb)

                        # Calculate L1 penalty
                        l1_norm = sum(p.abs().sum() for p in model.parameters())
                        total_loss = mse_loss + lambda_l1 * l1_norm # Add L1 penalty

                        total_loss.backward()
                        optimizer.step()
                        epoch_train_losses.append(total_loss.item())

                    train_loss = np.mean(epoch_train_losses)

                    model.eval()
                    with torch.no_grad():
                        val_loss = criterion(model(torch.tensor(X_val)), torch.tensor(y_val)).item()

                        # Save loss info
                    loss_tracking.append({
                        "Cycle": cycle,
                        "OuterFold": outer_fold + 1,
                        "ParamCombo": grid.index((hidden_sizes, lr, batch_size, dropout, wd, epochs, L1)),
                        "Epoch": epoch + 1,
                        "TrainLoss": train_loss,
                        "ValLoss": val_loss
                    })

                    # Early stopping
                    if best_val_loss - val_loss > min_delta:
                        best_val_loss = val_loss
                        counter = 0
                    else:
                        counter += 1
                    if counter >= patience:
                        break

                model.eval()
                with torch.no_grad():
                    y_val_pred = model(torch.tensor(X_val)).numpy()

                fold_PAs.append(pearsonr(y_val, y_val_pred)[0])
                fold_MSPEs.append(mean_squared_error(y_val, y_val_pred))
                fold_MAPEs.append(mean_absolute_error(y_val, y_val_pred))

            tuning_results.append({
                "hidden_sizes": hidden_sizes,
                "learning_rate": lr,
                "batch_size": batch_size,
                "dropout": dropout,
                "weight_decay": wd,
                "L1":L1,
                "epochs": epochs,
                "PA": np.mean(fold_PAs),
                "MSPE": np.mean(fold_MSPEs),
                "MAPE": np.mean(fold_MAPEs)
            })

        best_params = pd.DataFrame(tuning_results).loc[lambda df: df.MAPE.idxmin()]

        model = MLP(X.shape[1], best_params.hidden_sizes, best_params.dropout)
        optimizer = optim.Adam(model.parameters(), lr=best_params.learning_rate, weight_decay=best_params.weight_decay)
        criterion = nn.MSELoss()
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train_outer), torch.tensor(y_train_outer)), batch_size=int(best_params.batch_size), shuffle=True)

        for epoch in range(int(best_params.epochs)):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_train_pred = model(torch.tensor(X_train_outer)).numpy()
            y_test_pred = model(torch.tensor(X_test_outer)).numpy()

        all_cycles_results.append({
            "Cycle": cycle,
            "OuterFold": outer_fold + 1,
            "PA_train": pearsonr(y_train_outer, y_train_pred)[0],
            "PA_test": pearsonr(y_test_outer, y_test_pred)[0],
            "MSPE": mean_squared_error(y_test_outer, y_test_pred),
            "MAPE": mean_absolute_error(y_test_outer, y_test_pred),
            "Best_hidden_sizes": best_params.hidden_sizes,
            "Best_learning_rate": best_params.learning_rate,
            "Best_dropout": best_params.dropout,
            "Best_weight_decay": best_params.weight_decay,
            "Best_batch_size": best_params.batch_size,
            "Best_L1": best_params.L1
        })


# Save and plot
all_results_df = pd.DataFrame(all_cycles_results)
all_results_df["PA_gap"] = all_results_df["PA_train"] - all_results_df["PA_test"]
results_file = os.path.join(output_dir, "MLP3_PA_score1_bw_waves.csv")
all_results_df.to_csv(results_file, index=False)

loss_df = pd.DataFrame(loss_tracking)
loss_df.to_csv(os.path.join(output_dir, "MLP3_loss_per_epoch_score1_bw_waves.csv"), index=False)



