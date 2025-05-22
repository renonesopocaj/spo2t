import argparse
import os
import subprocess  # new for dashboard
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import optuna
from optuna.trial import Trial

# Added for plotting and evaluation exports
import matplotlib
matplotlib.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Spo2Dataset(Dataset):
    """Simple Dataset wrapping Spo2 windows and binary labels from CSV files.

    The dataset expects two ``.csv`` files in the data_path:
        data.csv   -> CSV file with N_samples rows and 25 columns (Spo2 values). No header.
        labels.csv -> CSV file with N_samples rows and 1 column (binary labels 0/1). No header.
    """

    def __init__(self, data_path: str):
        # Load data from CSV files using pandas
        data_df = pd.read_csv(os.path.join(data_path, "data.csv"), header=None)
        labels_df = pd.read_csv(os.path.join(data_path, "labels.csv"), header=None)

        data_np = data_df.values
        labels_np = labels_df.values.squeeze() # Squeeze to make it 1D

        if data_np.ndim != 2 or data_np.shape[1] != 25:
            raise ValueError(
                f"Expected data from data.csv to be of shape (N, 25) but got {data_np.shape}. Ensure windows of 25 Spo2 samples.")
        if labels_np.ndim != 1 or labels_np.shape[0] != data_np.shape[0]:
            raise ValueError(f"Labels shape mismatch with data. Expected labels from labels.csv to be (N,) but got {labels_np.shape}")

        self.data = torch.from_numpy(data_np).float()
        self.labels = torch.from_numpy(labels_np).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # reshape to (25, 1) -> only SpO2 values
        window = self.data[idx].unsqueeze(-1)  # (25, 1)
        label = self.labels[idx]
        return window, label


class ApneaMLP(nn.Module):
    """Multi-layer fully connected perceptron for binary classification."""

    def __init__(self, input_dim: int, hidden_units: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []

        # optional initial batch norm on input features (flattened)
        # layers.append(nn.BatchNorm1d(input_dim))

        prev = input_dim
        for i, h in enumerate(hidden_units):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))  # binary logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # x shape: (batch, time, 1)
        x = x.view(x.size(0), -1)  # flatten to (batch, 25)
        return self.net(x).squeeze(-1)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (logits.sigmoid() >= 0.5).float()
    return (preds == targets).float().mean().item()


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer) -> Tuple[float, float]:
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # Calculate batch accuracy
        batch_acc = accuracy_from_logits(logits.detach(), y)
        
        # Update running metrics
        running_loss += loss.item() * x.size(0)
        running_acc += batch_acc * x.size(0)
        
        # Print batch-level metrics
        print(f"  Batch {batch_idx+1}/{len(loader)}: Loss={loss.item():.4f}, Acc={batch_acc:.4f}")
        
    return running_loss / len(loader.dataset), running_acc / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion) -> Tuple[float, float]:
    model.eval()
    loss_sum, acc_sum = 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            acc_sum += accuracy_from_logits(logits, y) * x.size(0)
    return loss_sum / len(loader.dataset), acc_sum / len(loader.dataset)


def objective(trial: Trial,
              train_val_dataset: Dataset,
              test_dataset: Dataset,
              device: torch.device,
              patience: int,
              test_eval_freq: int):
    # Hyperparameter search space
    n_layers = trial.suggest_int("n_layers", 2, 5)
    hidden_units = [trial.suggest_int(f"n_units_l{i}", 64, 512, step=64) for i in range(n_layers)]
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)

    # Create model, criterion, optimizer, & scheduler
    input_dim = 25  # SpO2 only, no label feature
    model = ApneaMLP(input_dim=input_dim, hidden_units=hidden_units, dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Data loaders: split train_val_dataset into train and validation subsets every trial so that
    # hyper-parameter tuning selects models that generalise without ever peeking at the test set.
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_ds, val_ds = random_split(train_val_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    # Loader for hold-out test set (constant across trials)
    test_loader = DataLoader(test_dataset, batch_size=256)

    epochs = trial.suggest_int("epochs", 10, 50)

    # Lists to track metrics per epoch (for export)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    best_val_acc = 0.0
    early_stop_counter = 0
    best_model_state = None  # will hold weights of the best epoch

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping check within a trial
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            early_stop_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stop_counter += 1

        # Periodic evaluation on the test set (if enabled)
        if test_eval_freq > 0 and ((epoch + 1) % test_eval_freq == 0 or epoch == epochs - 1):
            test_loss, test_acc = evaluate(model, test_loader, criterion)
            print(f"    >>> Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

        if early_stop_counter >= patience:
            break

    # Ensure we have a model to save
    if best_model_state is None:
        best_model_state = model.state_dict()

    # Save best model weights for this trial
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", f"model_trial_{trial.number}.pt")
    torch.save(best_model_state, model_path)

    # Attach extra info to the trial for later retrieval
    trial.set_user_attr("model_path", model_path)
    trial.set_user_attr("train_loss", train_losses)
    trial.set_user_attr("val_loss", val_losses)
    trial.set_user_attr("train_acc", train_accs)
    trial.set_user_attr("val_acc", val_accs)

    return best_val_acc


def parse_args():
    parser = argparse.ArgumentParser(description="MLP for sleep apnea detection with Optuna tuning.")
    parser.add_argument("--data-path", required=True, help="Directory containing data.csv and labels.csv files.")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--study-name", default="apnea_mlp", help="Optuna study name.")
    parser.add_argument("--storage", default="sqlite:///optuna.db", help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    parser.add_argument("--dashboard", action="store_true", help="Launch Optuna dashboard on http://localhost:8080")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--test-split", type=float, default=0.1, help="Fraction of data reserved as hold-out test set (e.g. 0.1 = 10%).")
    parser.add_argument("--test-eval-freq", type=int, default=5, help="Evaluate on the test set every N epochs (0 to disable).")
    return parser.parse_args()


def main():
    args = parse_args()
    full_dataset = Spo2Dataset(args.data_path)

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study = optuna.create_study(direction="maximize", study_name=args.study_name, storage=args.storage,
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

    # ---------------------------------------------------------------------
    # Hold-out test split (fixed for the whole optimisation)
    # ---------------------------------------------------------------------
    test_size = int(args.test_split * len(full_dataset))
    test_size = max(1, test_size)  # ensure at least one sample
    train_val_size = len(full_dataset) - test_size

    # Use a fixed seed so every run gets the same test set
    generator = torch.Generator().manual_seed(42)
    train_val_ds, test_ds = random_split(full_dataset, [train_val_size, test_size], generator=generator)

    study.optimize(lambda trial: objective(trial,
                                          train_val_ds,
                                          test_ds,
                                          device,
                                          args.patience,
                                          args.test_eval_freq),
                   n_trials=args.trials)

    # Optionally launch dashboard (after optimization so DB is populated). If requested, keep process alive.
    if args.dashboard:
        try:
            print("\nStarting Optuna dashboard at http://localhost:8080 ...")
            subprocess.Popen(["optuna-dashboard", args.storage, "--host", "0.0.0.0", "--port", "8080"])
            print("Dashboard running in background. Press Ctrl+C to stop.")
        except FileNotFoundError:
            print("optuna-dashboard command not found. Install via `pip install optuna-dashboard`.")

    print("\nStudy statistics:")
    print("  Number of finished trials:", len(study.trials))
    print("  Best trial:")
    best_trial = study.best_trial

    print(f"    Value (best validation accuracy): {best_trial.value:.4f}")
    print("    Params:")
    for key, value in best_trial.params.items():
        print(f"      {key}: {value}")

    # ------------------------------------------------------------------
    # Export learning curves, confusion matrix and save best model path
    # ------------------------------------------------------------------

    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    # Retrieve metric lists from best trial
    train_loss_hist = best_trial.user_attrs.get("train_loss")
    val_loss_hist = best_trial.user_attrs.get("val_loss")
    train_acc_hist = best_trial.user_attrs.get("train_acc")
    val_acc_hist = best_trial.user_attrs.get("val_acc")

    if train_loss_hist and val_loss_hist:
        plt.figure()
        plt.plot(train_loss_hist, label="Train")
        plt.plot(val_loss_hist, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curves")
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, "loss_curve.png"))
        plt.close()

    if train_acc_hist and val_acc_hist:
        plt.figure()
        plt.plot(train_acc_hist, label="Train")
        plt.plot(val_acc_hist, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy Curves")
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, "accuracy_curve.png"))
        plt.close()

    # Re-create model with best hyperparams and load weights
    n_layers_best = best_trial.params["n_layers"]
    hidden_units_best = [best_trial.params[f"n_units_l{i}"] for i in range(n_layers_best)]
    dropout_best = best_trial.params["dropout"]

    best_model = ApneaMLP(input_dim=25, hidden_units=hidden_units_best, dropout=dropout_best).to(device)
    best_model_path = best_trial.user_attrs.get("model_path")
    if best_model_path and os.path.isfile(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        best_model.load_state_dict(state_dict)
    else:
        print("Warning: Saved model file not found, using randomly initialised weights for evaluation.")

    # Confusion matrix on test set
    best_model.eval()
    all_preds, all_targets = [], []
    test_loader_final = DataLoader(test_ds, batch_size=256)
    with torch.no_grad():
        for x, y in test_loader_final:
            x = x.to(device)
            logits = best_model(x)
            preds = (logits.sigmoid() >= 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Apnea", "Apnea"])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Also save confusion matrix counts to CSV
    cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
    cm_df.to_csv(os.path.join(artifacts_dir, "confusion_matrix.csv"))

    print(f"\nArtifacts saved in '{artifacts_dir}':")
    print("  • loss_curve.png / accuracy_curve.png")
    print("  • confusion_matrix.png / confusion_matrix.csv")
    print(f"  • Best model weights: {best_model_path}")


if __name__ == "__main__":
    main() 