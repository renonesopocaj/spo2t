import argparse
import os
import pandas as pd
import numpy as np


def process_patient_group(df_group: pd.DataFrame, window_size: int, mean: float, std: float) -> tuple[list[list[float]], list[int]]:
    """Generate windows and labels from a single patient's time series.

    Each SpO2 value is z-score normalised using the *global* mean and std computed over the full dataset.
    """

    # Normalise SpO2 (vectorised) -> zero mean, unit variance
    spo2_norm = (df_group["spo2"].values - mean) / std
    label_series = df_group["label"].values

    n_samples = len(spo2_norm)
    windows, window_labels = [], []
    for start in range(0, n_samples - window_size + 1, window_size):
        end = start + window_size
        window_spo2 = spo2_norm[start:end]
        if len(window_spo2) < window_size:
            break  # ignore incomplete window at end
        window_label = int(label_series[start:end].max())  # 1 if any apnea event inside window
        windows.append(window_spo2.tolist())
        window_labels.append(window_label)
    return windows, window_labels


def main():
    parser = argparse.ArgumentParser(description="Prepare data.csv and labels.csv from raw spo2_1hz_all CSV.")
    parser.add_argument("--input", required=True, help="Path to spo2_1hz_all.csv")
    parser.add_argument("--output-dir", default="dataset", help="Directory to write data.csv and labels.csv")
    parser.add_argument("--window", type=int, default=25, help="Window size (seconds)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read CSV
    print("Loading CSV ...")
    df = pd.read_csv(args.input)

    required_cols = {"spo2", "label", "time", "patient"}
    if not required_cols.issubset(df.columns):
        # patient/time could be missing; only spo2 and label strictly required
        required_cols = {"spo2", "label"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Input CSV must contain at least columns {required_cols} but has {df.columns}")

    # If patient/time exist, sort by them to keep temporal order per patient
    if "patient" in df.columns and "time" in df.columns:
        df = df.sort_values(["patient", "time"])

    # Compute global mean and std for SpO2 normalisation
    mean_spo2 = df["spo2"].mean()
    std_spo2 = df["spo2"].std(ddof=0)
    if std_spo2 == 0:
        raise ValueError("Standard deviation of SpO2 is zero. Cannot normalise.")

    all_windows, all_labels = [], []

    if "patient" in df.columns:
        group_iterator = df.groupby("patient")
    else:
        # treat entire dataframe as single group if no patient column
        group_iterator = [(None, df)]

    for _, group in group_iterator:
        wins, labs = process_patient_group(group, args.window, mean_spo2, std_spo2)
        all_windows.extend(wins)
        all_labels.extend(labs)

    print(f"Generated {len(all_windows)} windows of size {args.window} seconds.")
    print(f"SpO2 normalisation: mean={mean_spo2:.3f}, std={std_spo2:.3f}")

    # Save CSVs (no header)
    data_path = os.path.join(args.output_dir, "data.csv")
    labels_path = os.path.join(args.output_dir, "labels.csv")

    pd.DataFrame(all_windows).to_csv(data_path, index=False, header=False)
    pd.DataFrame(all_labels).to_csv(labels_path, index=False, header=False)

    print(f"Saved features to {data_path} and labels to {labels_path}.")


if __name__ == "__main__":
    main() 