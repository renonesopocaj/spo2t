import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:  # graceful fallback if seaborn missing
    sns = None
    print("[WARNING] seaborn not installed – box plots will use raw matplotlib styling.")
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exploratory data-analysis for SpO₂ windows. "
            "Computes per-window trend features and compares their discriminative power with absolute-value features."
        )
    )
    parser.add_argument("--data", required=True, help="Path to data.csv produced by prepare_csv_windows.py")
    parser.add_argument("--labels", required=True, help="Path to labels.csv produced by prepare_csv_windows.py")
    parser.add_argument(
        "--mean",
        type=float,
        default=None,
        help=(
            "Global mean SpO₂ used during normalisation (e.g. 93.815). "
            "If omitted, windows are assumed to be already in absolute units."
        ),
    )
    parser.add_argument(
        "--std",
        type=float,
        default=None,
        help=(
            "Global std SpO₂ used during normalisation (e.g. 8.207). "
            "If omitted, windows are assumed to be already in absolute units."
        ),
    )
    parser.add_argument(
        "--output-dir", default="eda", help="Directory where all EDA artifacts will be saved"
    )
    parser.add_argument("--sample", type=int, default=10000, help="Random subset of windows to speed-up EDA (0 = all)")
    return parser.parse_args()


def recover_absolute(window: np.ndarray, mean: float | None, std: float | None) -> np.ndarray:
    """Convert z-score normalised window back to absolute SpO₂ units if mean/std are given."""
    if mean is not None and std is not None:
        return window * std + mean
    return window


def compute_features(window: np.ndarray, abs_window: np.ndarray) -> List[float]:
    """Derive simple shape/trend and absolute features from a window."""

    # Trend-related features (unit-invariant)
    baseline_start = abs_window[:3].mean()
    baseline_end = abs_window[-3:].mean()
    min_val = abs_window.min()

    drop_mag = baseline_start - min_val  # amplitude of the dip
    recovery_mag = baseline_end - min_val  # how much recovery within the window
    time_of_min = abs_window.argmin()  # index where min occurs
    duration_below_95 = (abs_window < 95).sum()  # number of seconds SpO₂ < 95

    # Absolute features
    mean_val = abs_window.mean()
    std_val = abs_window.std()

    return [
        baseline_start,
        baseline_end,
        min_val,
        drop_mag,
        recovery_mag,
        time_of_min,
        duration_below_95,
        mean_val,
        std_val,
    ]


def feature_names() -> List[str]:
    return [
        "baseline_start",
        "baseline_end",
        "min_val",
        "drop_mag",
        "recovery_mag",
        "time_of_min",
        "duration_below_95",
        "mean_val",
        "std_val",
    ]


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    artifact_files: list[str] = []  # keep track of every artifact we save

    print("Loading CSV files …")
    data_df = pd.read_csv(args.data, header=None)
    labels = pd.read_csv(args.labels, header=None).squeeze("columns").astype(int)

    if len(data_df) != len(labels):
        raise ValueError("data and labels have different number of rows")

    # Optionally subsample to speed up interactive sessions
    if args.sample and args.sample > 0 and args.sample < len(data_df):
        sample_idx = np.random.choice(len(data_df), size=args.sample, replace=False)
        data_df = data_df.iloc[sample_idx].reset_index(drop=True)
        labels = labels.iloc[sample_idx].reset_index(drop=True)
        print(f"Sub-sampling to {len(data_df)} windows for faster analysis …")

    # Compute features
    means, stds = [], []
    features: list[list[float]] = []
    for row in data_df.itertuples(index=False):
        win = np.asarray(row, dtype=float)
        abs_win = recover_absolute(win, args.mean, args.std)
        feat_vec = compute_features(win, abs_win)
        features.append(feat_vec)

    feat_df = pd.DataFrame(features, columns=feature_names())
    feat_df["label"] = labels.values

    # Save per-window features for external analysis
    features_path = os.path.join(args.output_dir, "window_features.csv")
    feat_df.to_csv(features_path, index=False)
    artifact_files.append(features_path)

    print("Feature dataframe built:")
    print(feat_df.head())

    # 1. Descriptive statistics
    desc_path = os.path.join(args.output_dir, "feature_descriptive_stats.csv")
    feat_df.groupby("label").describe().to_csv(desc_path)
    print(f"Saved descriptive stats to {desc_path}")
    artifact_files.append(desc_path)

    # 2. Box plots for selected features
    if sns:
        sns.set(style="whitegrid")
    for col in ["min_val", "mean_val", "drop_mag", "duration_below_95"]:
        plt.figure(figsize=(6, 4))
        if sns:
            sns.boxplot(x="label", y=col, data=feat_df)
        else:
            # rudimentary matplotlib fallback
            feat_df.boxplot(column=col, by="label")
            plt.suptitle("")  # remove automatic title
        plt.title(f"{col} vs label")
        plt.xlabel("label")
        plt.savefig(os.path.join(args.output_dir, f"box_{col}.png"), bbox_inches="tight")
        plt.close()
        artifact_files.append(os.path.join(args.output_dir, f"box_{col}.png"))

    # 3. Mutual information to gauge feature relevance
    mi_scores = mutual_info_score(labels, feat_df["min_val"].round(1))  # one feature example
    print(f"Mutual information (min_val vs label): {mi_scores:.4f}")

    # 4. Simple logistic regression to compare absolute vs trend features
    X = feat_df.drop(columns=["label"])
    y = feat_df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    print("LogReg test accuracy: {:.3f}".format(clf.score(X_test, y_test)))

    coef_df = pd.Series(clf.coef_[0], index=X.columns).sort_values(ascending=False)
    coef_path = os.path.join(args.output_dir, "logreg_feature_weights.csv")
    coef_df.to_csv(coef_path)
    print(f"Saved logistic regression feature weights to {coef_path}")
    artifact_files.append(coef_path)

    # 5. Plot hypothetical sequences provided in the prompt
    hypothetical_1 = np.array([98, 98, 98, 95, 95, 95, 95, 98, 98, 98])
    hypothetical_2 = np.array([98, 98, 98, 97, 95, 95, 97, 98, 98, 98])

    plt.figure(figsize=(7, 3))
    plt.plot(hypothetical_1, label="seq 1")
    plt.plot(hypothetical_2, label="seq 2")
    plt.legend()
    plt.title("Hypothetical sequences")
    plt.ylabel("SpO₂ (%)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "hypothetical_sequences.png"), bbox_inches="tight")
    plt.close()
    artifact_files.append(os.path.join(args.output_dir, "hypothetical_sequences.png"))

    # Compute and save their trend features
    hypo_sequences = [hypothetical_1, hypothetical_2]
    hypo_feat_rows: list[list[float]] = []
    for i, seq in enumerate(hypo_sequences, start=1):
        feats = compute_features(seq, seq)
        hypo_feat_rows.append(feats)
        print(f"Sequence {i} features →", dict(zip(feature_names(), feats)))

    hypo_df = pd.DataFrame(hypo_feat_rows, columns=feature_names(), index=["seq1", "seq2"])
    hypo_path = os.path.join(args.output_dir, "hypothetical_sequence_features.csv")
    hypo_df.to_csv(hypo_path)
    artifact_files.append(hypo_path)

    # Create a zip archive with all artifacts for easy sharing
    zip_path = os.path.join(args.output_dir, "eda_results.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in artifact_files:
            # store files with just basename inside the zip
            zf.write(fp, arcname=os.path.basename(fp))
    print("EDA finished. Artefacts saved in:", args.output_dir)
    print("Saved artifacts ({} files) →".format(len(artifact_files)))
    for fp in artifact_files:
        print("  •", fp)
    print("A zipped bundle is available at:", zip_path)


if __name__ == "__main__":
    main() 