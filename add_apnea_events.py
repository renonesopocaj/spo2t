import pandas as pd
import numpy as np
import os
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Add synthetic apnea events (label=1) to dataset")
    parser.add_argument("--ratio", type=float, default=0.05, 
                        help="Desired ratio of positive (apnea) samples (0.0-1.0)")
    parser.add_argument("--data-path", default="dataset", 
                        help="Directory containing data.csv and labels.csv")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    labels_path = os.path.join(args.data_path, "labels.csv")
    
    # Read current labels
    print(f"Reading labels from {labels_path}...")
    labels_df = pd.read_csv(labels_path, header=None)
    labels = labels_df[0].values
    
    # Count current positives
    n_total = len(labels)
    n_positives = np.sum(labels)
    n_negatives = n_total - n_positives
    
    print(f"Current distribution: {n_total} total samples")
    print(f"  - Negative (non-apnea): {n_negatives} ({n_negatives/n_total:.2%})")
    print(f"  - Positive (apnea): {n_positives} ({n_positives/n_total:.2%})")
    
    # Calculate how many positives to add
    target_positives = int(args.ratio * n_total)
    additional_positives = max(0, target_positives - n_positives)
    
    if additional_positives == 0:
        print(f"Already have enough positive samples to meet target ratio of {args.ratio:.2%}")
        return
    
    print(f"Adding {additional_positives} positive samples to reach target ratio of {args.ratio:.2%}")
    
    # Randomly select negative samples to convert to positive
    negative_indices = np.where(labels == 0)[0]
    indices_to_change = np.random.choice(negative_indices, size=additional_positives, replace=False)
    
    # Change selected indices to positive
    labels[indices_to_change] = 1
    
    # Save updated labels
    new_labels_df = pd.DataFrame(labels)
    backup_path = labels_path + ".backup"
    print(f"Backing up original labels to {backup_path}")
    os.rename(labels_path, backup_path)
    
    print(f"Saving updated labels to {labels_path}")
    new_labels_df.to_csv(labels_path, index=False, header=False)
    
    # Verify new distribution
    n_positives_new = np.sum(labels)
    print(f"New distribution:")
    print(f"  - Negative (non-apnea): {n_total - n_positives_new} ({(n_total - n_positives_new)/n_total:.2%})")
    print(f"  - Positive (apnea): {n_positives_new} ({n_positives_new/n_total:.2%})")
    
if __name__ == "__main__":
    main() 