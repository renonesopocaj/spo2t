import argparse
import os
import torch
import optuna

# Reuse model definition
from mlp_apnea_optuna import ApneaMLP

def load_best_trial(study_name: str, storage: str, trial_number: int = None):
    study = optuna.load_study(study_name=study_name, storage=storage)
    if trial_number is not None:
        trials = study.get_trials()
        trial = next((t for t in trials if t.number == trial_number), None)
        if trial is None:
            raise ValueError(f"Trial {trial_number} not found in study {study_name}.")
        return trial
    return study.best_trial

def export_model_to_onnx(best_trial, output_path: str):
    # Retrieve hyperparameters and model path stored in user attrs
    n_layers = best_trial.params["n_layers"]
    hidden_units = [best_trial.params[f"n_units_l{i}"] for i in range(n_layers)]
    dropout = best_trial.params["dropout"]
    model_path = best_trial.user_attrs.get("model_path")
    if model_path is None or not os.path.isfile(model_path):
        raise FileNotFoundError("Saved model weights not found. Ensure training script was executed and artifacts present.")

    # Create model and load weights
    model = ApneaMLP(input_dim=25, hidden_units=hidden_units, dropout=dropout)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input: (batch, time, 1)
    dummy_input = torch.randn(1, 25, 1)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export best ApneaMLP model from Optuna study to ONNX.")
    parser.add_argument("--storage", default="sqlite:///optuna.db", help="Optuna storage URL")
    parser.add_argument("--study-name", default="apnea_mlp", help="Optuna study name")
    parser.add_argument("--output", default="artifacts/apnea_mlp.onnx", help="Output ONNX file path")
    parser.add_argument("--trial-number", type=int, help="Export a specific trial model (e.g., model_trial_0)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    best_trial = load_best_trial(args.study_name, args.storage, args.trial_number)
    export_model_to_onnx(best_trial, args.output)


if __name__ == "__main__":
    main() 