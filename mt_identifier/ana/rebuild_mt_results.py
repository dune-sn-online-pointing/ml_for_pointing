#!/usr/bin/env python3
"""Rebuild evaluation artifacts for an MT training run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_DIR / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import general_purpose_libs as gpl
import classification_libs as cl
from tensorflow import keras
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild MT evaluation artifacts")
    parser.add_argument("results_dir", help="Directory that contains model outputs (config.json, *.h5)")
    parser.add_argument("--config", dest="config_path", help="Override path to config.json")
    parser.add_argument("--model", dest="model_path", help="Override path to trained model file")
    parser.add_argument("--plane", help="Override plane (defaults to config)")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples while rebuilding")
    parser.add_argument("--data-dir", action="append", dest="data_dirs", help="Override data directories (repeatable)")
    return parser.parse_args()


def load_test_set(data_dirs, plane, dataset_parameters, output_folder):
    """Return concatenated test set across all data directories."""
    tests = []
    for idx, data_dir in enumerate(data_dirs, start=1):
        print(f"\nLoading directory {idx}/{len(data_dirs)}: {data_dir}")
        _, _, test = cl.prepare_data_from_npz(
            data_dir=data_dir,
            plane=plane,
            dataset_parameters=dataset_parameters,
            output_folder=output_folder,
        )
        tests.append(test)
    if len(tests) == 1:
        return tests[0]
    print("\nConcatenating test sets...")
    images = np.concatenate([t[0] for t in tests], axis=0)
    labels = np.concatenate([t[1] for t in tests], axis=0)
    return images, labels


def main():
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    config_path = Path(args.config_path).expanduser().resolve() if args.config_path else results_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_parameters = dict(config.get("dataset_parameters", {}))
    if args.max_samples is not None:
        dataset_parameters["max_samples"] = args.max_samples

    plane = args.plane or config.get("plane") or config.get("dataset_parameters", {}).get("plane") or "X"

    if args.data_dirs:
        data_dirs = args.data_dirs
    else:
        raw_dirs = config.get("data_directories")
        if not raw_dirs:
            raise ValueError("No data_directories found in config and none provided via --data-dir")
        data_dirs = raw_dirs if isinstance(raw_dirs, list) else [raw_dirs]

    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else results_dir / f"{config['model_name']}.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("\n==============================")
    print("REBUILDING MT RESULTS")
    print("==============================")
    print(f"Results dir: {results_dir}")
    print(f"Config:      {config_path}")
    print(f"Model:       {model_path}")
    print(f"Plane:       {plane}")
    print(f"Data dirs:   {len(data_dirs)} directory(ies)")

    test_images, test_labels = load_test_set(data_dirs, plane, dataset_parameters, str(results_dir))
    model = keras.models.load_model(model_path)

    evaluation = cl.test_model(
        model,
        (test_images, test_labels),
        str(results_dir),
        label_names=["Background", "Main Track"],
    )

    results_payload = {
        "config": {
            "model": {"name": config.get("model_name", "mt_model")},
            "task_label": config.get("task_label", "mt_identifier"),
            "dataset": {
                "plane": plane,
                "train_fraction": dataset_parameters.get("train_fraction"),
                "val_fraction": dataset_parameters.get("val_fraction"),
                "test_fraction": dataset_parameters.get("test_fraction"),
                "balance_data": dataset_parameters.get("balance_data"),
                "max_samples": dataset_parameters.get("max_samples"),
                "data_directories": data_dirs,
            },
        },
        "metrics": evaluation.get("metrics", {}),
        "history": {},
        "artifacts": evaluation.get("artifacts", {}),
    }

    gpl.write_results_json(str(results_dir), results_payload)
    print("\n✓ Results artifacts rebuilt successfully")


if __name__ == "__main__":
    main()
