#!/usr/bin/env python3
"""
Generate predictions for an already-trained CT model.
Loads test data and saves predictions for comprehensive analysis.
"""

import sys
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions for trained CT model')
    parser.add_argument('results_dir', help='Path to training results directory')
    parser.add_argument('--test-samples', type=int, default=3000,
                        help='Number of test samples per class (default: 3000)')
    return parser.parse_args()

def load_volume_batch(es_directory, cc_directory, plane='X', 
                      max_samples_per_class=3000, seed=42):
    """Load volume images for testing."""
    import glob
    
    np.random.seed(seed)
    
    print(f"\nLoading test data for plane {plane}...")
    print(f"Maximum {max_samples_per_class} samples per class")
    
    es_pattern = f'{es_directory}/*plane{plane}.npz'
    cc_pattern = f'{cc_directory}/*plane{plane}.npz'
    
    es_files = sorted(glob.glob(es_pattern))
    cc_files = sorted(glob.glob(cc_pattern))
    
    print(f"Found {len(es_files)} ES files, {len(cc_files)} CC files")
    
    np.random.shuffle(es_files)
    np.random.shuffle(cc_files)
    
    images_list = []
    labels_list = []
    
    # Load ES samples
    es_count = 0
    for f in es_files:
        if es_count >= max_samples_per_class:
            break
        try:
            data = np.load(f, allow_pickle=True)
            imgs = data['images']
            indices = np.arange(len(imgs))
            np.random.shuffle(indices)
            
            for idx in indices:
                if es_count >= max_samples_per_class:
                    break
                img = imgs[idx]
                img_array = np.array(img, dtype=np.float32)
                if img_array.shape == (208, 1242):
                    img_max = np.max(img_array)
                    if img_max > 0:
                        img_array = img_array / img_max
                    images_list.append(img_array)
                    labels_list.append(0)
                    es_count += 1
        except Exception as e:
            continue
        
        if es_count % 1000 == 0 and es_count > 0:
            print(f"  Loaded {es_count} ES samples...")
    
    # Load CC samples
    cc_count = 0
    for f in cc_files:
        if cc_count >= max_samples_per_class:
            break
        try:
            data = np.load(f, allow_pickle=True)
            imgs = data['images']
            indices = np.arange(len(imgs))
            np.random.shuffle(indices)
            
            for idx in indices:
                if cc_count >= max_samples_per_class:
                    break
                img = imgs[idx]
                img_array = np.array(img, dtype=np.float32)
                if img_array.shape == (208, 1242):
                    img_max = np.max(img_array)
                    if img_max > 0:
                        img_array = img_array / img_max
                    images_list.append(img_array)
                    labels_list.append(1)
                    cc_count += 1
        except Exception as e:
            continue
        
        if cc_count % 1000 == 0 and cc_count > 0:
            print(f"  Loaded {cc_count} CC samples...")
    
    images = np.array(images_list).reshape(-1, 208, 1242, 1)
    labels = np.array(labels_list)
    
    print(f"Total: {len(labels)} samples, Shape: {images.shape}")
    print(f"ES: {es_count}, CC: {cc_count}")
    
    return images, labels

def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    
    print("\n" + "="*80)
    print("GENERATING CT PREDICTIONS FOR COMPREHENSIVE ANALYSIS")
    print("="*80 + "\n")
    
    # Load config
    config_path = results_dir / 'config.json'
    if not config_path.exists():
        print(f"❌ No config.json found in {results_dir}")
        return 1
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Find model file
    model_files = list(results_dir.glob('*.keras'))
    if not model_files:
        print(f"❌ No .keras model file found in {results_dir}")
        return 1
    
    model_path = model_files[0]
    print(f"✓ Found model: {model_path.name}")
    
    # Extract plane from config
    plane = config.get('plane', 'X')
    es_dir = config.get('es_directory')
    cc_dir = config.get('cc_directory')
    
    if not es_dir or not cc_dir:
        print("❌ Config missing es_directory or cc_directory")
        return 1
    
    print(f"✓ Plane: {plane}")
    print(f"✓ ES dir: {es_dir}")
    print(f"✓ CC dir: {cc_dir}")
    
    # Load model
    print(f"\n📦 Loading model...")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded")
    
    # Load test data
    print(f"\n📊 Loading test data...")
    test_images, test_labels = load_volume_batch(
        es_dir, cc_dir, plane=plane,
        max_samples_per_class=args.test_samples,
        seed=42
    )
    
    # Generate predictions
    print(f"\n🔮 Generating predictions...")
    predictions = model.predict(test_images, verbose=1)
    
    # Evaluate
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"\n✓ Test Accuracy: {accuracy:.4f}")
    print(f"✓ Test Loss: {loss:.4f}")
    
    # Save predictions
    pred_file = results_dir / 'test_predictions.npz'
    np.savez(pred_file,
             predictions=predictions,
             true_labels=test_labels,
             test_images=test_images,
             energies=None)
    
    print(f"\n✅ Saved predictions to: {pred_file}")
    print(f"   - {len(test_labels)} samples")
    print(f"   - {predictions.shape[1]} classes")
    print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
