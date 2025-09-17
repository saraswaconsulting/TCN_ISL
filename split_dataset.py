#!/usr/bin/env python3
"""
Data splitting script for ISL Pose-GRU dataset.
Splits the dataset into train/val with 70-30 ratio while maintaining class balance.
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
import json

def get_all_classes_and_files(data_root):
    """
    Scan the data directory and collect all classes and their video files.
    
    Args:
        data_root (str): Path to the root data directory
    
    Returns:
        dict: Dictionary mapping class names to list of video file paths
    """
    data_path = Path(data_root)
    class_files = defaultdict(list)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for video_file in class_dir.iterdir():
                if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    class_files[class_name].append(video_file)
    
    return class_files

def split_files(file_list, train_ratio=0.7, random_seed=42):
    """
    Split a list of files into train and validation sets.
    
    Args:
        file_list (list): List of file paths
        train_ratio (float): Ratio for training set (default: 0.7)
        random_seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_files, val_files)
    """
    random.seed(random_seed)
    shuffled_files = file_list.copy()
    random.shuffle(shuffled_files)
    
    split_idx = int(len(shuffled_files) * train_ratio)
    train_files = shuffled_files[:split_idx]
    val_files = shuffled_files[split_idx:]
    
    return train_files, val_files

def create_directory_structure(output_root):
    """
    Create train and val directory structure.
    
    Args:
        output_root (str): Path to output directory
    """
    output_path = Path(output_root)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    return train_dir, val_dir

def copy_files_to_split(files, target_dir, class_name):
    """
    Copy files to the target directory maintaining class structure.
    
    Args:
        files (list): List of file paths to copy
        target_dir (Path): Target directory (train or val)
        class_name (str): Name of the class
    """
    class_dir = target_dir / class_name
    class_dir.mkdir(exist_ok=True)
    
    for file_path in files:
        target_path = class_dir / file_path.name
        shutil.copy2(file_path, target_path)

def generate_split_report(class_files, train_splits, val_splits, output_root):
    """
    Generate a detailed report of the data split.
    
    Args:
        class_files (dict): Original class files mapping
        train_splits (dict): Training split files mapping
        val_splits (dict): Validation split files mapping
        output_root (str): Output directory path
    """
    report = {
        "split_summary": {
            "total_classes": len(class_files),
            "train_ratio": 0.7,
            "val_ratio": 0.3,
            "random_seed": 42
        },
        "class_details": {}
    }
    
    total_train = 0
    total_val = 0
    
    print("\n" + "="*80)
    print("DATASET SPLIT REPORT")
    print("="*80)
    print(f"{'Class':<15} {'Total':<8} {'Train':<8} {'Val':<6} {'Train%':<8} {'Val%':<8}")
    print("-"*80)
    
    for class_name in sorted(class_files.keys()):
        total_files = len(class_files[class_name])
        train_files = len(train_splits[class_name])
        val_files = len(val_splits[class_name])
        
        train_pct = (train_files / total_files) * 100
        val_pct = (val_files / total_files) * 100
        
        total_train += train_files
        total_val += val_files
        
        report["class_details"][class_name] = {
            "total": total_files,
            "train": train_files,
            "val": val_files,
            "train_percentage": round(train_pct, 1),
            "val_percentage": round(val_pct, 1)
        }
        
        print(f"{class_name:<15} {total_files:<8} {train_files:<8} {val_files:<6} {train_pct:<7.1f}% {val_pct:<7.1f}%")
    
    print("-"*80)
    print(f"{'TOTAL':<15} {total_train + total_val:<8} {total_train:<8} {total_val:<6} {(total_train/(total_train + total_val))*100:<7.1f}% {(total_val/(total_train + total_val))*100:<7.1f}%")
    print("="*80)
    
    report["split_summary"]["total_files"] = total_train + total_val
    report["split_summary"]["total_train"] = total_train
    report["split_summary"]["total_val"] = total_val
    
    # Save report to JSON
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    report_path = output_path / "split_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")
    return report

def main():
    parser = argparse.ArgumentParser(description="Split ISL dataset into train/val splits")
    parser.add_argument("--data_root", default="data", 
                       help="Path to the root data directory (default: data)")
    parser.add_argument("--output_root", default="data_split", 
                       help="Path to output directory for split data (default: data_split)")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="Training set ratio (default: 0.7)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show split statistics without copying files")
    
    args = parser.parse_args()
    
    print(f"Starting dataset split...")
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Validation ratio: {1 - args.train_ratio}")
    print(f"Random seed: {args.random_seed}")
    print(f"Dry run: {args.dry_run}")
    
    # Get all classes and files
    print("\nScanning data directory...")
    class_files = get_all_classes_and_files(args.data_root)
    
    if not class_files:
        print("ERROR: No video files found in the data directory!")
        return
    
    print(f"Found {len(class_files)} classes with video files")
    
    # Split each class
    train_splits = {}
    val_splits = {}
    
    for class_name, files in class_files.items():
        train_files, val_files = split_files(files, args.train_ratio, args.random_seed)
        train_splits[class_name] = train_files
        val_splits[class_name] = val_files
    
    # Generate report
    report = generate_split_report(class_files, train_splits, val_splits, args.output_root)
    
    if args.dry_run:
        print("\nDRY RUN: No files were copied. Use --dry_run=False to perform actual split.")
        return
    
    # Create directory structure and copy files
    print(f"\nCreating directory structure at: {args.output_root}")
    train_dir, val_dir = create_directory_structure(args.output_root)
    
    print("Copying files to train/val directories...")
    for class_name in class_files.keys():
        print(f"  Processing class: {class_name}")
        copy_files_to_split(train_splits[class_name], train_dir, class_name)
        copy_files_to_split(val_splits[class_name], val_dir, class_name)
    
    print(f"\nDataset split completed successfully!")
    print(f"Train directory: {train_dir}")
    print(f"Val directory: {val_dir}")
    print(f"\nNext steps:")
    print(f"1. Extract keypoints: python extract_keypoints.py --data_root {args.output_root} --out_root features")
    print(f"2. Train model: python train.py --features_root features --epochs 30")

if __name__ == "__main__":
    main()