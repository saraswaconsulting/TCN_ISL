#!/usr/bin/env python3
"""
Script to merge two class_to_idx.json files into a complete mapping.
Combines the first set of classes (0-14) with the second set (15-83).
"""

import json
from pathlib import Path

def merge_class_mappings():
    # Load the first JSON (classes 0-14)
    first_json_path = Path("features_all/class_to_idx.json")
    with open(first_json_path, 'r') as f:
        first_mapping = json.load(f)
    
    # Load the second JSON (classes that should be 15-83)
    second_json_path = Path("class_to_idx.json")
    with open(second_json_path, 'r') as f:
        second_mapping = json.load(f)
    
    print("📋 Original mappings:")
    print(f"First file: {len(first_mapping)} classes (0-{len(first_mapping)-1})")
    print(f"Second file: {len(second_mapping)} classes (currently 0-{len(second_mapping)-1})")
    
    # Create the merged mapping
    merged_mapping = {}
    
    # Add first set (keep indices 0-14)
    for class_name, idx in first_mapping.items():
        merged_mapping[class_name] = idx
    
    # Add second set (renumber to start from 15)
    offset = len(first_mapping)  # This will be 15
    for class_name, idx in second_mapping.items():
        new_idx = idx + offset
        merged_mapping[class_name] = new_idx
    
    print(f"\n✅ Merged mapping:")
    print(f"Total classes: {len(merged_mapping)}")
    print(f"Index range: 0-{len(merged_mapping)-1}")
    
    # Show sample of the mapping
    print(f"\n📊 Sample mapping:")
    for i, (class_name, idx) in enumerate(merged_mapping.items()):
        if i < 5 or i >= len(merged_mapping) - 5:
            print(f"  {class_name}: {idx}")
        elif i == 5:
            print("  ...")
    
    # Save the merged mapping
    output_path = Path("features_all/class_to_idx_complete.json")
    with open(output_path, 'w') as f:
        json.dump(merged_mapping, f, indent=2)
    
    print(f"\n💾 Merged mapping saved to: {output_path}")
    
    # Also update the features_all version
    with open(first_json_path, 'w') as f:
        json.dump(merged_mapping, f, indent=2)
    
    print(f"📝 Updated original file: {first_json_path}")
    
    return merged_mapping

if __name__ == "__main__":
    print("🔄 Merging class mappings...")
    merged_mapping = merge_class_mappings()
    print("🎉 Merge completed successfully!")