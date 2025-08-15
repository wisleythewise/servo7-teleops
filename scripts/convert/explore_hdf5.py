#!/usr/bin/env python3
"""
Script to explore HDF5 dataset structure and show all keys/values nested.
Usage: python explore_hdf5.py --file path/to/dataset.hdf5
"""

import argparse
import h5py
import numpy as np

def explore_hdf5_structure(group, prefix="", max_depth=10, current_depth=0):
    """Recursively explore HDF5 structure."""
    if current_depth > max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    
    for key in group.keys():
        item = group[key]
        full_path = f"{prefix}/{key}" if prefix else key
        
        if isinstance(item, h5py.Group):
            print(f"{prefix}{key}/ (Group, {len(item)} items)")
            explore_hdf5_structure(item, prefix + "  ", max_depth, current_depth + 1)
        elif isinstance(item, h5py.Dataset):
            shape_str = f"shape={item.shape}" if item.shape else "scalar"
            dtype_str = f"dtype={item.dtype}"
            size_str = f"size={item.size}"
            
            print(f"{prefix}{key} (Dataset: {shape_str}, {dtype_str}, {size_str})")
            
            # Show first few values for small datasets or scalar values
            if item.size <= 10:
                try:
                    value = item[()]
                    if isinstance(value, (str, bytes)):
                        print(f"{prefix}  Value: {repr(value)}")
                    elif isinstance(value, np.ndarray) and value.size <= 5:
                        print(f"{prefix}  Value: {value}")
                    elif not isinstance(value, np.ndarray):
                        print(f"{prefix}  Value: {value}")
                except Exception as e:
                    print(f"{prefix}  (could not read value: {e})")
            elif len(item.shape) > 0 and item.shape[0] <= 5:
                try:
                    # Show first few elements for small arrays
                    sample = item[:min(3, item.shape[0])]
                    print(f"{prefix}  Sample: {sample}")
                except Exception as e:
                    print(f"{prefix}  (could not read sample: {e})")

def show_dataset_summary(file_path):
    """Show a summary of the dataset structure."""
    print(f"=== HDF5 Dataset Structure: {file_path} ===\n")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Root keys: {list(f.keys())}")
            print(f"Root attributes: {dict(f.attrs)}")
            print()
            
            # Explore the structure
            explore_hdf5_structure(f)
            
            # Special handling for common structures
            if 'data' in f:
                print(f"\n=== 'data' group details ===")
                data_group = f['data']
                if hasattr(data_group, 'keys'):
                    for episode_key in sorted(data_group.keys()):
                        episode = data_group[episode_key]
                        print(f"Episode {episode_key}:")
                        if 'obs' in episode:
                            print(f"  - obs keys: {list(episode['obs'].keys())}")
                        if 'actions' in episode:
                            actions_shape = episode['actions'].shape if hasattr(episode['actions'], 'shape') else "unknown"
                            print(f"  - actions shape: {actions_shape}")
                        if 'rewards' in episode:
                            rewards_shape = episode['rewards'].shape if hasattr(episode['rewards'], 'shape') else "unknown"
                            print(f"  - rewards shape: {rewards_shape}")
                        if 'dones' in episode:
                            dones_shape = episode['dones'].shape if hasattr(episode['dones'], 'shape') else "unknown"
                            print(f"  - dones shape: {dones_shape}")
                        
                        # Show episode length
                        if 'actions' in episode and hasattr(episode['actions'], 'shape'):
                            print(f"  - episode length: {episode['actions'].shape[0]} steps")
                        print()
    
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Explore HDF5 dataset structure")
    parser.add_argument("--file", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum depth to explore")
    
    args = parser.parse_args()
    
    show_dataset_summary(args.file)

if __name__ == "__main__":
    main()