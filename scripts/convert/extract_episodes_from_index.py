#!/usr/bin/env python3
"""
Script to extract all episodes from a specified index onwards from an HDF5 dataset.
Usage: python extract_episodes_from_index.py --input path/to/dataset.hdf5 --start-index 2 --output filtered_dataset.hdf5
"""

import argparse
import h5py
import json
import os

def copy_hdf5_structure(source_item, dest_group, name):
    """Recursively copy HDF5 structure from source to destination."""
    if isinstance(source_item, h5py.Group):
        # Create group and copy attributes
        new_group = dest_group.create_group(name)
        for attr_name, attr_value in source_item.attrs.items():
            new_group.attrs[attr_name] = attr_value
        
        # Recursively copy contents
        for key in source_item.keys():
            copy_hdf5_structure(source_item[key], new_group, key)
    
    elif isinstance(source_item, h5py.Dataset):
        # Copy dataset
        dest_group.create_dataset(name, data=source_item[()])
        
        # Copy attributes
        for attr_name, attr_value in source_item.attrs.items():
            dest_group[name].attrs[attr_name] = attr_value

def list_episodes(file_path):
    """List all available episodes in the dataset."""
    print(f"=== Episodes in {file_path} ===")
    
    with h5py.File(file_path, 'r') as f:
        if 'data' not in f:
            print("No 'data' group found in file")
            return []
        
        data_group = f['data']
        episodes = sorted(data_group.keys())
        
        print(f"Found {len(episodes)} episodes:")
        for i, episode_key in enumerate(episodes):
            episode = data_group[episode_key]
            
            # Get episode info
            episode_info = {}
            if 'actions' in episode and hasattr(episode['actions'], 'shape'):
                episode_info['length'] = episode['actions'].shape[0]
            if 'obs' in episode:
                episode_info['obs_keys'] = list(episode['obs'].keys())
            
            print(f"  [{i}] {episode_key}: {episode_info}")
        
        return episodes

def extract_episodes_from_index(input_file, start_index, output_file, add_metadata=True):
    """Extract all episodes from start_index onwards."""
    print(f"Extracting episodes from index {start_index} onwards from {input_file} to {output_file}")
    
    with h5py.File(input_file, 'r') as src_f:
        if 'data' not in src_f:
            raise ValueError("No 'data' group found in input file")
        
        data_group = src_f['data']
        episodes = sorted(data_group.keys())
        
        if start_index >= len(episodes):
            raise ValueError(f"Start index {start_index} out of range. Found {len(episodes)} episodes.")
        
        # Get episodes to extract
        episodes_to_extract = episodes[start_index:]
        print(f"Will extract {len(episodes_to_extract)} episodes: {episodes_to_extract}")
        
        # Create output file
        with h5py.File(output_file, 'w') as dst_f:
            # Copy root attributes
            for attr_name, attr_value in src_f.attrs.items():
                dst_f.attrs[attr_name] = attr_value
            
            # Create data group
            dst_data = dst_f.create_group('data')
            
            # Copy selected episodes
            for episode_key in episodes_to_extract:
                print(f"Copying episode: {episode_key}")
                episode_data = data_group[episode_key]
                copy_hdf5_structure(episode_data, dst_data, episode_key)
            
            # Add required metadata if missing (for Isaac Lab compatibility)
            if add_metadata:
                # Check if env_args exists, if not add it
                if 'env_args' not in dst_data.attrs:
                    env_args = {
                        "task": "LeIsaac-SO101-PickOrange-Mimic-v0",
                        "num_envs": 1,
                        "sim_device": "cuda",
                        "device": "cuda"
                    }
                    dst_data.attrs["env_args"] = json.dumps(env_args)
                    print(f"Added env_args metadata")
                
                # Add other commonly required attributes
                if 'env_name' not in dst_data.attrs:
                    dst_data.attrs["env_name"] = "LeIsaac-SO101-PickOrange-Mimic-v0"
                if 'date' not in dst_data.attrs:
                    dst_data.attrs["date"] = "2025-01-15"
                if 'repository_version' not in dst_data.attrs:
                    dst_data.attrs["repository_version"] = "isaac-lab-1.0.0"
                if 'num_episodes' not in dst_data.attrs:
                    dst_data.attrs["num_episodes"] = len(episodes_to_extract)
                
                print(f"Added Isaac Lab compatibility metadata")
            
            # Also copy any other root-level groups/datasets that might be needed
            for key in src_f.keys():
                if key != 'data':  # Don't copy data group again
                    print(f"Copying additional group/dataset: {key}")
                    copy_hdf5_structure(src_f[key], dst_f, key)
        
        # Show what was extracted
        print(f"\nExtraction summary:")
        print(f"  Start index: {start_index}")
        print(f"  Episodes extracted: {len(episodes_to_extract)}")
        print(f"  Episode keys: {episodes_to_extract}")
        
        # Show detailed info for each extracted episode
        with h5py.File(input_file, 'r') as src_f:
            data_group = src_f['data']
            for episode_key in episodes_to_extract:
                episode_data = data_group[episode_key]
                
                episode_info = []
                if 'actions' in episode_data:
                    actions_shape = episode_data['actions'].shape
                    episode_info.append(f"actions: {actions_shape}")
                
                if 'obs' in episode_data:
                    obs_keys = list(episode_data['obs'].keys())
                    episode_info.append(f"obs: {len(obs_keys)} types")
                
                if 'states' in episode_data:
                    episode_info.append("states: included")
                
                print(f"    {episode_key}: {', '.join(episode_info)}")
        
        print(f"\nSuccessfully extracted to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract episodes from specified index onwards from HDF5 dataset")
    parser.add_argument("--input", type=str, required=True, help="Input HDF5 file path")
    parser.add_argument("--start-index", type=int, help="Start index (inclusive) - extract this episode and all following ones")
    parser.add_argument("--output", type=str, help="Output HDF5 file path")
    parser.add_argument("--list", action="store_true", help="List all episodes and exit")
    parser.add_argument("--no-metadata", action="store_true", help="Don't add Isaac Lab metadata")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Always list episodes first
    episodes = list_episodes(args.input)
    
    if args.list:
        return
    
    if args.start_index is None:
        print("\nPlease specify --start-index to extract episodes")
        return
    
    if args.output is None:
        # Generate default output name
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}_from_index_{args.start_index}.hdf5"
        print(f"Using default output name: {args.output}")
    
    try:
        extract_episodes_from_index(
            args.input, 
            args.start_index, 
            args.output, 
            add_metadata=not args.no_metadata
        )
    except Exception as e:
        print(f"Error extracting episodes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()