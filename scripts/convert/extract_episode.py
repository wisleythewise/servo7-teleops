#!/usr/bin/env python3
"""
Script to extract a specific episode from an HDF5 dataset.
Usage: python extract_episode.py --input path/to/dataset.hdf5 --episode-index 0 --output extracted_episode.hdf5
"""

import argparse
import h5py
import numpy as np
import os

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

def extract_episode(input_file, episode_index, output_file):
    """Extract a specific episode from the dataset."""
    print(f"Extracting episode {episode_index} from {input_file} to {output_file}")
    
    with h5py.File(input_file, 'r') as src_f:
        if 'data' not in src_f:
            raise ValueError("No 'data' group found in input file")
        
        data_group = src_f['data']
        episodes = sorted(data_group.keys())
        
        if episode_index >= len(episodes):
            raise ValueError(f"Episode index {episode_index} out of range. Found {len(episodes)} episodes.")
        
        episode_key = episodes[episode_index]
        episode_data = data_group[episode_key]
        
        print(f"Extracting episode '{episode_key}'...")
        
        # Create output file
        with h5py.File(output_file, 'w') as dst_f:
            # Copy root attributes
            for attr_name, attr_value in src_f.attrs.items():
                dst_f.attrs[attr_name] = attr_value
            
            # Create data group
            dst_data = dst_f.create_group('data')
            
            # Copy the specific episode
            copy_hdf5_structure(episode_data, dst_data, episode_key)
            
            # Also copy any other root-level groups/datasets that might be needed
            for key in src_f.keys():
                if key != 'data':  # Don't copy data group again
                    print(f"Copying additional group/dataset: {key}")
                    copy_hdf5_structure(src_f[key], dst_f, key)
        
        # Show what was extracted
        print(f"\nExtracted episode summary:")
        print(f"  Episode key: {episode_key}")
        
        if 'actions' in episode_data:
            actions_shape = episode_data['actions'].shape
            print(f"  Actions shape: {actions_shape}")
        
        if 'obs' in episode_data:
            obs_keys = list(episode_data['obs'].keys())
            print(f"  Observation keys: {obs_keys}")
            
            # Show shapes of observations
            for obs_key in obs_keys:
                obs_shape = episode_data['obs'][obs_key].shape if hasattr(episode_data['obs'][obs_key], 'shape') else "unknown"
                print(f"    {obs_key}: {obs_shape}")
        
        if 'rewards' in episode_data:
            rewards_shape = episode_data['rewards'].shape
            print(f"  Rewards shape: {rewards_shape}")
        
        if 'dones' in episode_data:
            dones_shape = episode_data['dones'].shape
            print(f"  Dones shape: {dones_shape}")
        
        print(f"\nSuccessfully extracted to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract a specific episode from HDF5 dataset")
    parser.add_argument("--input", type=str, required=True, help="Input HDF5 file path")
    parser.add_argument("--episode-index", type=int, help="Episode index to extract (0-based)")
    parser.add_argument("--output", type=str, help="Output HDF5 file path")
    parser.add_argument("--list", action="store_true", help="List all episodes and exit")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Always list episodes first
    episodes = list_episodes(args.input)
    
    if args.list:
        return
    
    if args.episode_index is None:
        print("\nPlease specify --episode-index to extract an episode")
        return
    
    if args.output is None:
        # Generate default output name
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}_episode_{args.episode_index}.hdf5"
        print(f"Using default output name: {args.output}")
    
    try:
        extract_episode(args.input, args.episode_index, args.output)
    except Exception as e:
        print(f"Error extracting episode: {e}")

if __name__ == "__main__":
    main()