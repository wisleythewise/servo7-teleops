import h5py
import numpy as np
import os
from typing import List, Union

def remove_episodes_from_hdf5(
    input_file: str, 
    output_file: str, 
    episodes_to_remove: List[int],
    verbose: bool = True
):
    """
    Remove specified episodes from an HDF5 file and save to a new file.
    
    Args:
        input_file (str): Path to the input HDF5 file
        output_file (str): Path for the output HDF5 file
        episodes_to_remove (List[int]): List of episode indices to remove (0-based)
        verbose (bool): Whether to print progress information
    """
    
    if verbose:
        print(f"Processing {input_file}...")
        print(f"Episodes to remove: {episodes_to_remove}")
    
    # Convert to set for faster lookup
    episodes_to_remove_set = set(episodes_to_remove)
    
    with h5py.File(input_file, 'r') as source:
        with h5py.File(output_file, 'w') as dest:
            
            # Copy root level attributes
            for attr_name, attr_value in source.attrs.items():
                dest.attrs[attr_name] = attr_value
            
            # Create the data group
            data_group = dest.create_group('data')
            
            # Copy data group attributes if any
            if 'data' in source:
                for attr_name, attr_value in source['data'].attrs.items():
                    data_group.attrs[attr_name] = attr_value
            
            # Process each demo in the data group
            if 'data' in source:
                for demo_name in source['data'].keys():
                    # Extract episode number from demo_name (e.g., "demo_5" -> 5)
                    if demo_name.startswith('demo_'):
                        try:
                            episode_idx = int(demo_name.split('_')[1])
                            
                            if episode_idx in episodes_to_remove_set:
                                if verbose:
                                    print(f"Skipping episode: {demo_name} (index {episode_idx})")
                                continue
                            else:
                                if verbose:
                                    print(f"Copying episode: {demo_name} (index {episode_idx})")
                                
                                # Copy the entire demo group recursively
                                def copy_group_recursive(src_group, dest_group, group_name):
                                    new_group = dest_group.create_group(group_name)
                                    
                                    # Copy attributes
                                    for attr_name, attr_value in src_group.attrs.items():
                                        new_group.attrs[attr_name] = attr_value
                                    
                                    # Copy all items in the group
                                    for item_name, item in src_group.items():
                                        if isinstance(item, h5py.Group):
                                            copy_group_recursive(item, new_group, item_name)
                                        elif isinstance(item, h5py.Dataset):
                                            # Copy dataset with compression if it exists
                                            dataset = new_group.create_dataset(
                                                item_name, 
                                                data=item[:],
                                                compression=item.compression,
                                                compression_opts=item.compression_opts
                                            )
                                            
                                            # Copy dataset attributes
                                            for attr_name, attr_value in item.attrs.items():
                                                dataset.attrs[attr_name] = attr_value
                                
                                # Copy the demo group
                                copy_group_recursive(source['data'][demo_name], data_group, demo_name)
                        
                        except (ValueError, IndexError):
                            # If demo name doesn't follow expected format, copy it anyway
                            if verbose:
                                print(f"Copying non-standard demo: {demo_name}")
                            copy_group_recursive(source['data'][demo_name], data_group, demo_name)
    
    if verbose:
        print(f"Successfully created {output_file} with episodes {episodes_to_remove} removed.")


def inspect_episodes(file_path: str) -> List[int]:
    """
    Inspect the HDF5 file to find all episode indices.
    
    Args:
        file_path (str): Path to the HDF5 file
        
    Returns:
        List[int]: List of episode indices found in the file
    """
    episodes = set()
    
    with h5py.File(file_path, 'r') as f:
        def find_episodes(name, obj):
            path_parts = name.split('/')
            for part in path_parts:
                if part.isdigit():
                    episodes.add(int(part))
        
        f.visititems(find_episodes)
    
    return sorted(list(episodes))


def get_file_info(file_path: str):
    """
    Get basic information about the HDF5 file.
    
    Args:
        file_path (str): Path to the HDF5 file
    """
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print(f"Root keys: {list(f.keys())}")
        
        episodes = inspect_episodes(file_path)
        print(f"Episodes found: {episodes}")
        print(f"Total episodes: {len(episodes)}")


# Example usage
if __name__ == "__main__":
    # Configuration
    input_file = "./datasets/first_real_data_set.hdf5"
    output_file = "./datasets/filtered_data_set.hdf5"
    episodes_to_remove = [0, 1]  # Example: remove episodes 0, 2, and 5
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist!")
        exit(1)
    
    # Inspect the original file
    print("=== Original File Info ===")
    get_file_info(input_file)
    
    # Remove episodes
    print("\n=== Removing Episodes ===")
    remove_episodes_from_hdf5(
        input_file=input_file,
        output_file=output_file,
        episodes_to_remove=episodes_to_remove,
        verbose=True
    )
    
    # Inspect the filtered file
    print("\n=== Filtered File Info ===")
    if os.path.exists(output_file):
        get_file_info(output_file)
    else:
        print("Error: Output file was not created!")