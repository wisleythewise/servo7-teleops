import h5py
import numpy as np

# Open and inspect the dataset
with h5py.File('./datasets/filtered_data_set.hdf5', 'r') as f:
    print("Root level keys:", list(f.keys()))
    
    # Go one level deeper for each root key
    for key in f.keys():
        print(f"\n{key} contains:", list(f[key].keys()))
        
        # Go another level deeper for each subkey
        for subkey in f[key].keys():
            item = f[key][subkey]
            if hasattr(item, 'keys'):  # Check if it's a group
                print(f"  {key}/{subkey} contains:", list(item.keys()))
                
                # # Go one more level deeper
                # for subsubkey in item.keys():
                #     subitem = item[subsubkey]
                #     if hasattr(subitem, 'keys'):  # Check if it's a group
                #         print(f"    {key}/{subkey}/{subsubkey} contains:", list(subitem.keys()))
                #     else:  # It's a dataset
                #         print(f"    {key}/{subkey}/{subsubkey} is a dataset with shape:", subitem.shape)
            else:  # It's a dataset
                print(f"  {key}/{subkey} is a dataset with shape:", item.shape)