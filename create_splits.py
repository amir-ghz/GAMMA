import os
import numpy as np
import torch
from torch_geometric.datasets import WikipediaNetwork, WebKB, Actor
from torch_geometric.transforms import NormalizeFeatures

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_splits_directory():
    splits_dir = "splits"
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)

def save_splits(dataset_name, train_mask, val_mask, test_mask, split_idx):
    splits_dir = "splits"
    split_file = os.path.join(splits_dir, f"{dataset_name.lower()}_split_0.6_0.2_{split_idx}.npz")
    np.savez(split_file, 
             train_mask=train_mask.numpy(),
             val_mask=val_mask.numpy(),
             test_mask=test_mask.numpy())

def create_dataset_splits(dataset_name, dataset, num_splits=10):
    """
    Create multiple train/val/test splits for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset: PyG dataset object
        num_splits: Number of different splits to create
    
    Split ratios: 60% train, 20% val, 20% test
    """
    data = dataset[0]
    n_nodes = data.num_nodes
    
    for split_idx in range(num_splits):
        # Different seed for each split for reproducibility
        set_seed(102 + split_idx)
        
        # Create random permutation of nodes
        indices = torch.randperm(n_nodes)
        
        # Calculate split sizes (60/20/20)
        train_size = int(0.6 * n_nodes)
        val_size = int(0.2 * n_nodes)
        
        # Create masks
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        # Assign indices to masks
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        # Save splits
        save_splits(dataset_name, train_mask, val_mask, test_mask, split_idx)
        print(f"Created split {split_idx + 1}/{num_splits} for {dataset_name}")

def main():
    create_splits_directory()
    
    # Create splits for Chameleon
    print("Creating splits for Chameleon...")
    chameleon_dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon', transform=NormalizeFeatures())
    create_dataset_splits("Chameleon", chameleon_dataset)
    
    # Create splits for Cornell
    print("Creating splits for Cornell...")
    cornell_dataset = WebKB(root='/tmp/Cornell', name='Cornell', transform=NormalizeFeatures())
    create_dataset_splits("Cornell", cornell_dataset)

    # Create splits for Wisconsin
    print("Creating splits for Wisconsin...")
    Wisconsin_dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin', transform=NormalizeFeatures())
    create_dataset_splits("Wisconsin", Wisconsin_dataset)
    
    # Create splits for Texas
    print("Creating splits for Texas...")
    texas_dataset = WebKB(root='/tmp/Texas', name='Texas', transform=NormalizeFeatures())
    create_dataset_splits("Texas", texas_dataset)
    
    # Create splits for Squirrel
    print("Creating splits for Squirrel...")
    squirrel_dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel', transform=NormalizeFeatures())
    create_dataset_splits("Squirrel", squirrel_dataset)

    # Create splits for Actor
    print("Creating splits for Actor...")
    actor_dataset = Actor(root='/tmp/Actor', transform=NormalizeFeatures())
    create_dataset_splits("Actor", actor_dataset)
    
    print("All splits have been created successfully!")

if __name__ == "__main__":
    main() 