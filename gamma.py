import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Linear, LayerNorm
from torch_geometric.datasets import (
    Planetoid,  # For Cora, CiteSeer, PubMed
    HeterophilousGraphDataset,  # For roman-empire, amazon-ratings, minesweeper, tolokers, questions
    LINKXDataset,  # For penn94, genius
    Flickr,  # For flickr
    SNAPDataset,  # For soc-pokec, twitch-gamers
    DeezerEurope,  # For deezer-europe
)
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import argparse
from GAMMA_layer import GAMMA

def set_seed(seed=102):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GAMMA_GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, powers=[0, 1, 2], num_layers=3, num_iterations=1):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GAMMA(in_channels, hidden_channels, powers=powers, num_iterations=1))
        self.norms.append(LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GAMMA(hidden_channels, hidden_channels, powers=powers, num_iterations=1))
            self.norms.append(LayerNorm(hidden_channels))
        
        # Output layer
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.3, training=self.training)
        
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.dropout(x, p=0.9, training=self.training)
            
        return F.log_softmax(self.lin(x), dim=1)

def create_split_masks(num_nodes, train_ratio=0.48, val_ratio=0.32, seed=42):
    """Create train/val/test masks with specified ratios."""
    set_seed(seed)
    indices = torch.randperm(num_nodes)
    
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    return train_mask, val_mask, test_mask

def load_dataset(dataset_name):
    """Load the specified dataset."""
    if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name, transform=NormalizeFeatures())
    elif dataset_name.lower() in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        dataset = HeterophilousGraphDataset(root=f'/tmp/{dataset_name}', name=dataset_name, transform=NormalizeFeatures())
    elif dataset_name.lower() == 'penn94':
        dataset = LINKXDataset(root='/tmp/Penn94', name='penn94')
    elif dataset_name.lower() == 'genius':
        dataset = LINKXDataset(root='/tmp/Genius', name='genius', transform=NormalizeFeatures())
    elif dataset_name.lower() == 'flickr':
        dataset = Flickr(root='/tmp/Flickr', transform=NormalizeFeatures())
    elif dataset_name.lower() == 'soc-pokec':
        dataset = SNAPDataset(root='/tmp/SNAP', name='soc-pokec', transform=NormalizeFeatures())
    elif dataset_name.lower() == 'twitch-gamers':
        dataset = SNAPDataset(root='/tmp/SNAP', name='twitch-gamers', transform=NormalizeFeatures())
    elif dataset_name.lower() == 'deezer-europe':
        dataset = DeezerEurope(root='/tmp/DeezerEurope', transform=NormalizeFeatures())
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset

def train_node_classifier(model, graph, optimizer, scheduler, criterion, n_epochs=500):
    """Train the model with early stopping."""
    best_val_acc = 0
    best_model_state = None
    patience = 200
    patience_counter = 0
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            pred = model(graph.x, graph.edge_index).argmax(dim=1)
            val_correct = (pred[graph.val_mask] == graph.y[graph.val_mask]).sum()
            val_acc = int(val_correct) / int(graph.val_mask.sum())
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {val_acc:.3f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def eval_node_classifier(model, graph, mask):
    """Evaluate the model on the specified mask."""
    model.eval()
    with torch.no_grad():
        pred = model(graph.x, graph.edge_index).argmax(dim=1)
        correct = (pred[mask] == graph.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
    return acc

def train_and_evaluate(data, model, optimizer, scheduler, criterion, device):
    """Train and evaluate model on a single split."""
    model = train_node_classifier(model, data, optimizer, scheduler, criterion)
    test_acc = eval_node_classifier(model, data, data.test_mask)
    return test_acc

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train GAMMA on various datasets')
    parser.add_argument('--dataset', type=str, default='Cora',
                      choices=['Cora', 'CiteSeer', 'PubMed', 'roman-empire', 'amazon-ratings', 
                              'minesweeper', 'tolokers', 'questions', 'penn94', 'genius', 
                              'flickr', 'soc-pokec', 'twitch-gamers', 'deezer-europe'],
                      help='Dataset to use')
    parser.add_argument('--num_splits', type=int, default=10,
                      help='Number of random splits to use')
    parser.add_argument('--hidden_channels', type=int, default=32,
                      help='Number of hidden channels')
    parser.add_argument('--num_iterations', type=int, default=2,
                      help='num_iterations for GAMMA')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of GAMMA layers')
    args = parser.parse_args()
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    powers = [0, 1, 2]
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    data = dataset[0]
    
    test_accuracies = []
    
    for split in range(args.num_splits):
        print(f"\nSplit {split + 1}/{args.num_splits}")
        set_seed(seed=42 + split)  # Different seed for each split
        
        # Create split masks
        train_mask, val_mask, test_mask = create_split_masks(data.num_nodes, seed=42 + split)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        # Move to device
        data = data.to(device)
        
        # Initialize model
        model = GAMMA_GNN(
            in_channels=dataset.num_node_features,
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            powers=powers,
            num_layers=args.num_layers,
            num_iterations=args.num_iterations,
        ).to(device)
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=50, 
            min_lr=1e-6, verbose=True
        )
        criterion = CrossEntropyLoss()
        
        # Train and evaluate
        test_acc = train_and_evaluate(data, model, optimizer, scheduler, criterion, device)
        test_accuracies.append(test_acc)
        print(f'Test Accuracy: {test_acc:.3f}')
    
    # Calculate statistics
    mean_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies)
    
    print(f"\n{'='*50}")
    print(f"Results over {args.num_splits} splits:")
    print(f"Mean Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"All accuracies: {[f'{acc:.4f}' for acc in test_accuracies]}")
    print(f"Min: {min(test_accuracies):.4f}, Max: {max(test_accuracies):.4f}")

if __name__ == "__main__":
    main()