import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Linear, BatchNorm1d, LayerNorm
from torch_geometric.datasets import WikipediaNetwork, WebKB, Actor
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import os
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
        self.convs.append(GAMMA(in_channels, hidden_channels, powers=powers, num_iterations=num_iterations))
        self.norms.append(LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GAMMA(hidden_channels, hidden_channels, powers=powers, num_iterations=num_iterations))
            self.norms.append(LayerNorm(hidden_channels))
        
        # Output layer
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.dropout(x, p=0.8, training=self.training)
            
        return F.log_softmax(self.lin(x), dim=1)

def load_fixed_splits(dataset_name, split_idx):
    """Load fixed splits."""
    split_file = os.path.join('splits', f'{dataset_name.lower()}_split_0.6_0.2_{split_idx}.npz')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file {split_file} not found.")
    data = np.load(split_file)
    train_mask = torch.tensor(data['train_mask'].flatten(), dtype=torch.bool)
    val_mask = torch.tensor(data['val_mask'].flatten(), dtype=torch.bool)
    test_mask = torch.tensor(data['test_mask'].flatten(), dtype=torch.bool)
    return train_mask, val_mask, test_mask

def train_node_classifier(model, graph, optimizer, scheduler, criterion, n_epochs=500):
    best_val_acc = 0
    best_model_state = None
    patience = 50
    patience_counter = 0
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            pred = model(graph.x, graph.edge_index).argmax(dim=1)
            val_correct = (pred[graph.val_mask] == graph.y[graph.val_mask]).sum()
            val_acc = int(val_correct) / int(graph.val_mask.sum())
            
            # Update learning rate scheduler based on validation accuracy
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
    model.eval()
    with torch.no_grad():
        pred = model(graph.x, graph.edge_index).argmax(dim=1)
        correct = (pred[mask] == graph.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
    return acc

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GAMMA GNN Node Classification')
    parser.add_argument('--dataset', type=str, default='Texas', 
                       choices=['Chameleon', 'Cornell', 'Wisconsin', 'Texas', 'Squirrel', 'Actor'],
                       help='Dataset name')
    parser.add_argument('--hidden_channels', type=int, default=32,
                       help='Number of hidden channels')
    parser.add_argument('--num_splits', type=int, default=10,
                       help='Number of splits to run')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of GAMMA layers')
    parser.add_argument('--num_iterations', type=int, default=2,
                       help='Number of iterations for GAMMA layer')
    
    args = parser.parse_args()
    
    # Configuration
    dataset_name = args.dataset
    hidden_channels = args.hidden_channels
    num_splits = args.num_splits
    num_layers = args.num_layers
    num_iterations = args.num_iterations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # GAMMA specific configurations
    powers = [0, 1, 2]  # Powers for GAMMA layer
    
    # Load dataset
    if dataset_name == "Chameleon":
        dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon', transform=NormalizeFeatures())
    elif dataset_name == "Cornell":
        dataset = WebKB(root='/tmp/Cornell', name='Cornell', transform=NormalizeFeatures())
    elif dataset_name == "Wisconsin":
        dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin', transform=NormalizeFeatures())
    elif dataset_name == "Texas":
        dataset = WebKB(root='/tmp/Texas', name='Texas', transform=NormalizeFeatures())
    elif dataset_name == "Squirrel":
        dataset = WikipediaNetwork(root='/tmp/Squirrel', name='Squirrel', transform=NormalizeFeatures())
    elif dataset_name == "Actor":
        dataset = Actor(root='/tmp/Actor', transform=NormalizeFeatures())
    
    data = dataset[0]
    
    test_accuracies = []
    
    for split in range(num_splits):
        print(f"\nSplit {split + 1}/{num_splits}")
        set_seed(seed=42 + split)
        
        # Load fixed splits
        train_mask, val_mask, test_mask = load_fixed_splits(dataset_name, split)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        # Move to device
        data = data.to(device)
        
        # Initialize model
        model = GAMMA_GNN(
            in_channels=dataset.num_node_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
            powers=powers,
            num_layers=num_layers,
            num_iterations=num_iterations
        ).to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0.0001)  # Same as Chameleon setting

        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=50, 
            min_lr=1e-6)
        criterion = CrossEntropyLoss()
        
        # Train model
        model = train_node_classifier(model, data, optimizer, scheduler, criterion)
        
        # Evaluate
        test_acc = eval_node_classifier(model, data, data.test_mask)
        test_accuracies.append(test_acc)
        print(f'Test Accuracy: {test_acc:.3f}')
    
    # Calculate statistics
    mean_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies)
    
    print(f"\n{'='*50}")
    print(f"Results over {num_splits} splits:")
    print(f"Mean Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"All accuracies: {[f'{acc:.4f}' for acc in test_accuracies]}")
    print(f"Min: {min(test_accuracies):.4f}, Max: {max(test_accuracies):.4f}")

if __name__ == "__main__":
    main()