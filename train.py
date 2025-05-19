import torch
import torch.nn.functional as F
import json
import argparse
import numpy as np
import os
from pathlib import Path
from gcn_model import MultiLayerGCN
from data_processor import GraphDataProcessor
from torch_geometric.data import Data, Batch

def split_data(graphs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    
    Args:
        graphs (list): List of graph dictionaries
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_graphs, val_graphs, test_graphs)
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, 
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create random permutation of indices
    num_graphs = len(graphs)
    indices = torch.randperm(num_graphs)
    
    # Calculate split sizes
    train_size = int(num_graphs * train_ratio)
    val_size = int(num_graphs * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Split graphs
    train_graphs = [graphs[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    test_graphs = [graphs[i] for i in test_indices]
    
    return train_graphs, val_graphs, test_graphs

def save_checkpoint(model, optimizer, epoch, train_graphs, val_graphs, test_graphs, args, checkpoint_dir='checkpoints'):
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_graphs': train_graphs,
        'val_graphs': val_graphs,
        'test_graphs': test_graphs,
        'args': args
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['train_graphs'], checkpoint['val_graphs'], checkpoint['test_graphs']

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def convert_to_pyg_data(graph_dict, device):
    """Convert a single graph dictionary to PyG Data object."""
    x = torch.tensor(graph_dict['node_features'], dtype=torch.float).to(device)
    edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long).to(device)
    y = torch.tensor(graph_dict['y'], dtype=torch.float).view(-1).to(device)
    return Data(x=x, edge_index=edge_index, y=y)

def train_model(model, train_graphs, val_graphs, test_graphs, optimizer, args):
    """
    
    Args:
        model (MultiLayerGCN): The GCN model
        train_graphs (list): List of training graph dictionaries
        val_graphs (list): List of validation graph dictionaries
        test_graphs (list): List of test graph dictionaries
        optimizer (torch.optim.Optimizer): The optimizer
        args: Command line arguments
    """
    device = get_device()
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    model.train()
    best_val_loss = float('inf')
    best_model_state = None
    start_epoch = 0
    
    # Load checkpoint if specified
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        start_epoch, train_graphs, val_graphs, test_graphs = load_checkpoint(
            args.resume, model, optimizer
        )
        print(f"Resuming from epoch {start_epoch}")
    
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_train_loss = 0
            for graph_dict in train_graphs:
                data = convert_to_pyg_data(graph_dict, device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.mse_loss(out.squeeze(), data.y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for graph_dict in val_graphs:
                    data = convert_to_pyg_data(graph_dict, device)
                    out = model(data.x, data.edge_index)
                    val_loss = F.mse_loss(out.squeeze(), data.y)
                    total_val_loss += val_loss.item()
            
            avg_train_loss = total_train_loss / len(train_graphs)
            avg_val_loss = total_val_loss / len(val_graphs)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                # Save best model checkpoint
                save_checkpoint(model, optimizer, epoch, train_graphs, val_graphs, test_graphs, args)
            
            # Save periodic checkpoint
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(model, optimizer, epoch, train_graphs, val_graphs, test_graphs, args)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        save_checkpoint(model, optimizer, epoch, train_graphs, val_graphs, test_graphs, args)
        print("Checkpoint saved. You can resume training later using --resume")
        return
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for graph_dict in test_graphs:
            data = convert_to_pyg_data(graph_dict, device)
            out = model(data.x, data.edge_index)
            test_loss = F.mse_loss(out.squeeze(), data.y)
            total_test_loss += test_loss.item()
    
    avg_test_loss = total_test_loss / len(test_graphs)
    print(f'\nFinal Test Loss: {avg_test_loss:.4f}')

def main(args):
    # Load and process the data
    print("Loading data...")
    with open(args.data_path, 'r') as f:
        graphs = json.load(f)
    
    # Split data into train/val/test sets
    print("Splitting data...")
    train_graphs, val_graphs, test_graphs = split_data(
        graphs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Print split sizes
    print(f"Training graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")
    
    # Get input feature dimension from first graph
    input_dim = len(graphs[0]['node_features'][0])
    
    # Initialize model
    print("Initializing model...")
    model = MultiLayerGCN(
        in_channels=input_dim,
        hidden_channels=args.hidden_size,
        out_channels=1,  # For regression
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Train the model
    train_model(model, train_graphs, val_graphs, test_graphs, optimizer, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Path to the input data file')
    parser.add_argument('--hidden-size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio of training data')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Ratio of validation data')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Ratio of test data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to use for training')
    
    args = parser.parse_args()
    main(args)

# def example_usage():
#     """Example of how to use the GCN model with dictionary input."""
#     # Example dictionary input
#     sample_data = {
#         'node_features': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # 3 nodes with 3 features each
#         'edge_index': [[0, 1, 1, 2], [1, 0, 2, 1]],  # Bidirectional edges between nodes
#         'y': [0, 1, 2]  # Node labels
#     }
    
#     # Initialize data processor
#     processor = GraphDataProcessor(normalize_features=True)
    
#     # Convert dictionary to PyG Data object
#     graph_data = processor.process_dict_to_graph(sample_data)
    
#     # Create model (3 input features, 16 hidden features, 3 output classes)
#     model = TwoLayerGCN(
#         in_channels=3,
#         hidden_channels=16,
#         out_channels=3
#     )
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
#     graph_data.train_mask = torch.ones(graph_data.num_nodes, dtype=torch.bool)
    
#     train_model(model, graph_data, optimizer)

# if __name__ == "__main__":
#     example_usage() 