import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from gcn import TwoLayerGCN
from data_processor import GraphDataProcessor
from torch_geometric.data import Data

def load_dataset(data_path):
    """
    Load dataset from a file path. Supports JSON, JSONL and CSV formats.
    
    Expected JSON/JSONL format:
    Each line should be a JSON object with the structure:
    {
        "node_features": [[feat1, feat2, ...], ...],  # List of feature lists for each node
        "edge_index": [[source1, source2, ...], [target1, target2, ...]],  # List of source and target node indices
        "y": [label1, label2, ...]  # List of node labels
    }
    
    Expected CSV format:
    - nodes.csv: node_id, feature1, feature2, ..., label
    - edges.csv: source_id, target_id
    
    Args:
        data_path (str): Path to the data file or directory
        
    Returns:
        dict: Dictionary containing node features, edge indices, and labels
    """
    data_path = Path(data_path)
    
    if data_path.is_file():
        if data_path.suffix.lower() == '.json':
            with open(data_path, 'r') as f:
                return json.load(f)
        elif data_path.suffix.lower() == '.jsonl':
            # For JSONL, read line by line
            node_features = []
            edge_sources = []
            edge_targets = []
            labels = []
            
            with open(data_path, 'r') as f:
                for line in f:
                    # Parse each line as a separate JSON object
                    data = json.loads(line.strip())
                    
                    # Extract features and append
                    if 'node_features' in data:
                        node_features.extend(data['node_features'])
                    
                    # Extract edge indices and append
                    if 'edge_index' in data:
                        edge_sources.extend(data['edge_index'][0])
                        edge_targets.extend(data['edge_index'][1])
                    
                    # Extract labels and append
                    if 'y' in data:
                        labels.extend(data['y'])
            
            return {
                'node_features': node_features,
                'edge_index': [edge_sources, edge_targets],
                'y': labels
            }
        else:
            raise ValueError(f"Unsupported single file format: {data_path.suffix}")
    
    elif data_path.is_dir():
        # Directory containing CSV files
        nodes_path = data_path / 'nodes.csv'
        edges_path = data_path / 'edges.csv'
        
        if not (nodes_path.exists() and edges_path.exists()):
            raise FileNotFoundError(
                "Directory must contain 'nodes.csv' and 'edges.csv'"
            )
        
        # Read nodes data
        nodes_df = pd.read_csv(nodes_path)
        # Assume last column is label, rest are features
        features = nodes_df.iloc[:, 1:-1].values.tolist()
        labels = nodes_df.iloc[:, -1].tolist()
        
        # Read edges data
        edges_df = pd.read_csv(edges_path)
        edge_index = [
            edges_df['source_id'].tolist(),
            edges_df['target_id'].tolist()
        ]
        
        return {
            'node_features': features,
            'edge_index': edge_index,
            'y': labels
        }
    
    else:
        raise FileNotFoundError(f"Path does not exist: {data_path}")

def train_model(model, graph_data, optimizer):
    """
    Train the GCN model for regression.
    """
    print("Starting training...")
    model.train()
    
    for epoch in range(200):  # 200 epochs
        optimizer.zero_grad()
        
        # Forward pass - pass x and edge_index separately
        out = model(graph_data.x, graph_data.edge_index)
        
        # MSE Loss for regression
        loss = torch.nn.functional.mse_loss(out.squeeze(), graph_data.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')
            
def main(args):
    # Load and process the data
    print("Loading data...")
    with open(args.data_path, 'r') as f:
        data_dict = json.load(f)
    
    # Convert data to PyTorch tensors
    x = torch.tensor(data_dict['node_features'], dtype=torch.float)
    edge_index = torch.tensor(data_dict['edge_index'], dtype=torch.long)
    y = torch.tensor(data_dict['y'], dtype=torch.float)
    
    # Create PyG Data object
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    
    # Initialize model (input features = number of features in data)
    print("Initializing model...")
    model = TwoLayerGCN(
        in_channels=x.size(1),
        hidden_channels=args.hidden_size,
        out_channels=1  # Changed to 1 for regression
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Train the model
    train_model(model, graph_data, optimizer)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Path to the input data file')
    parser.add_argument('--hidden-size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    
    args = parser.parse_args()
    main(args) 