import torch
import json
import sys
from train import train_model
from gcn import TwoLayerGCN
from data_processor import GraphDataProcessor

def load_custom_data(json_file):
    """Load custom data from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def main():
    if len(sys.argv) != 5:
        print("Usage: python train_custom.py <json_data_file> <num_features> <hidden_size> <num_classes>")
        sys.exit(1)
    
    # Get command line arguments
    json_file = sys.argv[1]
    num_features = int(sys.argv[2])
    hidden_size = int(sys.argv[3])
    num_classes = int(sys.argv[4])
    
    # Load and process data
    custom_data = load_custom_data(json_file)
    processor = GraphDataProcessor(normalize_features=True)
    graph_data = processor.process_dict_to_graph(custom_data)
    
    # Create model
    model = TwoLayerGCN(
        in_channels=num_features,
        hidden_channels=hidden_size,
        out_channels=num_classes
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Add training mask (assuming all nodes are for training)
    graph_data.train_mask = torch.ones(graph_data.num_nodes, dtype=torch.bool)
    
    # Train the model
    train_model(model, graph_data, optimizer)

if __name__ == "__main__":
    main() 