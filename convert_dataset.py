import json
import torch
import numpy as np
from torch import nn

class AtomEmbedding(nn.Module):
    """
    Learnable embedding layer for atom types.
    
    Args:
        num_atom_types (int): Number of unique atom types
        embedding_dim (int): Dimension of the embedding vectors
    """
    def __init__(self, num_atom_types, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)

def convert_molecular_data(jsonl_file, output_file, embedding_dim=32):
    """
    Convert molecular data from JSONL format to GCN-compatible JSON format.
    Each molecule is kept as a separate graph.
    Uses learned embeddings for atom types instead of one-hot encoding.
    
    Expected JSONL format:
    Each line should be a JSON object with the structure:
    {
        "node_feat": [feat1, feat2, ...],  # List of feature values for each node
        "edge_index": [[source1, source2, ...], [target1, target2, ...]],  # List of source and target node indices
        "y": [property_value]  # List containing the molecular property value
    }
    """
    
    # First pass: determine the maximum feature value
    max_feat = -1
    print("Scanning file for maximum feature value...")
    with open(jsonl_file, 'r') as f:
        for line in f:
            try:
                mol_data = json.loads(line.strip())
                current_max = max(mol_data['node_feat'])
                max_feat = max(max_feat, current_max)
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    # Add 1 to max_feat to get the number of atom types
    num_atom_types = max_feat + 1
    print(f"Maximum atom type found: {max_feat}")
    print(f"Using {embedding_dim}-dimensional embeddings for {num_atom_types} atom types")
    
    # Initialize embedding layer
    embedding_layer = AtomEmbedding(num_atom_types, embedding_dim)
    
    # Process each molecule
    print("Converting molecules...")
    graphs = []
    
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                mol_data = json.loads(line.strip())
                
                # Convert node features to embeddings
                node_feats = torch.tensor(mol_data['node_feat'], dtype=torch.long)
                with torch.no_grad():
                    node_embeddings = embedding_layer(node_feats).numpy().tolist()
                
                # Create graph data
                graph_data = {
                    'node_features': node_embeddings,
                    'edge_index': mol_data['edge_index'],
                    'y': mol_data['y'][0]  
                }
                
                graphs.append(graph_data)
                
                if line_num % 100 == 0:
                    print(f"Processed {line_num} molecules...")
                
            except Exception as e:
                print(f"Error processing molecule {line_num}: {e}")
                continue
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(graphs, f, indent=2)
    
    print(f"Converted {len(graphs)} molecules")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python convert_dataset.py <input_jsonl> <output_json>")
        sys.exit(1)
    
    convert_molecular_data(sys.argv[1], sys.argv[2]) 