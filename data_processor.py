import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

class GraphDataProcessor:
    def __init__(self, normalize_features=False):
        """
        
        Args:
            normalize_features (bool): Whether to normalize node features
        """
        self.normalize_features = normalize_features
        self.scaler = StandardScaler() if normalize_features else None
        
    def process_dict_to_graph(self, data_dict):
        """
        
        Expected dictionary format:
        {
            'node_features': list or numpy array of node features,
            'edge_index': list of [source_nodes, target_nodes],
            'edge_attr': (optional) list or numpy array of edge features,
            'y': (optional) list or numpy array of node/graph labels
        }
        
        Args:
            data_dict (dict): Input dictionary containing graph information
            
        Returns:
            Data: PyTorch Geometric Data object
        """
        # Convert node features to tensor
        x = torch.tensor(np.array(data_dict['node_features']), dtype=torch.float32)
        
        # Normalize features if requested
        if self.normalize_features:
            x = torch.tensor(
                self.scaler.fit_transform(x.numpy()), 
                dtype=torch.float32
            )
        
        # Convert edge index to tensor
        edge_index = torch.tensor(
            data_dict['edge_index'], 
            dtype=torch.long
        )
        
        # Create the basic graph
        graph_data = Data(x=x, edge_index=edge_index)
        
        # Add edge attributes if they exist
        if 'edge_attr' in data_dict:
            edge_attr = torch.tensor(
                data_dict['edge_attr'],
                dtype=torch.float32
            )
            graph_data.edge_attr = edge_attr
        
        # Add labels if they exist
        if 'y' in data_dict:
            y = torch.tensor(data_dict['y'])
            graph_data.y = y
        
        return graph_data
    
    def process_batch(self, dict_list):
        """
        
        Args:
            dict_list (list): List of dictionaries containing graph information
            
        Returns:
            list: List of PyG Data objects
        """
        return [self.process_dict_to_graph(d) for d in dict_list] 