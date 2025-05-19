import torch
from torch_geometric.nn import GCNConv, global_mean_pool

class MultiLayerGCN(torch.nn.Module):
    """
    
    Args:
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden features
        out_channels (int): Number of output features
        num_layers (int): Number of GCN layers (default: 2)
        dropout (float): Dropout probability (default: 0.0)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0):
        super().__init__()
        
        # Input layer
        self.layers = torch.nn.ModuleList([
            GCNConv(in_channels, hidden_channels)
        ])
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # Final prediction layer
        self.pred = torch.nn.Linear(hidden_channels, out_channels)
        
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        """
        
        Args:
            x (torch.Tensor): Node feature matrix
            edge_index (torch.Tensor): Graph connectivity in COO format
            batch (torch.Tensor, optional): Batch assignment for each node
            
        Returns:
            torch.Tensor: Graph-level predictions
        """
        # Apply all layers except the last one with ReLU and dropout
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        
        # Last layer
        x = self.layers[-1](x, edge_index)
        
        # Global pooling 
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        
        # Final prediction
        x = self.pred(x)
        
        return x

# For backward compatibility
class TwoLayerGCN(MultiLayerGCN):
    """
    
    Args:
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden features
        out_channels (int): Number of output features
        dropout (float): Dropout probability (default: 0.0)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers=2, dropout=dropout) 