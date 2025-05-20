import torch
from torch_geometric.nn import GATConv, global_mean_pool

class MultiLayerGAT(torch.nn.Module):
    """
    
    Args:
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden features
        out_channels (int): Number of output features
        heads (int): Number of attention heads
        edge_dim (int, optional): Dimension of edge features
        num_layers (int): Number of GAT layers (default: 2)
        dropout (float): Dropout probability (default: 0.0)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, edge_dim=None, num_layers=2, dropout=0.0):
        super().__init__()
        
        # Input layer
        self.layers = torch.nn.ModuleList([
            GATConv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim)
        ])
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        
        # Output layer
        self.layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=1))  # Single head for output
        
        # Final prediction layer
        self.pred = torch.nn.Linear(hidden_channels, out_channels)
        
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        
        Args:
            x (torch.Tensor): Node feature matrix
            edge_index (torch.Tensor): Graph connectivity in COO format
            edge_attr (torch.Tensor, optional): Edge features
            batch (torch.Tensor, optional): Batch assignment for each node
            
        Returns:
            torch.Tensor: Graph-level predictions
        """
        # Apply all layers except the last one with ReLU and dropout
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_attr)
            x = torch.relu(x)
            x = self.dropout(x)
        
        # Last layer
        x = self.layers[-1](x, edge_index, edge_attr)
        
        # Global pooling to get graph-level representation
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        
        # Final prediction
        x = self.pred(x)
        
        return x

class TwoLayerGAT(MultiLayerGAT):
    """
    
    Args:
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden features
        out_channels (int): Number of output features
        heads (int): Number of attention heads
        edge_dim (int, optional): Dimension of edge features
        dropout (float): Dropout probability (default: 0.0)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, edge_dim=None, dropout=0.0):
        super().__init__(in_channels, hidden_channels, out_channels, heads=heads, edge_dim=edge_dim, num_layers=2, dropout=dropout)
