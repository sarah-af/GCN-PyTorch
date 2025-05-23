import torch
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
from torch_scatter import scatter_mean

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
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        
        # Input layer
        self.layers.append(GATConv(in_channels, hidden_channels, add_self_loops=False))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels, hidden_channels, add_self_loops=False))
        
        # Output layer
        self.layers.append(GATConv(hidden_channels, out_channels, add_self_loops=False))
        
        self.dropout = dropout
    
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
        # If no batch is provided, create one
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        print(f"Debug - Input x shape: {x.shape}")
        print(f"Debug - Batch shape: {batch.shape}")
        print(f"Debug - Batch unique values: {torch.unique(batch)}")
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            print(f"Debug - After layer {i}, x shape: {x.shape}")
            if i < len(self.layers) - 1:  # Don't apply dropout after last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global mean pooling to get one prediction per graph
        x = scatter_mean(x, batch, dim=0)
        print(f"Debug - After scatter_mean, x shape: {x.shape}")
        
        # Ensure output has shape [batch_size, 1]
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        print(f"Debug - Final output shape: {x.shape}")
        
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
        super().__init__(in_channels, hidden_channels, out_channels, num_layers=2, dropout=dropout)
