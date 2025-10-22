import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Linear, LayerNorm
from torch_geometric.datasets import WikipediaNetwork, WebKB, Actor
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GATConv, GCNConv, MixHopConv, JumpingKnowledge, APPNP, SGConv, GCN2Conv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, degree, remove_self_loops
from torch_sparse import SparseTensor
import torch_sparse
from torch_scatter import scatter
import scipy.sparse
import numpy as np
import argparse
import os
from Bernpro import Bern_prop
from GAMMA_layer import GAMMA

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
        x = F.dropout(x, p=0.5, training=self.training)
        
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.dropout(x, p=0.5, training=self.training)
            
        return F.log_softmax(self.lin(x), dim=1)
    

class H2GCN(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            num_layers=2,  # This will be used as 'k' parameter
            dropout=0.5,
            use_relu=True
    ):
        super(H2GCN, self).__init__()
        self.dropout = dropout
        self.k = num_layers - 1  # Convert num_layers to k (k=1 means 2 layers)
        self.act = F.relu if use_relu else lambda x: x
        self.use_relu = use_relu
        self.w_embed = torch.nn.Parameter(
            torch.zeros(size=(in_channels, hidden_channels)),
            requires_grad=True
        )
        self.w_classify = torch.nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_channels, out_channels)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_embed)
        torch.nn.init.xavier_uniform_(self.w_classify)
        
    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )
        
    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )
        
    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)
        
    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)
        
    def _edge_index_to_adj(self, edge_index, num_nodes):
        """Convert edge_index to sparse adjacency matrix"""
        device = edge_index.device
        row, col = edge_index
        values = torch.ones(edge_index.size(1), device=device)
        adj = torch.sparse_coo_tensor(
            indices=edge_index,
            values=values,
            size=(num_nodes, num_nodes)
        )
        return adj
        
    def forward(self, x, edge_index):
        """
        Forward pass with interface matching other models
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings with log_softmax applied [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        
        # Convert edge_index to adjacency matrix
        adj = self._edge_index_to_adj(edge_index, num_nodes)
        
        # Make sure everything is on the same device
        device = x.device
        adj = adj.to(device)
        
        # Initialize if needed
        if not self.initialized:
            self._prepare_prop(adj)
            
        # H2GCN propagation
        rs = [self.act(torch.mm(x, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
            
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        
        # Use log_softmax instead of softmax for consistency with other models
        return F.log_softmax(torch.mm(r_final, self.w_classify), dim=1)
    

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(MLP, self).__init__()
        
        self.lins = torch.nn.ModuleList()
        
        # Input layer
        self.lins.append(Linear(in_channels, hidden_channels))
        
        # Hidden layers (if any)
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        
        # Output layer
        self.lins.append(Linear(hidden_channels, out_channels))
        
        self.dropout = dropout
    
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
    
    def forward(self, x, edge_index):
        # First layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lins[0](x))
        
        # Hidden layers
        for i in range(1, len(self.lins) - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.lins[i](x))
        
        # Output layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        
        return F.log_softmax(x, dim=1)
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers (if any)
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, edge_index):
        # Process all layers except the last one
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, alpha=0.5, theta=1.0, shared_weights=True,
                 cached=False, add_self_loops=True):
        super(GCN2, self).__init__()
        
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            conv = GCN2Conv(
                channels=hidden_channels,
                alpha=alpha,
                theta=theta,
                layer=layer+1,  # Layer index starts from 1
                shared_weights=shared_weights,
                cached=cached,
                add_self_loops=add_self_loops
            )
            self.convs.append(conv)
            
        self.dropout = dropout
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
            
    def forward(self, x, edge_index):
        """
        Forward pass with interface matching other models
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings with log_softmax applied [num_nodes, out_channels]
        """
        # Apply first linear layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store the initial features for GCN2
        x_0 = x
        
        # Apply GCN2 convolution layers
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, x_0, edge_index)
            x = F.relu(x)
            
        # Apply final linear layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)
    

class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''
    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma
        self.temp = torch.nn.Parameter(torch.tensor(TEMP))
        
    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K
        
    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None
        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class BernNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, dprate=0.5, K=10):
        super(BernNet, self).__init__()
        
        self.num_layers = num_layers
        self.lins = torch.nn.ModuleList()
        
        # Input layer
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        
        # Hidden layers (if any)
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            
        # Output layer
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        
        self.m = torch.nn.BatchNorm1d(out_channels)
        self.prop1 = Bern_prop(K)
        
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        # For first n-1 layers: apply dropout, linear, ReLU
        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[i](x)
            x = F.relu(x)
        
        # Final layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        #x = self.m(x)  # Batch norm is commented out as in original
        
        # Propagation step
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            
        return F.log_softmax(x, dim=1)

class GPRGNN(torch.nn.Module):
    """GPRGNN with interface matching MixHop for fair comparison"""
    def __init__(self, in_channels, hidden_channels, out_channels, Init='PPR', 
                 dropout=0.5, K=10, alpha=0.1, Gamma=None, ppnp='GPR_prop', **kwargs):
        super(GPRGNN, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        
        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)
            
        self.Init = Init
        self.dropout = dropout
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()
        
    def forward(self, x, edge_index):
        """
        Forward pass with interface matching MixHop
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings with log_softmax applied
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        # Apply GPR propagation
        x = self.prop1(x, edge_index)
        
        # Apply log_softmax to match MixHop output format
        return F.log_softmax(x, dim=1)

class MixHop(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, hops=2):
        super(MixHop, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        
        # Input layer
        self.convs.append(MixHopConv(in_channels, hidden_channels, powers=[0, 1, 2]))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels * 3))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(MixHopConv(hidden_channels * 3, hidden_channels, powers=[0, 1, 2]))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels * 3))
        
        # Output layer
        self.convs.append(MixHopConv(hidden_channels * 3, out_channels, powers=[0, 1, 2]))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 dropout=0.5, heads=2, add_self_loops=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        
        # First layer: We use heads=3 to match the 3x dimension increase in MixHop
        # MixHop uses powers=[0,1,2] creating a 3x wider output, so we use 3 heads
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,  # Each head produces this many features
            heads=heads,                   # Number of attention heads (3 to match MixHop's x3)
            concat=True,                   # Concatenate head outputs (True matches MixHop behavior)
            dropout=dropout,
            add_self_loops=add_self_loops
        )
        
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels * heads)
        
        # Hidden layers (if num_layers > 2)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    in_channels=hidden_channels * heads,
                    out_channels=hidden_channels,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=add_self_loops
                )
            )
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        
        # Output layer
        self.conv_out = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,                     # Single head for final output
            concat=False,                # Don't concatenate heads for output layer
            dropout=dropout,
            add_self_loops=add_self_loops
        )

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.conv_out(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class SGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, cached=True):
        super(SGC, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(SGConv(in_channels, hidden_channels, K=2, cached=cached))
        
        # Hidden layers (if any)
        for _ in range(num_layers - 2):
            self.convs.append(SGConv(hidden_channels, hidden_channels, K=2, cached=cached))
        
        # Output layer
        self.convs.append(SGConv(hidden_channels, out_channels, K=2, cached=cached))
        
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, edge_index):
        """
        Forward pass with layered SGC operations
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings with log_softmax applied [num_nodes, out_channels]
        """
        # Apply dropout to input features
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Process all layers except the last one
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            # No nonlinearity (ReLU) between layers to maintain the SGC philosophy
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer
        x = self.convs[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)

class APPNP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 dropout=0.5, K=10, alpha=0.1):
        super(APPNP, self).__init__()
        
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        
        # Import APPNP propagation from PyTorch Geometric
        from torch_geometric.nn import APPNP as APPNPProp
        self.prop1 = APPNPProp(K=K, alpha=alpha)
        
        self.dropout = dropout
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        
    def forward(self, x, edge_index):
        """
        Forward pass with interface matching other models
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings with log_softmax applied [num_nodes, out_channels]
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class GCNJK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, jk_type='max'):
        super(GCNJK, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
        self.convs.append(
            GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            
        self.dropout = dropout
        self.activation = F.relu
        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels, num_layers=1)
        
        if jk_type == 'cat':
            self.final_project = torch.nn.Linear(hidden_channels * num_layers, out_channels)
        else:  # max or lstm
            self.final_project = torch.nn.Linear(hidden_channels, out_channels)
            
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()
        
    def forward(self, x, edge_index):
        """
        Forward pass with interface matching other models
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings with log_softmax applied [num_nodes, out_channels]
        """
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = self.convs[-1](x, edge_index)
        xs.append(x)
        
        x = self.jump(xs)
        x = self.final_project(x)
        
        return F.log_softmax(x, dim=1)


class GATJK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, jk_type='max'):
        super(GATJK, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True))
            
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels*heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*heads))
            
        self.convs.append(
            GATConv(hidden_channels*heads, hidden_channels, heads=heads))
            
        self.dropout = dropout
        self.activation = F.elu  # note: uses elu instead of relu
        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels*heads, num_layers=1)
        
        if jk_type == 'cat':
            self.final_project = torch.nn.Linear(hidden_channels*heads*num_layers, out_channels)
        else:  # max or lstm
            self.final_project = torch.nn.Linear(hidden_channels*heads, out_channels)
            
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()
        
    def forward(self, x, edge_index):
        """
        Forward pass with interface matching other models
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings with log_softmax applied [num_nodes, out_channels]
        """
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = self.convs[-1](x, edge_index)
        xs.append(x)
        
        x = self.jump(xs)
        x = self.final_project(x)
        
        return F.log_softmax(x, dim=1)

class M2M2_layer(torch.nn.Module):
    def __init__(self, in_feat, out_feat, c, dropout, temperature=1):
        super(M2M2_layer, self).__init__()
        self.lin = torch.nn.Linear(in_feat, out_feat, bias=False)
        self.att = torch.nn.Linear(out_feat, c, bias=False)
        self.para_lin = self.lin.parameters()
        self.para_att = self.att.parameters()
        self.temperature = temperature
        self.c = c
        self.dropout = dropout
        self.reg = None
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.att.reset_parameters()
        
    def forward(self, x, edge_index):
        x = self.lin(x)
        row, col = edge_index
        bin_rela = F.relu(0.5*x[row] + x[col])
        bin_rela = self.att(bin_rela)
        bin_rela = F.softmax(bin_rela/self.temperature, dim=1)
        self.reg = np.sqrt(self.c)/bin_rela.size(0)*torch.linalg.vector_norm(bin_rela.sum(dim=0), 2) - 1
        
        x_j = torch.cat([x[col] * bin_rela[:, i].view(-1, 1) for i in range(self.c)], dim=1)
        out = scatter(x_j, row, dim=0, dim_size=x.size(0))
        
        return out

class M2MGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 dropout=0.5, c=4, beta=0.1, dropout2=0.0, temperature=1.0, 
                 remove_self_loops=True):
        super(M2MGNN, self).__init__()
        
        # Input linear transformation
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels*c)
        
        # Normalization layers
        self.norms = torch.nn.ModuleList()
        self.norms.append(torch.nn.LayerNorm(hidden_channels*c))
        
        # M2M2 layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(M2M2_layer(hidden_channels*c, hidden_channels, c, dropout, temperature))
            self.norms.append(torch.nn.LayerNorm(hidden_channels*c))
            
        # Output projection
        self.lin2 = torch.nn.Linear(hidden_channels*c, out_channels)
        
        # Parameter groups for optimization
        self.params1 = list(self.lin2.parameters()) + list(self.lin1.parameters())
        self.params2 = list(self.convs.parameters()) + list(self.norms.parameters())
        
        # Model parameters
        self.dropout = dropout
        self.dropout2 = dropout2
        self.num_layers = num_layers
        self.beta = beta
        self.remove_self_loop = remove_self_loops
        
        # Regularization term
        self.reg = None
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
            
    def forward(self, x, edge_index):
        """
        Forward pass with interface matching other models
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings with log_softmax applied [num_nodes, out_channels]
        """
        # Remove self-loops if specified
        if self.remove_self_loop:
            edge_index, _ = remove_self_loops(edge_index)
            
        # Initialize regularization term
        self.reg = 0
        
        # Apply initial dropout to input features
        if self.dropout2 != 0:
            x = F.dropout(x, p=self.dropout2, training=self.training)
            
        # First linear transformation
        x = F.relu(self.lin1(x))
        x = self.norms[0](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store initial features for residual connections
        ego = x
        
        # Apply M2M2 layers
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.norms[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Apply residual connection with beta weighting
            x = (1-self.beta) * x + self.beta * ego
            
            # Accumulate regularization term
            self.reg = self.reg + self.convs[i].reg
            
        # Final projection
        x = self.lin2(x)
        
        # Normalize regularization term
        self.reg = self.reg / self.num_layers
        
        return F.log_softmax(x, dim=1)
    
