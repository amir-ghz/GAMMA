"""
GAMMA: Gated Multi-hop Message Passing for Homophily-Agnostic Node Representation

This module implements the GAMMA layer as described in the paper:
"GAMMA: Gated Multi-hop Message Passing for Homophily-Agnostic Node Representation in GNNs"

GAMMA addresses heterophilic graphs where connected nodes often belong to different classes.
It uses an iterative gating mechanism to adaptively leverage multi-hop neighborhood information
based on node-specific structural patterns, without the memory overhead of feature concatenation.

Key Features:
    - Adaptive multi-hop aggregation through dynamic routing
    - Weight sharing scheme with learnable channel-wise scaling
    - Efficient memory usage (maintains fixed dimensionality)
    - Captures both global (per-hop) and local (per-node) heterophily patterns

Reference:
    Paper Section 4: "GAMMA: Gated Multi-hop Message Passing"
    Algorithm 1 in Appendix A.2
"""

from typing import List, Optional, Dict, Any
import torch
from torch import Tensor, nn
from torch.nn import Parameter, Linear, ModuleList
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor


class GAMMA(MessagePassing):
    """
    GAMMA (Gated Multi-hop Message Passing) Layer
    
    This layer implements the adaptive multi-hop aggregation mechanism described in the paper.
    Instead of concatenating features from different hop distances, GAMMA uses an iterative
    routing mechanism to dynamically weight hop-specific embeddings based on their agreement
    with the node's evolving representation.
    
    The layer performs three main steps:
        1. Multi-hop propagation: Compute features at different hop distances (A^p * X * W)
        2. Channel-wise scaling: Apply learnable per-hop scaling factors (γ_p)
        3. Dynamic routing: Iteratively compute node-specific gating coefficients (α_{i,p})
    
    Mathematical Formulation:
        H^(p) = A^p (X W) ⊙ γ_p                    # Eq. 8-9 in paper
        α_i^(t) = softmax(b_i^(t))                 # Gating coefficients
        s_i^(t) = Σ_p α_{i,p}^(t) Ĥ_i^(p)        # Weighted aggregation
        v_i^(t) = squash(s_i^(t))                  # Squash normalization
        b_{i,p}^(t+1) = b_{i,p}^(t) + Ĥ_i^(p)·v_i^(t)  # Routing update (agreement)
    
    Attributes:
        in_channels (int): Number of input features per node
        out_channels (int): Number of output features per node
        powers (List[int]): List of hop distances to consider (e.g., [0, 1, 2])
        num_iterations (int): Number of routing iterations (R in paper, typically 2-3)
        use_routing (bool): Enable dynamic routing mechanism
        use_squash (bool): Apply squash function for normalization
        use_shared_weights (bool): Use single weight matrix W for all hops (recommended)
        use_scale_params (bool): Enable learnable per-hop scaling factors γ_p
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        powers: Optional[List[int]] = None,
        num_iterations: int = 1,
        add_self_loops: bool = True,
        bias: bool = True,
        use_routing: bool = True,
        use_squash: bool = True,
        use_shared_weights: bool = True, 
        use_scale_params: bool = True,  
        **kwargs,
    ):
        """
        Initialize the GAMMA layer.
        
        Args:
            in_channels (int): Number of input node features (d_in in paper)
            out_channels (int): Number of output node features (d_out in paper)
            powers (Optional[List[int]]): List of hop distances to aggregate from.
                Default is [0, 1, 2] representing 0-hop (self), 1-hop, and 2-hop neighbors.
                Corresponds to K in the paper (maximum hop distance).
            num_iterations (int): Number of routing iterations (R in Algorithm 1).
                Paper recommends 2-3 iterations. Default: 1
            add_self_loops (bool): Whether to add self-loops to the adjacency matrix.
                Default: True
            bias (bool): Whether to add a learnable bias vector to the output.
                Default: True
            use_routing (bool): Enable the dynamic routing mechanism. When False, uses
                uniform weighting across hops. Default: True (recommended)
            use_squash (bool): Apply the squash function for normalization during routing.
                Squash ensures output vectors have bounded magnitude. Default: True
            use_shared_weights (bool): Use a single shared weight matrix W for all hops
                (Section 5: Weight Sharing). When False, each hop gets its own W^(p).
                Default: True (recommended for efficiency and to ensure common feature space)
            use_scale_params (bool): Enable learnable channel-wise scaling factors γ_p
                for each hop (Eq. 9). Allows hop-specific feature importance.
                Default: True (recommended)
            **kwargs: Additional arguments for MessagePassing base class
        
        References:
            - Section 4: Core GAMMA mechanism
            - Section 5: Weight sharing and scaling parameters
            - Algorithm 1 (Appendix A.2): Complete forward pass pseudocode
        """
        super().__init__(aggr='add', **kwargs)
        
        # Set default hop configuration: 0-hop (self), 1-hop, and 2-hop neighbors
        if powers is None:
            powers = [0, 1, 2]

        # Store configuration parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.powers = powers  # K in paper: maximum hop distance
        self.num_iterations = num_iterations  # R in Algorithm 1
        self.add_self_loops = add_self_loops
        self.use_routing = use_routing
        self.use_squash = use_squash
        self.use_shared_weights = use_shared_weights  
        self.use_scale_params = use_scale_params     

        # ====================================================================
        # Weight Sharing Scheme (Section 5, Eq. 8)
        # ====================================================================
        # Purpose: Ensure all hop embeddings reside in a common feature space
        # for meaningful dot-product comparisons in routing mechanism
        
        if self.use_shared_weights:
            # RECOMMENDED: Single shared transformation W ∈ R^{d_in × d_out}
            # All hops use: H^(p) = A^p (X W)
            # Benefits:
            #   - Parameter efficiency: O(d_in × d_out) vs O(K × d_in × d_out)
            #   - Common coordinate system for all hop embeddings
            #   - Reduces overfitting risk
            self.shared_linear = Linear(in_channels, out_channels, bias=True)
            self.separate_linears = None
        else:
            # Alternative: Separate transformation W^(p) for each hop
            # Each hop uses: H^(p) = A^p X W^(p)
            # Use case: When different hops require drastically different transformations
            self.shared_linear = None
            self.separate_linears = ModuleList([
                Linear(in_channels, out_channels, bias=True) 
                for _ in range(len(powers))
            ])

        # ====================================================================
        # Channel-wise Scaling Factors (Section 5, Eq. 9)
        # ====================================================================
        # Purpose: Re-introduce hop-specific adaptability while preserving shared space
        # γ_p ∈ R^{d_out}: learnable scaling vector for hop p
        # H^(p)_scaled = H^(p) ⊙ γ_p (element-wise multiplication)
        
        if self.use_scale_params:
            # Initialize scaling factors with small variance around 1.0
            # Shape: [num_powers, out_channels]
            self.scale_params = Parameter(torch.randn(len(powers), out_channels))
        else:
            self.register_parameter('scale_params', None)

        # ====================================================================
        # Bias Term (Optional)
        # ====================================================================
        # Added to final output: H_i = v_i^(R) + b
        
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        # ====================================================================
        # Layer Normalization
        # ====================================================================
        # Applied to each hop embedding before routing (Lines 6-7 in Algorithm 1)
        # Ensures stable training and meaningful magnitude comparisons
        
        self.layer_norm = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize/reset all learnable parameters.
        
        Initialization strategy:
            - Linear layers: PyTorch default (Kaiming uniform)
            - Scale parameters γ_p: Normal(mean=1.0, std=0.1) to start near identity
            - Bias: Zero initialization
        
        This method is called automatically during __init__ and can be called
        manually to reinitialize the layer.
        """
        # Initialize weight matrices
        if self.use_shared_weights:
            self.shared_linear.reset_parameters()
        else:
            for linear in self.separate_linears:
                linear.reset_parameters()
        
        # Initialize scaling factors centered around 1.0
        # This allows the model to start with approximately equal hop importance
        if self.use_scale_params:
            torch.nn.init.normal_(self.scale_params, mean=1.0, std=0.1)
        
        # Initialize bias to zero
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """
        Forward pass implementing the GAMMA algorithm (Algorithm 1 in Appendix A.2).
        
        This method performs three main stages:
            1. Multi-hop Propagation: Compute H^(p) = A^p (X W) for each hop p
            2. Channel-wise Scaling: Apply H^(p) ⊙ γ_p
            3. Dynamic Routing: Iteratively compute gating coefficients α_{i,p}
        
        Args:
            x (Tensor): Input node features of shape [N, in_channels]
                where N is the number of nodes
            edge_index (Adj): Graph connectivity in COO format [2, num_edges]
            edge_weight (OptTensor): Optional edge weights of shape [num_edges]
                If None, all edges have weight 1.0
        
        Returns:
            Tensor: Output node representations of shape [N, out_channels]
        
        Algorithm Flow:
            Lines 1-4: Multi-hop propagation (shared transformation + iterative propagation)
            Lines 6-7: Hop-specific scaling and normalization
            Lines 10-16: Iterative routing mechanism (if enabled)
            Line 17: Add bias term
        
        References:
            - Algorithm 1 in Appendix A.2: Complete pseudocode
            - Section 4: Detailed explanation of routing mechanism
        """
        # Initialize edge weights if not provided (all edges weighted equally)
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=x.dtype, device=x.device
            )

        # ====================================================================
        # Stage 1: Multi-hop Propagation (Algorithm 1, Lines 1-4)
        # ====================================================================
        # For each hop distance p ∈ {0, 1, ..., K}:
        #   1. Apply shared linear transformation: H = X W
        #   2. Propagate p times: H^(p) = A^p H
        # This produces K+1 embeddings, one for each hop distance
        
        outputs = []
        for i, p in enumerate(self.powers):
            # Line 1: Apply shared linear transformation H ← X W
            if self.use_shared_weights:
                xp = self.shared_linear(x)  # [N, out_channels]
            else:
                # Alternative: Use hop-specific transformation
                xp = self.separate_linears[i](x)  # [N, out_channels]
            
            # Lines 2-4: Apply p-hop propagation iteratively
            # H^(0) = H (self-features, no propagation)
            # H^(1) = A H (1-hop neighbors)
            # H^(2) = A (A H) = A^2 H (2-hop neighbors)
            # etc.
            for _ in range(p):
                xp = self.propagate(edge_index, x=xp, edge_weight=edge_weight, size=None)
            
            # Line 6: Apply hop-specific scaling H^(p) ← H^(p) ⊙ γ_p
            # This allows different feature channels to have different importance
            # at different hop distances (Eq. 9 in paper)
            if self.use_scale_params:
                xp = xp * self.scale_params[i]
            
            outputs.append(xp)  # Each xp is [N, out_channels]

        # ====================================================================
        # Stage 2: Normalization (Algorithm 1, Lines 6-7)
        # ====================================================================
        # Apply L2 normalization (via LayerNorm) to each hop embedding
        # Purpose: Mitigate scale discrepancies and enable meaningful dot products
        
        outputs = [self.layer_norm(xp) for xp in outputs]
        stacked_outputs = torch.stack(outputs, dim=1)  # [N, num_powers, out_channels]

        # ====================================================================
        # Stage 3: Dynamic Routing Mechanism (Algorithm 1, Lines 10-16)
        # ====================================================================
        # This is the core innovation of GAMMA: instead of concatenating multi-hop
        # features, we iteratively determine node-specific gating coefficients
        # α_{i,p} that weight each hop's contribution based on its "agreement"
        # with the node's evolving representation.
        
        if self.use_routing:
            # Line 10: Initialize routing logits b_i ← 0 for all nodes i
            # Shape: [N, num_powers] where N = number of nodes
            # These logits will be iteratively updated based on agreement scores
            b = torch.zeros(x.size(0), len(self.powers), device=x.device)

            # Lines 11-16: Iterative routing loop (R iterations)
            for iteration in range(self.num_iterations):
                # Line 11: Compute gating coefficients via softmax
                # α_i = softmax(b_i) ensures Σ_p α_{i,p} = 1 for each node i
                c = torch.softmax(b, dim=1)  # [N, num_powers]
                c_expanded = c.unsqueeze(-1)  # [N, num_powers, 1]

                # Line 12: Compute weighted aggregation of hop embeddings
                # s_i^(t) = Σ_{p=0}^K α_{i,p}^(t) Ĥ_i^(p)
                # This is the intermediate representation for this iteration
                s = torch.sum(c_expanded * stacked_outputs, dim=1)  # [N, out_channels]

                # Line 13: Apply squash function for normalization
                # v_i^(t) = squash(s_i^(t))
                # Squash function: v = (||s||^2 / (1 + ||s||^2)) * (s / ||s||)
                # Purpose: Bound vector magnitudes while preserving direction
                if self.use_squash:
                    v = self.squash(s)  # [N, out_channels]
                else:
                    v = s

                # Lines 14-15: Update routing logits based on agreement
                # agreement_{i,p} = Ĥ_i^(p) · v_i^(t) (dot product)
                # b_{i,p}^(t+1) = b_{i,p}^(t) + agreement_{i,p}
                # 
                # Intuition: If hop p's embedding aligns well with the current
                # node representation v_i, the dot product is large (positive),
                # increasing b_{i,p} and thus α_{i,p} in the next iteration.
                # This creates a feedback loop that amplifies informative hops.
                agreement = torch.sum(stacked_outputs * v.unsqueeze(1), dim=-1)  # [N, num_powers]
                b = b + agreement

            # Line 16: Use final representation from last iteration
            out = v
            
        else:
            # ================================================================
            # Baseline: Uniform Weighting (No Routing)
            # ================================================================
            # For ablation studies: average all hop embeddings equally
            # α_{i,p} = 1/(K+1) for all nodes i and hops p
            
            uniform_weights = torch.ones(x.size(0), len(self.powers), device=x.device) / len(self.powers)
            uniform_weights = uniform_weights.unsqueeze(-1)  # [N, num_powers, 1]
            s = torch.sum(uniform_weights * stacked_outputs, dim=1)  # [N, out_channels]
            
            # Apply squash if enabled (usually disabled for non-routing)
            if self.use_squash:
                out = self.squash(s)
            else:
                out = s

        # ====================================================================
        # Final Step: Add Bias (Algorithm 1, Line 17)
        # ====================================================================
        # H_i = v_i^(R) + b
        
        if self.bias is not None:
            out = out + self.bias

        return out

    def squash(self, s: Tensor) -> Tensor:
        """
        Squash activation function (inspired by Capsule Networks).
        
        Mathematical formula:
            v = (||s||^2 / (1 + ||s||^2)) * (s / ||s||)
        
        Properties:
            - Preserves direction of input vector s
            - Bounds output magnitude: 0 ≤ ||v|| < 1
            - For small ||s||: v ≈ s (approximately linear)
            - For large ||s||: ||v|| → 1 (saturation)
        
        Purpose in GAMMA:
            - Maintains meaningful vector magnitudes during routing
            - Prevents numerical instabilities from unbounded growth
            - Preserves directional information for dot-product comparisons
        
        Args:
            s (Tensor): Input tensor of shape [N, out_channels]
        
        Returns:
            Tensor: Squashed tensor of shape [N, out_channels]
        
        Reference:
            - Sabour et al., "Dynamic Routing Between Capsules", NeurIPS 2017
            - Paper Section 4: GAMMA routing mechanism
        """
        # Compute squared L2 norm for each node: ||s_i||^2
        s_norm_sq = torch.sum(s ** 2, dim=-1, keepdim=True)  # [N, 1]
        
        # Compute scaling factor: ||s||^2 / (1 + ||s||^2)
        # This approaches 1 as ||s|| → ∞, providing saturation
        scale = s_norm_sq / (1 + s_norm_sq)
        
        # Compute L2 norm (add epsilon for numerical stability)
        s_norm = torch.sqrt(s_norm_sq + 1e-8)
        
        # Apply squash: scale * (s / ||s||)
        v = scale * s / s_norm  # [N, out_channels]
        return v

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        """
        Construct messages from neighboring nodes (PyG MessagePassing interface).
        
        This method is called internally by self.propagate() during multi-hop
        propagation to aggregate features from neighboring nodes.
        
        Args:
            x_j (Tensor): Features of neighboring nodes (source nodes)
                Shape: [num_edges, out_channels]
            edge_weight (Tensor): Edge weights for the graph
                Shape: [num_edges]
        
        Returns:
            Tensor: Weighted messages of shape [num_edges, out_channels]
        
        Note:
            This implements standard weighted message passing:
            message_ij = w_ij * x_j
            where w_ij is the edge weight from node j to node i.
        """
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def get_routing_weights(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None):
        """
        Extract the learned routing coefficients α_{i,p} for analysis and visualization.
        
        This method computes the final gating coefficients that determine how much
        weight each node assigns to different hop distances. These coefficients are
        node-specific and adaptive, capturing local heterophily patterns.
        
        Use cases:
            - Analyze which hop distances are most informative for different nodes
            - Visualize how GAMMA adapts to heterophilic graph structure
            - Debug and interpret the model's decision-making process
            - Study local vs. global heterophily patterns (see Paper Section 2)
        
        Args:
            x (Tensor): Input node features [N, in_channels]
            edge_index (Adj): Graph connectivity [2, num_edges]
            edge_weight (OptTensor): Optional edge weights [num_edges]
        
        Returns:
            Tensor: Routing coefficients α of shape [N, num_powers]
                where α_{i,p} ∈ [0, 1] and Σ_p α_{i,p} = 1 for each node i.
                Higher values indicate that hop p is more informative for node i.
        
        Example:
            >>> layer = GAMMA(in_channels=16, out_channels=32, powers=[0,1,2])
            >>> out = layer(x, edge_index)
            >>> coeffs = layer.get_routing_weights(x, edge_index)
            >>> # coeffs[i, 0] = weight for 0-hop (self)
            >>> # coeffs[i, 1] = weight for 1-hop neighbors
            >>> # coeffs[i, 2] = weight for 2-hop neighbors
        
        Note:
            - This method runs in eval mode (no gradients computed)
            - If routing is disabled, returns uniform weights 1/(K+1)
        """
        if not self.use_routing:
            # Return uniform weights when routing is disabled
            uniform = torch.ones(x.size(0), len(self.powers), device=x.device) / len(self.powers)
            return uniform
        
        # Run routing mechanism without gradient computation
        with torch.no_grad():
            # Initialize edge weights if not provided
            if edge_weight is None:
                edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=x.device)

            # Step 1: Compute multi-hop embeddings (same as forward pass)
            outputs = []
            for i, p in enumerate(self.powers):
                # Apply linear transformation
                if self.use_shared_weights:
                    xp = self.shared_linear(x)
                else:
                    xp = self.separate_linears[i](x)
                
                # Apply p-hop propagation
                for _ in range(p):
                    xp = self.propagate(edge_index, x=xp, edge_weight=edge_weight, size=None)
                
                # Apply channel-wise scaling
                if self.use_scale_params:
                    xp = xp * self.scale_params[i]
                    
                outputs.append(xp)

            # Step 2: Normalize hop embeddings
            outputs = [self.layer_norm(xp) for xp in outputs]
            stacked_outputs = torch.stack(outputs, dim=1)

            # Step 3: Run routing iterations to compute final logits
            b = torch.zeros(x.size(0), len(self.powers), device=x.device)
            
            for _ in range(self.num_iterations):
                c = torch.softmax(b, dim=1)
                c_expanded = c.unsqueeze(-1)
                s = torch.sum(c_expanded * stacked_outputs, dim=1)
                v = self.squash(s) if self.use_squash else s
                agreement = torch.sum(stacked_outputs * v.unsqueeze(1), dim=-1)
                b = b + agreement

            # Step 4: Convert final logits to coefficients via softmax
            final_coefficients = torch.softmax(b, dim=1)
            return final_coefficients

    def get_scale_params(self):
        """
        Extract the learned channel-wise scaling parameters γ_p for analysis.
        
        These parameters capture global (graph-level) hop-specific feature importance.
        Each γ_p ∈ R^{d_out} scales the features from hop p across all nodes,
        complementing the node-specific routing coefficients α_{i,p}.
        
        Returns:
            Tensor or None: Scaling parameters of shape [num_powers, out_channels]
                Returns None if use_scale_params=False
        
        Example:
            >>> layer = GAMMA(in_channels=16, out_channels=32, powers=[0,1,2])
            >>> scales = layer.get_scale_params()
            >>> # scales[0] = global scaling for 0-hop features
            >>> # scales[1] = global scaling for 1-hop features
            >>> # scales[2] = global scaling for 2-hop features
        
        Reference:
            - Paper Section 5, Eq. 9: Channel-wise scaling mechanism
        """
        if self.use_scale_params:
            return self.scale_params.detach().clone()
        else:
            return None

    def get_num_parameters(self):
        """
        Calculate the total number of learnable parameters in the layer.
        
        This method is useful for:
            - Comparing parameter efficiency with baseline methods
            - Memory footprint estimation
            - Model complexity analysis (see Paper Section 5, Figure 5)
        
        Parameter breakdown:
            - Shared weight matrix W: d_in × d_out (+ bias)
            - OR separate weights W^(p): K × d_in × d_out (if not shared)
            - Scaling factors γ_p: K × d_out (if enabled)
            - Output bias b: d_out (if enabled)
            - Layer normalization: 2 × d_out (scale + shift)
        
        Returns:
            int: Total number of learnable parameters
        
        Example:
            >>> layer = GAMMA(in_channels=16, out_channels=32, powers=[0,1,2],
            ...              use_shared_weights=True, use_scale_params=True)
            >>> num_params = layer.get_num_parameters()
            >>> # With shared weights: (16*32 + 32) + (3*32) + 32 + (2*32) = 736
            
        Reference:
            - Paper Section 5: Weight sharing reduces parameters significantly
            - Figure 5: Memory consumption comparison
        """
        total_params = 0
        
        # Count parameters in linear transformation(s)
        if self.use_shared_weights:
            # Single shared transformation W ∈ R^{d_in × d_out}
            total_params += sum(p.numel() for p in self.shared_linear.parameters())
        else:
            # Separate transformations W^(p) for each hop
            total_params += sum(sum(p.numel() for p in linear.parameters()) 
                              for linear in self.separate_linears)
        
        # Count scaling parameters γ_p
        if self.use_scale_params:
            total_params += self.scale_params.numel()
        
        # Count output bias
        if self.bias is not None:
            total_params += self.bias.numel()
        
        # Count layer normalization parameters (scale and shift)
        total_params += sum(p.numel() for p in self.layer_norm.parameters())
        
        return total_params

    def __repr__(self) -> str:
        """
        Return a string representation of the layer for debugging and logging.
        
        Returns:
            str: Human-readable description of the layer configuration
        
        Example:
            >>> layer = GAMMA(in_channels=16, out_channels=32, powers=[0,1,2],
            ...              num_iterations=3, use_routing=True)
            >>> print(layer)
            GAMMA(16, 32, powers=[0, 1, 2], num_iterations=3, use_routing=True,
                  use_squash=True, use_shared_weights=True, use_scale_params=True)
        """
        return (
            f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, '
            f'powers={self.powers}, num_iterations={self.num_iterations}, '
            f'use_routing={self.use_routing}, use_squash={self.use_squash}, '
            f'use_shared_weights={self.use_shared_weights}, use_scale_params={self.use_scale_params})'
        )
