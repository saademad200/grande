import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def softsign_split(z):
    """Scaled softsign function as differentiable split function (Equation 6 in paper).
    
    The softsign function is used instead of sigmoid because it:
    1. Maintains responsive gradients for large input values
    2. Has high gradients near the decision boundary
    3. Scales outputs to (0,1) range for probabilistic interpretation
    
    Args:
        z (torch.Tensor): Input tensor (typically feature_value - threshold)
    
    Returns:
        torch.Tensor: Scaled softsign output in range (0,1)
    """
    return 0.5 * (z / (1 + torch.abs(z)) + 1)

class SoftTree(nn.Module):
    """Neural soft decision tree with softsign splits and instance-wise weighting.
    
    Key Features:
    - Uses differentiable softsign split function
    - Supports feature subset selection for regularization
    - Implements instance-wise leaf weighting
    - Maintains probabilistic path routing
    """
    
    def __init__(self, depth, input_dim, num_classes, feature_subset_ratio=0.8):
        """Initialize a soft decision tree.
        
        Args:
            depth (int): Depth of the tree
            input_dim (int): Number of input features
            num_classes (int): Number of output classes
            feature_subset_ratio (float): Ratio of features to select for each split
        """
        super().__init__()
        self.depth = depth
        self.input_dim = input_dim
        self.num_leaves = 2 ** depth  # Number of leaves grows exponentially with depth
        self.num_classes = num_classes
        self.feature_subset_size = max(1, int(input_dim * feature_subset_ratio))

        # Learnable threshold parameters for each internal node
        self.thresholds = nn.Parameter(torch.randn(depth))
        
        # Feature selection mask for regularization (Section 3.4)
        # One-hot encoded mask to select which features to use at each split
        self.feature_mask = nn.Parameter(
            torch.zeros(depth, input_dim),
            requires_grad=False  # Fixed during training
        )
        self._initialize_feature_masks()
        
        # Learnable leaf node outputs (one per class)
        self.leaf_outputs = nn.Parameter(torch.randn(self.num_leaves, num_classes))
        
        # Instance-wise leaf weights for attention mechanism (Section 3.3)
        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves))

    def _initialize_feature_masks(self):
        """Initialize random feature subset for each split node.
        
        This implements the feature subset selection described in Section 3.4
        of the paper for regularization and improved scalability.
        """
        for d in range(self.depth):
            # Randomly select features for each split node
            selected_features = np.random.choice(
                self.input_dim, 
                self.feature_subset_size, 
                replace=False
            )
            self.feature_mask.data[d, selected_features] = 1.0

    def forward(self, x, return_leaf_probs=False):
        """Forward pass through the soft decision tree.
        
        This implements the core tree routing logic with:
        1. Feature subset selection
        2. Soft splits using softsign function
        3. Probabilistic path routing
        4. Instance-wise weighting
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            return_leaf_probs (bool): Whether to return leaf assignment probabilities
        
        Returns:
            torch.Tensor: Class probabilities [batch_size, num_classes]
            torch.Tensor: (optional) Leaf assignment probabilities [batch_size, num_leaves]
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize path probabilities for root node
        path_probs = torch.ones(batch_size, 1, device=device)
        
        # Track sample-to-leaf assignments for instance-wise weighting
        leaf_assignments = torch.zeros(batch_size, self.num_leaves, device=device)
        
        # Traverse the tree, computing split probabilities at each node
        for d in range(self.depth):
            # Apply feature mask to implement feature subset selection
            masked_features = x @ self.feature_mask[d]
            
            # Compute split probability using softsign function
            split_prob = softsign_split(masked_features - self.thresholds[d])
            
            # Update path probabilities by branching left and right
            path_probs = torch.cat([
                path_probs * split_prob.unsqueeze(1),        # Left branch
                path_probs * (1 - split_prob.unsqueeze(1))   # Right branch
            ], dim=1)
        
        # Final path probabilities represent leaf assignments
        leaf_assignments = path_probs
        
        # Compute final predictions using leaf values
        leaf_probs = F.softmax(self.leaf_outputs, dim=1)  # Convert to probabilities
        predictions = path_probs @ leaf_probs  # Weighted sum of leaf predictions
        
        if return_leaf_probs:
            return predictions, leaf_assignments
        return predictions

class GRANDE(nn.Module):
    """Gradient Boosted Neural Decision Trees with Instance-wise Attention.
    
    This implements the complete GRANDE model as described in the paper, featuring:
    1. Ensemble of soft decision trees
    2. Instance-wise attention mechanism
    3. Feature subset selection
    4. Dropout regularization
    """
    
    def __init__(self, n_trees, depth, input_dim, num_classes, 
                 feature_subset_ratio=0.8, dropout_rate=0.1):
        """Initialize GRANDE model.
        
        Args:
            n_trees (int): Number of trees in ensemble
            depth (int): Depth of each tree
            input_dim (int): Number of input features
            num_classes (int): Number of output classes
            feature_subset_ratio (float): Ratio of features to use in each tree
            dropout_rate (float): Probability of dropping trees during training
        """
        super().__init__()
        self.n_trees = n_trees
        self.depth = depth
        self.dropout_rate = dropout_rate
        
        # Create ensemble of soft decision trees
        self.trees = nn.ModuleList([
            SoftTree(depth, input_dim, num_classes, feature_subset_ratio) 
            for _ in range(n_trees)
        ])
        
        # Instance-wise attention network (Section 3.3)
        # Maps input features to tree weights
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_trees)
        )

    def forward(self, x, return_attention=False):
        """Forward pass through GRANDE ensemble.
        
        Implements equation (7) from the paper:
        G(x|W,L,T,I) = σ(w(x|W,L,T,I)) · p(x|L,T,I)
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            return_attention (bool): Whether to return attention weights
        
        Returns:
            torch.Tensor: Class probabilities [batch_size, num_classes]
            torch.Tensor: (optional) Attention weights [batch_size, n_trees]
        """
        batch_size = x.size(0)
        device = x.device
        
        # Get predictions from each tree
        tree_outputs = []
        leaf_assignments = []
        
        for tree in self.trees:
            pred, leaf_probs = tree(x, return_leaf_probs=True)
            tree_outputs.append(pred)
            leaf_assignments.append(leaf_probs)
        
        # Stack predictions and leaf assignments
        tree_outputs = torch.stack(tree_outputs, dim=1)      # [batch_size, n_trees, num_classes]
        leaf_assignments = torch.stack(leaf_assignments, dim=1)  # [batch_size, n_trees, num_leaves]
        
        # Calculate instance-wise attention weights
        attention_logits = self.attention_net(x)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # Apply dropout during training (Section 3.4)
        if self.training and self.dropout_rate > 0:
            mask = torch.bernoulli(
                torch.ones_like(attention_weights) * (1 - self.dropout_rate)
            )
            attention_weights = attention_weights * mask
            # Renormalize weights after dropout
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-6)
        
        # Combine predictions using instance-wise attention (Equation 7)
        weighted_preds = torch.sum(
            tree_outputs * attention_weights.unsqueeze(-1), 
            dim=1
        )
        
        if return_attention:
            return weighted_preds, attention_weights
        return weighted_preds

    def predict(self, x):
        """Return class predictions (argmax of probabilities)."""
        with torch.no_grad():
            probs = self.forward(x)
            return torch.argmax(probs, dim=1)

    def get_complexity_metrics(self):
        """Return model complexity metrics for analysis."""
        return {
            'num_trees': self.n_trees,
            'tree_depth': self.depth,
            'num_leaves_per_tree': self.trees[0].num_leaves,
            'total_leaves': self.n_trees * self.trees[0].num_leaves,
            'feature_subset_size': self.trees[0].feature_subset_size,
            'dropout_rate': self.dropout_rate
        } 