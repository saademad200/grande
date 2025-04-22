import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

def load_dataset(n_samples=1000, n_features=10, n_classes=3, n_informative=5, **kwargs):
    """Load or generate a classification dataset.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features
        n_classes (int): Number of classes
        n_informative (int): Number of informative features
        **kwargs: Additional arguments for make_classification
    
    Returns:
        tuple: (X, y) data arrays and dataset info
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        random_state=42,
        **kwargs
    )
    
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    class_names = [f'Class_{i}' for i in range(n_classes)]
    
    dataset_info = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': n_classes,
        'n_informative': n_informative,
        'feature_names': feature_names,
        'class_names': class_names
    }
    
    return X, y, dataset_info

def prepare_data(X, y, test_size=0.2, random_state=42):
    """Prepare data for training by splitting and scaling.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Target values
        test_size (float): Proportion of test set
        random_state (int): Random seed
    
    Returns:
        tuple: Processed data arrays and tensors
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)
    
    return {
        'numpy': {
            'train': {'X': X_train_scaled, 'y': y_train},
            'test': {'X': X_test_scaled, 'y': y_test}
        },
        'torch': {
            'train': {'X': X_train_tensor, 'y': y_train_tensor},
            'test': {'X': X_test_tensor, 'y': y_test_tensor}
        },
        'scaler': scaler
    } 