import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Tuple, List
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    """Base class for model training and evaluation."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.training_time = 0
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

class GRANDETrainer(ModelTrainer):
    """Trainer for GRANDE model."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__('GRANDE')
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_data: Dict[str, torch.Tensor], 
              val_data: Dict[str, torch.Tensor], 
              num_epochs: int = 100) -> Tuple[List[float], List[float]]:
        """Train the GRANDE model.
        
        Args:
            train_data (dict): Training data tensors
            val_data (dict): Validation data tensors
            num_epochs (int): Number of training epochs
        
        Returns:
            tuple: Lists of training and validation losses
        """
        X_train, y_train = train_data['X'].to(self.device), train_data['y'].to(self.device)
        X_val, y_val = val_data['X'].to(self.device), val_data['y'].to(self.device)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            self.optimizer.zero_grad()
            y_pred = self.model(X_train)
            loss = self.criterion(y_pred, y_train.long())
            loss.backward()
            self.optimizer.step()
            
            # Calculate training accuracy
            train_acc = (torch.argmax(y_pred, dim=1) == y_train).float().mean()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = self.criterion(val_pred, y_val.long())
                val_acc = (torch.argmax(val_pred, dim=1) == y_val).float().mean()
            
            self.history['train_loss'].append(loss.item())
            self.history['val_loss'].append(val_loss.item())
            self.history['train_acc'].append(train_acc.item())
            self.history['val_acc'].append(val_acc.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Train Loss: {loss.item():.4f}, "
                      f"Val Loss: {val_loss.item():.4f}, "
                      f"Train Acc: {train_acc.item():.4f}, "
                      f"Val Acc: {val_acc.item():.4f}")
        
        self.training_time = time.time() - start_time
        return self.history

    def evaluate(self, test_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate the model on test data."""
        self.model.eval()
        X_test, y_test = test_data['X'].to(self.device), test_data['y'].to(self.device)
        
        with torch.no_grad():
            y_pred = self.model(X_test)
            test_loss = self.criterion(y_pred, y_test.long())
            predictions = torch.argmax(y_pred, dim=1)
            accuracy = (predictions == y_test).float().mean()
            
        return {
            'test_loss': test_loss.item(),
            'accuracy': accuracy.item(),
            'training_time': self.training_time
        }

class XGBoostTrainer(ModelTrainer):
    """Trainer for XGBoost model."""
    
    def __init__(self, num_classes, params=None):
        super().__init__('XGBoost')
        self.num_classes = num_classes
        self.params = params or {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'objective': 'multi:softmax',
            'num_class': num_classes,
            'random_state': 42
        }
        self.model = xgb.XGBClassifier(**self.params)

    def train(self, train_data: Dict[str, np.ndarray], 
              val_data: Dict[str, np.ndarray]) -> None:
        """Train the XGBoost model."""
        start_time = time.time()
        self.model.fit(
            train_data['X'], train_data['y'],
            eval_set=[(val_data['X'], val_data['y'])],
            verbose=False
        )
        self.training_time = time.time() - start_time

    def evaluate(self, test_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate the model on test data."""
        y_pred = self.model.predict(test_data['X'])
        accuracy = accuracy_score(test_data['y'], y_pred)
        
        return {
            'accuracy': accuracy,
            'training_time': self.training_time
        }

class CatBoostTrainer(ModelTrainer):
    """Trainer for CatBoost model."""
    
    def __init__(self, num_classes, params=None):
        super().__init__('CatBoost')
        self.num_classes = num_classes
        self.params = params or {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 3,
            'loss_function': 'MultiClass',
            'random_seed': 42,
            'verbose': False
        }
        self.model = CatBoostClassifier(**self.params)

    def train(self, train_data: Dict[str, np.ndarray], 
              val_data: Dict[str, np.ndarray]) -> None:
        """Train the CatBoost model."""
        start_time = time.time()
        self.model.fit(
            train_data['X'], train_data['y'],
            eval_set=(val_data['X'], val_data['y'])
        )
        self.training_time = time.time() - start_time

    def evaluate(self, test_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate the model on test data."""
        y_pred = self.model.predict(test_data['X'])
        accuracy = accuracy_score(test_data['y'], y_pred)
        
        return {
            'accuracy': accuracy,
            'training_time': self.training_time
        } 