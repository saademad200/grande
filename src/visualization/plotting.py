import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict
import os

def set_style():
    """Set the style for all plots."""
    plt.style.use('seaborn')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 12

def plot_training_curves(history: Dict[str, List[float]], save_path: str = None):
    """Plot training and validation loss curves.
    
    Args:
        history (dict): Dictionary containing loss histories
        save_path (str, optional): Path to save the plot
    """
    set_style()
    plt.figure()
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_model_comparison(results: pd.DataFrame, metric: str, save_path: str = None):
    """Plot model comparison bar chart.
    
    Args:
        results (pd.DataFrame): DataFrame containing model results
        metric (str): Metric to plot ('test_loss' or 'training_time')
        save_path (str, optional): Path to save the plot
    """
    set_style()
    plt.figure()
    
    ax = sns.barplot(x='Model', y=metric, data=results)
    plt.title(f'Model Comparison - {metric}')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(results[metric]):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_results_plots(results: Dict[str, Dict], output_dir: str = 'results'):
    """Create and save all result plots.
    
    Args:
        results (dict): Dictionary containing results for all models
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare results DataFrame
    df_results = pd.DataFrame({
        'Model': list(results.keys()),
        'test_loss': [r['test_loss'] for r in results.values()],
        'training_time': [r['training_time'] for r in results.values()]
    })
    
    # Plot and save comparisons
    plot_model_comparison(
        df_results, 'test_loss',
        os.path.join(output_dir, 'test_loss_comparison.png')
    )
    plot_model_comparison(
        df_results, 'training_time',
        os.path.join(output_dir, 'training_time_comparison.png')
    )
    
    # Save results to CSV
    df_results.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    return df_results 