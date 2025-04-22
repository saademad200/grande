import os
import torch
from models.grande import GRANDE
from utils.data import load_dataset, prepare_data
from utils.training import GRANDETrainer, XGBoostTrainer, CatBoostTrainer
from visualization.plotting import plot_training_curves, create_results_plots
import pandas as pd
from sklearn.metrics import classification_report

def run_benchmark(n_samples=1000, n_features=10, n_classes=3, n_trees=5, depth=3):
    """Run the benchmark comparison between GRANDE, XGBoost, and CatBoost.
    
    Args:
        n_samples (int): Number of samples in the dataset
        n_features (int): Number of features
        n_classes (int): Number of classes
        n_trees (int): Number of trees for GRANDE
        depth (int): Depth of trees
    """
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Load and prepare data
    print("\nGenerating dataset...")
    X, y, dataset_info = load_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features // 2
    )
    data = prepare_data(X, y)
    
    print("\nDataset Information:")
    for key, value in dataset_info.items():
        if not isinstance(value, list):
            print(f"{key}: {value}")
    
    results = {}
    
    # Train and evaluate GRANDE
    print("\nTraining GRANDE...")
    grande_model = GRANDE(n_trees=n_trees, depth=depth, 
                        input_dim=n_features, num_classes=n_classes)
    grande_trainer = GRANDETrainer(grande_model)
    history = grande_trainer.train(data['torch']['train'], data['torch']['test'])
    
    # Plot GRANDE training curves
    plot_training_curves(
        history,
        'results/grande_training_curves.png'
    )
    
    results['GRANDE'] = grande_trainer.evaluate(data['torch']['test'])
    
    # Train and evaluate XGBoost
    print("\nTraining XGBoost...")
    xgb_trainer = XGBoostTrainer(num_classes=n_classes)
    xgb_trainer.train(data['numpy']['train'], data['numpy']['test'])
    results['XGBoost'] = xgb_trainer.evaluate(data['numpy']['test'])
    
    # Train and evaluate CatBoost
    print("\nTraining CatBoost...")
    catboost_trainer = CatBoostTrainer(num_classes=n_classes)
    catboost_trainer.train(data['numpy']['train'], data['numpy']['test'])
    results['CatBoost'] = catboost_trainer.evaluate(data['numpy']['test'])
    
    # Create comparison plots and save results
    print("\nGenerating comparison plots...")
    df_results = create_results_plots(results)
    
    # Generate detailed classification reports
    print("\nDetailed Classification Reports:")
    for name, trainer in [
        ('GRANDE', grande_trainer),
        ('XGBoost', xgb_trainer),
        ('CatBoost', catboost_trainer)
    ]:
        print(f"\n{name} Classification Report:")
        if name == 'GRANDE':
            with torch.no_grad():
                y_pred = torch.argmax(
                    trainer.model(data['torch']['test']['X']), dim=1
                ).cpu().numpy()
            y_true = data['torch']['test']['y'].cpu().numpy()
        else:
            y_pred = trainer.model.predict(data['numpy']['test']['X'])
            y_true = data['numpy']['test']['y']
        
        report = classification_report(
            y_true, y_pred,
            target_names=dataset_info['class_names']
        )
        print(report)
        
        # Save classification report to file
        with open(f'results/{name.lower()}_classification_report.txt', 'w') as f:
            f.write(report)
    
    # Print final results
    print("\nFinal Results:")
    print(df_results.to_string(index=False))
    
    return results, df_results, dataset_info

if __name__ == "__main__":
    run_benchmark() 