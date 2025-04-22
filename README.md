# GRANDE: Gradient Boosted Neural Decision Trees

This repository implements GRANDE (Gradient Boosted Neural Decision Trees) and compares its performance with XGBoost and CatBoost. The implementation features differentiable decision trees with attention mechanisms, making it end-to-end trainable using gradient descent.

## Project Structure

```
.
├── src/
│   ├── models/
│   │   └── grande.py         # GRANDE model implementation
│   ├── utils/
│   │   ├── data.py          # Data loading and preprocessing
│   │   └── training.py      # Model training utilities
│   ├── visualization/
│   │   └── plotting.py      # Visualization utilities
│   └── run_benchmark.py     # Main benchmark script
├── results/                 # Generated plots and metrics
├── data/                   # Dataset storage
├── docker-compose.yml      # Docker compose configuration
├── Dockerfile             # Docker build configuration
└── requirements.txt       # Python dependencies
```

## Features

- **GRANDE Implementation**:
  - Soft decision trees with differentiable splitting
  - Instance-wise tree weighting using attention
  - End-to-end training with gradient descent

- **Comparison Framework**:
  - Benchmarking against XGBoost and CatBoost
  - Performance metrics (MSE, training time)
  - Automated visualization generation

## Quick Start

### Using Docker (Recommended)

1. Build and run using Docker Compose:
```bash
docker-compose up --build
```

This will:
- Build the Docker image
- Run the benchmark
- Save results in the `results/` directory

### Manual Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the benchmark:
```bash
python -m src.run_benchmark
```

## Results

The benchmark generates several outputs in the `results/` directory:
- `grande_training_curves.png`: Training and validation loss curves for GRANDE
- `test_loss_comparison.png`: Comparison of test loss across models
- `training_time_comparison.png`: Comparison of training times
- `model_comparison.csv`: Detailed metrics in CSV format

## Model Architecture

### GRANDE
- Uses soft decision trees with differentiable splitting functions
- Employs attention mechanism for tree ensemble weighting
- Trained end-to-end using gradient descent

### Comparison Models
- **XGBoost**: Gradient boosting with traditional decision trees
- **CatBoost**: Gradient boosting with ordered boosting and categorical feature support

## Customization

You can modify the benchmark parameters in `src/run_benchmark.py`:
```python
run_benchmark(
    n_samples=1000,    # Number of samples in dataset
    n_features=10,     # Number of features
    n_trees=5,        # Number of trees in GRANDE
    depth=3           # Depth of trees
)
```

## Contributing

Feel free to open issues or submit pull requests for improvements. Some areas for potential enhancement:
- Additional model architectures
- More comprehensive metrics
- Support for different types of datasets
- Hyperparameter optimization 