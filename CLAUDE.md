# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a learning repository implementing machine learning algorithms from scratch, following concepts from the [Deep Learning Book](https://www.deeplearningbook.org/). Each algorithm is implemented as a standalone Python script in its own directory.

## Running Code

Each algorithm is a self-contained script that can be run directly:

```bash
# REGRESSION ALGORITHMS
# Linear regression - iterative optimization (PyTorch)
python regression/linear_regression/iterative/linear_regression.py

# Linear regression - analytical solution (to be implemented)
# python regression/linear_regression/analytical/normal_equation.py

# CLASSIFICATION ALGORITHMS
# Linear SVM (from scratch with NumPy)
python classification/linear_svm/linear_svm.py

# Neural network (PyTorch with FashionMNIST)
python classification/neural_network/simple_deep_learning.py
```

All scripts generate visualizations using matplotlib that display automatically.

## Code Architecture

### Implementation Philosophy

- **From-scratch implementations**: Algorithms use NumPy directly when possible to demonstrate the math
- **Mixed approach**: Some use PyTorch for convenience (linear_regression, simple_deep_learning), others are pure NumPy (linear_svm)
- **Self-contained scripts**: Each file is a complete, runnable example with data generation/loading, training, and visualization
- **No shared utilities**: Each algorithm is independent with no cross-dependencies

### Directory Structure

Algorithms are organized by **task type** (regression vs classification):

```
regression/                          # Predicting continuous values
├── least_squares/                   # General theory (applies to all regression)
│   └── README.md                    # Explains the least squares objective
│
└── linear_regression/               # Linear models
    ├── README.md
    ├── iterative/                   # Solution: Gradient descent
    │   └── linear_regression.py
    └── analytical/                  # Solution: Normal equation (to be implemented)
        └── README.md

classification/                      # Predicting discrete classes
├── linear_svm/                      # Binary classification with SVM
│   ├── linear_svm.py
│   ├── README.md
│   └── Figure_1.png
│
└── neural_network/                  # Multi-class with neural network
    ├── simple_deep_learning.py
    └── data/                        # Auto-downloaded FashionMNIST
        └── FashionMNIST/
```

**Key architectural principle:** `least_squares/` is at the `regression/` level because the least squares method (minimizing sum of squared residuals) is a general optimization objective that applies to both linear and non-linear regression, not just linear regression.

### Implementation Patterns

**Linear Regression - Iterative** (regression/linear_regression/iterative/linear_regression.py):
- Uses PyTorch nn.Module for model definition
- Implements train() and test() functions that take model/optimizer as parameters
- Target function: y = 2x + 10
- Uses SGD optimizer, MSE loss
- Visualizes predictions vs true values with dual y-axes

**Linear Regression - Analytical** (regression/linear_regression/analytical/):
- Analytical solution: θ = (X^T X)^(-1) X^T y
- To be implemented using NumPy
- No training loop needed - direct computation
- See README.md for implementation guide

**Linear SVM** (classification/linear_svm/linear_svm.py):
- Pure NumPy/scikit-learn implementation
- Custom Hinge_Loss class with gradient computation
- Linear_SVM class with fit() method
- Uses sklearn.datasets.make_blobs for binary classification data
- Labels are -1/+1 (not 0/1)
- Visualizes decision boundary with scatter plot

**Neural Network** (classification/neural_network/simple_deep_learning.py):
- PyTorch neural network on FashionMNIST
- 3-layer MLP: 784 → 512 → 512 → 10
- Downloads dataset to data/ directory automatically
- Train/test loop pattern with batch processing
- Uses DataLoader with batch_size=64

## Dependencies

Core dependencies used across implementations:
- numpy: Matrix operations and from-scratch implementations
- torch: Neural network framework (regression/linear_regression, classification/neural_network)
- matplotlib: Visualization
- scikit-learn: Dataset generation (classification/linear_svm only)
- torchvision: FashionMNIST dataset (classification/neural_network only)

**Dependency Management:**
- `requirements.txt` - Lists all required packages
- `setup_venv.sh` - Automated setup script for virtual environment
- `venv/` - Virtual environment directory (gitignored)

## Development Notes

- Use virtual environment (`venv/`) for consistent dependencies across platforms
- Run `./setup_venv.sh` to set up the environment
- No test suite
- Scripts print progress to stdout during training
- Matplotlib plots use plt.show() which blocks until window is closed
- Data files and caches are gitignored (.gitignore includes standard Python patterns)

## Documentation Guidelines

When creating or updating documentation (README.md files):
- Add references with URLs for difficult or technical terms (e.g., "closed-form solution", "convex optimization", "BLUE theorem")
- Prefer authoritative sources: Wikipedia, academic papers, official documentation
- Include the reference inline as a markdown link where the term first appears
- Explain the term in context before or after introducing it
