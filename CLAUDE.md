# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a learning repository implementing machine learning algorithms from scratch, following concepts from the [Deep Learning Book](https://www.deeplearningbook.org/). Each algorithm is implemented as a standalone Python script in its own directory.

## Running Code

Each algorithm is a self-contained script that can be run directly:

```bash
# Linear regression - iterative optimization (PyTorch)
python linear_regression/iterative/linear_regression.py

# Linear regression - analytical solution (to be implemented)
# python linear_regression/analytical/normal_equation.py

# Linear SVM (from scratch with NumPy)
python linear_svm/linear_svm.py

# Simple neural network (PyTorch with FashionMNIST)
python simple_deep_learing/simple_deep_learning.py
```

All scripts generate visualizations using matplotlib that display automatically.

## Code Architecture

### Implementation Philosophy

- **From-scratch implementations**: Algorithms use NumPy directly when possible to demonstrate the math
- **Mixed approach**: Some use PyTorch for convenience (linear_regression, simple_deep_learning), others are pure NumPy (linear_svm)
- **Self-contained scripts**: Each file is a complete, runnable example with data generation/loading, training, and visualization
- **No shared utilities**: Each algorithm is independent with no cross-dependencies

### Directory Structure

```
linear_regression/           # Linear regression implementations
├── iterative/              # Iterative optimization (gradient descent)
│   └── linear_regression.py
├── analytical/             # Analytical solution (normal equation)
│   └── README.md          # (implementation to be added)
└── README.md

linear_svm/                  # Linear SVM from scratch
├── linear_svm.py
├── README.md
└── Figure_1.png            # Example output visualization

simple_deep_learing/         # Neural network with FashionMNIST
├── simple_deep_learning.py
└── data/                   # Auto-downloaded FashionMNIST dataset
    └── FashionMNIST/
```

### Implementation Patterns

**Linear Regression - Iterative** (linear_regression/iterative/linear_regression.py):
- Uses PyTorch nn.Module for model definition
- Implements train() and test() functions that take model/optimizer as parameters
- Target function: y = 2x + 10
- Uses SGD optimizer, MSE loss
- Visualizes predictions vs true values with dual y-axes

**Linear Regression - Analytical** (linear_regression/analytical/):
- Analytical solution: θ = (X^T X)^(-1) X^T y
- To be implemented using NumPy
- No training loop needed - direct computation
- See README.md for implementation guide

**Linear SVM** (linear_svm/linear_svm.py):
- Pure NumPy/scikit-learn implementation
- Custom Hinge_Loss class with gradient computation
- Linear_SVM class with fit() method
- Uses sklearn.datasets.make_blobs for binary classification data
- Labels are -1/+1 (not 0/1)
- Visualizes decision boundary with scatter plot

**Simple Deep Learning** (simple_deep_learing/simple_deep_learning.py):
- PyTorch neural network on FashionMNIST
- 3-layer MLP: 784 → 512 → 512 → 10
- Downloads dataset to data/ directory automatically
- Train/test loop pattern with batch processing
- Uses DataLoader with batch_size=64

## Dependencies

Core dependencies used across implementations:
- numpy: Matrix operations and from-scratch implementations
- torch: Neural network framework (linear_regression, simple_deep_learning)
- matplotlib: Visualization
- scikit-learn: Dataset generation (linear_svm only)
- torchvision: FashionMNIST dataset (simple_deep_learning only)

## Development Notes

- No package manager config files (no requirements.txt, pyproject.toml, etc.)
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
