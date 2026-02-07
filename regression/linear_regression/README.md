# Linear Regression

Implementation of linear regression algorithm for learning purposes.

## Theory

Linear regression models the relationship between input features and continuous output by fitting a linear function:

```
y = wx + b
```

Where:
- `y`: Predicted output
- `x`: Input feature
- `w`: Weight (slope)
- `b`: Bias (intercept)

The goal is to find optimal parameters (w, b) that minimize the Mean Squared Error (MSE):

```
MSE = (1/m) Σ(prediction - actual)²
```

## Implementations

This directory contains two approaches to solving linear regression:

### 1. Iterative Optimization (`iterative/`)

Uses gradient descent with PyTorch to iteratively optimize parameters for `y = 2x + 10`.

**Architecture:**
- Single linear layer: `nn.Linear(1, 1)`
- Loss function: Mean Squared Error (MSE)
- Optimizer: Stochastic Gradient Descent (SGD) with learning rate 0.01
- Training: 500 epochs

**Data:**
- Training: x ∈ [0, 9], y = 2x + 10
- Testing: x ∈ [10, 19], y = 2x + 10

**Running:**
```bash
python iterative/linear_regression.py
```

**Expected Results:**
- Weight (w) ≈ 2.0
- Bias (b) ≈ 10.0

### 2. Analytical Solution (`analytical/`)

Closed-form solution using the Normal Equation that directly computes optimal parameters.

**Formula:** θ = (X^T X)^(-1) X^T y

**Advantages:**
- No hyperparameters (learning rate, epochs)
- Exact solution in single computation
- No iterative training needed

**To be implemented** - See `analytical/README.md` for implementation guide.

## Key Concepts

### Closed-form Solution

A [closed-form solution](https://en.wikipedia.org/wiki/Closed-form_expression) (or analytical solution) is an expression that can be evaluated in a finite number of standard operations. In contrast to iterative methods that approximate a solution through repeated calculations, a closed-form solution provides an exact answer using a direct formula.

**Example:**
- **Closed-form**: θ = (X^T X)^(-1) X^T y - computes optimal parameters directly
- **Iterative**: θ := θ - α∇J(θ) - approaches optimal parameters through many steps

The Normal Equation is a closed-form solution because it calculates the exact optimal parameters using matrix operations, without needing loops or convergence criteria.

## Visualization

The script plots predictions vs actual values using dual y-axes:
- Blue line: Model predictions
- Orange line: True values

Both lines should overlap if the model learned correctly.

## Reference

Based on concepts from [Deep Learning Book Chapter 5: Machine Learning Basics](https://www.deeplearningbook.org/contents/ml.html)
