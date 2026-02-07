# Linear Regression - Normal Equation

Analytical closed-form solution for linear regression that directly computes optimal parameters without iterative optimization.

## Overview

The Normal Equation is a **solution method** for linear regression that uses the [least squares](../../least_squares/) objective.

**Relationship:**
- **Least Squares** = The optimization objective (minimize Σ(y - ŷ)²) - see [../../least_squares/](../../least_squares/)
- **Normal Equation** = An analytical solution method (works only for models linear in parameters)
- **Linear Regression** = The model (ŷ = Xθ, linear in parameters)

For a deeper understanding of the least squares method and why we minimize squared residuals, see the [Least Squares documentation](../../least_squares/README.md).

## Theory

The Normal Equation solves for parameters θ that minimize the Mean Squared Error by setting the gradient to zero and solving analytically.

### The Formula

```
θ = (X^T X)^(-1) X^T y
```

Where:
- `θ`: Parameter vector [θ₀, θ₁, ..., θₙ]^T containing bias and weights
- `X`: Design matrix (m × n+1) including bias column
- `X^T`: Transpose of X
- `y`: Target values (m × 1)
- `(X^T X)^(-1)`: Inverse of (X^T X)

### Design Matrix Structure

For simple linear regression y = wx + b, the design matrix must include a bias column of ones:

```
X = [[1  x₁]
     [1  x₂]
     [1  x₃]
     [...  ...]]
```

The first column of 1s allows the equation to learn the bias term θ₀.

## Mathematical Derivation

Starting from the [least squares](../../least_squares/README.md) cost function:

1. **Cost function**: J(θ) = (1/2m)(Xθ - y)^T(Xθ - y)  [Least squares objective]
2. **Take derivative**: ∇_θ J(θ) = (1/m)(X^T Xθ - X^T y)
3. **Set gradient to zero**: X^T Xθ - X^T y = 0
4. **Solve for θ**: θ = (X^T X)^(-1) X^T y  [Normal Equation]

This derivation is only possible because the model is **linear in parameters**. For non-linear models, we must use iterative methods.

## Implementation Guide

### Step 1: Prepare Data

```python
import numpy as np

# Generate data: y = 2x + 10
X = np.arange(10).reshape(-1, 1)  # Shape: (10, 1)
y = 2 * X + 10  # Shape: (10, 1)

# Add bias column of ones
X_with_bias = np.column_stack([np.ones(len(X)), X])  # Shape: (10, 2)
```

### Step 2: Compute Normal Equation

```python
# θ = (X^T X)^(-1) X^T y
X_transpose = X_with_bias.T
theta = np.linalg.inv(X_transpose @ X_with_bias) @ X_transpose @ y
```

Or more compactly:
```python
theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
```

### Step 3: Make Predictions

```python
# For new data
X_test = np.arange(10, 20).reshape(-1, 1)
X_test_with_bias = np.column_stack([np.ones(len(X_test)), X_test])

predictions = X_test_with_bias @ theta
```

### Step 4: Handle Singular Matrices

If X^T X is not invertible (singular), use the Moore-Penrose pseudo-inverse:

```python
theta = np.linalg.pinv(X_with_bias) @ y
```

## Expected Results

For the target function y = 2x + 10:

```python
theta[0]  # Should equal 10.0 (bias)
theta[1]  # Should equal 2.0 (weight)
```

The Normal Equation finds the exact optimal parameters in one computation.

## Advantages

- **No hyperparameters**: No learning rate or iteration count to tune
- **Exact solution**: Computes mathematically optimal parameters
- **Single computation**: No iterative training loop needed
- **Deterministic**: Always produces the same result

## Disadvantages

- **Computational complexity**: O(n³) due to matrix inversion
- **Memory intensive**: Must compute and store X^T X
- **Large features**: Becomes slow when n > 10,000
- **Matrix inversion**: Fails if X^T X is singular (use pinv instead)

## Comparison with Gradient Descent

| Aspect | Normal Equation | Gradient Descent |
|--------|----------------|------------------|
| Iterations | None | Many (500+) |
| Learning rate | Not needed | Must tune (0.01) |
| Complexity | O(n³) | O(kn²m) |
| Large n | Slow | Fast |
| Exact solution | Yes | Approximate |
| Implementation | Simple | More complex |

Where k = iterations, n = features, m = examples

## Testing

To verify your implementation works:

1. **Test Case 1**: Perfect linear data (y = 2x + 10)
   - Expected: θ = [10, 2]

2. **Test Case 2**: Add Gaussian noise
   - Expected: θ ≈ [10, 2]

3. **Test Case 3**: Compare with gradient descent results
   - Both methods should converge to similar parameters

## Visualization

Plot predictions vs actual values to verify:
```python
import matplotlib.pyplot as plt

plt.scatter(X, y, label='Training data')
plt.plot(X_test, predictions, color='red', label='Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

## When Can You Use the Normal Equation?

The Normal Equation **only works** when your model is **linear in parameters**:

✅ **Can use Normal Equation:**
- Linear regression: y = θ₀ + θ₁x
- Polynomial regression: y = θ₀ + θ₁x + θ₂x² (linear in θ, non-linear in x)
- Multiple regression: y = θ₀ + θ₁x₁ + θ₂x₂

❌ **Cannot use Normal Equation:**
- Exponential: y = θ₀ * e^(θ₁*x) (non-linear in θ₁)
- Logistic: y = θ₀ / (1 + θ₁*e^(-θ₂*x)) (non-linear in parameters)

### Why the Limitation?

The analytical solution provided by the Normal Equation, derived by setting the gradient of the Mean Squared Error (MSE) to zero, implicitly assumes that the loss function is **convex**. A convex loss function guarantees a single, unique global minimum.

When a model is non-linear in its parameters, the resulting MSE loss function is typically **non-convex**. This creates a complex landscape with multiple "valleys" (local minima), as shown below for the model `y = a * sin(b*x)`.

![Non-Convex Loss Surface](./non_linear_loss_surface.png)

Non-convex functions can have:
- Multiple local minima
- Local maxima
- Saddle points

At all these points, the gradient with respect to the parameters is zero. Consequently, simply solving for where the gradient is zero (as the Normal Equation does) would not reliably identify the global minimum. It might find a local minimum, a maximum, or a saddle point, and it cannot distinguish between them. For such non-convex problems, iterative optimization methods like Gradient Descent are necessary, though they also face challenges like getting stuck in local minima.

For non-linear models, you must use iterative optimization (gradient descent, Gauss-Newton, etc.).

See [Least Squares - Applicability](../../least_squares/README.md#applicability-linear-vs-non-linear-functions) for details.

## Reference

- Deep Learning Book, Section 5.1.4: Example: Linear Regression
- The formula derives from solving ∇_w MSE(w) = 0 for the [least squares objective](../../least_squares/)
