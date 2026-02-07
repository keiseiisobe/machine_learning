# Least Squares Method

The method of least squares is a standard approach to finding the best-fit line (or hyperplane) by minimizing the sum of squared residuals.

## Theory

### The Problem

Given data points (x₁, y₁), (x₂, y₂), ..., (xₘ, yₘ), find parameters θ that minimize the squared error:

```
minimize: Σᵢ (yᵢ - ŷᵢ)²
```

Where ŷᵢ = θ₀ + θ₁xᵢ is the predicted value.

### Residuals

The **residual** (or error) for each data point is:

```
eᵢ = yᵢ - ŷᵢ = yᵢ - (θ₀ + θ₁xᵢ)
```

The least squares method minimizes the **sum of squared residuals (SSR)**:

```
SSR = Σᵢ eᵢ² = Σᵢ (yᵢ - θ₀ - θ₁xᵢ)²
```

### Why Square the Errors?

1. **Positive values**: Squaring ensures all errors are positive (otherwise positive and negative errors could cancel)
2. **Penalizes large errors**: Larger errors are penalized more heavily (quadratic penalty)
3. **Differentiable**: Squared function is smooth and differentiable everywhere
4. **Unique solution**: Convex optimization problem with single global minimum

## Mathematical Formulation

### In Matrix Form

For multiple data points, we can express this in matrix notation:

```
X = [[1  x₁]     y = [y₁]     θ = [θ₀]
     [1  x₂]          [y₂]          [θ₁]
     [... ...]        [...]
     [1  xₘ]]         [yₘ]]
```

The predictions are: ŷ = Xθ

The residual vector is: e = y - Xθ

The sum of squared residuals is:

```
SSR = ||e||² = ||y - Xθ||² = (y - Xθ)ᵀ(y - Xθ)
```

Expanding this:

```
SSR = yᵀy - 2θᵀXᵀy + θᵀXᵀXθ
```

### Cost Function

In machine learning, we often use Mean Squared Error (MSE):

```
MSE = (1/m) Σᵢ (yᵢ - ŷᵢ)² = (1/m) ||y - Xθ||²
```

Where m is the number of examples. Note: MSE = SSR/m

## Implementation Guide

### Step 1: Prepare Data

```python
import numpy as np

# Generate data: y = 2x + 10
X = np.arange(10).reshape(-1, 1)  # [0, 1, 2, ..., 9]
y = 2 * X + 10

# Add bias column
X_with_bias = np.column_stack([np.ones(len(X)), X])
```

### Step 2: Define Cost Function

```python
def compute_ssr(X, y, theta):
    """Compute sum of squared residuals"""
    predictions = X @ theta
    residuals = y - predictions
    ssr = np.sum(residuals ** 2)
    return ssr

def compute_mse(X, y, theta):
    """Compute mean squared error"""
    m = len(y)
    predictions = X @ theta
    mse = (1 / m) * np.sum((y - predictions) ** 2)
    return mse
```

### Step 3: Compute Residuals

```python
# Try some initial theta values
theta = np.array([[5.0], [1.0]])  # θ₀=5, θ₁=1

predictions = X_with_bias @ theta
residuals = y - predictions

print(f"Predictions: {predictions.flatten()}")
print(f"Actual: {y.flatten()}")
print(f"Residuals: {residuals.flatten()}")
print(f"SSR: {compute_ssr(X_with_bias, y, theta)}")
print(f"MSE: {compute_mse(X_with_bias, y, theta)}")
```

### Step 4: Visualize Fit Quality

```python
import matplotlib.pyplot as plt

plt.scatter(X, y, label='Data points', color='blue')
plt.plot(X, predictions, label='Fit line', color='red')

# Draw residual lines
for i in range(len(X)):
    plt.plot([X[i], X[i]], [y[i], predictions[i]],
             color='gray', linestyle='--', linewidth=0.5)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'Least Squares Fit (SSR = {compute_ssr(X_with_bias, y, theta):.2f})')
plt.show()
```

## Finding Optimal Parameters

The least squares method defines the **problem**: minimize SSR.

There are several ways to **solve** this optimization problem:

### 1. Analytical Solution (Normal Equation)
Solve directly by setting ∇_θ SSR = 0:
```
θ = (XᵀX)⁻¹Xᵀy
```
This is implemented in the parent `analytical/` directory.

### 2. Iterative Optimization (Gradient Descent)
Update θ iteratively using gradients:
```
θ := θ - α ∇_θ MSE
```
This is implemented in the `iterative/` directory.

### 3. Other Methods
- QR decomposition
- SVD (Singular Value Decomposition)
- Cholesky decomposition

## Expected Results

For the function y = 2x + 10:

**Optimal parameters:**
- θ₀ = 10.0 (intercept)
- θ₁ = 2.0 (slope)

**At optimal θ:**
- SSR = 0 (perfect fit, no residuals)
- MSE = 0

**With non-optimal θ = [5, 1]:**
- All predictions will be off
- Positive SSR value
- Visible residuals in plot

## Key Properties of Least Squares

1. **Unbiased**: Expected value of parameter estimates equals true parameters
2. **Best Linear Unbiased Estimator (BLUE)**: Among all linear unbiased estimators, least squares has minimum variance (Gauss-Markov theorem)
3. **Maximum Likelihood**: Under Gaussian noise assumption, least squares = maximum likelihood estimation
4. **Convex**: Cost function is convex, guaranteeing global minimum

## Testing Your Implementation

```python
# Test 1: Perfect linear data
X = np.array([[1], [2], [3]])
y = 2 * X + 10  # [12, 14, 16]
theta_optimal = np.array([[10], [2]])

# Should get SSR = 0
assert compute_ssr(X_with_bias, y, theta_optimal) == 0

# Test 2: Suboptimal parameters
theta_bad = np.array([[0], [1]])
ssr_bad = compute_ssr(X_with_bias, y, theta_bad)

# Should get SSR > 0
assert ssr_bad > 0
```

## Next Steps

After understanding the least squares cost function:

1. Derive the gradient: ∇_θ SSR = -2Xᵀ(y - Xθ)
2. Set gradient to zero and solve (leads to Normal Equation)
3. Implement the Normal Equation solution
4. Compare different methods (analytical vs iterative)

## Reference

- Deep Learning Book, Section 5.1.4: Example: Linear Regression
- The method of least squares was developed by Gauss and Legendre around 1800
