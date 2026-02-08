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

## Applicability: Linear vs Non-linear Functions

### Terminology Clarification

It's important to understand the relationship between these terms:

**"Least Squares"** = The **objective/problem**: minimize Σ(yᵢ - ŷᵢ)²

**"Normal Equation"** = A **solution method**: the closed-form formula θ = (X^T X)^(-1) X^T y

**"Ordinary Least Squares (OLS)"** = Least squares applied to models **linear in parameters**
- Can be solved using the Normal Equation (closed-form solution)
- Example: y = θ₀ + θ₁x₁ + θ₂x₂

**"Non-linear Least Squares (NLLS)"** = Least squares applied to models **non-linear in parameters**
- **Cannot** be solved using the Normal Equation (no closed-form solution)
- Must use iterative optimization methods
- Example: y = θ₀ * e^(θ₁*x)

**The hierarchy:**
```
Least Squares (objective: minimize squared errors)
    ├── Ordinary Least Squares (OLS)
    │   └── Solution: Normal Equation ✅
    │
    └── Non-linear Least Squares (NLLS)
        └── Solution: Iterative methods ❌ (no normal equation)
```

The least squares method (minimizing Σ(yᵢ - ŷᵢ)²) can be applied to both linear and non-linear functions, but there's an important distinction:

### Case 1: Linear in Parameters (Ordinary Least Squares)

Functions that are **non-linear in the input variables (x)** but **linear in the parameters (θ)**:

```
y = θ₀ + θ₁x₁ + θ₂x₂ + θ₃x₁² + θ₄x₂² + θ₅x₁x₂
```

Even though this includes x₁² and x₂², it's still **linear in θ** (each parameter appears only to the first power and isn't nested).

**Solution approach:**
- Transform your features: X' = [1, x₁, x₂, x₁², x₂², x₁x₂]
- Apply the normal equation: θ = (X'ᵀX')⁻¹X'ᵀy
- Get a **closed-form solution** (no iteration needed)

**Examples:**
- [Polynomial regression](https://en.wikipedia.org/wiki/Polynomial_regression): y = θ₀ + θ₁x + θ₂x² + θ₃x³
- Interaction terms: y = θ₀ + θ₁x₁ + θ₂x₂ + θ₃x₁x₂
- Transformed features: y = θ₀ + θ₁log(x) + θ₂√x

### Case 2: Non-linear in Parameters (Non-linear Least Squares)

Functions that are **non-linear in the parameters (θ)**:

```
y = θ₀ * e^(θ₁*x)           # Exponential
y = θ₀ / (1 + θ₁*e^(-θ₂*x)) # Logistic
y = θ₀ * sin(θ₁*x + θ₂)     # Sinusoidal
```

Here, parameters appear in exponents, denominators, or other non-linear ways.

**Solution approach:**
- **Cannot** use the normal equation (no closed-form solution)
- Must use **iterative optimization** methods:
  - [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)
  - [Gauss-Newton algorithm](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)
  - [Levenberg-Marquardt algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)

### Summary Table

| Function Type | Example | Solution Method | Closed-form? |
|--------------|---------|-----------------|--------------|
| Linear | y = θ₀ + θ₁x₁ + θ₂x₂ | Normal Equation | ✅ Yes |
| Non-linear in x, linear in θ | y = θ₀ + θ₁x² + θ₂sin(x) | Normal Equation | ✅ Yes |
| Non-linear in θ | y = θ₀ * e^(θ₁*x) | Iterative methods | ❌ No |

**Key insight:** The least squares **objective** (minimize squared errors) applies to all cases. What changes is the **solution method**.

## Finding Optimal Parameters

The least squares method defines the **problem**: minimize SSR.

There are several ways to **solve** this optimization problem:

### 1. Analytical Solution (Normal Equation)
Solve directly by setting ∇_θ SSR = 0:
```
θ = (XᵀX)⁻¹Xᵀy
```
This is implemented in `../linear_regression/analytical/` (to be implemented).

### 2. Iterative Optimization (Gradient Descent)
Update θ iteratively using gradients:
```
θ := θ - α ∇_θ MSE
```
This is implemented in `../linear_regression/iterative/`.

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

## Next Steps

After understanding the least squares cost function:

1. Derive the gradient: ∇_θ SSR = -2Xᵀ(y - Xθ)
2. Set gradient to zero and solve (leads to Normal Equation)
3. Implement the Normal Equation solution
4. Compare different methods (analytical vs iterative)

## Reference

- Deep Learning Book, Section 5.1.4: Example: Linear Regression
- The method of least squares was developed by Gauss and Legendre around 1800
