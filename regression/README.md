# Regression

Regression algorithms for predicting continuous values.

## Directory Organization

```
regression/
├── README.md (this file)
├── least_squares/              # Cost function / optimization objective
└── linear_regression/          # Model type
```

## Understanding the Relationship: Least Squares vs Linear Regression

**Important distinction:** "Least squares" and "linear regression" are closely linked but **not synonymous**.

### Least Squares = Cost Function

[Least squares](./least_squares/) is an **optimization objective** that minimizes the sum of squared residuals:

```
minimize: Σ(yᵢ - ŷᵢ)²
```

#### Terminology: SSR vs MSE vs Least Squares

You'll see different but related terms used interchangeably:

- **SSR (Sum of Squared Residuals)** = Σ(yᵢ - ŷᵢ)²
- **MSE (Mean Squared Error)** = (1/m) Σ(yᵢ - ŷᵢ)² = SSR / m
- **Least Squares** = The method that minimizes SSR

**Key relationship:** MSE and SSR differ only by a constant factor (1/m, where m = number of samples).

**For optimization:** Minimizing SSR ≡ Minimizing MSE
- Both give the **same optimal parameters θ**
- The constant (1/m) doesn't affect the location of the minimum
- Taking derivatives: ∇_θ MSE = (1/m) ∇_θ SSR (gradients are proportional)

**Usage conventions:**
- **Statistics:** Typically use "SSR" or "sum of squares"
- **Machine Learning:** Typically use "MSE" (represents average error per sample)
- **Deep Learning:** Often called "L2 loss" or "squared error loss"

**In this repository:** "Least squares" refers to minimizing squared errors (whether SSR or MSE), since they're optimization-equivalent. The key property is the **squaring** of errors, not the averaging.

### Linear Regression = Model Type

[Linear regression](./linear_regression/) is a **model** where predictions are linear in parameters:

```
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ...
```

## They Are NOT the Same Thing

### Linear Regression Can Use Different Cost Functions

[Linear regression](https://en.wikipedia.org/wiki/Linear_regression) models are **often fitted using least squares**, but they may also be fitted in other ways:

1. **Least Absolute Deviations (LAD)** - Minimize L1 norm instead of L2:
   ```
   minimize: Σ|yᵢ - ŷᵢ|
   ```
   - More robust to outliers than least squares
   - Also called [Least Absolute Residuals](https://en.wikipedia.org/wiki/Least_absolute_deviations)

2. **Ridge Regression** - Penalized least squares with L2 penalty:
   ```
   minimize: Σ(yᵢ - ŷᵢ)² + λΣθⱼ²
   ```
   - Adds regularization to prevent overfitting
   - See [Ridge Regression](https://en.wikipedia.org/wiki/Ridge_regression)

3. **Lasso** - Penalized least squares with L1 penalty:
   ```
   minimize: Σ(yᵢ - ŷᵢ)² + λΣ|θⱼ|
   ```
   - Performs feature selection by driving some coefficients to zero
   - See [Lasso Regression](https://en.wikipedia.org/wiki/Lasso_(statistics))

4. **Huber Loss** - Robust to outliers:
   ```
   L(e) = { ½e²           if |e| ≤ δ
          { δ(|e| - ½δ)   otherwise
   ```
   - Behaves like L2 for small errors, L1 for large errors

**Why use alternatives?** Least squares (whether SSR or MSE) assigns high importance to large errors due to the squaring operation. If your dataset has many outliers, least squares may fit the outliers more than the true data. Robust cost functions (LAD, Huber) handle outliers better.

### Least Squares Can Be Used for Non-Linear Models

Conversely, [least squares](./least_squares/) can fit models that are **not linear**:

1. **Polynomial Regression** - Non-linear in input, linear in parameters:
   ```
   y = θ₀ + θ₁x + θ₂x² + θ₃x³
   ```
   - Can use normal equation (closed-form solution)

2. **Non-linear Least Squares** - Non-linear in parameters:
   ```
   y = θ₀ * e^(θ₁*x)          (exponential)
   y = θ₀ / (1 + θ₁*e^(-θ₂*x)) (logistic)
   ```
   - Must use iterative methods (Gauss-Newton, Levenberg-Marquardt)

## Summary Table

| Aspect | Least Squares | Linear Regression |
|--------|--------------|-------------------|
| **What is it?** | Cost function / Objective | Model type |
| **Definition** | Minimize Σ(y - ŷ)² | ŷ = θ₀ + θ₁x₁ + ... |
| **Applies to** | Any model (linear or non-linear) | Only linear-in-parameters models |
| **Alternatives** | LAD, Huber, Quantile loss, etc. | Non-linear models (neural nets, trees, etc.) |

## Key Insight

**This directory structure reflects this distinction:**

- `least_squares/` - Contains theory about the optimization objective (applicable broadly)
- `linear_regression/` - Contains implementations of linear models (often uses least squares, but not exclusively)

This separation allows for future expansion:
- Add `non_linear_regression/` that also uses least squares
- Add alternative fitting methods to `linear_regression/` (LAD, ridge, lasso)

## References

- [Wikipedia: Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
- [Wikipedia: Least Squares](https://en.wikipedia.org/wiki/Least_squares)
- [Wikipedia: Robust Regression](https://en.wikipedia.org/wiki/Robust_regression)
- Deep Learning Book, Section 5.1.4: Example: Linear Regression
