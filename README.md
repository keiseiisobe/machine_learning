# Machine Learning Algorithms

Implementation of machine learning algorithms from scratch, following concepts from the [Deep Learning Book](https://www.deeplearningbook.org/).

## Repository Structure

Algorithms are organized by **task type**:

### Regression (Predicting Continuous Values)
- **Least Squares** - `regression/least_squares/`
  - Foundational theory of minimizing squared residuals
  - Applies to both linear and non-linear regression
- **Linear Regression** - `regression/linear_regression/`
  - Iterative approach (Gradient Descent)
  - Analytical approach (Normal Equation) - *to be implemented*

### Classification (Predicting Discrete Classes)
- **Linear SVM** - `classification/linear_svm/`
  - Binary classification with Support Vector Machines
- **Neural Network** - `classification/neural_network/`
  - Multi-class classification with feedforward neural network

## Running the Code

```bash
# Regression
python regression/linear_regression/iterative/linear_regression.py

# Classification
python classification/linear_svm/linear_svm.py
python classification/neural_network/simple_deep_learning.py
```

## Implementation Philosophy

- **From-scratch implementations**: Using NumPy to demonstrate the mathematics
- **Self-contained scripts**: Each algorithm is independent with no cross-dependencies
- **Visualizations**: All scripts generate matplotlib plots automatically
