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

## Setup

### 1. Create Virtual Environment

```bash
# Run the setup script
./setup_venv.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### 2. Activate Virtual Environment

**Every time** you work on the project:

```bash
# macOS/Linux
source venv/bin/activate

# You should see (venv) in your prompt
```

### 3. Deactivate When Done

```bash
deactivate
```

## Running the Code

**Make sure the virtual environment is activated first!**

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
