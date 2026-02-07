import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class NormalEquation:
    """
    Implements multiple linear regression using the Normal Equation
    and provides visualization tools.
    """
    def __init__(self, X_data, y_data):
        self.X = np.asarray(X_data)
        self.y = np.asarray(y_data).flatten()

        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)

        if self.X.shape[0] != len(self.y):
            raise ValueError("X_data and y_data must have the same number of samples.")
        if self.X.shape[0] < self.X.shape[1] + 1:
            raise ValueError("Not enough data points to fit the model.")

        self.theta = None
        self._fit_model() # Automatically fit the model upon initialization

    def _fit_model(self):
        """
        Calculates the optimal weights (theta) using the Normal Equation.
        The design matrix X will be [1, x1, x2, ...].
        theta = (X^T X)^(-1) X^T y
        """
        # Prepare the design matrix X by adding a bias column of ones
        X_design = np.column_stack([np.ones(self.X.shape[0]), self.X])
        y_matrix = self.y.reshape(-1, 1)

        try:
            # Calculate (X^T X)^(-1) X^T y
            XtX_inv = np.linalg.inv(X_design.T @ X_design)
            self.theta = XtX_inv @ X_design.T @ y_matrix
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if (X^T X) is singular
            self.theta = np.linalg.pinv(X_design) @ y_matrix

    def get_intercept(self):
        """Returns the calculated intercept (theta_0)."""
        if self.theta is None:
            self._fit_model()
        return self.theta[0, 0]

    def get_weights(self):
        """Returns the calculated feature weights (theta_1, theta_2, ...)."""
        if self.theta is None:
            self._fit_model()
        return self.theta[1:].flatten()

    def predict(self, X_test):
        """Predicts y values for new X_test data."""
        X_test = np.asarray(X_test)
        if X_test.ndim == 1:
            # Reshape for single sample with multiple features or multiple samples with one feature
            if len(self.get_weights()) == 1:
                 X_test = X_test.reshape(-1, 1)
            else:
                 X_test = X_test.reshape(1, -1)


        X_design = np.column_stack([np.ones(X_test.shape[0]), X_test])

        if X_design.shape[1] != len(self.theta):
             raise ValueError(f"Expected {len(self.theta) - 1} features, but got {X_test.shape[1]}")

        return (X_design @ self.theta).flatten()

    def _calculate_sse(self, theta_candidate):
        """Calculates the Sum of Squared Errors (SSE) for a given theta_candidate vector."""
        # Prepare the design matrix X by adding a bias column of ones
        X_design = np.column_stack([np.ones(self.X.shape[0]), self.X])
        
        predictions = X_design @ theta_candidate.reshape(-1, 1)
        errors = self.y.reshape(-1, 1) - predictions
        return np.sum(errors**2)

    def draw_regression_plane(self):
        """
        For 2 features, draws the data points and the fitted regression plane.
        """
        if self.X.shape[1] != 2:
            print("Can only draw regression plane for 2 features.")
            return

        x1_range = np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), 10)
        x2_range = np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), 10)
        x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

        y_pred_mesh = self.get_intercept() + self.get_weights()[0] * x1_mesh + self.get_weights()[1] * x2_mesh

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X[:, 0], self.X[:, 1], self.y, color='blue', label='Data Points')
        ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, color='orange', alpha=0.5, label='Regression Plane')

        ax.set_xlabel('Feature X1')
        ax.set_ylabel('Feature X2')
        ax.set_zlabel('Target Y')
        ax.set_title('Multiple Linear Regression Plane')
        plt.show()

    def draw_loss_func_for_intercept(self, n_points=50):
        """
        Draws the Sum of Squared Errors (SSE) as a function of the intercept,
        holding all feature weights constant at their optimal values.
        """
        if self.theta is None:
            self._fit_model()

        optimal_theta = self.theta.flatten()
        intercept_range = np.linspace(optimal_theta[0] * 0.5, optimal_theta[0] * 1.5, n_points)
        if np.isclose(optimal_theta[0], 0):
            intercept_range = np.linspace(-5, 5, n_points)

        sse_values = []
        for intercept_val in intercept_range:
            current_theta = optimal_theta.copy()
            current_theta[0] = intercept_val
            sse_values.append(self._calculate_sse(current_theta))

        plt.figure(figsize=(8, 5))
        plt.plot(intercept_range, sse_values, label=f'Weights fixed at optimal')
        plt.axvline(optimal_theta[0], color='red', linestyle='--', label=f'Optimal Intercept: {optimal_theta[0]:.2f}')
        plt.xlabel('Intercept (θ₀)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.title('SSE vs. Intercept (Weights fixed)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def draw_loss_func_for_weight(self, feature_index, n_points=50):
        """
        Draws the Sum of Squared Errors (SSE) as a function of a specific feature's weight,
        holding the intercept and all other weights constant at their optimal values.

        Parameters
        ----------
        feature_index : int
            The index of the feature (0 for x1, 1 for x2, etc.) whose weight
            loss function is to be plotted.
        """
        if self.theta is None:
            self._fit_model()

        num_features = self.X.shape[1]
        if not (0 <= feature_index < num_features):
            raise ValueError(f"feature_index must be between 0 and {num_features - 1}.")

        # The index in the theta vector is feature_index + 1 because theta[0] is the intercept
        theta_index_to_vary = feature_index + 1
        optimal_theta = self.theta.flatten()
        
        weight_range = np.linspace(
            optimal_theta[theta_index_to_vary] * 0.5,
            optimal_theta[theta_index_to_vary] * 1.5,
            n_points
        )
        if np.isclose(optimal_theta[theta_index_to_vary], 0):
            weight_range = np.linspace(-5, 5, n_points)

        sse_values = []
        for weight_val in weight_range:
            current_theta = optimal_theta.copy()
            current_theta[theta_index_to_vary] = weight_val
            sse_values.append(self._calculate_sse(current_theta))

        plt.figure(figsize=(8, 5))
        plt.plot(weight_range, sse_values, label=f'Other params fixed at optimal')
        plt.axvline(optimal_theta[theta_index_to_vary], color='red', linestyle='--',
                    label=f'Optimal Weight θ_{feature_index + 1}: {optimal_theta[theta_index_to_vary]:.2f}')
        plt.xlabel(f'Weight θ_{feature_index + 1}')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.title(f'SSE vs. Weight θ_{feature_index + 1} (Other parameters fixed)')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example Usage:
if __name__ == "__main__":
    # Sample data with 2 features
    # y = 10 + 3*x1 + 5*x2 + noise
    np.random.seed(42)
    x1_sample = 2 * np.random.rand(100, 1)
    x2_sample = 3 * np.random.rand(100, 1)
    noise = np.random.randn(100, 1)
    y_sample = 10 + (3 * x1_sample) + (5 * x2_sample) + noise

    X_sample = np.column_stack([x1_sample, x2_sample])

    print("--- Multiple Linear Regression Example ---")
    model = NormalEquation(X_sample, y_sample)

    print(f"Optimal Intercept (theta_0): {model.get_intercept():.2f} (Expected approx. 10)")
    weights = model.get_weights()
    print(f"Optimal Weight for x1 (theta_1): {weights[0]:.2f} (Expected approx. 3)")
    print(f"Optimal Weight for x2 (theta_2): {weights[1]:.2f} (Expected approx. 5)")


    # Make a prediction
    x_new = np.array([[1, 1.5]]) # New data point with 2 features
    prediction = model.predict(x_new)
    print(f"\nPrediction for x_new = [1, 1.5]: {prediction[0]:.2f}")

    # Draw the regression plane
    model.draw_regression_plane()

    # Draw loss function for intercept
    model.draw_loss_func_for_intercept()

    # Draw loss function for each weight
    for i in range(len(weights)):
        model.draw_loss_func_for_weight(i)