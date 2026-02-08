import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegressionPartialDerivatives:
    """
    Implements simple linear regression by analytically solving for the optimal
    intercept and slope using formulas derived from setting partial derivatives
    of the Sum of Squared Errors (SSE) to zero.
    Also provides visualization tools for the loss function.

    y = beta_1 * x + beta_0
    """
    def __init__(self, x_data, y_data):
        self.x = np.asarray(x_data).flatten()
        self.y = np.asarray(y_data).flatten()

        if len(self.x) != len(self.y):
            raise ValueError("x_data and y_data must have the same number of samples.")
        if len(self.x) < 2:
            raise ValueError("At least two data points are required for linear regression.")

        self.intercept = None
        self.slope = None
        self._fit_model() # Automatically fit the model upon initialization

    def _fit_model(self):
        """
        Calculates the optimal intercept and slope using the algebraic formulas
        derived from setting the partial derivatives of the SSE with respect
        to the intercept and slope to zero.
        """
        n = len(self.x)
        sum_x = np.sum(self.x)
        sum_y = np.sum(self.y)
        sum_x_sq = np.sum(self.x**2)
        sum_xy = np.sum(self.x * self.y)

        # Calculate slope (beta_1)
        # beta_1 = (n * sum(x_i * y_i) - sum(x_i) * sum(y_i)) / (n * sum(x_i^2) - (sum(x_i))^2)
        denominator = (n * sum_x_sq - sum_x * sum_x)
        if denominator == 0:
            raise ValueError("Cannot calculate slope. The variance of x is zero (all x values are the same).")
        self.slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Calculate intercept (beta_0)
        # beta_0 = mean(y) - beta_1 * mean(x)
        self.intercept = (sum_y - self.slope * sum_x) / n

    def get_intercept(self):
        """Returns the calculated intercept (beta_0)."""
        if self.intercept is None:
            self._fit_model()
        return self.intercept

    def get_slope(self):
        """Returns the calculated slope (beta_1)."""
        if self.slope is None:
            self._fit_model()
        return self.slope

    def _predict(self, x_values, intercept, slope):
        """Helper to predict y values given x, intercept, and slope."""
        return intercept + slope * x_values

    def _calculate_sse(self, intercept, slope):
        """Calculates the Sum of Squared Errors (SSE) for given intercept and slope."""
        predictions = self._predict(self.x, intercept, slope)
        errors = self.y - predictions
        return np.sum(errors**2)

    def draw_opt_func(self, x_sample, y_sample):
        # Plot the data and the regression line
        plt.figure(figsize=(8, 6))
        plt.scatter(x_sample, y_sample, label='Data Points')
        plt.plot(x_sample, self._predict(x_sample, self.intercept, self.slope), color='red', label='Regression Line')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Simple Linear Regression')
        plt.legend()
        plt.grid(True)
        plt.show()

        
    def draw_loss_func_for_intercept(self, n_points=50):
        """
        Draws the Sum of Squared Errors (SSE) as a function of the intercept,
        holding the slope constant at its optimal value.
        """
        if self.intercept is None or self.slope is None:
            self._fit_model()

        intercept_range = np.linspace(self.intercept * 0.5, self.intercept * 1.5, n_points)
        if np.isclose(self.intercept, 0):
            intercept_range = np.linspace(-5, 5, n_points)

        sse_values = [self._calculate_sse(i, self.slope) for i in intercept_range]

        plt.figure(figsize=(8, 5))
        plt.plot(intercept_range, sse_values, label=f'Slope fixed at {self.slope:.2f}')
        plt.axvline(self.intercept, color='red', linestyle='--', label=f'Optimal Intercept: {self.intercept:.2f}')
        plt.xlabel('Intercept (β₀)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.title('SSE vs. Intercept (Slope fixed)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def draw_loss_func_for_slope(self, n_points=50):
        """
        Draws the Sum of Squared Errors (SSE) as a function of the slope,
        holding the intercept constant at its optimal value.
        """
        if self.intercept is None or self.slope is None:
            self._fit_model()

        slope_range = np.linspace(self.slope * 0.5, self.slope * 1.5, n_points)
        if np.isclose(self.slope, 0):
            slope_range = np.linspace(-5, 5, n_points)

        sse_values = [self._calculate_sse(self.intercept, s) for s in slope_range]

        plt.figure(figsize=(8, 5))
        plt.plot(slope_range, sse_values, label=f'Intercept fixed at {self.intercept:.2f}')
        plt.axvline(self.slope, color='red', linestyle='--', label=f'Optimal Slope: {self.slope:.2f}')
        plt.xlabel('Slope (β₁)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.title('SSE vs. Slope (Intercept fixed)')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example Usage:
if __name__ == "__main__":
    # Sample data
    x_sample = np.array([1, 2, 3, 4, 5])
    y_sample = np.array([2, 4, 5, 4, 5]) # A simple linear trend with some noise

    print("--- Simple Linear Regression Example (Partial Derivatives Solution) ---")
    model = SimpleLinearRegressionPartialDerivatives(x_sample, y_sample)

    print(f"Optimal Intercept (β₀): {model.get_intercept():.2f}")
    print(f"Optimal Slope (β₁): {model.get_slope():.2f}")

    # Draw optimized function surface
    model.draw_opt_func(x_sample, y_sample)

    # Draw loss function for intercept
    model.draw_loss_func_for_intercept()

    # Draw loss function for slope
    model.draw_loss_func_for_slope()

    # Example with perfect linear data
    print("\n--- Perfect Linear Data Example (y = 2x + 10) ---")
    x_perfect = np.arange(10)
    y_perfect = 2 * x_perfect + 10
    model_perfect = SimpleLinearRegressionPartialDerivatives(x_perfect, y_perfect)
    print(f"Optimal Intercept (β₀): {model_perfect.get_intercept():.2f}")
    print(f"Optimal Slope (β₁): {model_perfect.get_slope():.2f}")
    assert np.isclose(model_perfect.get_intercept(), 10.0), "Perfect data: Intercept is not 10.0"
    assert np.isclose(model_perfect.get_slope(), 2.0), "Perfect data: Slope is not 2.0"
    print("Assertions passed for perfect linear data!")
