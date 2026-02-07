import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_non_linear_loss_image():
    """
    Generates and saves a 3D plot of a non-convex Sum of Squared Errors (SSE)
    loss surface for a model that is non-linear in its parameters.
    """
    # 1. Define a non-linear model and generate sample data
    # Model: y = a * sin(b*x) + noise
    np.random.seed(42)
    x_data = np.linspace(-np.pi, np.pi, 50)
    true_a, true_b = 1.5, 2.5
    y_data = true_a * np.sin(true_b * x_data) + np.random.normal(0, 0.2, size=x_data.shape)

    # 2. Define the SSE function for this model
    def calculate_sse(a, b):
        y_pred = a * np.sin(b * x_data)
        return np.sum((y_data - y_pred)**2)

    # 3. Create a grid of parameter values (a, b) to explore
    a_range = np.linspace(0.5, 3.5, 100)
    b_range = np.linspace(0.5, 4.5, 100)
    A, B = np.meshgrid(a_range, b_range)

    # 4. Calculate SSE for each point in the grid
    SSE = np.array([calculate_sse(a, b) for a, b in zip(np.ravel(A), np.ravel(B))])
    SSE = SSE.reshape(A.shape)

    # 5. Plot the 3D loss surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface, using a log scale for Z to better visualize minima
    ax.plot_surface(A, B, np.log(SSE + 1), cmap='viridis', alpha=0.9)

    # Mark the true parameter values
    ax.scatter(true_a, true_b, np.log(calculate_sse(true_a, true_b) + 1),
               color='red', marker='o', s=200, label='True Parameters', depthshade=True)

    ax.set_xlabel('Parameter "a"')
    ax.set_ylabel('Parameter "b"')
    ax.set_zlabel('log(SSE + 1)')
    ax.set_title('Non-Convex Loss Surface for a Non-Linear Model (y = a*sin(b*x))')
    ax.view_init(elev=30, azim=45) # Adjust viewing angle
    ax.legend()

    # 6. Save the plot to a file
    output_path = 'regression/linear_regression/analytical/non_linear_loss_surface.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    generate_non_linear_loss_image()
