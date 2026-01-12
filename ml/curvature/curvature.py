import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def construct_matrix_from_eigenvalues(lambda1, lambda2, theta):
    """
    Construct a 2x2 matrix from eigenvalues and rotation angle.
    Matrix = R * D * R^T where:
    - D is diagonal matrix with eigenvalues
    - R is rotation matrix with angle theta
    """
    # Rotation matrix
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    # Diagonal matrix of eigenvalues
    D = np.array([[lambda1, 0], [0, lambda2]])

    # Construct symmetric matrix
    matrix = R @ D @ R.T
    return matrix

def interactive_curvature():
    # Initial parameters
    init_lambda1 = 2.0
    init_lambda2 = 0.5
    init_theta = 0.0

    # Create the figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Adjust the main plot to make room for sliders
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Create grid points
    x = np.linspace(-1, 1, 30)
    y = np.linspace(-1, 1, 30)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()])

    # Function to update the plot
    def update(val):
        ax.clear()

        # Get current slider values
        lambda1 = slider_lambda1.val
        lambda2 = slider_lambda2.val
        theta = slider_theta.val

        # Construct matrix from eigenvalues
        matrix = construct_matrix_from_eigenvalues(lambda1, lambda2, theta)

        # Apply transformation
        transformed_points = matrix @ points
        X_t = transformed_points[0, :].reshape(30, 30)
        Y_t = transformed_points[1, :].reshape(30, 30)

        # Calculate surface height
        Z = X_t**2 + Y_t**2

        # Plot surface
        surf = ax.plot_surface(X_t, Y_t, Z, cmap='viridis', alpha=0.8, edgecolor='none')

        # Update title with current matrix
        ax.set_title(f'Surface Curvature - Interactive Eigenvalue Control\n'
                    f'λ₁={lambda1:.2f}, λ₂={lambda2:.2f}, θ={np.degrees(theta):.1f}°\n'
                    f'Matrix: [[{matrix[0,0]:.2f}, {matrix[0,1]:.2f}], '
                    f'[{matrix[1,0]:.2f}, {matrix[1,1]:.2f}]]')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Height (Z)')

        # Adjust z-limits for better viewing
        ax.set_zlim(0, np.max(Z))

        fig.canvas.draw_idle()

    # Create slider axes
    ax_lambda1 = plt.axes([0.2, 0.15, 0.65, 0.03])
    ax_lambda2 = plt.axes([0.2, 0.10, 0.65, 0.03])
    ax_theta = plt.axes([0.2, 0.05, 0.65, 0.03])

    # Create sliders
    slider_lambda1 = Slider(ax_lambda1, 'Eigenvalue λ₁', 0.1, 5.0,
                            valinit=init_lambda1, valstep=0.1)
    slider_lambda2 = Slider(ax_lambda2, 'Eigenvalue λ₂', 0.1, 5.0,
                            valinit=init_lambda2, valstep=0.1)
    slider_theta = Slider(ax_theta, 'Rotation θ (rad)', -np.pi, np.pi,
                         valinit=init_theta, valstep=0.1)

    # Connect sliders to update function
    slider_lambda1.on_changed(update)
    slider_lambda2.on_changed(update)
    slider_theta.on_changed(update)

    # Initial plot
    update(None)

    plt.show()

def plot_transformed_curvature(matrix):
    """Original static visualization function (kept for backward compatibility)"""
    # 1. Create a flat grid of points (x, y)
    x = np.linspace(-1, 1, 30)
    y = np.linspace(-1, 1, 30)
    X, Y = np.meshgrid(x, y)

    # Flatten grid for matrix multiplication
    points = np.vstack([X.ravel(), Y.ravel()])

    # 2. Apply the 2x2 Transformation Matrix
    # This matrix changes the "principal directions" of the surface
    transformed_points = matrix @ points

    X_t = transformed_points[0, :].reshape(30, 30)
    Y_t = transformed_points[1, :].reshape(30, 30)

    # 3. Calculate Z (the surface height)
    # We use a standard bowl shape: z = x^2 + y^2
    Z = X_t**2 + Y_t**2

    # 4. Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X_t, Y_t, Z, cmap='viridis', alpha=0.8, edgecolor='none')

    ax.set_title(f'Surface Curvature after Matrix Transformation\nMatrix: {matrix.tolist()}')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Height (Z)')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()

# --- Run the interactive visualization ---
if __name__ == "__main__":
    interactive_curvature()

    # Original examples (uncomment to use static version):
    # plot_transformed_curvature(np.array([[1, 0], [0, 1]]))  # Identity
    # plot_transformed_curvature(np.array([[2.5, 0], [0, 0.5]]))  # Stretch
    # plot_transformed_curvature(np.array([[1, 0.5], [0.5, 1]]))  # Shear