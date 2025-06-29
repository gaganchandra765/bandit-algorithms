import numpy as np
import matplotlib.pyplot as plt

# Fix random seed for reproducibility
np.random.seed(0)

# Create rank-deficient design matrix (e.g., all points lie on a line)
A = np.array([
    [1, 2],
    [2, 4],
    [3, 6]
])  # Rank 1, spans only a line in R^2

r = np.array([1, 2, 3])  # Rewards / targets

# Grid of theta values for contour plot
theta1 = np.linspace(-2, 2, 100)
theta2 = np.linspace(-2, 2, 100)
T1, T2 = np.meshgrid(theta1, theta2)
J_unreg = np.zeros_like(T1)
J_reg = np.zeros_like(T1)

# Evaluate cost at each grid point
for i in range(T1.shape[0]):
    for j in range(T1.shape[1]):
        theta = np.array([T1[i, j], T2[i, j]])
        residual = A @ theta - r
        J_unreg[i, j] = np.sum(residual**2)
        J_reg[i, j] = J_unreg[i, j] + 5.0 * np.sum(theta**2)  # L2 penalty λ=1.0

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Unregularized loss: flat valley along null space
axs[0].contour(T1, T2, J_unreg, levels=30)
axs[0].set_title("Unregularized Cost (Rank-deficient)")
axs[0].set_xlabel(r"$\theta_1$")
axs[0].set_ylabel(r"$\theta_2$")
axs[0].axhline(0, color='gray', linestyle='--', lw=0.5)
axs[0].axvline(0, color='gray', linestyle='--', lw=0.5)

# Regularized loss: unique minimizer
axs[1].contour(T1, T2, J_reg, levels=30)
axs[1].set_title("Ridge-Regularized Cost (λ=1)")
axs[1].set_xlabel(r"$\theta_1$")
axs[1].set_ylabel(r"$\theta_2$")
axs[1].axhline(0, color='gray', linestyle='--', lw=0.5)
axs[1].axvline(0, color='gray', linestyle='--', lw=0.5)

plt.tight_layout()
plt.savefig('L2-regularization-low-rank.png')
plt.show()
