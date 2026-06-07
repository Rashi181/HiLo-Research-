import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
# Load the data
df = pd.read_csv("Week_1/dataset_x_squared.csv")

x = df["x"].values
y = df["y"].values

# Split into train / test 
rng = np.random.default_rng(30)
idx = rng.permutation(1000)

X_train = x[idx[:300]].reshape(-1, 1) 
y_train = y[idx[:300]]
X_test  = x[idx[300:1000]].reshape(-1,1)
y_test  = y[idx[300:1000]]


#sklearn GPR 
gpr = GaussianProcessRegressor(kernel=RBF(), alpha=1e-8)
gpr.fit(X_train, y_train)

pred_test = gpr.predict(X_test)
mse_test = np.mean((y_test - pred_test) ** 2)


#GPR from scratch 

# Creating the RBF kernel
def my_RBF_kernel(A, B, l=1.0):
    # squared distance between every row of A and every row of B
    diff = A[:, None, :] - B[None, :, :]   # shape (n, m, d)
    dist2 = np.sum(diff ** 2, axis=-1)      # shape (n, m)
    return np.exp(-dist2 / (2 * l ** 2))

l = gpr.kernel_.length_scale   # reuse the length-scale sklearn found


def gp_predict(X_train, y_train, X_star, l=1.0, noise=1e-8):
    # Training kernel
    K = my_RBF_kernel(X_train, X_train, l)
    K += noise * np.eye(len(X_train))

    # K^{-1}
    K_inv = np.linalg.inv(K)

    # alpha = K^{-1} y
    alpha = K_inv @ y_train

    # Cross-kernel
    K_star = my_RBF_kernel(X_star, X_train, l)

    # Test kernel
    K_starstar = my_RBF_kernel(X_star, X_star, l)

    # Predictive mean
    mean = K_star @ alpha

    # Predictive covariance
    cov = K_starstar - K_star @ K_inv @ K_star.T

    # Predictive std
    std = np.sqrt(np.clip(np.diag(cov), 0, None))

    return mean, std

#Test 
scratch_pred, scratch_std = gp_predict(X_train, y_train, X_test, l=1.0, noise=1e-8)
mse_test_sc = np.mean((y_test - scratch_pred) ** 2)


# Validate both agree 
print(f"[sklearn]  Test MSE: {mse_test:.2e}")
print(f"[scratch]  Test MSE: {mse_test_sc:.2e}")
print(f"Absolute difference: {abs(mse_test - mse_test_sc):.2e}")

