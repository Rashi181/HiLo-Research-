import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
# Load the data
df = pd.read_csv("dataset_x_squared.csv")

x = df["x"].values
y = df["y"].values

# Split into train / test / val 
rng = np.random.default_rng(30)
idx = rng.permutation(1000)

X_train = x[idx[:700]].reshape(-1, 1) 
y_train = y[idx[:700]]
X_test  = x[idx[700:900]].reshape(-1,1)
y_test  = y[idx[700:900]]
X_val   = x[idx[900:]].reshape(-1, 1)
y_val   = y[idx[900:]]


#sklearn GPR 
gpr = GaussianProcessRegressor(kernel=RBF(), alpha=1e-8)
gpr.fit(X_train, y_train)

pred_test = gpr.predict(X_test)
pred_val  = gpr.predict(X_val)

mse_test = np.mean((y_test - pred_test) ** 2)
mse_val  = np.mean((y_val  - pred_val ) ** 2)

print(f"[sklearn]  Test MSE: {mse_test:.2e}   Val MSE: {mse_val:.2e}")

#GPR from scratch 

# RBF kernel to measure similarity between two sets of points
def rbf_kernel(A, B, l=1.0):
    # squared distance between every row of A and every row of B
    diff = A[:, None, :] - B[None, :, :]   # shape (n, m, d)
    dist2 = np.sum(diff ** 2, axis=-1)      # shape (n, m)
    return np.exp(-dist2 / (2 * l ** 2))

l = gpr.kernel_.length_scale   # reuse the length-scale sklearn found

# Build the kernel matrix on training data
K = rbf_kernel(X_train, X_train, l)
K += 1e-8 * np.eye(700)               # small jitter for stability

# Precompute alpha = K^{-1} y  (via Cholesky for stability)
L     = np.linalg.cholesky(K)
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

# Predict: mean = K(X*, X) @ alpha
def predict(X_star):
    K_star = rbf_kernel(X_star, X_train, l)
    return K_star @ alpha

pred_test_sc = predict(X_test)
pred_val_sc  = predict(X_val)

mse_test_sc = np.mean((y_test - pred_test_sc) ** 2)
mse_val_sc  = np.mean((y_val  - pred_val_sc ) ** 2)

print(f"[scratch]  Test MSE: {mse_test_sc:.2e}   Val MSE: {mse_val_sc:.2e}")

# ── 5. PART 3: Validate both agree ───────────────────────────────
max_diff = np.max(np.abs(pred_test - pred_test_sc))
print(f"\nMax difference between sklearn and scratch: {max_diff:.2e}")
