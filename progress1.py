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

X_train = x[idx[:750]].reshape(-1, 1) 
y_train = y[idx[:750]]
X_test  = x[idx[750:1000]].reshape(-1,1)
y_test  = y[idx[750:1000]]


#sklearn GPR 
gpr = GaussianProcessRegressor(kernel=RBF(), alpha=1e-8)
gpr.fit(X_train, y_train)

pred_test = gpr.predict(X_test)
mse_test = np.mean((y_test - pred_test) ** 2)


#GPR from scratch 

# RBF kernel to measure similarity between two sets of points
def my_RBF_kernel(A, B, l=1.0):
    # squared distance between every row of A and every row of B
    diff = A[:, None, :] - B[None, :, :]   # shape (n, m, d)
    dist2 = np.sum(diff ** 2, axis=-1)      # shape (n, m)
    return np.exp(-dist2 / (2 * l ** 2))

l = gpr.kernel_.length_scale   # reuse the length-scale sklearn found

# Build the kernel matrix on training data
K = my_RBF_kernel(X_train, X_train, l)
K += 1e-8 * np.eye(750)               # small jitter for stability

# alpha = K^{-1} y  
K_inv = np.linalg.inv(K)
alpha = K_inv @ y_train

# Predict: mean = K(X*, X) @ alpha
def predict(X_star):
    K_star =my_RBF_kernel(X_star, X_train, l)
    return K_star @ alpha

#Uncertainity: covariance:
# Kernel matrices needed

# K_* : kernel between test points and training points  
K_star  = my_RBF_kernel(X_test, X_train, l)

# K_** : kernel between test points and themselves 
K_starstar = my_RBF_kernel(X_test, X_test, l)

# Predictive mean 
pred_mean = K_star @ alpha                    

# --- Predictive covariance ---
# cov(f*) = K_** - K_*^T (K + σ²I)^{-1} K_*
# Note: K_star is (n_test, n_train), so K_star^T is (n_train, n_test)
# K_star @ K_inv @ K_star.T  gives (n_test, n_test)

cov_f_star = K_starstar - K_star @ K_inv @ K_star.T  

# --- Predictive std dev (one number per test point) ---
# diagonal of the covariance matrix = variance per point
var_f_star = np.diag(cov_f_star)                    # shape (n_test,)
std_f_star = np.sqrt(np.clip(var_f_star, 0, None))  # clip to avoid tiny negatives from floating point

pred_test_sc = predict(X_test)


mse_test_sc = np.mean((y_test - pred_test_sc) ** 2)



# 5. PART 3: Validate both agree 
print(f"\n[sklearn]  Test MSE: {mse_test:.2e}")
print(f"[scratch]  Test MSE: {mse_test_sc:.2e}")

difference = abs(mse_test - mse_test_sc)
print(f"\nAbsolute difference in MSE: {difference:.2e}")

