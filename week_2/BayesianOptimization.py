import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Week_1.progress1 import my_RBF_kernel, gp_predict

#To find minimum Y = x**2 + eps in the range [-2, 2] using Bayesian Optimization

def f(x):
    eps = np.random.normal(0, 0.1)
    return x**2 + eps

# Initial training data (only 5 random samples)
np.random.seed(42)
n_init = 5
X = np.random.uniform(-2, 2, size=(n_init, 1))
y = np.array([f(x[0]) for x in X])

# Create DataFrame
df = pd.DataFrame({
    'x': X.flatten(),
    'y': y
}) 


# Build a Gaussian process model using current data points


#implement the GCP-UB acquisition function
def beta_t(t, delta=0.1):
    beta = 2 * np.log((t**2 * np.pi**2) / (3 * delta))
    return beta

def gp_ucb(mean, std, beta):      
    return mean + beta * std



def propose_candidate(X_train, y_train, t, bounds=(-2, 2), n_candidates=1000):
    # 1. Sample random candidate points
    X_cand = np.random.uniform(bounds[0], bounds[1], size=(n_candidates, 1))

    # 2. GP predicts mean and std at every candidate
    mean, std = gp_predict(X_train, y_train, X_cand) 

    # 3. Evaluate acquisition at every candidate
    acq = gp_ucb(mean, std, beta=beta_t(t))

    # 4. x* = argmax a(x)
    x_star = X_cand[np.argmax(acq)]

    return x_star, X_cand, acq

n_iter = 15

for t in range(1, n_iter + 1):
    x_star, X_cand, acq = propose_candidate(X, y, t)

    y_star = f(x_star[0])                      # evaluate true function at x*

    X = np.vstack([X, x_star])                 # add x* to dataset
    y = np.append(y, y_star)                   # add y* to dataset

    best_idx = np.argmin(y)
    print(f"Iter {t:02d} | x* = {x_star[0]:+.4f} | y* = {y_star:.4f} | "
          f"best x = {X[best_idx][0]:+.4f} | best y = {y[best_idx]:.4f}")



best_idx = np.argmin(y)
print(f"\nMinimum found: x = {X[best_idx][0]:+.4f}, f(x) = {y[best_idx]:.4f}")