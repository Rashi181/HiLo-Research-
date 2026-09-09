import numpy as np
from itertools import combinations

np.random.seed(42)

def f(x):
    return x ** 2

eps = np.random.uniform(0, 1) #constant epsilon
print(f"eps = {eps:.6f}")

#creating the traning points
n = 10
X = np.random.uniform(-2, 2, n)       # x1 ... x10
Y = f(X) + eps                         # y1 ... y10

print(f"\nTraining data (x, y=x²+eps):")
for i in range(n):
    print(f"  x{i+1} = {X[i]:+.4f},  y{i+1} = {Y[i]:.4f}")

# difference dataset — all unique pairs (xi, xj), i < j 
# pairs: (x1,x2),(x1,x3),...,(x9,x10)  →  C(10,2) = 45 pairs
pair_x1  = []
pair_x2  = []
pair_ydiff = []

for i, j in combinations(range(n), 2):   # i < j, so i != j
    pair_x1.append(X[i])
    pair_x2.append(X[j])
    pair_ydiff.append(Y[i] - Y[j])        # eps cancels: (f(xi)+eps)-(f(xj)+eps)

pair_x1    = np.array(pair_x1)            # shape (45,)
pair_x2    = np.array(pair_x2)            # shape (45,)
pair_ydiff = np.array(pair_ydiff)         # shape (45,)  = f(xi)-f(xj)


# ── RBF kernel ────────────────────────────────────────────────────────────────
def my_RBF_kernel(A, B, l=1.0):
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    diff  = A[:, None, :] - B[None, :, :]
    dist2 = np.sum(diff ** 2, axis=-1)
    return np.exp(-dist2 / (2 * l ** 2))

# ── Vanilla GP ────────────────────────────────────────────────────────────────
def gp_predict(X_train, y_train, X_test, l=1.0, noise=1e-6):
    K      = my_RBF_kernel(X_train, X_train, l) + noise * np.eye(len(X_train))
    K_inv  = np.linalg.inv(K)
    alpha  = K_inv @ y_train
    K_star = my_RBF_kernel(X_test, X_train, l)
    mean   = K_star @ alpha
    return mean

# ── Difference kernel OC.2 ────────────────────────────────────────────────────
def paired_kernel_scalar(x, xp, xt, xtp, l=1.0):
    """Ǩ(x|x′, x̃|x̃′) = K(x,x̃) + K(x′,x̃′) - K(x,x̃′) - K(x̃,x′)"""
    def K(a, b):
        return my_RBF_kernel(np.array([[a]]), np.array([[b]]), l)[0, 0]
    return K(x, xt) + K(xp, xtp) - K(x, xtp) - K(xt, xp)

# ── Difference GP ─────────────────────────────────────────────────────────────
def diff_gp_predict(tr_x1, tr_x2, tr_ydiff,
                    te_x1, te_x2, l=1.0, noise=1e-6):
    n_tr = len(tr_x1)
    n_te = len(te_x1)


    K_train = np.zeros((n_tr, n_tr))
    for i in range(n_tr):
        for j in range(n_tr):
            K_train[i, j] = paired_kernel_scalar(
                tr_x1[i], tr_x2[i], tr_x1[j], tr_x2[j], l)
    K_train += noise * np.eye(n_tr)
    K_inv = np.linalg.inv(K_train)
    alpha = K_inv @ tr_ydiff


    K_cross = np.zeros((n_te, n_tr))
    for i in range(n_te):
        for j in range(n_tr):
            K_cross[i, j] = paired_kernel_scalar(
                te_x1[i], te_x2[i], tr_x1[j], tr_x2[j], l)

    mean = K_cross @ alpha
    return mean

# ── Test data: 5 clean points, no eps ────────────────────────────────────────
n_test       = 5
X_test       = np.random.uniform(-2, 2, n_test)
y_test_clean = f(X_test)                          # ground truth: x^2, no eps

# ── Step 5: Vanilla GP — train on (X, Y=x²+eps), predict f(x) ───────────────
vanilla_pred = gp_predict(
    X.reshape(-1, 1), Y,
    X_test.reshape(-1, 1)
)
vanilla_mse = np.mean((vanilla_pred - y_test_clean) ** 2)

# ── Step 6: Difference GP — train on 45 pairs, predict f(xi)-f(xj) ───────────
# Test pairs: (x_test_i, x_ref) where x_ref = mean of training X
# Then recover f(x_test) = predicted_diff + f(x_ref)
x_ref  = np.mean(X)
f_ref  = f(x_ref)

te_x1  = X_test
te_x2  = np.full(n_test, x_ref)

diff_pred_raw = diff_gp_predict(pair_x1, pair_x2, pair_ydiff, te_x1, te_x2)
diff_pred     = diff_pred_raw + f_ref              # recover absolute prediction
diff_mse      = np.mean((diff_pred - y_test_clean) ** 2)

# ── Step 7: Report MSEs ───────────────────────────────────────────────────────
print(f"\n── Test points ──")
for i in range(n_test):
    print(f"  x={X_test[i]:+.4f}  true f(x)={y_test_clean[i]:.4f}  "
          f"vanilla={vanilla_pred[i]:.4f}  diff_gp={diff_pred[i]:.4f}")

print(f"\n══ MSE Results ══")
print(f"  eps (shared noise)   = {eps:.6f}")
print(f"  Vanilla GP MSE       = {vanilla_mse:.6f}")
print(f"  Difference GP MSE    = {diff_mse:.6f}")
print()
if diff_mse < vanilla_mse:
    print("  → Difference GP wins: shared eps cancelled in the difference targets.")
else:
    print("  → Vanilla GP wins on this run.")
