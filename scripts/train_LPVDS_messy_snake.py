"""
LPV-DS (Linear Parameter Varying Dynamical System) learning
on the 2D messy-snake demonstration dataset.

Pipeline (Figueroa & Billard, CoRL 2018):
  1. Load demos and split into trajectories.
  2. Fit a GMM over positions -> mixing functions gamma_k(x).
  3. For each component k, solve a *jointly-constrained* SDP that finds
     {A_k} and a common Lyapunov matrix P such that
        A_k^T P + P A_k <= -eps * I        (global asymptotic stability)
     while minimising reconstruction error of the velocity field.
  4. Visualise streamlines + demo overlay and save the model.

Dependencies: numpy, scipy, scikit-learn, cvxpy, matplotlib
"""

import numpy as np
import scipy.io as sio
from scipy.linalg import solve_continuous_lyapunov
from sklearn.mixture import BayesianGaussianMixture
import cvxpy as cp
import matplotlib.pyplot as plt
import pickle
import os

# Data Loading
def load_demos(path):
    raw = sio.loadmat(path)["Data"]              # (4, N)
    pos_all, vel_all = raw[:2, :], raw[2:, :]

    # Heuristic split: large jumps in position mark demo boundaries.
    diffs = np.linalg.norm(np.diff(pos_all, axis=1), axis=0)
    breaks = np.where(diffs > 50 * np.median(diffs))[0]
    starts = np.r_[0, breaks + 1]
    ends   = np.r_[breaks + 1, raw.shape[1]]

    demos = []
    for s, e in zip(starts, ends):
        # Drop the final zero "attractor" sample if it is exactly zero,
        # otherwise the velocity dummy would bias the fit.
        if np.allclose(raw[:, e - 1], 0):
            e -= 1
        demos.append((pos_all[:, s:e], vel_all[:, s:e]))
    return demos


def stack(demos):
    X = np.hstack([p for p, _ in demos])
    Xd = np.hstack([v for _, v in demos])
    return X, Xd


def preprocess(demos):
    """Three-step preprocessing:
      1. Min-max normalize positions (and velocities) to [-1, 1].
      2. Shift attractor (mean of demo endpoints) to the origin.
      3. Re-normalize to [-1, 1] by dividing by the per-axis max absolute
         value — scale-only, no shift, so the attractor stays at the origin.

    Returns preprocessed demos; attractor is np.zeros(d) by construction.
    """
    # Step 1: global min-max normalise to [-1, 1]
    all_pos = np.hstack([p for p, _ in demos])
    p_min = all_pos.min()
    p_max = all_pos.max()
    s1 = 2.0 / (p_max - p_min)

    demos1 = [((p - p_min) * s1 - 1.0, v * s1) for p, v in demos]

    # Step 2: shift attractor (mean of endpoints) to origin
    att1 = np.mean(
        np.hstack([p[:, -1:] for p, _ in demos1]), axis=1, keepdims=True
    )
    demos2 = [(p - att1, v) for p, v in demos1]

    # Step 3: re-normalise to [-1, 1] via per-axis max-abs scaling (no shift)
    # This keeps the attractor at the origin.
    all_pos2 = np.hstack([p for p, _ in demos2])
    s2 = 1.0 / np.abs(all_pos2).max()  

    demos3 = [(p * s2, v * s2) for p, v in demos2]

    return demos3, np.zeros(all_pos.shape[0])


 # GMM mixing functions
def fit_gmm(X, max_components=10, random_state=0):

    bgmm = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1.0 / max_components,
        max_iter=500,
        random_state=random_state,
    ).fit(X.T)

    keep = bgmm.weights_ > 1e-2
    K = int(keep.sum())
    weights = bgmm.weights_[keep] / bgmm.weights_[keep].sum()
    means   = bgmm.means_[keep]                 # (K, d)
    covs    = bgmm.covariances_[keep] * 4.0 # multiply cov inflate           # (K, d, d)
    print(f"  -> kept {K} active components (out of {max_components})")
    return weights, means, covs


def posterior(X, weights, means, covs):
    """gamma_k(x_i) - shape (K, N), each column sums to 1."""
    from scipy.stats import multivariate_normal
    K, N = len(weights), X.shape[1]
    logp = np.empty((K, N))
    for k in range(K):
        logp[k] = np.log(weights[k] + 1e-300) + multivariate_normal.logpdf(
            X.T, mean=means[k], cov=covs[k], allow_singular=True
        )
    logp -= logp.max(axis=0, keepdims=True)      # stabilise
    p = np.exp(logp)
    return p / p.sum(axis=0, keepdims=True)

# Stable LPV-DS optimisation
def estimate_P(X, attractor, alpha=1.0):
    Xc = X - attractor[:, None]
    # Use inverse covariance scaled to be well-conditioned (whitening style).
    cov = np.cov(Xc) + 1e-3 * np.eye(Xc.shape[0])
    P = np.linalg.inv(cov)
    P = 0.5 * (P + P.T)
    # Normalise so largest eigenvalue == alpha (keeps SDP well-scaled).
    P = alpha * P / np.linalg.eigvalsh(P).max()
    return P


def learn_lpvds(X, Xd, gamma, attractor=None, eps=1e-3):
    """Two-stage LPV-DS:

        Stage 1: pick a positive-definite P from data.
        Stage 2: solve, with P fixed,

            min  sum_i || Xd_i - sum_k gamma_k(i) * A_k * (X_i - x*) ||^2
            s.t. A_k^T P + P A_k <<= -eps * I    for every k.

    This second problem is a convex SDP (linear in the A_k).
    """
    d, N = X.shape
    K = gamma.shape[0]
    if attractor is None:
        attractor = np.zeros(d)
    Xc = X - attractor[:, None]

    # P = estimate_P(X, attractor)
    P = np.eye(d)
    print(f"   P eigenvalues: {np.linalg.eigvalsh(P)}")

    Ak = [cp.Variable((d, d)) for _ in range(K)]

    pred = 0
    for k in range(K):
        pred = pred + cp.multiply(gamma[k][None, :], Ak[k] @ Xc)

    obj = cp.Minimize(cp.sum_squares(pred - Xd) / N)

    cons = []
    I = np.eye(d)
    for k in range(K):
        # A_k^T P + P A_k is now linear in A_k (P is a constant).
        cons.append(Ak[k].T @ P + P @ Ak[k] + eps * I << 0)

    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, verbose=False)
    print(f"   SDP status: {prob.status},  fit MSE = {prob.value:.4e}")

    A_val = np.stack([A.value for A in Ak], axis=0)
    b_val = np.stack([-A_val[k] @ attractor for k in range(K)], axis=0)
    return A_val, b_val, P

# Forward Simulation and Plotting
def lpvds_velocity(x, weights, means, covs, A, b):
    """f(x) = sum_k gamma_k(x) * (A_k x + b_k).  x: (d,) or (d, N)."""
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False
    g = posterior(x, weights, means, covs)       # (K, N)
    out = np.zeros_like(x)
    for k in range(len(weights)):
        out += g[k][None, :] * (A[k] @ x + b[k][:, None])
    return out[:, 0] if squeeze else out


def simulate(x0, weights, means, covs, A, b, dt=0.01, T=2000, tol=1e-3):
    traj = [x0.copy()]
    x = x0.copy()
    for _ in range(T):
        v = lpvds_velocity(x, weights, means, covs, A, b)
        x = x + dt * v
        traj.append(x.copy())
        if np.linalg.norm(v) < tol:
            break
    return np.array(traj).T


def dtw_distance(P, Q):
    """DTW distance between trajectories P (d, T1) and Q (d, T2)."""
    n, m = P.shape[1], Q.shape[1]
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(P[:, i - 1] - Q[:, j - 1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return dtw[n, m]


def compute_metrics(demos, weights, means, covs, A, b):
    """Compute per-demo DTWD and overall velocity RMSE."""
    # Velocity RMSE over all demo data
    X_all = np.hstack([p for p, _ in demos])
    Xd_all = np.hstack([v for _, v in demos])
    Xd_pred = lpvds_velocity(X_all, weights, means, covs, A, b)
    vel_rmse = np.sqrt(np.mean(np.sum((Xd_pred - Xd_all) ** 2, axis=0)))

    # DTWD: reproduced trajectory vs. demonstration
    dtwds = []
    for i, (p, _) in enumerate(demos):
        repro = simulate(p[:, 0].copy(), weights, means, covs, A, b,
                         dt=0.01, T=4000, tol=1e-3)
        d = dtw_distance(p, repro)
        dtwds.append(d)
        print(f"   demo {i+1}: DTWD = {d:.4f}")

    mean_dtwd = float(np.mean(dtwds))
    print(f"   Mean DTWD     = {mean_dtwd:.4f}")
    print(f"   Velocity RMSE = {vel_rmse:.4f}")
    return mean_dtwd, vel_rmse


def plot_results(demos, weights, means, covs, A, b, save_path):
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
 
    # Streamline grid
    nx, ny = 60, 60
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.vstack([XX.ravel(), YY.ravel()])
    V = lpvds_velocity(grid, weights, means, covs, A, b)
    U = V[0].reshape(ny, nx)
    W = V[1].reshape(ny, nx)
    speed = np.sqrt(U ** 2 + W ** 2)
 
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.streamplot(XX, YY, U, W, color=speed, cmap="viridis",
                  density=1.6, linewidth=0.8, arrowsize=0.9)
 
    # Demonstrations (originals)
    demo_colors = ["tab:red", "tab:orange", "tab:purple", "tab:brown"]
    for i, (p, _) in enumerate(demos):
        c = demo_colors[i % len(demo_colors)]
        ax.plot(p[0], p[1], color=c, lw=2.2, alpha=0.55,
                label=f"demo {i + 1}")
        ax.plot(p[0, 0], p[1, 0], "o", color=c, ms=8, mec="k", mew=0.5)
 
    # ---- Reproductions: integrate the learned DS from each demo's start ----
    print("   simulating reproductions from demo start points ...")
    for i, (p, _) in enumerate(demos):
        x0 = p[:, 0].copy()
        traj = simulate(x0, weights, means, covs, A, b,
                        dt=0.01, T=4000, tol=1e-3)
        label = "DS reproduction" if i == 0 else None
        ax.plot(traj[0], traj[1], "k--", lw=1.6, label=label)
        end = traj[:, -1]
        print(f"   demo {i+1}: reproduction ended at "
              f"({end[0]:+.3f}, {end[1]:+.3f}) in {traj.shape[1]} steps")
 
    # Attractor + GMM centres
    ax.plot(0, 0, "k*", ms=18, label="attractor")
    ax.plot(means[:, 0], means[:, 1], "kx", ms=10, mew=2, label="GMM means")
 
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("LPV-DS on 2D messy-snake")
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"  saved figure -> {save_path}")

# Main
def main():
    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)

    print("[1/5] Loading demonstrations ...")
    demos_raw = load_demos("third_party/corl18/2D_messy-snake.mat")
    for i, (p, v) in enumerate(demos_raw):
        print(f"   demo {i + 1}: {p.shape[1]} samples,"
              f" start=({p[0,0]:.2f},{p[1,0]:.2f}),"
              f" end=({p[0,-1]:.2f},{p[1,-1]:.2f})")

    print("[2/5] Preprocessing (min-max -> shift attractor -> re-normalize) ...")
    demos, attractor = preprocess(demos_raw)
    X, Xd = stack(demos)
    print(f"   total: {X.shape[1]} (pos, vel) pairs")

    print("[3/5] Fitting GMM (Bayesian, auto-K) ...")
    weights, means, covs = fit_gmm(X, max_components=10)
    gamma = posterior(X, weights, means, covs)

    print("[4/5] Solving stable LPV-DS SDP ...")
    A, b, P = learn_lpvds(X, Xd, gamma, attractor=attractor, eps=1e-3)
    
    # Gamma sharpness
    entropy = (-gamma * np.log(gamma + 1e-12)).sum(axis=0).mean()
    print(f"gamma avg entropy: {entropy:.3f} / log(K)={np.log(len(weights)):.3f}")

    # Sanity-check eigenvalues of A_k^T P + P A_k
    for k in range(len(weights)):
        eigs = np.linalg.eigvalsh(A[k].T @ P + P @ A[k])
        print(f"   comp {k}: max eig(AᵀP+PA) = {eigs.max(): .3e}  (should be < 0)")

    print("[5/6] Computing metrics ...")
    compute_metrics(demos, weights, means, covs, A, b)

    print("[6/6] Plotting & saving ...")
    fig_path = os.path.join(out_dir, "lpvds_messy_snake.png")
    plot_results(demos, weights, means, covs, A, b, fig_path)

    model_path = os.path.join(out_dir, "lpvds_messy_snake.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            dict(weights=weights, means=means, covs=covs, A=A, b=b, P=P,
                 attractor=attractor),
            f,
        )
    print(f"  saved model   -> {model_path}")


if __name__ == "__main__":
    main()