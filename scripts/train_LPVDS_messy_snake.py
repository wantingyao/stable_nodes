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


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_demos(path):
    """Load 2D_messy-snake.mat -> list of (pos, vel) trajectories.

    The .mat file stores Data as a 4 x N concatenation of all demos:
        rows 0..1 = position, rows 2..3 = velocity.
    Demos are separated by a trailing zero column (attractor padding).
    """
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
    """Concatenate every (pos, vel) pair column-wise."""
    X = np.hstack([p for p, _ in demos])
    Xd = np.hstack([v for _, v in demos])
    return X, Xd


# ---------------------------------------------------------------------------
# 2. GMM mixing functions
# ---------------------------------------------------------------------------
def fit_gmm(X, max_components=10, random_state=0):
    """Fit a Bayesian (Dirichlet-process style) GMM and prune empty components.

    Using BayesianGaussianMixture lets us pick K automatically:
    components whose effective weight is ~0 are dropped.
    """
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
    covs    = bgmm.covariances_[keep]           # (K, d, d)
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


# ---------------------------------------------------------------------------
# 3. Stable LPV-DS optimisation
# ---------------------------------------------------------------------------
# Joint optimisation over {A_k} AND P is bilinear (non-convex), so we follow
# the standard LPV-DS recipe: estimate P first from data, then solve a convex
# SDP for {A_k} with that P held fixed.
# ---------------------------------------------------------------------------
def estimate_P(X, attractor, alpha=1.0):
    """Data-driven Lyapunov matrix.

    Idea (Khansari-Zadeh & Billard '11; Figueroa & Billard '18):
    a quadratic V(x) = (x-x*)^T P (x-x*) is a Lyapunov function for the demos
    iff its time-derivative is negative along them. We pick P >> 0 that does
    a reasonable job of that. The simplest robust choice that always works
    on demos converging to x* is P = alpha * I — and we then *certify* it by
    checking sign(V_dot) on the demonstration samples after fitting.
    A slightly better choice rescales axes by demo extent.
    """
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

    P = estimate_P(X, attractor)
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


# ---------------------------------------------------------------------------
# 4. Forward simulation + plotting
# ---------------------------------------------------------------------------
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
def plot_results(demos, weights, means, covs, A, b, save_path):
    X_all, _ = stack(demos)
    pad = 0.5
    x_min, x_max = X_all[0].min() - pad, X_all[0].max() + pad
    y_min, y_max = X_all[1].min() - pad, X_all[1].max() + pad
 
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

# def plot_results(demos, weights, means, covs, A, b, save_path):
#     X_all, _ = stack(demos)
#     pad = 0.5
#     x_min, x_max = X_all[0].min() - pad, X_all[0].max() + pad
#     y_min, y_max = X_all[1].min() - pad, X_all[1].max() + pad

#     # Streamline grid
#     nx, ny = 60, 60
#     xs = np.linspace(x_min, x_max, nx)
#     ys = np.linspace(y_min, y_max, ny)
#     XX, YY = np.meshgrid(xs, ys)
#     grid = np.vstack([XX.ravel(), YY.ravel()])
#     V = lpvds_velocity(grid, weights, means, covs, A, b)
#     U = V[0].reshape(ny, nx)
#     W = V[1].reshape(ny, nx)
#     speed = np.sqrt(U ** 2 + W ** 2)

#     fig, ax = plt.subplots(figsize=(9, 7))
#     ax.streamplot(XX, YY, U, W, color=speed, cmap="viridis",
#                   density=1.6, linewidth=0.8, arrowsize=0.9)

#     # Demonstrations
#     colors = ["tab:red", "tab:orange", "tab:purple", "tab:brown"]
#     for i, (p, _) in enumerate(demos):
#         ax.plot(p[0], p[1], color=colors[i % len(colors)], lw=2.0,
#                 label=f"demo {i + 1}")
#         ax.plot(p[0, 0], p[1, 0], "o", color=colors[i % len(colors)], ms=7)

#     # Attractor + GMM centres
#     ax.plot(0, 0, "k*", ms=18, label="attractor")
#     ax.plot(means[:, 0], means[:, 1], "kx", ms=10, mew=2, label="GMM means")

#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_title("LPV-DS on 2D messy-snake")
#     ax.legend(loc="best", fontsize=9)
#     ax.set_aspect("equal")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=140)
#     plt.close(fig)
#     print(f"  saved figure -> {save_path}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)

    print("[1/4] Loading demonstrations ...")
    demos = load_demos("third_party/corl18/2D_messy-snake.mat")
    for i, (p, v) in enumerate(demos):
        print(f"   demo {i + 1}: {p.shape[1]} samples,"
              f" start=({p[0,0]:.2f},{p[1,0]:.2f}),"
              f" end=({p[0,-1]:.2f},{p[1,-1]:.2f})")
    X, Xd = stack(demos)
    print(f"   total: {X.shape[1]} (pos, vel) pairs")

    print("[2/4] Fitting GMM (Bayesian, auto-K) ...")
    weights, means, covs = fit_gmm(X, max_components=10)
    gamma = posterior(X, weights, means, covs)

    print("[3/4] Solving stable LPV-DS SDP ...")
    A, b, P = learn_lpvds(X, Xd, gamma, attractor=np.zeros(2), eps=1e-3)

    # Sanity-check eigenvalues of A_k^T P + P A_k
    for k in range(len(weights)):
        eigs = np.linalg.eigvalsh(A[k].T @ P + P @ A[k])
        print(f"   comp {k}: max eig(AᵀP+PA) = {eigs.max(): .3e}  (should be < 0)")

    print("[4/4] Plotting & saving ...")
    fig_path = os.path.join(out_dir, "lpvds_messy_snake.png")
    plot_results(demos, weights, means, covs, A, b, fig_path)

    model_path = os.path.join(out_dir, "lpvds_messy_snake.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            dict(weights=weights, means=means, covs=covs, A=A, b=b, P=P,
                 attractor=np.zeros(2)),
            f,
        )
    print(f"  saved model   -> {model_path}")


if __name__ == "__main__":
    main()