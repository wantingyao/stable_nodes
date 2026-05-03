"""
SEDS (Stable Estimator of Dynamical Systems)
============================================

Reference:
  S.M. Khansari-Zadeh and A. Billard,
  "Learning Stable Nonlinear Dynamical Systems with Gaussian Mixture Models",
  IEEE Transactions on Robotics, 2011.

Pipeline:
  1. Load demos and split into trajectories.
  2. Fit a joint GMM over (ξ, ξ̇) to initialise parameters.
  3. Optimise the GMM parameters {π_k, μ_k, Σ_k} via constrained SQP:
       - objective: mean-squared velocity reconstruction error of the
         Gaussian Mixture Regression (GMR) prediction
                ξ̇ = Σ_k h_k(ξ) [ μ_k^ξ̇ + A_k (ξ - μ_k^ξ) ]
            where  A_k = Σ_k^{ξ̇ξ} (Σ_k^{ξξ})^{-1}.
       - constraints (Theorem 1 of the paper):
                A_k + A_k^T  ≺  0          (negative-definite symmetric part)
                b_k = -A_k ξ*   ⇔   μ_k^ξ̇ = A_k (ξ* - μ_k^ξ)
                Σ_k^{ξξ} ≻ 0                (covariance positive-definite)
                π_k > 0,  Σ_k π_k = 1
  4. Visualise streamlines + demo overlay and save the model.

Dependencies: numpy, scipy, scikit-learn, matplotlib
"""

import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pickle
import os


# ---------------------------------------------------------------------------
# 1. Data loading (identical to the LPV-DS baseline)
# ---------------------------------------------------------------------------
def load_demos(path):
    """Load 2D_messy-snake.mat -> list of (pos, vel) trajectories."""
    raw = sio.loadmat(path)["Data"]
    pos_all, vel_all = raw[:2, :], raw[2:, :]
    diffs = np.linalg.norm(np.diff(pos_all, axis=1), axis=0)
    breaks = np.where(diffs > 50 * np.median(diffs))[0]
    starts = np.r_[0, breaks + 1]
    ends   = np.r_[breaks + 1, raw.shape[1]]
    demos = []
    for s, e in zip(starts, ends):
        if np.allclose(raw[:, e - 1], 0):
            e -= 1
        demos.append((pos_all[:, s:e], vel_all[:, s:e]))
    return demos


def stack(demos):
    X = np.hstack([p for p, _ in demos])
    Xd = np.hstack([v for _, v in demos])
    return X, Xd


# ---------------------------------------------------------------------------
# 2. SEDS parameter packing / unpacking
# ---------------------------------------------------------------------------
# A SEDS model is parametrised by {π_k, μ_k, Σ_k} with
#     μ_k = [μ_k^ξ; μ_k^ξ̇] ∈ R^{2d}
#     Σ_k = [[Σ_k^ξξ,  Σ_k^ξξ̇];
#            [Σ_k^ξ̇ξ, Σ_k^ξ̇ξ̇]] ∈ R^{2d x 2d}, symmetric PD.
# To handle constraints automatically we work with:
#   - log π_k                        (then softmax to get π_k)
#   - μ_k^ξ                          (free)
#   - L_k^ξξ                         (lower-tri Cholesky of Σ_k^ξξ)
#   - Σ_k^ξξ̇                         (free; gives A_k via Σ_k^ξξ̇ (Σ_k^ξξ)^-1)
#   - L_k^cond                       (lower-tri Cholesky of conditional cov
#                                     Σ_k^ξ̇|ξ = Σ_k^ξ̇ξ̇ - Σ_k^ξ̇ξ Σ_k^ξξ⁻¹ Σ_k^ξξ̇)
# This factorisation guarantees Σ_k ≻ 0 by construction.
# μ_k^ξ̇ is *not* a free variable: it is fixed to A_k(ξ* - μ_k^ξ) so that
# b_k = -A_k ξ* and f(ξ*) = 0 (Theorem 1 condition).
# ---------------------------------------------------------------------------
def _vec_lower(L, diag_floor=1e-3):
    """Pack lower-triangular matrix into a flat vector.

    Diagonal stored as log(L_ii - diag_floor) so that the unpacked diag is
    exp(v) + diag_floor ≥ diag_floor. (Inverse of `_unvec_lower`.)
    """
    d = L.shape[0]
    out = []
    for i in range(d):
        for j in range(i + 1):
            if i == j:
                out.append(np.log(max(L[i, j] - diag_floor, 1e-12)))
            else:
                out.append(L[i, j])
    return np.array(out)


def _unvec_lower(v, d, diag_floor=1e-3):
    """Inverse of _vec_lower. Keep diagonal ≥ diag_floor for stability."""
    L = np.zeros((d, d))
    k = 0
    for i in range(d):
        for j in range(i + 1):
            if i == j:
                L[i, j] = np.exp(v[k]) + diag_floor
            else:
                L[i, j] = v[k]
            k += 1
    return L


def _n_lower(d):
    return d * (d + 1) // 2


def _pack(K, d, log_pi, mu_x, L_xx, S_xxd, L_cond):
    """Flatten parameters into a single 1-D vector."""
    parts = [log_pi]
    for k in range(K):
        parts.append(mu_x[k])                        # d
        parts.append(_vec_lower(L_xx[k]))            # d(d+1)/2
        parts.append(S_xxd[k].ravel())               # d*d
        parts.append(_vec_lower(L_cond[k]))          # d(d+1)/2
    return np.concatenate(parts)


def _unpack(theta, K, d):
    """Inverse of _pack."""
    nL = _n_lower(d)
    log_pi = theta[:K]
    off = K
    mu_x   = np.zeros((K, d))
    L_xx   = np.zeros((K, d, d))
    S_xxd  = np.zeros((K, d, d))
    L_cond = np.zeros((K, d, d))
    for k in range(K):
        mu_x[k]   = theta[off:off + d];          off += d
        L_xx[k]   = _unvec_lower(theta[off:off + nL], d); off += nL
        S_xxd[k]  = theta[off:off + d * d].reshape(d, d); off += d * d
        L_cond[k] = _unvec_lower(theta[off:off + nL], d); off += nL
    return log_pi, mu_x, L_xx, S_xxd, L_cond


# ---------------------------------------------------------------------------
# 3. SEDS forward pass: GMR with parameters expressed via the factorisation
# ---------------------------------------------------------------------------
def _model_to_lpv(log_pi, mu_x, L_xx, S_xxd, L_cond, attractor):
    """Convert the factorised parameters into per-component (A_k, b_k, μ_k, Σ_k^ξξ).

    Returns
    -------
    pi   : (K,)              mixing weights (softmax of log_pi)
    mu_x : (K, d)            position means
    Sxx  : (K, d, d)          position covariances (= L_xx L_xx^T)
    A    : (K, d, d)          A_k = S_xxd^T (Sxx)^{-1}      (NOTE the transpose:
                               Σ^ξξ̇ has shape (ξ, ξ̇), so Σ^ξ̇ξ = (Σ^ξξ̇)^T)
    b    : (K, d)             b_k = -A_k ξ*
    mu_xd: (K, d)             μ_k^ξ̇ = A_k (ξ* - μ_k^ξ)
    """
    K, d = mu_x.shape
    pi = np.exp(log_pi - log_pi.max())
    pi /= pi.sum()

    Sxx = np.einsum("kij,klj->kil", L_xx, L_xx)        # L L^T
    A = np.zeros((K, d, d))
    b = np.zeros((K, d))
    mu_xd = np.zeros((K, d))
    I = np.eye(d)
    for k in range(K):
        # A_k = Σ^ξ̇ξ Σ^ξξ⁻¹ with regularisation for numerical safety
        Sxx_k = Sxx[k] + 1e-6 * I
        A[k] = S_xxd[k].T @ np.linalg.solve(Sxx_k, I)
        # Theorem 1 condition for f(ξ*) = 0:
        #   ξ̇* = μ_k^ξ̇ + A_k (ξ* - μ_k^ξ) = 0  =>  μ_k^ξ̇ = A_k (μ_k^ξ - ξ*)
        mu_xd[k] = A[k] @ (mu_x[k] - attractor)
        b[k] = -A[k] @ attractor
    return pi, mu_x, Sxx, A, b, mu_xd


def _h_weights(X, pi, mu_x, Sxx):
    """Posterior responsibility h_k(ξ_i)  -  shape (K, M)."""
    K, d = mu_x.shape
    M = X.shape[1]
    log_h = np.empty((K, M))
    for k in range(K):
        diff = X - mu_x[k][:, None]
        sign, logdet = np.linalg.slogdet(Sxx[k])
        sol = np.linalg.solve(Sxx[k], diff)
        maha = np.einsum("ij,ij->j", diff, sol)
        log_h[k] = (np.log(pi[k] + 1e-300)
                    - 0.5 * (d * np.log(2 * np.pi) + logdet + maha))
    log_h -= log_h.max(axis=0, keepdims=True)
    h = np.exp(log_h)
    return h / h.sum(axis=0, keepdims=True)


def _gmr_predict(X, pi, mu_x, Sxx, A, mu_xd):
    """ξ̇(ξ) = Σ_k h_k(ξ) [ μ_k^ξ̇ + A_k (ξ - μ_k^ξ) ].

    Returns shape (d, M).
    """
    h = _h_weights(X, pi, mu_x, Sxx)
    K, d = mu_x.shape
    out = np.zeros((d, X.shape[1]))
    for k in range(K):
        out += h[k][None, :] * (mu_xd[k][:, None] + A[k] @ (X - mu_x[k][:, None]))
    return out


# ---------------------------------------------------------------------------
# 4. Initialisation: standard EM-GMM on the joint (ξ, ξ̇) space
# ---------------------------------------------------------------------------
def _project_to_stable(A, eps=0.1):
    """Project A so that A + A^T is negative-definite.

    Decompose A = (A+Aᵀ)/2 + (A-Aᵀ)/2 = S + W (symmetric + skew).
    Eigendecompose S = U Λ Uᵀ, clip Λ_i to ≤ -eps, rebuild S' = U Λ' Uᵀ.
    Return A' = S' + W. Guarantees A' + A'ᵀ = 2 S' ≺ 0.
    """
    S = 0.5 * (A + A.T)
    W = 0.5 * (A - A.T)
    evals, evecs = np.linalg.eigh(S)
    evals = np.minimum(evals, -eps)
    S_clipped = evecs @ np.diag(evals) @ evecs.T
    return S_clipped + W


def _init_from_em(X, Xd, K, attractor, seed=0):
    d = X.shape[0]
    Z = np.vstack([X, Xd]).T                           # (M, 2d)
    gmm = GaussianMixture(n_components=K, covariance_type="full",
                          random_state=seed, max_iter=300).fit(Z)
    log_pi = np.log(gmm.weights_ + 1e-12)
    mu_x   = gmm.means_[:, :d]
    L_xx   = np.zeros((K, d, d))
    S_xxd  = np.zeros((K, d, d))
    L_cond = np.zeros((K, d, d))
    for k in range(K):
        S = gmm.covariances_[k]
        Sxx_k  = S[:d, :d]    + 1e-3 * np.eye(d)
        Sxxd_k = S[:d, d:]                              # ξ↔ξ̇ block
        Sxdxd  = S[d:, d:]    + 1e-3 * np.eye(d)

        # Project so that A_k = Sxxd_k^T Sxx_k^-1 is "stable" (sym part neg def)
        A_init = Sxxd_k.T @ np.linalg.solve(Sxx_k, np.eye(d))
        A_proj = _project_to_stable(A_init, eps=0.1)
        Sxxd_k_proj = (A_proj @ Sxx_k).T               # = Σ^ξξ̇  (Σ^ξ̇ξ = A Σ^ξξ)

        # conditional covariance (recompute with projected ξ̇ block consistent)
        Scond = Sxdxd - Sxxd_k_proj.T @ np.linalg.solve(Sxx_k, Sxxd_k_proj)
        Scond = 0.5 * (Scond + Scond.T) + 1e-3 * np.eye(d)
        # If still not PD, regularise more
        eig_min = np.linalg.eigvalsh(Scond).min()
        if eig_min < 1e-3:
            Scond = Scond + (1e-3 - eig_min) * np.eye(d)

        L_xx[k]   = np.linalg.cholesky(Sxx_k)
        S_xxd[k]  = Sxxd_k_proj
        L_cond[k] = np.linalg.cholesky(Scond)
    return log_pi, mu_x, L_xx, S_xxd, L_cond


# ---------------------------------------------------------------------------
# 5. Main SEDS optimisation (constrained SQP)
# ---------------------------------------------------------------------------
def fit_seds(X, Xd, K, attractor=None, max_iter=200, seed=0, verbose=True):
    """Fit a SEDS model via constrained nonlinear optimisation.

    Returns a dict with the LPV form (pi, mu_x, Sxx, A, b, mu_xd, attractor).
    """
    d = X.shape[0]
    if attractor is None:
        attractor = np.zeros(d)

    # 1) Initialise from a vanilla joint-space EM-GMM
    log_pi0, mu_x0, L_xx0, S_xxd0, L_cond0 = _init_from_em(X, Xd, K, attractor, seed)
    theta0 = _pack(K, d, log_pi0, mu_x0, L_xx0, S_xxd0, L_cond0)

    # 2) Objective: mean-squared GMR prediction error
    def objective(theta):
        log_pi, mu_x, L_xx, S_xxd, L_cond = _unpack(theta, K, d)
        pi, mu_x, Sxx, A, b, mu_xd = _model_to_lpv(
            log_pi, mu_x, L_xx, S_xxd, L_cond, attractor)
        pred = _gmr_predict(X, pi, mu_x, Sxx, A, mu_xd)
        return float(np.mean((pred - Xd) ** 2))

    # 3) Stability constraint: A_k + A_k^T ≺ 0 for every k (Theorem 1).
    # We enforce this through *smooth* polynomial inequalities, which SLSQP
    # handles much better than eigenvalue-max:
    #   2D case:  trace(A+Aᵀ) < 0  AND  det(A+Aᵀ) > 0
    #             (a 2x2 symmetric matrix is neg-def iff both hold)
    #   General d: rely on each leading principal minor of -(A+Aᵀ) being
    #              positive (Sylvester's criterion).
    eps = 5e-2
    def stability_con(theta):
        _, mu_x_, L_xx_, S_xxd_, L_cond_ = _unpack(theta, K, d)
        _, _, _, A_, _, _ = _model_to_lpv(
            np.zeros(K), mu_x_, L_xx_, S_xxd_, L_cond_, attractor)
        out = []
        for k in range(K):
            sym = A_[k] + A_[k].T
            for i in range(1, d + 1):
                m = sym[:i, :i]
                out.append(((-1) ** i) * np.linalg.det(m) - eps)
        return np.array(out)
    constraints = [{"type": "ineq", "fun": stability_con}]

    # 4) Solve
    history = {"loss": []}
    def callback(theta):
        history["loss"].append(objective(theta))

    if verbose:
        print(f"   initial loss = {objective(theta0):.4e}")
    res = minimize(
        objective, theta0,
        method="SLSQP",
        constraints=constraints,
        options=dict(maxiter=max_iter, ftol=1e-6, disp=verbose),
        callback=callback,
    )
    theta_opt = res.x
    if verbose:
        print(f"   final loss   = {res.fun:.4e}  ({res.message})")

    log_pi, mu_x, L_xx, S_xxd, L_cond = _unpack(theta_opt, K, d)
    pi, mu_x, Sxx, A, b, mu_xd = _model_to_lpv(
        log_pi, mu_x, L_xx, S_xxd, L_cond, attractor)

    # Safety net: SLSQP may finish marginally infeasible. Project any A_k
    # whose symmetric part is not yet negative-definite. This guarantees the
    # final model is GAS, at the cost of a tiny extra fit error.
    for k in range(K):
        sym = A[k] + A[k].T
        if np.linalg.eigvalsh(sym).max() > -1e-3:
            A[k] = _project_to_stable(A[k], eps=0.05)
            mu_xd[k] = A[k] @ (mu_x[k] - attractor)
            b[k] = -A[k] @ attractor
    if verbose:
        max_eigs = [np.linalg.eigvalsh(A[k] + A[k].T).max() for k in range(K)]
        print(f"   post-projection max eig(A+Aᵀ) per comp: "
              f"{[f'{e:+.2e}' for e in max_eigs]}")
    return dict(pi=pi, mu_x=mu_x, Sxx=Sxx, A=A, b=b, mu_xd=mu_xd,
                attractor=attractor, history=history)


# ---------------------------------------------------------------------------
# 6. Forward simulation
# ---------------------------------------------------------------------------
def seds_velocity(x, model):
    """f(ξ) = Σ_k h_k(ξ) [ μ_k^ξ̇ + A_k (ξ - μ_k^ξ) ].  x: (d,) or (d, N)."""
    one_d = x.ndim == 1
    if one_d:
        x = x[:, None]
    out = _gmr_predict(x, model["pi"], model["mu_x"], model["Sxx"],
                       model["A"], model["mu_xd"])
    return out[:, 0] if one_d else out


def simulate(x0, model, dt=0.01, T=4000, tol=1e-3):
    traj = [x0.copy()]
    x = x0.copy()
    for _ in range(T):
        v = seds_velocity(x, model)
        x = x + dt * v
        traj.append(x.copy())
        if np.linalg.norm(v) < tol:
            break
    return np.array(traj).T


# ---------------------------------------------------------------------------
# 7. Plotting (mirrors the LPV-DS baseline plot for easy comparison)
# ---------------------------------------------------------------------------
def plot_results(demos, model, save_path):
    X_all, _ = stack(demos)
    pad = 0.5
    x_min, x_max = X_all[0].min() - pad, X_all[0].max() + pad
    y_min, y_max = X_all[1].min() - pad, X_all[1].max() + pad

    nx = ny = 60
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.vstack([XX.ravel(), YY.ravel()])
    V = seds_velocity(grid, model)
    U = V[0].reshape(ny, nx); W = V[1].reshape(ny, nx)
    speed = np.hypot(U, W)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.streamplot(XX, YY, U, W, color=speed, cmap="viridis",
                  density=1.6, linewidth=0.8, arrowsize=0.9)

    demo_colors = ["tab:red", "tab:orange", "tab:purple", "tab:brown"]
    for i, (p, _) in enumerate(demos):
        c = demo_colors[i % len(demo_colors)]
        ax.plot(p[0], p[1], color=c, lw=2.2, alpha=0.55, label=f"demo {i+1}")
        ax.plot(p[0, 0], p[1, 0], "o", color=c, ms=8, mec="k", mew=0.5)

    print("   simulating reproductions from demo start points ...")
    for i, (p, _) in enumerate(demos):
        traj = simulate(p[:, 0].copy(), model)
        end = traj[:, -1]
        ax.plot(traj[0], traj[1], "k--", lw=1.6,
                label="DS reproduction" if i == 0 else None)
        print(f"   demo {i+1}: reproduction ended at "
              f"({end[0]:+.3f}, {end[1]:+.3f}) in {traj.shape[1]} steps")

    ax.plot(0, 0, "k*", ms=18, label="attractor")
    ax.plot(model["mu_x"][:, 0], model["mu_x"][:, 1], "kx", ms=10, mew=2,
            label="GMM means")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    ax.set_title("SEDS on 2D messy-snake")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140); plt.close(fig)
    print(f"  saved figure -> {save_path}")


# ---------------------------------------------------------------------------
# 8. Main
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

    K = 6                    # SEDS is sensitive to K; 4-8 is typical for LASA
    print(f"[2/4] Initialising SEDS from joint-space EM-GMM (K={K}) ...")

    print("[3/4] Optimising SEDS parameters under stability constraints ...")
    model = fit_seds(X, Xd, K=K, attractor=np.zeros(2),
                     max_iter=200, seed=0, verbose=True)

    # Sanity check: A_k + A_k^T must be negative-definite for all k
    for k in range(K):
        sym = model["A"][k] + model["A"][k].T
        emax = np.linalg.eigvalsh(sym).max()
        print(f"   comp {k}: max eig(A+Aᵀ) = {emax:+.3e}  (should be < 0)")

    print("[4/4] Plotting & saving ...")
    fig_path   = os.path.join(out_dir, "seds_messy_snake.png")
    model_path = os.path.join(out_dir, "seds_messy_snake.pkl")
    plot_results(demos, model, fig_path)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  saved model   -> {model_path}")


if __name__ == "__main__":
    main()