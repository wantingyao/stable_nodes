import sys
import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.rcParams.update({
    'font.size':        20,
    'axes.titlesize':   20,
    'axes.labelsize':   20,
    'xtick.labelsize':  20,
    'ytick.labelsize':  20,
    'legend.fontsize':  14,
    'figure.titlesize': 20,
})
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo, "third_party", "pyLasaDataset"))
sys.path.insert(0, os.path.join(repo, "utils"))
sys.path.insert(0, os.path.join(repo, "src"))   # must be first: shadows utils/node.py

from node import MLP, rollout, rollout_to_convergence
from lsddm import Dynamics, ICNN, MakePSD, WarpedMakePSD, RadialWarp, ICNNLimitCycleV, FactoredLimitCycleV, PICNN, PhiAngularV, PhiWarpedICNNV, PolarLimitCycleShapeFn


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_lasa(shape='Leaf_2', subsample=5):
    """
    Returns a batch dict with all demos stacked, plus a shared time grid from demo_0.

    Args:
        shape:     LASA dataset shape name, e.g. 'Leaf_2', 'PShape', 'Angle', etc.
        subsample: Take every N-th time step (1 = no subsampling, 5 = 200 points).

    Returns:
        data: dict with keys
            'pos'      : (N, T, 2) float32 tensor  — normalized & centered positions
            'x0'       : (N, 2)    float32 tensor  — initial states
            't'        : (T,)      float32 tensor  — shared normalized time
            'attractor': (2,)      float32 tensor  — attractor in normalized space (before centering)
        scale: float  — mm value corresponding to 1 in normalized space
    """
    import pyLasaDataset as lasa

    leaf2 = getattr(lasa.DataSet, shape)

    # Step 1: scale normalization — divide by global max abs value → data in [-1, 1]²
    all_pos = np.concatenate([d.pos for d in leaf2.demos], axis=1)  # (2, 7000)
    scale   = np.abs(all_pos).max()

    # Shared time grid from demo_0 (same approach as reference algo_testnode.py)
    t_ref  = leaf2.demos[0].t[0]                           # (1000,)
    t_norm = (t_ref - t_ref[0]) / (t_ref[-1] - t_ref[0])  # [0, 1]
    t_norm = t_norm[::subsample]                           # (T,)

    pos_list = []
    for demo in leaf2.demos:
        pos_norm = demo.pos / scale          # (2, 1000)
        pos_list.append(pos_norm.T[::subsample])  # (T, 2)

    pos_batch = np.stack(pos_list, axis=0)  # (7, T, 2)

    # Step 2: center on attractor first, then re-normalize so that the
    # subsequent shift cannot push any point outside [-1, 1]².
    attractor   = pos_batch[:, -1, :].mean(axis=0)  # (2,) in current normalized units
    pos_batch   = pos_batch - attractor[None, None, :]
    scale2      = float(np.abs(pos_batch).max())     # max-abs of centered data
    pos_batch   = pos_batch / scale2
    scale       = scale * scale2                     # physical mm per unit in final space

    data = {
        'pos':       torch.tensor(pos_batch,          dtype=torch.float32),  # (7, T, 2)
        'x0':        torch.tensor(pos_batch[:, 0, :], dtype=torch.float32),  # (7, 2)
        't':         torch.tensor(t_norm,             dtype=torch.float32),  # (T,)
        'attractor': torch.tensor(np.zeros(2),        dtype=torch.float32),  # origin by construction
        'dim': 2,
    }
    return data, scale


# ---------------------------------------------------------------------------
# IROS dataset
# ---------------------------------------------------------------------------

IROS_SHAPES = ['IShape', 'RShape', 'SShape', 'OShape']

def load_iros(shape='IShape', subsample=1, center_mode='endpoint'):
    """Load an IROS dataset shape (.npy file with shape (N, T, 2)).

    Args:
        shape:       one of 'IShape', 'RShape', 'SShape', 'OShape'
        subsample:   take every N-th time step
        center_mode: 'endpoint' — shift mean endpoint to origin (default)
                     'centroid' — shift trajectory centroid to origin; required for
                                  limit-cycle mode so the origin lies inside the curve
                                  and V_lc = phi(x)^2 has a non-trivial level set

    Returns the same dict format as load_lasa():
        data: 'pos' (N,T,2), 'x0' (N,2), 't' (T,), 'attractor' (2,), 'dim'
        scale: float — physical units per normalized unit
    """
    iros_dir = os.path.join(repo, 'third_party', 'CLF-CBF-NODE', 'Dataset', 'IROS_dataset')
    path = os.path.join(iros_dir, f'{shape}.npy')
    pos_batch = np.load(path).astype(np.float64)  # (N, T, 2)

    pos_batch = pos_batch[:, ::subsample, :]       # (N, T', 2)

    scale = float(np.abs(pos_batch).max())
    if scale > 0:
        pos_batch = pos_batch / scale

    if center_mode == 'centroid':
        center = pos_batch.mean(axis=(0, 1))       # geometric centroid of all points
    else:
        center = pos_batch[:, -1, :].mean(axis=0)  # mean endpoint
    pos_batch = pos_batch - center[None, None, :]

    scale2 = float(np.abs(pos_batch).max())
    if scale2 > 0:
        pos_batch = pos_batch / scale2
    scale = scale * scale2

    N, T, d = pos_batch.shape
    t_norm = np.linspace(0.0, 1.0, T, dtype=np.float32)

    data = {
        'pos':       torch.tensor(pos_batch, dtype=torch.float32),   # (N, T, 2)
        'x0':        torch.tensor(pos_batch[:, 0, :], dtype=torch.float32),  # (N, 2)
        't':         torch.tensor(t_norm, dtype=torch.float32),       # (T,)
        'attractor': torch.zeros(d, dtype=torch.float32),
        'dim': d,
    }
    return data, scale


# ---------------------------------------------------------------------------
# phys-gmm data
# ---------------------------------------------------------------------------

def load_phys_gmm(dataset='2D_snake', subsample=1, T=1000):
    """Load a phys-gmm dataset, returning the same format as load_lasa().

    Some phys-gmm files store trajectories as a cell array, while the 2D toy
    datasets store them as a flat [x; y; xdot; ydot] matrix. For flat matrices,
    trajectory boundaries are read from discontinuities in position.
    Positions are normalized first, then shifted so the mean endpoint attractor
    lands at the origin. Each trajectory is remapped to T points at 1 ms steps.
    """
    import scipy.io
    from scipy.interpolate import interp1d

    dataset_file = dataset if dataset.endswith('.mat') else f'{dataset}.mat'
    mat_path = os.path.join(repo, 'third_party', 'phys-gmm', 'datasets',
                            dataset_file)
    mat = scipy.io.loadmat(mat_path)

    def split_flat_positions(pos):
        """Read trajectory boundaries from discontinuous jumps in flat Data."""
        if len(pos) < 2:
            return [pos]

        jumps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        median = float(np.median(jumps))
        q1, q3 = np.percentile(jumps, [25, 75])
        iqr = float(q3 - q1)
        threshold = max(10.0 * median, float(q3 + 10.0 * iqr), 1e-12)
        split_after = np.flatnonzero(jumps > threshold)

        starts = np.r_[0, split_after + 1]
        ends = np.r_[split_after + 1, len(pos)]
        return [pos[s:e] for s, e in zip(starts, ends) if e - s >= 2]

    if 'Data' in mat:
        D = mat['Data']
        dim = D.shape[0] // 2
        pos_all = D[:dim, :].T
        trajs = [traj[::subsample] for traj in split_flat_positions(pos_all)]
    else:
        cells = mat['data'].ravel()
        dim = cells[0].shape[0] // 2
        trajs = [cell[:dim, ::subsample].T for cell in cells]

    trajs = [traj for traj in trajs if len(traj) >= 2]
    if not trajs:
        raise ValueError(f"No valid trajectories found in {dataset_file}.")
    traj_lengths = [int(len(traj)) for traj in trajs]

    # Use a fixed 1 ms step: 1000 remapped samples cover [0, 0.999] seconds.
    t_new = np.arange(T, dtype=np.float32) / 1000.0

    def resample(traj):
        t_old = np.linspace(0.0, float(t_new[-1]), len(traj))
        return interp1d(t_old, traj, axis=0)(t_new).astype(np.float32)

    pos_batch = np.stack([resample(traj) for traj in trajs], axis=0)  # (N, T, 2)

    # Normalize first, then shift the normalized attractor to the origin.
    scale = float(np.abs(pos_batch).max())
    pos_batch = pos_batch / scale
    attractor = pos_batch[:, -1, :].mean(axis=0)
    pos_batch = pos_batch - attractor[None, None, :]

    data = {
        'pos':       torch.tensor(pos_batch,           dtype=torch.float32),  # (N, T, dim)
        'x0':        torch.tensor(pos_batch[:, 0, :],  dtype=torch.float32),  # (N, dim)
        't':         torch.tensor(t_new,               dtype=torch.float32),  # (T,)
        'attractor': torch.zeros(dim,                  dtype=torch.float32),
        'traj_lengths': traj_lengths,
        'dim': dim,
    }
    return data, scale


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def train_shape_fn(phi, data, epochs=1000, lr=1e-3):
    """Phase 0: learn phi(x)=0 on demo trajectories via the polar parameterization.

    PolarLimitCycleShapeFn guarantees a closed zero set by construction, so no
    eikonal regularizer is needed — the on-curve loss alone is sufficient.
    """
    device  = next(phi.parameters()).device
    pos     = data['pos']
    x_curve = pos.reshape(-1, pos.shape[-1]).to(device)

    opt  = torch.optim.Adam(phi.parameters(), lr=lr)
    pbar = tqdm(range(epochs), desc="Phase 0 - phi", dynamic_ncols=True)
    for ep in pbar:
        loss = phi(x_curve).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0 or ep == epochs - 1:
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    tqdm.write(f"  phi done - |phi(demo)|.mean = {phi(x_curve).abs().mean().item():.6f}")


def save_shape_fn_plot(phi, data, save_path, lim=1.3, grid_n=300):
    """Visualize the learned polar shape function after Phase 0."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    device = next(phi.parameters()).device
    phi.eval()

    xs     = torch.linspace(-lim, lim, grid_n, device=device)
    XX, YY = torch.meshgrid(xs, xs, indexing='xy')
    grid   = torch.stack([XX.ravel(), YY.ravel()], dim=1)
    with torch.no_grad():
        phi_np = phi(grid).cpu().numpy().reshape(grid_n, grid_n)
    xs_np = xs.cpu().numpy()

    theta_np = np.linspace(0, 2 * np.pi, 720)
    cos_sin  = torch.tensor(np.stack([np.cos(theta_np), np.sin(theta_np)], axis=1),
                            dtype=torch.float32, device=device)
    with torch.no_grad():
        r_vals = phi.r_net(cos_sin).cpu().numpy().ravel()
    cycle_x = r_vals * np.cos(theta_np)
    cycle_y = r_vals * np.sin(theta_np)
    pos_gt  = data['pos'].cpu().numpy()

    fig = plt.figure(figsize=(22, 6))
    gs  = fig.add_gridspec(1, 4, wspace=0.35)

    ax = fig.add_subplot(gs[0])
    vabs = float(np.abs(phi_np).max())
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    cf   = ax.contourf(xs_np, xs_np, phi_np, levels=80, cmap='RdBu_r', norm=norm, alpha=0.85)
    ax.contour(xs_np, xs_np, phi_np, levels=[0.0], colors='white', linewidths=2.0)
    for i in range(pos_gt.shape[0]):
        ax.plot(pos_gt[i, :, 0], pos_gt[i, :, 1], c='lime', linewidth=1.2, alpha=0.8,
                label='Demo' if i == 0 else None)
    ax.scatter(0, 0, c='yellow', s=60, zorder=5, label='centroid')
    fig.colorbar(cf, ax=ax, label='phi(x)', shrink=0.85)
    ax.set_aspect('equal'); ax.set_title('phi(x)  [white = phi=0]')
    ax.legend(loc='lower right')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    ax = fig.add_subplot(gs[1])
    cf2 = ax.contourf(xs_np, xs_np, np.abs(phi_np), levels=60, cmap='viridis', alpha=0.85)
    ax.contour(xs_np, xs_np, phi_np, levels=[0.0], colors='white', linewidths=2.0)
    for i in range(pos_gt.shape[0]):
        ax.plot(pos_gt[i, :, 0], pos_gt[i, :, 1], c='lime', linewidth=1.2, alpha=0.8)
    fig.colorbar(cf2, ax=ax, label='|phi(x)|', shrink=0.85)
    ax.set_aspect('equal'); ax.set_title('|phi(x)|  [dark band = on cycle]')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    ax = fig.add_subplot(gs[2], projection='polar')
    ax.plot(theta_np, r_vals, c='crimson', linewidth=2.0)
    ax.fill(theta_np, r_vals, alpha=0.15, color='crimson')
    ax.set_title('r(theta) - learned radius', pad=18)

    ax = fig.add_subplot(gs[3])
    for i in range(pos_gt.shape[0]):
        ax.plot(pos_gt[i, :, 0], pos_gt[i, :, 1], c='dodgerblue', linewidth=1.5, alpha=0.8,
                label='Demo' if i == 0 else None)
    ax.plot(cycle_x, cycle_y, c='crimson', linewidth=2.0, label='phi=0')
    ax.scatter(0, 0, c='black', s=60, zorder=5, label='centroid')
    ax.set_aspect('equal'); ax.set_title('Demo vs learned cycle')
    ax.legend(loc='lower right')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    fig.suptitle('Phase 0 - phi(x) = ||x|| - r(theta)')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Shape fn plot saved to {save_path}")
    phi.train()


def build_model(hidden_dim=256, alpha=0.1, stability_mode='off', eps=1.0, d=1.0, dim=2,
                limit_cycle=False, eps_smooth=0.05, flow_layers=0, flow_hidden=64, clf_d=0.1,
                phi=None):
    fhat = MLP(in_dim=dim, out_dim=dim, hidden_dim=256, num_layers=5)
    if limit_cycle and phi is not None:
        icnn = ICNN([dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, 1])
        V    = PhiWarpedICNNV(phi, icnn, eps=eps, d=d)
    elif limit_cycle:
        icnn = ICNN([dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, 1])
        V    = ICNNLimitCycleV(icnn, v_gamma_init=1.0, eps_smooth=eps_smooth)
    elif flow_layers > 0:
        icnn = ICNN([dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, 1])
        flow = RadialWarp(n_basis=flow_hidden)
        V    = WarpedMakePSD(icnn, flow, n=dim, eps=eps, d=d)
    else:
        icnn = ICNN([dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, 1])
        V    = MakePSD(icnn, n=dim, eps=eps, d=d)
    dynamics = Dynamics(fhat, V, alpha=alpha, stability_mode=stability_mode, clf_d=clf_d)
    return dynamics


# ---------------------------------------------------------------------------
# Visualization helpers (reference style from algo_testnode.py)
# ---------------------------------------------------------------------------

def _compute_grid(dynamics, N=41, lim=1.2):
    device = next(dynamics.parameters()).device
    x_lin = torch.linspace(-lim, lim, N)
    y_lin = torch.linspace(-lim, lim, N)
    XX, YY = torch.meshgrid(x_lin, y_lin, indexing='xy')
    grid = torch.stack([XX.ravel(), YY.ravel()], dim=1).to(device)

    with torch.enable_grad():
        grid_g = grid.clone().requires_grad_(True)
        vel    = dynamics(grid_g)
        V_vals = dynamics.V(grid_g)
        gV     = torch.autograd.grad(V_vals.sum(), grid_g, create_graph=False)[0]
        dotV   = (gV * vel).sum(dim=1, keepdim=True)

        vel    = vel.detach()
        V_vals = V_vals.detach()
        dotV   = dotV.detach()

    U      = vel[:, 0].reshape(N, N).cpu().numpy()
    V_f    = vel[:, 1].reshape(N, N).cpu().numpy()
    V_lyap = V_vals[:, 0].reshape(N, N).cpu().numpy()
    dotV_h = dotV[:, 0].reshape(N, N).cpu().numpy()
    return x_lin.numpy(), y_lin.numpy(), U, V_f, V_lyap, dotV_h


def _draw_streamplot(ax, x_np, y_np, U, V_f):
    """Violet streamplot."""
    sp = ax.streamplot(x_np, y_np, U, V_f,
                       color='plum', linewidth=1.2, arrowsize=1.8, density=1.2)
    sp.lines.set_alpha(0.85)
    sp.arrows.set_alpha(0.85)
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.4)
    ax.axvline(0, color='k', linewidth=0.4, alpha=0.4)
    ax.set_xlim(x_np[0], x_np[-1]); ax.set_ylim(y_np[0], y_np[-1])
    ax.set_aspect('equal'); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")


def _draw_lyapunov_cf(ax, x_np, y_np, V_lyap, gamma=0.4, is_lc=False, phi_grid=None):
    """Draw Lyapunov contourf onto ax; return cf for colorbar attachment."""
    from matplotlib.colors import PowerNorm
    vmax = V_lyap.max()
    levels = np.concatenate([[0.0], np.geomspace(vmax * 1e-3, vmax, 20)])
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)
    cf = ax.contourf(x_np, y_np, V_lyap, levels=levels, cmap='viridis', alpha=0.75, norm=norm)
    ax.contour(x_np, y_np, V_lyap, levels=levels[::4], colors='white', linewidths=0.4, alpha=0.4)
    ax.axhline(0, color='w', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='w', linewidth=0.5, alpha=0.5)
    if is_lc and phi_grid is not None:
        # phi=0 is the exact learned limit cycle
        ax.contour(x_np, y_np, phi_grid, levels=[0.0], colors='lime',
                   linewidths=2.0, zorder=6)
    else:
        ax.plot(0.0, 0.0, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1.0,
                zorder=6, label='V=0')
    ax.set_xlim(x_np[0], x_np[-1]); ax.set_ylim(y_np[0], y_np[-1])
    return cf


def _attach_colorbar(fig, ax, cf, label='V(x)'):
    """Attach colorbar via make_axes_locatable so it doesn't shrink the main axis."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cf, cax=cax, label=label)


def _draw_lyapunov(fig, ax, x_np, y_np, V_lyap):
    """Standalone Lyapunov panel."""
    cf = _draw_lyapunov_cf(ax, x_np, y_np, V_lyap)
    _attach_colorbar(fig, ax, cf)
    ax.legend(loc='upper right')
    ax.set_aspect('equal'); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")


def _draw_lyapunov_surface_3d(ax, x_np, y_np, V_lyap, U=None, V_f=None, is_lc=False, phi_grid=None):
    """3D surface plot of V(x1, x2) with floor contours and f(x) arrow.

    _compute_grid uses torch indexing='xy': V_lyap[row, col] = V(x_np[col], y_np[row]).
    We apply log1p transform so the bowl shape is visible even when V spans many orders
    of magnitude — this is more reliable than PowerNorm with plot_surface.
    """
    # Match _compute_grid's indexing='xy'
    XX, YY = np.meshgrid(x_np, y_np, indexing='xy')

    # log1p compresses large V values while preserving bowl shape near origin
    V_disp = V_lyap ** 0.4
    vmin = float(V_disp.min())
    vmax = float(V_disp.max())

    surf = ax.plot_surface(XX, YY, V_disp, cmap='viridis',
                           vmin=vmin, vmax=vmax,
                           alpha=0.7, linewidth=0, antialiased=True,
                           rcount=60, ccount=60)

    # Floor contours projected at z=vmin
    n_floor = 16
    floor_levels = np.linspace(vmin, vmax, n_floor)
    ax.contourf(XX, YY, V_disp, levels=floor_levels, cmap='viridis',
                alpha=0.55, zdir='z', offset=vmin)
    ax.contour(XX, YY, V_disp, levels=floor_levels[::4], colors='white',
               linewidths=0.4, alpha=0.55, zdir='z', offset=vmin)

    # f(x) arrow — pick upper-left quadrant, V_lyap[row=iy, col=ix]
    if U is not None and V_f is not None:
        N_x, N_y = len(x_np), len(y_np)
        ix, iy = N_x // 3, 2 * N_y // 3
        px = float(x_np[ix])
        py = float(y_np[iy])
        pv = float(V_disp[iy, ix])
        scale = (x_np[-1] - x_np[0]) * 0.18
        raw = np.hypot(float(U[iy, ix]), float(V_f[iy, ix])) + 1e-8
        dx = float(U[iy, ix]) / raw * scale
        dy = float(V_f[iy, ix]) / raw * scale
        ax.quiver(px, py, pv, dx, dy, 0,
                  color='white', linewidth=3.0, arrow_length_ratio=0.5, zorder=8)
        ax.text(px + dx * 1.5, py + dy * 1.5, pv + (vmax - vmin) * 0.06,
                r'$f(x)$', color='white', fontsize=16, zorder=9)

    if is_lc and phi_grid is not None:
        # phi=0 is the exact learned limit cycle — draw it on the floor
        ax.contour(XX, YY, phi_grid, levels=[0.0], colors='lime',
                   linewidths=2.0, alpha=0.9, zdir='z', offset=vmin)
    elif is_lc:
        pass  # no phi_grid available, skip floor marker
    else:
        ax.scatter([0], [0], [vmin], c='red', s=80, zorder=10)

    ax.set_zlim(vmin, vmax)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$V^{0.4}$")
    ax.set_title("Lyapunov $V(x)$" if not is_lc else "Lyapunov $V_{lc}(x)$")
    # elev=30 shows depth; azim=-60 (near default) shows both axes clearly
    ax.view_init(elev=30, azim=-60)
    return surf


def _draw_dotv(fig, ax, x_np, y_np, dotV):
    """Standalone dV/dt panel. Positive regions indicate Lyapunov violation."""
    vmin = min(float(np.nanmin(dotV)), 0.0)
    vmax = max(float(np.nanmax(dotV)), 0.0)
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    levels = np.linspace(vmin, vmax, 81)
    cf = ax.contourf(x_np, y_np, dotV, levels=levels, cmap='seismic_r',
                     vmin=vmin, vmax=vmax, extend='both')
    if vmin <= 0.0 <= vmax:
        ax.contour(x_np, y_np, dotV, levels=[0.0], colors='black',
                   linewidths=0.8, alpha=0.8)
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.4)
    ax.axvline(0, color='k', linewidth=0.4, alpha=0.4)
    ax.set_xlim(x_np[0], x_np[-1]); ax.set_ylim(y_np[0], y_np[-1])
    ax.set_aspect('equal'); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    _attach_colorbar(fig, ax, cf, label=r"$\nabla V \cdot f$")


def _plot_traj_panel(ax, data, dynamics, t_eval, title="", perturb=0.02,
                     grid_data=None, probe_grid_n=8, probe_lim=1.2,
                     lyap_data=None, fig=None, show_probe=False, phi_grid=None):
    """
    Reference-style panel:
      - Optional Lyapunov contourf as bottom layer (if lyap_data provided)
      - Violet streamplot on top (if grid_data provided)
      - Ground truth demos in dodgerblue
      - Model rollouts in crimson (solid from original x0, dashed from perturbed)
      - Start: large black filled circle; Target: large black X
    """
    if t_eval is None:
        device = next(dynamics.parameters()).device
        t_eval = torch.linspace(0.0, 1.0, 1000, device=device)

    pos_gt   = data['pos']   # (N, T, 2)
    x0_batch = data['x0']    # (N, 2)
    target   = data.get('attractor', pos_gt[0, -1])

    # Lyapunov contourf as bottom layer
    is_lc = getattr(dynamics, 'stability_mode', '') == 'limit_cycle'
    if lyap_data is not None:
        x_np_l, y_np_l, V_lyap = lyap_data
        cf = _draw_lyapunov_cf(ax, x_np_l, y_np_l, V_lyap, is_lc=is_lc, phi_grid=phi_grid)
        if fig is not None:
            _attach_colorbar(fig, ax, cf)

    # Streamplot background
    if grid_data is not None:
        x_np, y_np, U, V_f = grid_data
        _draw_streamplot(ax, x_np, y_np, U, V_f)

    # Ground truth trajectories
    for i in range(pos_gt.shape[0]):
        traj = pos_gt[i].cpu().numpy()
        ax.plot(traj[:, 0], traj[:, 1], c='dodgerblue',
                linewidth=1.5, alpha=0.7, label='Real' if i == 0 else None,
                zorder=3)

    # Start points
    for i in range(pos_gt.shape[0]):
        ax.plot(pos_gt[i, 0, 0].item(), pos_gt[i, 0, 1].item(),
                marker='o', color='black', markersize=10,
                markeredgecolor='black', zorder=5,
                label='Start' if i == 0 else None)

    # Target
    ax.plot(target[0].item(), target[1].item(),
            marker='x', color='black', markersize=12,
            markeredgewidth=2.5, zorder=5, label='Target')

    def _mark_endpoints(traj_np, color, markersize, alpha=1.0):
        """Plot a triangle at the last point of each trajectory."""
        for i in range(traj_np.shape[0]):
            ax.plot(traj_np[i, -1, 0], traj_np[i, -1, 1],
                    marker='^', color=color, markersize=markersize,
                    markeredgecolor='white', markeredgewidth=0.5,
                    alpha=alpha, zorder=6, linestyle='none')

    # Limit cycle trajectories never converge to origin, so cap chunks lower and
    # batch all x0s (original + perturbed) into one ODE call for GPU efficiency.
    max_chunks = max(1, int(np.ceil(3 * pos_gt.shape[1] / len(t_eval))))
    x_pred_np = rollout_to_convergence(dynamics, x0_batch, t_eval,
                                       max_chunks=max_chunks).cpu().numpy()

    for i in range(x_pred_np.shape[0]):
        ax.plot(x_pred_np[i, :, 0], x_pred_np[i, :, 1], c='crimson',
                linewidth=1.5, zorder=4, label='Model' if i == 0 else None)
    _mark_endpoints(x_pred_np, 'crimson', markersize=7)

    # Perturbed rollouts disabled
    # offsets = [np.array([perturb, 0.0]), np.array([-perturb, 0.0])]
    # x0_perturb_list = [...]
    # ...

    # Probe rollouts from a uniform grid to reveal spurious attractors
    if show_probe:
        xs = np.linspace(-probe_lim, probe_lim, probe_grid_n)
        probe_x0 = torch.tensor(
            np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2),
            dtype=torch.float32, device=x0_batch.device,
        )
        # Single batched ODE call for all probe points
        probe_np = rollout_to_convergence(dynamics, probe_x0, t_eval,
                                          max_chunks=max_chunks).cpu().numpy()
        for i in range(probe_np.shape[0]):
            ax.plot(probe_np[i, :, 0], probe_np[i, :, 1], c='darkorange',
                    linewidth=0.6, alpha=0.35, zorder=2,
                    label='Probe' if i == 0 else None)
        _mark_endpoints(probe_np, 'darkorange', markersize=4, alpha=0.6)

    # For limit cycle mode, draw the learned cycle boundary as a contour
    if getattr(dynamics, 'stability_mode', '') == 'limit_cycle':
        lim = 1.3
        N_g = 200
        device = next(dynamics.parameters()).device
        xs = torch.linspace(-lim, lim, N_g, device=device)
        XX, YY = torch.meshgrid(xs, xs, indexing='xy')
        grid = torch.stack([XX.ravel(), YY.ravel()], dim=1)
        with torch.no_grad():
            if hasattr(dynamics.V, 'phi'):
                cycle_grid = dynamics.V.phi(grid).reshape(N_g, N_g).cpu().numpy()
                levels = [0.0]
            elif hasattr(dynamics.V, 'icnn'):
                v_gamma = dynamics.V.v_gamma.item()
                cycle_grid = (dynamics.V.icnn(grid) - v_gamma).reshape(N_g, N_g).cpu().numpy()
                levels = [0.0]
            else:
                cycle_grid = None
        if cycle_grid is not None:
            ax.contour(xs.cpu().numpy(), xs.cpu().numpy(), cycle_grid,
                       levels=levels, colors='white', linewidths=1.5, linestyles='--',
                       zorder=5)

    ax.set_aspect('equal'); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend(loc='lower right')


def _plot_traj_panel_3d(ax, data, dynamics, t_eval, title="", perturb=0.02):
    """3D trajectory panel: GT demos in blue, model rollouts in red."""
    if t_eval is None:
        device = next(dynamics.parameters()).device
        t_eval = torch.linspace(0.0, 1.0, 1000, device=device)

    pos_gt   = data['pos']    # (N, T, 3)
    x0_batch = data['x0']     # (N, 3)

    for i in range(pos_gt.shape[0]):
        traj = pos_gt[i].cpu().numpy()
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                c='dodgerblue', linewidth=1.2, alpha=0.7,
                label='Real' if i == 0 else None)
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
                   c='black', s=40, zorder=5,
                   label='Start' if i == 0 else None)

    ax.scatter(0.0, 0.0, 0.0, c='black', marker='x', s=100, linewidths=2,
               zorder=5, label='Target')

    max_chunks = max(1, int(np.ceil(3 * pos_gt.shape[1] / len(t_eval))))
    x_pred_np = rollout_to_convergence(dynamics, x0_batch, t_eval,
                                       max_chunks=max_chunks).cpu().numpy()
    for i in range(x_pred_np.shape[0]):
        ax.plot(x_pred_np[i, :, 0], x_pred_np[i, :, 1], x_pred_np[i, :, 2],
                c='crimson', linewidth=1.2, zorder=4,
                label='Model' if i == 0 else None)

    offsets = [np.array([perturb, 0.0, 0.0]), np.array([0.0, perturb, 0.0]),
               np.array([0.0, 0.0, perturb])]
    _perturb_labeled = False
    for offset in offsets:
        x0_p = (x0_batch + torch.tensor(offset, dtype=torch.float32,
                                         device=x0_batch.device)).clamp(-1.2, 1.2)
        x_p_np = rollout_to_convergence(dynamics, x0_p, t_eval,
                                        max_chunks=max_chunks).cpu().numpy()
        for i in range(x_p_np.shape[0]):
            label = 'Perturbed' if not _perturb_labeled else None
            _perturb_labeled = True
            ax.plot(x_p_np[i, :, 0], x_p_np[i, :, 1], x_p_np[i, :, 2],
                    c='crimson', linewidth=0.7, linestyle='--', alpha=0.4,
                    zorder=3, label=label)

    ax.invert_yaxis()
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.invert_yaxis()
    ax.view_init(elev=25, azim=160, roll=0)  # looking from negative x1 toward +x1


def _plot_lyapunov_surface(ax, dynamics, lim=0.9, grid_n=60, n_contours=12, slice_dim=2):
    """3D surface of V(x1,x2) at slice_dim=0 plane, with contour lines at the base."""
    from matplotlib.colors import PowerNorm

    device = next(dynamics.parameters()).device
    lin = np.linspace(-lim, lim, grid_n)
    A, B = np.meshgrid(lin, lin, indexing='ij')   # (G, G)

    axes = [0, 1, 2]
    free = [a for a in axes if a != slice_dim]
    pts_np = np.zeros((grid_n * grid_n, 3), dtype=np.float32)
    pts_np[:, free[0]] = A.ravel()
    pts_np[:, free[1]] = B.ravel()

    with torch.no_grad():
        V_surf = dynamics.V(
            torch.tensor(pts_np, device=device)
        ).cpu().numpy().ravel().reshape(grid_n, grid_n)

    vmax = float(V_surf.max())
    norm = PowerNorm(gamma=0.4, vmin=0.0, vmax=vmax)

    ax.plot_surface(A, B, V_surf, cmap='viridis', norm=norm,
                    alpha=0.45, linewidth=0, antialiased=True)

    levels = np.concatenate([[0.0], np.geomspace(vmax * 1e-3, vmax, n_contours - 1)])
    ax.contour(A, B, V_surf, levels=levels, zdir='z', offset=0,
               cmap='viridis', norm=norm, linewidths=0.8, alpha=0.8)

    ax.scatter(0, 0, 0, c='red', marker='o', s=60, zorder=5)
    labels = ["$x_1$", "$x_2$", "$x_3$"]
    ax.set_xlabel(labels[free[0]])
    ax.set_ylabel(labels[free[1]])
    ax.set_zlabel("$V(x)$")
    ax.set_title(f"Lyapunov $V(x)$  [slice $x_{{{slice_dim+1}}}=0$]")
    ax.view_init(elev=30, azim=-60)


def _make_figure_3d(dynamics, data, t_eval, title_suffix, fig_kw):
    """
    Warmup:    1 panel  — trajectories only
    icnn phase: 3 panels — trajectories | Lyapunov surface | combined
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    show_lyap = dynamics.stability_mode == 'icnn'

    if not show_lyap:
        fig = plt.figure(figsize=(9, 7), **fig_kw)
        ax3d = fig.add_subplot(111, projection='3d')
        _plot_traj_panel_3d(ax3d, data, dynamics, t_eval,
                            title=f"3D trajectories {title_suffix}")
        fig.tight_layout()
        return fig

    fig = plt.figure(figsize=(24, 7), **fig_kw)

    # Panel 1: trajectories only
    ax_traj = fig.add_subplot(131, projection='3d')
    _plot_traj_panel_3d(ax_traj, data, dynamics, t_eval,
                        title=f"Trajectories {title_suffix}")

    # Panel 2: Lyapunov surface only
    ax_ly = fig.add_subplot(132, projection='3d')
    _plot_lyapunov_surface(ax_ly, dynamics)

    # Panel 3: combined — Lyapunov surface + trajectories overlaid
    ax_comb = fig.add_subplot(133, projection='3d')
    _plot_lyapunov_surface(ax_comb, dynamics)
    _plot_traj_panel_3d(ax_comb, data, dynamics, t_eval, title=f"Combined {title_suffix}")

    fig.tight_layout()
    return fig


def _make_figure(dynamics, data, t_eval, title_suffix, fig_kw):
    """Shared 1×4 layout (or 1×1 pre-warmup): vector field | Lyapunov 3D surface | dV/dt | combined."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if dynamics.stability_mode not in ('icnn', 'limit_cycle'):
        # Warmup: skip V evaluation entirely
        x_np, y_np, U, V_f, _, _ = _compute_grid(dynamics)
        fig, ax = plt.subplots(figsize=(7, 7), **fig_kw)
        _plot_traj_panel(ax, data, dynamics, t_eval,
                         title=f"Vector field - {title_suffix}",
                         grid_data=(x_np, y_np, U, V_f))
        fig.tight_layout()
        return fig

    is_lc = (dynamics.stability_mode == 'limit_cycle')
    tqdm.write("  [vis] computing grid (V, ∇V, dV/dt) ...")
    x_np, y_np, U, V_f, V_lyap, dotV = _compute_grid(dynamics)
    grid_data = (x_np, y_np, U, V_f)
    lyap_data  = (x_np, y_np, V_lyap)

    # Compute cycle boundary grid for limit cycle contour overlays
    phi_grid = None
    if is_lc:
        tqdm.write("  [vis] computing phi grid for cycle contour ...")
        device = next(dynamics.parameters()).device
        N_g, lim_g = 41, 1.2
        xs_g = torch.linspace(-lim_g, lim_g, N_g, device=device)
        XX_g, YY_g = torch.meshgrid(xs_g, xs_g, indexing='xy')
        grid_g = torch.stack([XX_g.ravel(), YY_g.ravel()], dim=1)
        with torch.no_grad():
            if hasattr(dynamics.V, 'phi'):
                phi_grid = dynamics.V.phi(grid_g).reshape(N_g, N_g).cpu().numpy()
            elif hasattr(dynamics.V, 'icnn'):
                v_gamma = dynamics.V.v_gamma.item()
                phi_grid = (dynamics.V.icnn(grid_g) - v_gamma).reshape(N_g, N_g).cpu().numpy()

    v_label = '$V_{lc}(x)$' if is_lc else '$V(x)$'
    fig = plt.figure(figsize=(35, 7), **fig_kw)

    # Panel 1: vector field + rollouts (2D)
    tqdm.write("  [vis] panel 1/5: vector field + rollouts ...")
    ax1 = fig.add_subplot(1, 5, 1)
    _plot_traj_panel(ax1, data, dynamics, t_eval,
                     title=f"{title_suffix} — Vector field",
                     grid_data=grid_data)

    # Panel 2: Lyapunov 3D surface
    tqdm.write("  [vis] panel 2/5: Lyapunov 3D surface ...")
    ax2 = fig.add_subplot(1, 5, 2, projection='3d')
    surf = _draw_lyapunov_surface_3d(ax2, x_np, y_np, V_lyap, U=U, V_f=V_f,
                                     is_lc=is_lc, phi_grid=phi_grid)
    ax2.set_title(f"{title_suffix} — {v_label}", pad=12)
    fig.colorbar(surf, ax=ax2, shrink=0.55, pad=0.1, label=v_label)

    # Panel 3: dV/dt heatmap (2D)
    tqdm.write("  [vis] panel 3/5: dV/dt heatmap ...")
    ax3 = fig.add_subplot(1, 5, 3)
    _draw_dotv(fig, ax3, x_np, y_np, dotV)
    ax3.set_title(f"{title_suffix} — " + r"$\dot V = \nabla V \cdot f$")

    # Panel 4: combined overlay (2D): Lyapunov contourf + streamplot + rollouts
    tqdm.write("  [vis] panel 4/5: combined overlay + rollouts ...")
    ax4 = fig.add_subplot(1, 5, 4)
    _plot_traj_panel(ax4, data, dynamics, t_eval,
                     title=f"{title_suffix} — Combined",
                     grid_data=grid_data,
                     lyap_data=lyap_data, fig=fig, phi_grid=phi_grid)

    # Panel 5: raw fhat (no CLF projection) + Lyapunov contourf
    tqdm.write("  [vis] panel 5/5: raw fhat + Lyapunov ...")
    ax5 = fig.add_subplot(1, 5, 5)
    device = next(dynamics.parameters()).device
    N_g, lim_g = 41, 1.2
    x_lin = torch.linspace(-lim_g, lim_g, N_g)
    y_lin = torch.linspace(-lim_g, lim_g, N_g)
    XX, YY = torch.meshgrid(x_lin, y_lin, indexing='xy')
    grid_raw = torch.stack([XX.ravel(), YY.ravel()], dim=1).to(device)
    with torch.no_grad():
        vel_raw = dynamics.fhat(grid_raw)
    U_raw = vel_raw[:, 0].reshape(N_g, N_g).cpu().numpy()
    V_raw = vel_raw[:, 1].reshape(N_g, N_g).cpu().numpy()
    cf5 = _draw_lyapunov_cf(ax5, x_np, y_np, V_lyap, is_lc=is_lc, phi_grid=phi_grid)
    _attach_colorbar(fig, ax5, cf5, label=v_label)
    _draw_streamplot(ax5, x_lin.numpy(), y_lin.numpy(), U_raw, V_raw)
    ax5.set_title(f"{title_suffix} — Raw $\\hat{{f}}$ + {v_label}")
    ax5.set_aspect('equal'); ax5.set_xlabel("$x_1$"); ax5.set_ylabel("$x_2$")

    tqdm.write("  [vis] saving figure ...")
    fig.tight_layout()
    return fig


def save_intermediate_plot(dynamics, data, t_eval, epoch, logdir, dtwd=None, conv_mse=None, use_wandb=False, shape_name=''):
    """Save plot every eval_every epochs (2D: streamplot; 3D: trajectory axes)."""
    import matplotlib.pyplot as plt
    dynamics.eval()

    dim = data.get('dim', data['pos'].shape[-1])
    dtwd_tag    = f"_dtwd_{dtwd:.4f}"       if dtwd     is not None else ""
    convmse_tag = f"_convmse_{conv_mse:.4f}" if conv_mse is not None else ""
    title_suffix = shape_name
    if dim == 3:
        fig = _make_figure_3d(dynamics, data, t_eval, title_suffix=title_suffix, fig_kw={})
    else:
        fig = _make_figure(dynamics, data, t_eval, title_suffix=title_suffix, fig_kw={})
    fig.tight_layout()
    vis_dir = os.path.join(logdir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    fig_path = os.path.join(vis_dir, f"epoch_{epoch:04d}{dtwd_tag}{convmse_tag}.png")
    fig.savefig(fig_path, dpi=120)
    if use_wandb and _WANDB_AVAILABLE:
        wandb.log({"vis/trajectory": wandb.Image(fig_path)}, step=epoch)
    plt.close(fig)
    dynamics.train()



def save_final_plot(dynamics, data, t_eval, loss_history, save_path):
    """Final result plot (2D: streamplot panels; 3D: 3D trajectory axes)."""
    import matplotlib.pyplot as plt
    dynamics.eval()

    dim = data.get('dim', data['pos'].shape[-1])
    if dim == 3:
        fig = _make_figure_3d(dynamics, data, t_eval, title_suffix="", fig_kw={})
    else:
        fig = _make_figure(dynamics, data, t_eval, title_suffix="", fig_kw={})
    fig_path = save_path.replace(".pt", "_results.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Plot saved to {fig_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _dtw_distance(p: np.ndarray, q: np.ndarray) -> float:
    """DTW distance between two trajectories p (T1, d) and q (T2, d)."""
    T1, T2 = len(p), len(q)
    dtw = np.full((T1 + 1, T2 + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            cost = np.linalg.norm(p[i - 1] - q[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[T1, T2])


@torch.no_grad()
def evaluate(dynamics, data, solver='rk4') -> dict:
    """Compute RMSE_vel, MVD, and DTWD on the LASA demos.

    - RMSE_vel / MVD: computed at GT trajectory positions (independent of rollout).
    - DTWD: model rollout from each demo x0 vs GT trajectory shape.

    Returns a dict with keys: rmse_vel, mvd, dtwd (all Python floats).
    """
    dynamics.eval()

    pos_gt   = data['pos']   # (N, T, 2)
    t        = data['t']     # (T,)
    N, T, d  = pos_gt.shape

    dt       = t[1:] - t[:-1]
    vel_gt   = (pos_gt[:, 1:, :] - pos_gt[:, :-1, :]) / dt[None, :, None]
    x_flat   = pos_gt[:, :-1, :].reshape(-1, d).requires_grad_(True)
    with torch.enable_grad():
        vel_pred = dynamics(x_flat).reshape(N, T - 1, d)

    err      = (vel_pred - vel_gt).norm(dim=-1)
    mvd      = err.mean().item()
    rmse_vel = err.pow(2).mean().sqrt().item()

    x0_batch = data['x0']
    tqdm.write(f"  [eval]   rollout {N} trajectories ({T} steps) ...")
    with torch.enable_grad():
        x_pred = rollout(dynamics, x0_batch, t, method=solver).detach()
    tqdm.write(f"  [eval]   computing DTW ...")
    dtwd_vals = [_dtw_distance(x_pred[i].cpu().numpy(), pos_gt[i].cpu().numpy())
                 for i in range(N)]
    dtwd = float(np.mean(dtwd_vals))

    # MSE of final position from attractor (origin)
    x_final  = x_pred[:, -1, :]                              # (N, d)
    conv_mse = x_final.pow(2).sum(dim=-1).mean().item()      # mean ||x_T||²

    return {'rmse_vel': rmse_vel, 'mvd': mvd, 'dtwd': dtwd, 'conv_mse': conv_mse}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(dynamics, data, logdir, num_epochs=500, lr=1e-3, weight_decay=1e-4,
          warmup_epochs=200, pos_weight=1.0, vel_weight=1.0, use_wandb=False,
          icnn_lr_scale=0.1, eval_every=100, lyapunov_only_epochs=200,
          limit_cycle=False, levelset_weight=1.0, shape_name='', data_eval=None):
    """Three-phase training.

    Standard (limit_cycle=False):
      phase 1:  epoch 1..warmup_epochs       — stability_mode='off', all params
      phase 2a: warmup+1..warmup+lyap_only   — stability_mode='icnn', fhat frozen
      phase 2b: warmup+lyap_only+1..end      — stability_mode='icnn', all params

    Limit cycle (limit_cycle=True):
      phase 1:  epoch 1..warmup_epochs       — stability_mode='off', all params
                (phi already trained in phase 0 before this call)
      phase 2a: warmup+1..warmup+lyap_only   — stability_mode='limit_cycle', fhat frozen
      phase 2b: warmup+lyap_only+1..end      — stability_mode='limit_cycle', all params
    """
    x0_batch = data.get('x0')
    pos_gt   = data.get('pos')
    t        = data.get('t')

    phase2_mode   = 'limit_cycle' if limit_cycle else 'icnn'
    phase2a_start = warmup_epochs + 1
    phase2b_start = warmup_epochs + lyapunov_only_epochs + 1

    optimizer = torch.optim.Adam(
        dynamics.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=warmup_epochs, eta_min=lr * 0.1
    )

    data_eval = data_eval if data_eval is not None else data

    loss_history = []
    best_dtwd    = float('inf')
    best_path    = None
    solver       = 'rk4'
    pbar = tqdm(range(1, num_epochs + 1), desc="Training", dynamic_ncols=True)

    for epoch in pbar:
        # Phase 1 → 2a: enable stability projection, freeze fhat, train V / phi only
        if epoch == phase2a_start:
            dynamics.stability_mode = phase2_mode
            dynamics.V.invalidate_zero_cache()
            for p in dynamics.fhat.parameters():
                p.requires_grad_(False)
            phase2_lr = lr * icnn_lr_scale
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, dynamics.parameters()),
                lr=phase2_lr, weight_decay=weight_decay,
            )
            remaining = num_epochs - warmup_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=phase2_lr * 0.1
            )
            tqdm.write(f"\n[Epoch {epoch}] Phase 2a — {phase2_mode} V/phi only (fhat frozen), lr={phase2_lr:.2e}")

        # Phase 2a → 2b: unfreeze fhat, joint fine-tuning with separate LR groups
        if epoch == phase2b_start:
            for p in dynamics.fhat.parameters():
                p.requires_grad_(True)
            optimizer = torch.optim.Adam([
                {'params': dynamics.fhat.parameters(), 'lr': phase2_lr},
                {'params': dynamics.V.parameters(),    'lr': phase2_lr * 0.1},
            ], weight_decay=weight_decay)
            remaining2b = num_epochs - phase2b_start + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(remaining2b, 1), eta_min=phase2_lr * 0.01
            )
            tqdm.write(f"\n[Epoch {epoch}] Phase 2b — joint fine-tuning "
                       f"(fhat lr={phase2_lr:.2e}, V lr={phase2_lr*0.1:.2e})")

        dynamics.train()
        optimizer.zero_grad()

        if pos_weight > 0.0:
            x_pred   = rollout(dynamics, x0_batch, t, method=solver)
            loss_pos = F.mse_loss(x_pred, pos_gt)
        else:
            loss_pos = torch.tensor(0.0, device=x0_batch.device)

        dt       = t[1:] - t[:-1]
        vel_gt   = (pos_gt[:, 1:, :] - pos_gt[:, :-1, :]) / dt[None, :, None]
        N, _, d  = pos_gt.shape
        x_flat   = pos_gt[:, :-1, :].reshape(-1, d).requires_grad_(True)
        vel_pred = dynamics(x_flat).reshape(N, -1, d)
        loss_vel = F.mse_loss(vel_pred, vel_gt)

        loss = pos_weight * loss_pos + vel_weight * loss_vel

        if (dynamics.stability_mode == 'limit_cycle'
                and levelset_weight > 0.0
                and hasattr(dynamics.V, 'level_set_loss')):
            x_demo_flat = pos_gt[:, :-1, :].reshape(-1, d).detach()
            loss = loss + levelset_weight * dynamics.V.level_set_loss(x_demo_flat)

        loss.backward()
        clip_grad_norm_(dynamics.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if dynamics.stability_mode in ('icnn', 'limit_cycle'):
            dynamics.V.invalidate_zero_cache()

        loss_history.append(loss.item())
        cur_lr = scheduler.get_last_lr()[0]
        mode_tag = dynamics.stability_mode
        pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{cur_lr:.2e}", stab=mode_tag)

        if use_wandb and _WANDB_AVAILABLE:
            wandb.log({"train/loss": loss.item(), "train/loss_pos": loss_pos.item(),
                       "train/loss_vel": loss_vel.item(), "train/lr": cur_lr,
                       "train/stability_mode": mode_tag}, step=epoch)

        # Evaluate every eval_every epochs and at the final epoch
        post_warmup = epoch > warmup_epochs
        if epoch % eval_every == 0 or epoch == num_epochs:
            tqdm.write(f"  [eval] computing metrics (rollout + DTW) ...")
            metrics  = evaluate(dynamics, data_eval, solver=solver)
            dtwd_str = f"{metrics['dtwd']:.4f}" if not (metrics['dtwd'] != metrics['dtwd']) else "N/A"
            tqdm.write(
                f"[Epoch {epoch:>5}]  RMSE_vel={metrics['rmse_vel']:.6f}"
                f"  MVD={metrics['mvd']:.6f}  DTWD={dtwd_str}"
                f"  ConvMSE={metrics['conv_mse']:.6f}"
            )
            if use_wandb and _WANDB_AVAILABLE:
                log_d = {"eval/rmse_vel": metrics['rmse_vel'], "eval/mvd": metrics['mvd'],
                         "eval/conv_mse": metrics['conv_mse']}
                if not (metrics['dtwd'] != metrics['dtwd']):
                    log_d["eval/dtwd"] = metrics['dtwd']
                wandb.log(log_d, step=epoch)

            best_metric     = metrics['dtwd']
            best_metric_tag = "dtwd"

            if post_warmup:
                ckpt = {
                    'epoch':          epoch,
                    'dtwd':           metrics['dtwd'],
                    'model_state':    dynamics.state_dict(),
                    'metrics':        metrics,
                    'stability_mode': dynamics.stability_mode,
                }

                # Save periodic checkpoint every 200 epochs after warmup
                epochs_since_warmup = epoch - warmup_epochs
                conv_mse = metrics['conv_mse']
                if epochs_since_warmup % 200 == 0 or epoch == num_epochs:
                    ckpt_path = os.path.join(
                        logdir, f"ckpt_ep{epoch:04d}_{best_metric_tag}_{best_metric:.4f}_convmse_{conv_mse:.4f}.pt"
                    )
                    torch.save(ckpt, ckpt_path)
                    tqdm.write(f"  → {os.path.basename(ckpt_path)} saved")

                # Save best model
                if best_metric < best_dtwd:
                    best_dtwd = best_metric
                    if best_path is not None and os.path.exists(best_path):
                        os.remove(best_path)
                    best_path = os.path.join(
                        logdir, f"best_model_ep{epoch:04d}_{best_metric_tag}_{best_dtwd:.4f}_convmse_{conv_mse:.4f}.pt"
                    )
                    torch.save(ckpt, best_path)
                    tqdm.write(f"  → {os.path.basename(best_path)} saved (best)")

            save_intermediate_plot(dynamics, data_eval, t, epoch, logdir,
                                   dtwd=metrics['dtwd'], conv_mse=metrics['conv_mse'],
                                   use_wandb=use_wandb, shape_name=shape_name)

    return loss_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Stable NODE on 2D_messy-snake")
    parser.add_argument("--hidden_dim",    type=int,   default=64)
    parser.add_argument("--alpha",         type=float, default=0.01) # Higher alpha → stronger stability regularization (V(x) dominates over fhat(x))
    parser.add_argument("--epochs",        type=int,   default=10000)
    parser.add_argument("--lr",            type=float, default=3e-3)
    parser.add_argument("--weight_decay",  type=float, default=0.0)
    parser.add_argument("--shape",         type=str,   default='Leaf_2',
                        help="LASA shape name, e.g. Leaf_2, PShape, Angle, Sine ...")
    parser.add_argument("--subsample",     type=int,   default=1,
                        help="Take every N-th time step (1=no subsampling, 5=200pts)")
    parser.add_argument("--pos_weight",    type=float, default=0.0,
                        help="Weight for position rollout loss term (0=disable)")
    parser.add_argument("--vel_weight",    type=float, default=1.0,
                        help="Weight for velocity loss term (0=disable)")
    parser.add_argument("--logdir",        type=str,   default=None,
                        help="Override experiment dir (default: logs/snode/<shape>/<datetime>)")
    parser.add_argument("--warmup_epochs",  type=int,   default=1000,
                        help="Epochs with stability off (plain NODE warm-up), then ICNN projection")
    parser.add_argument("--icnn_lr_scale",        type=float, default=0.1,
                        help="LR multiplier for phase 2 (icnn), applied to both fhat and V")
    parser.add_argument("--lyapunov_only_epochs", type=int,   default=100,
                        help="Epochs after warmup where fhat is frozen and only V is trained (phase 2a)")
    parser.add_argument("--eval_every",      type=int,   default=100,
                        help="Run evaluation and visualization every N epochs (default: 10)")
    parser.add_argument("--source",          type=str,   default='phys-gmm',
                        choices=['lasa', 'phys-gmm', 'iros'],
                        help="Dataset source: 'phys-gmm' (default), 'lasa', or 'iros'")
    parser.add_argument("--phys_gmm_name",  type=str,   default='2D_messy-snake',
                        help="phys-gmm dataset name, e.g. '2D_snake', '2D_Sshape' (only used with --source phys-gmm)")
    parser.add_argument("--iros_shape",     type=str,   default='OShape',
                        choices=IROS_SHAPES,
                        help="IROS dataset shape name (only used with --source iros)")
    parser.add_argument("--eps",            type=float, default=0.1,
                        help="MakePSD quadratic floor coefficient (smaller → V less quadratic)")
    parser.add_argument("--d",              type=float, default=1e-5,
                        help="ReHU knee point in MakePSD (smaller → ICNN activates closer to origin)")
    parser.add_argument("--limit_cycle",      action="store_true",
                        help="Enable limit-cycle mode: trains phi in Phase 0, then V=phi^2*h(x)")
    parser.add_argument("--phase0_epochs",    type=int,   default=1000,
                        help="Epochs for Phase 0 polar shape fn (only used with --limit_cycle)")
    parser.add_argument("--center_mode",      type=str,   default='endpoint',
                        choices=['endpoint', 'centroid'],
                        help="IROS centering: 'endpoint' (default) or 'centroid' (required for limit_cycle)")
    parser.add_argument("--eps_smooth",       type=float, default=0.05,
                        help="Pseudo-Huber smoothing epsilon for ICNNLimitCycleV")
    parser.add_argument("--levelset_weight",  type=float, default=1.0,
                        help="Weight for level-set alignment loss (ICNN(x_demo) ~= v_gamma)")
    parser.add_argument("--flow_layers",     type=int,   default=4,
                        help="Warp enabled if >0; use 0 for plain ICNN (no radial warp)")
    parser.add_argument("--flow_hidden",     type=int,   default=64,
                        help="n_basis for RadialWarp MonotoneNet (more basis = more expressive s(r))")
    parser.add_argument("--clf_d",           type=float, default=0.1,
                        help="ReHU knee point for CLF projection (smaller → closer to ReLU, stricter guarantee)")
    parser.add_argument("--no_plot",        action="store_true")
    parser.add_argument("--wandb_project",  type=str,   default="stable-nodes",
                        help="W&B project name")
    parser.add_argument("--wandb_run",      type=str,   default=None,
                        help="W&B run name (default: <shape>_<datetime>)")
    parser.add_argument("--no_wandb",       action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--num_threads",    type=int, default=os.cpu_count(),
                        help="PyTorch intra-op thread count (default: all cores)")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(max(1, args.num_threads // 2))

    use_wandb = _WANDB_AVAILABLE and not args.no_wandb

    # Resolve experiment directory
    run_name   = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.source == 'phys-gmm':
        _ds_tag = args.phys_gmm_name
    elif args.source == 'iros':
        _ds_tag = args.iros_shape
    else:
        _ds_tag = args.shape
    if args.logdir is None:
        args.logdir = os.path.join(repo, "logs", "snode", _ds_tag, run_name)
    os.makedirs(args.logdir, exist_ok=True)
    save_path = os.path.join(args.logdir, "snode.pt")
    print(f"Experiment directory: {args.logdir}")

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"{_ds_tag}_{run_name}",
            config=vars(args),
            dir=args.logdir,
        )
        print(f"W&B run: {wandb.run.url}")
    elif not _WANDB_AVAILABLE:
        print("wandb not installed — logging disabled. Run: pip install wandb")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    if args.source == 'phys-gmm':
        print(f"Loading phys-gmm {args.phys_gmm_name} ...")
        data, scale = load_phys_gmm(dataset=args.phys_gmm_name,
                                    subsample=args.subsample)
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}
        print(f"  {data['pos'].shape[0]} trajectories, {data['pos'].shape[1]} time points each")
        print(f"  original trajectory lengths: {data['traj_lengths']}")
        print(f"  mean endpoint after shift: {data['pos'][:, -1, :].mean(dim=0).detach().cpu().numpy()}")
        print(f"  scale={scale:.4f}  (normalized before attractor shift)")
    elif args.source == 'iros':
        center_mode = 'centroid' if args.limit_cycle else args.center_mode
        print(f"Loading IROS {args.iros_shape} (center_mode={center_mode}) ...")
        data, scale = load_iros(shape=args.iros_shape, subsample=args.subsample,
                                center_mode=center_mode)
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}
        print(f"  {data['pos'].shape[0]} demos, {data['pos'].shape[1]} time points each")
        print(f"  mean endpoint after shift: {data['pos'][:, -1, :].mean(dim=0).detach().cpu().numpy()}")
        print(f"  scale={scale:.4f}  (normalized before attractor shift)")
    else:
        print(f"Loading LASA {args.shape} ...")
        data, scale = load_lasa(shape=args.shape, subsample=args.subsample)
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        print(f"  {data['pos'].shape[0]} demos, {data['pos'].shape[1]} time points each")
        print(f"  scale={scale:.2f} mm  (data in [-1, 1])")

    dim = int(data.get('dim', data['pos'].shape[-1]))
    print(f"  state dimension: {dim}D")

    # Train/eval split: first 5 demos for training, remaining for eval
    N_total = data['pos'].shape[0]
    n_train = min(5, N_total)
    def _split(d, start, end):
        return {k: v[start:end] if isinstance(v, torch.Tensor) and v.shape[0] == N_total else v
                for k, v in d.items()}
    if args.limit_cycle:
        data_train = data
        data_eval  = data
        print(f"  train demos: {N_total}  |  eval demos: {N_total}  (limit_cycle: no split)")
    else:
        data_train = _split(data, 0, n_train)
        data_eval  = _split(data, n_train, N_total)
        if data_eval['pos'].shape[0] == 0:
            data_eval = data_train
        print(f"  train demos: {n_train}  |  eval demos: {N_total - n_train}")

    phi = None
    if args.limit_cycle:
        print(f"\nPhase 0 — learning polar shape fn for {args.phase0_epochs} epochs ...")
        phi = PolarLimitCycleShapeFn(hidden=64).to(device)
        train_shape_fn(phi, data, epochs=args.phase0_epochs)
        save_shape_fn_plot(phi, data,
                           save_path=os.path.join(args.logdir, "phase0_phi.png"))
        for p in phi.parameters():
            p.requires_grad_(False)

    dynamics = build_model(hidden_dim=args.hidden_dim, alpha=args.alpha,
                           stability_mode='off', eps=args.eps, d=args.d, dim=dim,
                           limit_cycle=args.limit_cycle,
                           eps_smooth=args.eps_smooth,
                           flow_layers=args.flow_layers,
                           flow_hidden=args.flow_hidden,
                           clf_d=args.clf_d,
                           phi=phi).to(device)
    n_params = sum(p.numel() for p in dynamics.parameters() if p.requires_grad)
    print(f"Model: {n_params} trainable parameters")

    if use_wandb:
        wandb.config.update({"n_params": n_params, "device": str(device)})

    p2a_end = args.warmup_epochs + args.lyapunov_only_epochs
    mode_tag = 'limit_cycle' if args.limit_cycle else 'icnn'
    print(f"\nTraining for {args.epochs} epochs ...")
    print(f"  1–{args.warmup_epochs}: plain NODE  |  "
          f"{args.warmup_epochs+1}–{p2a_end}: {mode_tag} V/phi only (fhat frozen)  |  "
          f"{p2a_end+1}–{args.epochs}: joint fine-tune")
    loss_history = train(
        dynamics, data_train, args.logdir,
        num_epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, pos_weight=args.pos_weight, vel_weight=args.vel_weight,
        use_wandb=use_wandb, icnn_lr_scale=args.icnn_lr_scale,
        eval_every=args.eval_every, lyapunov_only_epochs=args.lyapunov_only_epochs,
        limit_cycle=args.limit_cycle, levelset_weight=args.levelset_weight,
        shape_name=_ds_tag, data_eval=data_eval,
    )

    torch.save({
        'model_state':  dynamics.state_dict(),
        'loss_history': loss_history,
        'scale':        scale,
        'args':         vars(args),
    }, save_path)
    print(f"\nModel saved to {save_path}")

    if not args.no_plot:
        save_final_plot(dynamics, data, data.get('t'), loss_history, save_path)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
