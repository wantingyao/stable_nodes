import sys
import os
import argparse
from datetime import datetime

import numpy as np
import torch
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
from lsddm import Dynamics, ICNN, MakePSD


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

    # Step 2: subtract attractor (mean of last positions across demos, in normalized space)
    # so that the attractor lands exactly at the origin where V=0
    attractor = pos_batch[:, -1, :].mean(axis=0)  # (2,)
    pos_batch  = pos_batch - attractor[None, None, :]

    data = {
        'pos':       torch.tensor(pos_batch,          dtype=torch.float32),  # (7, 1000, 2)
        'x0':        torch.tensor(pos_batch[:, 0, :], dtype=torch.float32),  # (7, 2)
        't':         torch.tensor(t_norm,             dtype=torch.float32),  # (1000,)
        'attractor': torch.tensor(attractor,          dtype=torch.float32),  # (2,)
    }
    return data, scale


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(hidden_dim=64, alpha=0.1, stability_mode='off'):
    fhat     = MLP(in_dim=2, out_dim=2, hidden_dim=256, num_layers=5)
    icnn     = ICNN([2, hidden_dim, hidden_dim, 1])
    V        = MakePSD(icnn, n=2, eps=1.0, d=1.0)
    dynamics = Dynamics(fhat, V, alpha=alpha, stability_mode=stability_mode)
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
        vel    = dynamics(grid_g).detach()
        V_vals = dynamics.V(grid_g).detach()

    U      = vel[:, 0].reshape(N, N).cpu().numpy()
    V_f    = vel[:, 1].reshape(N, N).cpu().numpy()
    V_lyap = V_vals[:, 0].reshape(N, N).cpu().numpy()
    return x_lin.numpy(), y_lin.numpy(), U, V_f, V_lyap


def _draw_streamplot(ax, x_np, y_np, U, V_f):
    """Violet streamplot matching PShape_2.png reference style."""
    ax.streamplot(x_np, y_np, U, V_f,
                  color='mediumorchid', linewidth=0.6, arrowsize=0.7, density=2.0)
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.4)
    ax.axvline(0, color='k', linewidth=0.4, alpha=0.4)
    ax.set_xlim(x_np[0], x_np[-1]); ax.set_ylim(y_np[0], y_np[-1])
    ax.set_aspect('equal'); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")


def _draw_lyapunov_cf(ax, x_np, y_np, V_lyap, gamma=0.4):
    """Draw Lyapunov contourf onto ax; return cf for colorbar attachment."""
    from matplotlib.colors import PowerNorm
    vmax = V_lyap.max()
    levels = np.concatenate([[0.0], np.geomspace(vmax * 1e-3, vmax, 79)])
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)
    cf = ax.contourf(x_np, y_np, V_lyap, levels=levels, cmap='viridis', alpha=0.80, norm=norm)
    ax.contour(x_np, y_np, V_lyap, levels=levels[::8], colors='white', linewidths=0.4, alpha=0.4)
    ax.axhline(0, color='w', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='w', linewidth=0.5, alpha=0.5)
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
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal'); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")


def _plot_traj_panel(ax, data, dynamics, t_eval, title="", perturb=0.1,
                     grid_data=None, probe_grid_n=8, probe_lim=1.2,
                     lyap_data=None, fig=None, show_probe=False):
    """
    Reference-style panel:
      - Optional Lyapunov contourf as bottom layer (if lyap_data provided)
      - Violet streamplot on top (if grid_data provided)
      - Ground truth demos in dodgerblue
      - Model rollouts in crimson (solid from original x0, dashed from perturbed)
      - Start: large black filled circle; Target: large black X
    """
    pos_gt   = data['pos']   # (7, T, 2)
    x0_batch = data['x0']    # (7, 2)
    target   = pos_gt[0, -1] # last point of demo_0 ≈ origin

    # Lyapunov contourf as bottom layer (identical appearance to standalone panel)
    if lyap_data is not None:
        x_np_l, y_np_l, V_lyap = lyap_data
        cf = _draw_lyapunov_cf(ax, x_np_l, y_np_l, V_lyap)
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

    # Start points (large black filled circle, like reference)
    for i in range(pos_gt.shape[0]):
        ax.plot(pos_gt[i, 0, 0].item(), pos_gt[i, 0, 1].item(),
                marker='o', color='black', markersize=10,
                markeredgecolor='black', zorder=5,
                label='Start' if i == 0 else None)

    # Target (large black X, like reference)
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

    # Model rollouts from original x0 — integrate until convergence
    x_pred_np = rollout_to_convergence(dynamics, x0_batch, t_eval).cpu().numpy()
    for i in range(x_pred_np.shape[0]):
        ax.plot(x_pred_np[i, :, 0], x_pred_np[i, :, 1], c='crimson',
                linewidth=1.8, zorder=4, label='Model' if i == 0 else None)
    _mark_endpoints(x_pred_np, 'crimson', markersize=7)

    # Model rollouts from perturbed x0 (dashed, lighter)
    offsets = [np.array([perturb, 0.0]), np.array([0.0, perturb]),
               np.array([-perturb, 0.0]), np.array([0.0, -perturb])]
    _perturb_labeled = False
    for offset in offsets:
        x0_p = (x0_batch + torch.tensor(offset, dtype=torch.float32,
                                         device=x0_batch.device)).clamp(-1.2, 1.2)
        x_p_np = rollout_to_convergence(dynamics, x0_p, t_eval).cpu().numpy()
        for i in range(x_p_np.shape[0]):
            label = 'Perturbed' if not _perturb_labeled else None
            _perturb_labeled = True
            ax.plot(x_p_np[i, :, 0], x_p_np[i, :, 1], c='crimson',
                    linewidth=0.8, linestyle='--', alpha=0.45, zorder=4, label=label)
        _mark_endpoints(x_p_np, 'crimson', markersize=4, alpha=0.6)

    # Probe rollouts from a uniform grid to reveal spurious attractors
    if show_probe:
        xs = np.linspace(-probe_lim, probe_lim, probe_grid_n)
        probe_x0 = torch.tensor(
            np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2),
            dtype=torch.float32, device=x0_batch.device,
        )  # (probe_grid_n², 2)
        probe_np = rollout_to_convergence(dynamics, probe_x0, t_eval).cpu().numpy()
        for i in range(probe_np.shape[0]):
            ax.plot(probe_np[i, :, 0], probe_np[i, :, 1], c='darkorange',
                    linewidth=0.6, alpha=0.35, zorder=2,
                    label='Probe' if i == 0 else None)
        _mark_endpoints(probe_np, 'darkorange', markersize=4, alpha=0.6)

    ax.set_aspect('equal'); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend(loc='lower left', fontsize=8)


def _make_figure(dynamics, data, t_eval, title_suffix, fig_kw):
    """Shared 1×3 layout (or 1×1 pre-warmup): vector field | Lyapunov | combined."""
    import matplotlib.pyplot as plt

    x_np, y_np, U, V_f, V_lyap = _compute_grid(dynamics)
    grid_data = (x_np, y_np, U, V_f)
    lyap_data  = (x_np, y_np, V_lyap)

    if dynamics.stability_mode != 'icnn':
        fig, ax = plt.subplots(figsize=(7, 7), **fig_kw)
        _plot_traj_panel(ax, data, dynamics, t_eval,
                         title=f"Vector field {title_suffix}",
                         grid_data=grid_data)
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), **fig_kw)

    # Panel 1: vector field + rollouts only
    _plot_traj_panel(axes[0], data, dynamics, t_eval,
                     title=f"Vector field {title_suffix}",
                     grid_data=grid_data)

    # Panel 2: Lyapunov only
    _draw_lyapunov(fig, axes[1], x_np, y_np, V_lyap)
    axes[1].set_title(f"Lyapunov V(x) {title_suffix}")

    # Panel 3: combined overlay
    _plot_traj_panel(axes[2], data, dynamics, t_eval,
                     title=f"Combined {title_suffix}",
                     grid_data=grid_data,
                     lyap_data=lyap_data, fig=fig)

    fig.tight_layout()
    return fig


def save_intermediate_plot(dynamics, data, t_eval, epoch, logdir, dtwd=None, use_wandb=False):
    """Save 1×3 plot every eval_every epochs."""
    import matplotlib.pyplot as plt
    dynamics.eval()

    dtwd_tag = f"_dtwd_{dtwd:.4f}" if dtwd is not None else ""
    fig = _make_figure(dynamics, data, t_eval,
                       title_suffix=f"— Epoch {epoch}  [{dynamics.stability_mode}]"
                                    + (f"  DTWD={dtwd:.4f}" if dtwd is not None else ""),
                       fig_kw={})
    fig.tight_layout()
    vis_dir = os.path.join(logdir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    fig_path = os.path.join(vis_dir, f"epoch_{epoch:04d}{dtwd_tag}.png")
    fig.savefig(fig_path, dpi=120)
    if use_wandb and _WANDB_AVAILABLE:
        wandb.log({"vis/trajectory": wandb.Image(fig_path)}, step=epoch)
    plt.close(fig)
    dynamics.train()



def save_final_plot(dynamics, data, t_eval, loss_history, save_path):
    """1×3: vector field | Lyapunov | combined."""
    import matplotlib.pyplot as plt
    dynamics.eval()

    fig = _make_figure(dynamics, data, t_eval,
                       title_suffix="", fig_kw={})
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

    # --- velocity metrics (at GT positions) ---
    dt       = t[1:] - t[:-1]                                           # (T-1,)
    vel_gt   = (pos_gt[:, 1:, :] - pos_gt[:, :-1, :]) / dt[None, :, None]  # (N, T-1, 2)
    x_flat   = pos_gt[:, :-1, :].reshape(-1, d).requires_grad_(True)
    with torch.enable_grad():
        vel_pred = dynamics(x_flat).reshape(N, T - 1, d)                # (N, T-1, 2)

    err      = (vel_pred - vel_gt).norm(dim=-1)   # (N, T-1)
    mvd      = err.mean().item()
    rmse_vel = err.pow(2).mean().sqrt().item()

    # --- DTWD (rollout vs GT) ---
    x0_batch = data['x0']                                               # (N, 2)
    with torch.enable_grad():
        x_pred = rollout(dynamics, x0_batch, t, method=solver).detach() # (N, T, 2)
    dtwd_vals = []
    for i in range(N):
        p = x_pred[i].cpu().numpy()
        q = pos_gt[i].cpu().numpy()
        dtwd_vals.append(_dtw_distance(p, q))
    dtwd = float(np.mean(dtwd_vals))

    return {'rmse_vel': rmse_vel, 'mvd': mvd, 'dtwd': dtwd}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(dynamics, data, logdir, num_epochs=500, lr=1e-3, weight_decay=1e-4,
          warmup_epochs=200, pos_weight=1.0, vel_weight=1.0, use_wandb=False,
          icnn_lr_scale=0.1, eval_every=100):
    """Two-phase training:
      epoch 1 .. warmup_epochs     : stability_mode='off'  (plain NODE, lr)
      epoch warmup_epochs+1 .. end : stability_mode='icnn' (all params, lr * icnn_lr_scale)
    """
    x0_batch = data['x0']   # (7, 2)
    pos_gt   = data['pos']  # (7, T_sub, 2)
    t        = data['t']    # (T_sub,)

    optimizer = torch.optim.Adam(
        dynamics.parameters(), lr=lr, weight_decay=weight_decay
    )
    # CosineAnnealingLR, mirroring algo_testnode.py
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=warmup_epochs, eta_min=lr * 0.1
    )

    loss_history = []
    best_dtwd    = float('inf')
    best_path    = None
    solver       = 'dopri5'   # phase 1: adaptive; switches to rk4 at phase 2
    pbar = tqdm(range(1, num_epochs + 1), desc="Training", dynamic_ncols=True)

    for epoch in pbar:
        # Phase 1 → 2: switch to ICNN projection after warm-up
        if epoch == warmup_epochs + 1:
            dynamics.stability_mode = 'icnn'
            dynamics.V.invalidate_zero_cache()
            phase2_lr = lr * icnn_lr_scale
            optimizer = torch.optim.Adam(
                dynamics.parameters(), lr=phase2_lr, weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - warmup_epochs, eta_min=phase2_lr * 0.1
            )
            solver = 'rk4'
            tqdm.write(f"\n[Epoch {epoch}] Stability ON — mode=icnn, lr={phase2_lr:.2e}, solver=rk4")

        dynamics.train()
        optimizer.zero_grad()

        # Batched rollout over all 7 demos in one odeint call (skip if pos_weight=0)
        if pos_weight > 0.0:
            x_pred   = rollout(dynamics, x0_batch, t, method=solver)   # (7, T_sub, 2)
            loss_pos = F.mse_loss(x_pred, pos_gt)
        else:
            loss_pos = torch.tensor(0.0, device=x0_batch.device)

        # Velocity loss: single-step f(x) vs finite-difference GT velocity
        dt       = t[1:] - t[:-1]                               # (T_sub-1,)
        vel_gt   = (pos_gt[:, 1:, :] - pos_gt[:, :-1, :]) / dt[None, :, None]  # (7, T_sub-1, 2)
        N, _, d  = pos_gt.shape
        x_flat   = pos_gt[:, :-1, :].reshape(-1, d).requires_grad_(True)       # (7*(T_sub-1), 2)
        vel_pred = dynamics(x_flat).reshape(N, -1, d)           # (7, T_sub-1, 2)
        loss_vel = F.mse_loss(vel_pred, vel_gt)

        loss = pos_weight * loss_pos + vel_weight * loss_vel

        loss.backward()
        clip_grad_norm_(dynamics.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        # Cache invalidation only needed when ICNN V is active
        if dynamics.stability_mode == 'icnn':
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
            metrics = evaluate(dynamics, data, solver=solver)
            tqdm.write(
                f"[Epoch {epoch:>5}]  RMSE_vel={metrics['rmse_vel']:.6f}"
                f"  MVD={metrics['mvd']:.6f}  DTWD={metrics['dtwd']:.4f}"
            )
            if use_wandb and _WANDB_AVAILABLE:
                wandb.log({
                    "eval/rmse_vel": metrics['rmse_vel'],
                    "eval/mvd":      metrics['mvd'],
                    "eval/dtwd":     metrics['dtwd'],
                }, step=epoch)

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
                if epochs_since_warmup % 200 == 0 or epoch == num_epochs:
                    ckpt_path = os.path.join(
                        logdir, f"ckpt_ep{epoch:04d}_dtwd_{metrics['dtwd']:.4f}.pt"
                    )
                    torch.save(ckpt, ckpt_path)
                    tqdm.write(f"  → {os.path.basename(ckpt_path)} saved")

                # Save best model
                if metrics['dtwd'] < best_dtwd:
                    best_dtwd = metrics['dtwd']
                    if best_path is not None and os.path.exists(best_path):
                        os.remove(best_path)
                    best_path = os.path.join(
                        logdir, f"best_model_ep{epoch:04d}_dtwd_{best_dtwd:.4f}.pt"
                    )
                    torch.save(ckpt, best_path)
                    tqdm.write(f"  → {os.path.basename(best_path)} saved (best)")

            save_intermediate_plot(dynamics, data, t, epoch, logdir,
                                   dtwd=metrics['dtwd'], use_wandb=use_wandb)

    return loss_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Stable NODE on LASA Leaf_2")
    parser.add_argument("--hidden_dim",    type=int,   default=64)
    parser.add_argument("--alpha",         type=float, default=0.001) # Higher alpha → stronger stability regularization (V(x) dominates over fhat(x))
    parser.add_argument("--epochs",        type=int,   default=7000)
    parser.add_argument("--lr",            type=float, default=3e-3)
    parser.add_argument("--weight_decay",  type=float, default=0.0)
    parser.add_argument("--shape",         type=str,   default='Leaf_2',
                        help="LASA shape name, e.g. Leaf_2, PShape, Angle, Sine ...")
    parser.add_argument("--subsample",     type=int,   default=1,
                        help="Take every N-th time step (1=no subsampling, 5=200pts)")
    parser.add_argument("--pos_weight",    type=float, default=1.0,
                        help="Weight for position rollout loss term (0=disable)")
    parser.add_argument("--vel_weight",    type=float, default=1.0,
                        help="Weight for velocity loss term (0=disable)")
    parser.add_argument("--logdir",        type=str,   default=None,
                        help="Override experiment dir (default: logs/snode/<shape>/<datetime>)")
    parser.add_argument("--warmup_epochs",  type=int,   default=1000,
                        help="Epochs with stability off (plain NODE warm-up), then ICNN projection")
    parser.add_argument("--icnn_lr_scale",   type=float, default=0.1,
                        help="LR multiplier for phase 2 (icnn), applied to both fhat and V")
    parser.add_argument("--eval_every",      type=int,   default=100,
                        help="Run evaluation every N epochs (default: 100)")
    parser.add_argument("--no_plot",        action="store_true")
    parser.add_argument("--wandb_project",  type=str,   default="stable-nodes",
                        help="W&B project name")
    parser.add_argument("--wandb_run",      type=str,   default=None,
                        help="W&B run name (default: <shape>_<datetime>)")
    parser.add_argument("--no_wandb",       action="store_true",
                        help="Disable W&B logging")
    args = parser.parse_args()

    use_wandb = _WANDB_AVAILABLE and not args.no_wandb

    # Resolve experiment directory
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.logdir is None:
        args.logdir = os.path.join(repo, "logs", "snode", args.shape, run_name)
    os.makedirs(args.logdir, exist_ok=True)
    save_path = os.path.join(args.logdir, "snode_lasa.pt")
    print(f"Experiment directory: {args.logdir}")

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"{args.shape}_{run_name}",
            config=vars(args),
            dir=args.logdir,
        )
        print(f"W&B run: {wandb.run.url}")
    elif not _WANDB_AVAILABLE:
        print("wandb not installed — logging disabled. Run: pip install wandb")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Loading LASA {args.shape} ...")
    data, scale = load_lasa(shape=args.shape, subsample=args.subsample)
    data = {k: v.to(device) for k, v in data.items()}
    print(f"  {data['pos'].shape[0]} demos, {data['pos'].shape[1]} time points each")
    print(f"  scale={scale:.2f} mm  (data in [-1, 1])")

    dynamics = build_model(hidden_dim=args.hidden_dim, alpha=args.alpha,
                           stability_mode='off').to(device)
    n_params = sum(p.numel() for p in dynamics.parameters() if p.requires_grad)
    print(f"Model: {n_params} trainable parameters")

    if use_wandb:
        wandb.config.update({"n_params": n_params, "device": str(device)})

    print(f"\nTraining for {args.epochs} epochs ...")
    print(f"  0–{args.warmup_epochs}: plain NODE  |  "
          f"{args.warmup_epochs+1}–{args.epochs}: ICNN V")
    loss_history = train(
        dynamics, data, args.logdir,
        num_epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, pos_weight=args.pos_weight, vel_weight=args.vel_weight,
        use_wandb=use_wandb, icnn_lr_scale=args.icnn_lr_scale,
        eval_every=args.eval_every,
    )

    torch.save({
        'model_state':  dynamics.state_dict(),
        'loss_history': loss_history,
        'scale':        scale,
        'args':         vars(args),
    }, save_path)
    print(f"\nModel saved to {save_path}")

    if not args.no_plot:
        save_final_plot(dynamics, data, data['t'], loss_history, save_path)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
