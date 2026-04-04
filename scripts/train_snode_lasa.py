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

def load_lasa(shape='Leaf_2'):
    """
    Returns a batch dict with all demos stacked (all 1000 points, no subsampling),
    plus a shared time grid from demo_0.

    Args:
        shape: LASA dataset shape name, e.g. 'Leaf_2', 'PShape', 'Angle', etc.

    Returns:
        data: dict with keys
            'pos'  : (N, 1000, 2) float32 tensor  — normalized positions
            'x0'   : (N, 2)       float32 tensor  — initial states
            't'    : (1000,)      float32 tensor  — shared normalized time
        scale: float  — mm value corresponding to 1 in normalized space
    """
    import pyLasaDataset as lasa

    leaf2 = getattr(lasa.DataSet, shape)

    # Scale normalization: divide by global max abs value → data in [-1, 1]²
    all_pos = np.concatenate([d.pos for d in leaf2.demos], axis=1)  # (2, 7000)
    scale   = np.abs(all_pos).max()

    # Shared time grid from demo_0 (same approach as reference algo_testnode.py)
    t_ref  = leaf2.demos[0].t[0]                           # (1000,)
    t_norm = (t_ref - t_ref[0]) / (t_ref[-1] - t_ref[0])  # [0, 1]

    pos_list = []
    for demo in leaf2.demos:
        pos_norm = demo.pos / scale   # (2, 1000)
        pos_list.append(pos_norm.T)   # (1000, 2)

    pos_batch = np.stack(pos_list, axis=0)  # (7, 1000, 2)

    data = {
        'pos': torch.tensor(pos_batch,          dtype=torch.float32),  # (7, 1000, 2)
        'x0':  torch.tensor(pos_batch[:, 0, :], dtype=torch.float32),  # (7, 2)
        't':   torch.tensor(t_norm,             dtype=torch.float32),  # (1000,)
    }
    return data, scale


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(hidden_dim=64, alpha=0.1, stability_mode='off'):
    fhat     = MLP(in_dim=2, out_dim=2, hidden_dim=256, num_layers=5)
    icnn     = ICNN([2, hidden_dim, hidden_dim, 1])
    V        = MakePSD(icnn, n=2, eps=0.01, d=1.0)
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
                  color='mediumorchid', linewidth=0.6, arrowsize=0.7, density=1.5)
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.4)
    ax.axvline(0, color='k', linewidth=0.4, alpha=0.4)
    ax.set_xlim(x_np[0], x_np[-1]); ax.set_ylim(y_np[0], y_np[-1])
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")


def _draw_lyapunov(fig, ax, x_np, y_np, V_lyap, vmax_pct=70):
    """Lyapunov contour with viridis colormap.
    vmax_pct: colormap saturates above this percentile of V values,
              giving finer color resolution near the origin (low V).
    """
    vmax = np.percentile(V_lyap, vmax_pct)
    cf = ax.contourf(x_np, y_np, V_lyap, levels=80, cmap='viridis', alpha=0.80,
                     vmin=0.0, vmax=vmax)
    fig.colorbar(cf, ax=ax, label='V(x)', extend='max')
    ax.contour(x_np, y_np, V_lyap, levels=12, colors='white', linewidths=0.4, alpha=0.4)
    ax.axhline(0, color='w', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='w', linewidth=0.5, alpha=0.5)
    # Mark V=0 at origin (guaranteed by MakePSD construction)
    ax.plot(0.0, 0.0, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1.0,
            zorder=6, label='V=0')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(x_np[0], x_np[-1]); ax.set_ylim(y_np[0], y_np[-1])
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")


def _plot_traj_panel(ax, data, dynamics, t_eval, title="", perturb=0.1,
                     grid_data=None):
    """
    Reference-style panel (mirrors PShape_2.png):
      - Violet streamplot as background (if grid_data provided)
      - Ground truth demos in dodgerblue
      - Model rollouts in crimson (solid from original x0, dashed from perturbed)
      - Start: large black filled circle; Target: large black X
    """
    pos_gt   = data['pos']   # (7, T, 2)
    x0_batch = data['x0']    # (7, 2)
    target   = pos_gt[0, -1] # last point of demo_0 ≈ origin

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

    # Model rollouts from original x0 — integrate until convergence
    x_pred_np = rollout_to_convergence(dynamics, x0_batch, t_eval).cpu().numpy()
    for i in range(x_pred_np.shape[0]):
        ax.plot(x_pred_np[i, :, 0], x_pred_np[i, :, 1], c='crimson',
                linewidth=1.8, zorder=4, label='Model' if i == 0 else None)

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

    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend(loc='lower left', fontsize=8)


def save_intermediate_plot(dynamics, data, t_eval, epoch, logdir, use_wandb=False):
    """Save reference-style plot every 100 epochs.
    Post-warmup: 1x2 (trajectory | Lyapunov). Pre-warmup: single trajectory panel.
    """
    import matplotlib.pyplot as plt
    dynamics.eval()

    x_np, y_np, U, V_f, V_lyap = _compute_grid(dynamics)

    if dynamics.stability_mode == 'icnn':
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        _plot_traj_panel(axes[0], data, dynamics, t_eval,
                         title=f"Learned vector field — Epoch {epoch}",
                         grid_data=(x_np, y_np, U, V_f))
        _draw_lyapunov(fig, axes[1], x_np, y_np, V_lyap)
        axes[1].set_title(f"Lyapunov V(x) — Epoch {epoch}")
    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        _plot_traj_panel(ax, data, dynamics, t_eval,
                         title=f"Learned vector field — Epoch {epoch}",
                         grid_data=(x_np, y_np, U, V_f))
    fig.tight_layout()
    vis_dir = os.path.join(logdir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    fig_path = os.path.join(vis_dir, f"epoch_{epoch:04d}.png")
    fig.savefig(fig_path, dpi=120)
    if use_wandb and _WANDB_AVAILABLE:
        wandb.log({"vis/trajectory": wandb.Image(fig_path)}, step=epoch)
    plt.close(fig)
    dynamics.train()



def save_final_plot(dynamics, data, t_eval, loss_history, save_path):
    """1x2: streamplot+trajectories | Lyapunov contour."""
    import matplotlib.pyplot as plt
    dynamics.eval()

    x_np, y_np, U, V_f, V_lyap = _compute_grid(dynamics)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Panel 1: streamplot + trajectories (reference style)
    _plot_traj_panel(axes[0], data, dynamics, t_eval,
                     title="Learned vector field",
                     grid_data=(x_np, y_np, U, V_f))

    # Panel 2: Lyapunov contour only
    _draw_lyapunov(fig, axes[1], x_np, y_np, V_lyap)
    axes[1].set_title("Lyapunov V(x)")

    fig.tight_layout()
    fig_path = save_path.replace(".pt", "_results.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Plot saved to {fig_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(dynamics, data, logdir, num_epochs=500, lr=1e-3, weight_decay=1e-4,
          warmup_epochs=200, use_wandb=False):
    """Two-phase training:
      epoch 1 .. warmup_epochs     : stability_mode='off'  (plain NODE)
      epoch warmup_epochs+1 .. end : stability_mode='icnn' (learned V)
    """
    x0_batch = data['x0']   # (7, 2)
    pos_gt   = data['pos']  # (7, T_sub, 2)
    t        = data['t']    # (T_sub,)

    optimizer = torch.optim.Adam(
        dynamics.parameters(), lr=lr, weight_decay=weight_decay
    )
    # CosineAnnealingLR, mirroring algo_testnode.py
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.1
    )

    loss_history = []
    pbar = tqdm(range(1, num_epochs + 1), desc="Training", dynamic_ncols=True)

    for epoch in pbar:
        # Phase 1 → 2: switch to ICNN projection after warm-up
        if epoch == warmup_epochs + 1:
            dynamics.stability_mode = 'icnn'
            dynamics.V.invalidate_zero_cache()
            tqdm.write(f"\n[Epoch {epoch}] Stability ON — mode=icnn")

        dynamics.train()
        optimizer.zero_grad()

        # Batched rollout over all 7 demos in one odeint call
        x_pred = rollout(dynamics, x0_batch, t)   # (7, T_sub, 2)
        loss   = F.mse_loss(x_pred, pos_gt)

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
            wandb.log({"train/loss": loss.item(), "train/lr": cur_lr,
                       "train/stability_mode": mode_tag}, step=epoch)

        # Save intermediate visualization every 100 epochs and at the final epoch
        if epoch % 100 == 0 or epoch == num_epochs:
            save_intermediate_plot(dynamics, data, t, epoch, logdir, use_wandb=use_wandb)

    return loss_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Stable NODE on LASA Leaf_2")
    parser.add_argument("--hidden_dim",    type=int,   default=64)
    parser.add_argument("--alpha",         type=float, default=0.1)
    parser.add_argument("--epochs",        type=int,   default=500)
    parser.add_argument("--lr",            type=float, default=3e-3)
    parser.add_argument("--weight_decay",  type=float, default=0.0)
    parser.add_argument("--shape",         type=str,   default='PShape',
                        help="LASA shape name, e.g. Leaf_2, PShape, Angle, Sine ...")
    parser.add_argument("--logdir",        type=str,   default=None,
                        help="Override experiment dir (default: logs/snode/<shape>/<datetime>)")
    parser.add_argument("--warmup_epochs",  type=int,   default=200,
                        help="Epochs with stability off (plain NODE warm-up), then ICNN projection")
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Loading LASA {args.shape} ...")
    data, scale = load_lasa(shape=args.shape)
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
        warmup_epochs=args.warmup_epochs,
        use_wandb=use_wandb,
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
