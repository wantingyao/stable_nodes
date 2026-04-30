import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import sys
from torchdiffeq import odeint

# ---------------------------------------------------------------------------
# Rollout utilities — used by train_snode_lasa.py and any other training script
# ---------------------------------------------------------------------------

def make_ode_fn(dynamics):
    """Wrap a Dynamics object so x.requires_grad_(True) is set before each call.
    Required by Dynamics.forward, which calls torch.autograd.grad([V(x)], [x]).
    """
    def ode_fn(t, x):
        x = x.requires_grad_(True)
        return dynamics(x)
    return ode_fn


def rollout(dynamics, x0_batch, t, method='rk4'):
    """Batched ODE rollout.

    Args:
        dynamics  : Dynamics object (or any nn.Module with forward(x) → dx/dt)
        x0_batch  : (N, d) initial states
        t         : (T,)   time grid
        method    : odeint solver string
    Returns:
        x_pred    : (N, T, d) predicted trajectories
    """
    ode_fn = make_ode_fn(dynamics)
    out = odeint(ode_fn, x0_batch, t, method=method)
    # odeint returns (T, N, d) → (N, T, d)
    return out.permute(1, 0, 2)


def rollout_to_convergence(dynamics, x0_batch, t_chunk, threshold=0.001, max_chunks=2):
    """Integrate autonomously in fixed-length chunks until convergence.

    Args:
        dynamics   : Dynamics object
        x0_batch   : (N, d) initial states
        t_chunk    : (T,) one chunk's time array (same dt as training, [0,1])
        threshold  : convergence criterion on ||x||
        max_chunks : safety cap on number of integration chunks
    Returns:
        traj       : (N, T_total, d) full trajectory (all chunks concatenated)
    """
    ode_fn = make_ode_fn(dynamics)
    x_cur  = x0_batch.clone()
    chunks = []

    for _ in range(max_chunks):
        with torch.enable_grad():
            out = odeint(ode_fn, x_cur, t_chunk, method='rk4')
        chunk = out.permute(1, 0, 2).detach()   # (N, T, d)
        chunks.append(chunk)
        x_cur = chunk[:, -1, :].clone()

        if (x_cur.norm(dim=-1) < threshold).all():
            break

    return torch.cat(chunks, dim=1)   # (N, T_total, d)


class MLP(nn.Module):
    '''
    A simple MLP with a variable number of layers and hidden dimensions.
    '''

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation=F.relu):
        super().__init__()
        if num_layers == 1:
            self.fcs = nn.ModuleList([nn.Linear(in_dim, out_dim)])
        else:
            self.fcs = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
            self.fcs.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
            self.fcs.append(nn.Linear(hidden_dim, out_dim))
        self.activation = activation

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            if i == len(self.fcs) - 1:
                x = fc(x)  # Remove the activation function from the last layer
            else:
                x = self.activation(fc(x))
        return x
    

class NODE(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation=F.relu):
        super().__init__()
        self.func = MLP(in_dim, out_dim, hidden_dim, num_layers, activation)

    def forward(self, ts, y0):
        if not torch.is_tensor(ts):
            ts = torch.tensor(ts, dtype=y0.dtype, device=y0.device)
        ts = ts.to(device=y0.device, dtype=y0.dtype)[0]

        if not torch.is_tensor(y0):
            y0 = torch.tensor(y0, dtype=ts.dtype, device=ts.device)
        y0 = y0.to(device=ts.device, dtype=ts.dtype)

        def func_bound(t, y):
            return self.func(y)
        ys = odeint(func_bound, y0, ts, method="rk4")

        return ys.permute(1, 0, 2).contiguous()