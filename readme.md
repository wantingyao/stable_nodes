# stable_nodes

## Setup

**1. Create conda environment**

```bash
conda create -n snode python=3.10
conda activate snode
```

**2. Install dependencies**

```bash
pip install torch torchdiffeq tqdm
pip install wandb
```

**3. Initialize submodules**

https://github.com/justagist/pyLasaDataset.git

```bash
git submodule update --init third_party/pyLasaDataset
```

**4. Run training**

```bash
conda activate snode
python scripts/train_snode_lasa.py
```

## Tunable Parameters

Run `python scripts/train_snode_lasa.py` with these arguments:

| Argument | Default | Description |
|---|---|---|
| `--shape` | `Leaf_2` | LASA shape name (e.g. `PShape`, `Angle`, `Sine`) |
| `--subsample` | `1` | Downsample time steps (1 = full ~1000 pts) |
| `--hidden_dim` | `64` | Hidden dim of the Lyapunov network V |
| `--alpha` | `0.001` | Stability rate: `dV/dt ≤ -alpha·V`. Higher → tighter stability |
| `--epochs` | `7000` | Total training epochs |
| `--warmup_epochs` | `1000` | Phase-1 (no projection) length; switches to ICNN mode after |
| `--lr` | `3e-3` | Phase-1 learning rate |
| `--icnn_lr_scale` | `0.1` | Phase-2 LR = `lr × icnn_lr_scale` |
| `--weight_decay` | `0.0` | Adam weight decay |
| `--pos_weight` | `1.0` | Weight on trajectory MSE loss (0 = disable) |
| `--vel_weight` | `1.0` | Weight on velocity MSE loss (0 = disable) |
| `--eval_every` | `100` | Evaluate and save plot every N epochs |

## Structure

```
utils/
  node.py    # Neural ODE wrappers (NODE, SINODETaskEmbedding, MASNODETaskEmbedding)
  lsddm.py   # Lyapunov-stable dynamics (Dynamics, ICNN, MakePSD)
scripts/     # Training scripts
third_party/ # pyLasaDataset、
```
