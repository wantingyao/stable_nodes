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
pip install wandb  # optional, for logging
```

**3. Initialize submodules**

```bash
git submodule update --init third_party/pyLasaDataset
```

**4. Run training**

```bash
conda activate snode
python scripts/train_snode_lasa.py
```

## Structure

```
utils/
  node.py    # Neural ODE wrappers (NODE, SINODETaskEmbedding, MASNODETaskEmbedding)
  lsddm.py   # Lyapunov-stable dynamics (Dynamics, ICNN, MakePSD)
scripts/     # Training scripts
third_party/ # pyLasaDataset, hyper-node
```
