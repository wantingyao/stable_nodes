#!/usr/bin/env bash
set -e

SHAPES=(
    Angle BendedLine CShape DoubleBendedLine GShape heee JShape JShape_2
    Khamesh Leaf_1 Leaf_2 Line LShape NShape PShape RShape Saeghe Sharpc
    Sine Snake Spoon Sshape Trapezoid Worm WShape Zshape
    Multi_Models_1 Multi_Models_2 Multi_Models_3 Multi_Models_4
)

for SHAPE in "${SHAPES[@]}"; do
    echo "========================================"
    echo "  Training shape: $SHAPE"
    echo "========================================"
    python scripts/train_snode_lasa.py \
        --shape "$SHAPE" \
        --epochs 7000 \
        --warmup_epochs 1000 \
        --hidden_dim 64 \
        --alpha 0.001 \
        --lr 3e-3 \
        --icnn_lr_scale 0.1 \
        --subsample 5 \
        --pos_weight 1.0 \
        --vel_weight 1.0 \
        --eval_every 100
done
