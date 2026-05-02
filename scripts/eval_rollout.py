#!/usr/bin/env python3
"""
Usage:
    # checkpoint 里有 args（snode.pt 格式）:
    python scripts/eval_rollout.py --checkpoint <path>.pt

    # periodic checkpoint（ckpt_ep*.pt / best_model*.pt），手动指定数据集:
    python scripts/eval_rollout.py --checkpoint <path>.pt \
        --source lasa --shape Leaf_2

    python scripts/eval_rollout.py --checkpoint <path>.pt \
        --source iros --iros_shape IShape --limit_cycle
"""
import sys, os, argparse
import torch

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo, "third_party", "pyLasaDataset"))
sys.path.insert(0, os.path.join(repo, "utils"))
sys.path.insert(0, os.path.join(repo, "src"))

from train_snode_lasa import (
    load_lasa, load_iros, load_phys_gmm,
    evaluate, save_intermediate_plot, IROS_SHAPES,
)
from node import MLP
from lsddm import (
    Dynamics, ICNN, MakePSD, WarpedMakePSD, RadialWarp,
    ICNNLimitCycleV, PhiWarpedICNNV, PolarLimitCycleShapeFn,
)


def build_from_sd(sd, stability_mode, alpha=1.0, eps=0.1, d=1e-5, clf_d=0.1):
    """Reconstruct Dynamics by reading layer sizes directly from state-dict shapes."""

    # fhat
    fw = sorted(k for k in sd if k.startswith("fhat.fcs.") and k.endswith(".weight"))
    fhat = MLP(in_dim=sd[fw[0]].shape[1], out_dim=sd[fw[0]].shape[1],
               hidden_dim=sd[fw[0]].shape[0], num_layers=len(fw))

    # V type detection
    vk = {k.split(".")[1] for k in sd if k.startswith("V.")}

    if "phi" in vk:                          # PhiWarpedICNNV
        phi_h = sd["V.phi.r_net.0.weight"].shape[0]
        phi   = PolarLimitCycleShapeFn(hidden=phi_h)
        icnn  = _icnn(sd, "V.icnn")
        V     = PhiWarpedICNNV(phi, icnn, eps=eps, d=d)

    elif "v_gamma" in vk:                    # ICNNLimitCycleV
        V = ICNNLimitCycleV(_icnn(sd, "V.icnn"), v_gamma_init=1.0, eps_smooth=0.05)

    elif "flow" in vk:                       # WarpedMakePSD
        n_basis = sd["V.flow.s_net.log_w"].shape[0]
        V = WarpedMakePSD(_icnn(sd, "V.f"), RadialWarp(n_basis),
                          n=sd[fw[0]].shape[1], eps=eps, d=d)

    else:                                    # MakePSD
        V = MakePSD(_icnn(sd, "V.f"), n=sd[fw[0]].shape[1], eps=eps, d=d)

    return Dynamics(fhat, V, alpha=alpha, stability_mode=stability_mode, clf_d=clf_d)


def _icnn(sd, prefix):
    wk = sorted(k for k in sd if k.startswith(f"{prefix}.W."))
    sizes = [sd[wk[0]].shape[1]] + [sd[k].shape[0] for k in wk]
    return ICNN(sizes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output",     default=None, help="输出 PNG 路径（默认 checkpoint 同目录）")
    p.add_argument("--no_eval",    action="store_true", help="跳过 evaluate() 指标计算")
    p.add_argument("--device",     default=None)
    # 数据集（有 args 的 checkpoint 自动填充，否则必须手动给）
    p.add_argument("--source",        default=None, choices=["lasa", "phys-gmm", "iros"])
    p.add_argument("--shape",         default=None)
    p.add_argument("--phys_gmm_name", default=None)
    p.add_argument("--iros_shape",    default=None, choices=IROS_SHAPES + [None])
    p.add_argument("--subsample",     type=int, default=None)
    p.add_argument("--center_mode",   default=None, choices=["endpoint", "centroid"])
    p.add_argument("--limit_cycle",   action="store_true", default=False)
    # 标量超参（不在 state dict 里，有 args 自动填，否则用默认值）
    p.add_argument("--alpha",  type=float, default=None)
    p.add_argument("--eps",    type=float, default=None)
    p.add_argument("--d",      type=float, default=None)
    p.add_argument("--clf_d",  type=float, default=None)
    p.add_argument("--stability_mode", default=None,
                   choices=["off", "icnn", "limit_cycle"])
    args = p.parse_args()

    # ---------- 加载 checkpoint ----------
    ckpt_path = os.path.abspath(args.checkpoint)
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 用 checkpoint 里的 args 填充未指定的参数
    saved = ckpt.get("args", {})
    def _get(attr, default):
        cli_val = getattr(args, attr, None)
        if cli_val is not None and cli_val is not False:
            return cli_val
        return saved.get(attr, default)

    source       = _get("source", "lasa")
    shape        = _get("shape", "Leaf_2")
    phys_name    = _get("phys_gmm_name", "2D_messy-snake")
    iros_shape   = _get("iros_shape", "IShape")
    subsample    = _get("subsample", 1) or 1
    limit_cycle  = _get("limit_cycle", False)
    center_mode  = _get("center_mode", "centroid" if limit_cycle else "endpoint")
    alpha        = _get("alpha", 1.0) or 1.0
    eps          = _get("eps",   0.1) or 0.1
    d            = _get("d",     1e-5) or 1e-5
    clf_d        = _get("clf_d", 0.1) or 0.1

    # stability_mode: CLI > checkpoint key > limit_cycle 推断 > 默认 icnn
    stab = args.stability_mode or ckpt.get("stability_mode")
    if stab is None:
        stab = "limit_cycle" if limit_cycle else "icnn"
    print(f"  stability_mode = {stab}")

    # ---------- 设备 ----------
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"  device = {device}")

    # ---------- 数据集 ----------
    if source == "phys-gmm":
        data, scale = load_phys_gmm(dataset=phys_name, subsample=subsample)
    elif source == "iros":
        data, scale = load_iros(shape=iros_shape, subsample=subsample, center_mode=center_mode)
    else:
        data, scale = load_lasa(shape=shape, subsample=subsample)
    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    N, T, dim = data["pos"].shape
    print(f"  dataset: {N} demos × {T} steps × {dim}D")

    # ---------- 构建网络 + 加载权重 ----------
    sd = ckpt["model_state"]
    dynamics = build_from_sd(sd, stab, alpha=alpha, eps=eps, d=d, clf_d=clf_d).to(device)
    dynamics.load_state_dict(sd)
    dynamics.eval()
    print(f"  model: {type(dynamics.V).__name__}  "
          f"({sum(p.numel() for p in dynamics.parameters())} params)")

    # ---------- 指标 ----------
    if not args.no_eval:
        print("Evaluating ...")
        m = evaluate(dynamics, data)
        print(f"  RMSE_vel={m['rmse_vel']:.6f}  MVD={m['mvd']:.6f}  "
              f"DTWD={m['dtwd']:.4f}  ConvMSE={m['conv_mse']:.6f}")
        if "metrics" in ckpt:
            ref = ckpt["metrics"]
            r_dtwd = ref.get("dtwd")
            r_conv = ref.get("conv_mse")
            if r_dtwd is not None:
                print(f"  (ckpt ref: DTWD={r_dtwd:.4f}  ConvMSE={r_conv:.4f})")
        epoch = ckpt.get("epoch", 0)
        dtwd  = m["dtwd"]
        conv  = m["conv_mse"]
    else:
        epoch = ckpt.get("epoch", 0)
        dtwd  = conv = None

    # ---------- 可视化（完全复用 training 里的 save_intermediate_plot）----------
    # shape_name: 优先从 checkpoint args 取，其次从 CLI 推断
    shape_name = (saved.get("shape") or saved.get("iros_shape") or
                  saved.get("phys_gmm_name") or
                  getattr(args, "shape", None) or
                  getattr(args, "iros_shape", None) or
                  getattr(args, "phys_gmm_name", None) or "")
    logdir = args.output or os.path.dirname(ckpt_path)
    save_intermediate_plot(dynamics, data, data["t"], epoch, logdir,
                           dtwd=dtwd, conv_mse=conv, shape_name=shape_name)
    print(f"Plot saved to {os.path.join(logdir, 'vis', f'epoch_{epoch:04d}*.png')}")


if __name__ == "__main__":
    main()
