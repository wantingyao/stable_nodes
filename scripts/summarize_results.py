import sys
import os
import argparse
import csv
import glob

import torch

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def find_best_model(shape_dir):
    """Return (run_dir, best_model_path) for the latest run under shape_dir, or (None, None)."""
    run_dirs = sorted(
        [d for d in glob.glob(os.path.join(shape_dir, "*")) if os.path.isdir(d)],
        key=os.path.getmtime,
    )
    if not run_dirs:
        return None, None
    latest = run_dirs[-1]
    candidates = glob.glob(os.path.join(latest, "best_model_*.pt"))
    if not candidates:
        return latest, None
    return latest, candidates[0]


def load_metrics(pt_path):
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", {})
    epoch = ckpt.get("epoch", -1)
    return epoch, metrics.get("rmse_vel"), metrics.get("mvd"), metrics.get("dtwd")


def main():
    parser = argparse.ArgumentParser(description="Summarize best SNODE results across LASA shapes")
    parser.add_argument("--logdir", type=str, default=os.path.join(repo, "logs", "snode"),
                        help="Root log directory (default: logs/snode/ in repo root)")
    args = parser.parse_args()

    logdir = args.logdir
    if not os.path.isdir(logdir):
        print(f"Log directory not found: {logdir}")
        sys.exit(1)

    shape_dirs = sorted(
        [d for d in glob.glob(os.path.join(logdir, "*")) if os.path.isdir(d)
         and os.path.basename(d) != "__pycache__"]
    )

    rows = []
    for shape_dir in shape_dirs:
        shape = os.path.basename(shape_dir)
        _, best_path = find_best_model(shape_dir)
        if best_path is None:
            rows.append({"shape": shape, "epoch": "MISSING", "rmse_vel": "MISSING",
                         "mvd": "MISSING", "dtwd": "MISSING"})
        else:
            epoch, rmse_vel, mvd, dtwd = load_metrics(best_path)
            rows.append({"shape": shape, "epoch": epoch, "rmse_vel": rmse_vel,
                         "mvd": mvd, "dtwd": dtwd})

    col_w = [20, 8, 12, 12, 12]
    header = f"{'Shape':<{col_w[0]}} {'Epoch':>{col_w[1]}} {'RMSE_vel':>{col_w[2]}} {'MVD':>{col_w[3]}} {'DTWD':>{col_w[4]}}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    numeric_rows = []
    for row in rows:
        if row["rmse_vel"] == "MISSING":
            print(f"{row['shape']:<{col_w[0]}} {'MISSING':>{col_w[1]}} {'MISSING':>{col_w[2]}} {'MISSING':>{col_w[3]}} {'MISSING':>{col_w[4]}}")
        else:
            print(f"{row['shape']:<{col_w[0]}} {row['epoch']:>{col_w[1]}} {row['rmse_vel']:>{col_w[2]}.6f} {row['mvd']:>{col_w[3]}.6f} {row['dtwd']:>{col_w[4]}.4f}")
            numeric_rows.append(row)

    if numeric_rows:
        print(sep)
        mean_rmse = sum(r["rmse_vel"] for r in numeric_rows) / len(numeric_rows)
        mean_mvd  = sum(r["mvd"]      for r in numeric_rows) / len(numeric_rows)
        mean_dtwd = sum(r["dtwd"]     for r in numeric_rows) / len(numeric_rows)
        print(f"{'MEAN':<{col_w[0]}} {'':>{col_w[1]}} {mean_rmse:>{col_w[2]}.6f} {mean_mvd:>{col_w[3]}.6f} {mean_dtwd:>{col_w[4]}.4f}")

    print(sep)

    csv_path = os.path.join(logdir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["shape", "epoch", "rmse_vel", "mvd", "dtwd"])
        writer.writeheader()
        writer.writerows(rows)
        if numeric_rows:
            writer.writerow({"shape": "MEAN", "epoch": "",
                             "rmse_vel": f"{mean_rmse:.6f}",
                             "mvd": f"{mean_mvd:.6f}",
                             "dtwd": f"{mean_dtwd:.4f}"})

    print(f"\nSummary saved to {csv_path}")


if __name__ == "__main__":
    main()
