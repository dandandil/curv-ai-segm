"""Кривые прогона по эпохам в notebooks/curves/<прогон>.png: лоссы train/val, IoU нарезки (всё и по
полуширине), точность/полнота/F1 стыков, mIoU типа, шаг обучения. Берёт runs/<прогон>/log.json;
шаг — из строк «эпоха N/M  lr энкодера …» в логах прогона (в log.json его нет).

    python scripts/plot_curves.py tile50              # один раз
    python scripts/plot_curves.py tile50 --watch      # перерисовывать после каждой эпохи, до последней
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "curves"
LR_LINE = re.compile(r"^эпоха (\d+)/(\d+)\s+lr энкодера ([0-9.e+-]+)")


def learning_rates(run: Path) -> tuple[dict[int, float], int | None]:
    """Шаг энкодера по эпохам и число эпох прогона, по всем логам прогона; при продолжении
    более поздний лог перекрывает ранний."""
    lr, total = {}, None
    logs = sorted(list(run.glob("*.log")) + list(run.glob("ranks/*/attempt_*/0/stdout.log")),
                  key=lambda p: p.stat().st_mtime)
    for path in logs:
        for line in path.read_text(errors="ignore").splitlines():
            m = LR_LINE.match(line)
            if m:
                lr[int(m[1])], total = float(m[3]), int(m[2])
    return lr, total


def plot(name: str) -> int:
    run = ROOT / "runs" / name
    log = json.loads((run / "log.json").read_text())
    ep = [r["epoch"] for r in log]
    col = lambda k: [r.get(k, float("nan")) for r in log]          # noqa: E731
    lr, total = learning_rates(run)
    iou = col("graphv_mean_iou")
    best = max(range(len(iou)), key=iou.__getitem__)

    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))
    for a, key in zip(ax[0], ("edge", "triplet", "type")):
        a.plot(ep, col(f"train_{key}"), label="train")
        a.plot(ep, col(f"val_{key}"), label="val")
        a.set_title(f"лосс {key}")
        a.legend()
    a = ax[1, 0]
    a.plot(ep, iou, "k", lw=2, label="граф с вето")
    a.plot(ep[best], iou[best], "r*", ms=14, label=f"лучший {iou[best]:.3f} (эп. {ep[best]})")
    a.set_title("IoU нарезки")
    a.legend()
    a = ax[1, 1]
    for band in ("le0.5", "0.5-1", "1-3", "gt3"):
        a.plot(ep, col(f"graphv_iou_width_{band}"), label=f"полуширина {band} мм")
    a.set_title("IoU по полуширине грани")
    a.legend(fontsize=9)
    a = ax[1, 2]
    for key, label in (("val_edge_precision", "точность стыков"), ("val_edge_recall", "полнота стыков"),
                       ("val_edge_f1", "F1 стыков"), ("val_type_miou", "mIoU типа")):
        a.plot(ep, col(key), label=label)
    a.set_title("стыки и тип")
    a.legend(fontsize=9)
    if lr:
        t = a.twinx()
        t.semilogy(sorted(lr), [lr[e] for e in sorted(lr)], "k:", lw=1)
        t.set_ylabel("шаг энкодера (пунктир)")
    for a in ax.ravel():
        a.grid(alpha=0.3)
        a.set_xlabel("эпоха")
    fig.suptitle(f"{name}: эпох {len(log)}" + (f" из {total}" if total else ""), fontsize=14)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f".{name}.png"
    fig.savefig(tmp, dpi=80)
    plt.close(fig)
    tmp.replace(OUT / f"{name}.png")                 # чтобы ноутбук не открыл недописанный файл
    return len(log) if total is None else total - len(log)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()
    log = ROOT / "runs" / args.run / "log.json"
    seen = None
    while True:
        stamp = log.stat().st_mtime
        if stamp != seen:
            seen = stamp
            left = plot(args.run)
            print(f"{time.strftime('%H:%M')} нарисовано, осталось эпох {left}", flush=True)
            if not args.watch or left <= 0:
                return
        time.sleep(60)


if __name__ == "__main__":
    main()
