"""Обучение эталона ParSeNet на наших деталях, чистые точки.

    python scripts/parsenet/train_parsenet.py --limit 48 --epochs 1 --val-limit 16   # проверка
    python scripts/parsenet/train_parsenet.py --epochs 40 --out runs/parsenet/clean
    python scripts/parsenet/train_parsenet.py --epochs 40 --out runs/parsenet/clean --resume

Цикл — как `train_parsenet.py` авторов: Adam lr 0.01, шаг оптимизатора на
3 батча по 8 форм, лосс батча не делится на число батчей, ReduceLROnPlateau по
эмбеддинг-лоссу валидации, лучший чекпоинт — по нему же. Эпоха здесь — один
проход по train; у авторов «эпоха» — три прохода (train/8 шагов по 24 формы).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model import PRIMITIVE_TYPES  # noqa: E402
from src.parsenet import data as PD  # noqa: E402
from src.parsenet.config import PCFG  # noqa: E402
from src.parsenet.loss import primitive_loss, triplet_loss  # noqa: E402
from src.parsenet.model import ParSeNet  # noqa: E402


class Batches(Dataset):
    """Микробатчи одной эпохи. Семя — из генератора torch воркера: новое на каждой выдаче."""

    def __init__(self, split: PD.PointSplit, groups: list[np.ndarray], augment: bool):
        self.split, self.groups, self.augment = split, groups, augment

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int):
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 62, ()).item()))
        return PD.batch(self.split, self.groups[i], rng, PCFG.data.train_points, self.augment, PCFG.data.normal_noise)


def groups(n: int, size: int, rng: np.random.Generator | None) -> list[np.ndarray]:
    order = rng.permutation(n) if rng is not None else np.arange(n)
    order = order[: n - n % size]
    return [np.sort(order[i:i + size]) for i in range(0, len(order), size)]


def forward(model, item, device, rng):
    points, normals, labels, prim = item
    x = torch.cat([torch.as_tensor(points), torch.as_tensor(normals)], 2).permute(0, 2, 1).to(device)
    embedding, log_prob = model(x)
    t = triplet_loss(embedding, np.asarray(labels), PCFG.train.margin, PCFG.train.max_segment_pairs,
                     PCFG.train.samples_per_segment, rng)
    c = primitive_loss(log_prob, torch.as_tensor(prim).to(device))
    return t, c


def save(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--val-limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "parsenet" / "clean")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    train, val = PD.PointSplit("train"), PD.PointSplit("val")
    n_train = min(len(train), args.limit or len(train))
    n_val = min(len(val), args.val_limit or len(val))
    micro = PCFG.train.micro_batch
    per_step = PCFG.train.batch_size * PCFG.train.batches_per_step // micro
    scale = micro / PCFG.train.batch_size

    model = ParSeNet(PCFG.model.emb_size, len(PRIMITIVE_TYPES), PCFG.model.neighbors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PCFG.train.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=PCFG.train.lr_factor, patience=PCFG.train.lr_patience, min_lr=PCFG.train.lr_min)
    args.out.mkdir(parents=True, exist_ok=True)
    log, best, start = [], float("inf"), 0
    if args.resume:
        state = torch.load(args.out / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        log, best, start = state["log"], state["best"], state["epoch"]
        torch.set_rng_state(state["torch_rng"].cpu())
        rng = np.random.default_rng(args.seed + start)
        print(f"продолжаю после эпохи {start}", flush=True)
    print(f"train {n_train}, val {n_val}, микробатч {micro}, шаг оптимизатора на {per_step} микробатчей, "
          f"параметров {sum(p.numel() for p in model.parameters()):,}", flush=True)

    for epoch in range(start, args.epochs):
        model.train()
        loader = DataLoader(Batches(train, groups(n_train, micro, rng), True), batch_size=None,
                            num_workers=args.workers, shuffle=False)
        optimizer.zero_grad(set_to_none=True)
        hist = {"triplet": [], "type": []}
        t0 = time.perf_counter()
        print(f"эпоха {epoch + 1}/{args.epochs}  lr {optimizer.param_groups[0]['lr']:.1e}", flush=True)
        for i, item in enumerate(loader):
            t, c = forward(model, item, device, rng)
            ((t + c).sum() * scale).backward()
            if (i + 1) % per_step == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            hist["triplet"].append(float(t))
            hist["type"].append(float(c))
            if (i + 1) % args.log_every == 0:
                dt = (time.perf_counter() - t0) / ((i + 1) * micro)
                print(f"  {(i + 1) * micro}/{n_train}  triplet {np.mean(hist['triplet'][-args.log_every:]):.4f}  "
                      f"type {np.mean(hist['type'][-args.log_every:]):.4f}  {dt * 1000:.0f} мс/форма  "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} ГиБ", flush=True)

        model.eval()
        vt, vc = [], []
        val_rng = np.random.default_rng(args.seed)
        with torch.no_grad():
            for g in groups(n_val, micro, None):
                t, c = forward(model, PD.batch(val, g, val_rng, PCFG.data.train_points, False,
                                               PCFG.data.normal_noise), device, val_rng)
                vt.append(float(t)); vc.append(float(c))
        row = {"epoch": epoch + 1, "train_triplet": float(np.mean(hist["triplet"])),
               "train_type": float(np.mean(hist["type"])), "val_triplet": float(np.mean(vt)),
               "val_type": float(np.mean(vc)), "lr": optimizer.param_groups[0]["lr"],
               "minutes": (time.perf_counter() - t0) / 60}
        log.append(row)
        print("  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch"), flush=True)
        scheduler.step(row["val_triplet"])
        if row["val_triplet"] < best:
            best = row["val_triplet"]
            save(args.out / "best.pt", {"model": model.state_dict(), "epoch": epoch + 1, "val_triplet": best})
        save(args.out / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "log": log,
                                    "best": best, "torch_rng": torch.get_rng_state()})
        (args.out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
