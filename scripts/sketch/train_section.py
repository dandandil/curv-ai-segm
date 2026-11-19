"""Обучение 2D сети сечения (README, «Эскиз из сечения»).

    python scripts/sketch/train_section.py --out runs/section_base
    python scripts/sketch/train_section.py --limit 2000 --epochs 1 --out runs/section_probe

Детали — с сечениями (`scripts/sketch/build_sections.py`), разделение по дизайнам то же, что у
opseq (`src/opseq/data.py`, `split_uids`). Чекпоинты `last.pt` и `best.pt` (по лоссу val),
история — `history.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.opseq.data import split_uids  # noqa: E402
from src.sketch.config import SCFG  # noqa: E402
from src.sketch.data import Parts, SegmentBatches, collate  # noqa: E402
from src.sketch.losses import rates, section_losses  # noqa: E402
from src.sketch.model import SectionNet  # noqa: E402


def section_uids() -> list[str]:
    return sorted(p.name[:-4] for p in (Path(SCFG.data.root) / "sections").glob("*.npz") if not p.name.endswith(".tmp.npz"))


def loader_of(ds: Parts, shuffle: bool, workers: int, seed: int = 0) -> DataLoader:
    return DataLoader(ds, batch_sampler=SegmentBatches(ds.segments, SCFG.train.batch_segments, shuffle, seed),
                      collate_fn=collate, num_workers=workers, persistent_workers=shuffle)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, dict[str, float], dict[str, float]]:
    model.eval()
    hits, losses, total, n = Counter(), Counter(), 0.0, 0
    for batch in loader:
        loss, parts, h = section_losses(model, batch, device)
        total, n = total + float(loss), n + 1
        losses.update(parts)
        hits.update(h)
    model.train()
    return total / max(n, 1), {k: v / max(n, 1) for k, v in losses.items()}, rates(hits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=SCFG.train.epochs)
    ap.add_argument("--limit", type=int, help="деталей train (val — пропорционально)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = SCFG.train
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    split = split_uids(section_uids())
    train, val = split["train"], split["val"]
    if args.limit:
        rng = np.random.default_rng(args.seed)
        train = sorted(rng.permutation(train)[:args.limit].tolist())
        val = val[:max(1, args.limit * len(val) // len(split["train"]))]
    loader = loader_of(Parts(train, True), True, args.workers, args.seed)
    val_loader = loader_of(Parts(val, False), False, args.workers)
    print(f"деталей train {len(loader.dataset)}, val {len(val_loader.dataset)}, батчей {len(loader)}", flush=True)

    model = SectionNet().to(device)
    print(f"параметров {sum(p.numel() for p in model.parameters()) / 1e6:.2f} млн", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))
    start, best = 0, float("inf")
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck["best"]
    (out / "args.json").write_text(json.dumps({**vars(args), "config": str(SCFG)}, ensure_ascii=False))

    for epoch in range(start, args.epochs):
        t0, seen, run = time.perf_counter(), 0, Counter()
        for it, batch in enumerate(loader, 1):
            loss, parts, _ = section_losses(model, batch, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            run.update(parts)
            seen += 1
            if it % 200 == 0:
                print(f"эпоха {epoch} шаг {it}/{len(loader)}  " +
                      "  ".join(f"{k} {v / seen:.4f}" for k, v in sorted(run.items())) +
                      f"  {(time.perf_counter() - t0) / it:.2f} с/шаг", flush=True)
        val_loss, val_parts, val_rates = evaluate(model, val_loader, device)
        rec = {"epoch": epoch, "train": {k: v / max(seen, 1) for k, v in run.items()}, "val_loss": val_loss,
               "val": val_parts, "val_rates": val_rates, "minutes": (time.perf_counter() - t0) / 60}
        with open(out / "history.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"=== эпоха {epoch}  val {val_loss:.4f}  " +
              "  ".join(f"{k} {v:.4f}" for k, v in sorted(val_rates.items())), flush=True)
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "epoch": epoch, "best": min(best, val_loss)}
        torch.save(ck, out / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(ck, out / "best.pt")


if __name__ == "__main__":
    main()
