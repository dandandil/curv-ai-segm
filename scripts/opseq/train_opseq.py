"""Обучение модели операций: базовая линия на идеальной сегментации (README, «Тело последовательностью операций»).

    python scripts/opseq/train_opseq.py --limit 2000 --epochs 2 --out runs/opseq_probe
    python scripts/opseq/train_opseq.py --out runs/opseq_base
    python scripts/opseq/train_opseq.py --out runs/opseq_base --resume
    python scripts/opseq/train_opseq.py --fraction 0.25 --out runs/opseq_frac25

Тела — те, у которых собран граф (`scripts/opseq/build_graphs.py`), разделение по дизайнам
(`src/opseq/data.py`, `split_uids`). Чекпоинты `last.pt` и `best.pt` (по лоссу val), история —
`history.jsonl` в каталоге прогона.
"""
from __future__ import annotations

import argparse
import hashlib
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

from src.opseq.config import OCFG  # noqa: E402
from src.opseq.data import Bodies, BudgetBatches, collate, design, graph_keys, split_uids  # noqa: E402
from src.opseq.losses import batch_losses  # noqa: E402
from src.opseq.model import OpseqNet  # noqa: E402


def graph_uids(corpora=None) -> list[str]:
    """Ключи `<корпус>/<uid>` тел с графом; по умолчанию — корпуса `data.train_corpora`."""
    return graph_keys(corpora or OCFG.data.train_corpora)


def loader_of(ds: Bodies, shuffle: bool, workers: int, seed: int = 0) -> DataLoader:
    cfg = OCFG.train
    return DataLoader(ds, batch_sampler=BudgetBatches(ds.nodes, cfg.batch, cfg.batch_pairs, shuffle, seed),
                      collate_fn=collate, num_workers=workers, persistent_workers=shuffle)


def rates(hits: Counter) -> dict[str, float]:
    """Доли верных по головам, F1 отношений, стенок операции, ролей и «одной операции», средняя ошибка глубины /
    диагональ."""
    out = {k[:-6]: hits[k] / hits[k[:-6] + ":всего"] for k in hits if k.endswith(":верно") and hits[k[:-6] + ":всего"]}
    for k in ("relation", "стенки", "role", "same_op"):
        tp, fp, fn = hits[f"{k}:tp"], hits[f"{k}:fp"], hits[f"{k}:fn"]
        if tp + fp + fn:
            out[f"{k}:f1"] = 2 * tp / (2 * tp + fp + fn)
    if hits["depth:всего"]:
        out["depth:mae"] = hits["depth:abs"] / hits["depth:всего"]
    if hits["шаг без опоры"]:
        out["шагов без опоры"] = hits["шаг без опоры"]
    return out


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, dict[str, float], dict[str, float]]:
    model.eval()
    hits, losses, total, n = Counter(), Counter(), 0.0, 0
    for batch in loader:
        tally = batch_losses(model, batch, device)
        loss, parts = tally.total()
        total, n = total + float(loss), n + 1
        losses.update(parts)
        hits.update(tally.hits)
    model.train()
    return total / max(n, 1), {k: v / max(n, 1) for k, v in losses.items()}, rates(hits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=OCFG.train.epochs)
    ap.add_argument("--limit", type=int, help="тел train (val — пропорционально)")
    ap.add_argument("--corpora", nargs="+", help="корпуса вместо data.train_corpora")
    ap.add_argument("--fraction", type=float, help="доля дизайнов train, val целиком (кривая по объёму)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = OCFG.train
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    split = split_uids(graph_uids(args.corpora))
    train, val = split["train"], split["val"]
    if args.limit:
        rng = np.random.default_rng(args.seed)
        train = sorted(rng.permutation(train)[:args.limit].tolist())
        val = val[:max(1, args.limit * len(val) // len(split["train"]))]
    if args.fraction:
        # свой хэш, не тот, что в split_uids; доли вложены: 0.25 входит в 0.5
        keep = lambda u: int(hashlib.md5(f"fraction:{design(u)}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < args.fraction
        train = [u for u in train if keep(u)]
    loader = loader_of(Bodies(train), True, args.workers, args.seed)
    val_loader = loader_of(Bodies(val), False, args.workers)
    print(f"тел train {len(loader.dataset)}, val {len(val_loader.dataset)}", flush=True)

    model = OpseqNet().to(device)
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
    (out / "args.json").write_text(json.dumps({**vars(args), "config": str(OCFG)}, ensure_ascii=False))

    for epoch in range(start, args.epochs):
        t0, seen, run = time.perf_counter(), 0, Counter()
        for it, batch in enumerate(loader, 1):
            tally = batch_losses(model, batch, device)
            loss, parts = tally.total()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            run.update(parts)
            seen += 1
            if it % 100 == 0:
                print(f"эпоха {epoch} шаг {it}/{len(loader)}  " +
                      "  ".join(f"{k} {v / seen:.3f}" for k, v in sorted(run.items())) +
                      f"  {(time.perf_counter() - t0) / it:.2f} с/шаг", flush=True)
        val_loss, val_parts, val_rates = evaluate(model, val_loader, device)
        rec = {"epoch": epoch, "train": {k: v / max(seen, 1) for k, v in run.items()}, "val_loss": val_loss,
               "val": val_parts, "val_rates": val_rates, "minutes": (time.perf_counter() - t0) / 60}
        with open(out / "history.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"=== эпоха {epoch}  val {val_loss:.4f}  " +
              "  ".join(f"{k} {v:.3f}" for k, v in sorted(val_rates.items())), flush=True)
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "epoch": epoch, "best": min(best, val_loss)}
        torch.save(ck, out / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(ck, out / "best.pt")


if __name__ == "__main__":
    main()
