"""Лосс и счётчики 2D сети сечения."""
from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import SCFG


def section_losses(model, batch: dict, device, cfg=SCFG.train) -> tuple[Tensor, dict[str, float], Counter]:
    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    type_logits, bnd, tan = model(b)
    parts, hits = {}, Counter()
    t = b["type"]
    parts["type"] = F.cross_entropy(type_logits, t, ignore_index=-1)
    lab = t >= 0
    hits["type:верно"] += int((type_logits.argmax(1)[lab] == t[lab]).sum())
    hits["type:всего"] += int(lab.sum())
    y = b["boundary"]
    m = y >= 0
    parts["boundary"] = F.binary_cross_entropy_with_logits(bnd[m], y[m].float())
    pred = bnd[m] > 0
    hits["boundary:tp"] += int((pred & (y[m] == 1)).sum())
    hits["boundary:fp"] += int((pred & (y[m] == 0)).sum())
    hits["boundary:fn"] += int((~pred & (y[m] == 1)).sum())
    g = b["tangent"]
    m = (y == 1) & (g >= 0)
    if m.any():
        parts["tangent"] = F.binary_cross_entropy_with_logits(tan[m], g[m].float())
        hits["tangent:верно"] += int(((tan[m] > 0) == (g[m] == 1)).sum())
        hits["tangent:всего"] += int(m.sum())
    loss = sum(getattr(cfg, f"{k}_weight") * v for k, v in parts.items())
    return loss, {k: v.item() for k, v in parts.items()}, hits


def rates(hits: Counter) -> dict[str, float]:
    out = {k[:-6]: hits[k] / hits[k[:-6] + ":всего"] for k in hits if k.endswith(":верно") and hits[k[:-6] + ":всего"]}
    tp, fp, fn = hits["boundary:tp"], hits["boundary:fp"], hits["boundary:fn"]
    if tp + fp:
        out["boundary:точность"] = tp / (tp + fp)
    if tp + fn:
        out["boundary:полнота"] = tp / (tp + fn)
    return out
