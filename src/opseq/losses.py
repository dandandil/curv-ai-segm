"""Лосс по последовательности операций.

Правильный ответ шага — множество (свободный порядок, несколько опор, несколько «до k»), лосс
`−log Σ p` по нему. Teacher forcing — самый вероятный по мнению сети из правильных (решено
2026-10-22, вариант (b)): по нему идёт и следующая голова шага, и следующий шаг. Состояние между
шагами — по меткам (решено 2026-10-24). Скругления и фаски — операции цикла (решено 2026-10-25).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .config import OCFG
from .data import BOOLEANS, OP_TYPES, PRIMITIVE_TYPES, STATE_DIM, STEP_CLASSES, Body

# Пар на тело в голове отношений: у тела в 234 узла их 27 тысяч, большинство — ни одного
# отношения. Выбрано.
RELATION_PAIRS = 2048
# Параллельность плоскостей в простом правиле «до k»: |cos| нормалей, подогнанных по скану,
# больше 1 − 1e-3 (~2.6°) — запас на шум подгонки. Выбрано.
PARALLEL_COS = 1 - 1e-3
PLANE = PRIMITIVE_TYPES.index("plane")
EXTRUDE = OP_TYPES.index("ExtrudeFeature")
DONE = STEP_CLASSES.index("done")
HEADS = ("type", "support", "faces", "boolean", "to", "depth", "direction", "relation", "role", "same_op")


def set_nll(logits: Tensor, targets: list[int]) -> Tensor:
    """`−log Σ_{t ∈ targets} softmax(logits)_t`; недопустимые варианты — `−inf` в `logits`."""
    return torch.logsumexp(logits, 0) - torch.logsumexp(logits[targets], 0)


def _pick(logits: Tensor, targets: list[int]) -> int:
    return targets[int(logits[targets].argmax())]


def _pad_state(bodies: list[Body], done: list[list[str]], n: int, device) -> Tensor:
    out = torch.zeros(len(bodies), n, STATE_DIM)
    for j, (body, d) in enumerate(zip(bodies, done)):
        out[j, :body.n] = torch.from_numpy(body.state(d))
    return out.to(device)


class Tally:
    """Суммы лоссов и счётчики верных ответов по головам; `wrong` — тела с ошибкой хоть на одном шаге."""

    def __init__(self):
        self.loss = {k: [] for k in HEADS}
        self.hits = Counter()
        self.wrong: set[str] = set()

    def add(self, head: str, loss: Tensor, hit: bool | None = None, n: int = 1, uid: str | None = None) -> None:
        self.loss[head].append(loss)
        if hit is not None:
            self.add_hit(head, hit, uid, n)

    def add_hit(self, head: str, hit: bool, uid: str | None = None, n: int = 1) -> None:
        self.hits[f"{head}:верно"] += int(hit)
        self.hits[f"{head}:всего"] += n
        if uid is not None and not hit:
            self.wrong.add(uid)

    def rule(self, name: str, hit: float) -> None:
        """Простое правило на том же шаге, что и сеть (`hit` — доля, у случайного — матожидание)."""
        self.hits[f"правило {name}:верно"] += hit
        self.hits[f"правило {name}:всего"] += 1

    def total(self, cfg=OCFG.train) -> tuple[Tensor, dict[str, float]]:
        parts = {k: torch.stack(v).mean() for k, v in self.loss.items() if v}
        loss = sum(getattr(cfg, f"{k}_weight") * v for k, v in parts.items())
        return loss, {k: v.item() for k, v in parts.items()}


def _f1_counts(tally: Tally, name: str, p: Tensor, y: Tensor) -> None:
    tally.hits[f"{name}:tp"] += int((p & y).sum())
    tally.hits[f"{name}:fp"] += int((p & ~y).sum())
    tally.hits[f"{name}:fn"] += int((~p & y).sum())


def pre_loop(model, h: Tensor, pairs: Tensor, batch: dict, tally: Tally) -> None:
    """Головы по прогону с пустым состоянием: отношения пар, роль грани, «одна операция»."""
    device = h.device
    index = []
    for j, body in enumerate(batch["bodies"]):
        i, k = torch.triu_indices(body.n, body.n, 1)
        if len(i) > RELATION_PAIRS:
            keep = torch.randperm(len(i))[:RELATION_PAIRS]
            i, k = i[keep], k[keep]
        index.append(torch.stack([torch.full_like(i, j), i, k], 1))
    index = torch.cat(index).to(device)
    if len(index):
        z = model.relation_logits(h, pairs, index)
        y = batch["relations"].to(device)[index[:, 0], index[:, 1], index[:, 2]]
        tally.add("relation", F.binary_cross_entropy_with_logits(z, y))
        _f1_counts(tally, "relation", z > 0, y > 0)
        z = model.same_op_logits(h, pairs, index)
        y = batch["same_op"].to(device)[index[:, 0], index[:, 1], index[:, 2]]
        tally.add("same_op", F.binary_cross_entropy_with_logits(z, y))
        _f1_counts(tally, "same_op", z > 0, y > 0)
    mask = batch["mask"].to(device)
    z = model.role_logits(h[:, :-1][mask])
    y = batch["roles"].to(device)[mask]
    tally.add("role", F.binary_cross_entropy_with_logits(z, y))
    _f1_counts(tally, "role", z > 0, y > 0)
    right = ((z > 0) == (y > 0)).all(1)
    tally.hits["role:верно"] += int(right.sum())
    tally.hits["role:всего"] += len(right)


def support_rules(body: Body, allowed: np.ndarray, targets: list[int], tally: Tally) -> None:
    """Опора выдавливания: случайная допустимая и самая большая плоскость."""
    cand = np.flatnonzero(allowed)
    tally.rule("опора случайная", len(targets) / len(cand))
    tally.rule("опора большая плоскость", float(cand[body.g["static"][cand, 14].argmax()] in targets))


def to_rule(body: Body, s: int, targets: list[int], tally: Tally) -> None:
    """«До k» — самая дальняя плоскость, параллельная опоре; нет такой или опора не плоскость —
    «без поверхности»."""
    g, pred = body.g, body.n
    if g["plane"][s]:
        ok = g["plane"] & (np.abs(g["dir"] @ g["dir"][s]) > PARALLEL_COS)
        ok[s] = False
        if ok.any():
            dist = np.abs((g["anchor"] - g["anchor"][s]) @ g["dir"][s])
            pred = int(np.where(ok, dist, -1).argmax())
    tally.rule("до дальняя параллельная", float(pred in targets))


def face_heads(model, h: Tensor, body: Body, ops: list[str], s: int, kind: int, tally: Tally) -> str:
    """Класс граней (нет / стенка / торец); правильный ответ — разметка любой из операций `ops` с
    этой затравкой, лосс `−log Σ p(разметки)` на узел. Возвращает выбранную операцию."""
    n, device = body.n, h.device
    logp = F.log_softmax(model.face_logits(h[:n], h[s], kind), -1)
    labels = torch.from_numpy(np.stack([body.ops[u].faces for u in ops])).to(device)
    per_op = logp.gather(1, labels.T).sum(0)
    tally.add("faces", -torch.logsumexp(per_op, 0) / n)
    u = ops[int(per_op.argmax())]
    pred = logp.argmax(1).cpu().numpy()
    gold = body.ops[u].faces
    tally.add_hit("faces", bool((pred == gold).all()), body.uid)
    tally.hits["стенки:tp"] += int(((pred == 1) & (gold == 1)).sum())
    tally.hits["стенки:fp"] += int(((pred == 1) & (gold != 1)).sum())
    tally.hits["стенки:fn"] += int(((pred != 1) & (gold == 1)).sum())
    return u


def step_heads(model, h: Tensor, pairs: Tensor, body: Body, done: list[str], tally: Tally) -> str | None:
    """Головы одного шага тела; возвращает выполненную операцию (по выбору сети среди правильных)
    или None, если тело закончено. `h` — `(N_max + 1, d)` тела, `pairs` — `(N_max, N_max, pair_dim)`."""
    n, device, uid = body.n, h.device, body.uid
    z = model.type_logits(h)
    if not body.available(done):
        tally.add("type", set_nll(z, [DONE]), int(z.argmax()) == DONE, uid=uid)
        return None
    entries = body.entries(done)
    if not entries:
        # опоры всех доступных операций потеряны сканером или не того типа: шаг без метки, операция по истории
        tally.hits["шаг без опоры"] += 1
        tally.wrong.add(uid)
        return body.available(done)[0]
    targets = sorted({body.ops[u].kind for u, _ in entries})
    tally.add("type", set_nll(z, targets), int(z.argmax()) in targets, uid=uid)
    tally.rule("тип всегда выдавливание", float(EXTRUDE in targets))
    kind = _pick(z, targets)
    cand = [(u, sup) for u, sup in entries if body.ops[u].kind == kind]

    logits = model.support_logits(h[:n], kind)
    allowed = torch.from_numpy(body.support_masks[kind]).to(device)
    logits = logits.masked_fill(~allowed, float("-inf"))
    targets = sorted({sup.node for _, sup in cand})
    tally.add("support", set_nll(logits, targets), int(logits.argmax()) in targets, uid=uid)
    if kind == EXTRUDE:
        support_rules(body, body.support_masks[kind], targets, tally)
    s = _pick(logits, targets)
    cand = [(u, sup) for u, sup in cand if sup.node == s]

    u = face_heads(model, h, body, sorted({u for u, _ in cand}), s, kind, tally)
    if kind != EXTRUDE:
        return u
    sup = next(sup for v, sup in cand if v == u)
    op = body.ops[u]
    sides = np.flatnonzero(op.faces == 1)
    profile = h[sides].mean(0) if len(sides) else torch.zeros_like(h[s])

    z = model.boolean_logits(h[s], profile)
    tally.add("boolean", set_nll(z, [op.boolean]), int(z.argmax()) == op.boolean, uid=uid)
    tally.rule("булева всегда add", float(op.boolean == BOOLEANS.index("add")))

    none = n
    targets = sorted(sup.to | ({none} if sup.depth is not None else set()))
    if targets:
        z, q = model.to_heads(h[:n], pairs[:n, :n], s, profile, op.boolean)
        z = z.masked_fill(torch.arange(n + 1, device=device) == s, float("-inf"))
        tally.add("to", set_nll(z, targets), int(z.argmax()) in targets, uid=uid)
        to_rule(body, s, targets, tally)
        if _pick(z, targets) == none:
            depth, direction = model.depth_direction(q)
            tally.add("depth", F.smooth_l1_loss(depth, torch.tensor(sup.depth, device=device), beta=0.01))
            tally.hits["depth:abs"] += abs(depth.item() - sup.depth)
            tally.hits["depth:всего"] += 1
            y = torch.tensor(float(sup.direction > 0), device=device)
            tally.add("direction", F.binary_cross_entropy_with_logits(direction, y), bool((direction > 0) == (y > 0)),
                      uid=uid)
    return u


def batch_losses(model, batch: dict, device) -> Tally:
    """Все головы на батче тел: прогон до цикла и шаги цикла до «готово» у каждого тела."""
    tally = Tally()
    bodies = batch["bodies"]
    mask = batch["mask"].to(device)
    nodes = model.embed_nodes(batch["static"].to(device), batch["points"].to(device), mask)
    pairs = model.embed_pairs(batch["pair"].to(device), batch["spd"].to(device))
    h = model.encode(nodes, nodes.new_zeros(*mask.shape, STATE_DIM), pairs, mask)
    pre_loop(model, h, pairs, batch, tally)

    done = [[] for _ in bodies]
    active = list(range(len(bodies)))
    while active:
        idx = torch.tensor(active, device=device)
        state = _pad_state([bodies[j] for j in active], [done[j] for j in active], mask.shape[1], device)
        h = model.encode(nodes[idx], state, pairs[idx], mask[idx])
        still = []
        for r, j in enumerate(active):
            op = step_heads(model, h[r], pairs[j], bodies[j], done[j], tally)
            if op is not None:
                done[j].append(op)
                still.append(j)
        active = still
    # тело верно целиком: на всех шагах верны тип (и «готово»), указатель, грани операции, булева,
    # «до k», направление; глубина — регрессия, не входит
    tally.hits["тело:верно"] += sum(b.uid not in tally.wrong for b in bodies)
    tally.hits["тело:всего"] += len(bodies)
    return tally
