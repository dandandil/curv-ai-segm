"""Обучение варианта D: голова границ поверх энкодера PrimitiveSegNet.

    python scripts/boundary/train_boundary.py --limit 2000 --epochs 3 --out runs/boundary_ab
    python scripts/boundary/train_boundary.py --epochs 10 --out runs/boundary
    python scripts/boundary/train_boundary.py --epochs 10 --out runs/boundary --resume

Энкодер и старые головы стартуют с чекпоинта основной архитектуры (`--init`),
голова границ — случайной. Батч, чекпоинты и продолжение прогона — из
scripts/train.py.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train as base  # noqa: E402
from src import cluster as C  # noqa: E402
from src.boundary import partition as P  # noqa: E402
from src.boundary.config import BCFG  # noqa: E402
from src.boundary.data import long_range_pairs, tangent_mask, tangent_pairs  # noqa: E402
from src.boundary.losses import contrast_loss, edge_loss, edge_scores, edge_targets, repulsive_loss  # noqa: E402
from src.boundary.model import BoundarySegNet, boundary_strength, head_parameters, load_backbone  # noqa: E402
from src.config import CFG, load_config  # noqa: E402
from src.data import CachedParts, collate_single, input_names, part_meta, split_uids  # noqa: E402
from src.losses import triplet_loss, type_loss  # noqa: E402
from src.metrics import MIN_FACE, face_records, grouped_scores, segment_scores, type_miou  # noqa: E402
from src.model import PRIMITIVE_TYPES  # noqa: E402
from src.runs import run_config  # noqa: E402


class EpochSubset(torch.utils.data.Sampler):
    """`k` случайных из `n` деталей, новые на каждой эпохе."""

    def __init__(self, n: int, k: int, seed: int):
        self.n, self.k, self.seed, self.epoch = n, k, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        return iter(np.random.default_rng((self.seed, self.epoch)).permutation(self.n)[:self.k].tolist())

    def __len__(self) -> int:
        return self.k


def build_batch(item: dict, device, generator, rep: int = 0) -> dict:
    """`rep` — дальних пар на треугольник и слой для головы отталкивающих связей; 0 — без них."""
    batch = base.build_batch(item, device, generator)
    batch["edges"] = torch.from_numpy(item["adjacency"].astype(np.int64)).to(device)
    batch["ambiguous"] = ~batch["keep"]
    batch["core"] = torch.from_numpy(item["core"]).to(device)
    pairs = tangent_pairs(item["uid"])
    if "remap" in item:
        # Касание записано в метках кэша, а метки детали после правил разметки другие.
        pairs = np.sort(item["remap"][pairs[(pairs < len(item["remap"])).all(1)]], axis=1)
        pairs = np.unique(pairs[(pairs[:, 0] >= 0) & (pairs[:, 0] != pairs[:, 1])], axis=0).reshape(-1, 2)
    pairs = torch.from_numpy(pairs).to(device)
    batch["tangent"] = tangent_mask(batch["labels"], batch["edges"], pairs)
    if rep:
        batch["pairs"] = long_range_pairs(batch["idx"], rep, generator)
    return batch


def core_edge_scores(edge, batch) -> dict:
    """Метрики стыков только по рёбрам ядра куска: кайма — контекст, лосса на ней нет, и она же у
    края окна скана (README, «Края кусков»). У кэша без каймы ядро — вся деталь."""
    e = batch["edges"]
    sel = batch["core"][e[:, 0]] & batch["core"][e[:, 1]]
    return edge_scores(edge[sel], batch["labels"], batch["ambiguous"], e[sel], batch["tangent"][sel])


def losses(model, batch, generator, tcfg=BCFG.train):
    """Прямой проход и лоссы: triplet, тип, границы, перепад для водораздела и, если в
    батче есть дальние пары, отталкивающие связи.

    `tcfg` — настройки обучения головы: конфиг с поправками флагов прогона (потолок
    веса стыка, вес и запас лосса на перепад). Одни и те же для обучения и
    валидации, иначе их лоссы считались бы по-разному.
    """
    out = model(batch["x"], batch["idx"], batch["edges"], batch.get("pairs"))
    emb, logits, edge = out[:3]
    core = batch["core"]                                # кайма куска лосса не получает
    mask = batch["keep"] & core if CFG.train.exclude_ambiguous else core
    t = triplet_loss(emb[0], batch["labels"], CFG.train.margin, mask, generator)
    c = type_loss(logits[0], batch["types"], mask)
    band = batch["ambiguous"] if tcfg.exclude_band else torch.zeros_like(batch["ambiguous"])
    target, weight = edge_targets(batch["labels"], band, batch["edges"], tcfg.pos_weight_max,
                                  batch["tangent"], tcfg.tangent_weight, tcfg.face_balance)
    weight = weight * (core[batch["edges"][:, 0]] & core[batch["edges"][:, 1]]).float()
    e = edge_loss(edge, target, weight)
    k = contrast_loss(edge, batch["labels"], batch["edges"], tcfg.contrast_margin, tcfg.contrast_tau)
    total = t + CFG.train.type_weight * c + tcfg.edge_weight * e + tcfg.contrast_weight * k
    parts = {"triplet": float(t.detach()), "type": float(c.detach()), "edge": float(e.detach()),
             "contrast": float(k.detach())}
    if len(out) == 4:
        r = repulsive_loss(out[3], batch["labels"], batch["pairs"])
        total = total + tcfg.repulsive_weight * r
        parts["repulsive"] = float(r.detach())
    return total, parts, (emb, logits, edge)


def run_epoch(model, loader, optimizer, device, generator, log_every: int, accum: int, seen: int,
              tcfg=BCFG.train, quiet: bool = False, rep: int = 0) -> dict:
    """`accum` — сколько деталей приходится на шаг оптимизатора в этом процессе.

    На нескольких картах между шагами градиенты не синхронизируются (`no_sync`):
    обмен идёт раз в шаг, а не на каждой детали, и 5.4 МБ градиента по гигабитной
    сети стоят около 4% времени шага.
    """
    model.train()
    history: dict[str, list[float]] = {}
    t0 = time.perf_counter()
    for i, item in enumerate(loader):
        step = (seen + i + 1) % accum == 0
        sync = model.no_sync() if not step and hasattr(model, "no_sync") else contextlib.nullcontext()
        with sync:
            batch = build_batch(item, device, generator, rep)
            loss, parts, _ = losses(model, batch, generator, tcfg)
            (loss / accum).backward()
        if step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for k, v in parts.items():
            history.setdefault(k, []).append(v)
        if log_every and (i + 1) % log_every == 0 and not quiet:
            dt = (time.perf_counter() - t0) / (i + 1)
            print(f"  {i + 1}/{len(loader)}  "
                  + "  ".join(f"{k} {np.mean(v[-log_every:]):.4f}" for k, v in history.items())
                  + f"  {dt * 1000:.0f} мс/деталь", flush=True)
    return history


@torch.no_grad()
def evaluate(model, loader, device, seed: int, reduce: bool = True, tcfg=BCFG.train, rep: int = 0):
    """Лоссы и метрики стыков на валидации. `reduce=False` — сырые строки по деталям,
    чтобы собрать их со всех процессов и усреднить один раз."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = []
    for item in loader:
        batch = build_batch(item, device, generator, rep)
        _, parts, (_, logits, edge) = losses(model, batch, generator, tcfg)
        scores = core_edge_scores(edge, batch)
        rows.append({**parts, "type_miou": type_miou(logits[0], batch["types"], len(PRIMITIVE_TYPES)),
                     **{f"edge_{k}": v for k, v in scores.items()}})
    return reduce_losses(rows) if reduce else rows


def reduce_losses(rows: list[dict]) -> dict:
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def graph_labels(emb, edge, batch, cfg, use_veto: bool) -> torch.Tensor:
    strength = boundary_strength(edge)
    z = F.normalize(emb[0].t(), dim=1).double().cpu().numpy() if use_veto else None
    labels = P.partition(batch["x"].shape[2], batch["edges"].cpu().numpy(), strength, cfg, z)
    return torch.from_numpy(labels).to(batch["labels"].device)


@torch.no_grad()
def evaluate_segmentation(model, loader, device, seed: int, min_face: int,
                          mean_shift: bool = True, reduce: bool = True, cfg=BCFG.partition,
                          rep: int = 0, mws_limit: int = 0):
    """Нарезки одной модели: граф без вето, граф с вето и, по желанию, mean-shift.

    mean-shift стоит 1.9 с на деталь против 0.15 с у нарезки графом, то есть
    почти всю оценку эпохи. Без него оценку можно считать на всей валидации, и
    шум IoU падает втрое (README, «Что и когда меряется»).

    С головой отталкивающих связей на первых `mws_limit` деталях ещё и Mutex
    Watershed — сырой (`mws_`) и с поглощением мелких регионов (`mwsa_`), — а рядом
    граф с вето на тех же деталях (`graphvs_`): Mutex Watershed на чистом Python
    стоит секунды на деталь, и на всей оценке его не посчитать.
    """
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    rows, records, extra = [], [], {"mwsa": [], "graphvs": []}
    for i, item in enumerate(loader):
        batch = build_batch(item, device, generator, rep)
        out = model(batch["x"], batch["idx"], batch["edges"], batch.get("pairs"))
        emb, edge = out[0], out[2]
        labels = batch["labels"]
        # нарезка — по всему куску, метрики — по ядру
        big = (torch.bincount(labels)[labels] >= min_face) & batch["core"]
        row = {}
        if mean_shift:
            pred, _ = C.segment(emb[0], CFG.cluster.quantile, CFG.cluster.iterations,
                                CFG.cluster.bandwidth_samples, CFG.cluster.bandwidth_floor,
                                CFG.cluster.chunk, CFG.cluster.max_clusters, width=CFG.cluster.bandwidth)
            row = {f"seg_{k}": v for k, v in segment_scores(labels[big], pred[big]).items()}
        final = None
        for name, veto in (("graph", False), ("graphv", True)):
            lab = graph_labels(emb, edge, batch, cfg, veto)
            row.update({f"{name}_{k}": v for k, v in segment_scores(labels[big], lab[big]).items()})
            if veto:
                final = lab                                  # разбивка — по итоговой нарезке
        halfwidth = item["halfwidth_mm"]
        records += face_records(labels[big], final[big], batch["types"][big], halfwidth, PRIMITIVE_TYPES)
        if len(out) == 4 and i < mws_limit:
            e = batch["edges"].cpu().numpy()
            t0 = time.perf_counter()
            mws = P.mutex_watershed(len(labels), e, torch.sigmoid(edge).double().cpu().numpy(),
                                    batch["pairs"].cpu().numpy(), torch.sigmoid(-out[3]).double().cpu().numpy())
            row["mws_seconds"] = time.perf_counter() - t0
            mwsa = np.unique(P.absorb_small(mws, e, cfg.min_region), return_inverse=True)[1]
            for name, lab in (("mws", mws), ("mwsa", mwsa)):
                lab = torch.from_numpy(lab).to(labels.device)
                row.update({f"{name}_{k}": v for k, v in segment_scores(labels[big], lab[big]).items()})
            row.update({f"graphvs_{k[7:]}": v for k, v in row.items() if k.startswith("graphv_")})
            mwsa = torch.from_numpy(mwsa).to(labels.device)
            extra["mwsa"] += face_records(labels[big], mwsa[big], batch["types"][big], halfwidth, PRIMITIVE_TYPES)
            extra["graphvs"] += face_records(labels[big], final[big], batch["types"][big], halfwidth,
                                             PRIMITIVE_TYPES)
        rows.append(row)
    return reduce_segmentation(rows, records, extra) if reduce else (rows, records, extra)


def reduce_segmentation(rows: list[dict], records: list, extra: dict | None = None) -> dict:
    """Среднее по деталям — по тем, где величина посчитана (Mutex Watershed — не на всех).
    Разбивка по типам и полуширине — по итоговой нарезке, граф с вето, и для Mutex
    Watershed с графом на тех же деталях."""
    keys = dict.fromkeys(k for r in rows for k in r)
    out = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
    out.update({f"graphv_{k}": v for k, v in grouped_scores(records).items()})
    for name, recs in (extra or {}).items():
        if recs:
            out.update({f"{name}_{k}": v for k, v in grouped_scores(recs).items()})
    return out


def drop_channels(state: dict, init: Path, names: list[str]) -> None:
    """Каналы чекпоинта, которых нет в кэше, срезать в первом EdgeConv: в конфигах с
    2026-10-10 нет масштабов струи 0.02 и 0.05 (README, «Масштабы струи на будущее»).
    Имена каналов чекпоинта — по конфигу его прогона, в порядке `dataset/features.py`."""
    scales = load_config(run_config(init.parent.name)).features.scales_rel
    old = ["cx", "cy", "cz", "nx", "ny", "nz"] + [f"{nm}@{rel}" for rel in scales for nm in ("k1", "k2", "S", "dII")]
    if len(old) != state["channels"] or not set(names) <= set(old):
        raise SystemExit(f"{init} обучен на другом входе, чем задаёт кэш и конфиг")
    keep = torch.tensor([old.index(n) for n in names])
    key = "encoder.conv1.block.0.weight"           # вход EdgeConv: [сосед − центр, центр]
    w = state["model"][key]
    state["model"][key] = torch.cat([w[:, keep], w[:, len(old) + keep]], dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="noisy")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--val-limit", type=int, default=100)
    ap.add_argument("--seg-limit", type=int, default=100)
    ap.add_argument("--no-mean-shift", action="store_true",
                    help="не считать нарезку mean-shift после эпохи: она стоит всей оценки")
    ap.add_argument("--min-face", type=int, default=MIN_FACE)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=CFG.train.accum)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--absorb-blends-mm", type=float, default=0.0,
                    help="скругления и фаски до стольких мм отдать соседним граням в разметке (0 — нет)")
    ap.add_argument("--init", type=lambda v: None if v == "none" else Path(v),
                    default=ROOT / "runs" / "fine10_ft" / "last.pt",
                    help="чекпоинт основной архитектуры: энкодер и старые головы; none — всё с нуля")
    ap.add_argument("--backbone-lr", type=float, default=BCFG.train.backbone_lr)
    ap.add_argument("--head-lr", type=float, default=BCFG.train.head_lr)
    ap.add_argument("--pos-weight-max", type=float, default=BCFG.train.pos_weight_max,
                    help="потолок веса стыкового ребра в лоссе границ")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "boundary")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--plateau-on", choices=("edge", "iou"), default="edge",
                    help="по чему планировщик снижает шаг: лосс границ или IoU нарезки")
    ap.add_argument("--reset-lr", type=float,
                    help="при продолжении задать шаг заново, счётчик планировщика обнулить")
    ap.add_argument("--shard-data", action="store_true",
                    help="у каждой машины своя половина кэша: детали делятся списком, а не сэмплером")
    ap.add_argument("--min-region", type=int, help="переопределить размер поглощаемого региона")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--no-jet", action="store_true",
                    help="без каналов кривизны на входе, даже если они в кэше (README, «Фиксированное ребро и куски»)")
    ap.add_argument("--narrow-rule", action="store_true",
                    help="правило узких граней из brep.json (README, «Узкие грани: ширина в мм и правило разметки»)")
    ap.add_argument("--epoch-parts", type=int,
                    help="деталей train на эпоху на все машины, каждую эпоху новые случайные; по умолчанию все")
    ap.add_argument("--face-balance", action=argparse.BooleanOptionalAction, default=BCFG.train.face_balance,
                    help="вес лосса границ на грань, а не на ребро: каждая грань весит поровну")
    ap.add_argument("--contrast-weight", type=float, default=BCFG.train.contrast_weight,
                    help="вес лосса на перепад дна грани и перевала стыка")
    ap.add_argument("--contrast-margin", type=float, default=BCFG.train.contrast_margin)
    ap.add_argument("--contrast-tau", type=float, default=BCFG.train.contrast_tau)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="учить только голову границ: энкодер и прочие головы заморожены")
    ap.add_argument("--repulsive", action=argparse.BooleanOptionalAction, default=BCFG.model.repulsive,
                    help="голова отталкивающих связей и Mutex Watershed в оценке")
    ap.add_argument("--repulsive-weight", type=float, default=BCFG.train.repulsive_weight)
    ap.add_argument("--mws-limit", type=int, default=150,
                    help="на скольких деталях оценки каждой машины считать Mutex Watershed")
    args = ap.parse_args()

    if args.freeze_backbone and args.init is None and not args.resume:
        raise SystemExit("--freeze-backbone при --init none заморозил бы случайные веса")
    tcfg = dataclasses.replace(BCFG.train, pos_weight_max=args.pos_weight_max, face_balance=args.face_balance,
                               contrast_weight=args.contrast_weight,
                               contrast_margin=args.contrast_margin, contrast_tau=args.contrast_tau,
                               repulsive_weight=args.repulsive_weight)
    head_cfg = dataclasses.replace(BCFG.model, repulsive=args.repulsive)
    rep = head_cfg.repulsive_per_layer if args.repulsive else 0
    rank, world, device = base.dist_setup()
    main_rank = rank == 0
    say = print if main_rank else (lambda *a, **k: None)
    if args.accum % world:
        raise SystemExit(f"накопление {args.accum} не делится на {world} процессов")
    torch.manual_seed(args.seed + rank)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    # При разделённом кэше на машине лежит только своя доля деталей, и `split_uids`
    # видит именно её; лимиты тогда тоже на машину, а не на весь прогон.
    share = (lambda n: n // world if args.shard_data and n else n)
    train_uids = split_uids(args.train_split, args.variant)[:share(args.limit)]
    val_uids = split_uids(args.val_split, args.variant)[:share(args.val_limit)]
    # Порог поглощения задан в треугольниках, поэтому при другой цели упрощения
    # он означает другую долю детали (README, «Разрешение»).
    partition = (BCFG.partition if args.min_region is None
                 else dataclasses.replace(BCFG.partition, min_region=args.min_region))
    names = input_names(part_meta(train_uids[0], args.variant)["feature_names"], args.no_jet)
    if names[3:6] != ["nx", "ny", "nz"]:
        raise SystemExit(f"голова границ ждёт нормаль в каналах 3-5, а там {names[3:6]}")
    say(f"train {len(train_uids)}, val {len(val_uids)}, устройство {device}, процессов {world}")

    def loader(uids, shuffle, rotate=False, sampler=None):
        return DataLoader(CachedParts(uids, args.variant, rotate=rotate, absorb_mm=args.absorb_blends_mm,
                                      narrow=args.narrow_rule, no_jet=args.no_jet, random_tile=shuffle),
                          batch_size=1,
                          shuffle=shuffle if sampler is None else None, sampler=sampler,
                          num_workers=args.workers, collate_fn=collate_single,
                          persistent_workers=bool(args.workers))

    # Обучение делится сэмплером, оценка — нарезкой списка: её строки потом собираются
    # со всех процессов, и делить их поровну не обязательно. При разделённом кэше
    # сэмплер не годится — он тасует весь список, а на машине только своя половина;
    # тогда список уже свой, а число деталей на эпоху уравнивается между машинами.
    if args.shard_data:
        sampler = None
        same = base.dist_same_count(len(train_uids), world, device)
        if args.epoch_parts:
            same = min(same, args.epoch_parts // world)
            sampler = EpochSubset(len(train_uids), same, args.seed + rank)
        else:
            train_uids = train_uids[:same]
        say(f"своя доля кэша: {len(train_uids)} деталей train, {len(val_uids)} val; "
            f"на эпоху берётся {same} — поровну с другими машинами")
    else:
        sampler = (DistributedSampler(CachedParts(train_uids, args.variant), num_replicas=world,
                                      rank=rank, shuffle=True, seed=args.seed) if world > 1 else None)
    train_loader = loader(train_uids, True, rotate=CFG.train.rotate, sampler=sampler)
    mine = (lambda items: items) if args.shard_data else (lambda items: items[rank::world])
    val_loader = loader(mine(val_uids), False)
    # --seg-limit — на весь прогон, как --val-limit: без своей доли кэша делится между процессами
    seg_loader = loader(val_uids[:share(args.seg_limit)] if args.shard_data else mine(val_uids[:args.seg_limit]),
                        False)
    model = BoundarySegNet(len(names), CFG.model.emb_size, head_cfg, dropout=CFG.model.dropout).to(device)
    head = head_parameters(model)
    head_ids = {id(p) for p in head}
    optimizer = torch.optim.Adam([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": args.backbone_lr},
        {"params": head, "lr": args.head_lr}], weight_decay=CFG.train.weight_decay)
    # Лосс границ мерит одну голову, IoU — весь конвейер, и они расходятся: в joint50
    # лосс встал на 57-й эпохе, а IoU рос ещё десять (README, «Совместное обучение с нуля»).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min" if args.plateau_on == "edge" else "max", patience=CFG.train.lr_patience)
    say(f"параметров {sum(p.numel() for p in model.parameters()):,}, из них голова границ "
        f"{sum(p.numel() for p in head):,}")

    if main_rank:
        args.out.mkdir(parents=True, exist_ok=True)
    log, seen, best, start = [], 0, -1.0, 0
    if args.resume:
        r = base.resume_state(args.out / "last.pt", model, optimizer, scheduler, generator, args.accum)
        log, seen, best, start = r["log"], r["seen"], r["best"], r["epoch"]
        say(f"продолжаю после эпохи {start}: лучший IoU {best:.4f}", flush=True)
        if args.reset_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = args.reset_lr
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min" if args.plateau_on == "edge" else "max",
                patience=CFG.train.lr_patience)
            say(f"шаг задан заново: {args.reset_lr:.1e}", flush=True)
    elif args.init is None:
        # Вся сеть учится с нуля, тремя лоссами сразу: энкодер с первой эпохи
        # строит признаки и под эмбеддинг, и под стыки. При дообучении поверх
        # готового энкодера он двигался в десять раз медленнее головы и почти
        # не подстраивался под лосс границ.
        say("все веса случайные", flush=True)
    else:
        state = torch.load(args.init, map_location=device)
        if state.get("log_curvature") != CFG.model.log_curvature:
            raise SystemExit(f"{args.init} обучен на другом входе, чем задаёт кэш и конфиг")
        if state["channels"] != len(names):
            drop_channels(state, args.init, names)
            say(f"каналы {args.init} сведены к каналам кэша: {state['channels']} -> {len(names)}", flush=True)
        missing = load_backbone(model, state["model"])
        note = f"голова границ случайная: {len(missing)} тензоров" if missing else "голова границ оттуда же"
        say(f"веса с {args.init} (эпоха {state['epoch']}), {note}", flush=True)
    if args.freeze_backbone:
        # Замороженные параметры в граф не попадают: прямой проход по энкодеру не
        # запоминается, обратный идёт только через голову.
        trainable = {id(p) for p in head_parameters(model)}
        for p in model.parameters():
            p.requires_grad_(id(p) in trainable)
        say(f"энкодер заморожен, учится голова границ: {sum(p.numel() for p in head_parameters(model)):,} "
            "параметров", flush=True)
    ids = [device.index] if device.type == "cuda" else None
    trainee = DistributedDataParallel(model, device_ids=ids) if world > 1 else model
    best_key = "graphv_mean_iou" if BCFG.partition.veto is not None else "graph_mean_iou"

    for epoch in range(start, args.epochs):
        say(f"эпоха {epoch + 1}/{args.epochs}  lr энкодера {optimizer.param_groups[0]['lr']:.1e}, "
            f"головы {optimizer.param_groups[1]['lr']:.1e}", flush=True)
        if sampler is not None:
            sampler.set_epoch(epoch)
        history = run_epoch(trainee, train_loader, optimizer, device, generator,
                            args.log_every, args.accum // world, seen, tcfg, quiet=not main_rank, rep=rep)
        seen += len(train_loader)
        val = reduce_losses(base.dist_gather(
            evaluate(model, val_loader, device, args.seed, reduce=False, tcfg=tcfg, rep=rep), world))
        rows, records, extra = evaluate_segmentation(model, seg_loader, device, args.seed, args.min_face,
                                                     not args.no_mean_shift, reduce=False, cfg=partition,
                                                     rep=rep, mws_limit=args.mws_limit)
        seg = reduce_segmentation(base.dist_gather(rows, world), base.dist_gather(records, world),
                                  {k: base.dist_gather(v, world) for k, v in extra.items()})
        row = {"epoch": epoch + 1, **{f"train_{k}": float(np.mean(v)) for k, v in history.items()},
               **{f"val_{k}": v for k, v in val.items()}, **seg}
        log.append(row)
        say("  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch"), flush=True)
        shift = f"mean-shift {seg['seg_mean_iou']:.3f}, " if "seg_mean_iou" in seg else ""
        mws = (f"; Mutex Watershed {seg['mws_mean_iou']:.3f}, с поглощением мелких {seg['mwsa_mean_iou']:.3f}, "
               f"граф с вето на тех же деталях {seg['graphvs_mean_iou']:.3f}" if "mws_mean_iou" in seg else "")
        say(f"  IoU: {shift}граф {seg['graph_mean_iou']:.3f}, "
            f"граф с вето {seg['graphv_mean_iou']:.3f}{mws}", flush=True)
        scheduler.step(val["edge"] if args.plateau_on == "edge" else seg[best_key])
        if seg[best_key] > best:
            best = seg[best_key]
            if main_rank:
                base.save_checkpoint(args.out / "best.pt", model, epoch + 1, len(names), best)
        # генераторы всех рангов: сбор — коллективная операция, поэтому до проверки ранга
        rank_rng = base.dist_gather([base.rng_state(generator)], world)
        training = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "best": best, "seen": seen, "log": log, "accum": args.accum, **rank_rng[0],
                    "rank_rng": rank_rng}
        if main_rank:
            base.save_checkpoint(args.out / "last.pt", model, epoch + 1, len(names), seg[best_key], training)
            (args.out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
