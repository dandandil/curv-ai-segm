"""Обучение сегментации.

    python scripts/train.py --limit 40 --epochs 6          # проверка, что лосс падает
    python scripts/train.py --lr-find                      # развёртка lr
    python scripts/train.py --epochs 100 --out runs/base
    python scripts/train.py --epochs 100 --out runs/base --resume   # продолжить оборванный

Батч — одна деталь с накоплением градиента: деталь в 30 тысяч треугольников
занимает около пяти гигабайт, две сразу не помещаются (README, «Бюджет»).
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
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import cluster as C  # noqa: E402
from src.config import CFG  # noqa: E402
from src.data import CachedParts, collate_single, part_meta, ring_indices, split_uids  # noqa: E402
from src.losses import triplet_loss, type_loss  # noqa: E402
from src.metrics import MIN_FACE, face_records, grouped_scores, segment_scores, type_miou  # noqa: E402
from src.model import PRIMITIVE_TYPES, PrimitiveSegNet  # noqa: E402


def build_batch(item: dict, device, generator) -> dict:
    """Перенос на устройство и соседство по кольцам — из кэша колец, если он
    собран (`scripts/build_rings.py`), иначе обходом на GPU."""
    x = torch.from_numpy(item["features"]).to(device).unsqueeze(0)
    idx = ring_indices(item, device, generator)
    return {
        "x": x,
        "idx": idx,
        "labels": torch.from_numpy(item["labels"]).to(device),
        "types": torch.from_numpy(item["types"]).to(device),
        "keep": ~torch.from_numpy(item["ambiguous"]).to(device),
    }


def step(model, batch, generator) -> tuple[torch.Tensor, float, float]:
    """Один прямой проход и оба лосса."""
    emb, logits = model(batch["x"], batch["idx"])
    mask = batch["keep"] if CFG.train.exclude_ambiguous else None
    t = triplet_loss(emb[0], batch["labels"], CFG.train.margin, mask, generator)
    c = type_loss(logits[0], batch["types"], mask)
    return t + CFG.train.type_weight * c, float(t.detach()), float(c.detach())


def run_epoch(model, loader, optimizer, device, generator, log_every: int,
              accum: int, seen: int = 0) -> dict:
    """Проход по деталям. `seen` — сколько деталей уже прошло с начала обучения:
    накопление градиента считается сквозным, иначе остаток последней эпохи
    пропадал бы, а при коротком прогоне шаг не делался бы вовсе."""
    model.train()
    history = {"lr": [], "loss": [], "triplet": [], "type": []}
    t0 = time.perf_counter()

    for i, item in enumerate(loader):
        batch = build_batch(item, device, generator)
        loss, t, c = step(model, batch, generator)
        (loss / accum).backward()

        if (seen + i + 1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["loss"].append(float(loss.detach()))
        history["triplet"].append(t)
        history["type"].append(c)
        if log_every and (i + 1) % log_every == 0:
            dt = (time.perf_counter() - t0) / (i + 1)
            print(f"  {i + 1}/{len(loader)}  triplet {np.mean(history['triplet'][-log_every:]):.4f}"
                  f"  type {np.mean(history['type'][-log_every:]):.4f}"
                  f"  {dt * 1000:.0f} мс/деталь"
                  + (f"  {torch.cuda.max_memory_allocated() / 2**30:.1f} ГиБ"
                     if device.type == "cuda" else ""), flush=True)
    return history


def lr_range(model, loader, optimizer, device, generator, accum: int) -> dict:
    """Развёртка скорости обучения: `lr` держится постоянным на группе деталей.

    Первая версия меняла `lr` на каждой детали и не показала ничего: лосс на
    детали зависит от числа её граней сильнее, чем от `lr`, и разброс между
    деталями перекрывал тренд. Здесь точка развёртки — группа в `accum` деталей,
    градиент по ней копится, лосс усредняется, и только потом `lr` растёт.
    """
    model.train()
    history = {"lr": [], "loss": [], "triplet": [], "type": []}
    n_points = len(loader) // accum
    if n_points < 8:
        raise SystemExit(f"на развёртку нужно хотя бы {8 * accum} деталей")
    ratio = CFG.train.lr_find_max / CFG.train.lr_find_min
    buffer: list[tuple[float, float, float]] = []
    optimizer.zero_grad(set_to_none=True)

    for i, item in enumerate(loader):
        point = i // accum
        if point >= n_points:
            break
        lr = CFG.train.lr_find_min * ratio ** (point / (n_points - 1))
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch = build_batch(item, device, generator)
        loss, t, c = step(model, batch, generator)
        (loss / accum).backward()
        buffer.append((float(loss.detach()), t, c))

        if (i + 1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            mean = np.mean(buffer, axis=0)
            buffer = []
            history["lr"].append(lr)
            history["loss"].append(float(mean[0]))
            history["triplet"].append(float(mean[1]))
            history["type"].append(float(mean[2]))
            if (point + 1) % 20 == 0:
                print(f"  {point + 1}/{n_points}  lr {lr:.2e}  лосс {mean[0]:.4f}", flush=True)
    return history


def report_lr_range(history: dict, window: int = 5) -> None:
    """Где лосс ниже всего и где он начинает расходиться."""
    lr = np.array(history["lr"])
    loss = np.array(history["loss"])
    smooth = np.convolve(loss, np.ones(window) / window, mode="valid")
    axis = lr[window - 1:]
    best = int(np.argmin(smooth))
    print(f"\nминимум сглаженного лосса: lr {axis[best]:.2e}, лосс {smooth[best]:.4f}")
    # Расхождение: первая точка ПОСЛЕ минимума, где лосс поднялся заметно выше него.
    after = np.where((axis > axis[best]) & (smooth > smooth.min() * 1.3))[0]
    if after.size:
        print(f"расхождение начинается около lr {axis[after[0]]:.2e}")
        print(f"рабочее значение — на порядок ниже: lr {axis[after[0]] / 10:.2e}")
    else:
        print("расхождения не видно: развёртка снова неинформативна, "
              "нужен более широкий диапазон или больше деталей на точку")


@torch.no_grad()
def evaluate(model, loader, device, seed: int) -> dict:
    """Те же лоссы на отложенных деталях. Кластеризация сюда не входит: она
    только на инференсе, и мерить по ней каждую эпоху слишком дорого.

    Генератор свой и заново засеянный: соседи в кольце выбираются случайно, и
    с общим генератором валидация меряла бы ещё и эту случайность, а разница
    между эпохами перестала бы читаться.
    """
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    t_all, c_all, miou = [], [], []
    for item in loader:
        batch = build_batch(item, device, generator)
        emb, logits = model(batch["x"], batch["idx"])
        mask = batch["keep"] if CFG.train.exclude_ambiguous else None
        t_all.append(float(triplet_loss(emb[0], batch["labels"], CFG.train.margin,
                                        mask, generator)))
        c_all.append(float(type_loss(logits[0], batch["types"], mask)))
        miou.append(type_miou(logits[0], batch["types"], len(PRIMITIVE_TYPES)))
    return {"triplet": float(np.mean(t_all)), "type": float(np.mean(c_all)),
            "type_miou": float(np.mean(miou))}


@torch.no_grad()
def evaluate_segmentation(model, loader, device, seed: int, min_face: int) -> dict:
    """Метрики по итоговой нарезке: кластеризация и сравнение с разметкой.

    Считается на небольшом наборе, потому что здесь работает mean-shift — около
    2 секунд на деталь против 0.4 на обычный проход. Это единственная величина,
    которая меряет то, ради чего всё делается, поэтому по ней и выбирается
    лучший чекпоинт.

    Грани мельче `min_face` треугольников выбрасываются из счёта: их не
    восстановит никакая кластеризация, и они дают стабильный процент
    «потерянных», ничего не говоря о качестве.
    """
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    rows, records = [], []
    for item in loader:
        batch = build_batch(item, device, generator)
        embedding, _ = model(batch["x"], batch["idx"])
        pred, _ = C.segment(embedding[0], CFG.cluster.quantile, CFG.cluster.iterations,
                            CFG.cluster.bandwidth_samples, CFG.cluster.bandwidth_floor,
                            CFG.cluster.chunk, CFG.cluster.max_clusters,
                            width=CFG.cluster.bandwidth)
        labels = batch["labels"]
        big = torch.bincount(labels)[labels] >= min_face
        # Оракул: треугольник к ближайшему центру своей истинной грани. Без
        # mean-shift и ширины ядра — меряет, разделимы ли грани в эмбеддинге.
        z = F.normalize(embedding[0].t(), dim=1)
        faces, inverse = labels.unique(return_inverse=True)
        centers = F.normalize(torch.zeros(len(faces), z.shape[1], device=z.device)
                              .index_add_(0, inverse, z), dim=1)
        oracle = faces[(z @ centers.t()).argmax(1)]
        rows.append({**segment_scores(labels[big], pred[big]),
                     "oracle_iou": segment_scores(labels[big], oracle[big])["mean_iou"]})
        records += face_records(labels[big], pred[big], batch["types"][big], item["halfwidth_mm"],
                                PRIMITIVE_TYPES)
    return {**{k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, **grouped_scores(records)}


def save_checkpoint(path: Path, model, epoch: int, channels: int, score: float,
                    training: dict | None = None) -> None:
    """Веса, эпоха и достигнутое значение метрики — чтобы потом было видно,
    откуда чекпоинт взялся. `training` — всё, что нужно для продолжения
    прогона; пишется только в `last.pt`.

    Запись через временный файл: оборванная на полпути запись испортила бы
    `last.pt`, а с ним и возможность продолжить.
    """
    state = {"model": model.state_dict(), "epoch": epoch, "channels": channels,
             "score": score, "log_curvature": CFG.model.log_curvature}
    if training is not None:
        state["training"] = training
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def dist_setup() -> tuple[int, int, torch.device]:
    """Ранг, число процессов и устройство. Без torchrun — обычное обучение на одной карте.

    Процессов ровно по числу карт: одна деталь на карту, градиенты складываются
    перед шагом оптимизатора (README, «Обучение на двух машинах»).
    """
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    return dist.get_rank(), dist.get_world_size(), device


def dist_same_count(n: int, world: int, device: torch.device) -> int:
    """Наименьшее число деталей среди процессов.

    Шаги оптимизатора должны совпасть у всех: иначе на хвосте эпохи один процесс
    ещё складывает градиент, другой уже собирает оценку, и NCCL валит обоих на
    рассогласовании коллективных операций (README, «Обучение на двух машинах»).
    """
    if world == 1:
        return n
    count = torch.tensor([n], device=device)
    dist.all_reduce(count, op=dist.ReduceOp.MIN)
    return int(count.item())


def dist_gather(rows: list, world: int) -> list:
    """Собрать списки со всех процессов в один: оценка делится между картами."""
    if world == 1:
        return rows
    buf = [None] * world
    dist.all_gather_object(buf, rows)
    return [row for part in buf for row in part]


def rng_state(generator) -> dict:
    """Генераторы случайности процесса — для продолжения прогона (`resume_state`)."""
    state = {"torch_rng": torch.get_rng_state(), "generator": generator.get_state()}
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


def resume_state(path: Path, model, optimizer, scheduler, generator, accum: int) -> dict:
    """Восстановить прогон из `last.pt`: веса, оптимизатор, планировщик, генераторы
    случайности и счётчики. Возвращает эпоху, лучшую метрику, счётчик деталей и лог.

    Накопление градиента обязано совпадать: счётчик деталей с начала обучения
    отмеряет шаги оптимизатора, и с другим накоплением он бы их сместил. Градиент,
    накопленный на хвосте последней эпохи (9461 деталь не делится на 8), при
    продолжении теряется — это пять деталей из десяти тысяч.
    """
    # На нескольких машинах чекпоинт читает только нулевой ранг и раздаёт остальным:
    # у других рангов своего last.pt может не быть вовсе или он устарел, и тогда
    # прогон либо висел бы на рандеву, либо молча продолжался бы с разных весов.
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if distributed else 0
    # Отсутствие чекпоинта тоже раздаётся: иначе нулевой ранг падал бы один, а
    # остальные вечно ждали бы раздачи.
    state = torch.load(path, map_location="cpu") if rank == 0 and path.exists() else None
    if distributed:
        box = [state]
        dist.broadcast_object_list(box, src=0)
        state = box[0]
    if state is None:
        raise SystemExit(f"продолжать нечего: нет {path}")
    training = state.get("training")
    if training is None:
        raise SystemExit(f"{path} сохранён без состояния оптимизатора (до появления --resume): "
                         "продолжить его нельзя, только начать заново")
    if state.get("log_curvature") != CFG.model.log_curvature:
        raise SystemExit("чекпоинт обучен с другим преобразованием входа, чем задаёт конфиг")
    if training["accum"] != accum:
        raise SystemExit(f"чекпоинт обучался с накоплением {training['accum']}, "
                         f"а задано {accum}: счётчик шагов оптимизатора сместится")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(training["optimizer"])
    scheduler.load_state_dict(training["scheduler"])
    # Генераторы у каждого ранга свои, иначе все ранги тасовали бы и поворачивали детали одной
    # последовательностью. В чекпоинтах до `rank_rng` записаны только генераторы нулевого ранга,
    # и остальные начинают заново со своего семени.
    per_rank = training.get("rank_rng")
    own = per_rank[rank] if per_rank and rank < len(per_rank) else (training if rank == 0 else None)
    if own is not None:
        torch.set_rng_state(own["torch_rng"].cpu())
        generator.set_state(own["generator"].cpu())
        if "cuda_rng" in own and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in own["cuda_rng"]])
    return {"epoch": state["epoch"], "best": training["best"],
            "seen": training["seen"], "log": training["log"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="noisy")
    ap.add_argument("--limit", type=int, help="только первые N деталей train")
    ap.add_argument("--val-limit", type=int, default=40)
    ap.add_argument("--seg-limit", type=int, default=16,
                    help="деталей для метрик сегментации, там работает mean-shift")
    ap.add_argument("--min-face", type=int, default=MIN_FACE,
                    help="грани мельче стольких треугольников из метрик выбрасываются")
    ap.add_argument("--epochs", type=int, default=CFG.train.epochs)
    ap.add_argument("--lr", type=float, default=CFG.train.lr)
    ap.add_argument("--lr-find", action="store_true", help="развёртка lr вместо обучения")
    ap.add_argument("--accum", type=int, default=CFG.train.accum,
                    help="деталей на шаг оптимизатора")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "last")
    ap.add_argument("--resume", action="store_true",
                    help="продолжить прогон из OUT/last.pt; --lr при этом не действует")
    ap.add_argument("--set-lr", type=float,
                    help="при --resume заменить lr из чекпоинта: дообучение с меньшим шагом")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    train_uids = split_uids("train", args.variant)[:args.limit]
    val_uids = split_uids("val", args.variant)[:args.val_limit]
    if not train_uids:
        raise SystemExit(f"кэш варианта {args.variant!r} пуст — нечего обучать")
    print(f"train {len(train_uids)}, val {len(val_uids)}, устройство {device}")

    def loader(uids, shuffle, rotate=False):
        return DataLoader(CachedParts(uids, args.variant, rotate=rotate), batch_size=1, shuffle=shuffle,
                          num_workers=args.workers, collate_fn=collate_single,
                          persistent_workers=bool(args.workers))

    # Повороты только на обучении: на валидации они добавили бы шум в метрики.
    train_loader = loader(train_uids, True, rotate=CFG.train.rotate)
    val_loader = loader(val_uids, False)
    seg_loader = loader(val_uids[:args.seg_limit], False)
    n_channels = len(part_meta(train_uids[0], args.variant)["feature_names"])
    model = PrimitiveSegNet(n_channels, CFG.model.emb_size, dropout=CFG.model.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=CFG.train.weight_decay)
    optimizer.zero_grad(set_to_none=True)
    print(f"каналов {n_channels}, параметров {sum(p.numel() for p in model.parameters()):,}")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.lr_find:
        print(f"развёртка: {len(train_loader) // args.accum} точек по {args.accum} деталей, "
              f"lr от {CFG.train.lr_find_min:.0e} до {CFG.train.lr_find_max:.0e}")
        history = lr_range(model, train_loader, optimizer, device, generator, args.accum)
        (args.out / "lr_find.json").write_text(json.dumps(history, indent=1))
        report_lr_range(history)
        print(f"развёртка записана в {args.out / 'lr_find.json'}")
        return

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=CFG.train.lr_patience)
    log, seen, best, start = [], 0, -1.0, 0
    if args.resume:
        r = resume_state(args.out / "last.pt", model, optimizer, scheduler, generator, args.accum)
        if args.set_lr is not None:
            # Состояние Adam и счётчики остаются, меняется только шаг.
            for group in optimizer.param_groups:
                group["lr"] = args.set_lr
        log, seen, best, start = r["log"], r["seen"], r["best"], r["epoch"]
        print(f"продолжаю после эпохи {start}: лучший IoU {best:.4f}, "
              f"lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)
        if start >= args.epochs:
            raise SystemExit(f"уже пройдено {start} эпох из {args.epochs}")

    for epoch in range(start, args.epochs):
        print(f"эпоха {epoch + 1}/{args.epochs}  lr {optimizer.param_groups[0]['lr']:.2e}")
        history = run_epoch(model, train_loader, optimizer, device, generator,
                            args.log_every, args.accum, seen)
        seen += len(train_loader)
        val = evaluate(model, val_loader, device, args.seed) if val_uids else {}
        seg = evaluate_segmentation(model, seg_loader, device, args.seed,
                                    args.min_face) if val_uids else {}
        row = {"epoch": epoch + 1,
               "train_triplet": float(np.mean(history["triplet"])),
               "train_type": float(np.mean(history["type"])),
               **{f"val_{k}": v for k, v in val.items()},
               **{f"seg_{k}": v for k, v in seg.items()}}
        log.append(row)
        print("  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch"), flush=True)

        if seg:
            # Планировщик — по эмбеддинг-лоссу, как в оригинале. Выбор весов —
            # по IoU сегментации: лосс на нашей стадии сидит у margin и почти не
            # реагирует, а нарезка деталей — это ровно то, что нам нужно.
            scheduler.step(val["triplet"])
            if seg["mean_iou"] > best:
                best = seg["mean_iou"]
                save_checkpoint(args.out / "best.pt", model, epoch + 1, n_channels, best)
                print(f"  улучшение: IoU {best:.4f}, протечек {seg['leaks']:.1f}, "
                      f"лишних разрезов {seg['splits']:.1f} — чекпоинт сохранён", flush=True)
        training = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "best": best, "seen": seen, "log": log, "accum": args.accum, **rng_state(generator)}
        save_checkpoint(args.out / "last.pt", model, epoch + 1, n_channels,
                        seg.get("mean_iou", float("nan")), training)
        (args.out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
