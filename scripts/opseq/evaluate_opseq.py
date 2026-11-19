"""Оценка модели операций на val или test: доли верных ответов голов при teacher forcing.

    python scripts/opseq/evaluate_opseq.py --run runs/opseq_base --split val
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "opseq"))

from train_opseq import evaluate, graph_uids, loader_of  # noqa: E402

from src.opseq.data import Bodies, split_uids  # noqa: E402
from src.opseq.model import OpseqNet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--corpora", nargs="+", help="корпуса вместо data.train_corpora")
    args = ap.parse_args()
    device = torch.device("cuda")
    model = OpseqNet().to(device)
    model.load_state_dict(torch.load(ROOT / args.run / args.checkpoint, map_location=device)["model"])
    loader = loader_of(Bodies(split_uids(graph_uids(args.corpora))[args.split]), False, args.workers)
    loss, parts, rates = evaluate(model, loader, device)
    print(json.dumps({"split": args.split, "тел": len(loader.dataset), "loss": loss, "головы": parts,
                      "верно": rates}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
