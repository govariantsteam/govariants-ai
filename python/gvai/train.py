"""Train the graph network on self-play data from the Rust engine.

    python -m gvai.train --data ../data/grid9 ../data/cube3 --epochs 20

Mixed topologies in one run are the point: the model takes the adjacency
matrix as an input, so grids, tori and cube surfaces train the same weights.

`--overfit N` is the wiring check. If the model, loss and batching are wired
correctly it should drive the loss on N positions to near zero; if it cannot,
something is wrong with the plumbing and no amount of real training will help.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import time

import torch

from .data import SelfPlayData, collate
from .model import GraphNet, ModelConfig, policy_value_loss, top1_agreement


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def iterate_batches(
    data: SelfPlayData, indices: list[int], batch_size: int, shuffle: bool
):
    order = list(indices)
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        if chunk:
            yield collate([data[i] for i in chunk])


@torch.no_grad()
def evaluate(
    net: GraphNet, data: SelfPlayData, indices: list[int], batch_size: int, device
) -> dict:
    net.eval()
    totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "top1": 0.0}
    seen = 0
    for batch in iterate_batches(data, indices, batch_size, shuffle=False):
        batch = to_device(batch, device)
        size = batch["value"].shape[0]
        node_logits, pass_logit, value = net(batch["x"], batch["a"], batch["mask"])
        loss, policy_loss, value_loss = policy_value_loss(
            node_logits, pass_logit, value, batch
        )
        totals["loss"] += loss.item() * size
        totals["policy"] += policy_loss.item() * size
        totals["value"] += value_loss.item() * size
        totals["top1"] += top1_agreement(node_logits, pass_logit, batch).item() * size
        seen += size
    return {key: total / max(seen, 1) for key, total in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", nargs="+", required=True, help="directories of .bin self-play files"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit-games", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overfit",
        type=int,
        default=None,
        help="train on this many positions only, as a wiring check",
    )
    parser.add_argument("--out", default="models/gvai.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = SelfPlayData(args.data, limit_games=args.limit_games)
    print(f"{len(data)} positions from {len(data.games)} games")
    print(f"positions per board size: {data.board_sizes()}")
    floor = data.policy_entropy()
    print(
        f"policy loss floor (mean target entropy): {floor:.3f} "
        "- cross-entropy to a soft target cannot go below this"
    )

    all_indices = list(range(len(data)))
    random.shuffle(all_indices)

    if args.overfit:
        train_indices = all_indices[: args.overfit]
        val_indices = train_indices
        print(f"overfit mode: {len(train_indices)} positions, train == val")
    else:
        split = max(1, int(len(all_indices) * args.val_fraction))
        val_indices = all_indices[:split]
        train_indices = all_indices[split:]
        print(f"train {len(train_indices)} / val {len(val_indices)}")

    device = pick_device(args.device)
    config = ModelConfig(
        num_features=data.num_features, hidden=args.hidden, layers=args.layers
    )
    net = GraphNet(config).to(device)
    print(
        f"device {device} | {sum(p.numel() for p in net.parameters())} parameters "
        f"| hidden {config.hidden} x {config.layers} layers"
    )

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best = float("inf")
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        net.train()
        started = time.time()
        running = count = 0.0
        for batch in iterate_batches(
            data, train_indices, args.batch_size, shuffle=True
        ):
            batch = to_device(batch, device)
            node_logits, pass_logit, value = net(batch["x"], batch["a"], batch["mask"])
            loss, _, _ = policy_value_loss(
                node_logits, pass_logit, value, batch, args.value_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            size = batch["value"].shape[0]
            running += loss.item() * size
            count += size

        stats = evaluate(net, data, val_indices, args.batch_size, device)
        print(
            f"epoch {epoch:3d}  train {running / max(count, 1):.4f}  "
            f"val {stats['loss']:.4f} (policy {stats['policy']:.4f}, "
            f"value {stats['value']:.4f})  top1 {stats['top1'] * 100:5.1f}%  "
            f"{time.time() - started:.1f}s"
        )

        if stats["loss"] < best:
            best = stats["loss"]
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "config": config.to_dict(),
                    "val": stats,
                    "epoch": epoch,
                },
                out_path,
            )

    print(f"best val loss {best:.4f}; checkpoint written to {out_path}")
    print(json.dumps({"best_val_loss": best}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
