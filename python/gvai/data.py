"""Reading the Rust engine's self-play format, and batching it for training.

The format (``GVAI``, little-endian) is written by ``src/selfplay.rs``:

    Header (28 bytes)
      magic           [u8; 4]  b"GVAI"
      version         u32      1
      num_examples    u32      positions in this file
      num_nodes       u32      board intersections
      num_features    u32      features per node
      num_edge_pairs  u32      directed edges (each a [src, dst] pair)
      action_size     u32      num_nodes + 1, the extra slot being pass

    Shared graph, once per file
      edges           [u32; num_edge_pairs * 2]

    Per example, repeated num_examples times
      node_features   [f32; num_nodes * num_features]   row-major
      policy          [f32; action_size]                 MCTS visit shares
      value           f32                                outcome for the
                                                         player to move

One file is one game, so the graph is stored once rather than per position.
Boards differ in size and topology between files, which is the whole point:
batching pads to the largest board present and carries a node mask.
"""

from __future__ import annotations

import pathlib
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"GVAI"
SUPPORTED_VERSION = 1
HEADER = struct.Struct("<4s6I")


class FormatError(ValueError):
    pass


@dataclass
class Game:
    """One self-play game: a shared board graph plus a position per move."""

    path: pathlib.Path
    num_nodes: int
    num_features: int
    action_size: int
    edges: np.ndarray  # [E, 2] uint32, directed (both directions present)
    node_features: np.ndarray  # [T, num_nodes, num_features] float32
    policies: np.ndarray  # [T, action_size] float32
    values: np.ndarray  # [T] float32

    def __len__(self) -> int:
        return self.values.shape[0]


def read_game(path: str | pathlib.Path) -> Game:
    """Parse one .bin file. Raises FormatError on anything unexpected."""
    path = pathlib.Path(path)
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise FormatError(f"{path}: too short to hold a header")

    magic, version, num_examples, num_nodes, num_features, num_edge_pairs, action_size = (
        HEADER.unpack_from(raw, 0)
    )
    if magic != MAGIC:
        raise FormatError(f"{path}: bad magic {magic!r}")
    if version != SUPPORTED_VERSION:
        raise FormatError(f"{path}: version {version}, expected {SUPPORTED_VERSION}")
    if action_size != num_nodes + 1:
        raise FormatError(
            f"{path}: action_size {action_size} is not num_nodes + 1 ({num_nodes + 1})"
        )

    offset = HEADER.size
    edge_count = num_edge_pairs * 2
    edges = np.frombuffer(raw, dtype="<u4", count=edge_count, offset=offset)
    edges = edges.reshape(-1, 2)
    offset += edge_count * 4

    feature_count = num_nodes * num_features
    stride = feature_count * 4 + action_size * 4 + 4
    expected = offset + num_examples * stride
    if len(raw) != expected:
        raise FormatError(
            f"{path}: expected {expected} bytes for {num_examples} examples, found {len(raw)}"
        )

    features = np.empty((num_examples, num_nodes, num_features), dtype=np.float32)
    policies = np.empty((num_examples, action_size), dtype=np.float32)
    values = np.empty((num_examples,), dtype=np.float32)

    for i in range(num_examples):
        base = offset + i * stride
        features[i] = np.frombuffer(
            raw, dtype="<f4", count=feature_count, offset=base
        ).reshape(num_nodes, num_features)
        policies[i] = np.frombuffer(
            raw, dtype="<f4", count=action_size, offset=base + feature_count * 4
        )
        values[i] = np.frombuffer(
            raw, dtype="<f4", count=1, offset=base + feature_count * 4 + action_size * 4
        )[0]

    if edges.size and int(edges.max()) >= num_nodes:
        raise FormatError(f"{path}: edge references node >= num_nodes")

    return Game(
        path=path,
        num_nodes=int(num_nodes),
        num_features=int(num_features),
        action_size=int(action_size),
        edges=edges,
        node_features=features,
        policies=policies,
        values=values,
    )


def normalized_adjacency(num_nodes: int, edges: np.ndarray) -> np.ndarray:
    """Symmetrically normalized adjacency with self-loops: D^-1/2 (A + I) D^-1/2.

    This is the GCN propagation matrix. **The Rust side must build the exact
    same matrix**, or a trained net will see different inputs at play time than
    it saw during training; `tests/` on the Rust side checks one against a
    fixture generated here.

    `edges` is a directed list containing both directions, so A is symmetric
    by construction; duplicates are collapsed.
    """
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    if edges.size:
        src = edges[:, 0].astype(np.int64)
        dst = edges[:, 1].astype(np.int64)
        adjacency[src, dst] = 1.0
        adjacency[dst, src] = 1.0
    np.fill_diagonal(adjacency, 1.0)

    degree = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    return (adjacency * inv_sqrt[:, None] * inv_sqrt[None, :]).astype(np.float32)


def find_games(roots: list[str | pathlib.Path]) -> list[pathlib.Path]:
    """All .bin files under the given directories (or the files themselves)."""
    found: list[pathlib.Path] = []
    for root in roots:
        root = pathlib.Path(root)
        if root.is_file():
            found.append(root)
        elif root.is_dir():
            found.extend(sorted(root.rglob("*.bin")))
    return found


@dataclass
class Example:
    node_features: np.ndarray  # [N, F]
    adjacency: np.ndarray  # [N, N]
    policy_nodes: np.ndarray  # [N]
    policy_pass: float
    value: float


class SelfPlayData:
    """Every position from every game, with adjacency shared per game.

    Positions are held in memory; a few hundred games of 9x9 is tens of
    megabytes, so there is no reason to stream.
    """

    def __init__(self, roots: list[str | pathlib.Path], limit_games: int | None = None):
        paths = find_games(roots)
        if limit_games is not None:
            paths = paths[:limit_games]
        if not paths:
            raise FileNotFoundError(f"no .bin self-play files under {roots}")

        self.games: list[Game] = []
        self.adjacencies: list[np.ndarray] = []
        self.index: list[tuple[int, int]] = []

        features_seen = set()
        for path in paths:
            game = read_game(path)
            features_seen.add(game.num_features)
            adjacency = normalized_adjacency(game.num_nodes, game.edges)
            game_id = len(self.games)
            self.games.append(game)
            self.adjacencies.append(adjacency)
            self.index.extend((game_id, t) for t in range(len(game)))

        if len(features_seen) != 1:
            raise FormatError(
                f"files disagree on feature count: {sorted(features_seen)}"
            )
        self.num_features = features_seen.pop()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Example:
        game_id, t = self.index[i]
        game = self.games[game_id]
        policy = game.policies[t]
        return Example(
            node_features=game.node_features[t],
            adjacency=self.adjacencies[game_id],
            policy_nodes=policy[:-1],
            policy_pass=float(policy[-1]),
            value=float(game.values[t]),
        )

    def policy_entropy(self) -> float:
        """Mean entropy of the target policies, in nats.

        This is the **floor** the policy cross-entropy can reach: the targets
        are MCTS visit distributions, not one-hot labels, and
        H(target, prediction) >= H(target). A policy loss near this number
        means the net has fit the data, not that training has stalled. Worth
        printing, because "why won't the loss go to zero" is otherwise an easy
        hour to lose.
        """
        total = 0.0
        for i in range(len(self)):
            example = self[i]
            probs = np.concatenate([example.policy_nodes, [example.policy_pass]])
            probs = probs[probs > 0.0]
            total += float(-(probs * np.log(probs)).sum())
        return total / max(len(self), 1)

    def board_sizes(self) -> dict[int, int]:
        """Positions per board size, for reporting what a dataset contains."""
        counts: dict[int, int] = {}
        for game_id, _ in self.index:
            n = self.games[game_id].num_nodes
            counts[n] = counts.get(n, 0) + 1
        return dict(sorted(counts.items()))


def collate(batch: list[Example]) -> dict:
    """Pad a batch to its largest board and build a node mask.

    Padding is what lets one model train on mixed topologies in a single
    batch: padded nodes are masked out of the policy and excluded from the
    pooled value features, and their adjacency rows are zero.
    """
    import torch

    max_nodes = max(example.node_features.shape[0] for example in batch)
    num_features = batch[0].node_features.shape[1]
    size = len(batch)

    x = np.zeros((size, max_nodes, num_features), dtype=np.float32)
    a = np.zeros((size, max_nodes, max_nodes), dtype=np.float32)
    mask = np.zeros((size, max_nodes), dtype=np.float32)
    policy_nodes = np.zeros((size, max_nodes), dtype=np.float32)
    policy_pass = np.zeros((size,), dtype=np.float32)
    values = np.zeros((size,), dtype=np.float32)

    for i, example in enumerate(batch):
        n = example.node_features.shape[0]
        x[i, :n] = example.node_features
        a[i, :n, :n] = example.adjacency
        mask[i, :n] = 1.0
        policy_nodes[i, :n] = example.policy_nodes
        policy_pass[i] = example.policy_pass
        values[i] = example.value

    return {
        "x": torch.from_numpy(x),
        "a": torch.from_numpy(a),
        "mask": torch.from_numpy(mask),
        "policy_nodes": torch.from_numpy(policy_nodes),
        "policy_pass": torch.from_numpy(policy_pass),
        "value": torch.from_numpy(values),
    }
