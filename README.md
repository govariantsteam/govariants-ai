# govariants-ai

AlphaZero-style training engine for [Go Variants](https://github.com/govariantsteam/govariants). Trains AI opponents that can play across any board topology — standard grids, cubes, circular boards, and more.

## Why build this?

The govariants project has 19+ game variants. We evaluated existing frameworks:

- **OpenSpiel** (DeepMind) — great game abstraction layer, but its AlphaZero implementation is self-described as "illustrative" and not designed for scale. No one has trained Go with it.
- **kZero** — good Rust architecture for fast self-play, but dormant solo project since mid-2024.
- **alpha-zero-general** — unmaintained Python project, too slow for anything beyond Connect Four.
- **KataGo** — excellent for standard Go, but can't handle non-standard board topologies or variant rules.

None of these fit a project that needs AI across many variants with different board topologies. So we're building our own.

## Architecture

```
Rust (fast self-play)          Python (training)
┌─────────────────────┐       ┌──────────────────────┐
│ Game trait           │       │ GNN model (PyG)      │
│ ├─ Baduk             │       │ Training loop         │
│ ├─ Capture Go        │  ──>  │ ONNX export           │
│ ├─ Cube Go           │ data/ │                      │
│ └─ ...               │       └──────────┬───────────┘
│                      │  <──             │
│ MCTS + Self-play     │ models/          │
└─────────────────────┘       ┌──────────┴───────────┐
                              │ ONNX model served in  │
                              │ govariants (Node/browser) │
                              └──────────────────────┘
```

- **Rust**: game rules, MCTS search, parallel self-play. Fast enough to generate training data on modest hardware.
- **Python/PyTorch Geometric**: GNN architecture, training loop, ONNX export.
- **ONNX**: trained models served in govariants via onnxruntime-node or onnxruntime-web.

## Why a GNN (not a CNN)?

Standard AlphaZero uses convolutional neural networks, which assume a 2D grid. That breaks for cube boards, circular boards, Sierpinski boards, etc.

A **Graph Neural Network** treats every board as a graph — nodes are intersections, edges are adjacency. This means:

- **One architecture for all topologies**: grids, cubes, circles, arbitrary graphs
- **Variable board sizes for free**: GNNs naturally handle different-sized graphs
- **Transfer learning across topologies**: a model trained on standard Go learns about adjacency relationships (not spatial positions), so it can fine-tune to cube Go or circular Go

The tradeoff is that GNNs are less proven for Go than CNNs and may be somewhat weaker on standard grids. For our goal of "fun to play against" (not superhuman), this is a good trade.

## What's built so far

### Game trait (`src/game.rs`)

Core interface that all variants implement. Exposes graph structure for the GNN:

```rust
trait Game: Clone + Send {
    // Game logic
    fn legal_moves(&self) -> Vec<Self::Move>;
    fn play_move(&mut self, player: usize, mv: Self::Move);
    fn is_terminal(&self) -> bool;
    fn result(&self) -> Option<GameResult>;

    // Graph structure for GNN
    fn num_nodes(&self) -> usize;
    fn adjacency(&self) -> Vec<[usize; 2]>;
    fn node_features(&self) -> Vec<Vec<f32>>;

    // Action space
    fn action_size(&self) -> usize;
    fn move_to_index(&self, mv: Self::Move) -> usize;
    fn index_to_move(&self, idx: usize) -> Self::Move;
}
```

### Baduk (`src/games/baduk.rs`)

Standard Go implementation: stone placement, capture logic, positional superko, area scoring. Configurable board size (default 9x9).

Node features: `[is_current_player, is_opponent, is_empty, normalized_liberty_count]`

### MCTS (`src/mcts.rs`)

Monte Carlo Tree Search with PUCT selection and a pluggable `Evaluator` trait. Currently uses random rollouts; will use the GNN once the Python side is ready.

```rust
trait Evaluator<G: Game> {
    fn evaluate(&self, game: &G) -> (Vec<f32>, f32); // (policy, value)
}
```

### Self-play pipeline (`src/selfplay.rs`)

Parallel self-play engine using rayon. Plays MCTS games and serializes training data for the Python training pipeline. Dirichlet noise is applied at the MCTS root for exploration.

Each game produces a `.bin` file in the output directory containing all training examples (one per position).

## Building and running

```bash
cargo build
cargo test

# Run self-play (use --release for real runs)
cargo run --release -- --games 100 --simulations 200

# Quick test run on a small board
cargo run -- --games 5 --simulations 30 --width 5 --height 5 --rollouts 2
```

See all options with `cargo run -- --help`.

## Training data format

Self-play writes `.bin` files to `data/` (one per game). Format is custom binary (`GVAI`), little-endian:

```
Header (28 bytes):
  magic:          [u8; 4]   "GVAI"
  version:        u32       1
  num_examples:   u32       positions in this game
  num_nodes:      u32       board intersections (e.g., 81 for 9x9)
  num_features:   u32       features per node (4 for Baduk)
  num_edge_pairs: u32       directed edges in graph
  action_size:    u32       legal action slots (num_nodes + 1 for pass)

Shared graph (written once per file):
  edges:          [u32; num_edge_pairs * 2]   flattened [src, dst] pairs

Per example (repeated num_examples times):
  node_features:  [f32; num_nodes * num_features]   row-major
  policy:         [f32; action_size]                 MCTS visit distribution
  value:          f32                                game outcome for current player (+1/-1/0)
```

Python reading example:
```python
import numpy as np

with open("data/game_000000.bin", "rb") as f:
    magic = f.read(4)
    assert magic == b"GVAI"
    header = np.frombuffer(f.read(24), dtype=np.uint32)
    version, num_examples, num_nodes, num_features, num_edges, action_size = header

    edges = np.frombuffer(f.read(num_edges * 2 * 4), dtype=np.uint32).reshape(-1, 2)

    for _ in range(num_examples):
        node_features = np.frombuffer(f.read(num_nodes * num_features * 4), dtype=np.float32).reshape(num_nodes, num_features)
        policy = np.frombuffer(f.read(action_size * 4), dtype=np.float32)
        value = np.frombuffer(f.read(4), dtype=np.float32)[0]
```

## Training strategy

1. Train a base model on standard 9x9 Go
2. Fine-tune for each variant from the base model
3. For variants very far from Go (quantum, phantom), train from scratch on smaller boards

## Roadmap

See [open issues](https://github.com/govariantsteam/govariants-ai/issues) for the full roadmap:

1. ~~Self-play data pipeline (#1)~~ ✅
2. GNN model and training loop (#2) ← **next up**
3. Full training loop integration (#3)
4. Train and validate 9x9 Go model (#4)
5. Port variant game rules to Rust (#5)
6. Serving integration with govariants (#6)
