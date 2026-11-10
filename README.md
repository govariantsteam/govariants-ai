# govariants-ai

AlphaZero-style training engine for [Go Variants](https://github.com/govariantsteam/govariants). Trains AI opponents that can play across any board topology — standard grids, cubes, circular boards, and more.

There is no neural net yet, but there is a bot you can play right now: `cargo run --release -- play --board cube:3`.

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

### Board topologies (`src/games/topology.rs`)

Because the rules only consult adjacency, a board is just a graph. Three topologies ship today:

| Topology | Spec | Nodes | Notes |
|----------|------|-------|-------|
| Grid | `9x9`, `grid:5x7` | w×h | standard board; corners and edges |
| Torus | `torus:7x7` | w×h | wraps on both axes, 4-regular, no edges or corners |
| Cube | `cube:3` | 6n² | surface of a cube, 4-regular, stitched across faces |

Cube adjacency is **derived from 3D cell-center positions**, not hand-stitched face edges: two cells are neighbors iff their centers sit a half-step apart across a cube edge, or one step apart within a face. That removes the orientation bookkeeping that makes cube boards error-prone, and it yields 3D coordinates for free — which is exactly what edge features (#8) need.

Adding a topology means writing one adjacency builder. The rules, search, and self-play pipeline need no changes.

### Baduk (`src/games/baduk.rs`)

Go on an arbitrary board graph: stone placement, capture, suicide, simple ko, positional superko, and area scoring — all expressed purely in terms of adjacency, so the same engine plays every topology above.

Node features: `[is_current_player, is_opponent, is_empty, normalized_liberty_count]`

### MCTS (`src/mcts.rs`)

Monte Carlo Tree Search with PUCT selection and a pluggable `Evaluator` trait. Values are stored from the perspective of the player to move at each node and negated on the way back up; terminal positions are scored by the parent, which still knows whose turn it was.

```rust
trait Evaluator<G: Game> {
    fn evaluate(&self, game: &G) -> (Vec<f32>, f32); // (policy, value)
}
```

An evaluator may assign a prior of zero to veto a move, and the tree prunes it. The heuristic evaluator uses this to refuse to fill its own eyes.

### Heuristic rollouts (`src/rollout.rs`)

Uniformly random rollouts are close to worthless in Go: random play fills its own eyes and lets settled groups die, so the value estimate barely correlates with the position. Three classic pre-AlphaGo rules fix most of it:

1. never fill your own eye,
2. never put your own group in atari for nothing,
3. answer the opponent's last move when it created or escaped an atari.

All three are stated in terms of adjacency alone, so they work on any topology. (On a general graph there is no diagonal relation, so the classical false-eye test does not apply; the eye test here is the topology-agnostic analogue — all neighbors friendly, none of those groups in atari.)

### Playable bot (`src/play.rs`)

A terminal UI to play the engine, plus an engine-vs-engine harness for measuring strength.

### Self-play pipeline (`src/selfplay.rs`)

Parallel self-play engine using rayon. Plays MCTS games and serializes training data for the Python training pipeline. Dirichlet noise is applied at the MCTS root for exploration.

Each game produces a `.bin` file in the output directory containing all training examples (one per position).

## Building and running

```bash
cargo build --release
cargo test
```

### Play against it

```bash
# 9x9 grid
cargo run --release -- play --board 9x9

# on a cube surface (6 faces of 3x3 = 54 points)
cargo run --release -- play --board cube:3 --komi 0.5

# on a torus, as white
cargo run --release -- play --board torus:7x7 --white
```

Points are named `D4` on grids and tori, and `Fc2` on cubes (face `U`/`L`/`F`/`R`/`B`/`D`, then column, then row). In-game commands: `pass`, `hint`, `score`, `board`, `nbrs <point>` (list a point's neighbors — useful when a cube edge makes adjacency non-obvious), `resign`, `help`, `quit`.

Trade strength for speed with `--simulations`.

### Measure strength

```bash
# vs uniformly random play
cargo run --release -- match --board cube:3 --games 16 --opponent random

# vs plain MCTS at the same simulation count
cargo run --release -- match --board cube:3 --games 16 --opponent uniform

# more search vs less search
cargo run --release -- match --board cube:3 --simulations 800 \
    --opponent heuristic --opponent-simulations 200
```

### Generate training data

```bash
cargo run --release -- selfplay --board 9x9 --games 100 --simulations 200
```

See all options with `cargo run --release -- <subcommand> --help`.

## Strength and performance

Measured on an M-series laptop. The bot is a rollout bot with no neural net yet, so "decent" here means *plays real Go* — it captures, defends, builds eyes, keeps its own territory, and passes when nothing is left — not that it is strong.

| Match (cube:3, 54 points) | Result |
|---------------------------|--------|
| heuristic (200 sims) vs random | 20-0 on 5x5, 16-0 on cube |
| heuristic vs uniform MCTS, equal sims | 81% (cube), 100% (9x9) |
| heuristic 800 sims vs heuristic 200 sims | 81% |

The last row is the one that matters most: more search wins, which is the signature of a search that is actually working.

### Three things were making the engine unusably bad

**1. Move generation cloned the whole game state per candidate move.** `legal_moves()` cloned the position — board, `Vec<Vec<usize>>` adjacency (one heap allocation per node), and the superko `HashSet` — once for every empty point, just to test legality. Now legality is decided analytically without mutating or cloning, the topology and Zobrist table are shared between clones behind an `Arc`, board hashing is incremental, and the hot paths reuse caller-owned scratch buffers.

**2. PUCT selection had an inverted sign.** Each node stores its value from the perspective of the player to move at that node, but selection compared `child.q_value()` without negating it. Search therefore preferred the moves the *opponent* liked best. This is why there was never a playable bot.

**3. Terminal positions were scored from the wrong player's point of view.** `simulate` returned `result.scores[0]` — always player 0 — regardless of whose turn it was, so roughly half of all decided lines were scored backwards. Terminal positions are now scored by the parent.

A regression test for each lives in `src/mcts.rs`; the sign bugs are pinned by a mutual-atari race on a 3x3 board where passing loses by half a point and the single alternative move wins.

| Benchmark (`cargo run --release --bin bench`) | Before | After |
|---|---|---|
| `legal_moves()`, 9x9 | 5,553/s | 387,000/s (70x) |
| MCTS, 9x9, 200 sims × 5 rollouts | 20 sims/s | 396 sims/s (20x) |
| One 9x9 search | 10.0s | 0.5s |

## Conformance with govariants

The rules in this repo are a *second* implementation of rules that already
exist in TypeScript in [govariants](https://github.com/govariantsteam/govariants)
(`packages/shared/src/variants/`). Two independent implementations of the same
rules always diverge somewhere, and a divergence is not a cosmetic problem: a
bot playing on the live server would submit moves the server rejects, or play
on from a position it has misread.

So ports are verified against real games rather than only against hand-written
tests:

```bash
# refresh or extend the corpus (hits the live API, read-only)
python3 tools/fetch_corpus.py --board-type grid --games 6

# replay it offline
cargo test --test conformance -- --nocapture
```

`tools/fetch_corpus.py` downloads finished games and, critically, the board
after **every round** as the server computed it. `GET /games/:id/state?round=N`
reports `last_move` for each round, so the full move sequence is recoverable
for every board type — not just the rectangular boards that SGF export
supports. Boards are stored as glyph strings, which keeps a 199-round 19x19
game under 90 KB instead of 637 KB.

The harness replays each game move by move and compares the board and both
capture counts at every round. Current corpus: **5 games, 489 rounds, 80
captures**, all matching.

A conformance test that cannot fail is worthless, so it is mutation-checked:
transposing the coordinate decode makes all five games fail by round 2.

### Board types

Only `grid` is implemented here so far. Across baduk games on the live server
the real distribution is roughly:

| Board type | Games seen | Implemented |
|---|---|---|
| grid | 128 | yes |
| circular | 62 | no |
| gridWithHoles | 61 | no |
| polygonal | 14 | no |
| sierpinsky | 12 | no |
| trihexagonal | 8 | no |
| sunflower | 6 | no |
| custom | 6 | no |

Grid is only about 37% of real games, so the graph topologies are the majority
of the work, not an exotic afterthought. govariants' `boardFactory.ts` and its
helpers are the reference; unsupported board types are reported as skipped by
the harness rather than silently passing.

## Playing on govariants.com

A bot needs no server-side support — it is an ordinary API client:

| Step | Endpoint |
|---|---|
| Register the bot account | `POST /api/register` |
| Log in | `POST /api/login` |
| Fetch the CSRF token | `GET /api/checkLogin` -> `{user, csrf_token}` |
| Find open games | `GET /api/games?variant=…&count=&offset=` |
| Take a seat | `POST /api/games/:id/sit/:seat` |
| Read the position | `GET /api/games/:id/state/initial`, `…/state?round=&seat=` |
| Play | `POST /api/games/:id/move`, body `{"<seat>": "<move>"}` |

Auth is a session cookie plus a `CSRF-Token` header. Live updates arrive over
Socket.IO on `game/:id`, `game/:id/seats` and `game/:id/:seat` (the last
carries hidden state and the server checks seat occupancy, which is how
Phantom Go would work).

Use a **full password login with an obviously-named account**, not a guest
session, so it is clear a bot is playing. There is no `bot` user role yet — the
`UserRole` type is `"admin"` only — so the username is the disclosure.

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

Train on mixed topologies from day one — don't start with standard 9x9 Go and fine-tune. Training on a fixed grid risks learning grid-specific biases (opening theory, fixed-size patterns) that don't transfer to irregular boards.

**Curriculum:**

| Phase | Node range | Goal |
|-------|-----------|------|
| 1 | 50–100 nodes | Learn basic Go: capturing, not filling eyes, basic life/death |
| 2 | 100–250 nodes | Scale reasoning: ladders, territory, larger fights |
| 3 | 250–400 nodes | Full-size optimization toward 361-node target |

- Mix all topologies (2D, 3D, irregular) freely within each phase
- Include a small number of 25-node games throughout for coverage
- Transition between phases based on observed learning, not fixed epochs
- Game length scales with board size — front-load phase 1 heavily

## Roadmap

See [open issues](https://github.com/govariantsteam/govariants-ai/issues) for the full roadmap:

1. ~~Self-play data pipeline (#1)~~ ✅
2. ~~A bot you can actually play, on more than one topology~~ ✅ (grid, torus, cube)
3. GNN model and training loop (#2) ← **next up**
4. Full training loop integration (#3)
5. Train and validate model on mixed topologies (#4)
6. Edge features for geometric boards (#8) — cube coordinates now exist, so this is mostly plumbing
7. Port variant game rules to Rust (#5), verified by the conformance harness
8. Port the remaining board types (circular, gridWithHoles, polygonal, sierpinsky, trihexagonal, sunflower)
9. Adapter that joins games on govariants.com as a named bot account
10. Serving integration with govariants (#6)
