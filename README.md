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

Two halves. Both halves now exist: the Rust engine plays with either hand-written rollout
heuristics or a trained network (see **The neural network** below).

```
Rust — rules, search, self-play, and a client that plays on govariants
┌────────────────────────────────────────────────────────────────────┐
│ game.rs          Game trait: rules + graph structure for a GNN     │
│ games/                                                             │
│   topology.rs    board graphs: grid, torus, cube, circular,        │
│                  gridWithHoles                                     │
│   baduk.rs       Go on an arbitrary board graph                    │
│ mcts.rs          PUCT search over a pluggable Evaluator            │
│ rollout.rs       heuristic rollout evaluator (needs no net)        │
│ selfplay.rs      parallel self-play -> data/*.bin                  │
│ play.rs          terminal UI + engine-vs-engine match harness      │
│ govariants.rs    API client + wire-format translation              │
└────────────────────────────────────────────────────────────────────┘
        │ data/*.bin                            ▲ models/*.onnx
        ▼                                       │
Python — training              ** NOT BUILT YET (issue #2) **
┌────────────────────────────────────────────────────────────────────┐
│ GNN with policy + value heads, training loop, ONNX export          │
└────────────────────────────────────────────────────────────────────┘
        │
        ▼  served in govariants via onnxruntime-node / -web (issue #6)
```

- **Rust**: game rules, MCTS search, parallel self-play, and the live bot. The
  `Evaluator` trait is the seam a neural net drops into; `rollout.rs` is one
  implementation of it and an ONNX-backed one would be another.
- **Python**: GNN architecture, training loop, ONNX export.
- **ONNX**: trained models served in govariants via onnxruntime-node or
  onnxruntime-web.

### A revision to the Python plan

This was originally specced as PyTorch Geometric. **Prefer plain PyTorch with a
dense adjacency matrix instead.** A message-passing layer is just
`X' = relu(X @ W_self + (A_norm @ X) @ W_neigh)`, and at ≤361 nodes a dense
adjacency matrix is ~130 KB and a matmul beats sparse scatter anyway. PyG's
gather/scatter ops are awkward to export to ONNX, which issue #6 needs for
onnxruntime-web; passing `A` as a model input with a dynamic axis keeps one set
of weights serving every topology, which is the whole point of the GNN. Edge
features (#8) then become one adjacency matrix per direction bucket — still
just matmuls.

### Binaries

| Binary | Purpose |
|---|---|
| `govariants-ai` | `play` (terminal game), `match` (strength), `selfplay` (training data) |
| `bot` | plays on a govariants server (see below) |
| `bench` | micro-benchmarks of the hot paths, per topology |
| `budget` | how many playouts a fixed thinking time actually buys |

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
| Circular | — | rings×per-ring | govariants' circular boards; see below |
| GridWithHoles | — | non-zero cells | govariants' bitmap boards |

Circular and GridWithHoles exist to match govariants rather than to be
configured by hand, so they have no `--board` spelling yet; the conformance
harness builds them from a game's config.

**Circular** is not a stack of independent circles. Each ring is rotated half
a step from the one inside it, so every node has two radial neighbors inward,
and **only the innermost and outermost rings carry intra-ring edges** — the
middle rings connect purely radially. It is a triangular lattice on an
annulus. Making every ring a cycle is the natural misreading and it changes
which stones form a group.

**GridWithHoles** is a bitmap where each cell is 0 (no intersection) or a
mask: bit 0 connects right, bit 1 connects down, and 4 marks an intersection
that only receives edges from above or the left. It is *not* "a grid with
cells deleted" — a cell can sit next to a surviving neighbor with no edge
between them.

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

### govariants client (`src/govariants.rs`)

The API client plus translation between govariants' wire format and this
engine: board config to topology, `state.board` to a position, and move
encoding (SGF on grids, decimal node index everywhere else). The conformance
harness drives these same functions, so replaying real games verifies the code
the live bot runs.

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

### Benchmark and budget

```bash
# throughput of the hot paths, across topologies
cargo run --release --bin bench

# what a fixed thinking time buys, with the tree cost included
cargo run --release --bin budget
```

`bench` times move generation, node features and whole searches. `budget`
calibrates a real search to a 5-second budget and reports playouts, playouts
per second and tree nodes per second, at several rollouts-per-node settings —
which is how the "more tree beats less noise" finding below was measured.

See all options with `cargo run --release -- <subcommand> --help`, and
`cargo run --release --bin bot -- help` for the bot.

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
capture counts at every round. Current corpus: **14 games, 1532 rounds**
across grid, circular and gridWithHoles boards, all matching.

Moves are decoded per topology: grids use SGF coordinates, and every other
board type uses the decimal node index, which is how govariants encodes them
on the wire.

A conformance test that cannot fail is worthless, so it is mutation-checked:
transposing the coordinate decode makes all five games fail by round 2.

### Board types

Across baduk games on the live server the real distribution is roughly:

| Board type | Games seen | Implemented |
|---|---|---|
| grid | 128 | yes |
| circular | 62 | yes |
| gridWithHoles | 61 | yes |
| polygonal | 14 | no |
| sierpinsky | 12 | no |
| trihexagonal | 8 | no |
| sunflower | 6 | no |
| custom | 6 | no |

Grid is only about 37% of real games, so the graph topologies are the majority
of the work, not an exotic afterthought. The three implemented types cover
about 85% of games played. govariants' `boardFactory.ts` and its helpers are
the reference; unsupported board types are reported as skipped by the harness
rather than silently passing.

### Replaying real games is necessary but not sufficient

Worth recording, because it is easy to over-trust a green corpus.

Both new topologies were mutation-tested. Making every circular ring a cycle
is caught by the corpus — but at rounds 21 to 49, not round 1, because extra
edges only change the outcome once a capture depends on them. That is exactly
the kind of case hand-written tests miss and real games catch.

Ignoring the GridWithHoles connectivity bits, however, **passed the entire
corpus**. All four finished boards sampled used only values 0 and 3 with zero
adjacent-but-unconnected pairs — they were plain grids with five cells
removed, on which the correct and incorrect readings are identical. Targeting
a board with 72 such pairs did not help either: across its 91 rounds, two
same-coloured stones never once landed on both ends of one of them, and the
game contained a single capture.

So the bitmap semantics are pinned by direct unit tests instead
(`grid_with_holes_respects_connectivity_bits`), which do fail under that
mutant. The lesson for the remaining board types: fetch real games *and* write
a structural test, because whether a corpus exercises a given rule is a matter
of what players happened to play.

## Playing on govariants.com

There is a working bot client (`src/bin/bot.rs`). It needs no server-side
support: it registers, logs in with a password, takes a seat and posts moves
exactly as a browser does.

```bash
# one-off: create the account (writes .govariants-bot-credentials, gitignored)
cargo run --release --bin bot -- register --username govariantsAiBot

# host a game and wait for a human to take the other seat
cargo run --release --bin bot -- host --board 9x9 --simulations 4000

# read a game and report the move it would play, without writing anything
cargo run --release --bin bot -- probe --game-id <id>

# list games, marking which ones it can actually play
cargo run --release --bin bot -- games

# join an existing game, or give up a seat
cargo run --release --bin bot -- play --game-id <id> [--seat 0] [--dry-run]
cargo run --release --bin bot -- leave --game-id <id> --seat 0
```

Point `--base` at a local instance to test against one.

### How it behaves

`host` is the preferred way to play it: the bot creates its own game and waits,
so nobody ends up playing a bot without choosing to. It prints the URL to open.

It **refuses to sit at a board it cannot parse** rather than taking a seat and
then playing badly, and `games` shows exactly why a game is unplayable:

```
game id                    board            phase     seats   playable
6acea8f34af6cf110acfe007   grid             play      open    yes
6aca445aec4422373d5a2a93   sunflower        play      full    no (board type 'sunflower' is not implemented)
```

**If the server rejects a move, the bot falls back to its next-best choice.**
That is a design decision, not a patch: `Baduk::from_position` builds a
position from a board rather than from a move list, because that is what
joining a game in progress gives you — and **ko state cannot be recovered from
a single position**. Positional superko history is genuinely gone, and even
simple ko needs the previous board. So a rejection is expected rather than
exceptional, and retrying the next candidate turns a correctness problem into
a retry. It doubles as a safety net for exactly the rules divergence the
conformance harness exists to catch.

Credentials come from `GOVARIANTS_BOT_USERNAME` / `GOVARIANTS_BOT_PASSWORD`,
or from `.govariants-bot-credentials` (mode 0600, gitignored) which `register`
writes. The password is generated and never printed.

Only the `baduk` variant is supported, on grid, circular and gridWithHoles
boards. Everything else it declines to play. Note that the torus and cube
topologies in this repo are **not** govariants board types — cube is a separate
variant there — so they cannot be hosted on the site yet.

Use an obviously-bot account name. There is no `bot` user role on the server
(`UserRole` is `"admin"` only), so the username is the only disclosure. Names
must be alphanumeric: the server rejects hyphens and underscores.

### The API, for reference

| Step | Endpoint |
|---|---|
| Register | `POST /api/register` (`{username, password}`; email optional, not sent) |
| Log in | `POST /api/login` |
| CSRF token | `GET /api/checkLogin` -> `{user, csrf_token}` |
| List games | `GET /api/games?variant=&count=&offset=` |
| Create a game | `POST /api/games` (`{variant, config}`) |
| Take a seat | `POST /api/games/:id/sit/:seat` |
| Read the position | `GET /api/games/:id/state/initial`, `…/state?round=&seat=` |
| Play | `POST /api/games/:id/move`, body `{"<seat>": "<move>"}` |

Auth is a session cookie plus a `CSRF-Token` header. Moves are SGF coordinates
on grids (`"dd"`) and the decimal node index on every other board type. Live
updates are available over Socket.IO on `game/:id`, `game/:id/seats` and
`game/:id/:seat` (the last carries hidden state and the server checks seat
occupancy, which is how Phantom Go would work) — but the bot polls instead,
which is plenty for a turn-based game and avoids a websocket client.

## The neural network

Exists, trains, exports, and plays. `python/` holds the training half; the
engine plays with it through the same `Evaluator` trait the rollouts use.

```bash
python3 -m venv .venv && .venv/bin/pip install -r python/requirements.txt

# generate data (one directory per topology, so files do not collide)
cargo run --release -- selfplay --board 9x9    --games 150 --simulations 200 --output data/grid9
cargo run --release -- selfplay --board cube:3 --games 150 --simulations 200 --output data/cube3

# train on mixed topologies, then export
cd python
../.venv/bin/python -m gvai.train --data ../data/grid9 ../data/cube3 --epochs 40
../.venv/bin/python -m gvai.export --checkpoint ../models/gvai_mixed.pt --out ../models/gvai.onnx
../.venv/bin/python -m gvai.selftest      # checks that run without pytest

# play with it (needs the onnx feature)
cargo run --release --features onnx --bin netplay -- --board 9x9 inspect
cargo run --release --features onnx --bin netplay -- --board cube:3 selfmatch --games 16
```

### What it is

A graph net: `h' = relu(h W_self + (A_norm h) W_neigh)` stacked a few deep,
then per-node policy logits, a separate pass logit, and a tanh value head.
84k parameters at hidden 96 x 4 layers.

`A_norm` is an **input**, not a constant, so **one model file plays every
topology** — verified in `src/net.rs` tests, where the same `gvai.onnx` runs on
an 81-node grid and a 54-node cube surface.

The policy is split into per-node logits plus one pass logit rather than a flat
`num_nodes + 1` vector, because mixed board sizes in a batch would otherwise
put the pass slot at a different index per example.

### What it learned

On an empty 9x9 its four favourite moves are the four 3-3 points, at identical
probability — it found both a reasonable 9x9 opening and the board's symmetry
from self-play alone. The priors are only ~1.3x uniform though, so the policy
is diffuse: 30% top-1 agreement with MCTS, value MSE 0.615 against 1.0 for
always guessing zero, policy cross-entropy 2.64 against a 1.95 floor.

### Measuring it honestly

| Match (9x9) | Result |
|---|---|
| net 800 sims vs net 100 sims | 81%, mean margin 8.3 points |
| net vs heuristic rollouts, equal sims | 100% — **but see below** |

The first row is the trustworthy one: the same evaluator on both sides, so
neither can exploit a quirk of the other, and more search wins — which is what
says the net is a usable evaluator at all.

**The 100% against the rollout bot is inflated.** Mean margin was 41 points on
an 81-point board and the net never passed first in 12 games. The rollout
bot's prior gives zero weight to eye-fills and self-ataris, so once only those
remain it passes while points are still convertible, and the net keeps playing.
That is a real weakness of the rollout prior rather than net strength, and it
is why the honest comparison is net against itself.

### Two traps worth knowing

**The policy loss floor is not zero.** Targets are MCTS visit distributions,
not one-hot labels, and cross-entropy to a soft target cannot go below the
target's entropy (1.95 nats here). `train.py` prints the floor, because
"why won't it converge to zero" is otherwise an easy hour to lose.

**Padding caused a NaN that only mixed topologies could reveal.** Padded nodes
have target 0 and had logit -inf, and `0 * -inf` is NaN. Training on one board
size never pads, so single-topology runs looked healthy while mixed-topology
runs silently lost the policy objective — with the value head still improving,
so nothing announced it. Pinned by
`test_mixed_board_sizes_give_a_finite_loss`, alongside a padding-invariance
test that a board predicts the same alone as it does padded.

### Python and Rust must agree on the adjacency

The net takes the adjacency as an input, so the matrix Rust builds at play time
must equal the one Python built during training — and a mismatch would not
crash, it would just play worse for no visible reason. Same failure mode as the
TypeScript/Rust rules divergence, so it gets the same treatment:
`python -m gvai.make_fixture` writes `tests/fixtures/adjacency.json` from real
self-play edge lists, and `tests/adjacency_parity.rs` asserts it for grid and
cube boards. ONNX export also checks itself against PyTorch at two board sizes
before declaring success (worst difference 1.5e-5).

Inference is `tract`, pure Rust, so there is no native onnxruntime to install.
A single evaluation is 0.48 ms on 9x9, giving ~2,100 search simulations per
second.

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
3. ~~GNN model and training loop (#2)~~ ✅ first version trains, exports and plays
4. Full training loop integration (#3)
5. Train and validate model on mixed topologies (#4)
6. Edge features for geometric boards (#8) — cube coordinates now exist, so this is mostly plumbing
7. Port variant game rules to Rust (#5), verified by the conformance harness
8. Port the remaining board types (polygonal, sierpinsky, trihexagonal, sunflower, custom) — grid, circular and gridWithHoles are done
9. ~~Adapter that joins games on govariants.com as a named bot account~~ ✅
10. Serving integration with govariants (#6)
