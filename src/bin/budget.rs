//! What do you actually get for a fixed thinking time?
//!
//! Reports raw playout throughput, and then calibrates a real search to the
//! given budget so the numbers include the cost of a tree that grows as the
//! search runs (rather than extrapolating from a short run).

use govariants_ai::game::Game;
use govariants_ai::games::baduk::{Baduk, BadukConfig};
use govariants_ai::games::topology::Topology;
use govariants_ai::mcts::{Evaluator, MctsConfig, search};
use govariants_ai::rollout::{HeuristicRolloutEvaluator, playout};
use rand::rng;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const BUDGET_SECS: f64 = 5.0;

/// Wraps an evaluator to count how many times it was asked for a verdict.
/// Each call runs exactly `rollouts` playouts, so this counts playouts too.
struct Counting<'a, E> {
    inner: &'a E,
    evaluations: AtomicUsize,
}

impl<'a, E: Evaluator<Baduk>> Evaluator<Baduk> for Counting<'a, E> {
    fn evaluate(&self, game: &Baduk) -> (Vec<f32>, f32) {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        self.inner.evaluate(game)
    }
}

/// Raw playout speed, independent of the tree.
fn raw_playouts(game: &Baduk) -> (f64, f64) {
    let mut scratch = game.new_scratch();
    let mut rng = rng();
    let max_moves = game.num_nodes() * 2;

    // Warm up, then run for a fixed slice of wall time.
    for _ in 0..50 {
        std::hint::black_box(playout(game, max_moves, 0.9, &mut scratch, &mut rng));
    }

    let start = Instant::now();
    let mut count = 0usize;
    while start.elapsed().as_secs_f64() < 1.0 {
        for _ in 0..200 {
            std::hint::black_box(playout(game, max_moves, 0.9, &mut scratch, &mut rng));
            count += 1;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    (count as f64 / elapsed, elapsed)
}

/// Run a search calibrated to take about `BUDGET_SECS`, and report what fit.
fn search_in_budget(game: &Baduk, rollouts: usize) {
    let evaluator = HeuristicRolloutEvaluator {
        num_rollouts: rollouts,
        ..Default::default()
    };

    // Calibration pass to estimate the simulation rate.
    let probe_sims = 150;
    let counter = Counting {
        inner: &evaluator,
        evaluations: AtomicUsize::new(0),
    };
    let config = MctsConfig {
        num_simulations: probe_sims,
        temperature: 1.0,
        noise: None,
        ..Default::default()
    };
    let start = Instant::now();
    std::hint::black_box(search(game, &counter, &config));
    let probe_secs = start.elapsed().as_secs_f64();
    let estimated_rate = probe_sims as f64 / probe_secs;

    // Real run, sized to the budget.
    let sims = ((estimated_rate * BUDGET_SECS) as usize).max(probe_sims);
    let counter = Counting {
        inner: &evaluator,
        evaluations: AtomicUsize::new(0),
    };
    let config = MctsConfig {
        num_simulations: sims,
        temperature: 1.0,
        noise: None,
        ..Default::default()
    };
    let start = Instant::now();
    std::hint::black_box(search(game, &counter, &config));
    let elapsed = start.elapsed().as_secs_f64();

    let evaluations = counter.evaluations.load(Ordering::Relaxed);
    let playouts = evaluations * rollouts;
    let scale = BUDGET_SECS / elapsed;

    println!(
        "    {:>2} rollouts/node: {:>9} playouts   ({:>7.0} playouts/s, {:>6.0} tree nodes/s, {:.1}s run)",
        rollouts,
        (playouts as f64 * scale) as usize,
        playouts as f64 / elapsed,
        evaluations as f64 / elapsed,
        elapsed
    );
}

fn main() {
    println!(
        "Thinking time budget: {:.0}s per move, single threaded\n",
        BUDGET_SECS
    );

    let boards = [
        Topology::Grid {
            width: 9,
            height: 9,
        },
        Topology::Grid {
            width: 19,
            height: 19,
        },
        Topology::Cube { size: 3 },
        Topology::Cube { size: 5 },
        Topology::Torus {
            width: 7,
            height: 7,
        },
    ];

    for topology in boards {
        let game = Baduk::new(BadukConfig {
            topology,
            komi: 6.5,
        });
        let (rate, _) = raw_playouts(&game);
        println!(
            "{} - {} points",
            topology.describe(),
            game.num_nodes()
        );
        println!(
            "    raw playout speed: {:.0} playouts/s  ({} in {:.0}s, ignoring tree overhead)",
            rate,
            (rate * BUDGET_SECS) as usize,
            BUDGET_SECS
        );
        for rollouts in [1usize, 2, 8] {
            search_in_budget(&game, rollouts);
        }
        println!();
    }
}
