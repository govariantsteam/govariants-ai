//! Micro-benchmarks for the hot paths: move generation, node features, and
//! full MCTS searches, across board topologies and both evaluators.

use govariants_ai::game::Game;
use govariants_ai::games::baduk::{Baduk, BadukConfig};
use govariants_ai::games::topology::Topology;
use govariants_ai::mcts::{Evaluator, MctsConfig, RandomRolloutEvaluator, search};
use govariants_ai::rollout::HeuristicRolloutEvaluator;
use std::time::Instant;

fn bench_legal_moves(game: &Baduk, iters: usize) {
    let start = Instant::now();
    let mut total = 0usize;
    for _ in 0..iters {
        total += game.legal_moves().len();
    }
    let el = start.elapsed().as_secs_f64();
    println!(
        "  legal_moves()         {:>9.0} calls/s   {:>7.1} us/call",
        iters as f64 / el,
        el * 1e6 / iters as f64
    );
    std::hint::black_box(total);
}

fn bench_node_features(game: &Baduk, iters: usize) {
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(game.node_features());
    }
    let el = start.elapsed().as_secs_f64();
    println!(
        "  node_features()       {:>9.0} calls/s   {:>7.1} us/call",
        iters as f64 / el,
        el * 1e6 / iters as f64
    );
}

fn bench_search<E: Evaluator<Baduk>>(game: &Baduk, evaluator: &E, label: &str, sims: usize) {
    let config = MctsConfig {
        num_simulations: sims,
        temperature: 1.0,
        noise: None,
        ..Default::default()
    };
    let start = Instant::now();
    std::hint::black_box(search(game, evaluator, &config));
    let el = start.elapsed().as_secs_f64();
    println!(
        "  search {:<14} {:>9.0} sims/s    {:>7.2}s for {} sims",
        label,
        sims as f64 / el,
        el,
        sims
    );
}

fn main() {
    let boards = [
        Topology::Grid {
            width: 5,
            height: 5,
        },
        Topology::Grid {
            width: 9,
            height: 9,
        },
        Topology::Torus {
            width: 7,
            height: 7,
        },
        Topology::Cube { size: 3 },
        Topology::Cube { size: 5 },
    ];

    for topology in boards {
        let game = Baduk::new(BadukConfig {
            topology: topology.clone(),
            komi: 6.5,
        });
        println!("{} - {} nodes:", topology.describe(), game.num_nodes());
        bench_legal_moves(&game, 20000);
        bench_node_features(&game, 5000);
        bench_search(
            &game,
            &RandomRolloutEvaluator { num_rollouts: 5 },
            "uniform",
            500,
        );
        bench_search(
            &game,
            &HeuristicRolloutEvaluator {
                num_rollouts: 8,
                ..Default::default()
            },
            "heuristic",
            500,
        );
        println!();
    }
}
