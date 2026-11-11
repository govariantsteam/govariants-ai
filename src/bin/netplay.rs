//! Play the trained network against the rollout bot, and look at what it wants.
//!
//! This is the measurement that decides whether the net is worth anything yet.
//! The rollout evaluator searches to the end of the game with hand-written
//! heuristics; the net is a static evaluator trained on that bot's own games.
//! Beating it is not a given, and the honest comparison is at equal simulation
//! counts.

use clap::{Parser, Subcommand};
use govariants_ai::game::Game;
use govariants_ai::games::baduk::{Baduk, BadukConfig, BadukMove};
use govariants_ai::mcts::{Evaluator, MctsConfig, search_detailed};
use govariants_ai::net::NetEvaluator;
use govariants_ai::play::parse_board;
use govariants_ai::rollout::HeuristicRolloutEvaluator;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "netplay")]
#[command(about = "Play or inspect a trained ONNX model")]
struct Cli {
    /// Path to the exported model.
    #[arg(long, default_value = "models/gvai.onnx")]
    model: PathBuf,

    /// Board: "9x9", "grid:5x7", "torus:7x7", "cube:3".
    #[arg(long, default_value = "9x9")]
    board: String,

    #[arg(long, default_value_t = 6.5)]
    komi: f32,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show the net's preferred moves and value for a fresh board.
    Inspect {
        /// How many of the top moves to list.
        #[arg(long, default_value_t = 8)]
        top: usize,
    },
    /// Play the net against the heuristic rollout bot.
    Match {
        #[arg(long, default_value_t = 20)]
        games: usize,
        #[arg(long, default_value_t = 200)]
        simulations: usize,
        /// Rollouts per evaluation for the heuristic opponent.
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
        /// Opening moves played at temperature 1, for variety.
        #[arg(long, default_value_t = 6)]
        opening_randomness: usize,
    },
    /// Net against itself at two search depths.
    ///
    /// The only self-consistent strength test available: the same evaluator on
    /// both sides means neither can exploit a quirk of the other. If more
    /// search does not win, the net is not a usable evaluator.
    Selfmatch {
        #[arg(long, default_value_t = 800)]
        simulations: usize,
        #[arg(long, default_value_t = 200)]
        opponent_simulations: usize,
        #[arg(long, default_value_t = 16)]
        games: usize,
        #[arg(long, default_value_t = 6)]
        opening_randomness: usize,
    },
    /// Play one game and print the final board. Diagnostic: a win that comes
    /// from exploiting the scoring looks different from a win on the board.
    Showgame {
        #[arg(long, default_value_t = 200)]
        simulations: usize,
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
        /// Net plays black.
        #[arg(long)]
        net_black: bool,
    },
    /// Time a single evaluation and a full search.
    Bench {
        #[arg(long, default_value_t = 400)]
        simulations: usize,
    },
}

fn config_for(cli: &Cli) -> BadukConfig {
    let topology = match parse_board(&cli.board) {
        Ok(topology) => topology,
        Err(message) => {
            eprintln!("error: {}", message);
            std::process::exit(2);
        }
    };
    BadukConfig {
        topology,
        komi: cli.komi,
    }
}

fn choose<E: Evaluator<Baduk>>(
    game: &Baduk,
    evaluator: &E,
    simulations: usize,
    temperature: f32,
) -> BadukMove {
    let config = MctsConfig {
        num_simulations: simulations,
        temperature,
        noise: None,
        ..Default::default()
    };
    let outcome = search_detailed(game, evaluator, &config);
    match outcome.best_action {
        Some(action) if temperature < 1e-6 => game.index_to_move(action),
        _ => {
            let total: f32 = outcome.policy.iter().sum();
            if total <= 0.0 {
                return BadukMove::Pass;
            }
            // Sample from the visit distribution for opening variety.
            use rand::prelude::*;
            let mut rng = rand::rng();
            let distribution = rand::distr::weighted::WeightedIndex::new(&outcome.policy);
            match distribution {
                Ok(d) => game.index_to_move(d.sample(&mut rng)),
                Err(_) => outcome
                    .best_action
                    .map(|a| game.index_to_move(a))
                    .unwrap_or(BadukMove::Pass),
            }
        }
    }
}

struct GameRecord {
    outcome: i32,
    moves: usize,
    black: f32,
    white: f32,
    /// Who ran out of moves first, if anyone passed twice.
    net_passed_first: Option<bool>,
}

/// One game. Returns +1 if the net won, -1 if it lost, 0 for a draw.
fn play_one(
    config: &BadukConfig,
    net: &NetEvaluator,
    heuristic: &HeuristicRolloutEvaluator,
    net_plays_black: bool,
    simulations: usize,
    opening_randomness: usize,
) -> GameRecord {
    let mut game = Baduk::new(config.clone());
    let max_moves = game.num_nodes() * 3;
    let mut played = 0;
    let mut net_passed_first = None;

    while !game.is_terminal() && played < max_moves {
        let to_play = game.next_to_play()[0];
        let net_turn = (to_play == 0) == net_plays_black;
        let temperature = if played < opening_randomness {
            1.0
        } else {
            0.0
        };
        let mv = if net_turn {
            choose(&game, net, simulations, temperature)
        } else {
            choose(&game, heuristic, simulations, temperature)
        };
        if mv == BadukMove::Pass && net_passed_first.is_none() {
            net_passed_first = Some(net_turn);
        }
        game.play_move(to_play, mv);
        played += 1;
    }

    let (black, white) = game.score();
    let black_result = if black > white {
        1
    } else if white > black {
        -1
    } else {
        0
    };
    GameRecord {
        outcome: if net_plays_black {
            black_result
        } else {
            -black_result
        },
        moves: played,
        black,
        white,
        net_passed_first,
    }
}

fn main() {
    let cli = Cli::parse();
    let config = config_for(&cli);
    let game = Baduk::new(config.clone());

    if !cli.model.exists() {
        eprintln!(
            "error: no model at {}. Train one and run `python -m gvai.export`.",
            cli.model.display()
        );
        std::process::exit(1);
    }

    let net = match NetEvaluator::load(&cli.model, &game) {
        Ok(net) => net,
        Err(error) => {
            eprintln!("error: could not load {}: {}", cli.model.display(), error);
            std::process::exit(1);
        }
    };
    println!(
        "model {} on {} ({} nodes)",
        cli.model.display(),
        config.topology.describe(),
        game.num_nodes()
    );

    match cli.command {
        Command::Inspect { top } => {
            let (policy, value) = net.evaluate(&game);
            println!("net value for the player to move: {:+.3}", value);
            let mut ranked: Vec<(usize, f32)> = policy
                .iter()
                .enumerate()
                .filter(|(_, p)| **p > 0.0)
                .map(|(i, p)| (i, *p))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!("top {} moves by prior:", top.min(ranked.len()));
            for (action, probability) in ranked.iter().take(top) {
                let label = match game.index_to_move(*action) {
                    BadukMove::Pass => "pass".to_string(),
                    BadukMove::Place(node) => config.topology.label(node),
                };
                println!("  {:>6}  {:.4}", label, probability);
            }
            let uniform = 1.0 / ranked.len() as f32;
            println!(
                "(uniform over {} legal moves would be {:.4}; a flat policy means the net learned nothing)",
                ranked.len(),
                uniform
            );
        }

        Command::Selfmatch {
            simulations,
            opponent_simulations,
            games,
            opening_randomness,
        } => {
            println!(
                "net({} sims) vs net({} sims), {} games",
                simulations, opponent_simulations, games
            );
            let start = Instant::now();
            let mut strong_wins = 0;
            let mut weak_wins = 0;
            let mut draws = 0;
            let mut total_margin = 0.0f32;

            for i in 0..games {
                let strong_is_black = i % 2 == 0;
                let mut board = Baduk::new(config.clone());
                let max_moves = board.num_nodes() * 3;
                let mut played = 0;
                while !board.is_terminal() && played < max_moves {
                    let to_play = board.next_to_play()[0];
                    let strong_turn = (to_play == 0) == strong_is_black;
                    let sims = if strong_turn {
                        simulations
                    } else {
                        opponent_simulations
                    };
                    let temperature = if played < opening_randomness {
                        1.0
                    } else {
                        0.0
                    };
                    let mv = choose(&board, &net, sims, temperature);
                    board.play_move(to_play, mv);
                    played += 1;
                }
                let (black, white) = board.score();
                total_margin += (black - white).abs();
                let black_result = if black > white {
                    1
                } else if white > black {
                    -1
                } else {
                    0
                };
                let strong_result = if strong_is_black {
                    black_result
                } else {
                    -black_result
                };
                match strong_result {
                    1 => strong_wins += 1,
                    -1 => weak_wins += 1,
                    _ => draws += 1,
                }
                print!("\r  {}/{} played", i + 1, games);
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            let elapsed = start.elapsed().as_secs_f64();
            let score = strong_wins as f64 + 0.5 * draws as f64;
            println!();
            println!(
                "more search {} - {} less search ({} draws)  winrate {:.1}%  mean |margin| {:.1}  [{:.0}s]",
                strong_wins,
                weak_wins,
                draws,
                100.0 * score / games as f64,
                total_margin / games as f32,
                elapsed
            );
        }

        Command::Showgame {
            simulations,
            rollouts,
            net_black,
        } => {
            let heuristic = HeuristicRolloutEvaluator {
                num_rollouts: rollouts,
                ..Default::default()
            };
            let mut board = Baduk::new(config.clone());
            let max_moves = board.num_nodes() * 3;
            let mut played = 0;
            let mut first_pass_at = None;
            while !board.is_terminal() && played < max_moves {
                let to_play = board.next_to_play()[0];
                let net_turn = (to_play == 0) == net_black;
                let mv = if net_turn {
                    choose(&board, &net, simulations, 0.0)
                } else {
                    choose(&board, &heuristic, simulations, 0.0)
                };
                if mv == BadukMove::Pass && first_pass_at.is_none() {
                    first_pass_at = Some((played, net_turn));
                }
                board.play_move(to_play, mv);
                played += 1;
            }
            let (black, white) = board.score();
            println!(
                "net plays {} | {} moves",
                if net_black { "black (X)" } else { "white (O)" },
                played
            );
            if let Some((at, by_net)) = first_pass_at {
                println!(
                    "first pass at move {} by {}",
                    at,
                    if by_net {
                        "the net"
                    } else {
                        "the heuristic bot"
                    }
                );
            }
            print!("{}", govariants_ai::play::render(&board));
            println!("final area score: black {:.1}, white {:.1}", black, white);
            println!("captures: {:?}", board.captures());
        }

        Command::Bench { simulations } => {
            let start = Instant::now();
            let rounds = 200;
            for _ in 0..rounds {
                std::hint::black_box(net.evaluate(&game));
            }
            let per_eval = start.elapsed().as_secs_f64() / rounds as f64;
            println!("single evaluation: {:.3} ms", per_eval * 1000.0);

            let start = Instant::now();
            std::hint::black_box(choose(&game, &net, simulations, 0.0));
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "search of {} simulations: {:.2}s ({:.0} sims/s)",
                simulations,
                elapsed,
                simulations as f64 / elapsed
            );
        }

        Command::Match {
            games,
            simulations,
            rollouts,
            opening_randomness,
        } => {
            let heuristic = HeuristicRolloutEvaluator {
                num_rollouts: rollouts,
                ..Default::default()
            };
            println!(
                "net vs heuristic-rollouts, {} games, {} simulations each",
                games, simulations
            );

            let start = Instant::now();
            let mut net_wins = 0;
            let mut heuristic_wins = 0;
            let mut draws = 0;
            let mut total_moves = 0usize;
            let mut total_margin = 0.0f32;
            let mut net_passed_first_count = 0usize;
            let mut someone_passed = 0usize;
            for i in 0..games {
                let record = play_one(
                    &config,
                    &net,
                    &heuristic,
                    i % 2 == 0,
                    simulations,
                    opening_randomness,
                );
                match record.outcome {
                    1 => net_wins += 1,
                    -1 => heuristic_wins += 1,
                    _ => draws += 1,
                }
                total_moves += record.moves;
                total_margin += (record.black - record.white).abs();
                if let Some(net_first) = record.net_passed_first {
                    someone_passed += 1;
                    if net_first {
                        net_passed_first_count += 1;
                    }
                }
                print!("\r  {}/{} played", i + 1, games);
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            println!();
            println!(
                "mean game length {:.0} moves on a {}-point board; mean |margin| {:.1} points",
                total_moves as f64 / games as f64,
                game.num_nodes(),
                total_margin / games as f32
            );
            println!(
                "first to pass: net {}/{} games where anyone passed",
                net_passed_first_count, someone_passed
            );
            let elapsed = start.elapsed().as_secs_f64();
            let score = net_wins as f64 + 0.5 * draws as f64;
            println!(
                "\nnet {} - {} heuristic ({} draws)  net winrate {:.1}%  [{:.0}s, {:.1}s/game]",
                net_wins,
                heuristic_wins,
                draws,
                100.0 * score / games as f64,
                elapsed,
                elapsed / games as f64
            );
        }
    }
}
