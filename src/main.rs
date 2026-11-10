use clap::{Parser, Subcommand};
use govariants_ai::game::Game;
use govariants_ai::games::baduk::{Baduk, BadukConfig};
use govariants_ai::mcts::{MctsConfig, NoiseConfig};
use govariants_ai::play::{Agent, interactive, parse_board, play_match};
use govariants_ai::rollout::HeuristicRolloutEvaluator;
use govariants_ai::selfplay::{SelfPlayConfig, run_self_play};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "govariants-ai")]
#[command(about = "Self-play, training data, and a playable bot for Go variants")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate self-play training data.
    Selfplay(SelfplayArgs),
    /// Play a game against the engine in the terminal.
    Play(PlayArgs),
    /// Play two engines against each other to measure strength.
    Match(MatchArgs),
}

/// Board specification shared by the subcommands.
#[derive(Parser, Clone)]
struct BoardArgs {
    /// Board: "9x9", "grid:5x7", "torus:5x5", or "cube:3".
    #[arg(long, default_value = "9x9")]
    board: String,

    /// Komi (points given to white).
    #[arg(long, default_value_t = 6.5)]
    komi: f32,
}

impl BoardArgs {
    fn config(&self) -> BadukConfig {
        let topology = match parse_board(&self.board) {
            Ok(t) => t,
            Err(err) => {
                eprintln!("error: {}", err);
                std::process::exit(2);
            }
        };
        BadukConfig {
            topology,
            komi: self.komi,
        }
    }
}

#[derive(Parser)]
struct SelfplayArgs {
    #[command(flatten)]
    board: BoardArgs,

    /// Number of self-play games to generate.
    #[arg(long, default_value_t = 100)]
    games: usize,

    /// Number of MCTS simulations per move.
    #[arg(long, default_value_t = 200)]
    simulations: usize,

    /// Temperature for move selection (0 = deterministic).
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    /// Dirichlet noise alpha (0 = disabled).
    #[arg(long, default_value_t = 0.3)]
    noise_alpha: f32,

    /// Dirichlet noise mixing fraction.
    #[arg(long, default_value_t = 0.25)]
    noise_epsilon: f32,

    /// Rollouts per evaluation (used until the neural net is available).
    #[arg(long, default_value_t = 8)]
    rollouts: usize,

    /// Output directory for training data.
    #[arg(long, default_value = "data")]
    output: String,
}

#[derive(Parser)]
struct PlayArgs {
    #[command(flatten)]
    board: BoardArgs,

    /// MCTS simulations the engine uses per move.
    #[arg(long, default_value_t = 2000)]
    simulations: usize,

    /// Rollouts per evaluation.
    #[arg(long, default_value_t = 8)]
    rollouts: usize,

    /// Play as white instead of black.
    #[arg(long)]
    white: bool,
}

#[derive(Parser)]
struct MatchArgs {
    #[command(flatten)]
    board: BoardArgs,

    /// Number of games to play.
    #[arg(long, default_value_t = 40)]
    games: usize,

    /// Simulations for the first engine.
    #[arg(long, default_value_t = 400)]
    simulations: usize,

    /// Rollouts per evaluation for the first engine.
    #[arg(long, default_value_t = 4)]
    rollouts: usize,

    /// Opponent: "random", "uniform", or "heuristic".
    #[arg(long, default_value = "random")]
    opponent: String,

    /// Simulations for the opponent (defaults to --simulations).
    #[arg(long)]
    opponent_simulations: Option<usize>,

    /// Rollouts per evaluation for the opponent (defaults to --rollouts).
    /// Together with --opponent-simulations this allows equal-playout or
    /// equal-time comparisons between differently tuned engines.
    #[arg(long)]
    opponent_rollouts: Option<usize>,

    /// Moves at the start played with temperature 1, for variety.
    #[arg(long, default_value_t = 6)]
    opening_randomness: usize,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Selfplay(args) => run_selfplay(args),
        Command::Play(args) => run_play(args),
        Command::Match(args) => run_match(args),
    }
}

fn run_selfplay(args: SelfplayArgs) {
    let board_config = args.board.config();

    let noise = if args.noise_alpha > 0.0 {
        Some(NoiseConfig {
            alpha: args.noise_alpha,
            epsilon: args.noise_epsilon,
        })
    } else {
        None
    };

    let mcts_config = MctsConfig {
        num_simulations: args.simulations,
        temperature: args.temperature,
        noise,
        ..Default::default()
    };

    let selfplay_config = SelfPlayConfig {
        num_games: args.games,
        mcts_config,
        output_dir: args.output.clone(),
    };

    let evaluator = HeuristicRolloutEvaluator {
        num_rollouts: args.rollouts,
        ..Default::default()
    };

    eprintln!(
        "Starting self-play: {} games on {}, {} sims/move",
        args.games,
        board_config.topology.describe(),
        args.simulations
    );

    let start = Instant::now();
    let result = run_self_play(
        &selfplay_config,
        move || Baduk::new(board_config),
        &evaluator,
    );
    let elapsed = start.elapsed();

    eprintln!(
        "Done: {} games, {} training examples in {:.1}s ({:.1} games/s)",
        result.games_completed,
        result.total_examples,
        elapsed.as_secs_f64(),
        result.games_completed as f64 / elapsed.as_secs_f64()
    );
}

fn run_play(args: PlayArgs) {
    let config = args.board.config();
    let bot = Agent::Heuristic {
        simulations: args.simulations,
        rollouts: args.rollouts,
    };
    let human_player = if args.white { 1 } else { 0 };

    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let stdout = std::io::stdout();
    let mut output = stdout.lock();

    if let Err(err) = interactive(config, bot, human_player, &mut input, &mut output) {
        eprintln!("io error: {}", err);
        std::process::exit(1);
    }
}

fn run_match(args: MatchArgs) {
    let config = args.board.config();
    let opponent_sims = args.opponent_simulations.unwrap_or(args.simulations);
    let opponent_rollouts = args.opponent_rollouts.unwrap_or(args.rollouts);

    let agent_a = Agent::Heuristic {
        simulations: args.simulations,
        rollouts: args.rollouts,
    };
    let agent_b = match args.opponent.as_str() {
        "random" => Agent::Random,
        "uniform" => Agent::Uniform {
            simulations: opponent_sims,
            rollouts: opponent_rollouts,
        },
        "heuristic" => Agent::Heuristic {
            simulations: opponent_sims,
            rollouts: opponent_rollouts,
        },
        other => {
            eprintln!(
                "unknown opponent '{}' (expected random, uniform, or heuristic)",
                other
            );
            std::process::exit(2);
        }
    };

    let game = Baduk::new(config);
    eprintln!(
        "Match on {} ({} nodes): {} vs {}, {} games",
        config.topology.describe(),
        game.num_nodes(),
        agent_a.describe(),
        agent_b.describe(),
        args.games
    );

    let start = Instant::now();
    let result = play_match(
        config,
        agent_a,
        agent_b,
        args.games,
        args.opening_randomness,
    );
    let elapsed = start.elapsed().as_secs_f64();

    println!(
        "{} wins {} - {} ({} draws)  winrate {:.1}%  [{:.1}s, {:.1}s/game]",
        agent_a.describe(),
        result.a_wins,
        result.b_wins,
        result.draws,
        100.0 * result.a_winrate(),
        elapsed,
        elapsed / args.games.max(1) as f64
    );
}
