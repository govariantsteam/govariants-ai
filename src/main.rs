use clap::Parser;
use govariants_ai::game::Game;
use govariants_ai::games::baduk::{Baduk, BadukConfig};
use govariants_ai::mcts::{MctsConfig, NoiseConfig, RandomRolloutEvaluator};
use govariants_ai::selfplay::{SelfPlayConfig, run_self_play};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "govariants-ai")]
#[command(about = "Self-play data generation for Go variant AI training")]
struct Cli {
    /// Game variant to play.
    #[arg(long, default_value = "baduk")]
    variant: String,

    /// Board width.
    #[arg(long, default_value_t = 9)]
    width: usize,

    /// Board height.
    #[arg(long, default_value_t = 9)]
    height: usize,

    /// Komi (points given to white to compensate for black's first-move advantage).
    #[arg(long, default_value_t = 6.5)]
    komi: f32,

    /// Number of self-play games to generate.
    #[arg(long, default_value_t = 100)]
    games: usize,

    /// Number of MCTS simulations per move.
    #[arg(long, default_value_t = 200)]
    simulations: usize,

    /// Temperature for move selection (0 = deterministic, 1 = proportional to visits).
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    /// Dirichlet noise alpha (0 = disabled). Typical: 0.03 for 19x19, 0.3 for 9x9.
    #[arg(long, default_value_t = 0.3)]
    noise_alpha: f32,

    /// Dirichlet noise mixing fraction.
    #[arg(long, default_value_t = 0.25)]
    noise_epsilon: f32,

    /// Number of random rollouts per evaluation (used until neural net is available).
    #[arg(long, default_value_t = 5)]
    rollouts: usize,

    /// Output directory for training data.
    #[arg(long, default_value = "data")]
    output: String,
}

fn main() {
    let cli = Cli::parse();

    match cli.variant.as_str() {
        "baduk" => run_baduk(&cli),
        other => {
            eprintln!("Unknown variant: {}. Available: baduk", other);
            std::process::exit(1);
        }
    }
}

fn run_baduk(cli: &Cli) {
    let board_config = BadukConfig {
        width: cli.width,
        height: cli.height,
        komi: cli.komi,
    };

    let noise = if cli.noise_alpha > 0.0 {
        Some(NoiseConfig {
            alpha: cli.noise_alpha,
            epsilon: cli.noise_epsilon,
        })
    } else {
        None
    };

    let mcts_config = MctsConfig {
        num_simulations: cli.simulations,
        temperature: cli.temperature,
        noise,
        ..Default::default()
    };

    let selfplay_config = SelfPlayConfig {
        num_games: cli.games,
        mcts_config,
        output_dir: cli.output.clone(),
    };

    let evaluator = RandomRolloutEvaluator {
        num_rollouts: cli.rollouts,
    };

    eprintln!(
        "Starting self-play: {} games of {}x{} Baduk, {} sims/move",
        cli.games, cli.width, cli.height, cli.simulations
    );

    let start = Instant::now();
    let result = run_self_play(&selfplay_config, move || Baduk::new(board_config.clone()), &evaluator);
    let elapsed = start.elapsed();

    eprintln!(
        "Done: {} games, {} training examples in {:.1}s ({:.1} games/s)",
        result.games_completed,
        result.total_examples,
        elapsed.as_secs_f64(),
        result.games_completed as f64 / elapsed.as_secs_f64()
    );
}
