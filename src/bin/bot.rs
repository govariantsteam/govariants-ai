//! A bot that plays on a govariants server.
//!
//! It is an ordinary client: it logs in with a password like any user, takes a
//! seat, polls for its turn, and posts moves. Nothing server-side is required.
//!
//! Safety properties, because this talks to a live site people are playing on:
//!   * It refuses to sit at a board it cannot parse, rather than sitting and
//!     then playing badly.
//!   * `probe` is read-only and logs the move it *would* play.
//!   * `play --dry-run` does everything except submit.
//!   * If the server rejects a move it falls back to the next-best one. Ko
//!     state cannot be recovered from a single board (see
//!     `Baduk::from_position`), so a rejection is expected, not exceptional.
//!   * The account name is meant to be obviously a bot; there is no `bot` user
//!     role on the server yet, so the username is the only disclosure.

use clap::{Parser, Subcommand};
use govariants_ai::game::Game;
use govariants_ai::games::baduk::{Baduk, BadukMove};
use govariants_ai::games::topology::Topology;
use govariants_ai::govariants::{
    Client, DEFAULT_BASE, LivePosition, encode_move, position_from_state, topology_from_config,
};
use serde_json::json;
use govariants_ai::mcts::{MctsConfig, search_detailed};
use govariants_ai::rollout::HeuristicRolloutEvaluator;
use serde_json::Value;
use std::time::Duration;

const CREDENTIALS_FILE: &str = ".govariants-bot-credentials";

#[derive(Parser)]
#[command(name = "bot")]
#[command(about = "Play govariants as a bot account")]
struct Cli {
    /// API base URL. Point this at a local instance to test.
    #[arg(long, default_value = DEFAULT_BASE, global = true)]
    base: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create the bot account. Writes credentials to a local file.
    Register {
        /// Account name. Must be alphanumeric (the server rejects hyphens and
        /// underscores), and should read unmistakably as a bot.
        #[arg(long, default_value = "govariantsAiBot")]
        username: String,
    },
    /// Log in and show who we are.
    Whoami,
    /// List games, marking which ones this bot could join.
    Games {
        #[arg(long, default_value = "baduk")]
        variant: String,
        #[arg(long, default_value_t = 20)]
        count: usize,
    },
    /// Read a game and report the move the engine would play. No writes.
    Probe {
        #[arg(long)]
        game_id: String,
        #[arg(long, default_value_t = 2000)]
        simulations: usize,
    },
    /// Create a game, sit down, and wait for a human to join.
    ///
    /// Preferred over joining a stranger's open game: nobody ends up playing a
    /// bot without choosing to.
    Host {
        /// Board: "9x9", "grid:5x7", "torus:7x7", "cube:3".
        #[arg(long, default_value = "9x9")]
        board: String,
        #[arg(long, default_value_t = 6.5)]
        komi: f32,
        /// Seat for the bot. Default 1 (white), so the human plays black and
        /// moves first.
        #[arg(long, default_value_t = 1)]
        seat: usize,
        #[arg(long, default_value_t = 2000)]
        simulations: usize,
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
        #[arg(long, default_value_t = 3)]
        poll_seconds: u64,
    },
    /// Give up a seat, e.g. if the bot got stuck in a game.
    Leave {
        #[arg(long)]
        game_id: String,
        #[arg(long)]
        seat: usize,
    },
    /// Join an existing game and play it.
    Play {
        #[arg(long)]
        game_id: String,
        /// Seat to take. Defaults to the first empty one.
        #[arg(long)]
        seat: Option<usize>,
        #[arg(long, default_value_t = 2000)]
        simulations: usize,
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
        /// Seconds between polls while waiting for the opponent.
        #[arg(long, default_value_t = 3)]
        poll_seconds: u64,
        /// Decide moves but never submit them.
        #[arg(long)]
        dry_run: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Register { username } => register(&cli.base, username),
        Command::Whoami => whoami(&cli.base),
        Command::Games { variant, count } => list_games(&cli.base, variant, *count),
        Command::Probe {
            game_id,
            simulations,
        } => probe(&cli.base, game_id, *simulations),
        Command::Leave { game_id, seat } => leave(&cli.base, game_id, *seat),
        Command::Host {
            board,
            komi,
            seat,
            simulations,
            rollouts,
            poll_seconds,
        } => host(
            &cli.base,
            board,
            *komi,
            *seat,
            *simulations,
            *rollouts,
            *poll_seconds,
        ),
        Command::Play {
            game_id,
            seat,
            simulations,
            rollouts,
            poll_seconds,
            dry_run,
        } => play(
            &cli.base,
            game_id,
            *seat,
            *simulations,
            *rollouts,
            *poll_seconds,
            *dry_run,
        ),
    };

    if let Err(message) = result {
        eprintln!("error: {}", message);
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

fn generate_password() -> String {
    use rand::Rng;
    use rand::distr::Alphanumeric;
    rand::rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect()
}

/// Credentials come from the environment, falling back to the local file that
/// `register` writes. They are never committed and never printed.
fn load_credentials() -> Result<(String, String), String> {
    if let (Ok(user), Ok(password)) = (
        std::env::var("GOVARIANTS_BOT_USERNAME"),
        std::env::var("GOVARIANTS_BOT_PASSWORD"),
    ) {
        return Ok((user, password));
    }

    let text = std::fs::read_to_string(CREDENTIALS_FILE).map_err(|_| {
        format!(
            "no credentials: set GOVARIANTS_BOT_USERNAME and GOVARIANTS_BOT_PASSWORD, \
             or run `bot register` to create {}",
            CREDENTIALS_FILE
        )
    })?;

    let mut username = None;
    let mut password = None;
    for line in text.lines() {
        if let Some((key, value)) = line.split_once('=') {
            match key.trim() {
                "GOVARIANTS_BOT_USERNAME" => username = Some(value.trim().to_string()),
                "GOVARIANTS_BOT_PASSWORD" => password = Some(value.trim().to_string()),
                _ => {}
            }
        }
    }
    match (username, password) {
        (Some(u), Some(p)) => Ok((u, p)),
        _ => Err(format!("{} is missing a username or password", CREDENTIALS_FILE)),
    }
}

fn save_credentials(username: &str, password: &str) -> Result<(), String> {
    let body = format!(
        "# Credentials for the govariants bot account. Not for committing.\n\
         GOVARIANTS_BOT_USERNAME={}\n\
         GOVARIANTS_BOT_PASSWORD={}\n",
        username, password
    );
    std::fs::write(CREDENTIALS_FILE, body)
        .map_err(|e| format!("could not write {}: {}", CREDENTIALS_FILE, e))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(
            CREDENTIALS_FILE,
            std::fs::Permissions::from_mode(0o600),
        );
    }
    Ok(())
}

fn connect(base: &str) -> Result<Client, String> {
    let (username, password) = load_credentials()?;
    let mut client = Client::new(base).map_err(|e| e.to_string())?;
    client
        .login(&username, &password)
        .map_err(|e| format!("login failed: {}", e))?;
    Ok(client)
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn register(base: &str, username: &str) -> Result<(), String> {
    if std::path::Path::new(CREDENTIALS_FILE).exists() {
        return Err(format!(
            "{} already exists; delete it first if you really want a new account",
            CREDENTIALS_FILE
        ));
    }
    let password = std::env::var("GOVARIANTS_BOT_PASSWORD").unwrap_or_else(|_| generate_password());

    let mut client = Client::new(base).map_err(|e| e.to_string())?;
    client
        .register(username, &password)
        .map_err(|e| format!("registration failed: {}", e))?;

    save_credentials(username, &password)?;
    println!("registered '{}' on {}", username, base);
    println!(
        "credentials written to {} (mode 0600, gitignored); the password was not printed",
        CREDENTIALS_FILE
    );
    if let Some(id) = client.user_id() {
        println!("user id: {}", id);
    }
    Ok(())
}

fn whoami(base: &str) -> Result<(), String> {
    let client = connect(base)?;
    println!(
        "logged in as {} (id {})",
        client.username().unwrap_or("<unknown>"),
        client.user_id().unwrap_or("<unknown>")
    );
    Ok(())
}

/// Seat index occupied by `user_id`, if any.
fn our_seat(initial: &Value, user_id: &str) -> Option<usize> {
    initial
        .get("players")
        .and_then(Value::as_array)?
        .iter()
        .position(|player| player.get("id").and_then(Value::as_str) == Some(user_id))
}

fn empty_seat(initial: &Value) -> Option<usize> {
    initial
        .get("players")
        .and_then(Value::as_array)?
        .iter()
        .position(|player| player.is_null())
}

fn list_games(base: &str, variant: &str, count: usize) -> Result<(), String> {
    let client = connect(base)?;
    let games = client
        .games(variant, count, 0)
        .map_err(|e| format!("could not list games: {}", e))?;

    println!(
        "{:<26} {:<16} {:<9} {:<7} playable",
        "game id", "board", "phase", "seats"
    );
    for game in &games {
        let id = game.get("id").and_then(Value::as_str).unwrap_or("?");
        let board_config = game.get("config").and_then(|c| c.get("board"));
        let board_type = board_config
            .and_then(|b| b.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("?");
        let phase = game.get("phase").and_then(Value::as_str).unwrap_or("?");
        let open = empty_seat(game).is_some();
        let supported = board_config.map(topology_from_config);
        let playable = match (&supported, phase, open) {
            (Some(Ok(_)), "play", true) => "yes".to_string(),
            (Some(Ok(_)), "play", false) => "no (full)".to_string(),
            (Some(Ok(_)), _, _) => format!("no ({})", phase),
            (Some(Err(why)), _, _) => format!("no ({})", why),
            (None, _, _) => "no (no board config)".to_string(),
        };
        println!(
            "{:<26} {:<16} {:<9} {:<7} {}",
            id,
            board_type,
            phase,
            if open { "open" } else { "full" },
            playable
        );
    }
    Ok(())
}

/// Fetch a game and translate it into a position.
fn fetch_position(client: &Client, game_id: &str) -> Result<(Value, LivePosition), String> {
    let initial = client
        .initial_state(game_id)
        .map_err(|e| format!("could not read game: {}", e))?;
    let config = initial
        .get("config")
        .cloned()
        .ok_or("game has no config")?;
    let state = client
        .state(game_id)
        .map_err(|e| format!("could not read state: {}", e))?;
    let position = position_from_state(&config, &state)?;
    Ok((initial, position))
}

/// Moves the engine likes, best first, so a rejected move has successors.
fn ranked_moves(game: &Baduk, simulations: usize, rollouts: usize) -> (Vec<BadukMove>, f32) {
    let evaluator = HeuristicRolloutEvaluator {
        num_rollouts: rollouts,
        ..Default::default()
    };
    let config = MctsConfig {
        num_simulations: simulations,
        temperature: 0.0,
        noise: None,
        ..Default::default()
    };
    let outcome = search_detailed(game, &evaluator, &config);

    let mut scored: Vec<(usize, f32)> = outcome
        .visits
        .iter()
        .enumerate()
        .filter(|(_, visits)| **visits > 0.0)
        .map(|(action, visits)| (action, *visits))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let moves = scored
        .into_iter()
        .map(|(action, _)| game.index_to_move(action))
        .collect();
    (moves, outcome.value)
}

fn describe_move(position: &LivePosition, mv: BadukMove) -> String {
    match mv {
        BadukMove::Pass => "pass".to_string(),
        BadukMove::Place(node) => format!(
            "{} (node {})",
            encode_move(&position.topology, node),
            node
        ),
    }
}

fn wire_move(position: &LivePosition, mv: BadukMove) -> String {
    match mv {
        BadukMove::Pass => "pass".to_string(),
        BadukMove::Place(node) => encode_move(&position.topology, node),
    }
}

fn probe(base: &str, game_id: &str, simulations: usize) -> Result<(), String> {
    // Reading a game needs no session.
    let client = Client::new(base).map_err(|e| e.to_string())?;
    let (initial, position) = fetch_position(&client, game_id)?;

    let variant = initial
        .get("variant")
        .and_then(Value::as_str)
        .unwrap_or("?");
    println!("game {} | variant {}", game_id, variant);
    println!(
        "board: {} | round {} | phase {}",
        position.topology.describe(),
        position.round,
        position.phase
    );
    println!("next to play: {:?}", position.next_to_play);
    println!("captures: {:?}", position.game.captures());
    let (black, white) = position.game.score();
    println!("area score as it stands: black {:.1}, white {:.1}", black, white);
    print!("{}", govariants_ai::play::render(&position.game));

    if position.next_to_play.is_empty() {
        println!("\ngame is over; nothing to play");
        return Ok(());
    }

    let (moves, value) = ranked_moves(&position.game, simulations, 1);
    println!("\nengine eval for the player to move: {:+.2}", value);
    println!("it would play, best first:");
    for mv in moves.iter().take(5) {
        println!("  {}", describe_move(&position, *mv));
    }
    Ok(())
}

fn leave(base: &str, game_id: &str, seat: usize) -> Result<(), String> {
    let client = connect(base)?;
    client
        .leave(game_id, seat)
        .map_err(|e| format!("could not leave seat {}: {}", seat, e))?;
    println!("left seat {} of game {}", seat, game_id);
    Ok(())
}

/// Our topology spec -> the board config govariants expects.
fn board_config_json(topology: &Topology) -> Result<Value, String> {
    match topology {
        Topology::Grid { width, height } => {
            Ok(json!({"type": "grid", "width": width, "height": height}))
        }
        // The server has no torus or cube *board type*; cube is a separate
        // variant there. Hosting those would need variant support first.
        other => Err(format!(
            "{} cannot be hosted on govariants: the server has no such board type",
            other.describe()
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn host(
    base: &str,
    board: &str,
    komi: f32,
    seat: usize,
    simulations: usize,
    rollouts: usize,
    poll_seconds: u64,
) -> Result<(), String> {
    let topology = govariants_ai::play::parse_board(board)?;
    let board_json = board_config_json(&topology)?;

    let mut client = connect(base)?;
    client.refresh_session().map_err(|e| e.to_string())?;

    let config = json!({
        "komi": komi,
        "board": board_json,
        "time_control": null,
    });
    let game = client
        .create_game("baduk", config)
        .map_err(|e| format!("could not create a game: {}", e))?;
    let game_id = game
        .get("id")
        .and_then(Value::as_str)
        .ok_or("server did not return a game id")?
        .to_string();

    client
        .sit(&game_id, seat)
        .map_err(|e| format!("created game {} but could not sit: {}", game_id, e))?;

    let site = base.trim_end_matches("/api");
    println!("created game {} on {}", game_id, topology.describe());
    println!("bot is seated at seat {} ({})", seat, if seat == 0 { "black" } else { "white" });
    println!();
    println!("  open {}/game/{} and take the other seat", site, game_id);
    println!();
    println!("waiting for an opponent...");

    play_loop(
        &client,
        &game_id,
        seat,
        simulations,
        rollouts,
        poll_seconds,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn play(
    base: &str,
    game_id: &str,
    seat: Option<usize>,
    simulations: usize,
    rollouts: usize,
    poll_seconds: u64,
    dry_run: bool,
) -> Result<(), String> {
    let mut client = connect(base)?;
    client.refresh_session().map_err(|e| e.to_string())?;
    let user_id = client
        .user_id()
        .ok_or("logged in but the server did not say who we are")?
        .to_string();

    let (initial, position) = fetch_position(&client, game_id)?;
    println!(
        "game {} | {} | round {} | phase {}",
        game_id,
        position.topology.describe(),
        position.round,
        position.phase
    );

    // An explicit --seat is honored: take it if we are not already in it,
    // rather than silently playing whichever seat we happen to occupy.
    let already = our_seat(&initial, &user_id);
    let target = match (seat, already) {
        (Some(requested), Some(existing)) if requested != existing => {
            println!(
                "requested seat {} but already seated at {}; taking {}",
                requested, existing, requested
            );
            requested
        }
        (Some(requested), _) => requested,
        (None, Some(existing)) => {
            println!("already seated at seat {}", existing);
            existing
        }
        (None, None) => empty_seat(&initial).ok_or("no empty seat in this game")?,
    };

    // Refuse to sit at a board we cannot play, rather than occupying a seat.
    let our = if already == Some(target) {
        target
    } else if dry_run {
        println!("dry run: would take seat {}", target);
        target
    } else {
        client
            .sit(game_id, target)
            .map_err(|e| format!("could not take seat {}: {}", target, e))?;
        println!("took seat {}", target);
        target
    };

    println!(
        "playing as seat {} with {} simulations x {} rollouts{}",
        our,
        simulations,
        rollouts,
        if dry_run { " (dry run)" } else { "" }
    );

    play_loop(
        &client,
        game_id,
        our,
        simulations,
        rollouts,
        poll_seconds,
        dry_run,
    )
}

#[allow(clippy::too_many_arguments)]
fn play_loop(
    client: &Client,
    game_id: &str,
    our: usize,
    simulations: usize,
    rollouts: usize,
    poll_seconds: u64,
    dry_run: bool,
) -> Result<(), String> {
    loop {
        let (_, position) = fetch_position(client, game_id)?;

        if position.phase == "gameover" || position.next_to_play.is_empty() {
            let (black, white) = position.game.score();
            println!(
                "game over at round {} | area score black {:.1}, white {:.1}",
                position.round, black, white
            );
            return Ok(());
        }

        if !position.next_to_play.contains(&our) {
            std::thread::sleep(Duration::from_secs(poll_seconds));
            continue;
        }

        let (moves, value) = ranked_moves(&position.game, simulations, rollouts);
        if moves.is_empty() {
            return Err("engine produced no moves".to_string());
        }
        println!(
            "round {}: thinking... eval {:+.2}, first choice {}",
            position.round,
            value,
            describe_move(&position, moves[0])
        );

        if dry_run {
            println!("dry run: not submitting. Stopping here.");
            return Ok(());
        }

        // A rejection is expected rather than exceptional: ko state is not
        // recoverable from a single board, so try the next-best move.
        let mut submitted = false;
        for (attempt, mv) in moves.iter().take(8).enumerate() {
            let text = wire_move(&position, *mv);
            match client.submit_move(game_id, our, &text) {
                Ok(_) => {
                    println!("  played {}", describe_move(&position, *mv));
                    submitted = true;
                    break;
                }
                Err(err) => {
                    println!(
                        "  server rejected {} ({}); trying next choice",
                        describe_move(&position, *mv),
                        err
                    );
                    if attempt == 7 {
                        return Err("server rejected every candidate move".to_string());
                    }
                }
            }
        }
        if !submitted {
            return Err("could not submit a move".to_string());
        }
    }
}
