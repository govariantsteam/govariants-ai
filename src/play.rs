//! Playing against the engine, and measuring how strong it is.

use crate::game::Game;
use crate::games::baduk::{Baduk, BadukConfig, BadukMove, Color};
use crate::games::topology::Topology;
use crate::mcts::{Evaluator, MctsConfig, RandomRolloutEvaluator, search_detailed};
use crate::rollout::HeuristicRolloutEvaluator;
use rand::prelude::*;
use rand::rng;
use std::io::{BufRead, Write};

/// How a side picks its moves.
#[derive(Clone, Copy, Debug)]
pub enum Agent {
    /// Uniformly random legal move. The floor to measure against.
    Random,
    /// MCTS with uniform priors and purely random rollouts.
    Uniform { simulations: usize, rollouts: usize },
    /// MCTS with the shaped prior and heuristic rollouts.
    Heuristic { simulations: usize, rollouts: usize },
}

impl Agent {
    pub fn describe(&self) -> String {
        match *self {
            Agent::Random => "random".to_string(),
            Agent::Uniform {
                simulations,
                rollouts,
            } => format!("uniform-mcts({}x{})", simulations, rollouts),
            Agent::Heuristic {
                simulations,
                rollouts,
            } => format!("heuristic-mcts({}x{})", simulations, rollouts),
        }
    }

    /// Choose a move. `temperature` only affects the MCTS agents.
    pub fn choose(&self, game: &Baduk, temperature: f32) -> (BadukMove, Option<f32>) {
        match *self {
            Agent::Random => {
                let legal = game.legal_moves();
                let mut rng = rng();
                // Prefer a real move over passing while any exists, otherwise
                // the baseline passes immediately and every game is a draw.
                let placements: Vec<BadukMove> = legal
                    .iter()
                    .copied()
                    .filter(|mv| !matches!(mv, BadukMove::Pass))
                    .collect();
                let pool = if placements.is_empty() {
                    legal
                } else {
                    placements
                };
                (pool[rng.random_range(0..pool.len())], None)
            }
            Agent::Uniform {
                simulations,
                rollouts,
            } => {
                let evaluator = RandomRolloutEvaluator {
                    num_rollouts: rollouts,
                };
                search_move(game, &evaluator, simulations, temperature)
            }
            Agent::Heuristic {
                simulations,
                rollouts,
            } => {
                let evaluator = HeuristicRolloutEvaluator {
                    num_rollouts: rollouts,
                    ..Default::default()
                };
                search_move(game, &evaluator, simulations, temperature)
            }
        }
    }
}

fn search_move<E: Evaluator<Baduk>>(
    game: &Baduk,
    evaluator: &E,
    simulations: usize,
    temperature: f32,
) -> (BadukMove, Option<f32>) {
    let config = MctsConfig {
        num_simulations: simulations,
        temperature,
        noise: None,
        ..Default::default()
    };
    let outcome = search_detailed(game, evaluator, &config);

    let action = if temperature < 1e-6 {
        outcome.best_action
    } else {
        let sum: f32 = outcome.policy.iter().sum();
        if sum > 0.0 {
            let mut rng = rng();
            let dist = rand::distr::weighted::WeightedIndex::new(&outcome.policy).ok();
            dist.map(|d| d.sample(&mut rng)).or(outcome.best_action)
        } else {
            outcome.best_action
        }
    };

    match action {
        Some(idx) => (game.index_to_move(idx), Some(outcome.value)),
        None => (BadukMove::Pass, Some(outcome.value)),
    }
}

/// Render the board, marking the last move if there was one.
pub fn render(game: &Baduk) -> String {
    let stones = game.stones();
    let last = match game.last_move() {
        Some(BadukMove::Place(node)) => Some(node),
        _ => None,
    };
    let cell = |node: usize| -> char {
        if Some(node) == last {
            match stones[node] {
                Color::Black => '#',
                Color::White => '@',
                Color::Empty => '.',
            }
        } else {
            stones[node].glyph()
        }
    };
    game.topology().render(&cell)
}

fn status_line(game: &Baduk) -> String {
    let (black, white) = game.score();
    let caps = game.captures();
    format!(
        "black X: {:.1} pts, {} captured   |   white O: {:.1} pts (komi {:.1}), {} captured",
        black,
        caps[0],
        white,
        game.komi(),
        caps[1]
    )
}

const HELP: &str = "\
Commands:
  <point>     play a stone, e.g. D4 (grid/torus) or Fc2 (cube: face U/L/F/R/B/D)
  pass        pass
  hint        ask the engine what it would play
  score       show the current area score
  nbrs <pt>   list the neighbors of a point (useful on cube boards)
  board       redraw the board
  resign      resign the game
  help        this message
  quit        leave";

/// Interactive game against an agent. Returns when the game ends or the human
/// quits.
pub fn interactive(
    config: &BadukConfig,
    bot: Agent,
    human_player: usize,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
) -> std::io::Result<()> {
    let mut game = Baduk::new(config.clone());
    let human_color = if human_player == 0 {
        "black (X)"
    } else {
        "white (O)"
    };

    writeln!(
        out,
        "\n{} | you are {} | opponent: {}",
        config.topology.describe(),
        human_color,
        bot.describe()
    )?;
    writeln!(out, "Type 'help' for commands.\n")?;
    write!(out, "{}", render(&game))?;
    out.flush()?;

    let mut line = String::new();
    loop {
        if game.is_terminal() {
            break;
        }

        let to_play = game.next_to_play()[0];

        if to_play != human_player {
            let start = std::time::Instant::now();
            let (mv, value) = bot.choose(&game, 0.0);
            let elapsed = start.elapsed().as_secs_f64();
            let label = match mv {
                BadukMove::Place(node) => game.topology().label(node),
                BadukMove::Pass => "pass".to_string(),
            };
            let eval = value
                .map(|v| format!(", engine eval {:+.2}", v))
                .unwrap_or_default();
            writeln!(out, "\nengine plays {} ({:.1}s{})\n", label, elapsed, eval)?;
            game.play_move(to_play, mv);
            write!(out, "{}", render(&game))?;
            writeln!(out, "{}", status_line(&game))?;
            out.flush()?;
            continue;
        }

        write!(out, "\nyour move> ")?;
        out.flush()?;
        line.clear();
        if input.read_line(&mut line)? == 0 {
            writeln!(out, "\nbye")?;
            return Ok(());
        }
        let command = line.trim();
        let lower = command.to_ascii_lowercase();

        match lower.as_str() {
            "" => continue,
            "help" | "?" => {
                writeln!(out, "{}", HELP)?;
            }
            "quit" | "exit" => {
                writeln!(out, "bye")?;
                return Ok(());
            }
            "board" => {
                write!(out, "{}", render(&game))?;
            }
            "score" => {
                writeln!(out, "{}", status_line(&game))?;
            }
            "resign" => {
                writeln!(
                    out,
                    "You resign. {} wins.",
                    if human_player == 0 { "White" } else { "Black" }
                )?;
                return Ok(());
            }
            "pass" => {
                game.play_move(human_player, BadukMove::Pass);
                writeln!(out, "you pass")?;
            }
            "hint" => {
                let (mv, value) = bot.choose(&game, 0.0);
                let label = match mv {
                    BadukMove::Place(node) => game.topology().label(node),
                    BadukMove::Pass => "pass".to_string(),
                };
                let eval = value
                    .map(|v| format!(" (eval {:+.2} for you)", v))
                    .unwrap_or_default();
                writeln!(out, "engine suggests {}{}", label, eval)?;
            }
            _ => {
                if let Some(rest) = lower.strip_prefix("nbrs") {
                    match game.topology().parse(rest.trim()) {
                        Some(node) => {
                            let labels: Vec<String> = game
                                .graph()
                                .neighbors(node)
                                .iter()
                                .map(|&nb| game.topology().label(nb as usize))
                                .collect();
                            writeln!(
                                out,
                                "{} touches {}",
                                game.topology().label(node),
                                labels.join(", ")
                            )?;
                        }
                        None => writeln!(out, "don't know that point")?,
                    }
                    continue;
                }

                match game.topology().parse(command) {
                    Some(node) => {
                        let legal = game.legal_moves();
                        if legal.contains(&BadukMove::Place(node)) {
                            game.play_move(human_player, BadukMove::Place(node));
                            write!(out, "{}", render(&game))?;
                        } else {
                            writeln!(
                                out,
                                "{} is not a legal move there",
                                game.topology().label(node)
                            )?;
                        }
                    }
                    None => {
                        writeln!(out, "didn't understand '{}' - try 'help'", command)?;
                    }
                }
            }
        }
        out.flush()?;
    }

    let (black, white) = game.score();
    write!(out, "{}", render(&game))?;
    writeln!(out, "\nGame over. Black {:.1} - White {:.1}", black, white)?;
    let winner = if black > white {
        "Black (X) wins"
    } else if white > black {
        "White (O) wins"
    } else {
        "Draw"
    };
    writeln!(out, "{} by {:.1}", winner, (black - white).abs())?;
    out.flush()?;
    Ok(())
}

/// Outcome of an engine-vs-engine match.
#[derive(Clone, Copy, Debug, Default)]
pub struct MatchResult {
    pub a_wins: usize,
    pub b_wins: usize,
    pub draws: usize,
}

impl MatchResult {
    pub fn games(&self) -> usize {
        self.a_wins + self.b_wins + self.draws
    }

    pub fn a_winrate(&self) -> f64 {
        if self.games() == 0 {
            return 0.0;
        }
        (self.a_wins as f64 + 0.5 * self.draws as f64) / self.games() as f64
    }
}

/// Play one game between two agents. `a_plays_black` decides the colors.
/// `opening_randomness` moves are played at temperature 1 for variety.
pub fn play_game(
    config: &BadukConfig,
    agent_a: Agent,
    agent_b: Agent,
    a_plays_black: bool,
    opening_randomness: usize,
) -> i32 {
    let mut game = Baduk::new(config.clone());
    let max_moves = game.num_nodes() * 3;
    let mut played = 0;

    while !game.is_terminal() && played < max_moves {
        let to_play = game.next_to_play()[0];
        let a_turn = (to_play == 0) == a_plays_black;
        let agent = if a_turn { agent_a } else { agent_b };
        let temperature = if played < opening_randomness {
            1.0
        } else {
            0.0
        };
        let (mv, _) = agent.choose(&game, temperature);
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

    // Translate to agent A's perspective.
    if a_plays_black {
        black_result
    } else {
        -black_result
    }
}

/// Play `games` games with alternating colors and tally the result.
pub fn play_match(
    config: &BadukConfig,
    agent_a: Agent,
    agent_b: Agent,
    games: usize,
    opening_randomness: usize,
) -> MatchResult {
    use rayon::prelude::*;

    let outcomes: Vec<i32> = (0..games)
        .into_par_iter()
        .map(|i| play_game(config, agent_a, agent_b, i % 2 == 0, opening_randomness))
        .collect();

    let mut result = MatchResult::default();
    for outcome in outcomes {
        match outcome {
            1 => result.a_wins += 1,
            -1 => result.b_wins += 1,
            _ => result.draws += 1,
        }
    }
    result
}

/// Parse a board specification: "9x9", "grid:9x9", "torus:5x5", "cube:3".
pub fn parse_board(spec: &str) -> Result<Topology, String> {
    let lower = spec.trim().to_ascii_lowercase();
    let (kind, dims) = match lower.split_once(':') {
        Some((k, d)) => (k, d),
        None => ("grid", lower.as_str()),
    };

    let parse_dims = |d: &str| -> Result<(usize, usize), String> {
        match d.split_once('x') {
            Some((w, h)) => {
                let w = w
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| format!("bad width '{}'", w))?;
                let h = h
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| format!("bad height '{}'", h))?;
                Ok((w, h))
            }
            None => {
                let n = d
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| format!("bad size '{}'", d))?;
                Ok((n, n))
            }
        }
    };

    match kind {
        "grid" => {
            let (width, height) = parse_dims(dims)?;
            if width == 0 || height == 0 {
                return Err("board must be at least 1x1".to_string());
            }
            Ok(Topology::Grid { width, height })
        }
        "torus" => {
            let (width, height) = parse_dims(dims)?;
            if width < 3 || height < 3 {
                return Err("a torus needs to be at least 3x3 to wrap".to_string());
            }
            Ok(Topology::Torus { width, height })
        }
        "cube" => {
            let size = dims
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("bad cube size '{}'", dims))?;
            if size == 0 {
                return Err("cube size must be at least 1".to_string());
            }
            Ok(Topology::Cube { size })
        }
        other => Err(format!(
            "unknown board kind '{}' (expected grid, torus, or cube)",
            other
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parse_board_forms() {
        assert_eq!(
            parse_board("9x9").unwrap(),
            Topology::Grid {
                width: 9,
                height: 9
            }
        );
        assert_eq!(
            parse_board("grid:5x7").unwrap(),
            Topology::Grid {
                width: 5,
                height: 7
            }
        );
        assert_eq!(
            parse_board("torus:5x5").unwrap(),
            Topology::Torus {
                width: 5,
                height: 5
            }
        );
        assert_eq!(parse_board("cube:3").unwrap(), Topology::Cube { size: 3 });
        assert!(parse_board("torus:2x2").is_err());
        assert!(parse_board("sphere:3").is_err());
        assert!(parse_board("cube:x").is_err());
    }

    #[test]
    fn interactive_session_accepts_moves_and_quits() {
        let config = BadukConfig::grid(5, 5, 0.5);
        let script = "help\nscore\nC3\nnbrs C3\nquit\n";
        let mut input = Cursor::new(script.as_bytes());
        let mut output: Vec<u8> = Vec::new();
        interactive(&config, Agent::Random, 0, &mut input, &mut output).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("Commands:"));
        assert!(text.contains("touches"));
        assert!(text.contains("bye"));
    }

    #[test]
    fn interactive_rejects_illegal_point() {
        let config = BadukConfig::grid(5, 5, 0.5);
        // C3 twice: the second attempt is occupied.
        let script = "C3\nC3\nquit\n";
        let mut input = Cursor::new(script.as_bytes());
        let mut output: Vec<u8> = Vec::new();
        interactive(&config, Agent::Random, 0, &mut input, &mut output).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("not a legal move"));
    }

    #[test]
    fn a_full_game_finishes_and_is_scored() {
        let config = BadukConfig::grid(5, 5, 0.5);
        let outcome = play_game(
            &config,
            Agent::Heuristic {
                simulations: 40,
                rollouts: 2,
            },
            Agent::Random,
            true,
            0,
        );
        assert!(outcome == 1 || outcome == -1 || outcome == 0);
    }
}
