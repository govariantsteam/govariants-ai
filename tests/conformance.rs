//! Replays real games from govariants.com against the Rust rules engine.
//!
//! The Rust engine is a second implementation of rules that already exist in
//! TypeScript, and two independent implementations of the same rules always
//! diverge somewhere. Hand-written tests only cover the cases we thought of;
//! these are real games, including whatever strange shapes real players
//! produced, with the board after every round as the server computed it.
//!
//! Refresh or extend the corpus with:
//!     python3 tools/fetch_corpus.py --board-type grid --games 6
//!
//! A failure here means the port disagrees with the server. Pointing a bot at
//! the live site with a divergence present would make it submit moves the
//! server rejects, or play on from a position it has misread.

use govariants_ai::game::Game;
use govariants_ai::games::baduk::{Baduk, BadukConfig, BadukMove, Color, Scratch};
use govariants_ai::govariants::{decode_move, grid_dimensions, topology_from_config};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// How the corpus encodes stone colors.
const GLYPHS: &[u8] = b".XO#*+";

struct Divergence {
    game: String,
    round: usize,
    detail: String,
}

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus")
}

fn glyph(color: Color) -> char {
    match color {
        Color::Empty => '.',
        Color::Black => 'X',
        Color::White => 'O',
    }
}

fn render_rows(game: &Baduk, grid: Option<(usize, usize)>) -> Vec<String> {
    let stones = game.stones();
    match grid {
        Some((width, height)) => (0..height)
            .map(|y| (0..width).map(|x| glyph(stones[y * width + x])).collect())
            .collect(),
        // Graph boards serialize as one flat array in node order.
        None => vec![stones.iter().map(|&c| glyph(c)).collect()],
    }
}

fn expected_rows(board: &Value) -> Result<Vec<String>, String> {
    if let Some(flat) = board.as_str() {
        return Ok(vec![flat.to_string()]);
    }
    let rows = board
        .as_array()
        .ok_or("round board is neither a string nor an array")?;
    rows.iter()
        .map(|row| {
            row.as_str()
                .map(str::to_string)
                .ok_or_else(|| "corpus board row is not a string; re-fetch the corpus".to_string())
        })
        .collect()
}

fn diff_boards(expected: &[String], actual: &[String]) -> String {
    let mut out = String::from("\n      server              engine\n");
    for (e, a) in expected.iter().zip(actual.iter()) {
        let marker = if e == a { "  " } else { " <" };
        out.push_str(&format!("      {}   {}{}\n", e, a, marker));
    }
    out
}

/// Replay one corpus game. Returns the first divergence found, if any.
fn replay(path: &Path) -> Result<usize, Divergence> {
    let text = std::fs::read_to_string(path).expect("corpus file is readable");
    let record: Value = serde_json::from_str(&text).expect("corpus file is valid JSON");

    let game_id = record
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>")
        .to_string();
    let fail = |round: usize, detail: String| Divergence {
        game: game_id.clone(),
        round,
        detail,
    };

    assert_eq!(
        record.get("glyphs").and_then(Value::as_str),
        Some(std::str::from_utf8(GLYPHS).unwrap()),
        "corpus glyph table changed; update GLYPHS"
    );

    let config = record.get("config").expect("game has a config");
    let board_config = config.get("board").expect("config has a board");
    let topology = match topology_from_config(board_config) {
        Ok(t) => t,
        Err(reason) => return Err(fail(0, format!("unsupported board: {}", reason))),
    };
    // Grids are addressed as SGF coordinates in a 2D board; every other
    // topology is a flat node list addressed by index.
    let grid = grid_dimensions(&topology);
    let komi = config.get("komi").and_then(Value::as_f64).unwrap_or(6.5) as f32;

    let mut game = Baduk::new(BadukConfig {
        topology: topology.clone(),
        komi,
    });
    let mut scratch: Scratch = game.new_scratch();

    let rounds = record
        .get("rounds")
        .and_then(Value::as_array)
        .expect("game has rounds");

    // Round 0 is the initial position; round N is the board after N moves,
    // and its `last_move` is the move that produced it.
    for (index, round) in rounds.iter().enumerate() {
        if index > 0 {
            let move_text = round
                .get("last_move")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();

            // Who had the move is taken from the previous round rather than
            // assumed to alternate, so variants with other turn orders replay
            // correctly too.
            let player = rounds[index - 1]
                .get("next_to_play")
                .and_then(Value::as_array)
                .and_then(|p| p.first())
                .and_then(Value::as_u64)
                .unwrap_or(((index - 1) % 2) as u64) as usize;

            match move_text.as_str() {
                "resign" | "timeout" => break, // ends the game; no board change
                "pass" => {
                    if !game.try_play(BadukMove::Pass, &mut scratch) {
                        return Err(fail(index, "engine refused a pass".into()));
                    }
                }
                other => {
                    let node = match decode_move(&topology, other) {
                        Ok(n) => n,
                        Err(reason) => return Err(fail(index, reason)),
                    };
                    if game.next_to_play() != vec![player] {
                        return Err(fail(
                            index,
                            format!(
                                "turn mismatch: server says player {} plays '{}', engine expects {:?}",
                                player, other, game.next_to_play()
                            ),
                        ));
                    }
                    if !game.try_play(BadukMove::Place(node), &mut scratch, ) {
                        return Err(fail(
                            index,
                            format!(
                                "engine rejected '{}' (node {}) as illegal, but the server accepted it",
                                other, node
                            ),
                        ));
                    }
                }
            }
        }

        // Compare the board.
        let Some(board) = round.get("board") else {
            continue;
        };
        let expected = match expected_rows(board) {
            Ok(rows) => rows,
            Err(reason) => return Err(fail(index, reason)),
        };
        let actual = render_rows(&game, grid);
        if expected != actual {
            return Err(fail(
                index,
                format!(
                    "board differs after move '{}':{}",
                    round
                        .get("last_move")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                    diff_boards(&expected, &actual)
                ),
            ));
        }

        // Compare capture counts.
        if let Some(captures) = round.get("captures") {
            let server = [
                captures.get("0").and_then(Value::as_u64).unwrap_or(0) as usize,
                captures.get("1").and_then(Value::as_u64).unwrap_or(0) as usize,
            ];
            let ours = game.captures();
            if server != ours {
                return Err(fail(
                    index,
                    format!(
                        "captures differ: server {:?}, engine {:?}",
                        server, ours
                    ),
                ));
            }
        }
    }

    Ok(rounds.len())
}

#[test]
fn replays_real_games_from_govariants() {
    let dir = corpus_dir();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read corpus dir {}: {}", dir.display(), e))
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();

    assert!(
        !files.is_empty(),
        "no corpus games in {}; run tools/fetch_corpus.py",
        dir.display()
    );

    let mut replayed_games = 0;
    let mut replayed_rounds = 0;
    let mut skipped = Vec::new();
    let mut failures = Vec::new();

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        match replay(path) {
            Ok(rounds) => {
                replayed_games += 1;
                replayed_rounds += rounds;
                println!("  ok   {} ({} rounds)", name, rounds);
            }
            Err(div) if div.detail.starts_with("unsupported board") => {
                skipped.push(format!("{}: {}", name, div.detail));
            }
            Err(div) => {
                println!(
                    "  FAIL {} at round {}: {}",
                    name, div.round, div.detail
                );
                failures.push(div);
            }
        }
    }

    println!(
        "\nreplayed {} games / {} rounds against govariants",
        replayed_games, replayed_rounds
    );
    for note in &skipped {
        println!("  skipped {}", note);
    }

    assert!(
        failures.is_empty(),
        "{} game(s) diverged from the server; first: {} at round {}: {}",
        failures.len(),
        failures[0].game,
        failures[0].round,
        failures[0].detail
    );
    assert!(
        replayed_games > 0,
        "every corpus game was skipped; nothing was actually verified"
    );
}
