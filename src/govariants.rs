//! Client for a govariants server, and translation between its wire format
//! and this engine.
//!
//! A bot needs no server-side support: it registers, logs in, takes a seat and
//! posts moves exactly as a browser does. Auth is a session cookie plus a
//! `CSRF-Token` header obtained from `/checkLogin`.
//!
//! Live updates are available over Socket.IO, but this polls instead. The game
//! is turn-based, so polling costs a request every second or two and avoids a
//! websocket client entirely.
//!
//! The conformance harness (`tests/conformance.rs`) drives the translation
//! functions here against real recorded games, so this module is what gets
//! verified, not a test-only copy of it.

use crate::games::baduk::{Baduk, BadukConfig, Color};
use crate::games::topology::Topology;
use serde_json::{Value, json};
use std::time::Duration;

pub const DEFAULT_BASE: &str = "https://www.govariants.com/api";

/// Colors as govariants serializes them.
const EMPTY: u64 = 0;
const BLACK: u64 = 1;
const WHITE: u64 = 2;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Http(String),
    /// The server answered, but with a refusal.
    Server { status: u16, body: String },
    /// The server's data could not be translated into a position.
    Translate(String),
    /// Something we need is missing from a response.
    Missing(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Http(msg) => write!(f, "http error: {}", msg),
            Error::Server { status, body } => write!(f, "server returned {}: {}", status, body),
            Error::Translate(msg) => write!(f, "cannot translate: {}", msg),
            Error::Missing(what) => write!(f, "response is missing {}", what),
        }
    }
}

impl std::error::Error for Error {}

impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        Error::Http(err.to_string())
    }
}

/// An authenticated govariants session.
pub struct Client {
    http: reqwest::blocking::Client,
    base: String,
    csrf_token: Option<String>,
    user_id: Option<String>,
    username: Option<String>,
}

impl Client {
    pub fn new(base: &str) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .cookie_store(true)
            .timeout(Duration::from_secs(30))
            // Identify the client generically. No user details go on the wire.
            .user_agent("govariants-ai-bot/0.1")
            .build()?;
        Ok(Client {
            http,
            base: base.trim_end_matches('/').to_string(),
            csrf_token: None,
            user_id: None,
            username: None,
        })
    }

    pub fn user_id(&self) -> Option<&str> {
        self.user_id.as_deref()
    }

    pub fn username(&self) -> Option<&str> {
        self.username.as_deref()
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    fn get(&self, path: &str) -> Result<Value> {
        let response = self.http.get(self.url(path)).send()?;
        Self::decode(response)
    }

    fn post(&self, path: &str, body: Option<Value>) -> Result<Value> {
        let mut request = self.http.post(self.url(path));
        if let Some(token) = &self.csrf_token {
            request = request.header("CSRF-Token", token);
        }
        if let Some(body) = body {
            request = request.json(&body);
        }
        Self::decode(request.send()?)
    }

    fn decode(response: reqwest::blocking::Response) -> Result<Value> {
        let status = response.status();
        let text = response.text().unwrap_or_default();
        if !status.is_success() {
            return Err(Error::Server {
                status: status.as_u16(),
                body: text.chars().take(400).collect(),
            });
        }
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).map_err(|e| {
            Error::Http(format!(
                "could not parse response as JSON ({}): {}",
                e,
                text.chars().take(200).collect::<String>()
            ))
        })
    }

    /// Create an account. Email is optional and deliberately not sent.
    pub fn register(&mut self, username: &str, password: &str) -> Result<Value> {
        let user = self.post(
            "/register",
            Some(json!({"username": username, "password": password})),
        )?;
        self.refresh_session()?;
        Ok(user)
    }

    pub fn login(&mut self, username: &str, password: &str) -> Result<Value> {
        let user = self.post(
            "/login",
            Some(json!({"username": username, "password": password})),
        )?;
        self.refresh_session()?;
        Ok(user)
    }

    /// Pick up the CSRF token and our own user id, which mutating calls need.
    pub fn refresh_session(&mut self) -> Result<()> {
        let body = self.get("/checkLogin")?;
        if body.is_null() {
            return Err(Error::Missing("a logged-in session"));
        }
        self.csrf_token = body
            .get("csrf_token")
            .and_then(Value::as_str)
            .map(str::to_string);
        self.user_id = body
            .pointer("/user/id")
            .and_then(Value::as_str)
            .map(str::to_string);
        self.username = body
            .pointer("/user/username")
            .and_then(Value::as_str)
            .map(str::to_string);
        Ok(())
    }

    pub fn games(&self, variant: &str, count: usize, offset: usize) -> Result<Vec<Value>> {
        let body = self.get(&format!(
            "/games?variant={}&count={}&offset={}",
            variant, count, offset
        ))?;
        Ok(body.as_array().cloned().unwrap_or_default())
    }

    /// Create a game. `config` is the variant's own config object, e.g.
    /// `{"komi": 6.5, "board": {"type": "grid", "width": 9, "height": 9}}`.
    pub fn create_game(&self, variant: &str, config: Value) -> Result<Value> {
        self.post(
            "/games",
            Some(json!({"variant": variant, "config": config})),
        )
    }

    pub fn initial_state(&self, game_id: &str) -> Result<Value> {
        self.get(&format!("/games/{}/state/initial", game_id))
    }

    pub fn state(&self, game_id: &str) -> Result<Value> {
        self.get(&format!("/games/{}/state?seat=&round=", game_id))
    }

    pub fn sit(&self, game_id: &str, seat: usize) -> Result<Value> {
        self.post(&format!("/games/{}/sit/{}", game_id, seat), None)
    }

    pub fn leave(&self, game_id: &str, seat: usize) -> Result<Value> {
        self.post(&format!("/games/{}/leave/{}", game_id, seat), None)
    }

    /// Submit a move. `text` is already in the variant's encoding.
    pub fn submit_move(&self, game_id: &str, seat: usize, text: &str) -> Result<Value> {
        self.post(
            &format!("/games/{}/move", game_id),
            Some(json!({seat.to_string(): text})),
        )
    }
}

// ---------------------------------------------------------------------------
// Translation
// ---------------------------------------------------------------------------

/// govariants board config -> our topology.
pub fn topology_from_config(board: &Value) -> std::result::Result<Topology, String> {
    let kind = board
        .get("type")
        .and_then(Value::as_str)
        .ok_or("board config has no type")?;
    match kind {
        "grid" => {
            let width = board
                .get("width")
                .and_then(Value::as_u64)
                .ok_or("grid board has no width")? as usize;
            let height = board
                .get("height")
                .and_then(Value::as_u64)
                .ok_or("grid board has no height")? as usize;
            Ok(Topology::Grid { width, height })
        }
        "circular" => {
            let rings = board
                .get("rings")
                .and_then(Value::as_u64)
                .ok_or("circular board has no rings")? as usize;
            let nodes_per_ring = board
                .get("nodesPerRing")
                .and_then(Value::as_u64)
                .ok_or("circular board has no nodesPerRing")? as usize;
            Ok(Topology::Circular {
                rings,
                nodes_per_ring,
            })
        }
        "gridWithHoles" => {
            let rows = board
                .get("bitmap")
                .and_then(Value::as_array)
                .ok_or("gridWithHoles board has no bitmap")?;
            let bitmap = rows
                .iter()
                .map(|row| {
                    row.as_array()
                        .ok_or_else(|| "bitmap row is not an array".to_string())
                        .map(|cells| {
                            cells
                                .iter()
                                .map(|c| c.as_u64().unwrap_or(0) as u8)
                                .collect::<Vec<u8>>()
                        })
                })
                .collect::<std::result::Result<Vec<Vec<u8>>, String>>()?;
            Ok(Topology::GridWithHoles { bitmap })
        }
        // polygonal, sierpinsky, trihexagonal, sunflower and custom are not
        // implemented. See govariants' boardFactory and its board helpers.
        other => Err(format!("board type '{}' is not implemented", other)),
    }
}

/// Whether a topology is addressed as a 2D grid (SGF coordinates) rather than
/// as a flat list of nodes (decimal indices).
pub fn grid_dimensions(topology: &Topology) -> Option<(usize, usize)> {
    match topology {
        Topology::Grid { width, height } => Some((*width, *height)),
        _ => None,
    }
}

/// Encode a node the way govariants expects it for this board.
pub fn encode_move(topology: &Topology, node: usize) -> String {
    match grid_dimensions(topology) {
        Some((width, _)) => {
            let x = node % width;
            let y = node / width;
            let letter = |v: usize| (b'a' + v as u8) as char;
            format!("{}{}", letter(x), letter(y))
        }
        None => node.to_string(),
    }
}

/// Decode a move string from govariants into a node index.
pub fn decode_move(topology: &Topology, text: &str) -> std::result::Result<usize, String> {
    match grid_dimensions(topology) {
        Some((width, height)) => {
            let bytes = text.as_bytes();
            if bytes.len() != 2 {
                return Err(format!("expected a 2-character SGF move, got '{}'", text));
            }
            let x = bytes[0].wrapping_sub(b'a') as usize;
            let y = bytes[1].wrapping_sub(b'a') as usize;
            if x >= width || y >= height {
                return Err(format!("move '{}' is off a {}x{} board", text, width, height));
            }
            Ok(y * width + x)
        }
        None => {
            let node: usize = text
                .parse()
                .map_err(|_| format!("expected a node index, got '{}'", text))?;
            let count = topology.num_nodes();
            if node >= count {
                return Err(format!("node {} is outside a {}-node board", node, count));
            }
            Ok(node)
        }
    }
}

/// Read the stones out of a `state.board`, which is a 2D array for grids and a
/// flat array for graph boards.
pub fn stones_from_board(
    topology: &Topology,
    board: &Value,
) -> std::result::Result<Vec<Color>, String> {
    let translate = |raw: u64| -> std::result::Result<Color, String> {
        match raw {
            EMPTY => Ok(Color::Empty),
            BLACK => Ok(Color::Black),
            WHITE => Ok(Color::White),
            other => Err(format!(
                "stone color {} is not plain baduk (multicolor variants are not supported)",
                other
            )),
        }
    };

    let cells = board.as_array().ok_or("state.board is not an array")?;
    let mut stones = Vec::with_capacity(topology.num_nodes());

    if cells.first().is_some_and(|c| c.is_array()) {
        for row in cells {
            let row = row.as_array().ok_or("board row is not an array")?;
            for cell in row {
                let raw = cell.as_u64().ok_or("board cell is not a number")?;
                stones.push(translate(raw)?);
            }
        }
    } else {
        for cell in cells {
            let raw = cell.as_u64().ok_or("board cell is not a number")?;
            stones.push(translate(raw)?);
        }
    }

    if stones.len() != topology.num_nodes() {
        return Err(format!(
            "board has {} points but the topology has {}",
            stones.len(),
            topology.num_nodes()
        ));
    }
    Ok(stones)
}

/// Everything needed to play one position of a live game.
pub struct LivePosition {
    pub game: Baduk,
    pub topology: Topology,
    pub next_to_play: Vec<usize>,
    pub phase: String,
    pub round: u64,
}

/// Build a playable position from a `/state` or `/state/initial` response.
pub fn position_from_state(
    config: &Value,
    state_response: &Value,
) -> std::result::Result<LivePosition, String> {
    let board_config = config.get("board").ok_or("config has no board")?;
    let topology = topology_from_config(board_config)?;
    let komi = config.get("komi").and_then(Value::as_f64).unwrap_or(6.5) as f32;

    let state = state_response.get("state").ok_or("response has no state")?;
    let board = state.get("board").ok_or("state has no board")?;
    let stones = stones_from_board(&topology, board)?;

    let next_to_play: Vec<usize> = state_response
        .get("next_to_play")
        .and_then(Value::as_array)
        .map(|players| {
            players
                .iter()
                .filter_map(Value::as_u64)
                .map(|p| p as usize)
                .collect()
        })
        .unwrap_or_default();

    let captures = state.get("captures");
    let read_capture = |key: &str| {
        captures
            .and_then(|c| c.get(key))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize
    };

    // The engine needs to know whose turn it is. When the game is over there
    // is no player to move, so default to black; nothing will be played.
    let to_move = next_to_play.first().copied().unwrap_or(0);

    let game = Baduk::from_position(
        BadukConfig {
            topology: topology.clone(),
            komi,
        },
        stones,
        to_move,
        [read_capture("0"), read_capture("1")],
    )?;

    Ok(LivePosition {
        game,
        topology,
        next_to_play,
        phase: state_response
            .get("phase")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        round: state_response
            .get("round")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    use serde_json::json;

    #[test]
    fn grid_moves_use_sgf_coordinates() {
        let t = Topology::Grid {
            width: 9,
            height: 9,
        };
        // "dd" is (3,3) -> row-major 3 * 9 + 3.
        assert_eq!(decode_move(&t, "dd").unwrap(), 30);
        assert_eq!(encode_move(&t, 30), "dd");
        // First letter is the column.
        assert_eq!(decode_move(&t, "ab").unwrap(), 9);
        assert_eq!(encode_move(&t, 9), "ab");
        for node in 0..81 {
            assert_eq!(decode_move(&t, &encode_move(&t, node)).unwrap(), node);
        }
        assert!(decode_move(&t, "zz").is_err());
        assert!(decode_move(&t, "d").is_err());
    }

    #[test]
    fn graph_moves_use_node_indices() {
        let t = Topology::Circular {
            rings: 3,
            nodes_per_ring: 6,
        };
        assert_eq!(decode_move(&t, "12").unwrap(), 12);
        assert_eq!(encode_move(&t, 12), "12");
        assert!(decode_move(&t, "18").is_err(), "out of range");
        assert!(decode_move(&t, "dd").is_err());
    }

    #[test]
    fn reads_a_grid_position() {
        let config = json!({"komi": 6.5, "board": {"type": "grid", "width": 3, "height": 3}});
        let state = json!({
            "state": {
                "board": [[0, 1, 0], [0, 2, 0], [0, 0, 0]],
                "captures": {"0": 1, "1": 2},
                "last_move": "bb",
            },
            "next_to_play": [1],
            "phase": "play",
            "round": 4,
        });
        let live = position_from_state(&config, &state).unwrap();
        assert_eq!(live.game.num_nodes(), 9);
        assert_eq!(live.game.stones()[1], Color::Black);
        assert_eq!(live.game.stones()[4], Color::White);
        assert_eq!(live.game.captures(), [1, 2]);
        assert_eq!(live.next_to_play, vec![1]);
        assert_eq!(live.phase, "play");
    }

    #[test]
    fn reads_a_flat_graph_position() {
        let config = json!({"komi": 0.5, "board": {"type": "circular", "rings": 2, "nodesPerRing": 3}});
        let state = json!({
            "state": {"board": [0, 1, 2, 0, 0, 0], "captures": {"0": 0, "1": 0}},
            "next_to_play": [0],
            "phase": "play",
        });
        let live = position_from_state(&config, &state).unwrap();
        assert_eq!(live.game.num_nodes(), 6);
        assert_eq!(live.game.stones()[1], Color::Black);
        assert_eq!(live.game.stones()[2], Color::White);
    }

    #[test]
    fn rejects_unsupported_boards_and_colors() {
        assert!(topology_from_config(&json!({"type": "sunflower", "size": 2})).is_err());
        let t = Topology::Grid {
            width: 2,
            height: 1,
        };
        // Multicolor variants use colors beyond black and white.
        assert!(stones_from_board(&t, &json!([0, 7])).is_err());
        // Wrong number of points.
        assert!(stones_from_board(&t, &json!([0])).is_err());
    }
}
