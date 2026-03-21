use crate::game::{Game, GameResult};
use std::collections::{HashSet, VecDeque};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Color {
    Empty,
    Black,
    White,
}

impl Color {
    fn opponent(self) -> Color {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
            Color::Empty => Color::Empty,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BadukMove {
    Place(usize), // node index
    Pass,
}

#[derive(Clone, Debug)]
pub struct BadukConfig {
    pub width: usize,
    pub height: usize,
    pub komi: f32,
}

impl Default for BadukConfig {
    fn default() -> Self {
        BadukConfig {
            width: 9,
            height: 9,
            komi: 6.5,
        }
    }
}

#[derive(Clone)]
pub struct Baduk {
    config: BadukConfig,
    board: Vec<Color>,
    /// Adjacency list: for each node, the list of neighbor indices.
    adj: Vec<Vec<usize>>,
    /// Edge list for GNN (both directions).
    edges: Vec<[usize; 2]>,
    next_to_play: usize, // 0 = black, 1 = white
    captures: [usize; 2],
    consecutive_passes: usize,
    game_over: bool,
    result: Option<GameResult>,
    /// Positional superko: set of seen board+player hashes.
    seen_states: HashSet<u64>,
}

impl Baduk {
    fn player_color(&self) -> Color {
        if self.next_to_play == 0 {
            Color::Black
        } else {
            Color::White
        }
    }

    /// Build adjacency list for a rectangular grid.
    fn build_grid_adjacency(width: usize, height: usize) -> (Vec<Vec<usize>>, Vec<[usize; 2]>) {
        let n = width * height;
        let mut adj = vec![vec![]; n];
        let mut edges = Vec::new();

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if x + 1 < width {
                    let right = y * width + (x + 1);
                    adj[idx].push(right);
                    adj[right].push(idx);
                    edges.push([idx, right]);
                    edges.push([right, idx]);
                }
                if y + 1 < height {
                    let down = (y + 1) * width + x;
                    adj[idx].push(down);
                    adj[down].push(idx);
                    edges.push([idx, down]);
                    edges.push([down, idx]);
                }
            }
        }

        (adj, edges)
    }

    /// Get the group (connected stones of same color) containing `start`.
    fn get_group(&self, start: usize) -> Vec<usize> {
        let color = self.board[start];
        if color == Color::Empty {
            return vec![start];
        }

        let mut group = Vec::new();
        let mut visited = vec![false; self.board.len()];
        let mut queue = VecDeque::new();

        visited[start] = true;
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            group.push(node);
            for &neighbor in &self.adj[node] {
                if !visited[neighbor] && self.board[neighbor] == color {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        group
    }

    /// Check if a group has at least one liberty.
    fn group_has_liberties(&self, group: &[usize]) -> bool {
        for &node in group {
            for &neighbor in &self.adj[node] {
                if self.board[neighbor] == Color::Empty {
                    return true;
                }
            }
        }
        false
    }

    /// Simple hash of board state + next player for superko detection.
    fn state_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.board.hash(&mut hasher);
        self.next_to_play.hash(&mut hasher);
        hasher.finish()
    }

    /// Try placing a stone. Returns true if legal, false otherwise.
    /// If legal, mutates the board. If illegal, board is unchanged.
    fn try_place(&mut self, node: usize) -> bool {
        if self.board[node] != Color::Empty {
            return false;
        }

        let color = self.player_color();
        let opponent = color.opponent();

        // Place the stone
        self.board[node] = color;

        // Capture opponent groups with no liberties
        let mut captured = Vec::new();
        for &neighbor in &self.adj[node].clone() {
            if self.board[neighbor] == opponent {
                let group = self.get_group(neighbor);
                if !self.group_has_liberties(&group) {
                    captured.extend(group);
                }
            }
        }

        for &cap in &captured {
            self.board[cap] = Color::Empty;
        }
        self.captures[self.next_to_play] += captured.len();

        // Check suicide (our group has no liberties after captures)
        let our_group = self.get_group(node);
        if !self.group_has_liberties(&our_group) {
            // Undo everything
            self.board[node] = Color::Empty;
            for &cap in &captured {
                self.board[cap] = opponent;
            }
            self.captures[self.next_to_play] -= captured.len();
            return false;
        }

        // Check superko
        let hash = self.state_hash();
        if self.seen_states.contains(&hash) {
            // Undo everything
            self.board[node] = Color::Empty;
            for &cap in &captured {
                self.board[cap] = opponent;
            }
            self.captures[self.next_to_play] -= captured.len();
            return false;
        }

        self.seen_states.insert(hash);
        true
    }

    /// Area scoring: count territory + stones for each color.
    fn score(&self) -> (f32, f32) {
        let mut counted = vec![false; self.board.len()];
        let mut black_score: f32 = 0.0;
        let mut white_score: f32 = 0.0;

        for i in 0..self.board.len() {
            if counted[i] {
                continue;
            }

            match self.board[i] {
                Color::Black => {
                    black_score += 1.0;
                    counted[i] = true;
                }
                Color::White => {
                    white_score += 1.0;
                    counted[i] = true;
                }
                Color::Empty => {
                    // Flood fill to find the empty region
                    let group = self.get_empty_region(i);
                    let mut touches_black = false;
                    let mut touches_white = false;

                    for &node in &group {
                        for &neighbor in &self.adj[node] {
                            match self.board[neighbor] {
                                Color::Black => touches_black = true,
                                Color::White => touches_white = true,
                                Color::Empty => {}
                            }
                        }
                    }

                    let points = group.len() as f32;
                    if touches_black && !touches_white {
                        black_score += points;
                    } else if touches_white && !touches_black {
                        white_score += points;
                    }
                    // Neutral territory (touches both or neither) scores for nobody

                    for &node in &group {
                        counted[node] = true;
                    }
                }
            }
        }

        white_score += self.config.komi;
        (black_score, white_score)
    }

    fn get_empty_region(&self, start: usize) -> Vec<usize> {
        let mut region = Vec::new();
        let mut visited = vec![false; self.board.len()];
        let mut queue = VecDeque::new();

        visited[start] = true;
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            region.push(node);
            for &neighbor in &self.adj[node] {
                if !visited[neighbor] && self.board[neighbor] == Color::Empty {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        region
    }
}

impl Game for Baduk {
    type Move = BadukMove;
    type Config = BadukConfig;

    fn new(config: Self::Config) -> Self {
        let n = config.width * config.height;
        let (adj, edges) = Self::build_grid_adjacency(config.width, config.height);

        let mut game = Baduk {
            config,
            board: vec![Color::Empty; n],
            adj,
            edges,
            next_to_play: 0,
            captures: [0, 0],
            consecutive_passes: 0,
            game_over: false,
            result: None,
            seen_states: HashSet::new(),
        };

        // Record initial state
        let hash = game.state_hash();
        game.seen_states.insert(hash);
        game
    }

    fn legal_moves(&self) -> Vec<Self::Move> {
        if self.game_over {
            return vec![];
        }

        let mut moves = Vec::new();

        // Try each board position
        for i in 0..self.board.len() {
            if self.board[i] != Color::Empty {
                continue;
            }

            // Quick check: try placing and see if it's legal
            let mut clone = self.clone();
            if clone.try_place(i) {
                moves.push(BadukMove::Place(i));
            }
        }

        // Pass is always legal
        moves.push(BadukMove::Pass);
        moves
    }

    fn play_move(&mut self, player: usize, mv: Self::Move) {
        assert_eq!(player, self.next_to_play, "Not this player's turn");
        assert!(!self.game_over, "Game is already over");

        match mv {
            BadukMove::Place(node) => {
                assert!(self.try_place(node), "Illegal move: Place({})", node);
                self.consecutive_passes = 0;
            }
            BadukMove::Pass => {
                self.consecutive_passes += 1;
            }
        }

        if self.consecutive_passes >= 2 {
            self.game_over = true;
            let (black, white) = self.score();
            let diff = black - white;
            self.result = Some(GameResult {
                scores: if diff > 0.0 {
                    vec![1.0, -1.0]
                } else if diff < 0.0 {
                    vec![-1.0, 1.0]
                } else {
                    vec![0.0, 0.0]
                },
            });
        } else {
            self.next_to_play = 1 - self.next_to_play;
        }
    }

    fn is_terminal(&self) -> bool {
        self.game_over
    }

    fn result(&self) -> Option<GameResult> {
        self.result.clone()
    }

    fn next_to_play(&self) -> Vec<usize> {
        if self.game_over {
            vec![]
        } else {
            vec![self.next_to_play]
        }
    }

    fn num_players(&self) -> usize {
        2
    }

    fn num_nodes(&self) -> usize {
        self.board.len()
    }

    fn adjacency(&self) -> Vec<[usize; 2]> {
        self.edges.clone()
    }

    fn node_features(&self) -> Vec<Vec<f32>> {
        let current_color = self.player_color();
        let opponent_color = current_color.opponent();

        self.board
            .iter()
            .enumerate()
            .map(|(i, &color)| {
                let is_current = if color == current_color { 1.0 } else { 0.0 };
                let is_opponent = if color == opponent_color { 1.0 } else { 0.0 };
                let is_empty = if color == Color::Empty { 1.0 } else { 0.0 };

                // Liberty count for the group this node belongs to
                let liberty_count = if color != Color::Empty {
                    let group = self.get_group(i);
                    let liberties: usize = self.adj[i]
                        .iter()
                        .chain(
                            group
                                .iter()
                                .flat_map(|&n| self.adj[n].iter()),
                        )
                        .collect::<HashSet<_>>()
                        .iter()
                        .filter(|&&n| self.board[*n] == Color::Empty)
                        .count();
                    (liberties as f32).min(8.0) / 8.0 // normalize
                } else {
                    0.0
                };

                vec![is_current, is_opponent, is_empty, liberty_count]
            })
            .collect()
    }

    fn action_size(&self) -> usize {
        // One action per board position + pass
        self.board.len() + 1
    }

    fn move_to_index(&self, mv: Self::Move) -> usize {
        match mv {
            BadukMove::Place(i) => i,
            BadukMove::Pass => self.board.len(),
        }
    }

    fn index_to_move(&self, idx: usize) -> Self::Move {
        if idx == self.board.len() {
            BadukMove::Pass
        } else {
            BadukMove::Place(idx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_9x9() -> Baduk {
        Baduk::new(BadukConfig::default())
    }

    #[test]
    fn test_new_game() {
        let game = new_9x9();
        assert_eq!(game.num_nodes(), 81);
        assert_eq!(game.next_to_play(), vec![0]);
        assert!(!game.is_terminal());
        assert_eq!(game.action_size(), 82); // 81 + pass
    }

    #[test]
    fn test_place_stone() {
        let mut game = new_9x9();
        game.play_move(0, BadukMove::Place(0)); // black at (0,0)
        assert_eq!(game.board[0], Color::Black);
        assert_eq!(game.next_to_play(), vec![1]);
    }

    #[test]
    fn test_simple_capture() {
        // Place white stone surrounded by black stones
        let mut game = Baduk::new(BadukConfig {
            width: 5,
            height: 5,
            komi: 0.5,
        });

        // White at center (2,2) = index 12
        // Black surrounding at (1,2)=11, (3,2)=13, (2,1)=7, (2,3)=17
        game.play_move(0, BadukMove::Place(11)); // B
        game.play_move(1, BadukMove::Place(12)); // W center
        game.play_move(0, BadukMove::Place(13)); // B
        game.play_move(1, BadukMove::Pass);       // W pass
        game.play_move(0, BadukMove::Place(7));  // B
        game.play_move(1, BadukMove::Pass);       // W pass
        game.play_move(0, BadukMove::Place(17)); // B captures!

        assert_eq!(game.board[12], Color::Empty); // white stone captured
        assert_eq!(game.captures[0], 1); // black captured 1 stone
    }

    #[test]
    fn test_two_passes_end_game() {
        let mut game = new_9x9();
        game.play_move(0, BadukMove::Pass);
        assert!(!game.is_terminal());
        game.play_move(1, BadukMove::Pass);
        assert!(game.is_terminal());
        assert!(game.result().is_some());
    }

    #[test]
    fn test_suicide_is_illegal() {
        // On a 3x3 board, surround a corner and try to play into it
        // Layout:
        //   . B .
        //   B . .
        //   . . .
        // White to play at (0,0) would be suicide
        let mut game = Baduk::new(BadukConfig {
            width: 3,
            height: 3,
            komi: 0.5,
        });

        game.play_move(0, BadukMove::Place(1)); // B at (1,0)
        game.play_move(1, BadukMove::Place(8)); // W at (2,2) - elsewhere
        game.play_move(0, BadukMove::Place(3)); // B at (0,1)
        // Now it's White's turn. Playing at 0 would be suicide.

        let legal = game.legal_moves();
        assert!(
            !legal.contains(&BadukMove::Place(0)),
            "Suicide move should not be legal"
        );
    }

    #[test]
    fn test_node_features_shape() {
        let game = new_9x9();
        let features = game.node_features();
        assert_eq!(features.len(), 81);
        assert_eq!(features[0].len(), 4); // 4 feature channels
    }

    #[test]
    fn test_adjacency() {
        let game = Baduk::new(BadukConfig {
            width: 3,
            height: 3,
            komi: 0.5,
        });
        let edges = game.adjacency();
        // 3x3 grid: 12 edges * 2 directions = 24
        assert_eq!(edges.len(), 24);
    }

    #[test]
    fn test_move_index_roundtrip() {
        let game = new_9x9();
        for i in 0..81 {
            let mv = BadukMove::Place(i);
            assert_eq!(game.index_to_move(game.move_to_index(mv)), mv);
        }
        assert_eq!(
            game.index_to_move(game.move_to_index(BadukMove::Pass)),
            BadukMove::Pass
        );
    }

    #[test]
    fn test_komi_decides_empty_board() {
        let mut game = Baduk::new(BadukConfig {
            width: 5,
            height: 5,
            komi: 6.5,
        });
        game.play_move(0, BadukMove::Pass);
        game.play_move(1, BadukMove::Pass);

        let result = game.result().unwrap();
        // White wins by komi on empty board
        assert_eq!(result.scores, vec![-1.0, 1.0]);
    }
}
