/// Core game trait that all variants implement.
/// Designed for AlphaZero-style training with GNN evaluation.
///
/// The graph-based interface (num_nodes, adjacency, node_features) allows
/// a single GNN architecture to work across all board topologies:
/// grids, cubes, circular boards, etc.

/// Result of a terminated game, from the perspective of each player.
#[derive(Debug, Clone)]
pub struct GameResult {
    /// Score per player, where positive = good. For two-player Go,
    /// this might be [1.0, -1.0] for a black win.
    pub scores: Vec<f32>,
}

pub trait Game: Clone + Send {
    type Move: Copy + Send + Eq + std::fmt::Debug;
    type Config: Clone;

    /// Create a new game from configuration.
    fn new(config: Self::Config) -> Self;

    /// All legal moves in the current position.
    fn legal_moves(&self) -> Vec<Self::Move>;

    /// Play a move. Panics if the move is illegal.
    fn play_move(&mut self, player: usize, mv: Self::Move);

    /// Whether the game is over.
    fn is_terminal(&self) -> bool;

    /// Game result. None if the game is still in progress.
    fn result(&self) -> Option<GameResult>;

    /// Which players must move next. Empty if game is over.
    fn next_to_play(&self) -> Vec<usize>;

    /// Number of players in this game.
    fn num_players(&self) -> usize;

    // -- Graph structure for GNN --

    /// Number of board intersections / nodes.
    fn num_nodes(&self) -> usize;

    /// Edge list representing adjacency. Each entry is [node_a, node_b].
    /// Edges are undirected; include both directions for the GNN.
    fn adjacency(&self) -> Vec<[usize; 2]>;

    /// Per-node feature vectors for GNN input.
    /// Outer vec: one entry per node. Inner vec: feature channels.
    /// Example features: [is_black, is_white, is_empty, liberty_count, ...]
    fn node_features(&self) -> Vec<Vec<f32>>;

    // -- Action space --

    /// Total number of possible actions (typically num_nodes + special moves like pass).
    fn action_size(&self) -> usize;

    /// Convert a move to an index in [0, action_size).
    fn move_to_index(&self, mv: Self::Move) -> usize;

    /// Convert an index back to a move.
    fn index_to_move(&self, idx: usize) -> Self::Move;
}
