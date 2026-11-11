//! Monte Carlo Tree Search with PUCT selection and a pluggable evaluator.
//!
//! Values are always expressed from the point of view of the player to move at
//! the node holding them, and negated on the way back up the tree. Terminal
//! positions are scored by the parent, which still knows whose turn it was.

use crate::game::Game;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rng;
use rand_distr::Gamma;

/// Configuration for Dirichlet noise at the MCTS root node.
/// Noise is mixed into the prior: p = (1 - epsilon) * p + epsilon * Dir(alpha)
#[derive(Clone, Debug)]
pub struct NoiseConfig {
    /// Dirichlet alpha parameter. Typical: 0.03 for 19x19 Go, 0.3 for 9x9.
    pub alpha: f32,
    /// Fraction of noise to mix in. Typical: 0.25.
    pub epsilon: f32,
}

/// Configuration for MCTS search.
#[derive(Clone, Debug)]
pub struct MctsConfig {
    /// Number of simulations (tree traversals) per move.
    pub num_simulations: usize,
    /// Exploration constant for PUCT formula.
    pub cpuct: f32,
    /// Temperature for move selection. Higher = more random.
    /// 0 = always pick best move.
    pub temperature: f32,
    /// Optional Dirichlet noise for exploration during self-play.
    pub noise: Option<NoiseConfig>,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            num_simulations: 200,
            cpuct: 1.4,
            temperature: 1.0,
            noise: None,
        }
    }
}

/// Evaluator trait: provides policy prior and value estimate for a position.
///
/// `evaluate` is only called on non-terminal positions. The returned value is
/// from the perspective of the player to move, in [-1, 1]. Actions given a
/// prior of zero are pruned from the tree, which lets an evaluator veto moves
/// it considers pointless (filling one's own eyes, for instance).
pub trait Evaluator<G: Game> {
    /// Returns (policy, value) for the current position.
    fn evaluate(&self, game: &G) -> (Vec<f32>, f32);
}

/// Simple evaluator that uses random rollouts to estimate position value.
/// Policy is uniform over legal moves.
pub struct RandomRolloutEvaluator {
    pub num_rollouts: usize,
}

impl Default for RandomRolloutEvaluator {
    fn default() -> Self {
        RandomRolloutEvaluator { num_rollouts: 5 }
    }
}

impl<G: Game> Evaluator<G> for RandomRolloutEvaluator {
    fn evaluate(&self, game: &G) -> (Vec<f32>, f32) {
        let action_size = game.action_size();
        let legal = game.legal_moves();

        let mut policy = vec![0.0; action_size];
        if legal.is_empty() {
            return (policy, 0.0);
        }
        let prob = 1.0 / legal.len() as f32;
        for mv in &legal {
            policy[game.move_to_index(*mv)] = prob;
        }

        let players = game.next_to_play();
        if players.is_empty() {
            // Terminal: no player to move, so no meaningful perspective.
            // `simulate` scores terminal positions itself and never gets here.
            return (policy, 0.0);
        }

        let current_player = players[0];
        let mut total_value = 0.0;
        let mut rng = rng();

        for _ in 0..self.num_rollouts {
            let result = random_rollout(game, &mut rng);
            total_value += result.scores[current_player];
        }

        (policy, total_value / self.num_rollouts as f32)
    }
}

/// Play a game to completion with uniformly random legal moves.
fn random_rollout<G: Game>(game: &G, rng: &mut impl Rng) -> crate::game::GameResult {
    let mut state = game.clone();
    let mut moves_played = 0;
    let max_moves = game.num_nodes() * 3; // safety limit

    while !state.is_terminal() && moves_played < max_moves {
        let legal = state.legal_moves();
        if legal.is_empty() {
            break;
        }
        let players = state.next_to_play();
        if players.is_empty() {
            break;
        }
        let player = players[0];
        let mv = legal[rng.random_range(0..legal.len())];
        state.play_move(player, mv);
        moves_played += 1;
    }

    state.result().unwrap_or(crate::game::GameResult {
        scores: vec![0.0; game.num_players()],
    })
}

/// A node in the MCTS tree.
///
/// Children are stored as a compact `(action, node)` list rather than a slot
/// per possible action: a slot-per-action layout costs `action_size` pointers
/// per node, which on a 9x9 board means tens of kilobytes per expansion.
struct MctsNode {
    visits: u32,
    total_value: f32,
    prior: f32,
    children: Vec<(u32, MctsNode)>,
    expanded: bool,
}

impl MctsNode {
    fn new(prior: f32) -> Self {
        MctsNode {
            visits: 0,
            total_value: 0.0,
            prior,
            children: Vec::new(),
            expanded: false,
        }
    }

    #[inline]
    fn q_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / self.visits as f32
        }
    }
}

/// Sample from a Dirichlet distribution by sampling independent Gamma(alpha, 1) variates.
fn sample_dirichlet(alpha: f32, n: usize, rng: &mut impl Rng) -> Vec<f32> {
    let gamma = Gamma::new(alpha as f64, 1.0).unwrap();
    let samples: Vec<f64> = (0..n).map(|_| rng.sample(gamma)).collect();
    let sum: f64 = samples.iter().sum();
    if sum > 0.0 {
        samples.iter().map(|&s| (s / sum) as f32).collect()
    } else {
        vec![1.0 / n as f32; n]
    }
}

/// Apply Dirichlet noise to the prior probabilities of a node's children.
fn apply_dirichlet_noise(node: &mut MctsNode, alpha: f32, epsilon: f32) {
    if node.children.is_empty() {
        return;
    }
    let mut rng = rng();
    let noise = sample_dirichlet(alpha, node.children.len(), &mut rng);
    for (slot, (_, child)) in node.children.iter_mut().enumerate() {
        child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[slot];
    }
}

/// What a search found: the visit distribution plus the root's own evaluation.
pub struct SearchOutcome {
    /// Visit distribution over actions, after temperature.
    pub policy: Vec<f32>,
    /// Root value estimate for the player to move, in [-1, 1].
    pub value: f32,
    /// Most-visited action, if any.
    pub best_action: Option<usize>,
    /// Visit count per action, before temperature.
    pub visits: Vec<f32>,
}

/// Run MCTS search and return the visit-count distribution over actions.
pub fn search<G: Game, E: Evaluator<G> + ?Sized>(
    game: &G,
    evaluator: &E,
    config: &MctsConfig,
) -> Vec<f32> {
    search_detailed(game, evaluator, config).policy
}

/// Run MCTS search, returning the visit distribution and the root evaluation.
pub fn search_detailed<G: Game, E: Evaluator<G> + ?Sized>(
    game: &G,
    evaluator: &E,
    config: &MctsConfig,
) -> SearchOutcome {
    let action_size = game.action_size();

    if game.is_terminal() {
        return SearchOutcome {
            policy: vec![0.0; action_size],
            value: 0.0,
            best_action: None,
            visits: vec![0.0; action_size],
        };
    }

    let mut root = MctsNode::new(1.0);

    // Expand the root and seed its statistics, so the first simulation already
    // has a non-zero visit count to drive the exploration term.
    let (policy, root_value) = evaluator.evaluate(game);
    expand_node(&mut root, &policy, game);
    root.visits = 1;
    root.total_value = root_value;

    if let Some(noise) = &config.noise {
        apply_dirichlet_noise(&mut root, noise.alpha, noise.epsilon);
    }

    for _ in 0..config.num_simulations {
        let mut state = game.clone();
        simulate(&mut root, &mut state, evaluator, config);
    }

    let mut visits = vec![0.0f32; action_size];
    for (action, child) in root.children.iter() {
        visits[*action as usize] = child.visits as f32;
    }

    let best_action = root
        .children
        .iter()
        .max_by_key(|(_, child)| child.visits)
        .map(|(action, _)| *action as usize);

    let policy = apply_temperature(&visits, config.temperature);

    SearchOutcome {
        policy,
        value: root.q_value(),
        best_action,
        visits,
    }
}

fn apply_temperature(visits: &[f32], temperature: f32) -> Vec<f32> {
    if temperature < 1e-6 {
        let mut result = vec![0.0; visits.len()];
        if let Some((best, _)) = visits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            && visits[best] > 0.0
        {
            result[best] = 1.0;
        }
        return result;
    }

    let inv_temp = 1.0 / temperature;
    let powered: Vec<f32> = visits.iter().map(|&v| v.powf(inv_temp)).collect();
    let sum: f32 = powered.iter().sum();
    if sum > 0.0 {
        powered.iter().map(|&v| v / sum).collect()
    } else {
        visits.to_vec()
    }
}

/// Select a move from an MCTS policy distribution.
pub fn select_move<G: Game>(game: &G, policy: &[f32]) -> G::Move {
    let sum: f32 = policy.iter().sum();
    let mut rng = rng();
    if sum < 1e-8 {
        let legal = game.legal_moves();
        return legal[rng.random_range(0..legal.len())];
    }
    let dist = WeightedIndex::new(policy).expect("Invalid policy distribution");
    game.index_to_move(dist.sample(&mut rng))
}

/// Create children for every legal move with a non-zero prior.
fn expand_node<G: Game>(node: &mut MctsNode, policy: &[f32], game: &G) {
    let legal = game.legal_moves();
    node.children.clear();
    node.children.reserve(legal.len());

    for mv in &legal {
        let idx = game.move_to_index(*mv);
        let prior = policy.get(idx).copied().unwrap_or(0.0);
        if prior > 0.0 {
            node.children.push((idx as u32, MctsNode::new(prior)));
        }
    }

    // An evaluator that vetoed everything still needs somewhere to go.
    if node.children.is_empty() {
        for mv in &legal {
            let idx = game.move_to_index(*mv);
            node.children.push((idx as u32, MctsNode::new(1.0)));
        }
    }

    node.expanded = true;
}

/// Run one simulation from `node`, whose position is `game`.
///
/// Returns the value from the perspective of the player to move in `game`.
fn simulate<G: Game, E: Evaluator<G> + ?Sized>(
    node: &mut MctsNode,
    game: &mut G,
    evaluator: &E,
    config: &MctsConfig,
) -> f32 {
    debug_assert!(
        !game.is_terminal(),
        "simulate called on a terminal position"
    );

    let players = game.next_to_play();
    if players.is_empty() {
        return 0.0;
    }
    let player = players[0];

    if !node.expanded {
        let (policy, value) = evaluator.evaluate(game);
        expand_node(node, &policy, game);
        node.visits += 1;
        node.total_value += value;
        return value;
    }

    // PUCT selection. Unvisited children inherit the negated parent estimate
    // as their first-play urgency, rather than a flat zero: a flat zero makes
    // unexplored moves look attractive in losing positions and unattractive in
    // winning ones, and biases selection toward low action indices.
    let sqrt_total = (node.visits as f32).max(1.0).sqrt();
    // Every node stores its value from the perspective of the player to move
    // at that node, so a child's stored value belongs to the opponent and must
    // be negated to score it for the player choosing here. Without the
    // negation, search systematically picks the moves the opponent likes best.
    let parent_q = node.q_value();

    let mut best_slot = None;
    let mut best_score = f32::NEG_INFINITY;
    for (slot, (_, child)) in node.children.iter().enumerate() {
        let q = if child.visits == 0 {
            parent_q // first-play urgency: assume a new move is about as good
        } else {
            -child.q_value()
        };
        let u = config.cpuct * child.prior * sqrt_total / (1.0 + child.visits as f32);
        let score = q + u;
        if score > best_score {
            best_score = score;
            best_slot = Some(slot);
        }
    }

    let Some(slot) = best_slot else {
        node.visits += 1;
        return 0.0;
    };

    let action = node.children[slot].0 as usize;
    let mv = game.index_to_move(action);
    game.play_move(player, mv);

    let value = if game.is_terminal() {
        // Score it here, where we still know whose move it was.
        let result = game.result().expect("terminal position must have a result");
        let value_for_player = result.scores[player];
        let child = &mut node.children[slot].1;
        child.visits += 1;
        child.total_value += -value_for_player; // child's turn is the opponent's
        value_for_player
    } else {
        let child = &mut node.children[slot].1;
        -simulate(child, game, evaluator, config)
    };

    node.visits += 1;
    node.total_value += value;
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::baduk::{Baduk, BadukConfig, BadukMove};

    fn small_game() -> Baduk {
        Baduk::new(BadukConfig::grid(5, 5, 0.5))
    }

    #[test]
    fn test_mcts_returns_valid_policy() {
        let game = small_game();
        let evaluator = RandomRolloutEvaluator { num_rollouts: 2 };
        let config = MctsConfig {
            num_simulations: 50,
            ..Default::default()
        };
        let policy = search(&game, &evaluator, &config);

        assert_eq!(policy.len(), game.action_size());
        let sum: f32 = policy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "policy should sum to 1, got {}",
            sum
        );
        assert!(policy.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_mcts_only_suggests_legal_moves() {
        let mut game = small_game();
        game.play_move(0, BadukMove::Place(12));
        let evaluator = RandomRolloutEvaluator { num_rollouts: 2 };
        let config = MctsConfig {
            num_simulations: 50,
            ..Default::default()
        };
        let policy = search(&game, &evaluator, &config);

        let legal = game.legal_moves();
        for (idx, &p) in policy.iter().enumerate() {
            if p > 0.0 {
                let mv = game.index_to_move(idx);
                assert!(legal.contains(&mv), "{:?} is not legal", mv);
            }
        }
    }

    #[test]
    fn test_temperature_zero_is_deterministic() {
        let game = small_game();
        let evaluator = RandomRolloutEvaluator { num_rollouts: 2 };
        let config = MctsConfig {
            num_simulations: 50,
            temperature: 0.0,
            ..Default::default()
        };
        let policy = search(&game, &evaluator, &config);
        let nonzero = policy.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(nonzero, 1, "temperature 0 should pick exactly one move");
    }

    #[test]
    fn test_dirichlet_noise_runs() {
        let game = small_game();
        let evaluator = RandomRolloutEvaluator { num_rollouts: 2 };
        let config = MctsConfig {
            num_simulations: 30,
            noise: Some(NoiseConfig {
                alpha: 0.3,
                epsilon: 0.25,
            }),
            ..Default::default()
        };
        let policy = search(&game, &evaluator, &config);
        assert_eq!(policy.len(), game.action_size());
    }

    #[test]
    fn test_search_on_terminal_position_is_safe() {
        let mut game = small_game();
        game.play_move(0, BadukMove::Pass);
        game.play_move(1, BadukMove::Pass);
        assert!(game.is_terminal());
        let evaluator = RandomRolloutEvaluator { num_rollouts: 1 };
        let outcome = search_detailed(&game, &evaluator, &MctsConfig::default());
        assert!(outcome.best_action.is_none());
    }

    #[test]
    fn test_does_not_pass_when_passing_loses() {
        // A mutual-atari race on 3x3, komi 0.5. Black and white each have a
        // group of four whose single shared liberty is node 0, and node 0 is
        // the only empty point. White has just passed, so black has exactly
        // two options:
        //
        //   play 0  -> captures white's four stones and wins 9 to 0.5
        //   pass    -> ends the game, losing 4 to 4.5
        //
        //     .  W  W          . = node 0 (the shared liberty)
        //     B  B  W
        //     B  B  W
        //
        // This is the regression test for terminal-value perspective: with the
        // sign of a terminal score taken from the wrong player's point of
        // view, an immediate loss looks like an immediate win and search
        // happily passes.
        let mut game = Baduk::new(BadukConfig::grid(3, 3, 0.5));
        game.play_move(0, BadukMove::Pass); // black defers so white ends on a pass
        for (player, node) in [
            (1usize, 1usize),
            (0, 3),
            (1, 2),
            (0, 4),
            (1, 5),
            (0, 6),
            (1, 8),
            (0, 7),
        ] {
            game.play_move(player, BadukMove::Place(node));
        }
        game.play_move(1, BadukMove::Pass);

        // Premise checks: black to move, one empty point, white just passed.
        assert_eq!(game.next_to_play(), vec![0]);
        assert_eq!(game.consecutive_passes(), 1);
        let mut scratch = game.new_scratch();
        assert_eq!(
            game.liberties_at(3, &mut scratch),
            1,
            "black group in atari"
        );
        assert_eq!(
            game.liberties_at(1, &mut scratch),
            1,
            "white group in atari"
        );

        // Passing ends the game as a half-point loss.
        let mut passed = game.clone();
        passed.play_move(0, BadukMove::Pass);
        assert!(passed.is_terminal());
        assert_eq!(passed.result().unwrap().scores[0], -1.0, "passing loses");

        // Playing the shared liberty captures and wins.
        let mut captured = game.clone();
        captured.play_move(0, BadukMove::Place(0));
        assert_eq!(captured.captures()[0], 4, "black takes four stones");

        let evaluator = RandomRolloutEvaluator { num_rollouts: 4 };
        let config = MctsConfig {
            num_simulations: 200,
            temperature: 0.0,
            ..Default::default()
        };
        let outcome = search_detailed(&game, &evaluator, &config);
        assert_eq!(
            outcome.best_action,
            Some(0),
            "search must capture at node 0 rather than pass into a lost game"
        );
        assert!(
            outcome.value > 0.0,
            "black is winning here; root value should be positive, got {}",
            outcome.value
        );
    }

    #[test]
    fn test_value_tracks_who_is_winning() {
        // Black has stones everywhere and white has none, so the root value
        // must be positive for black and negative for white. A sign error in
        // the negamax backup shows up here immediately.
        let mut game = Baduk::new(BadukConfig::grid(5, 5, 0.5));
        for node in [0usize, 1, 2, 5, 6, 7, 10, 11, 12] {
            game.play_move(0, BadukMove::Place(node));
            game.play_move(1, BadukMove::Pass);
        }

        let evaluator = RandomRolloutEvaluator { num_rollouts: 16 };
        let config = MctsConfig {
            num_simulations: 150,
            temperature: 0.0,
            ..Default::default()
        };

        // Black to move, and winning.
        assert_eq!(game.next_to_play(), vec![0]);
        let black_view = search_detailed(&game, &evaluator, &config);
        assert!(
            black_view.value > 0.0,
            "black is winning; root value should be positive, got {}",
            black_view.value
        );

        // Hand the turn to white; the same position must evaluate negative.
        let mut white_turn = game.clone();
        white_turn.play_move(0, BadukMove::Place(13));
        assert_eq!(white_turn.next_to_play(), vec![1]);
        let white_view = search_detailed(&white_turn, &evaluator, &config);
        assert!(
            white_view.value < 0.0,
            "white is losing; root value should be negative, got {}",
            white_view.value
        );
    }
}
