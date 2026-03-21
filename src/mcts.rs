use crate::game::{Game, GameResult};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rng;

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
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            num_simulations: 200,
            cpuct: 1.4,
            temperature: 1.0,
        }
    }
}

/// Evaluator trait: provides policy prior and value estimate for a position.
/// For now we have a random rollout evaluator; later this will be a neural network.
pub trait Evaluator<G: Game> {
    /// Returns (policy, value) for the current position.
    /// - policy: probability distribution over actions (length = action_size)
    /// - value: estimated value for the current player (in [-1, 1])
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

        // Uniform policy over legal moves
        let mut policy = vec![0.0; action_size];
        let prob = 1.0 / legal.len() as f32;
        for mv in &legal {
            policy[game.move_to_index(*mv)] = prob;
        }

        // Estimate value via random rollouts
        let players = game.next_to_play();
        if players.is_empty() {
            // Already terminal
            let result = game.result().unwrap();
            return (policy, result.scores[0]); // from player 0's perspective
        }

        let current_player = players[0];
        let mut total_value = 0.0;
        let mut rng = rng();

        for _ in 0..self.num_rollouts {
            let value = random_rollout(game, &mut rng);
            total_value += value.scores[current_player];
        }

        (policy, total_value / self.num_rollouts as f32)
    }
}

/// Play a game to completion with random moves.
fn random_rollout<G: Game>(game: &G, rng: &mut impl Rng) -> GameResult {
    let mut state = game.clone();
    let mut moves_played = 0;
    let max_moves = game.num_nodes() * 3; // safety limit

    while !state.is_terminal() && moves_played < max_moves {
        let legal = state.legal_moves();
        if legal.is_empty() {
            break;
        }
        let players = state.next_to_play();
        let player = players[0];
        let mv = legal[rng.random_range(0..legal.len())];
        state.play_move(player, mv);
        moves_played += 1;
    }

    state.result().unwrap_or(GameResult {
        scores: vec![0.0; game.num_players()],
    })
}

/// A node in the MCTS tree.
struct MctsNode {
    /// Visit count.
    visits: u32,
    /// Total value accumulated.
    total_value: f32,
    /// Prior probability from the evaluator.
    prior: f32,
    /// Child nodes, indexed by action index.
    children: Vec<Option<MctsNode>>,
    /// Whether this node has been expanded.
    expanded: bool,
}

impl MctsNode {
    fn new(prior: f32, action_size: usize) -> Self {
        MctsNode {
            visits: 0,
            total_value: 0.0,
            prior,
            children: (0..action_size).map(|_| None).collect(),
            expanded: false,
        }
    }

    fn q_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / self.visits as f32
        }
    }
}

/// Run MCTS search and return a policy (visit count distribution over actions).
pub fn search<G: Game, E: Evaluator<G>>(
    game: &G,
    evaluator: &E,
    config: &MctsConfig,
) -> Vec<f32> {
    let action_size = game.action_size();
    let mut root = MctsNode::new(1.0, action_size);

    // Expand root
    let (policy, _) = evaluator.evaluate(game);
    expand_node(&mut root, &policy, game);

    for _ in 0..config.num_simulations {
        let mut state = game.clone();
        simulate(&mut root, &mut state, evaluator, config);
    }

    // Build visit count distribution
    let mut visits = vec![0.0f32; action_size];
    for (i, child) in root.children.iter().enumerate() {
        if let Some(child) = child {
            visits[i] = child.visits as f32;
        }
    }

    // Apply temperature
    if config.temperature < 1e-6 {
        // Pick best move deterministically
        let mut result = vec![0.0; action_size];
        if let Some(best) = visits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        {
            result[best.0] = 1.0;
        }
        result
    } else {
        let inv_temp = 1.0 / config.temperature;
        let powered: Vec<f32> = visits.iter().map(|&v| v.powf(inv_temp)).collect();
        let sum: f32 = powered.iter().sum();
        if sum > 0.0 {
            powered.iter().map(|&v| v / sum).collect()
        } else {
            visits
        }
    }
}

/// Select a move from the MCTS policy distribution.
pub fn select_move<G: Game>(game: &G, policy: &[f32]) -> G::Move {
    let sum: f32 = policy.iter().sum();
    if sum < 1e-8 {
        // Fallback: pick a random legal move
        let legal = game.legal_moves();
        let mut rng = rng();
        return legal[rng.random_range(0..legal.len())];
    }
    let mut rng = rng();
    let dist = WeightedIndex::new(policy).expect("Invalid policy distribution");
    let idx = dist.sample(&mut rng);
    game.index_to_move(idx)
}

fn expand_node<G: Game>(node: &mut MctsNode, policy: &[f32], game: &G) {
    let legal = game.legal_moves();
    let legal_indices: Vec<usize> = legal.iter().map(|mv| game.move_to_index(*mv)).collect();

    for &idx in &legal_indices {
        node.children[idx] = Some(MctsNode::new(policy[idx], game.action_size()));
    }

    node.expanded = true;
}

/// Run one simulation from the given node.
/// Returns the value from the perspective of the player at this node.
fn simulate<G: Game, E: Evaluator<G>>(
    node: &mut MctsNode,
    game: &mut G,
    evaluator: &E,
    config: &MctsConfig,
) -> f32 {
    if game.is_terminal() {
        let result = game.result().unwrap();
        // Return from perspective of whoever just played (previous player)
        return result.scores[0]; // simplified: player 0's perspective
    }

    if !node.expanded {
        let (policy, value) = evaluator.evaluate(game);
        expand_node(node, &policy, game);
        node.visits += 1;
        node.total_value += value;
        return value;
    }

    // Select child with highest PUCT score
    let sqrt_total = (node.visits as f32).sqrt();
    let mut best_idx = 0;
    let mut best_score = f32::NEG_INFINITY;

    for (idx, child) in node.children.iter().enumerate() {
        if let Some(child) = child {
            let q = child.q_value();
            let u = config.cpuct * child.prior * sqrt_total / (1.0 + child.visits as f32);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
    }

    // Play the move
    let players = game.next_to_play();
    let player = players[0];
    let mv = game.index_to_move(best_idx);
    game.play_move(player, mv);

    // Recurse
    let child = node.children[best_idx].as_mut().unwrap();
    let value = -simulate(child, game, evaluator, config);

    // Backpropagate
    node.visits += 1;
    node.total_value += value;
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::baduk::{Baduk, BadukConfig};

    #[test]
    fn test_mcts_returns_valid_policy() {
        let game = Baduk::new(BadukConfig {
            width: 5,
            height: 5,
            komi: 6.5,
        });
        let evaluator = RandomRolloutEvaluator { num_rollouts: 3 };
        let config = MctsConfig {
            num_simulations: 50,
            cpuct: 1.4,
            temperature: 1.0,
        };

        let policy = search(&game, &evaluator, &config);
        assert_eq!(policy.len(), game.action_size());

        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "Policy should sum to ~1.0, got {}", sum);
    }

    #[test]
    fn test_mcts_selects_legal_move() {
        let game = Baduk::new(BadukConfig {
            width: 5,
            height: 5,
            komi: 6.5,
        });
        let evaluator = RandomRolloutEvaluator { num_rollouts: 3 };
        let config = MctsConfig {
            num_simulations: 50,
            cpuct: 1.4,
            temperature: 1.0,
        };

        let policy = search(&game, &evaluator, &config);
        let mv = select_move(&game, &policy);

        let legal = game.legal_moves();
        assert!(legal.contains(&mv), "Selected move should be legal");
    }

    #[test]
    fn test_mcts_plays_full_game() {
        let mut game = Baduk::new(BadukConfig {
            width: 5,
            height: 5,
            komi: 6.5,
        });
        let evaluator = RandomRolloutEvaluator { num_rollouts: 2 };
        let config = MctsConfig {
            num_simulations: 30,
            temperature: 1.0,
            ..Default::default()
        };

        let mut moves = 0;
        while !game.is_terminal() && moves < 100 {
            let players = game.next_to_play();
            let player = players[0];
            let policy = search(&game, &evaluator, &config);
            let mv = select_move(&game, &policy);
            game.play_move(player, mv);
            moves += 1;
        }

        // Game should eventually end (via passes or move limit)
        // With random rollouts on 5x5, it should terminate
        println!("Game ended after {} moves. Result: {:?}", moves, game.result());
    }

    #[test]
    fn test_deterministic_temperature() {
        let game = Baduk::new(BadukConfig {
            width: 5,
            height: 5,
            komi: 6.5,
        });
        let evaluator = RandomRolloutEvaluator { num_rollouts: 3 };
        let config = MctsConfig {
            num_simulations: 50,
            cpuct: 1.4,
            temperature: 0.0, // deterministic
        };

        let policy = search(&game, &evaluator, &config);

        // With temperature 0, exactly one action should have probability 1.0
        let nonzero: Vec<f32> = policy.iter().filter(|&&p| p > 0.0).cloned().collect();
        assert_eq!(nonzero.len(), 1);
        assert!((nonzero[0] - 1.0).abs() < 1e-6);
    }
}
