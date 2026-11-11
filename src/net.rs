//! Playing with a trained network instead of rollouts.
//!
//! This is the other implementation of `Evaluator`: where `rollout.rs`
//! estimates a position by playing it out with hand-written heuristics, this
//! asks a graph network trained on self-play data, as AlphaZero does.
//!
//! Inference runs through `tract`, which is pure Rust. There is no native
//! onnxruntime to install, so `cargo test --features onnx` works anywhere and
//! CI needs no extra setup.
//!
//! The network takes the board's normalized adjacency as an **input**, so one
//! model file serves every topology — grids, tori, cube surfaces, circular
//! boards. That matrix has to match the one the trainer used; see
//! `Graph::normalized_adjacency` and `tests/adjacency_parity.rs`.

use crate::game::Game;
use crate::games::baduk::{Baduk, BadukMove, Color};
use crate::mcts::Evaluator;
use std::path::Path;
use tract_onnx::prelude::*;

type Plan = TypedSimplePlan<TypedModel>;

/// A trained graph network, specialized to one board size.
///
/// `tract` wants concrete input shapes to optimize, and a given game has a
/// fixed board, so the model is specialized at load time. Playing a different
/// board size means loading again.
pub struct NetEvaluator {
    plan: Plan,
    num_nodes: usize,
    num_features: usize,
    /// Row-major `[num_nodes * num_nodes]`, built once.
    adjacency: Vec<f32>,
}

impl NetEvaluator {
    /// Load `path` and specialize it to this game's board.
    pub fn load(path: impl AsRef<Path>, game: &Baduk) -> TractResult<Self> {
        let num_nodes = game.num_nodes();
        let num_features = game.node_features().first().map(Vec::len).unwrap_or(0);

        let plan = tract_onnx::onnx()
            .model_for_path(path.as_ref())?
            .with_input_fact(0, f32::fact([1, num_nodes, num_features]).into())?
            .with_input_fact(1, f32::fact([1, num_nodes, num_nodes]).into())?
            .with_input_fact(2, f32::fact([1, num_nodes]).into())?
            .into_optimized()?
            .into_runnable()?;

        Ok(NetEvaluator {
            plan,
            num_nodes,
            num_features,
            adjacency: game.graph().normalized_adjacency(),
        })
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Raw network outputs: per-node logits, the pass logit, and the value.
    fn forward(&self, game: &Baduk) -> TractResult<(Vec<f32>, f32, f32)> {
        let mut features = Vec::with_capacity(self.num_nodes * self.num_features);
        for row in game.node_features() {
            features.extend_from_slice(&row);
        }

        let x = tract_ndarray::Array3::from_shape_vec(
            (1, self.num_nodes, self.num_features),
            features,
        )?
        .into_tensor();
        let a = tract_ndarray::Array3::from_shape_vec(
            (1, self.num_nodes, self.num_nodes),
            self.adjacency.clone(),
        )?
        .into_tensor();
        // No padding at play time, so every node is real.
        let mask = tract_ndarray::Array2::from_shape_vec(
            (1, self.num_nodes),
            vec![1.0f32; self.num_nodes],
        )?
        .into_tensor();

        let outputs = self.plan.run(tvec!(x.into(), a.into(), mask.into()))?;

        let node_logits: Vec<f32> = outputs[0].to_array_view::<f32>()?.iter().copied().collect();
        let pass_logit = outputs[1]
            .to_array_view::<f32>()?
            .iter()
            .copied()
            .next()
            .unwrap_or(0.0);
        let value = outputs[2]
            .to_array_view::<f32>()?
            .iter()
            .copied()
            .next()
            .unwrap_or(0.0);
        Ok((node_logits, pass_logit, value))
    }
}

impl Evaluator<Baduk> for NetEvaluator {
    fn evaluate(&self, game: &Baduk) -> (Vec<f32>, f32) {
        let action_size = game.action_size();
        let mut policy = vec![0.0f32; action_size];

        let Ok((node_logits, pass_logit, value)) = self.forward(game) else {
            // A failed inference should not abort a game; fall back to a
            // uniform prior over legal moves and a neutral value.
            let legal = game.legal_moves();
            if !legal.is_empty() {
                let share = 1.0 / legal.len() as f32;
                for mv in &legal {
                    policy[game.move_to_index(*mv)] = share;
                }
            }
            return (policy, 0.0);
        };

        // Softmax over legal actions only. Illegal moves keep a zero prior,
        // which prunes them from the search tree.
        let mut scratch = game.new_scratch();
        let color = game.player_color();
        let mut legal: Vec<(usize, f32)> = Vec::with_capacity(self.num_nodes + 1);

        // `node` indexes the board, and is also needed for the stone lookup
        // and the probe, so iterating node_logits directly would not help.
        #[allow(clippy::needless_range_loop)]
        for node in 0..self.num_nodes {
            if game.stones()[node] != Color::Empty {
                continue;
            }
            if game.probe(node, color, &mut scratch).is_some() {
                legal.push((node, node_logits[node]));
            }
        }
        legal.push((game.move_to_index(BadukMove::Pass), pass_logit));

        let highest = legal
            .iter()
            .map(|(_, logit)| *logit)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0f32;
        for (action, logit) in &legal {
            let weight = (logit - highest).exp();
            policy[*action] = weight;
            total += weight;
        }
        if total > 0.0 {
            for value in policy.iter_mut() {
                *value /= total;
            }
        }

        (policy, value.clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::baduk::BadukConfig;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/gvai.onnx")
    }

    /// The model is generated by training, not committed, so skip when absent.
    fn load_or_skip(game: &Baduk) -> Option<NetEvaluator> {
        let path = model_path();
        if !path.exists() {
            eprintln!(
                "skipping: no model at {} (train one, then `python -m gvai.export`)",
                path.display()
            );
            return None;
        }
        Some(NetEvaluator::load(&path, game).expect("model loads"))
    }

    #[test]
    fn produces_a_distribution_over_legal_moves_only() {
        let game = Baduk::new(BadukConfig::grid(9, 9, 6.5));
        let Some(evaluator) = load_or_skip(&game) else {
            return;
        };

        let (policy, value) = evaluator.evaluate(&game);
        assert_eq!(policy.len(), game.action_size());

        let total: f32 = policy.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "policy should sum to 1, got {}",
            total
        );
        assert!(
            (-1.0..=1.0).contains(&value),
            "value {} out of range",
            value
        );

        let legal = game.legal_moves();
        for (action, &probability) in policy.iter().enumerate() {
            if probability > 0.0 {
                assert!(
                    legal.contains(&game.index_to_move(action)),
                    "gave prior to illegal action {}",
                    action
                );
            }
        }
    }

    #[test]
    fn illegal_moves_get_no_prior() {
        let mut game = Baduk::new(BadukConfig::grid(5, 5, 0.5));
        game.play_move(0, BadukMove::Place(12));
        let Some(evaluator) = load_or_skip(&game) else {
            return;
        };
        let (policy, _) = evaluator.evaluate(&game);
        assert_eq!(policy[12], 0.0, "occupied point must have zero prior");
    }

    #[test]
    fn runs_on_a_cube_board_with_the_same_model() {
        // The claim the whole architecture rests on: one model file, any
        // topology, because the adjacency is an input.
        let game = Baduk::new(BadukConfig::cube(3, 0.5));
        let Some(evaluator) = load_or_skip(&game) else {
            return;
        };
        assert_eq!(evaluator.num_nodes(), 54);
        let (policy, value) = evaluator.evaluate(&game);
        let total: f32 = policy.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "cube policy sums to {}", total);
        assert!((-1.0..=1.0).contains(&value));
    }
}
