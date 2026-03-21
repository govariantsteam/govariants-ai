use crate::game::Game;
use crate::mcts::{self, Evaluator, MctsConfig};
use rayon::prelude::*;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Configuration for self-play data generation.
#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    /// Number of games to play.
    pub num_games: usize,
    /// MCTS configuration (includes noise settings).
    pub mcts_config: MctsConfig,
    /// Directory to write training data files.
    pub output_dir: String,
}

/// A single training example from one position in a game.
#[derive(Clone, Debug)]
pub struct TrainingExample {
    /// Per-node feature vectors: [num_nodes][num_features].
    pub node_features: Vec<Vec<f32>>,
    /// MCTS visit count policy: [action_size].
    pub policy: Vec<f32>,
    /// Index of the player to move at this position.
    pub current_player: usize,
    /// Game outcome from current player's perspective (filled after game ends).
    pub value: f32,
}

/// Result of a self-play run.
pub struct SelfPlayResult {
    pub games_completed: usize,
    pub total_examples: usize,
}

/// Run parallel self-play and write training data to disk.
pub fn run_self_play<G, E, F>(
    config: &SelfPlayConfig,
    make_game: F,
    evaluator: &E,
) -> SelfPlayResult
where
    G: Game,
    E: Evaluator<G> + Sync,
    F: Fn() -> G + Sync,
{
    fs::create_dir_all(&config.output_dir).expect("Failed to create output directory");

    let games_completed = AtomicUsize::new(0);
    let total_examples = AtomicUsize::new(0);

    (0..config.num_games)
        .into_par_iter()
        .for_each(|game_idx| {
            let game = make_game();
            let examples = play_one_game(&game, evaluator, &config.mcts_config);
            let num_examples = examples.len();

            if !examples.is_empty() {
                let path = Path::new(&config.output_dir).join(format!("game_{:06}.bin", game_idx));
                serialize_examples(&examples, &game, &path);
            }

            let completed = games_completed.fetch_add(1, Ordering::Relaxed) + 1;
            total_examples.fetch_add(num_examples, Ordering::Relaxed);

            if completed % 10 == 0 || completed == config.num_games {
                eprintln!(
                    "Self-play progress: {}/{} games",
                    completed, config.num_games
                );
            }
        });

    SelfPlayResult {
        games_completed: games_completed.load(Ordering::Relaxed),
        total_examples: total_examples.load(Ordering::Relaxed),
    }
}

/// Play a single game using MCTS and collect training examples.
fn play_one_game<G: Game, E: Evaluator<G>>(
    initial_state: &G,
    evaluator: &E,
    config: &MctsConfig,
) -> Vec<TrainingExample> {
    let mut game = initial_state.clone();
    let mut examples = Vec::new();

    while !game.is_terminal() {
        let policy = mcts::search(&game, evaluator, config);

        let players = game.next_to_play();
        let current_player = players[0];

        examples.push(TrainingExample {
            node_features: game.node_features(),
            policy: policy.clone(),
            current_player,
            value: 0.0, // filled after game ends
        });

        let mv = mcts::select_move(&game, &policy);
        game.play_move(current_player, mv);
    }

    // Fill in game results from each position's player perspective
    if let Some(result) = game.result() {
        for example in &mut examples {
            example.value = result.scores[example.current_player];
        }
    }

    examples
}

// -- Binary serialization --
//
// File format (all values little-endian):
//
//   Header:
//     magic:          [u8; 4]  = "GVAI"
//     version:        u32      = 1
//     num_examples:   u32
//     num_nodes:      u32
//     num_features:   u32
//     num_edge_pairs: u32      (number of directed edges)
//     action_size:    u32
//
//   Shared graph structure (same for all positions in one game):
//     edges:          [u32; num_edge_pairs * 2]  (flattened [src, dst] pairs)
//
//   Per example (repeated num_examples times):
//     node_features:  [f32; num_nodes * num_features]  (row-major)
//     policy:         [f32; action_size]
//     value:          f32

const MAGIC: &[u8; 4] = b"GVAI";
const FORMAT_VERSION: u32 = 1;

fn serialize_examples<G: Game>(examples: &[TrainingExample], game: &G, path: &Path) {
    let file = fs::File::create(path).expect("Failed to create output file");
    let mut w = BufWriter::new(file);

    let adjacency = game.adjacency();
    let num_nodes = game.num_nodes() as u32;
    let num_features = if examples.is_empty() {
        0u32
    } else {
        examples[0].node_features[0].len() as u32
    };
    let num_edge_pairs = adjacency.len() as u32;
    let action_size = game.action_size() as u32;

    // Header
    w.write_all(MAGIC).unwrap();
    w.write_all(&FORMAT_VERSION.to_le_bytes()).unwrap();
    w.write_all(&(examples.len() as u32).to_le_bytes()).unwrap();
    w.write_all(&num_nodes.to_le_bytes()).unwrap();
    w.write_all(&num_features.to_le_bytes()).unwrap();
    w.write_all(&num_edge_pairs.to_le_bytes()).unwrap();
    w.write_all(&action_size.to_le_bytes()).unwrap();

    // Shared edge list
    for edge in &adjacency {
        w.write_all(&(edge[0] as u32).to_le_bytes()).unwrap();
        w.write_all(&(edge[1] as u32).to_le_bytes()).unwrap();
    }

    // Per-example data
    for example in examples {
        // Node features (row-major: node0_feat0, node0_feat1, ..., node1_feat0, ...)
        for node_feats in &example.node_features {
            for &f in node_feats {
                w.write_all(&f.to_le_bytes()).unwrap();
            }
        }

        // Policy
        for &p in &example.policy {
            w.write_all(&p.to_le_bytes()).unwrap();
        }

        // Value
        w.write_all(&example.value.to_le_bytes()).unwrap();
    }

    w.flush().unwrap();
}

/// Deserialize training examples from a binary file (useful for testing).
pub fn deserialize_examples(path: &Path) -> (Vec<[u32; 2]>, Vec<TrainingExample>) {
    let data = fs::read(path).expect("Failed to read file");
    let mut pos = 0;

    let read_u32 = |data: &[u8], pos: &mut usize| -> u32 {
        let val = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        val
    };
    let read_f32 = |data: &[u8], pos: &mut usize| -> f32 {
        let val = f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        val
    };

    // Header
    assert_eq!(&data[pos..pos + 4], MAGIC);
    pos += 4;
    let version = read_u32(&data, &mut pos);
    assert_eq!(version, FORMAT_VERSION);
    let num_examples = read_u32(&data, &mut pos) as usize;
    let num_nodes = read_u32(&data, &mut pos) as usize;
    let num_features = read_u32(&data, &mut pos) as usize;
    let num_edge_pairs = read_u32(&data, &mut pos) as usize;
    let action_size = read_u32(&data, &mut pos) as usize;

    // Edges
    let mut edges = Vec::with_capacity(num_edge_pairs);
    for _ in 0..num_edge_pairs {
        let src = read_u32(&data, &mut pos);
        let dst = read_u32(&data, &mut pos);
        edges.push([src, dst]);
    }

    // Examples
    let mut examples = Vec::with_capacity(num_examples);
    for _ in 0..num_examples {
        let mut node_features = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let mut feats = Vec::with_capacity(num_features);
            for _ in 0..num_features {
                feats.push(read_f32(&data, &mut pos));
            }
            node_features.push(feats);
        }

        let mut policy = Vec::with_capacity(action_size);
        for _ in 0..action_size {
            policy.push(read_f32(&data, &mut pos));
        }

        let value = read_f32(&data, &mut pos);

        examples.push(TrainingExample {
            node_features,
            policy,
            current_player: 0, // not stored in file; not needed for training
            value,
        });
    }

    assert_eq!(pos, data.len(), "File size mismatch");
    (edges, examples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::baduk::{Baduk, BadukConfig};
    use crate::mcts::{MctsConfig, NoiseConfig, RandomRolloutEvaluator};
    fn small_game() -> Baduk {
        Baduk::new(BadukConfig {
            width: 5,
            height: 5,
            komi: 6.5,
        })
    }

    #[test]
    fn test_play_one_game_produces_examples() {
        let game = small_game();
        let evaluator = RandomRolloutEvaluator { num_rollouts: 2 };
        let config = MctsConfig {
            num_simulations: 20,
            temperature: 1.0,
            noise: Some(NoiseConfig {
                alpha: 0.3,
                epsilon: 0.25,
            }),
            ..Default::default()
        };

        let examples = play_one_game(&game, &evaluator, &config);

        assert!(!examples.is_empty(), "Game should produce training examples");

        for ex in &examples {
            assert_eq!(ex.node_features.len(), 25); // 5x5
            assert_eq!(ex.node_features[0].len(), 4); // 4 feature channels
            assert_eq!(ex.policy.len(), 26); // 25 + pass

            let policy_sum: f32 = ex.policy.iter().sum();
            // Policy should sum to ~1.0; can be 0 at near-terminal positions
            // where MCTS has no meaningful visits
            assert!(
                (policy_sum - 1.0).abs() < 1e-2 || policy_sum < 1e-6,
                "Policy should sum to ~1.0 or 0.0, got {}",
                policy_sum
            );

            // Value should be +1, -1, or 0
            assert!(
                ex.value == 1.0 || ex.value == -1.0 || ex.value == 0.0,
                "Value should be -1, 0, or 1, got {}",
                ex.value
            );
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let game = small_game();
        let evaluator = RandomRolloutEvaluator { num_rollouts: 2 };
        let config = MctsConfig {
            num_simulations: 20,
            temperature: 1.0,
            ..Default::default()
        };

        let examples = play_one_game(&game, &evaluator, &config);
        assert!(!examples.is_empty());

        let dir = std::env::temp_dir().join("govariants_test");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_roundtrip.bin");

        serialize_examples(&examples, &game, &path);
        let (edges, loaded) = deserialize_examples(&path);

        // Check edges match
        let expected_edges = game.adjacency();
        assert_eq!(edges.len(), expected_edges.len());
        for (e, expected) in edges.iter().zip(expected_edges.iter()) {
            assert_eq!(e[0], expected[0] as u32);
            assert_eq!(e[1], expected[1] as u32);
        }

        // Check examples match
        assert_eq!(loaded.len(), examples.len());
        for (loaded_ex, orig_ex) in loaded.iter().zip(examples.iter()) {
            assert_eq!(loaded_ex.node_features, orig_ex.node_features);
            for (lp, op) in loaded_ex.policy.iter().zip(orig_ex.policy.iter()) {
                assert!((lp - op).abs() < 1e-6);
            }
            assert!((loaded_ex.value - orig_ex.value).abs() < 1e-6);
        }

        // Cleanup
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_run_self_play_parallel() {
        let dir = std::env::temp_dir().join("govariants_selfplay_test");
        let _ = fs::remove_dir_all(&dir);

        let evaluator = RandomRolloutEvaluator { num_rollouts: 1 };
        let config = SelfPlayConfig {
            num_games: 4,
            mcts_config: MctsConfig {
                num_simulations: 10,
                temperature: 1.0,
                noise: Some(NoiseConfig {
                    alpha: 0.3,
                    epsilon: 0.25,
                }),
                ..Default::default()
            },
            output_dir: dir.to_string_lossy().to_string(),
        };

        let result = run_self_play(&config, small_game, &evaluator);

        assert_eq!(result.games_completed, 4);
        assert!(result.total_examples > 0);

        // Check files were created
        let files: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "bin"))
            .collect();
        assert_eq!(files.len(), 4);

        // Verify each file is readable
        for entry in &files {
            let (edges, examples) = deserialize_examples(&entry.path());
            assert!(!edges.is_empty());
            assert!(!examples.is_empty());
        }

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }
}
