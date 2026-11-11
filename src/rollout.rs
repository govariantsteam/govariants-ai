//! A Go-specific evaluator: heuristic rollouts plus a shaped policy prior.
//!
//! Uniformly random rollouts are nearly worthless in Go, because random play
//! fills its own eyes and lets settled groups die. The result is a value
//! estimate uncorrelated with the position. Three cheap rules fix most of it,
//! and they are the classic ones from pre-AlphaGo MCTS engines:
//!
//!   1. never fill your own eye,
//!   2. never put your own group into atari for nothing,
//!   3. answer the opponent's last move when it created or escaped an atari.
//!
//! All three are expressed purely in terms of adjacency, so they work on any
//! topology — grids, tori, cube surfaces.

use crate::game::{Game, GameResult};
use crate::games::baduk::{Baduk, BadukMove, Color, Scratch};
use crate::mcts::Evaluator;
use rand::prelude::*;
use rand::rng;

#[derive(Clone, Debug)]
pub struct HeuristicRolloutEvaluator {
    /// Rollouts averaged per evaluation.
    pub num_rollouts: usize,
    /// Probability of answering a tactical threat instead of playing randomly.
    pub tactical_probability: f32,
    /// Rollout length cap, as a multiple of the number of board points.
    pub max_moves_factor: usize,
}

impl Default for HeuristicRolloutEvaluator {
    fn default() -> Self {
        HeuristicRolloutEvaluator {
            num_rollouts: 8,
            tactical_probability: 0.9,
            max_moves_factor: 2,
        }
    }
}

impl HeuristicRolloutEvaluator {
    /// Shaped prior over actions: zero for moves a sane player never makes,
    /// boosted for captures. Zero-prior actions are pruned by the tree.
    fn prior(&self, game: &Baduk, scratch: &mut Scratch) -> Vec<f32> {
        let n = game.num_nodes();
        let color = game.player_color();
        let mut policy = vec![0.0f32; n + 1];
        let mut sum = 0.0f32;

        // `policy` has one slot more than the board (the pass action), so the
        // index is the board node, not a position in `policy`.
        #[allow(clippy::needless_range_loop)]
        for node in 0..n {
            if game.stones()[node] != Color::Empty {
                continue;
            }
            let Some(probe) = game.probe(node, color, scratch) else {
                continue;
            };
            if game.is_eye_for(node, color, scratch) {
                continue; // filling our own eye
            }
            if probe.captured.is_empty() && probe.liberties <= 1 {
                continue; // self-atari for no compensation
            }

            let mut weight = 1.0;
            if !probe.captured.is_empty() {
                weight += 2.0 + probe.captured.len() as f32;
            }
            if probe.liberties >= 3 {
                weight += 0.25;
            }
            policy[node] = weight;
            sum += weight;
        }

        // Passing stays available, but is never the preferred option while
        // real moves exist.
        let pass_weight = if sum > 0.0 { 0.05 } else { 1.0 };
        policy[n] = pass_weight;
        sum += pass_weight;

        for p in policy.iter_mut() {
            *p /= sum;
        }
        policy
    }
}

impl Evaluator<Baduk> for HeuristicRolloutEvaluator {
    fn evaluate(&self, game: &Baduk) -> (Vec<f32>, f32) {
        let mut scratch = game.new_scratch();
        let policy = self.prior(game, &mut scratch);

        let players = game.next_to_play();
        if players.is_empty() {
            return (policy, 0.0);
        }
        let player = players[0];

        let max_moves = game.num_nodes() * self.max_moves_factor;
        let mut rng = rng();
        let mut total = 0.0f32;
        for _ in 0..self.num_rollouts {
            let result = playout(
                game,
                max_moves,
                self.tactical_probability,
                &mut scratch,
                &mut rng,
            );
            total += result.scores[player];
        }

        (policy, total / self.num_rollouts as f32)
    }
}

/// Play the position out with the heuristic policy and score the result.
pub fn playout(
    game: &Baduk,
    max_moves: usize,
    tactical_probability: f32,
    scratch: &mut Scratch,
    rng: &mut impl Rng,
) -> GameResult {
    let mut state = game.rollout_copy();
    let mut played = 0;

    while !state.is_terminal() && played < max_moves {
        let mv = choose_move(&state, tactical_probability, scratch, rng);
        if !state.try_play(mv, scratch) {
            // Should not happen; fall back to passing rather than spinning.
            if !state.try_play(BadukMove::Pass, scratch) {
                break;
            }
        }
        played += 1;
    }

    state.result().unwrap_or_else(|| state.result_from_score())
}

/// Pick a rollout move: answer a tactical threat if there is one, otherwise
/// play a random point that is neither an own eye nor a pointless self-atari.
pub fn choose_move(
    state: &Baduk,
    tactical_probability: f32,
    scratch: &mut Scratch,
    rng: &mut impl Rng,
) -> BadukMove {
    let color = state.player_color();

    if let Some(BadukMove::Place(last)) = state.last_move()
        && rng.random::<f32>() < tactical_probability
        && let Some(mv) = tactical_reply(state, last, color, scratch)
    {
        return mv;
    }

    let n = state.num_nodes();
    if n == 0 {
        return BadukMove::Pass;
    }
    let start = rng.random_range(0..n);
    for offset in 0..n {
        let node = (start + offset) % n;
        if state.stones()[node] != Color::Empty {
            continue;
        }
        let Some(probe) = state.probe(node, color, scratch) else {
            continue;
        };
        if probe.captured.is_empty() && probe.liberties <= 1 {
            continue;
        }
        if state.is_eye_for(node, color, scratch) {
            continue;
        }
        return BadukMove::Place(node);
    }

    BadukMove::Pass
}

/// Capture an adjacent enemy group in atari, or save one of ours.
fn tactical_reply(
    state: &Baduk,
    last: usize,
    color: Color,
    scratch: &mut Scratch,
) -> Option<BadukMove> {
    let opponent = color.opponent();
    let neighbors = state.graph().neighbors(last);

    // Capture first. The group to take may be the one the opponent just
    // played -- that is the common case, a stone played into atari -- or a
    // group next to it that the move left short of liberties.
    let capture_candidates = std::iter::once(last).chain(neighbors.iter().map(|&n| n as usize));
    for nb in capture_candidates {
        if state.stones()[nb] != opponent {
            continue;
        }
        if state.liberties_at(nb, scratch) != 1 {
            continue;
        }
        if let Some(liberty) = state.first_liberty(nb, scratch)
            && state.probe(liberty, color, scratch).is_some()
        {
            return Some(BadukMove::Place(liberty));
        }
    }

    // Otherwise, rescue one of our own groups that the last move put in atari.
    for &nb in neighbors.iter() {
        let nb = nb as usize;
        if state.stones()[nb] != color {
            continue;
        }
        if state.liberties_at(nb, scratch) != 1 {
            continue;
        }
        if let Some(liberty) = state.first_liberty(nb, scratch)
            && let Some(probe) = state.probe(liberty, color, scratch)
        {
            // Only worth it if it actually gains liberties or captures.
            if !probe.captured.is_empty() || probe.liberties >= 2 {
                return Some(BadukMove::Place(liberty));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::baduk::BadukConfig;

    #[test]
    fn rollout_never_fills_own_eye() {
        // Black has a one-point eye at node 0 surrounded by live stones.
        let mut game = Baduk::new(BadukConfig::grid(5, 5, 0.5));
        game.play_move(0, BadukMove::Place(1));
        game.play_move(1, BadukMove::Place(24));
        game.play_move(0, BadukMove::Place(5));
        game.play_move(1, BadukMove::Place(23));
        game.play_move(0, BadukMove::Place(6));
        game.play_move(1, BadukMove::Place(22));

        let mut scratch = game.new_scratch();
        let mut rng = rng();
        assert!(game.is_eye_for(0, Color::Black, &mut scratch));

        // Black to move: the chosen move must never be the eye itself.
        assert_eq!(game.next_to_play(), vec![0]);
        for _ in 0..200 {
            let mv = choose_move(&game, 0.9, &mut scratch, &mut rng);
            assert_ne!(mv, BadukMove::Place(0), "rollout policy filled its own eye");
        }
    }

    #[test]
    fn rollout_takes_a_capture() {
        // White's last move left its own stone in atari next to black stones,
        // so black (to move) should fill the last liberty and capture.
        // 5x5 board, index = y * 5 + x.
        let mut game = Baduk::new(BadukConfig::grid(5, 5, 0.5));
        game.play_move(0, BadukMove::Place(0)); // B (0,0)
        game.play_move(1, BadukMove::Place(20)); // W far away
        game.play_move(0, BadukMove::Place(2)); // B (2,0)
        game.play_move(1, BadukMove::Place(1)); // W (1,0): neighbors 0 B, 2 B, 6 empty

        let mut scratch = game.new_scratch();
        assert_eq!(
            game.liberties_at(1, &mut scratch),
            1,
            "white stone at 1 should be in atari with its liberty at 6"
        );
        assert_eq!(game.next_to_play(), vec![0], "black to move");

        // With the tactical rule always firing, the capture is forced.
        let mut rng = rng();
        for _ in 0..100 {
            let mv = choose_move(&game, 1.0, &mut scratch, &mut rng);
            assert_eq!(
                mv,
                BadukMove::Place(6),
                "tactical policy should capture at node 6"
            );
        }
    }

    #[test]
    fn heuristic_prior_prunes_eyes_and_self_atari() {
        let mut game = Baduk::new(BadukConfig::grid(5, 5, 0.5));
        game.play_move(0, BadukMove::Place(1));
        game.play_move(1, BadukMove::Place(24));
        game.play_move(0, BadukMove::Place(5));
        game.play_move(1, BadukMove::Place(23));

        let evaluator = HeuristicRolloutEvaluator::default();
        let mut scratch = game.new_scratch();
        assert_eq!(game.next_to_play(), vec![0]);
        let policy = evaluator.prior(&game, &mut scratch);
        assert_eq!(policy[0], 0.0, "own eye should get zero prior");
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn heuristic_beats_random_decisively() {
        // The strength claim, as a test: shaped priors plus heuristic rollouts
        // should beat uniformly random play nearly every game, even at a low
        // simulation count. Kept small so it stays a unit test.
        use crate::play::{Agent, play_match};

        let config = BadukConfig::grid(5, 5, 0.5);
        let result = play_match(
            &config,
            Agent::Heuristic {
                simulations: 80,
                rollouts: 2,
            },
            Agent::Random,
            8,
            0,
        );
        assert!(
            result.a_wins >= 7,
            "heuristic agent should dominate random, won {}/{}",
            result.a_wins,
            result.games()
        );
    }

    #[test]
    fn playout_terminates_on_cube() {
        let game = Baduk::new(BadukConfig::cube(2, 0.5));
        let mut scratch = game.new_scratch();
        let mut rng = rng();
        let result = playout(&game, game.num_nodes() * 2, 0.9, &mut scratch, &mut rng);
        assert_eq!(result.scores.len(), 2);
        assert!((result.scores[0] + result.scores[1]).abs() < 1e-6);
    }
}
