use govariants_ai::game::Game;
use govariants_ai::games::baduk::{Baduk, BadukConfig, BadukMove};

fn main() {
    let mut game = Baduk::new(BadukConfig::default());
    println!("New 9x9 Baduk game");
    println!("Nodes: {}, Edges: {}", game.num_nodes(), game.adjacency().len());
    println!("Action size: {}", game.action_size());
    println!("Legal moves: {}", game.legal_moves().len());

    // Play a few moves
    game.play_move(0, BadukMove::Place(40)); // center
    game.play_move(1, BadukMove::Place(30));
    game.play_move(0, BadukMove::Place(50));

    println!("\nAfter 3 moves:");
    println!("Next to play: {:?}", game.next_to_play());
    println!("Legal moves: {}", game.legal_moves().len());

    let features = game.node_features();
    println!("Node features shape: {}x{}", features.len(), features[0].len());
}
