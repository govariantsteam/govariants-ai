//! Checks the Rust adjacency against the one the Python trainer builds.
//!
//! The network takes the normalized adjacency as an input, so the matrix built
//! here at play time must equal the matrix built in `python/gvai/data.py`
//! during training. A mismatch would not crash anything — the net would simply
//! see unfamiliar inputs and play worse, with no error to explain it. This is
//! the same failure mode as the TypeScript/Rust rules divergence, so it gets
//! the same treatment.
//!
//! Regenerate the fixture with: `cd python && python -m gvai.make_fixture`

use govariants_ai::games::topology::Topology;
use serde_json::Value;
use std::path::Path;

const TOLERANCE: f32 = 1e-5;

fn fixture() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/adjacency.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn check(name: &str, topology: &Topology) {
    let all = fixture();
    let entry = all
        .get(name)
        .unwrap_or_else(|| panic!("fixture has no entry '{}'", name));

    let expected_nodes = entry
        .get("num_nodes")
        .and_then(Value::as_u64)
        .expect("fixture entry has num_nodes") as usize;
    assert_eq!(
        topology.num_nodes(),
        expected_nodes,
        "{}: node count disagrees with the fixture",
        name
    );

    let expected: Vec<f32> = entry
        .get("adjacency")
        .and_then(Value::as_array)
        .expect("fixture entry has adjacency")
        .iter()
        .map(|v| v.as_f64().expect("adjacency entry is a number") as f32)
        .collect();

    let actual = topology.build().normalized_adjacency();
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: matrix size differs ({} vs {})",
        name,
        actual.len(),
        expected.len()
    );

    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > worst {
            worst = diff;
            worst_at = i;
        }
    }

    let n = topology.num_nodes();
    assert!(
        worst <= TOLERANCE,
        "{}: adjacency differs from the Python fixture by {} at ({}, {}): rust {} vs python {}",
        name,
        worst,
        worst_at / n,
        worst_at % n,
        actual[worst_at],
        expected[worst_at]
    );
}

#[test]
fn grid_adjacency_matches_python() {
    check(
        "grid5",
        &Topology::Grid {
            width: 5,
            height: 5,
        },
    );
}

#[test]
fn cube_adjacency_matches_python() {
    // The interesting case: cube adjacency is derived from 3D positions here
    // and read from the engine's own edge list on the Python side, so this
    // checks the whole path rather than one formula.
    check("cube3", &Topology::Cube { size: 3 });
}

#[test]
fn adjacency_is_symmetric_and_normalized() {
    for topology in [
        Topology::Grid {
            width: 9,
            height: 9,
        },
        Topology::Torus {
            width: 7,
            height: 7,
        },
        Topology::Cube { size: 3 },
        Topology::Circular {
            rings: 4,
            nodes_per_ring: 9,
        },
    ] {
        let n = topology.num_nodes();
        let matrix = topology.build().normalized_adjacency();
        for row in 0..n {
            assert!(
                matrix[row * n + row] > 0.0,
                "{:?}: missing self-loop at {}",
                topology,
                row
            );
            for col in 0..n {
                let diff = (matrix[row * n + col] - matrix[col * n + row]).abs();
                assert!(
                    diff < 1e-6,
                    "{:?}: not symmetric at ({}, {})",
                    topology,
                    row,
                    col
                );
            }
        }
    }
}
