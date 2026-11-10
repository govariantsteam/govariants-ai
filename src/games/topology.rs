//! Board topologies.
//!
//! Go rules only need an adjacency relation, so the same rule engine runs on
//! any graph. A `Topology` knows how to build its graph, label its nodes for
//! human input/display, and render itself as text.
//!
//! Adjacency for `Cube` is derived from 3D cell-center positions rather than
//! hand-stitched face edges: two cells are neighbors iff their centers are a
//! half-step apart across a cube edge, or one step apart within a face. That
//! removes the orientation bookkeeping that makes cube boards error-prone.

use std::sync::Arc;

/// Column letters, skipping "I" per Go convention.
const COLUMNS: &[u8] = b"ABCDEFGHJKLMNOPQRSTUVWXYZ";

/// Cube face names, in the order faces are indexed.
pub const CUBE_FACES: [&str; 6] = ["U", "L", "F", "R", "B", "D"];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Topology {
    /// Standard rectangular grid.
    Grid { width: usize, height: usize },
    /// Rectangular grid with wraparound on both axes. Every node has degree 4
    /// and there are no edges or corners, so it is a genuinely different board
    /// from a grid of the same size.
    Torus { width: usize, height: usize },
    /// Surface of a cube: 6 faces of `size` x `size` cells, stitched along the
    /// cube edges. Every cell has exactly 4 neighbors.
    Cube { size: usize },
}

impl Topology {
    pub fn num_nodes(&self) -> usize {
        match *self {
            Topology::Grid { width, height } | Topology::Torus { width, height } => width * height,
            Topology::Cube { size } => 6 * size * size,
        }
    }

    /// Short human-readable name, e.g. "9x9 grid" or "cube 3".
    pub fn describe(&self) -> String {
        match *self {
            Topology::Grid { width, height } => format!("{}x{} grid", width, height),
            Topology::Torus { width, height } => format!("{}x{} torus", width, height),
            Topology::Cube { size } => format!("cube {} ({} per face)", size, size * size),
        }
    }

    /// Build the graph: CSR adjacency, directed edge list, and 3D coordinates.
    pub fn build(&self) -> Graph {
        match *self {
            Topology::Grid { width, height } => Self::build_rect(width, height, false),
            Topology::Torus { width, height } => Self::build_rect(width, height, true),
            Topology::Cube { size } => Self::build_cube(size),
        }
    }

    fn build_rect(width: usize, height: usize, wrap: bool) -> Graph {
        let n = width * height;
        let mut neighbor_lists: Vec<Vec<usize>> = vec![Vec::with_capacity(4); n];
        let mut coords = vec![[0.0f32; 3]; n];

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                coords[idx] = [x as f32, y as f32, 0.0];

                // Right neighbor.
                if x + 1 < width {
                    neighbor_lists[idx].push(y * width + x + 1);
                } else if wrap && width > 2 {
                    neighbor_lists[idx].push(y * width);
                }
                // Left neighbor.
                if x > 0 {
                    neighbor_lists[idx].push(y * width + x - 1);
                } else if wrap && width > 2 {
                    neighbor_lists[idx].push(y * width + width - 1);
                }
                // Down neighbor.
                if y + 1 < height {
                    neighbor_lists[idx].push((y + 1) * width + x);
                } else if wrap && height > 2 {
                    neighbor_lists[idx].push(x);
                }
                // Up neighbor.
                if y > 0 {
                    neighbor_lists[idx].push((y - 1) * width + x);
                } else if wrap && height > 2 {
                    neighbor_lists[idx].push((height - 1) * width + x);
                }
            }
        }

        Graph::from_lists(neighbor_lists, coords)
    }

    fn build_cube(size: usize) -> Graph {
        let n = size as f32;
        let cells = 6 * size * size;
        let mut coords = vec![[0.0f32; 3]; cells];

        for face in 0..6 {
            for row in 0..size {
                for col in 0..size {
                    let idx = face * size * size + row * size + col;
                    let c = col as f32 + 0.5;
                    let r = row as f32 + 0.5;
                    let r_flip = n - r;
                    let c_flip = n - c;
                    coords[idx] = match face {
                        0 => [c, n, r],             // U (+Y)
                        1 => [0.0, r_flip, c],      // L (-X)
                        2 => [c, r_flip, n],        // F (+Z)
                        3 => [n, r_flip, c_flip],   // R (+X)
                        4 => [c_flip, r_flip, 0.0], // B (-Z)
                        _ => [c, 0.0, r_flip],      // D (-Y)
                    };
                }
            }
        }

        // Two cells are adjacent if they are one step apart on the same face,
        // or a half-step apart in two axes (sharing a cube edge).
        let face_of = |i: usize| i / (size * size);
        let mut neighbor_lists: Vec<Vec<usize>> = vec![Vec::with_capacity(4); cells];
        for a in 0..cells {
            for b in (a + 1)..cells {
                let d2: f32 = (0..3)
                    .map(|k| {
                        let d = coords[a][k] - coords[b][k];
                        d * d
                    })
                    .sum();
                let same_face = face_of(a) == face_of(b);
                let adjacent = if same_face {
                    (d2 - 1.0).abs() < 1e-4
                } else {
                    (d2 - 0.5).abs() < 1e-4
                };
                if adjacent {
                    neighbor_lists[a].push(b);
                    neighbor_lists[b].push(a);
                }
            }
        }

        Graph::from_lists(neighbor_lists, coords)
    }

    /// Human-readable label for a node, e.g. "D4" or "F c2".
    pub fn label(&self, node: usize) -> String {
        match *self {
            Topology::Grid { width, height } | Topology::Torus { width, height } => {
                let x = node % width;
                let y = node / width;
                format!(
                    "{}{}",
                    COLUMNS[x % COLUMNS.len()] as char,
                    height - y // rows counted from the bottom, as in Go
                )
            }
            Topology::Cube { size } => {
                let face = node / (size * size);
                let rem = node % (size * size);
                let row = rem / size;
                let col = rem % size;
                format!(
                    "{}{}{}",
                    CUBE_FACES[face],
                    COLUMNS[col % COLUMNS.len()] as char,
                    size - row
                )
            }
        }
    }

    /// Parse a node label. Accepts labels as produced by `label`, case
    /// insensitively, with optional spaces (e.g. "d4", "F c2", "Fc2").
    pub fn parse(&self, text: &str) -> Option<usize> {
        let cleaned: String = text
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .to_ascii_uppercase();
        if cleaned.is_empty() {
            return None;
        }

        let (face, rest) = match *self {
            Topology::Cube { .. } => {
                let (head, tail) = cleaned.split_at(1);
                let face = CUBE_FACES.iter().position(|f| *f == head)?;
                (face, tail.to_string())
            }
            _ => (0, cleaned),
        };

        let mut chars = rest.chars();
        let col_char = chars.next()?;
        let col = COLUMNS.iter().position(|c| *c as char == col_char)?;
        let row_text: String = chars.collect();
        let row_from_bottom: usize = row_text.parse().ok()?;

        match *self {
            Topology::Grid { width, height } | Topology::Torus { width, height } => {
                if col >= width || row_from_bottom == 0 || row_from_bottom > height {
                    return None;
                }
                let y = height - row_from_bottom;
                Some(y * width + col)
            }
            Topology::Cube { size } => {
                if col >= size || row_from_bottom == 0 || row_from_bottom > size {
                    return None;
                }
                let row = size - row_from_bottom;
                Some(face * size * size + row * size + col)
            }
        }
    }

    /// Render the board as text. `cell` maps a node index to a display glyph.
    pub fn render(&self, cell: &dyn Fn(usize) -> char) -> String {
        match *self {
            Topology::Grid { width, height } | Topology::Torus { width, height } => {
                let mut out = String::new();
                for y in 0..height {
                    out.push_str(&format!("{:>3} ", height - y));
                    for x in 0..width {
                        out.push(cell(y * width + x));
                        out.push(' ');
                    }
                    out.push('\n');
                }
                out.push_str("    ");
                for x in 0..width {
                    out.push(COLUMNS[x % COLUMNS.len()] as char);
                    out.push(' ');
                }
                out.push('\n');
                out
            }
            Topology::Cube { size } => {
                // Unfolded net:
                //       U
                //   L   F   R   B
                //       D
                let face_at = |face: usize, row: usize, col: usize| {
                    cell(face * size * size + row * size + col)
                };
                let pad = " ".repeat(2 * size + 2);
                let mut out = String::new();

                let header = |faces: &[usize]| {
                    let mut s = String::from("    ");
                    for (i, f) in faces.iter().enumerate() {
                        if i > 0 {
                            s.push_str("  ");
                        }
                        let name = CUBE_FACES[*f];
                        let width = 2 * size - 1;
                        let left = (width - name.len()) / 2;
                        s.push_str(&" ".repeat(left));
                        s.push_str(name);
                        s.push_str(&" ".repeat(width - name.len() - left));
                    }
                    s.push('\n');
                    s
                };

                // Top face.
                out.push_str(&format!("    {}", pad));
                out.push_str(header(&[0]).trim_start_matches("    "));
                for row in 0..size {
                    out.push_str(&format!("{:>3} {}", size - row, pad));
                    for col in 0..size {
                        out.push(face_at(0, row, col));
                        out.push(' ');
                    }
                    out.push('\n');
                }

                // Middle band: L F R B.
                out.push_str(&header(&[1, 2, 3, 4]));
                for row in 0..size {
                    out.push_str(&format!("{:>3} ", size - row));
                    for (i, face) in [1usize, 2, 3, 4].iter().enumerate() {
                        if i > 0 {
                            out.push_str("  ");
                        }
                        for col in 0..size {
                            out.push(face_at(*face, row, col));
                            if col + 1 < size {
                                out.push(' ');
                            }
                        }
                    }
                    out.push('\n');
                }

                // Bottom face.
                out.push_str(&format!("    {}", pad));
                out.push_str(header(&[5]).trim_start_matches("    "));
                for row in 0..size {
                    out.push_str(&format!("{:>3} {}", size - row, pad));
                    for col in 0..size {
                        out.push(face_at(5, row, col));
                        out.push(' ');
                    }
                    out.push('\n');
                }

                out.push_str("    ");
                out.push_str(&pad);
                for col in 0..size {
                    out.push(COLUMNS[col % COLUMNS.len()] as char);
                    out.push(' ');
                }
                out.push('\n');
                out
            }
        }
    }
}

/// A board graph in CSR form, plus coordinates for display and (later) edge
/// features.
#[derive(Clone, Debug)]
pub struct Graph {
    /// `offsets[i]..offsets[i + 1]` indexes `neighbors` for node `i`.
    offsets: Vec<u32>,
    neighbors: Vec<u32>,
    /// Directed edge list (both directions), for GNN input.
    edges: Vec<[usize; 2]>,
    /// 3D position of each node. Grids and tori are embedded in the z = 0
    /// plane; cube cells sit on the cube surface.
    coords: Vec<[f32; 3]>,
}

impl Graph {
    fn from_lists(mut lists: Vec<Vec<usize>>, coords: Vec<[f32; 3]>) -> Self {
        let n = lists.len();
        let mut offsets = Vec::with_capacity(n + 1);
        let mut neighbors = Vec::new();
        let mut edges = Vec::new();

        offsets.push(0);
        for list in lists.iter_mut() {
            list.sort_unstable();
            list.dedup();
            for &nb in list.iter() {
                neighbors.push(nb as u32);
            }
            offsets.push(neighbors.len() as u32);
        }

        for (node, list) in lists.iter().enumerate() {
            for &nb in list {
                edges.push([node, nb]);
            }
        }

        Graph {
            offsets,
            neighbors,
            edges,
            coords,
        }
    }

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Neighbors of `node`, as a slice (no allocation).
    #[inline]
    pub fn neighbors(&self, node: usize) -> &[u32] {
        let start = self.offsets[node] as usize;
        let end = self.offsets[node + 1] as usize;
        &self.neighbors[start..end]
    }

    #[inline]
    pub fn edges(&self) -> &[[usize; 2]] {
        &self.edges
    }

    #[inline]
    pub fn coords(&self) -> &[[f32; 3]] {
        &self.coords
    }

    /// Maximum degree, used to size scratch buffers.
    pub fn max_degree(&self) -> usize {
        (0..self.num_nodes())
            .map(|i| self.neighbors(i).len())
            .max()
            .unwrap_or(0)
    }
}

/// Build and cache a graph for a topology. Graphs are immutable and shared by
/// every clone of a game state, so building them once matters.
pub fn graph_for(topology: &Topology) -> Arc<Graph> {
    Arc::new(topology.build())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn degree(graph: &Graph, node: usize) -> usize {
        graph.neighbors(node).len()
    }

    fn assert_symmetric(graph: &Graph) {
        for a in 0..graph.num_nodes() {
            for &b in graph.neighbors(a) {
                assert!(
                    graph.neighbors(b as usize).contains(&(a as u32)),
                    "adjacency not symmetric: {} -> {}",
                    a,
                    b
                );
                assert_ne!(a, b as usize, "node {} is its own neighbor", a);
            }
        }
    }

    #[test]
    fn grid_degrees() {
        let t = Topology::Grid {
            width: 3,
            height: 3,
        };
        let g = t.build();
        assert_eq!(g.num_nodes(), 9);
        assert_eq!(degree(&g, 0), 2); // corner
        assert_eq!(degree(&g, 1), 3); // edge
        assert_eq!(degree(&g, 4), 4); // center
        assert_eq!(g.edges().len(), 24);
        assert_symmetric(&g);
    }

    #[test]
    fn torus_is_four_regular() {
        let t = Topology::Torus {
            width: 5,
            height: 5,
        };
        let g = t.build();
        assert_eq!(g.num_nodes(), 25);
        for i in 0..25 {
            assert_eq!(degree(&g, i), 4, "node {} should have degree 4", i);
        }
        assert_symmetric(&g);
    }

    #[test]
    fn cube_is_four_regular() {
        for size in 1..=4 {
            let g = Topology::Cube { size }.build();
            assert_eq!(g.num_nodes(), 6 * size * size);
            for i in 0..g.num_nodes() {
                assert_eq!(
                    degree(&g, i),
                    4,
                    "cube size {}: node {} has degree {}",
                    size,
                    i,
                    degree(&g, i)
                );
            }
            // 4-regular graph on 6n^2 nodes has 12n^2 undirected edges.
            assert_eq!(g.edges().len(), 2 * 12 * size * size);
            assert_symmetric(&g);
        }
    }

    #[test]
    fn cube_surface_is_connected() {
        let g = Topology::Cube { size: 3 }.build();
        let mut seen = vec![false; g.num_nodes()];
        let mut stack = vec![0usize];
        seen[0] = true;
        let mut count = 0;
        while let Some(node) = stack.pop() {
            count += 1;
            for &nb in g.neighbors(node) {
                if !seen[nb as usize] {
                    seen[nb as usize] = true;
                    stack.push(nb as usize);
                }
            }
        }
        assert_eq!(count, g.num_nodes(), "cube surface should be connected");
    }

    #[test]
    fn label_parse_roundtrip() {
        let topologies = [
            Topology::Grid {
                width: 9,
                height: 9,
            },
            Topology::Torus {
                width: 5,
                height: 5,
            },
            Topology::Cube { size: 3 },
        ];
        for t in topologies {
            for node in 0..t.num_nodes() {
                let label = t.label(node);
                assert_eq!(
                    t.parse(&label),
                    Some(node),
                    "{:?}: label {} did not round-trip",
                    t,
                    label
                );
            }
        }
    }

    #[test]
    fn parse_rejects_junk() {
        let t = Topology::Grid {
            width: 9,
            height: 9,
        };
        assert_eq!(t.parse(""), None);
        assert_eq!(t.parse("Z9"), None); // column out of range
        assert_eq!(t.parse("A0"), None); // row out of range
        assert_eq!(t.parse("A99"), None);
        assert_eq!(t.parse("hello"), None);
    }

    #[test]
    fn grid_labels_match_go_convention() {
        let t = Topology::Grid {
            width: 9,
            height: 9,
        };
        assert_eq!(t.label(0), "A9"); // top-left is A9
        assert_eq!(t.label(80), "J1"); // bottom-right, "I" skipped
    }

    #[test]
    fn small_torus_does_not_self_loop() {
        // Width 2 would wrap a node onto its own neighbor twice; wrapping is
        // suppressed below width 3.
        let g = Topology::Torus {
            width: 2,
            height: 5,
        }
        .build();
        assert_symmetric(&g);
        for i in 0..g.num_nodes() {
            assert!(degree(&g, i) <= 4);
        }
    }
}
