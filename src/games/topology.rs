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

#[derive(Clone, PartialEq, Eq, Debug)]
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
    /// Concentric rings, as used by govariants' circular boards.
    ///
    /// Each ring is rotated half a step from the one inside it, so a node has
    /// two radial neighbors inward. Only the innermost and outermost rings
    /// carry intra-ring edges; the middle rings connect purely radially,
    /// which makes this a triangular lattice on an annulus rather than a
    /// stack of independent circles.
    Circular { rings: usize, nodes_per_ring: usize },
    /// A grid with cells removed and per-cell connectivity, matching
    /// govariants' `gridWithHoles` boards.
    ///
    /// Each bitmap cell is 0 for "no intersection", or a bitmask: bit 0 (1)
    /// connects right, bit 1 (2) connects down, and 4 marks an intersection
    /// that only receives connections from above or the left. Nodes are
    /// numbered row-major over the non-zero cells.
    GridWithHoles { bitmap: Vec<Vec<u8>> },
}

impl Topology {
    pub fn num_nodes(&self) -> usize {
        match self {
            Topology::Grid { width, height } | Topology::Torus { width, height } => width * height,
            Topology::Cube { size } => 6 * size * size,
            Topology::Circular {
                rings,
                nodes_per_ring,
            } => rings * nodes_per_ring,
            Topology::GridWithHoles { bitmap } => bitmap
                .iter()
                .flat_map(|row| row.iter())
                .filter(|&&bits| bits != 0)
                .count(),
        }
    }

    /// Short human-readable name, e.g. "9x9 grid" or "cube 3".
    pub fn describe(&self) -> String {
        match self {
            Topology::Grid { width, height } => format!("{}x{} grid", width, height),
            Topology::Torus { width, height } => format!("{}x{} torus", width, height),
            Topology::Cube { size } => format!("cube {} ({} per face)", size, size * size),
            Topology::Circular {
                rings,
                nodes_per_ring,
            } => format!("circular {} rings x {} per ring", rings, nodes_per_ring),
            Topology::GridWithHoles { bitmap } => format!(
                "{}x{} grid with holes ({} points)",
                bitmap.first().map(|r| r.len()).unwrap_or(0),
                bitmap.len(),
                self.num_nodes()
            ),
        }
    }

    /// Build the graph: CSR adjacency, directed edge list, and 3D coordinates.
    pub fn build(&self) -> Graph {
        match self {
            Topology::Grid { width, height } => Self::build_rect(*width, *height, false),
            Topology::Torus { width, height } => Self::build_rect(*width, *height, true),
            Topology::Cube { size } => Self::build_cube(*size),
            Topology::Circular {
                rings,
                nodes_per_ring,
            } => Self::build_circular(*rings, *nodes_per_ring),
            Topology::GridWithHoles { bitmap } => Self::build_grid_with_holes(bitmap),
        }
    }

    /// Mirrors govariants' `CreateCircularBoard`.
    ///
    /// Node `ring * nodes_per_ring + j`, each node joined to `j` and `j + 1`
    /// of the ring inside it, with intra-ring cycles only on the innermost
    /// and outermost rings.
    fn build_circular(rings: usize, nodes_per_ring: usize) -> Graph {
        let count = rings * nodes_per_ring;
        let mut neighbor_lists: Vec<Vec<usize>> = vec![Vec::with_capacity(4); count];
        let mut coords = vec![[0.0f32; 3]; count];
        if count == 0 {
            return Graph::from_lists(neighbor_lists, coords);
        }

        let index = |ring: usize, j: usize| ring * nodes_per_ring + j;

        // Positions follow govariants' polar layout, so coordinates stay
        // meaningful for display and for edge features later.
        let node_angle = std::f32::consts::TAU / nodes_per_ring as f32;
        for ring in 0..rings {
            for j in 0..nodes_per_ring {
                let angle = (j as f32 + ring as f32 * 0.5) * node_angle;
                let mut radius =
                    nodes_per_ring as f32 / 2.0 / std::f32::consts::PI + ring as f32 / 2.0;
                if ring == rings - 1 {
                    radius += 0.4;
                }
                if ring == 0 {
                    radius -= 0.2;
                }
                coords[index(ring, j)] = [radius * angle.sin(), radius * angle.cos(), 0.0];
            }
        }

        let mut connect = |a: usize, b: usize| {
            if a != b {
                neighbor_lists[a].push(b);
                neighbor_lists[b].push(a);
            }
        };

        // Radial edges: each node reaches j and j+1 of the ring inside it.
        for ring in 1..rings {
            for j in 0..nodes_per_ring {
                connect(index(ring, j), index(ring - 1, j));
                connect(index(ring, j), index(ring - 1, (j + 1) % nodes_per_ring));
            }
        }

        // Only the inner and outer rings close into cycles.
        for j in 1..nodes_per_ring {
            connect(index(0, j), index(0, j - 1));
            connect(index(rings - 1, j), index(rings - 1, j - 1));
        }
        connect(index(0, nodes_per_ring - 1), index(0, 0));
        connect(index(rings - 1, nodes_per_ring - 1), index(rings - 1, 0));

        Graph::from_lists(neighbor_lists, coords)
    }

    /// Node index of each non-zero bitmap cell, row-major, matching the order
    /// govariants flattens its intersection grid in.
    fn hole_indices(bitmap: &[Vec<u8>]) -> Vec<Vec<Option<usize>>> {
        let mut next = 0usize;
        bitmap
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&bits| {
                        if bits == 0 {
                            None
                        } else {
                            let id = next;
                            next += 1;
                            Some(id)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn build_grid_with_holes(bitmap: &[Vec<u8>]) -> Graph {
        let ids = Self::hole_indices(bitmap);
        let count = ids
            .iter()
            .flat_map(|row| row.iter())
            .filter(|slot| slot.is_some())
            .count();

        let mut neighbor_lists: Vec<Vec<usize>> = vec![Vec::with_capacity(4); count];
        let mut coords = vec![[0.0f32; 3]; count];

        for (y, row) in bitmap.iter().enumerate() {
            for (x, &bits) in row.iter().enumerate() {
                let Some(from) = ids[y][x] else { continue };
                coords[from] = [x as f32, y as f32, 0.0];

                // bit 0: connect right. bit 1: connect down. 4 means the cell
                // exists but adds no edges of its own.
                if bits & 1 != 0
                    && let Some(Some(to)) = ids[y].get(x + 1)
                {
                    neighbor_lists[from].push(*to);
                    neighbor_lists[*to].push(from);
                }
                if bits & 2 != 0
                    && let Some(Some(to)) = ids.get(y + 1).and_then(|r| r.get(x))
                {
                    neighbor_lists[from].push(*to);
                    neighbor_lists[*to].push(from);
                }
            }
        }

        Graph::from_lists(neighbor_lists, coords)
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
        match self {
            // govariants encodes moves on non-grid boards as the decimal node
            // index, so labels match the wire format and the adapter needs no
            // translation table.
            Topology::Circular { .. } | Topology::GridWithHoles { .. } => node.to_string(),
            Topology::Grid { width, height } | Topology::Torus { width, height } => {
                let x = node % *width;
                let y = node / *width;
                format!(
                    "{}{}",
                    COLUMNS[x % COLUMNS.len()] as char,
                    height - y // rows counted from the bottom, as in Go
                )
            }
            Topology::Cube { size } => {
                let size = *size;
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

        // Graph boards are addressed by node index.
        if matches!(
            self,
            Topology::Circular { .. } | Topology::GridWithHoles { .. }
        ) {
            let node: usize = cleaned.parse().ok()?;
            return if node < self.num_nodes() {
                Some(node)
            } else {
                None
            };
        }

        let (face, rest) = match self {
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
            Topology::Circular { .. } | Topology::GridWithHoles { .. } => {
                unreachable!("graph boards are parsed as node indices above")
            }
        }
    }

    /// Render the board as text. `cell` maps a node index to a display glyph.
    pub fn render(&self, cell: &dyn Fn(usize) -> char) -> String {
        match self {
            Topology::Circular {
                rings,
                nodes_per_ring,
            } => {
                // One line per ring, innermost first. Each ring is offset half
                // a step from the one inside it, which the indentation shows.
                let mut out = String::new();
                for ring in 0..*rings {
                    out.push_str(&format!("{:>3} ", ring));
                    if ring % 2 == 1 {
                        out.push(' ');
                    }
                    for j in 0..*nodes_per_ring {
                        out.push(cell(ring * nodes_per_ring + j));
                        out.push(' ');
                    }
                    out.push('\n');
                }
                out.push_str(
                    "    (ring index on the left; play by node number, 'nbrs <n>' to inspect)\n",
                );
                out
            }
            Topology::GridWithHoles { bitmap } => {
                let ids = Self::hole_indices(bitmap);
                let mut out = String::new();
                for (y, row) in ids.iter().enumerate() {
                    out.push_str(&format!("{:>3} ", y));
                    for slot in row {
                        match slot {
                            Some(node) => out.push(cell(*node)),
                            None => out.push(' '),
                        }
                        out.push(' ');
                    }
                    out.push('\n');
                }
                out.push_str("    (play by node number, 'nbrs <n>' to inspect)\n");
                out
            }
            _ => Self::render_regular(self, cell),
        }
    }

    fn render_regular(&self, cell: &dyn Fn(usize) -> char) -> String {
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
            Topology::Circular { .. } | Topology::GridWithHoles { .. } => {
                unreachable!("graph boards are rendered above")
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
        for (node, list) in lists.iter_mut().enumerate() {
            list.retain(|&nb| nb != node); // a node is never its own neighbor
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

    /// Symmetrically normalized adjacency with self-loops, row-major `[N * N]`.
    ///
    /// This is the GCN propagation matrix `D^-1/2 (A + I) D^-1/2`, and it is
    /// the adjacency the trained network takes as an *input*. It must match
    /// `normalized_adjacency` in `python/gvai/data.py` exactly: if the two
    /// drift, a trained net sees different inputs at play time than it saw
    /// during training and quietly plays worse, with nothing to indicate why.
    /// `tests/fixtures/adjacency.json` is generated by the Python side and
    /// asserted here.
    pub fn normalized_adjacency(&self) -> Vec<f32> {
        let n = self.num_nodes();
        let mut matrix = vec![0.0f32; n * n];

        for node in 0..n {
            matrix[node * n + node] = 1.0; // self-loop
            for &neighbor in self.neighbors(node) {
                let neighbor = neighbor as usize;
                matrix[node * n + neighbor] = 1.0;
                matrix[neighbor * n + node] = 1.0;
            }
        }

        let inv_sqrt_degree: Vec<f32> = (0..n)
            .map(|node| {
                let degree: f32 = matrix[node * n..(node + 1) * n].iter().sum();
                if degree > 0.0 {
                    1.0 / degree.sqrt()
                } else {
                    0.0
                }
            })
            .collect();

        for row in 0..n {
            for col in 0..n {
                matrix[row * n + col] *= inv_sqrt_degree[row] * inv_sqrt_degree[col];
            }
        }
        matrix
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
    fn grid_with_holes_respects_connectivity_bits() {
        // Real games cannot be relied on to cover this: a board may carry
        // dozens of adjacent-but-unconnected pairs and still never see two
        // same-coloured stones land across one, in which case replaying it
        // proves nothing about the bits. So pin the format directly.
        //
        //   bitmap        meaning
        //   3 1           (0,0) connects right and down
        //   2 4           (1,0) connects right only -> NOT down to (1,1)
        //                 (0,1) connects down only  -> NOT right to (1,1)
        //                 (1,1) adds no edges of its own
        let topology = Topology::GridWithHoles {
            bitmap: vec![vec![3, 1], vec![2, 4]],
        };
        let g = topology.build();

        // Node numbering is row-major over the non-zero cells.
        assert_eq!(g.num_nodes(), 4);
        let (a, b, c, d) = (0usize, 1usize, 2usize, 3usize);

        assert_eq!(
            g.neighbors(a),
            &[b as u32, c as u32],
            "(0,0) joins right and down"
        );
        assert_eq!(g.neighbors(b), &[a as u32]);
        assert_eq!(g.neighbors(c), &[a as u32]);

        // The whole point: b and d are adjacent cells, and so are c and d,
        // but no bit joins them. Treating the bitmap as "a grid with cells
        // deleted" would wrongly connect both.
        assert!(
            !g.neighbors(b).contains(&(d as u32)),
            "right-only cell must not connect downward"
        );
        assert!(
            !g.neighbors(c).contains(&(d as u32)),
            "down-only cell must not connect rightward"
        );
        assert!(
            g.neighbors(d).is_empty(),
            "a 4 cell with no incoming edges is isolated, got {:?}",
            g.neighbors(d)
        );
        assert_symmetric(&g);
    }

    #[test]
    fn grid_with_holes_value_four_receives_edges() {
        // 4 contributes no edges itself but is a normal intersection, so a
        // neighbor pointing at it still connects.
        let g = Topology::GridWithHoles {
            bitmap: vec![vec![1, 4]],
        }
        .build();
        assert_eq!(g.num_nodes(), 2);
        assert_eq!(g.neighbors(0), &[1u32]);
        assert_eq!(g.neighbors(1), &[0u32]);
    }

    #[test]
    fn grid_with_holes_skips_zero_cells() {
        let g = Topology::GridWithHoles {
            bitmap: vec![vec![3, 0, 3], vec![0, 0, 0]],
        }
        .build();
        // Two intersections, not adjacent (the gap between them is a hole).
        assert_eq!(g.num_nodes(), 2);
        assert!(g.neighbors(0).is_empty());
        assert!(g.neighbors(1).is_empty());
    }

    #[test]
    fn circular_middle_rings_connect_only_radially() {
        // govariants' circular board is a triangular lattice on an annulus:
        // each node reaches j and j+1 of the ring inside it, and only the
        // innermost and outermost rings close into cycles. Making every ring
        // a cycle is the obvious misreading and changes which stones form a
        // group, so it has to be pinned.
        let rings = 3;
        let per = 6;
        let topology = Topology::Circular {
            rings,
            nodes_per_ring: per,
        };
        let g = topology.build();
        assert_eq!(g.num_nodes(), rings * per);
        assert_symmetric(&g);

        let at = |ring: usize, j: usize| (ring * per + j) as u32;

        // Innermost ring is a cycle.
        for j in 0..per {
            let neighbors = g.neighbors(at(0, j) as usize);
            assert!(
                neighbors.contains(&at(0, (j + 1) % per)),
                "inner ring should close into a cycle at {}",
                j
            );
        }
        // Outermost ring is a cycle.
        for j in 0..per {
            let neighbors = g.neighbors(at(rings - 1, j) as usize);
            assert!(
                neighbors.contains(&at(rings - 1, (j + 1) % per)),
                "outer ring should close into a cycle at {}",
                j
            );
        }
        // The middle ring is not.
        for j in 0..per {
            let neighbors = g.neighbors(at(1, j) as usize);
            assert!(
                !neighbors.contains(&at(1, (j + 1) % per)),
                "middle ring node {} must not touch its ring neighbor",
                j
            );
            // It reaches j and j+1 inward, and j and j-1 outward.
            assert!(neighbors.contains(&at(0, j)));
            assert!(neighbors.contains(&at(0, (j + 1) % per)));
            assert!(neighbors.contains(&at(2, j)));
            assert!(neighbors.contains(&at(2, (j + per - 1) % per)));
            assert_eq!(neighbors.len(), 4);
        }
    }

    #[test]
    fn circular_is_four_regular_and_connected() {
        let g = Topology::Circular {
            rings: 4,
            nodes_per_ring: 9,
        }
        .build();
        assert_eq!(g.num_nodes(), 36);
        for i in 0..g.num_nodes() {
            assert_eq!(degree(&g, i), 4, "node {} has degree {}", i, degree(&g, i));
        }
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
        assert_eq!(count, g.num_nodes(), "circular board should be connected");
    }

    #[test]
    fn graph_boards_are_addressed_by_node_index() {
        // govariants encodes moves on non-grid boards as a decimal index, so
        // labels round-trip through plain numbers.
        for topology in [
            Topology::Circular {
                rings: 3,
                nodes_per_ring: 6,
            },
            Topology::GridWithHoles {
                bitmap: vec![vec![3, 3], vec![1, 4]],
            },
        ] {
            for node in 0..topology.num_nodes() {
                assert_eq!(topology.label(node), node.to_string());
                assert_eq!(topology.parse(&topology.label(node)), Some(node));
            }
            assert_eq!(topology.parse("999"), None, "out of range index");
            assert_eq!(topology.parse("abc"), None);
        }
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
