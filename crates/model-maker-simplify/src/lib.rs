//! A shape model's dense surface brought down to a triangle budget: meshoptimizer's edge collapses by quadric
//! error keep the shape where a grid clustering would merge the two sides of a thin part and step the outline.
//! The vertices kept are the dense surface's own, so the caller's colours and normals follow them. Only where the
//! edge collapses cannot reach the budget (a surface of many small pieces) does a grid clustering take over; the
//! result says which ran, and the command prints it.

use meshopt::{SimplifyOptions, VertexDataAdapter};

/// How far the shape may move, as a share of the mesh's size: enough that the budget, not the error, decides
/// where it stops.
const MOST_ERROR: f32 = 1.0;
const POSITION_STRIDE: usize = std::mem::size_of::<[f32; 3]>();
const MESH_MAGIC: &[u8; 4] = b"MMSH";
const INDICES_MAGIC: &[u8; 4] = b"MMIX";
const FORMAT_VERSION: u32 = 1;
const WORD: usize = 4;

/// A surface: positions and triangles (three corners each).
#[derive(Debug, Clone, PartialEq)]
pub struct Surface {
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl Surface {
    pub fn triangles(&self) -> usize {
        self.indices.len() / 3
    }
}

/// How the budget was reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Already within it: every triangle kept.
    AsItWas,
    /// Edge collapses by quadric error: the shape kept.
    EdgeCollapse,
    /// The edge collapses stopped above the budget, so meshoptimizer's sloppy simplifier (a grid clustering that
    /// ignores the topology and may step a thin part's outline) brought it down.
    GridClustering,
}

impl Method {
    pub fn words(self) -> &'static str {
        match self {
            Method::AsItWas => "kept as it was",
            Method::EdgeCollapse => "edge collapse",
            Method::GridClustering => {
                "grid clustering (the edge collapses stopped above the budget)"
            }
        }
    }
}

/// The kept triangles' corners (indices into `surface.positions`), at most `most` triangles, and how. Loose pieces
/// are kept (no pruning): the object checks list them. Where the topology will not give way (many small pieces), the
/// sloppy simplifier, which ignores it, reaches the budget, and the method says so.
pub fn simplified(surface: &Surface, most: usize) -> Result<(Vec<u32>, Method), String> {
    if surface.triangles() <= most {
        return Ok((surface.indices.clone(), Method::AsItWas));
    }
    let positions = VertexDataAdapter::new(
        meshopt::typed_to_bytes(&surface.positions),
        POSITION_STRIDE,
        0,
    )
    .map_err(|e| format!("the positions: {e:?}"))?;
    let mut kept = meshopt::simplify(
        &surface.indices,
        &positions,
        most * 3,
        MOST_ERROR,
        SimplifyOptions::None,
        None,
    );
    let mut method = Method::EdgeCollapse;
    if kept.len() > most * 3 {
        kept = meshopt::simplify_sloppy(&surface.indices, &positions, most * 3, MOST_ERROR, None);
        method = Method::GridClustering;
    }
    if kept.is_empty() {
        return Err(format!(
            "nothing was left of the surface ({})",
            method.words()
        ));
    }
    Ok((kept, method))
}

fn word(bytes: &[u8], at: usize) -> Result<[u8; 4], String> {
    bytes
        .get(at..at + WORD)
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| format!("the file ends at byte {at}"))
}

fn u32_at(bytes: &[u8], at: usize) -> Result<u32, String> {
    word(bytes, at).map(u32::from_le_bytes)
}

/// A surface from its `MMSH` file's bytes.
pub fn read_surface(bytes: &[u8]) -> Result<Surface, String> {
    if bytes.get(..4) != Some(MESH_MAGIC.as_slice()) {
        return Err("not an MMSH file".into());
    }
    let version = u32_at(bytes, 4)?;
    if version != FORMAT_VERSION {
        return Err(format!(
            "MMSH version {version}; this reads {FORMAT_VERSION}"
        ));
    }
    let vertices = u32_at(bytes, 8)? as usize;
    let triangles = u32_at(bytes, 12)? as usize;
    let positions_at = 16;
    let indices_at = positions_at + vertices * 3 * WORD;
    let end = indices_at + triangles * 3 * WORD;
    if bytes.len() != end {
        return Err(format!("{} bytes; the counts say {end}", bytes.len()));
    }
    let mut positions = Vec::with_capacity(vertices);
    for v in 0..vertices {
        let at = positions_at + v * 3 * WORD;
        let mut p = [0.0_f32; 3];
        for (k, x) in p.iter_mut().enumerate() {
            *x = f32::from_le_bytes(word(bytes, at + k * WORD)?);
        }
        if !p.iter().all(|x| x.is_finite()) {
            return Err(format!("vertex {v} is not a finite position"));
        }
        positions.push(p);
    }
    let indices = (0..triangles * 3)
        .map(|i| u32_at(bytes, indices_at + i * WORD))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(bad) = indices.iter().find(|&&i| i as usize >= vertices) {
        return Err(format!("a triangle names vertex {bad} of {vertices}"));
    }
    Ok(Surface { positions, indices })
}

/// A surface as an `MMSH` file's bytes.
pub fn surface_bytes(surface: &Surface) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + surface.positions.len() * 12 + surface.indices.len() * 4);
    out.extend_from_slice(MESH_MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(
        &u32::try_from(surface.positions.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    out.extend_from_slice(
        &u32::try_from(surface.triangles())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    for p in &surface.positions {
        for x in p {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    for i in &surface.indices {
        out.extend_from_slice(&i.to_le_bytes());
    }
    out
}

/// Kept corners as an `MMIX` file's bytes.
pub fn indices_bytes(indices: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + indices.len() * 4);
    out.extend_from_slice(INDICES_MAGIC);
    out.extend_from_slice(
        &u32::try_from(indices.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    for i in indices {
        out.extend_from_slice(&i.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sphere of many small triangles.
    #[allow(clippy::cast_precision_loss)] // small counts
    fn sphere(rings: u32) -> Surface {
        let mut surface = Surface {
            positions: vec![],
            indices: vec![],
        };
        for ring in 0..=rings {
            for step in 0..=rings * 2 {
                let (a, b) = (
                    std::f32::consts::PI * ring as f32 / rings as f32,
                    std::f32::consts::PI * step as f32 / rings as f32,
                );
                surface
                    .positions
                    .push([a.sin() * b.cos(), a.cos(), a.sin() * b.sin()]);
            }
        }
        let row = rings * 2 + 1;
        for ring in 0..rings {
            for step in 0..rings * 2 {
                let (a, b, c, d) = (
                    ring * row + step,
                    ring * row + step + 1,
                    (ring + 1) * row + step,
                    (ring + 1) * row + step + 1,
                );
                surface.indices.extend([a, c, b, b, c, d]);
            }
        }
        surface
    }

    #[test]
    fn a_dense_sphere_comes_to_its_budget_on_its_own_surface() {
        let dense = sphere(60);
        let (kept, method) = simplified(&dense, 500).unwrap();
        assert_eq!(method, Method::EdgeCollapse);
        let triangles = kept.len() / 3;
        assert!(triangles <= 500 && triangles > 250, "{triangles}");
        for &i in &kept {
            let p = dense.positions[i as usize];
            let radius = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((radius - 1.0).abs() < 1e-4, "{radius}");
        }
    }

    #[test]
    fn a_loose_piece_is_kept_for_the_checks_to_list() {
        let mut both = sphere(40);
        let small = sphere(8);
        let offset = u32::try_from(both.positions.len()).unwrap();
        both.positions.extend(
            small
                .positions
                .iter()
                .map(|p| [p[0] * 0.1 + 3.0, p[1] * 0.1, p[2] * 0.1]),
        );
        both.indices
            .extend(small.indices.iter().map(|i| i + offset));
        let (kept, _) = simplified(&both, 1500).unwrap();
        assert!(
            kept.iter().any(|&i| i >= offset),
            "the small sphere was pruned"
        );
    }

    /// Many small pieces, even a budget of one triangle: meshoptimizer's edge collapses reach it (it removes
    /// whole pieces), so the grid clustering is not run; and a surface brought to nothing names what did it.
    #[test]
    fn many_small_pieces_are_brought_down_by_edge_collapse_and_nothing_left_is_refused() {
        let mut pieces = Surface {
            positions: vec![],
            indices: vec![],
        };
        for piece in 0..2000_u16 {
            let (x, z) = (f32::from(piece % 50), f32::from(piece / 50));
            let first = u32::try_from(pieces.positions.len()).unwrap();
            pieces.positions.extend([
                [x, 0.0, z],
                [x + 0.3, 0.0, z],
                [x, 0.0, z + 0.3],
                [x, 0.3, z],
            ]);
            let [a, b, c, d] = [first, first + 1, first + 2, first + 3];
            pieces.indices.extend([a, c, b, a, b, d, a, d, c, b, c, d]);
        }
        let (kept, method) = simplified(&pieces, 20).unwrap();
        assert_eq!(method, Method::EdgeCollapse);
        assert!(kept.len() / 3 <= 20);
        let refused = simplified(&pieces, 1).unwrap_err();
        assert!(
            refused.starts_with("nothing was left of the surface ("),
            "{refused}"
        );
    }

    #[test]
    fn a_surface_under_its_budget_is_kept_as_it_is() {
        let dense = sphere(10);
        assert_eq!(
            simplified(&dense, 10_000).unwrap(),
            (dense.indices.clone(), Method::AsItWas)
        );
    }

    #[test]
    fn files_read_back_what_was_written_and_refuse_what_is_broken() {
        let surface = sphere(6);
        let bytes = surface_bytes(&surface);
        assert_eq!(read_surface(&bytes).unwrap(), surface);
        assert!(read_surface(&bytes[..bytes.len() - 1]).is_err());
        let mut wrong_vertex = bytes.clone();
        let last = wrong_vertex.len() - 4;
        wrong_vertex[last..].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(read_surface(&wrong_vertex)
            .unwrap_err()
            .contains("names vertex"));
        assert!(read_surface(b"XXXX").is_err());
        let indices = indices_bytes(&[1, 2, 3]);
        assert_eq!(&indices[..4], b"MMIX");
        assert_eq!(indices.len(), 8 + 12);
    }
}
