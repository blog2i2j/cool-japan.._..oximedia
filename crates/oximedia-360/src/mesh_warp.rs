//! Mesh-based projection for custom lens models via lookup-table warping.
//!
//! This module implements a flexible, geometry-driven image warp that is driven
//! by a *warp mesh* — a regular grid of (src_u, src_v) source coordinates, one
//! per output grid vertex.  The mesh can encode arbitrary lens distortions,
//! chromatic aberration corrections, or any other per-pixel remap.
//!
//! ## Overview
//!
//! A [`WarpMesh`] stores a 2-D grid of [`MeshVertex`] values.  Each vertex
//! gives the fractional (u, v) source coordinate that should appear at the
//! corresponding output pixel location.  During [`WarpMesh::apply`], bilinear
//! interpolation is used both to sample within a mesh cell (to find the source
//! UV for every output pixel) and to sample the source image at that UV.
//!
//! ## Building a Mesh
//!
//! A [`MeshBuilder`] provides a fluent interface:
//!
//! ```rust
//! use oximedia_360::mesh_warp::{MeshBuilder, WarpMesh};
//!
//! // Identity mesh: each output pixel maps directly to the same UV in the source.
//! let mesh = MeshBuilder::new(5, 5)
//!     .identity()
//!     .build()
//!     .expect("build failed");
//!
//! let src = vec![128u8; 64 * 64 * 3];
//! let dst = mesh.apply(&src, 64, 64, 64, 64).expect("warp failed");
//! assert_eq!(dst.len(), 64 * 64 * 3);
//! ```

use crate::VrError;

// ─── Mesh vertex ─────────────────────────────────────────────────────────────

/// A single vertex in a warp mesh.
///
/// `src_u` and `src_v` are fractional source-image coordinates in `[0.0, 1.0]`
/// that correspond to the output pixel whose grid position this vertex
/// represents.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshVertex {
    /// Fractional horizontal source coordinate (0 = left edge, 1 = right edge).
    pub src_u: f32,
    /// Fractional vertical source coordinate (0 = top edge, 1 = bottom edge).
    pub src_v: f32,
}

impl MeshVertex {
    /// Create a new vertex.
    pub fn new(src_u: f32, src_v: f32) -> Self {
        Self { src_u, src_v }
    }
}

// ─── Warp mesh ────────────────────────────────────────────────────────────────

/// A 2-D grid of [`MeshVertex`] values defining an image warp.
///
/// Grid dimensions are `cols × rows`; each vertex records the fractional
/// source (u, v) for that grid node.  The mesh is applied to an image via
/// bilinear cell interpolation: for each output pixel the enclosing mesh cell
/// is found and its four corner UVs are blended to derive the source lookup
/// coordinate.
#[derive(Debug, Clone)]
pub struct WarpMesh {
    /// Number of grid columns (horizontal).
    pub cols: u32,
    /// Number of grid rows (vertical).
    pub rows: u32,
    /// Vertices stored in row-major order: `vertices[row * cols + col]`.
    pub vertices: Vec<MeshVertex>,
}

impl WarpMesh {
    /// Create a new mesh from raw vertex data.
    ///
    /// # Errors
    /// Returns [`VrError::InvalidDimensions`] if `cols` or `rows` is zero.
    /// Returns [`VrError::BufferTooSmall`] if `vertices.len() < cols * rows`.
    pub fn new(cols: u32, rows: u32, vertices: Vec<MeshVertex>) -> Result<Self, VrError> {
        if cols == 0 || rows == 0 {
            return Err(VrError::InvalidDimensions(
                "WarpMesh: cols and rows must be > 0".into(),
            ));
        }
        let required = (cols * rows) as usize;
        if vertices.len() < required {
            return Err(VrError::BufferTooSmall {
                expected: required,
                got: vertices.len(),
            });
        }
        Ok(Self {
            cols,
            rows,
            vertices,
        })
    }

    /// Return the vertex at `(col, row)`.
    ///
    /// Returns `None` if the indices are out of bounds.
    pub fn vertex(&self, col: u32, row: u32) -> Option<MeshVertex> {
        if col >= self.cols || row >= self.rows {
            return None;
        }
        self.vertices.get((row * self.cols + col) as usize).copied()
    }

    /// Apply this warp mesh to an RGB source image, producing a warped output.
    ///
    /// For each output pixel the mesh cell that contains its normalised (u, v)
    /// coordinate is located; then bilinear interpolation over the four corner
    /// vertices gives the source (u, v); finally the source image is sampled at
    /// that location using bilinear interpolation.
    ///
    /// * `src`        — source pixel data (RGB, 3 bpp, row-major)
    /// * `src_w`      — source image width in pixels
    /// * `src_h`      — source image height in pixels
    /// * `out_w`      — output image width in pixels
    /// * `out_h`      — output image height in pixels
    ///
    /// # Errors
    /// Returns [`VrError::InvalidDimensions`] if any dimension is zero, or if
    /// the source buffer is too small.
    pub fn apply(
        &self,
        src: &[u8],
        src_w: u32,
        src_h: u32,
        out_w: u32,
        out_h: u32,
    ) -> Result<Vec<u8>, VrError> {
        if src_w == 0 || src_h == 0 || out_w == 0 || out_h == 0 {
            return Err(VrError::InvalidDimensions(
                "WarpMesh::apply: all dimensions must be > 0".into(),
            ));
        }
        let required = (src_w * src_h * 3) as usize;
        if src.len() < required {
            return Err(VrError::BufferTooSmall {
                expected: required,
                got: src.len(),
            });
        }

        let mut out = vec![0u8; (out_w * out_h * 3) as usize];

        for oy in 0..out_h {
            for ox in 0..out_w {
                // Normalised output coordinate (0..1)
                let out_u = (ox as f32 + 0.5) / out_w as f32;
                let out_v = (oy as f32 + 0.5) / out_h as f32;

                let (src_u, src_v) = self.lookup_source_uv(out_u, out_v);
                let sample = bilinear_sample(src, src_w, src_h, src_u, src_v);
                let dst = (oy * out_w + ox) as usize * 3;
                out[dst..dst + 3].copy_from_slice(&sample);
            }
        }

        Ok(out)
    }

    /// Apply the warp mesh in parallel using rayon.
    ///
    /// Equivalent to [`apply`][WarpMesh::apply] but processes scanlines in
    /// parallel for improved throughput on multi-core systems.
    ///
    /// # Errors
    /// Same conditions as [`apply`][WarpMesh::apply].
    pub fn apply_par(
        &self,
        src: &[u8],
        src_w: u32,
        src_h: u32,
        out_w: u32,
        out_h: u32,
    ) -> Result<Vec<u8>, VrError> {
        use rayon::prelude::*;

        if src_w == 0 || src_h == 0 || out_w == 0 || out_h == 0 {
            return Err(VrError::InvalidDimensions(
                "WarpMesh::apply_par: all dimensions must be > 0".into(),
            ));
        }
        let required = (src_w * src_h * 3) as usize;
        if src.len() < required {
            return Err(VrError::BufferTooSmall {
                expected: required,
                got: src.len(),
            });
        }

        let row_bytes = (out_w * 3) as usize;

        let rows: Vec<Vec<u8>> = (0..out_h)
            .into_par_iter()
            .map(|oy| {
                let mut row = vec![0u8; row_bytes];
                for ox in 0..out_w {
                    let out_u = (ox as f32 + 0.5) / out_w as f32;
                    let out_v = (oy as f32 + 0.5) / out_h as f32;
                    let (src_u, src_v) = self.lookup_source_uv(out_u, out_v);
                    let sample = bilinear_sample(src, src_w, src_h, src_u, src_v);
                    let dst = (ox * 3) as usize;
                    row[dst..dst + 3].copy_from_slice(&sample);
                }
                row
            })
            .collect();

        let mut out = vec![0u8; (out_w * out_h * 3) as usize];
        for (oy, row) in rows.into_iter().enumerate() {
            let dst_base = oy * row_bytes;
            out[dst_base..dst_base + row_bytes].copy_from_slice(&row);
        }

        Ok(out)
    }

    /// Look up the source (u, v) for a given normalised output coordinate.
    ///
    /// Finds the enclosing mesh cell and bilinearly interpolates the four
    /// corner source UVs.  Coordinates outside `[0, 1]` are clamped to the
    /// mesh boundary.
    pub fn lookup_source_uv(&self, out_u: f32, out_v: f32) -> (f32, f32) {
        // Clamp output coordinates to [0, 1]
        let out_u = out_u.clamp(0.0, 1.0);
        let out_v = out_v.clamp(0.0, 1.0);

        // There are `cols` vertices per row and `rows` of vertices, defining
        // (cols−1) × (rows−1) cells.
        let num_cols = self.cols as f32;
        let num_rows = self.rows as f32;

        // Position within the vertex grid (0..cols−1 for x, 0..rows−1 for y)
        let gx = out_u * (num_cols - 1.0);
        let gy = out_v * (num_rows - 1.0);

        let col0 = (gx.floor() as u32).min(self.cols.saturating_sub(2));
        let row0 = (gy.floor() as u32).min(self.rows.saturating_sub(2));
        let col1 = col0 + 1;
        let row1 = row0 + 1;

        let tx = gx - col0 as f32;
        let ty = gy - row0 as f32;

        // Safety: indices were clamped above
        let v00 = self.vertices[(row0 * self.cols + col0) as usize];
        let v10 = self.vertices[(row0 * self.cols + col1) as usize];
        let v01 = self.vertices[(row1 * self.cols + col0) as usize];
        let v11 = self.vertices[(row1 * self.cols + col1) as usize];

        let src_u = lerp(
            lerp(v00.src_u, v10.src_u, tx),
            lerp(v01.src_u, v11.src_u, tx),
            ty,
        );
        let src_v = lerp(
            lerp(v00.src_v, v10.src_v, tx),
            lerp(v01.src_v, v11.src_v, tx),
            ty,
        );

        (src_u.clamp(0.0, 1.0), src_v.clamp(0.0, 1.0))
    }

    /// Compute the warp residual (max deviation from identity) of this mesh.
    ///
    /// Returns the maximum Euclidean distance between any vertex's source UV
    /// and the identity UV that would be at that grid position.  An identity
    /// mesh returns `0.0`; a mesh with strong distortion returns a larger value.
    pub fn max_distortion(&self) -> f32 {
        let mut max_d = 0.0f32;
        for row in 0..self.rows {
            for col in 0..self.cols {
                let ident_u = col as f32 / (self.cols - 1).max(1) as f32;
                let ident_v = row as f32 / (self.rows - 1).max(1) as f32;
                let v = self.vertices[(row * self.cols + col) as usize];
                let du = v.src_u - ident_u;
                let dv = v.src_v - ident_v;
                let d = (du * du + dv * dv).sqrt();
                if d > max_d {
                    max_d = d;
                }
            }
        }
        max_d
    }
}

// ─── Mesh builder ─────────────────────────────────────────────────────────────

/// Fluent builder for [`WarpMesh`].
///
/// The builder provides preset mesh configurations (identity, barrel/pincushion
/// distortion, radial vignette, horizontal/vertical flip) and allows arbitrary
/// vertex-level overrides.
#[derive(Debug, Clone)]
pub struct MeshBuilder {
    cols: u32,
    rows: u32,
    vertices: Vec<MeshVertex>,
}

impl MeshBuilder {
    /// Create a new builder for a `cols × rows` mesh.
    ///
    /// All vertices default to the identity UV (pass-through warp).
    ///
    /// # Panics
    /// Panics if `cols` or `rows` is zero; use `build()` errors for
    /// production code.
    pub fn new(cols: u32, rows: u32) -> Self {
        assert!(cols > 0 && rows > 0, "cols and rows must be > 0");
        let mut vertices = Vec::with_capacity((cols * rows) as usize);
        for row in 0..rows {
            for col in 0..cols {
                let u = col as f32 / (cols - 1).max(1) as f32;
                let v = row as f32 / (rows - 1).max(1) as f32;
                vertices.push(MeshVertex::new(u, v));
            }
        }
        Self {
            cols,
            rows,
            vertices,
        }
    }

    /// Reset all vertices to the identity UV mapping (pass-through warp).
    #[must_use]
    pub fn identity(mut self) -> Self {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let u = col as f32 / (self.cols - 1).max(1) as f32;
                let v = row as f32 / (self.rows - 1).max(1) as f32;
                self.vertices[(row * self.cols + col) as usize] = MeshVertex::new(u, v);
            }
        }
        self
    }

    /// Apply barrel distortion to all vertices.
    ///
    /// The `k` parameter is the barrel coefficient: positive values cause
    /// barrel distortion (outward bulge); negative values cause pincushion
    /// distortion (inward pinch).  Typical values are in `[−0.5, 0.5]`.
    ///
    /// The distortion model used is: `r' = r × (1 + k × r²)`, applied in
    /// normalised `[−1, +1]` centred coordinates.
    #[must_use]
    pub fn barrel_distortion(mut self, k: f32) -> Self {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let u = col as f32 / (self.cols - 1).max(1) as f32;
                let v = row as f32 / (self.rows - 1).max(1) as f32;
                // Map to [−1, +1]
                let cx = u * 2.0 - 1.0;
                let cy = v * 2.0 - 1.0;
                let r2 = cx * cx + cy * cy;
                let scale = 1.0 + k * r2;
                // Map distorted [−1, +1] back to [0, 1]
                let src_u = ((cx * scale) + 1.0) * 0.5;
                let src_v = ((cy * scale) + 1.0) * 0.5;
                self.vertices[(row * self.cols + col) as usize] =
                    MeshVertex::new(src_u.clamp(0.0, 1.0), src_v.clamp(0.0, 1.0));
            }
        }
        self
    }

    /// Apply a horizontal flip (mirror left–right) to all vertices.
    #[must_use]
    pub fn flip_horizontal(mut self) -> Self {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let u = col as f32 / (self.cols - 1).max(1) as f32;
                let v = row as f32 / (self.rows - 1).max(1) as f32;
                self.vertices[(row * self.cols + col) as usize] = MeshVertex::new(1.0 - u, v);
            }
        }
        self
    }

    /// Apply a vertical flip (mirror top–bottom) to all vertices.
    #[must_use]
    pub fn flip_vertical(mut self) -> Self {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let u = col as f32 / (self.cols - 1).max(1) as f32;
                let v = row as f32 / (self.rows - 1).max(1) as f32;
                self.vertices[(row * self.cols + col) as usize] = MeshVertex::new(u, 1.0 - v);
            }
        }
        self
    }

    /// Apply a 180° rotation to all vertices (equivalent to combined H+V flip).
    #[must_use]
    pub fn rotate_180(mut self) -> Self {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let u = col as f32 / (self.cols - 1).max(1) as f32;
                let v = row as f32 / (self.rows - 1).max(1) as f32;
                self.vertices[(row * self.cols + col) as usize] = MeshVertex::new(1.0 - u, 1.0 - v);
            }
        }
        self
    }

    /// Apply a uniform scale and shift to all source UVs.
    ///
    /// The transformation applied to each vertex is:
    /// `src_u' = (src_u - cx) * scale_x + cx + shift_x`
    /// `src_v' = (src_v - cy) * scale_y + cy + shift_y`
    ///
    /// where `(cx, cy)` is the image centre (`0.5, 0.5`).
    /// Results are clamped to `[0, 1]`.
    #[must_use]
    pub fn scale_shift(mut self, scale_x: f32, scale_y: f32, shift_x: f32, shift_y: f32) -> Self {
        for v in &mut self.vertices {
            let src_u = (v.src_u - 0.5) * scale_x + 0.5 + shift_x;
            let src_v = (v.src_v - 0.5) * scale_y + 0.5 + shift_y;
            v.src_u = src_u.clamp(0.0, 1.0);
            v.src_v = src_v.clamp(0.0, 1.0);
        }
        self
    }

    /// Override a single vertex at `(col, row)` with a custom source UV.
    ///
    /// Does nothing if the indices are out of range.
    #[must_use]
    pub fn set_vertex(mut self, col: u32, row: u32, src_u: f32, src_v: f32) -> Self {
        if col < self.cols && row < self.rows {
            self.vertices[(row * self.cols + col) as usize] =
                MeshVertex::new(src_u.clamp(0.0, 1.0), src_v.clamp(0.0, 1.0));
        }
        self
    }

    /// Consume the builder and produce a [`WarpMesh`].
    ///
    /// # Errors
    /// Returns [`VrError::InvalidDimensions`] if `cols` or `rows` is zero (not
    /// reachable when constructed via [`MeshBuilder::new`], but included for
    /// completeness).
    pub fn build(self) -> Result<WarpMesh, VrError> {
        WarpMesh::new(self.cols, self.rows, self.vertices)
    }
}

// ─── Radial lens model helpers ────────────────────────────────────────────────

/// Parameters for a polynomial radial distortion model.
///
/// The model is `r' = r × (1 + k1×r² + k2×r⁴ + k3×r⁶)`, applied in
/// normalised `[−1, +1]` centred coordinates.  This corresponds to the
/// Brown-Conrady (OpenCV) radial distortion model.
///
/// Use [`radial_distortion_mesh`] to bake these parameters into a [`WarpMesh`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadialDistortionParams {
    /// First radial distortion coefficient (k1).
    pub k1: f32,
    /// Second radial distortion coefficient (k2).
    pub k2: f32,
    /// Third radial distortion coefficient (k3).
    pub k3: f32,
    /// Optical centre X as a fraction of image width (default: 0.5).
    pub cx: f32,
    /// Optical centre Y as a fraction of image height (default: 0.5).
    pub cy: f32,
}

impl RadialDistortionParams {
    /// Create parameters with k1, k2, k3 and centred optical axis.
    pub fn new(k1: f32, k2: f32, k3: f32) -> Self {
        Self {
            k1,
            k2,
            k3,
            cx: 0.5,
            cy: 0.5,
        }
    }

    /// Identity distortion (no warp).
    pub fn identity() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

impl Default for RadialDistortionParams {
    fn default() -> Self {
        Self::identity()
    }
}

/// Build a [`WarpMesh`] that corrects (undistorts) radial lens distortion.
///
/// The mesh encodes the *undistortion* mapping — each output vertex points to
/// the distorted source pixel that should appear there.  Use this to correct
/// barrel/pincushion distortion from real-world lens calibration data.
///
/// # Parameters
/// * `cols`, `rows` — mesh resolution (higher = more accurate correction,
///   suggested: 33 × 17 for typical use)
/// * `params` — radial distortion coefficients
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if `cols` or `rows` is zero.
pub fn radial_distortion_mesh(
    cols: u32,
    rows: u32,
    params: &RadialDistortionParams,
) -> Result<WarpMesh, VrError> {
    if cols == 0 || rows == 0 {
        return Err(VrError::InvalidDimensions(
            "radial_distortion_mesh: cols and rows must be > 0".into(),
        ));
    }

    let mut vertices = Vec::with_capacity((cols * rows) as usize);

    for row in 0..rows {
        for col in 0..cols {
            // Normalised [0, 1] output coordinates
            let u = col as f32 / (cols - 1).max(1) as f32;
            let v = row as f32 / (rows - 1).max(1) as f32;

            // Map to centred coordinates relative to optical axis
            let cx = u - params.cx;
            let cy = v - params.cy;
            let r2 = cx * cx + cy * cy;
            let r4 = r2 * r2;
            let r6 = r4 * r2;

            let radial = 1.0 + params.k1 * r2 + params.k2 * r4 + params.k3 * r6;
            let src_u = params.cx + cx * radial;
            let src_v = params.cy + cy * radial;

            vertices.push(MeshVertex::new(
                src_u.clamp(0.0, 1.0),
                src_v.clamp(0.0, 1.0),
            ));
        }
    }

    WarpMesh::new(cols, rows, vertices)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Linear interpolation between `a` and `b` at `t ∈ [0, 1]`.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Bilinear sample an RGB image buffer with edge clamping.
fn bilinear_sample(data: &[u8], w: u32, h: u32, u: f32, v: f32) -> [u8; 3] {
    let fw = w as f32;
    let fh = h as f32;

    let px = (u * fw - 0.5).max(0.0);
    let py = (v * fh - 0.5).max(0.0);

    let x0 = (px.floor() as u32).min(w.saturating_sub(1));
    let y0 = (py.floor() as u32).min(h.saturating_sub(1));
    let x1 = (x0 + 1).min(w.saturating_sub(1));
    let y1 = (y0 + 1).min(h.saturating_sub(1));

    let tx = px - px.floor();
    let ty = py - py.floor();

    let stride = w as usize * 3;
    let b00 = y0 as usize * stride + x0 as usize * 3;
    let b10 = y0 as usize * stride + x1 as usize * 3;
    let b01 = y1 as usize * stride + x0 as usize * 3;
    let b11 = y1 as usize * stride + x1 as usize * 3;

    let mut out = [0u8; 3];
    for c in 0..3 {
        let p00 = data[b00 + c] as f32;
        let p10 = data[b10 + c] as f32;
        let p01 = data[b01 + c] as f32;
        let p11 = data[b11 + c] as f32;
        let top = p00 + (p10 - p00) * tx;
        let bottom = p01 + (p11 - p01) * tx;
        out[c] = (top + (bottom - top) * ty).round().clamp(0.0, 255.0) as u8;
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let n = (w * h * 3) as usize;
        let mut v = vec![0u8; n];
        for i in (0..n).step_by(3) {
            v[i] = r;
            v[i + 1] = g;
            v[i + 2] = b;
        }
        v
    }

    // ── MeshVertex ───────────────────────────────────────────────────────────

    #[test]
    fn mesh_vertex_new_stores_values() {
        let v = MeshVertex::new(0.25, 0.75);
        assert!((v.src_u - 0.25).abs() < 1e-6);
        assert!((v.src_v - 0.75).abs() < 1e-6);
    }

    // ── WarpMesh::new errors ─────────────────────────────────────────────────

    #[test]
    fn warp_mesh_zero_cols_returns_error() {
        let result = WarpMesh::new(0, 3, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn warp_mesh_zero_rows_returns_error() {
        let result = WarpMesh::new(3, 0, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn warp_mesh_too_few_vertices_returns_error() {
        let result = WarpMesh::new(3, 3, vec![MeshVertex::new(0.0, 0.0)]);
        assert!(result.is_err());
    }

    // ── Identity mesh ────────────────────────────────────────────────────────

    #[test]
    fn identity_mesh_apply_preserves_solid_colour() {
        let src = solid_rgb(16, 16, 200, 100, 50);
        let mesh = MeshBuilder::new(5, 5).identity().build().expect("build");
        let dst = mesh.apply(&src, 16, 16, 16, 16).expect("apply");

        assert_eq!(dst.len(), 16 * 16 * 3);
        // Centre pixel should be unchanged
        let cx = 8usize;
        let cy = 8usize;
        let base = (cy * 16 + cx) * 3;
        assert_eq!(dst[base], 200, "R channel");
        assert_eq!(dst[base + 1], 100, "G channel");
        assert_eq!(dst[base + 2], 50, "B channel");
    }

    #[test]
    fn identity_mesh_apply_output_has_correct_size() {
        let src = solid_rgb(32, 16, 128, 128, 128);
        let mesh = MeshBuilder::new(3, 3).identity().build().expect("build");
        let dst = mesh.apply(&src, 32, 16, 64, 32).expect("apply");
        assert_eq!(dst.len(), 64 * 32 * 3);
    }

    // ── Flip operations ──────────────────────────────────────────────────────

    #[test]
    fn flip_horizontal_swaps_left_right() {
        // Build a gradient image: left column is red (255,0,0), right column is blue (0,0,255)
        let w = 16u32;
        let h = 8u32;
        let mut src = vec![0u8; (w * h * 3) as usize];
        for row in 0..h as usize {
            for col in 0..w as usize {
                let base = (row * w as usize + col) * 3;
                src[base] = (col * 255 / (w as usize - 1)) as u8; // R gradient
            }
        }
        let mesh = MeshBuilder::new(9, 5)
            .flip_horizontal()
            .build()
            .expect("build");
        let dst = mesh.apply(&src, w, h, w, h).expect("apply");

        // Left column of output should be ~255 (was right column of src)
        let left_r = dst[0]; // pixel (0,0) R channel
        let right_r = dst[((w - 1) * 3) as usize]; // pixel (w-1, 0) R channel
        assert!(
            left_r > 200,
            "flipped left should be ~red (255), got {left_r}"
        );
        assert!(right_r < 55, "flipped right should be ~zero, got {right_r}");
    }

    #[test]
    fn flip_vertical_swaps_top_bottom() {
        let w = 8u32;
        let h = 16u32;
        let mut src = vec![0u8; (w * h * 3) as usize];
        for row in 0..h as usize {
            for col in 0..w as usize {
                let base = (row * w as usize + col) * 3;
                src[base] = (row * 255 / (h as usize - 1)) as u8; // R gradient top→bottom
            }
        }
        let mesh = MeshBuilder::new(5, 9)
            .flip_vertical()
            .build()
            .expect("build");
        let dst = mesh.apply(&src, w, h, w, h).expect("apply");

        // Top row of output should be ~255 (was bottom row of src)
        let top_r = dst[0];
        let bottom_base = ((h - 1) * w * 3) as usize;
        let bottom_r = dst[bottom_base];
        assert!(top_r > 200, "flipped top should be ~255, got {top_r}");
        assert!(bottom_r < 55, "flipped bottom should be ~0, got {bottom_r}");
    }

    // ── Barrel distortion ────────────────────────────────────────────────────

    #[test]
    fn barrel_distortion_identity_k0_matches_passthrough() {
        let src = solid_rgb(16, 16, 80, 160, 240);
        let mesh = MeshBuilder::new(9, 9)
            .barrel_distortion(0.0)
            .build()
            .expect("build");
        let dst = mesh.apply(&src, 16, 16, 16, 16).expect("apply");
        // Should still produce the correct solid colour at centre
        let cx = 8usize;
        let cy = 8usize;
        let base = (cy * 16 + cx) * 3;
        let diff_r = (dst[base] as i32 - 80).abs();
        assert!(
            diff_r <= 2,
            "barrel k=0 should be passthrough, got R diff {diff_r}"
        );
    }

    #[test]
    fn barrel_distortion_nonzero_k_changes_output() {
        let w = 32u32;
        let h = 32u32;
        // Create a checkerboard-like image
        let mut src = vec![0u8; (w * h * 3) as usize];
        for row in 0..h as usize {
            for col in 0..w as usize {
                let v = if (row + col) % 2 == 0 { 200u8 } else { 50u8 };
                let base = (row * w as usize + col) * 3;
                src[base] = v;
                src[base + 1] = v;
                src[base + 2] = v;
            }
        }
        let identity_mesh = MeshBuilder::new(9, 9).identity().build().expect("build");
        let barrel_mesh = MeshBuilder::new(9, 9)
            .barrel_distortion(0.3)
            .build()
            .expect("build");
        let dst_id = identity_mesh.apply(&src, w, h, w, h).expect("apply");
        let dst_barrel = barrel_mesh.apply(&src, w, h, w, h).expect("apply");
        // They should not be identical (barrel distortion changes the mapping)
        assert_ne!(
            dst_id, dst_barrel,
            "barrel k=0.3 should differ from identity"
        );
    }

    // ── max_distortion ───────────────────────────────────────────────────────

    #[test]
    fn identity_mesh_max_distortion_is_zero() {
        let mesh = MeshBuilder::new(5, 5).identity().build().expect("build");
        let d = mesh.max_distortion();
        assert!(
            d < 1e-6,
            "identity mesh should have zero distortion, got {d}"
        );
    }

    #[test]
    fn flip_mesh_max_distortion_is_nonzero() {
        let mesh = MeshBuilder::new(5, 5)
            .flip_horizontal()
            .build()
            .expect("build");
        let d = mesh.max_distortion();
        assert!(d > 0.0, "flipped mesh should have nonzero distortion");
    }

    // ── radial_distortion_mesh ───────────────────────────────────────────────

    #[test]
    fn radial_distortion_mesh_identity_params_matches_identity_mesh() {
        let params = RadialDistortionParams::identity();
        let radial = radial_distortion_mesh(5, 5, &params).expect("build");
        let identity = MeshBuilder::new(5, 5).identity().build().expect("build");

        for (rv, iv) in radial.vertices.iter().zip(identity.vertices.iter()) {
            let du = (rv.src_u - iv.src_u).abs();
            let dv = (rv.src_v - iv.src_v).abs();
            assert!(du < 1e-6, "radial identity u mismatch: {du}");
            assert!(dv < 1e-6, "radial identity v mismatch: {dv}");
        }
    }

    #[test]
    fn radial_distortion_mesh_nonzero_k1_differs_from_identity() {
        let params = RadialDistortionParams::new(0.2, 0.0, 0.0);
        let radial = radial_distortion_mesh(9, 9, &params).expect("build");
        let identity = MeshBuilder::new(9, 9).identity().build().expect("build");
        let any_diff = radial
            .vertices
            .iter()
            .zip(identity.vertices.iter())
            .any(|(r, i)| (r.src_u - i.src_u).abs() > 1e-6 || (r.src_v - i.src_v).abs() > 1e-6);
        assert!(any_diff, "radial k1=0.2 should differ from identity");
    }

    #[test]
    fn radial_distortion_mesh_zero_cols_returns_error() {
        let params = RadialDistortionParams::identity();
        assert!(radial_distortion_mesh(0, 5, &params).is_err());
    }

    // ── apply_par vs apply ───────────────────────────────────────────────────

    #[test]
    fn apply_par_matches_apply_on_identity_mesh() {
        let src = solid_rgb(32, 16, 100, 150, 200);
        let mesh = MeshBuilder::new(5, 5).identity().build().expect("build");
        let seq = mesh.apply(&src, 32, 16, 32, 16).expect("seq");
        let par = mesh.apply_par(&src, 32, 16, 32, 16).expect("par");
        assert_eq!(seq, par, "sequential and parallel must agree");
    }

    // ── scale_shift ──────────────────────────────────────────────────────────

    #[test]
    fn scale_shift_uniform_does_not_panic() {
        let src = solid_rgb(16, 16, 128, 64, 32);
        let mesh = MeshBuilder::new(5, 5)
            .identity()
            .scale_shift(0.9, 0.9, 0.0, 0.0)
            .build()
            .expect("build");
        let dst = mesh.apply(&src, 16, 16, 16, 16).expect("apply");
        assert_eq!(dst.len(), 16 * 16 * 3);
    }

    // ── WarpMesh::vertex ─────────────────────────────────────────────────────

    #[test]
    fn warp_mesh_vertex_out_of_bounds_returns_none() {
        let mesh = MeshBuilder::new(3, 3).identity().build().expect("build");
        assert!(mesh.vertex(3, 0).is_none());
        assert!(mesh.vertex(0, 3).is_none());
    }

    #[test]
    fn warp_mesh_vertex_in_bounds_returns_some() {
        let mesh = MeshBuilder::new(3, 3).identity().build().expect("build");
        let v = mesh.vertex(1, 1);
        assert!(v.is_some());
        let mv = v.expect("exists");
        assert!((mv.src_u - 0.5).abs() < 1e-6);
        assert!((mv.src_v - 0.5).abs() < 1e-6);
    }

    // ── apply dimension errors ────────────────────────────────────────────────

    #[test]
    fn apply_zero_src_w_returns_error() {
        let mesh = MeshBuilder::new(3, 3).identity().build().expect("build");
        assert!(mesh.apply(&[], 0, 8, 8, 8).is_err());
    }

    #[test]
    fn apply_zero_out_h_returns_error() {
        let src = solid_rgb(8, 8, 0, 0, 0);
        let mesh = MeshBuilder::new(3, 3).identity().build().expect("build");
        assert!(mesh.apply(&src, 8, 8, 8, 0).is_err());
    }

    #[test]
    fn apply_buffer_too_small_returns_error() {
        let mesh = MeshBuilder::new(3, 3).identity().build().expect("build");
        let tiny = vec![0u8; 10]; // way too small for a 16×16 image
        assert!(mesh.apply(&tiny, 16, 16, 16, 16).is_err());
    }

    // ── set_vertex ────────────────────────────────────────────────────────────

    #[test]
    fn set_vertex_updates_specific_vertex() {
        let mesh = MeshBuilder::new(3, 3)
            .identity()
            .set_vertex(1, 1, 0.9, 0.1)
            .build()
            .expect("build");
        let v = mesh.vertex(1, 1).expect("vertex exists");
        assert!((v.src_u - 0.9).abs() < 1e-6);
        assert!((v.src_v - 0.1).abs() < 1e-6);
    }

    // ── rotate_180 ───────────────────────────────────────────────────────────

    #[test]
    fn rotate_180_maps_corner_to_opposite_corner() {
        // rotate_180: vertex at (0,0) should map to src (1.0, 1.0) and
        // vertex at (cols-1, rows-1) should map to src (0.0, 0.0).
        let mesh = MeshBuilder::new(5, 5).rotate_180().build().expect("r180");
        let tl = mesh.vertex(0, 0).expect("top-left");
        let br = mesh.vertex(4, 4).expect("bottom-right");
        assert!(
            (tl.src_u - 1.0).abs() < 1e-6,
            "TL src_u should be 1.0, got {}",
            tl.src_u
        );
        assert!(
            (tl.src_v - 1.0).abs() < 1e-6,
            "TL src_v should be 1.0, got {}",
            tl.src_v
        );
        assert!(
            br.src_u.abs() < 1e-6,
            "BR src_u should be 0.0, got {}",
            br.src_u
        );
        assert!(
            br.src_v.abs() < 1e-6,
            "BR src_v should be 0.0, got {}",
            br.src_v
        );
    }

    #[test]
    fn rotate_180_centre_vertex_maps_to_centre() {
        // The centre vertex (col=2, row=2) of a 5×5 mesh has grid UV (0.5, 0.5),
        // so rotate_180 maps it to src (0.5, 0.5).
        let mesh = MeshBuilder::new(5, 5).rotate_180().build().expect("r180");
        let centre = mesh.vertex(2, 2).expect("centre");
        assert!((centre.src_u - 0.5).abs() < 1e-6);
        assert!((centre.src_v - 0.5).abs() < 1e-6);
    }
}
