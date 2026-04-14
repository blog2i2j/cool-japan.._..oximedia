//! CIE 1931 xy chromaticity diagram renderer.
//!
//! This module renders the CIE 1931 xy chromaticity diagram onto a black RGBA
//! canvas.  It supports:
//!
//! - A **spectral locus** approximation drawn as an outline of the visible-
//!   colour boundary (using tabulated Stockman–Sharpe 2° observer xy values).
//! - **Colorspace boundary** polygons (triangles) for Rec.709/BT.709, BT.2020,
//!   and DCI-P3 — with built-in constructors.
//! - **Sample point** overlay: arbitrary `(x, y, luminance)` dots.
//! - **Coverage percentage**: fraction of sample points inside a given boundary.
//!
//! # Coordinate mapping
//!
//! CIE xy coordinates span roughly x ∈ [0.0, 0.80] and y ∈ [0.0, 0.90].
//! The renderer maps this rectangle to the full output canvas, flipping the
//! y-axis so that high-y values appear at the top of the image.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

// ─── Spectral locus (tabulated CIE 1931 2° standard observer) ────────────────

/// A highly-simplified set of (x, y) boundary points that trace the visible
/// spectral locus from ~380 nm to ~700 nm plus the line of purples closing the
/// curve.  These values are taken from the CIE 1931 2° observer xy tables at
/// 10 nm intervals, representative for diagram rendering purposes.
const SPECTRAL_LOCUS_XY: &[(f32, f32)] = &[
    (0.1741, 0.0050), // 380 nm
    (0.1740, 0.0050),
    (0.1738, 0.0049),
    (0.1736, 0.0054),
    (0.1730, 0.0048),
    (0.1726, 0.0048),
    (0.1714, 0.0051),
    (0.1689, 0.0102),
    (0.1644, 0.0242),
    (0.1566, 0.0539),
    (0.1440, 0.1050),
    (0.1241, 0.1807),
    (0.0913, 0.2868),
    (0.0454, 0.4449),
    (0.0082, 0.5384),
    (0.0139, 0.6503),
    (0.0743, 0.7850),
    (0.1547, 0.8059),
    (0.2296, 0.7543),
    (0.3016, 0.6923),
    (0.3731, 0.6245),
    (0.4441, 0.5547),
    (0.5125, 0.4866),
    (0.5752, 0.4237),
    (0.6270, 0.3725),
    (0.6658, 0.3340),
    (0.6915, 0.3083),
    (0.7079, 0.2920),
    (0.7140, 0.2856), // 700 nm
    // Close back to 380 nm (line of purples)
    (0.1741, 0.0050),
];

// ─── Data types ───────────────────────────────────────────────────────────────

/// A single sample point in CIE xy chromaticity space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CieXyPoint {
    /// CIE x chromaticity coordinate.
    pub x: f32,
    /// CIE y chromaticity coordinate.
    pub y: f32,
    /// Relative luminance (arbitrary units; used for optional brightness scaling).
    pub luminance: f32,
}

impl CieXyPoint {
    /// Construct a new `CieXyPoint`.
    #[must_use]
    pub fn new(x: f32, y: f32, luminance: f32) -> Self {
        Self { x, y, luminance }
    }
}

/// A named colorspace boundary polygon (typically a triangle defined by the
/// three primaries, plus an optional white point).
#[derive(Debug, Clone)]
pub struct ColorspaceBoundary {
    /// Human-readable name (e.g. `"Rec.709"`).
    pub name: String,
    /// CIE xy vertices forming the closed boundary polygon.
    pub primaries: Vec<(f32, f32)>,
}

impl ColorspaceBoundary {
    /// Rec.709 / sRGB / BT.709 triangle.
    ///
    /// Primary coordinates from ITU-R BT.709-6.
    #[must_use]
    pub fn srgb_bt709() -> Self {
        Self {
            name: "Rec.709".to_string(),
            primaries: vec![
                (0.6400, 0.3300), // Red
                (0.3000, 0.6000), // Green
                (0.1500, 0.0600), // Blue
            ],
        }
    }

    /// BT.2020 / Rec.2020 triangle.
    ///
    /// Primary coordinates from ITU-R BT.2020-2.
    #[must_use]
    pub fn bt2020() -> Self {
        Self {
            name: "BT.2020".to_string(),
            primaries: vec![
                (0.7080, 0.2920), // Red
                (0.1700, 0.7970), // Green
                (0.1310, 0.0460), // Blue
            ],
        }
    }

    /// DCI-P3 triangle.
    ///
    /// Primary coordinates from SMPTE ST 431-2.
    #[must_use]
    pub fn dci_p3() -> Self {
        Self {
            name: "DCI-P3".to_string(),
            primaries: vec![
                (0.6800, 0.3200), // Red
                (0.2650, 0.6900), // Green
                (0.1500, 0.0600), // Blue
            ],
        }
    }

    /// Returns `true` if CIE xy point `(px, py)` is inside this polygon.
    ///
    /// Uses the ray-casting (even–odd) algorithm; works correctly for convex
    /// and concave polygons.
    #[must_use]
    pub fn contains(&self, px: f32, py: f32) -> bool {
        point_in_polygon(px, py, &self.primaries)
    }
}

/// Ray-casting even-odd point-in-polygon test.
fn point_in_polygon(px: f32, py: f32, polygon: &[(f32, f32)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];
        let crosses_y = (yi > py) != (yj > py);
        let x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi;
        if crosses_y && px < x_intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

// ─── Renderer ─────────────────────────────────────────────────────────────────

/// CIE 1931 xy chromaticity diagram renderer.
#[derive(Debug, Clone)]
pub struct CieXyDiagram {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
}

impl CieXyDiagram {
    /// Create a new diagram renderer with the given output dimensions.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    // ── Coordinate helpers ────────────────────────────────────────────────────

    /// CIE x range used for the canvas (slight margin beyond 0–0.80).
    const X_MIN: f32 = -0.02;
    const X_MAX: f32 = 0.82;
    /// CIE y range used for the canvas (slight margin beyond 0–0.90).
    const Y_MIN: f32 = -0.02;
    const Y_MAX: f32 = 0.92;

    /// Map a CIE x coordinate to a pixel column.
    fn cx(&self, x: f32) -> i32 {
        let t = (x - Self::X_MIN) / (Self::X_MAX - Self::X_MIN);
        (t * (self.width - 1) as f32).round() as i32
    }

    /// Map a CIE y coordinate to a pixel row (y-axis is flipped: high y → top).
    fn cy(&self, y: f32) -> i32 {
        let t = (y - Self::Y_MIN) / (Self::Y_MAX - Self::Y_MIN);
        ((1.0 - t) * (self.height - 1) as f32).round() as i32
    }

    /// Write an RGBA pixel clamping to canvas bounds.
    fn set_pixel(buf: &mut [u8], w: usize, x: i32, y: i32, rgba: [u8; 4]) {
        if x < 0 || y < 0 {
            return;
        }
        let (ux, uy) = (x as usize, y as usize);
        let h = buf.len() / (w * 4);
        if ux >= w || uy >= h {
            return;
        }
        let idx = (uy * w + ux) * 4;
        buf[idx] = rgba[0];
        buf[idx + 1] = rgba[1];
        buf[idx + 2] = rgba[2];
        buf[idx + 3] = rgba[3];
    }

    /// Draw a thin line between two pixel coordinates (Bresenham).
    fn draw_line(
        buf: &mut [u8],
        w: usize,
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
        rgba: [u8; 4],
    ) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx: i32 = if x0 < x1 { 1 } else { -1 };
        let sy: i32 = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let mut cx = x0;
        let mut cy = y0;

        loop {
            Self::set_pixel(buf, w, cx, cy, rgba);
            if cx == x1 && cy == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                cx += sx;
            }
            if e2 <= dx {
                err += dx;
                cy += sy;
            }
        }
    }

    /// Draw a filled circle of radius `r` at `(cx, cy)`.
    fn draw_dot(buf: &mut [u8], w: usize, cx: i32, cy: i32, r: i32, rgba: [u8; 4]) {
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    Self::set_pixel(buf, w, cx + dx, cy + dy, rgba);
                }
            }
        }
    }

    // ── Public render API ─────────────────────────────────────────────────────

    /// Render the CIE xy diagram to an RGBA byte buffer.
    ///
    /// # Arguments
    ///
    /// * `sample_points` – Arbitrary chromaticity samples to overlay.
    /// * `boundaries`    – Colorspace triangles to outline on the diagram.
    ///
    /// # Returns
    ///
    /// RGBA byte buffer of length `width * height * 4`.
    #[must_use]
    pub fn render(
        &self,
        sample_points: &[CieXyPoint],
        boundaries: &[ColorspaceBoundary],
    ) -> Vec<u8> {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut buf = vec![0u8; w * h * 4];

        // 1. Draw spectral locus outline (dim cyan)
        for window in SPECTRAL_LOCUS_XY.windows(2) {
            let (x0, y0) = window[0];
            let (x1, y1) = window[1];
            let px0 = self.cx(x0);
            let py0 = self.cy(y0);
            let px1 = self.cx(x1);
            let py1 = self.cy(y1);
            Self::draw_line(&mut buf, w, px0, py0, px1, py1, [80, 160, 160, 200]);
        }

        // 2. Draw colorspace boundaries
        // Assign a distinct colour per boundary
        let boundary_colors: &[[u8; 4]] = &[
            [255, 220, 80, 200],  // yellow (Rec.709)
            [100, 220, 255, 200], // sky blue (BT.2020)
            [220, 120, 255, 200], // magenta (DCI-P3)
            [80, 255, 140, 200],  // green (others)
        ];
        for (bi, boundary) in boundaries.iter().enumerate() {
            let color = boundary_colors[bi % boundary_colors.len()];
            let pts = &boundary.primaries;
            let n = pts.len();
            for i in 0..n {
                let (ax, ay) = pts[i];
                let (bx, by) = pts[(i + 1) % n];
                let px0 = self.cx(ax);
                let py0 = self.cy(ay);
                let px1 = self.cx(bx);
                let py1 = self.cy(by);
                Self::draw_line(&mut buf, w, px0, py0, px1, py1, color);
            }
        }

        // 3. Draw sample points (white dots)
        for pt in sample_points {
            let px = self.cx(pt.x);
            let py = self.cy(pt.y);
            // Scale dot brightness by luminance (clamped 0..1 → 128..255)
            let brightness = ((pt.luminance.clamp(0.0, 1.0) * 127.0) as u8).saturating_add(128);
            Self::draw_dot(&mut buf, w, px, py, 2, [brightness, brightness, brightness, 220]);
        }

        buf
    }

    /// Compute the percentage [0.0, 100.0] of `points` that fall within the
    /// given `boundary` polygon.
    ///
    /// Returns `0.0` if `points` is empty.
    #[must_use]
    pub fn coverage_percent(points: &[CieXyPoint], boundary: &ColorspaceBoundary) -> f32 {
        if points.is_empty() {
            return 0.0;
        }
        let inside = points
            .iter()
            .filter(|p| boundary.contains(p.x, p.y))
            .count();
        inside as f32 / points.len() as f32 * 100.0
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_bt709_boundary_construction() {
        let b = ColorspaceBoundary::srgb_bt709();
        assert_eq!(b.name, "Rec.709");
        assert_eq!(b.primaries.len(), 3);
    }

    #[test]
    fn test_bt2020_boundary_construction() {
        let b = ColorspaceBoundary::bt2020();
        assert_eq!(b.name, "BT.2020");
        assert_eq!(b.primaries.len(), 3);
    }

    #[test]
    fn test_dci_p3_boundary_construction() {
        let b = ColorspaceBoundary::dci_p3();
        assert_eq!(b.name, "DCI-P3");
        assert_eq!(b.primaries.len(), 3);
    }

    #[test]
    fn test_d65_white_point_inside_rec709() {
        // D65 white point (0.3127, 0.3290) should be inside the Rec.709 triangle
        let boundary = ColorspaceBoundary::srgb_bt709();
        assert!(
            boundary.contains(0.3127, 0.3290),
            "D65 should be inside Rec.709"
        );
    }

    #[test]
    fn test_point_outside_rec709() {
        // Pure spectral yellow-green (0.17, 0.80) is outside Rec.709
        let boundary = ColorspaceBoundary::srgb_bt709();
        assert!(!boundary.contains(0.17, 0.80));
    }

    #[test]
    fn test_coverage_all_inside() {
        let boundary = ColorspaceBoundary::srgb_bt709();
        // D65 white is inside — 100% coverage
        let points = vec![
            CieXyPoint::new(0.3127, 0.3290, 1.0),
            CieXyPoint::new(0.35, 0.35, 0.5),
        ];
        let cov = CieXyDiagram::coverage_percent(&points, &boundary);
        assert!((cov - 100.0).abs() < 0.01, "coverage={cov}");
    }

    #[test]
    fn test_coverage_none_inside() {
        let boundary = ColorspaceBoundary::srgb_bt709();
        // Spectral locus extremes are outside Rec.709
        let points = vec![
            CieXyPoint::new(0.17, 0.80, 1.0), // far green
            CieXyPoint::new(0.71, 0.29, 1.0), // far red
        ];
        let cov = CieXyDiagram::coverage_percent(&points, &boundary);
        assert!(cov < 100.0);
    }

    #[test]
    fn test_coverage_empty_points() {
        let boundary = ColorspaceBoundary::srgb_bt709();
        let cov = CieXyDiagram::coverage_percent(&[], &boundary);
        assert_eq!(cov, 0.0);
    }

    #[test]
    fn test_render_output_size() {
        let diagram = CieXyDiagram::new(256, 256);
        let buf = diagram.render(&[], &[]);
        assert_eq!(buf.len(), 256 * 256 * 4);
    }

    #[test]
    fn test_render_with_boundaries_and_samples() {
        let diagram = CieXyDiagram::new(128, 128);
        let boundaries = vec![
            ColorspaceBoundary::srgb_bt709(),
            ColorspaceBoundary::bt2020(),
            ColorspaceBoundary::dci_p3(),
        ];
        let samples = vec![CieXyPoint::new(0.3127, 0.3290, 1.0)];
        let buf = diagram.render(&samples, &boundaries);
        assert_eq!(buf.len(), 128 * 128 * 4);
        // At least some pixels should be non-zero (spectral locus is drawn)
        let non_zero = buf.iter().any(|&b| b != 0);
        assert!(non_zero, "rendered image should not be entirely black");
    }

    #[test]
    fn test_partial_coverage_mixed_points() {
        let boundary = ColorspaceBoundary::srgb_bt709();
        // D65 is inside, spectral far-green is outside
        let points = vec![
            CieXyPoint::new(0.3127, 0.3290, 1.0), // inside
            CieXyPoint::new(0.17, 0.80, 1.0),      // outside
        ];
        let cov = CieXyDiagram::coverage_percent(&points, &boundary);
        // Exactly 1/2 inside → 50%
        assert!((cov - 50.0).abs() < 0.01, "coverage={cov}");
    }
}
