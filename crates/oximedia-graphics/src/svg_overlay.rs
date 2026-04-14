//! SVG overlay rendering for broadcast graphics.
//!
//! Provides [`SvgOverlay`] which loads a Scalable Vector Graphics document via the
//! pure-Rust `resvg` / `usvg` pipeline and composites it over an RGBA frame buffer.
//! The design is intentionally analogous to FFmpeg's `drawsvg` filter so that
//! integrators familiar with that tool will feel at home.
//!
//! # Example
//!
//! ```rust
//! use oximedia_graphics::svg_overlay::{SvgOverlay, SvgOverlayBuilder};
//!
//! let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
//!   <rect width="200" height="100" fill="red"/>
//! </svg>"#;
//!
//! let overlay = SvgOverlayBuilder::new()
//!     .from_str(svg)
//!     .expect("parse failed")
//!     .position(10, 20)
//!     .scale(1.0)
//!     .opacity(0.8)
//!     .build();
//!
//! // Render the SVG to a standalone RGBA buffer (200×100 px)
//! let rgba = overlay.render_to_rgba(200, 100).expect("render failed");
//! assert_eq!(rgba.len(), 200 * 100 * 4);
//! ```

use resvg::tiny_skia;
use resvg::usvg;
use thiserror::Error;

// ── Error type ───────────────────────────────────────────────────────────────

/// Errors that can occur during SVG overlay operations.
#[derive(Debug, Error)]
pub enum SvgError {
    /// The SVG source could not be parsed.
    #[error("SVG parse error: {0}")]
    ParseError(String),

    /// The pixmap could not be allocated or the renderer returned nothing.
    #[error("SVG render error: {0}")]
    RenderError(String),

    /// A width or height of zero was supplied.
    #[error("invalid dimensions: width and height must both be greater than zero")]
    InvalidDimensions,
}

// ── Core struct ───────────────────────────────────────────────────────────────

/// A parsed and ready-to-composite SVG overlay.
///
/// Construct one via [`SvgOverlayBuilder`].
#[derive(Debug, Clone)]
pub struct SvgOverlay {
    /// The parsed usvg tree (kept in serialised form so we can re-parse cheaply
    /// without storing a `usvg::Tree` which is not `Clone`).
    svg_data: Vec<u8>,

    /// X position of the overlay's top-left corner on the destination frame (pixels).
    pub x: i32,

    /// Y position of the overlay's top-left corner on the destination frame (pixels).
    pub y: i32,

    /// Uniform scale factor applied to the SVG before compositing.
    ///
    /// `1.0` = render at the SVG's intrinsic size.
    pub scale: f64,

    /// Global opacity applied to the overlay during compositing.
    ///
    /// Clamped to `[0.0, 1.0]`.
    pub opacity: f64,
}

impl SvgOverlay {
    /// Render the SVG at the given `width` × `height` pixel dimensions and return
    /// a flat `Vec<u8>` in RGBA order (4 bytes per pixel, row-major, top-left origin).
    ///
    /// The `scale` factor stored in the overlay is applied on top of whatever
    /// transform is needed to fill `width` × `height`.
    ///
    /// # Errors
    ///
    /// Returns [`SvgError::InvalidDimensions`] when `width == 0 || height == 0`.
    /// Returns [`SvgError::RenderError`] when `tiny_skia` cannot allocate the pixmap.
    pub fn render_to_rgba(&self, width: u32, height: u32) -> Result<Vec<u8>, SvgError> {
        if width == 0 || height == 0 {
            return Err(SvgError::InvalidDimensions);
        }

        let tree = self.parse_tree()?;
        let svg_size = tree.size();

        // Build a transform: scale the SVG to fill [width × height], then
        // apply the user-supplied scale factor.
        let scale_x = (width as f64 / svg_size.width() as f64) * self.scale;
        let scale_y = (height as f64 / svg_size.height() as f64) * self.scale;

        // Use the smaller axis to preserve aspect ratio (letterbox / pillarbox).
        let uniform_scale = scale_x.min(scale_y) as f32;

        let transform = tiny_skia::Transform::from_scale(uniform_scale, uniform_scale);

        let mut pixmap = tiny_skia::Pixmap::new(width, height)
            .ok_or_else(|| SvgError::RenderError("failed to allocate pixmap".to_owned()))?;

        resvg::render(&tree, transform, &mut pixmap.as_mut());

        // Apply per-overlay opacity by scaling every alpha channel value.
        let clamped_opacity = self.opacity.clamp(0.0, 1.0);
        let alpha_factor = clamped_opacity as f32;

        if (alpha_factor - 1.0_f32).abs() > f32::EPSILON {
            apply_opacity(pixmap.data_mut(), alpha_factor);
        }

        Ok(pixmap.take())
    }

    /// Render the SVG at the given `width` × `height` for a specific animation
    /// `frame_index`.
    ///
    /// `resvg` / `usvg` only support static SVGs (no SMIL animation), so
    /// `frame_index` is accepted for API compatibility and forward-compatibility
    /// but has no effect on the rendered output.  Future versions may support
    /// animated SVG via separate frame data sources.
    ///
    /// # Errors
    ///
    /// Same as [`render_to_rgba`](SvgOverlay::render_to_rgba).
    pub fn render_frame_to_rgba(
        &self,
        width: u32,
        height: u32,
        _frame_index: u32,
    ) -> Result<Vec<u8>, SvgError> {
        // NOTE: Static SVGs are frame-invariant; a dynamic implementation would
        // select different document states per frame_index here.
        self.render_to_rgba(width, height)
    }

    /// Alpha-composite this overlay onto the provided RGBA `frame` buffer.
    ///
    /// `frame` must be exactly `frame_width * frame_height * 4` bytes.
    /// Pixels of the SVG that land outside the frame boundary are clipped.
    ///
    /// The compositing formula follows the Porter-Duff *over* operation:
    ///
    /// ```text
    /// out.rgb = src.rgb * src.a + dst.rgb * (1 − src.a)
    /// out.a   = src.a + dst.a   * (1 − src.a)
    /// ```
    ///
    /// where `src` is the (already opacity-scaled) SVG pixel and `dst` is the
    /// frame pixel.
    ///
    /// # Errors
    ///
    /// Returns [`SvgError::InvalidDimensions`] when either dimension is zero.
    /// Returns [`SvgError::RenderError`] on render failures.
    pub fn composite_over(
        &self,
        frame: &mut [u8],
        frame_width: u32,
        frame_height: u32,
    ) -> Result<(), SvgError> {
        if frame_width == 0 || frame_height == 0 {
            return Err(SvgError::InvalidDimensions);
        }

        let tree = self.parse_tree()?;
        let svg_size = tree.size();

        // Determine the render size of the SVG at the requested scale.
        let svg_render_w = ((svg_size.width() as f64) * self.scale).round() as u32;
        let svg_render_h = ((svg_size.height() as f64) * self.scale).round() as u32;

        let render_w = svg_render_w.max(1);
        let render_h = svg_render_h.max(1);

        // Render the SVG into its own pixmap.
        let scale_f = self.scale as f32;
        let transform = tiny_skia::Transform::from_scale(scale_f, scale_f);

        let mut pixmap = tiny_skia::Pixmap::new(render_w, render_h)
            .ok_or_else(|| SvgError::RenderError("failed to allocate pixmap".to_owned()))?;

        resvg::render(&tree, transform, &mut pixmap.as_mut());

        // Apply per-overlay opacity.
        let clamped_opacity = self.opacity.clamp(0.0, 1.0);
        let alpha_factor = clamped_opacity as f32;
        if (alpha_factor - 1.0_f32).abs() > f32::EPSILON {
            apply_opacity(pixmap.data_mut(), alpha_factor);
        }

        let svg_data = pixmap.data();

        // Composite row by row.
        for row in 0..render_h {
            let dst_y = self.y + row as i32;
            if dst_y < 0 || dst_y >= frame_height as i32 {
                continue;
            }
            for col in 0..render_w {
                let dst_x = self.x + col as i32;
                if dst_x < 0 || dst_x >= frame_width as i32 {
                    continue;
                }

                let src_offset = (row * render_w + col) as usize * 4;
                let dst_offset = (dst_y as u32 * frame_width + dst_x as u32) as usize * 4;

                if src_offset + 3 >= svg_data.len() || dst_offset + 3 >= frame.len() {
                    continue;
                }

                porter_duff_over(
                    &svg_data[src_offset..src_offset + 4],
                    &mut frame[dst_offset..dst_offset + 4],
                );
            }
        }

        Ok(())
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Parse the stored SVG bytes into a `usvg::Tree`.
    fn parse_tree(&self) -> Result<usvg::Tree, SvgError> {
        let opt = usvg::Options::default();
        usvg::Tree::from_data(&self.svg_data, &opt)
            .map_err(|e| SvgError::ParseError(e.to_string()))
    }
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Fluent builder for [`SvgOverlay`].
///
/// # Example
///
/// ```rust
/// use oximedia_graphics::svg_overlay::SvgOverlayBuilder;
///
/// let overlay = SvgOverlayBuilder::new()
///     .from_str(r#"<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50">
///       <circle cx="25" cy="25" r="20" fill="blue"/>
///     </svg>"#)
///     .expect("parse")
///     .position(0, 0)
///     .scale(2.0)
///     .opacity(1.0)
///     .build();
/// ```
#[derive(Debug, Default)]
pub struct SvgOverlayBuilder {
    svg_data: Option<Vec<u8>>,
    x: i32,
    y: i32,
    scale: f64,
    opacity: f64,
}

impl SvgOverlayBuilder {
    /// Create a new builder with sane defaults (position 0,0; scale 1.0; opacity 1.0).
    pub fn new() -> Self {
        Self {
            svg_data: None,
            x: 0,
            y: 0,
            scale: 1.0,
            opacity: 1.0,
        }
    }

    /// Load SVG source from a string slice.
    ///
    /// Validates the SVG by performing a trial parse.
    ///
    /// # Errors
    ///
    /// Returns [`SvgError::ParseError`] if the SVG is malformed.
    pub fn from_str(mut self, svg: &str) -> Result<Self, SvgError> {
        let bytes = svg.as_bytes().to_vec();
        validate_svg_bytes(&bytes)?;
        self.svg_data = Some(bytes);
        Ok(self)
    }

    /// Load SVG source from a byte slice.
    ///
    /// Validates the SVG by performing a trial parse.
    ///
    /// # Errors
    ///
    /// Returns [`SvgError::ParseError`] if the SVG is malformed.
    pub fn from_bytes(mut self, data: &[u8]) -> Result<Self, SvgError> {
        let bytes = data.to_vec();
        validate_svg_bytes(&bytes)?;
        self.svg_data = Some(bytes);
        Ok(self)
    }

    /// Set the pixel position of the overlay's top-left corner on the destination frame.
    #[must_use]
    pub fn position(mut self, x: i32, y: i32) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    /// Set the uniform scale factor (`1.0` = intrinsic SVG size).
    #[must_use]
    pub fn scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Set the global opacity (`0.0` = fully transparent, `1.0` = fully opaque).
    ///
    /// Values outside `[0.0, 1.0]` are clamped at composite time.
    #[must_use]
    pub fn opacity(mut self, opacity: f64) -> Self {
        self.opacity = opacity;
        self
    }

    /// Consume the builder and produce an [`SvgOverlay`].
    ///
    /// # Panics
    ///
    /// Panics if neither [`from_str`](Self::from_str) nor [`from_bytes`](Self::from_bytes)
    /// was called.  In production code, call the SVG-loading method before `build`.
    pub fn build(self) -> SvgOverlay {
        SvgOverlay {
            svg_data: self
                .svg_data
                .expect("SvgOverlayBuilder: no SVG source provided; call from_str or from_bytes"),
            x: self.x,
            y: self.y,
            scale: self.scale,
            opacity: self.opacity,
        }
    }

    /// Fallible alternative to [`build`](Self::build).
    ///
    /// Returns `None` when no SVG source was loaded.
    pub fn try_build(self) -> Option<SvgOverlay> {
        Some(SvgOverlay {
            svg_data: self.svg_data?,
            x: self.x,
            y: self.y,
            scale: self.scale,
            opacity: self.opacity,
        })
    }
}

// ── Standalone loading helpers ────────────────────────────────────────────────

/// Parse and construct an [`SvgOverlay`] from an SVG string.
///
/// Equivalent to:
/// ```rust
/// # use oximedia_graphics::svg_overlay::SvgOverlayBuilder;
/// # let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>"#;
/// SvgOverlayBuilder::new().from_str(svg)?.position(0, 0).scale(1.0).opacity(1.0).build()
/// # ; Ok::<(), oximedia_graphics::svg_overlay::SvgError>(())
/// ```
///
/// # Errors
///
/// Returns [`SvgError::ParseError`] if the SVG is malformed.
pub fn svg_overlay_from_str(svg: &str) -> Result<SvgOverlay, SvgError> {
    SvgOverlayBuilder::new().from_str(svg).map(|b| b.build())
}

/// Parse and construct an [`SvgOverlay`] from raw SVG bytes.
///
/// # Errors
///
/// Returns [`SvgError::ParseError`] if the SVG is malformed.
pub fn svg_overlay_from_bytes(data: &[u8]) -> Result<SvgOverlay, SvgError> {
    SvgOverlayBuilder::new()
        .from_bytes(data)
        .map(|b| b.build())
}

// ── Private pixel-level helpers ───────────────────────────────────────────────

/// Multiply every pixel's alpha channel by `factor` in-place.
///
/// `data` must be a flat RGBA buffer (4 bytes per pixel).
fn apply_opacity(data: &mut [u8], factor: f32) {
    for pixel in data.chunks_exact_mut(4) {
        let new_alpha = (pixel[3] as f32 * factor).round() as u8;
        pixel[3] = new_alpha;
        // Pre-multiply RGB by the same factor to maintain consistency with
        // tiny-skia's pre-multiplied alpha representation.
        pixel[0] = (pixel[0] as f32 * factor).round() as u8;
        pixel[1] = (pixel[1] as f32 * factor).round() as u8;
        pixel[2] = (pixel[2] as f32 * factor).round() as u8;
    }
}

/// Porter-Duff *over* compositing of one RGBA pixel.
///
/// `src` and `dst` are 4-byte slices `[R, G, B, A]` with values in `[0, 255]`.
/// `dst` is modified in place.
fn porter_duff_over(src: &[u8], dst: &mut [u8]) {
    let sa = src[3] as f32 / 255.0;
    let da = dst[3] as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);

    if out_a < f32::EPSILON {
        dst[0] = 0;
        dst[1] = 0;
        dst[2] = 0;
        dst[3] = 0;
        return;
    }

    for i in 0..3 {
        let s = src[i] as f32 / 255.0;
        let d = dst[i] as f32 / 255.0;
        let out_c = (s * sa + d * da * (1.0 - sa)) / out_a;
        dst[i] = (out_c * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    dst[3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
}

/// Perform a trial parse of SVG bytes to validate the source.
fn validate_svg_bytes(data: &[u8]) -> Result<(), SvgError> {
    let opt = usvg::Options::default();
    usvg::Tree::from_data(data, &opt).map_err(|e| SvgError::ParseError(e.to_string()))?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal, well-formed SVG used across multiple tests.
    fn simple_svg() -> &'static str {
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <rect width="100" height="100" fill="red"/>
        </svg>"#
    }

    // ── Test 1: parse a simple SVG ────────────────────────────────────────────

    #[test]
    fn test_parse_simple_svg() {
        let result = svg_overlay_from_str(simple_svg());
        assert!(result.is_ok(), "expected Ok, got {result:?}");
        let overlay = result.expect("parse failed");
        // The SVG data must be stored.
        assert!(!overlay.svg_data.is_empty());
    }

    // ── Test 2: render to RGBA with correct buffer size ───────────────────────

    #[test]
    fn test_render_to_rgba_correct_dimensions() {
        let overlay = svg_overlay_from_str(simple_svg()).expect("parse");
        let rgba = overlay.render_to_rgba(100, 100).expect("render");
        // RGBA: 4 bytes per pixel.
        assert_eq!(rgba.len(), 100 * 100 * 4);
    }

    // ── Test 3: composite at (0,0) onto a black frame ─────────────────────────

    #[test]
    fn test_composite_at_origin() {
        let overlay = SvgOverlayBuilder::new()
            .from_str(simple_svg())
            .expect("parse")
            .position(0, 0)
            .scale(1.0)
            .opacity(1.0)
            .build();

        let frame_w = 200u32;
        let frame_h = 200u32;
        let mut frame = vec![0u8; (frame_w * frame_h * 4) as usize];

        overlay
            .composite_over(&mut frame, frame_w, frame_h)
            .expect("composite");

        // The top-left 100×100 region should have been painted by the red SVG rect.
        // Pixel (0,0) should have a non-zero red channel.
        let r = frame[0];
        assert!(
            r > 0,
            "expected red channel > 0 at pixel (0,0), got {r}"
        );
    }

    // ── Test 4: composite at an offset position ───────────────────────────────

    #[test]
    fn test_composite_at_offset() {
        let offset_x = 50i32;
        let offset_y = 60i32;

        let overlay = SvgOverlayBuilder::new()
            .from_str(simple_svg())
            .expect("parse")
            .position(offset_x, offset_y)
            .scale(1.0)
            .opacity(1.0)
            .build();

        let frame_w = 300u32;
        let frame_h = 300u32;
        let mut frame = vec![0u8; (frame_w * frame_h * 4) as usize];

        overlay
            .composite_over(&mut frame, frame_w, frame_h)
            .expect("composite");

        // Pixel at the offset origin should be non-transparent (red rect covers it).
        let idx = (offset_y as u32 * frame_w + offset_x as u32) as usize * 4;
        let r = frame[idx];
        assert!(
            r > 0,
            "expected red channel > 0 at offset pixel, got {r}"
        );

        // Pixel (0,0) should still be black (zero alpha) because the SVG starts at offset.
        assert_eq!(
            frame[3], 0,
            "expected alpha == 0 at pixel (0,0) before offset, got {}",
            frame[3]
        );
    }

    // ── Test 5: invalid SVG returns ParseError ────────────────────────────────

    #[test]
    fn test_invalid_svg_returns_parse_error() {
        let result = svg_overlay_from_str("<this is not svg>");
        match result {
            Err(SvgError::ParseError(_)) => {}
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    // ── Test 6: zero-size frame returns InvalidDimensions ────────────────────

    #[test]
    fn test_zero_size_returns_invalid_dimensions() {
        let overlay = svg_overlay_from_str(simple_svg()).expect("parse");

        let err = overlay.render_to_rgba(0, 100).expect_err("expected error");
        assert!(
            matches!(err, SvgError::InvalidDimensions),
            "unexpected error variant: {err:?}"
        );

        let err2 = overlay.render_to_rgba(100, 0).expect_err("expected error");
        assert!(
            matches!(err2, SvgError::InvalidDimensions),
            "unexpected error variant: {err2:?}"
        );

        let mut frame = vec![0u8; 400];
        let err3 = overlay
            .composite_over(&mut frame, 0, 100)
            .expect_err("expected error");
        assert!(
            matches!(err3, SvgError::InvalidDimensions),
            "unexpected error variant: {err3:?}"
        );
    }

    // ── Test 7: opacity 0.0 produces transparent overlay ─────────────────────

    #[test]
    fn test_opacity_zero_produces_transparent() {
        let overlay = SvgOverlayBuilder::new()
            .from_str(simple_svg())
            .expect("parse")
            .position(0, 0)
            .scale(1.0)
            .opacity(0.0)
            .build();

        let rgba = overlay.render_to_rgba(100, 100).expect("render");

        // Every alpha byte should be 0.
        let all_alpha_zero = rgba.chunks_exact(4).all(|px| px[3] == 0);
        assert!(
            all_alpha_zero,
            "expected all alpha bytes to be 0 with opacity 0.0"
        );
    }

    // ── Test 8: opacity 1.0 produces fully opaque overlay ────────────────────

    #[test]
    fn test_opacity_one_produces_opaque() {
        let overlay = SvgOverlayBuilder::new()
            .from_str(simple_svg())
            .expect("parse")
            .position(0, 0)
            .scale(1.0)
            .opacity(1.0)
            .build();

        let rgba = overlay.render_to_rgba(100, 100).expect("render");

        // At least one pixel should be fully opaque (the solid red rect).
        let any_opaque = rgba.chunks_exact(4).any(|px| px[3] == 255);
        assert!(any_opaque, "expected at least one fully opaque pixel with opacity 1.0");
    }

    // ── Test 9: builder from_bytes ────────────────────────────────────────────

    #[test]
    fn test_from_bytes() {
        let bytes = simple_svg().as_bytes();
        let overlay = SvgOverlayBuilder::new()
            .from_bytes(bytes)
            .expect("parse")
            .build();
        let rgba = overlay.render_to_rgba(50, 50).expect("render");
        assert_eq!(rgba.len(), 50 * 50 * 4);
    }

    // ── Test 10: render_frame_to_rgba is equivalent to render_to_rgba ─────────

    #[test]
    fn test_animated_frame_api() {
        let overlay = svg_overlay_from_str(simple_svg()).expect("parse");
        let normal = overlay.render_to_rgba(64, 64).expect("render");
        let framed = overlay.render_frame_to_rgba(64, 64, 42).expect("render frame");
        assert_eq!(normal, framed, "frame API should match regular render for static SVG");
    }
}
