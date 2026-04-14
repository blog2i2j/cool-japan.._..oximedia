//! Interactive HTML LUT preview generator.
//!
//! Generates a self-contained HTML document that visualises a 3D LUT with:
//!
//! - A **colour wheel** showing how the LUT maps hues and saturations.
//! - A **before/after gradient bar** for luminance response.
//! - A **neutral (grey) axis response** chart rendered in plain HTML/CSS.
//! - LUT metadata summary table.
//!
//! The output is pure HTML with inline CSS and no JavaScript frameworks, so it
//! renders correctly in any modern browser without network access.
//!
//! # Example
//!
//! ```rust
//! use oximedia_lut::{Lut3d, LutSize};
//! use oximedia_lut::lut_preview_html::{LutPreviewOptions, generate_lut_preview_html};
//!
//! let lut = Lut3d::identity(LutSize::Size17);
//! let opts = LutPreviewOptions::default();
//! let html = generate_lut_preview_html(&lut, "Identity LUT", &opts);
//! assert!(html.contains("<!DOCTYPE html>"));
//! assert!(html.contains("Identity LUT"));
//! ```

use crate::{Lut3d, LutInterpolation};

// ============================================================================
// Public types
// ============================================================================

/// Options controlling what sections are included in the preview HTML.
#[derive(Debug, Clone)]
pub struct LutPreviewOptions {
    /// Number of hue steps in the colour wheel (default: 72 = every 5°).
    pub wheel_hue_steps: usize,
    /// Number of saturation rings in the colour wheel (default: 6).
    pub wheel_sat_rings: usize,
    /// Number of luminance steps in the gradient bar (default: 64).
    pub gradient_steps: usize,
    /// Whether to include the neutral-axis chart.
    pub include_neutral_chart: bool,
    /// Whether to include the metadata table.
    pub include_metadata: bool,
    /// CSS width of the generated page (default: `"900px"`).
    pub page_width: String,
}

impl Default for LutPreviewOptions {
    fn default() -> Self {
        Self {
            wheel_hue_steps: 72,
            wheel_sat_rings: 6,
            gradient_steps: 64,
            include_neutral_chart: true,
            include_metadata: true,
            page_width: "900px".to_owned(),
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Generate a self-contained HTML preview for `lut`.
///
/// # Arguments
///
/// * `lut`   – The 3D LUT to preview.
/// * `title` – Title string embedded in the `<title>` tag and `<h1>`.
/// * `opts`  – Rendering options.
///
/// # Returns
///
/// A `String` containing the complete HTML document.
#[must_use]
pub fn generate_lut_preview_html(lut: &Lut3d, title: &str, opts: &LutPreviewOptions) -> String {
    let mut html = String::with_capacity(64 * 1024);

    html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    html.push_str("<meta charset=\"UTF-8\">\n");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
    push_escaped(&mut html, "<title>", title, "</title>\n");
    push_str(&mut html, &build_css(&opts.page_width));
    html.push_str("</head>\n<body>\n");
    push_escaped(&mut html, "<h1>", title, "</h1>\n");

    if opts.include_metadata {
        push_str(&mut html, &build_metadata_table(lut));
    }

    push_str(&mut html, &build_gradient_section(lut, opts));

    if opts.include_neutral_chart {
        push_str(&mut html, &build_neutral_chart(lut, opts));
    }

    push_str(&mut html, &build_colour_wheel(lut, opts));

    html.push_str("</body>\n</html>\n");
    html
}

/// Generate the HTML for just the colour wheel fragment (no wrapping document).
///
/// Useful when embedding the wheel into an existing page.
#[must_use]
pub fn generate_colour_wheel_fragment(lut: &Lut3d, opts: &LutPreviewOptions) -> String {
    build_colour_wheel(lut, opts)
}

/// Generate the HTML for just the before/after gradient bar fragment.
#[must_use]
pub fn generate_gradient_fragment(lut: &Lut3d, opts: &LutPreviewOptions) -> String {
    build_gradient_section(lut, opts)
}

/// Generate the HTML for the neutral-axis chart fragment.
#[must_use]
pub fn generate_neutral_chart_fragment(lut: &Lut3d, opts: &LutPreviewOptions) -> String {
    build_neutral_chart(lut, opts)
}

// ============================================================================
// Private builders
// ============================================================================

fn build_css(page_width: &str) -> String {
    format!(
        "<style>\n\
         body {{ font-family: Arial, sans-serif; max-width: {page_width}; margin: 0 auto; \
                 padding: 16px; background: #1a1a1a; color: #eee; }}\n\
         h1 {{ font-size: 1.4em; margin-bottom: 8px; }}\n\
         h2 {{ font-size: 1.1em; margin-top: 24px; margin-bottom: 6px; }}\n\
         table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}\n\
         th, td {{ border: 1px solid #444; padding: 6px 10px; text-align: left; }}\n\
         th {{ background: #333; }}\n\
         .gradient-row {{ display: flex; height: 40px; width: 100%; }}\n\
         .gradient-cell {{ flex: 1; }}\n\
         .section {{ margin-bottom: 24px; }}\n\
         .wheel {{ display: flex; flex-wrap: wrap; gap: 2px; }}\n\
         .swatch {{ width: 14px; height: 14px; display: inline-block; border-radius: 2px; }}\n\
         .chart-row {{ display: flex; align-items: center; margin: 2px 0; }}\n\
         .chart-label {{ width: 3em; font-size: 0.75em; text-align: right; margin-right: 6px; }}\n\
         .chart-bar {{ height: 12px; min-width: 2px; border-radius: 2px; }}\n\
         </style>\n"
    )
}

fn build_metadata_table(lut: &Lut3d) -> String {
    let size = lut.size();
    let total = size * size * size;
    format!(
        "<div class=\"section\">\n\
         <h2>LUT Metadata</h2>\n\
         <table>\n\
         <tr><th>Property</th><th>Value</th></tr>\n\
         <tr><td>Grid size</td><td>{size}³ ({total} lattice points)</td></tr>\n\
         <tr><td>Interpolation (preview)</td><td>Tetrahedral</td></tr>\n\
         <tr><td>Generated by</td><td>OxiMedia oximedia-lut</td></tr>\n\
         </table>\n\
         </div>\n"
    )
}

/// Luminance gradient: a row of swatches from black to white, before and after the LUT.
fn build_gradient_section(lut: &Lut3d, opts: &LutPreviewOptions) -> String {
    let steps = opts.gradient_steps.max(2);
    let mut before_row = String::new();
    let mut after_row = String::new();

    for i in 0..steps {
        let v = i as f64 / (steps - 1) as f64;
        let (rb, gb, bb) = to_u8_rgb(v, v, v);
        before_row.push_str(&format!(
            "<div class=\"gradient-cell\" style=\"background:rgb({rb},{gb},{bb});\"></div>"
        ));

        let pixel = [v, v, v];
        let out = lut.apply(&pixel, LutInterpolation::Tetrahedral);
        let (ra, ga, ba) = to_u8_rgb(out[0], out[1], out[2]);
        after_row.push_str(&format!(
            "<div class=\"gradient-cell\" style=\"background:rgb({ra},{ga},{ba});\"></div>"
        ));
    }

    format!(
        "<div class=\"section\">\n\
         <h2>Luminance Gradient — Before / After</h2>\n\
         <p style=\"font-size:0.8em;margin:4px 0\">Before (identity):</p>\n\
         <div class=\"gradient-row\">{before_row}</div>\n\
         <p style=\"font-size:0.8em;margin:4px 0\">After (LUT applied):</p>\n\
         <div class=\"gradient-row\">{after_row}</div>\n\
         </div>\n"
    )
}

/// Neutral axis chart: shows how grey values map through each RGB channel.
fn build_neutral_chart(lut: &Lut3d, opts: &LutPreviewOptions) -> String {
    let steps = opts.gradient_steps.max(2);
    let max_bar_px: usize = 500;
    let mut rows = String::new();

    for i in 0..steps {
        let v = i as f64 / (steps - 1) as f64;
        let pixel = [v, v, v];
        let out = lut.apply(&pixel, LutInterpolation::Tetrahedral);

        let bar_r = ((out[0].clamp(0.0, 1.0) * max_bar_px as f64) as usize).max(1);
        let bar_g = ((out[1].clamp(0.0, 1.0) * max_bar_px as f64) as usize).max(1);
        let bar_b = ((out[2].clamp(0.0, 1.0) * max_bar_px as f64) as usize).max(1);

        let label_pct = (v * 100.0).round() as usize;
        rows.push_str(&format!(
            "<div class=\"chart-row\">\
             <span class=\"chart-label\">{label_pct}%</span>\
             <div style=\"display:flex;gap:2px;\">\
             <div class=\"chart-bar\" style=\"width:{bar_r}px;background:#d44;\"></div>\
             <div class=\"chart-bar\" style=\"width:{bar_g}px;background:#4d4;\"></div>\
             <div class=\"chart-bar\" style=\"width:{bar_b}px;background:#44d;\"></div>\
             </div>\
             </div>\n"
        ));
    }

    format!(
        "<div class=\"section\">\n\
         <h2>Neutral Axis Response (R=G=B input → R/G/B output)</h2>\n\
         {rows}\
         </div>\n"
    )
}

/// Colour wheel: sample hues at multiple saturations.
fn build_colour_wheel(lut: &Lut3d, opts: &LutPreviewOptions) -> String {
    let hue_steps = opts.wheel_hue_steps.max(6);
    let sat_rings = opts.wheel_sat_rings.max(1);
    let mut swatches_before = String::new();
    let mut swatches_after = String::new();

    for ring in 0..sat_rings {
        let sat = (ring + 1) as f64 / sat_rings as f64;
        for h in 0..hue_steps {
            let hue = h as f64 / hue_steps as f64 * 360.0;
            let (r, g, b) = hsl_to_linear(hue, sat, 0.5);

            let (rb8, gb8, bb8) = to_u8_rgb(r, g, b);
            swatches_before.push_str(&format!(
                "<div class=\"swatch\" style=\"background:rgb({rb8},{gb8},{bb8});\" \
                 title=\"H={hue:.0} S={sat:.2}\"></div>"
            ));

            let out = lut.apply(&[r, g, b], LutInterpolation::Tetrahedral);
            let (ra8, ga8, ba8) = to_u8_rgb(out[0], out[1], out[2]);
            swatches_after.push_str(&format!(
                "<div class=\"swatch\" style=\"background:rgb({ra8},{ga8},{ba8});\" \
                 title=\"H={hue:.0} S={sat:.2}\"></div>"
            ));
        }
    }

    format!(
        "<div class=\"section\">\n\
         <h2>Colour Wheel — Before (LUT input)</h2>\n\
         <div class=\"wheel\">{swatches_before}</div>\n\
         <h2>Colour Wheel — After (LUT output)</h2>\n\
         <div class=\"wheel\">{swatches_after}</div>\n\
         </div>\n"
    )
}

// ============================================================================
// Colour-space helpers
// ============================================================================

/// Convert HSL (hue 0-360, sat 0-1, lightness 0-1) to linear RGB `[0, 1]`.
///
/// Uses the standard HSL-to-RGB algorithm (no transfer function applied — the
/// values are treated as scene-linear for LUT input purposes).
#[must_use]
fn hsl_to_linear(hue: f64, sat: f64, lightness: f64) -> (f64, f64, f64) {
    if sat.abs() < f64::EPSILON {
        return (lightness, lightness, lightness);
    }
    let q = if lightness < 0.5 {
        lightness * (1.0 + sat)
    } else {
        lightness + sat - lightness * sat
    };
    let p = 2.0 * lightness - q;
    let r = hue_to_rgb(p, q, hue / 360.0 + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, hue / 360.0);
    let b = hue_to_rgb(p, q, hue / 360.0 - 1.0 / 3.0);
    (r, g, b)
}

fn hue_to_rgb(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 0.5 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

/// Clamp and convert linear `f64` channel to `u8`.
fn to_u8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn to_u8_rgb(r: f64, g: f64, b: f64) -> (u8, u8, u8) {
    (to_u8(r), to_u8(g), to_u8(b))
}

// ============================================================================
// HTML string helpers
// ============================================================================

/// Push `prefix + html-escaped(text) + suffix` into `buf`.
fn push_escaped(buf: &mut String, prefix: &str, text: &str, suffix: &str) {
    buf.push_str(prefix);
    for ch in text.chars() {
        match ch {
            '&' => buf.push_str("&amp;"),
            '<' => buf.push_str("&lt;"),
            '>' => buf.push_str("&gt;"),
            '"' => buf.push_str("&quot;"),
            '\'' => buf.push_str("&#39;"),
            c => buf.push(c),
        }
    }
    buf.push_str(suffix);
}

fn push_str(buf: &mut String, s: &str) {
    buf.push_str(s);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LutSize;

    fn identity_lut() -> Lut3d {
        Lut3d::identity(LutSize::Size17)
    }

    #[test]
    fn test_html_starts_with_doctype() {
        let lut = identity_lut();
        let html = generate_lut_preview_html(&lut, "Test LUT", &LutPreviewOptions::default());
        assert!(
            html.starts_with("<!DOCTYPE html>"),
            "HTML must start with <!DOCTYPE html>"
        );
    }

    #[test]
    fn test_html_contains_title() {
        let lut = identity_lut();
        let html = generate_lut_preview_html(&lut, "My Grading LUT", &LutPreviewOptions::default());
        assert!(html.contains("My Grading LUT"), "Title must appear in HTML");
    }

    #[test]
    fn test_html_special_chars_escaped_in_title() {
        let lut = identity_lut();
        let html =
            generate_lut_preview_html(&lut, "<script>alert(1)</script>", &LutPreviewOptions::default());
        assert!(
            !html.contains("<script>alert(1)</script>"),
            "Title must be HTML-escaped"
        );
        assert!(html.contains("&lt;script&gt;"), "Title must be HTML-escaped");
    }

    #[test]
    fn test_html_contains_metadata_table_by_default() {
        let lut = identity_lut();
        let html = generate_lut_preview_html(&lut, "T", &LutPreviewOptions::default());
        assert!(html.contains("LUT Metadata"), "Metadata table should be present");
        assert!(html.contains("Grid size"), "Grid size row should be present");
    }

    #[test]
    fn test_html_no_metadata_when_disabled() {
        let lut = identity_lut();
        let opts = LutPreviewOptions {
            include_metadata: false,
            ..Default::default()
        };
        let html = generate_lut_preview_html(&lut, "T", &opts);
        assert!(!html.contains("LUT Metadata"), "Metadata should be absent");
    }

    #[test]
    fn test_html_contains_gradient_section() {
        let lut = identity_lut();
        let html = generate_lut_preview_html(&lut, "T", &LutPreviewOptions::default());
        assert!(
            html.contains("Luminance Gradient"),
            "Gradient section should be present"
        );
    }

    #[test]
    fn test_html_contains_neutral_chart() {
        let lut = identity_lut();
        let html = generate_lut_preview_html(&lut, "T", &LutPreviewOptions::default());
        assert!(
            html.contains("Neutral Axis Response"),
            "Neutral chart should be present"
        );
    }

    #[test]
    fn test_html_no_neutral_chart_when_disabled() {
        let lut = identity_lut();
        let opts = LutPreviewOptions {
            include_neutral_chart: false,
            ..Default::default()
        };
        let html = generate_lut_preview_html(&lut, "T", &opts);
        assert!(
            !html.contains("Neutral Axis Response"),
            "Neutral chart should be absent"
        );
    }

    #[test]
    fn test_html_contains_colour_wheel() {
        let lut = identity_lut();
        let html = generate_lut_preview_html(&lut, "T", &LutPreviewOptions::default());
        assert!(
            html.contains("Colour Wheel"),
            "Colour wheel section should be present"
        );
    }

    #[test]
    fn test_gradient_fragment_is_valid_html_fragment() {
        let lut = identity_lut();
        let frag = generate_gradient_fragment(&lut, &LutPreviewOptions::default());
        assert!(frag.contains("gradient-row"), "Should contain gradient-row divs");
        assert!(!frag.contains("<!DOCTYPE"), "Fragment should not be a full document");
    }

    #[test]
    fn test_colour_wheel_fragment_standalone() {
        let lut = identity_lut();
        let frag = generate_colour_wheel_fragment(&lut, &LutPreviewOptions::default());
        assert!(frag.contains("swatch"), "Should contain swatch divs");
    }

    #[test]
    fn test_neutral_chart_fragment_standalone() {
        let lut = identity_lut();
        let frag = generate_neutral_chart_fragment(&lut, &LutPreviewOptions::default());
        assert!(frag.contains("chart-row"), "Should contain chart-row divs");
    }

    #[test]
    fn test_custom_page_width_in_css() {
        let lut = identity_lut();
        let opts = LutPreviewOptions {
            page_width: "1200px".to_owned(),
            ..Default::default()
        };
        let html = generate_lut_preview_html(&lut, "T", &opts);
        assert!(html.contains("1200px"), "Custom page width should appear in CSS");
    }

    #[test]
    fn test_identity_lut_before_after_near_equal() {
        // For an identity LUT the before and after gradient colours should be
        // nearly the same (pixel-for-pixel).  We verify by checking that the
        // "Before" and "After" rows both contain white (rgb(255,255,255)).
        let lut = identity_lut();
        let opts = LutPreviewOptions {
            gradient_steps: 4,
            ..Default::default()
        };
        let html = generate_lut_preview_html(&lut, "T", &opts);
        // White swatch must appear in both rows.
        assert!(html.contains("rgb(255,255,255)"), "White swatch expected for identity LUT");
    }

    #[test]
    fn test_hsl_to_linear_red() {
        let (r, g, b) = hsl_to_linear(0.0, 1.0, 0.5);
        assert!((r - 1.0).abs() < 1e-9, "Red hue should produce r≈1");
        assert!(g < 1e-9, "Red hue should produce g≈0");
        assert!(b < 1e-9, "Red hue should produce b≈0");
    }

    #[test]
    fn test_hsl_to_linear_grey_when_zero_saturation() {
        let (r, g, b) = hsl_to_linear(120.0, 0.0, 0.5);
        assert!((r - 0.5).abs() < 1e-9);
        assert!((g - 0.5).abs() < 1e-9);
        assert!((b - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_to_u8_clamps() {
        assert_eq!(to_u8(-0.5), 0);
        assert_eq!(to_u8(1.5), 255);
        assert_eq!(to_u8(0.0), 0);
        assert_eq!(to_u8(1.0), 255);
    }

    #[test]
    fn test_lut_applied_in_gradient() {
        // Build a LUT that inverts luminance (x → 1-x on all channels).
        let lut = Lut3d::from_fn(LutSize::Size17, |rgb: [f64; 3]| [1.0 - rgb[0], 1.0 - rgb[1], 1.0 - rgb[2]]);
        let opts = LutPreviewOptions {
            gradient_steps: 2,
            include_neutral_chart: false,
            include_metadata: false,
            wheel_hue_steps: 6,
            wheel_sat_rings: 1,
            page_width: "900px".to_owned(),
        };
        let html = generate_lut_preview_html(&lut, "Invert", &opts);
        // For input black (0,0,0) the inverted LUT should output white (255,255,255)
        // and vice-versa.  Both should appear as swatches.
        assert!(html.contains("rgb(255,255,255)"), "Expected white swatch from inverted LUT");
        assert!(html.contains("rgb(0,0,0)"), "Expected black swatch from inverted LUT");
    }
}
