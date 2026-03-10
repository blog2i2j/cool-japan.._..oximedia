//! Internal utility functions for color space conversions.
//!
//! These are private helpers shared between grading submodules.

use super::types::{HslColor, RgbColor};

/// Convert RGB to HSL color space.
pub(super) fn rgb_to_hsl(r: f64, g: f64, b: f64) -> HslColor {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let l = (max + min) / 2.0;

    if delta < f64::EPSILON {
        return HslColor::new(0.0, 0.0, l);
    }

    let s = if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };

    let h = if (r - max).abs() < f64::EPSILON {
        ((g - b) / delta).rem_euclid(6.0)
    } else if (g - max).abs() < f64::EPSILON {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    };

    HslColor::new(h * 60.0, s, l)
}

/// Convert HSL to RGB color space.
pub(super) fn hsl_to_rgb(h: f64, s: f64, l: f64) -> RgbColor {
    if s < f64::EPSILON {
        return RgbColor::gray(l);
    }

    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());

    let (r1, g1, b1) = match h_prime as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    let m = l - c / 2.0;

    RgbColor::new(r1 + m, g1 + m, b1 + m)
}

/// Apply saturation adjustment to a color.
pub(super) fn apply_saturation(color: RgbColor, saturation: f64) -> RgbColor {
    let luma = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;

    RgbColor::new(
        luma + (color.r - luma) * saturation,
        luma + (color.g - luma) * saturation,
        luma + (color.b - luma) * saturation,
    )
}
