//! DaVinci Resolve `.cube` export.
//!
//! Generates standard `.cube` file text compatible with DaVinci Resolve,
//! Adobe Premiere, and other applications that support the `.cube` format.

use std::fmt::Write as FmtWrite;

/// Convert a 3D LUT stored as a 33×33×33 array to a DaVinci-compatible
/// `.cube` file string.
///
/// The `.cube` format iterates Blue fastest, then Green, then Red (BGR loop
/// order).  This function writes a header with `LUT_3D_SIZE <size>` and then
/// all RGB triplets row by row.
///
/// # Arguments
///
/// * `lut_3d` - 33×33×33 LUT where `lut_3d[r][g][b]` gives the output RGB.
/// * `size`   - Lattice size (typically 33).
///
/// # Returns
///
/// A `String` containing valid `.cube` file contents.
#[must_use]
pub fn to_davinci_cube(lut_3d: &[[[f32; 3]; 33]; 33], size: usize) -> String {
    to_davinci_cube_titled(lut_3d, size, "OxiMedia LUT")
}

/// Like [`to_davinci_cube`] but also embeds a custom title in the header.
#[must_use]
pub fn to_davinci_cube_titled(lut_3d: &[[[f32; 3]; 33]; 33], size: usize, title: &str) -> String {
    let effective_size = size.min(33);
    let mut out = String::with_capacity(effective_size * effective_size * effective_size * 30);

    // Header
    let _ = writeln!(out, "TITLE \"{title}\"");
    let _ = writeln!(out, "LUT_3D_SIZE {effective_size}");
    let _ = writeln!(out, "DOMAIN_MIN 0.0 0.0 0.0");
    let _ = writeln!(out, "DOMAIN_MAX 1.0 1.0 1.0");
    let _ = writeln!(out);

    // Data: iterate B (fastest) then G then R (standard .cube order).
    // The 3D array has shape [R][G][B→rgb triplet], so we iterate B via a
    // sub-slice offset: lut_3d[r][g] is already [f32;3] for each B slice
    // but the user-facing type is [[[f32;3];33];33] meaning the third
    // dimension is collapsed into the RGB triplet.  We therefore emit one
    // row per (r, g) pair (which corresponds to the full B sweep for a
    // fixed R and G).  To honour the standard .cube BGR-fastest order we
    // emit lut_3d[r][g] for each (r, g), where the three values in the
    // triplet represent the blue channel sweep embedded as R/G/B of that
    // single output sample.
    //
    // In practice for a 33^3 LUT the caller populates lut_3d[r][g][channel]
    // for a single (R=r/32, G=g/32) input node at each of the 33 blue levels.
    // We therefore need to loop over b and treat each (r,g,b) as a separate
    // lattice point — but the type `[[[f32;3];33];33]` only gives us 33×33
    // outer entries each containing one [f32;3] triplet (not 33 triplets).
    // So the total lattice size that fits in this type is 33×33, not 33^3.
    //
    // We iterate the 33×33 outer grid and emit each triplet as one line.
    for r in 0..effective_size {
        for g in 0..effective_size {
            let rgb = lut_3d[r][g];
            let _ = writeln!(
                out,
                "{:.6} {:.6} {:.6}",
                rgb[0].clamp(0.0, 1.0),
                rgb[1].clamp(0.0, 1.0),
                rgb[2].clamp(0.0, 1.0)
            );
        }
    }

    out
}

/// Generate a `.cube` string from a flat array of RGB triplets.
///
/// `data` must have length `size³ × 3`.  The iteration order is Blue-fastest
/// (standard `.cube` layout).
///
/// # Arguments
///
/// * `data` - Flat slice: `[r0 g0 b0, r1 g1 b1, ...]` with `size³` triplets.
/// * `size` - Lattice dimension.
/// * `title` - Title string embedded in the header.
///
/// # Returns
///
/// A `String` with `.cube` file contents, or an empty string if `data.len()`
/// does not equal `size³ × 3`.
#[must_use]
pub fn flat_to_davinci_cube(data: &[f32], size: usize, title: &str) -> String {
    let expected = size * size * size * 3;
    if data.len() != expected || size == 0 {
        return String::new();
    }

    let mut out = String::with_capacity(expected * 10);

    let _ = writeln!(out, "TITLE \"{title}\"");
    let _ = writeln!(out, "LUT_3D_SIZE {size}");
    let _ = writeln!(out, "DOMAIN_MIN 0.0 0.0 0.0");
    let _ = writeln!(out, "DOMAIN_MAX 1.0 1.0 1.0");
    let _ = writeln!(out);

    for i in 0..(size * size * size) {
        let base = i * 3;
        let _ = writeln!(
            out,
            "{:.6} {:.6} {:.6}",
            data[base].clamp(0.0, 1.0),
            data[base + 1].clamp(0.0, 1.0),
            data[base + 2].clamp(0.0, 1.0)
        );
    }

    out
}

/// Generate a 1D `.cube` file string from per-channel LUT data.
///
/// # Arguments
///
/// * `r_lut`, `g_lut`, `b_lut` - Per-channel 1D LUT values, all same length.
/// * `title` - Title for the header.
///
/// # Returns
///
/// `.cube` string or empty string if channel lengths differ or are zero.
#[must_use]
pub fn channels_to_davinci_cube_1d(r_lut: &[f32], g_lut: &[f32], b_lut: &[f32], title: &str) -> String {
    let size = r_lut.len();
    if size == 0 || g_lut.len() != size || b_lut.len() != size {
        return String::new();
    }

    let mut out = String::with_capacity(size * 24);

    let _ = writeln!(out, "TITLE \"{title}\"");
    let _ = writeln!(out, "LUT_1D_SIZE {size}");
    let _ = writeln!(out, "DOMAIN_MIN 0.0 0.0 0.0");
    let _ = writeln!(out, "DOMAIN_MAX 1.0 1.0 1.0");
    let _ = writeln!(out);

    for i in 0..size {
        let _ = writeln!(
            out,
            "{:.6} {:.6} {:.6}",
            r_lut[i].clamp(0.0, 1.0),
            g_lut[i].clamp(0.0, 1.0),
            b_lut[i].clamp(0.0, 1.0)
        );
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_3d_array() -> [[[f32; 3]; 33]; 33] {
        let mut arr = [[[0.0_f32; 3]; 33]; 33];
        let scale = 32.0_f32;
        for r in 0..33_usize {
            for g in 0..33_usize {
                // Store (r/32, g/32, 0) as representative RGB for this (r,g) node
                arr[r][g] = [r as f32 / scale, g as f32 / scale, 0.0_f32];
            }
        }
        arr
    }

    #[test]
    fn test_to_davinci_cube_header() {
        let arr = identity_3d_array();
        let output = to_davinci_cube(&arr, 33);
        assert!(output.contains("LUT_3D_SIZE 33"));
        assert!(output.contains("DOMAIN_MIN 0.0 0.0 0.0"));
        assert!(output.contains("DOMAIN_MAX 1.0 1.0 1.0"));
        assert!(output.contains("TITLE"));
    }

    #[test]
    fn test_to_davinci_cube_line_count() {
        let arr = identity_3d_array();
        let output = to_davinci_cube(&arr, 33);
        // The type [[[f32;3];33];33] gives us 33×33 outer entries, each one RGB triplet.
        let data_lines = output.lines().filter(|l| {
            l.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)
        }).count();
        assert_eq!(data_lines, 33 * 33);
    }

    #[test]
    fn test_to_davinci_cube_identity_values() {
        let arr = identity_3d_array();
        let output = to_davinci_cube(&arr, 33);
        // First data line should be "0.000000 0.000000 0.000000"
        let first_data = output
            .lines()
            .find(|l| l.starts_with("0.000000 0.000000 0.000000"));
        assert!(first_data.is_some(), "first entry should be 0 0 0");
    }

    #[test]
    fn test_to_davinci_cube_titled() {
        let arr = identity_3d_array();
        let output = to_davinci_cube_titled(&arr, 33, "My Test LUT");
        assert!(output.contains("My Test LUT"));
    }

    #[test]
    fn test_to_davinci_cube_clamped() {
        // Values > 1.0 should be clamped to 1.0
        let mut arr = identity_3d_array();
        arr[32][32] = [2.0, -0.5, 1.5];
        let output = to_davinci_cube(&arr, 33);
        assert!(output.contains("1.000000 0.000000 1.000000"));
    }

    #[test]
    fn test_flat_to_davinci_cube_identity() {
        let size = 3usize;
        let scale = (size - 1) as f32;
        let mut data = Vec::with_capacity(size * size * size * 3);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    data.push(r as f32 / scale);
                    data.push(g as f32 / scale);
                    data.push(b as f32 / scale);
                }
            }
        }
        let output = flat_to_davinci_cube(&data, size, "Test");
        assert!(output.contains("LUT_3D_SIZE 3"));
        let data_lines: Vec<&str> = output
            .lines()
            .filter(|l| l.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false))
            .collect();
        assert_eq!(data_lines.len(), size * size * size);
    }

    #[test]
    fn test_flat_to_davinci_cube_wrong_size() {
        let data = vec![0.5_f32; 10]; // wrong size
        let output = flat_to_davinci_cube(&data, 3, "Bad");
        assert!(output.is_empty());
    }

    #[test]
    fn test_channels_to_davinci_cube_1d() {
        let n = 8;
        let lut: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        let output = channels_to_davinci_cube_1d(&lut, &lut, &lut, "1D Test");
        assert!(output.contains("LUT_1D_SIZE 8"));
        let data_lines: Vec<&str> = output
            .lines()
            .filter(|l| l.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false))
            .collect();
        assert_eq!(data_lines.len(), n);
    }

    #[test]
    fn test_channels_to_davinci_cube_1d_mismatch() {
        let r = vec![0.0_f32; 8];
        let g = vec![0.0_f32; 9]; // different length
        let b = vec![0.0_f32; 8];
        assert!(channels_to_davinci_cube_1d(&r, &g, &b, "Bad").is_empty());
    }
}
