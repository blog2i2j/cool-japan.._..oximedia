//! White balance and color matrix processing for DNG images.

use super::types::WhiteBalance;

// ==========================================
// White balance and color matrix
// ==========================================

/// Apply white balance multipliers to demosaiced RGB data in place.
///
/// Normalizes the as-shot neutral values and scales each channel.
pub fn apply_white_balance(data: &mut [u16], white_balance: &WhiteBalance, white_level: u32) {
    let neutral = &white_balance.as_shot_neutral;

    // Convert neutral to gain: gain = 1/neutral, then normalize so max gain = 1
    let gains = [
        if neutral[0].abs() > f64::EPSILON {
            1.0 / neutral[0]
        } else {
            1.0
        },
        if neutral[1].abs() > f64::EPSILON {
            1.0 / neutral[1]
        } else {
            1.0
        },
        if neutral[2].abs() > f64::EPSILON {
            1.0 / neutral[2]
        } else {
            1.0
        },
    ];

    // Find min gain to normalize (avoid clipping)
    let min_gain = gains.iter().copied().fold(f64::INFINITY, f64::min);

    if min_gain <= 0.0 {
        return;
    }

    let norm_gains = [
        gains[0] / min_gain,
        gains[1] / min_gain,
        gains[2] / min_gain,
    ];

    let wl = white_level as f64;

    // Apply to each RGB triplet
    let triplet_count = data.len() / 3;
    for i in 0..triplet_count {
        let base = i * 3;
        for ch in 0..3 {
            let val = f64::from(data[base + ch]) * norm_gains[ch];
            data[base + ch] = val.min(wl).max(0.0) as u16;
        }
    }
}

/// Apply a 3x3 color matrix to f64 RGB data in place.
///
/// The matrix maps from camera RGB to the target color space (typically XYZ or sRGB).
/// Data is expected as interleaved [R, G, B, R, G, B, ...].
pub fn apply_color_matrix(data: &mut [f64], matrix: &[[f64; 3]; 3]) {
    let triplet_count = data.len() / 3;
    for i in 0..triplet_count {
        let base = i * 3;
        let r = data[base];
        let g = data[base + 1];
        let b = data[base + 2];

        data[base] = matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b;
        data[base + 1] = matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b;
        data[base + 2] = matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b;
    }
}
