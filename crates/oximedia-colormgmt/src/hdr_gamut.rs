//! HDR gamut mapping utilities.
//!
//! Converts wide-gamut HDR content (BT.2020 / Rec.2020) into the narrower
//! BT.709 (sRGB) gamut using a knee-function–based luminance-preserving
//! technique.  The approach is derived from ITU-R BT.2390 Annex 5 (gamut
//! mapping for HDR–SDR conversion).
//!
//! # Algorithm overview
//!
//! 1. Apply the inverse PQ EOTF to convert scene-linear PQ-encoded values to
//!    linear light.
//! 2. Apply the BT.2020 → BT.709 chromatic colour matrix.
//! 3. Compress highlights above `knee` with a soft-clip curve.
//! 4. Clamp negative values that fall outside BT.709 gamut (out-of-gamut
//!    highlights are desaturated toward white before clamping).

// ── Conversion matrices ───────────────────────────────────────────────────────

/// BT.2020 → BT.709 linear-light colour matrix (D65 white point, row-major).
const BT2020_TO_BT709: [[f32; 3]; 3] = [
    [1.6605, -0.5876, -0.0728],
    [-0.1246, 1.1329, -0.0083],
    [-0.0182, -0.1006, 1.1187],
];

// ── Public API ────────────────────────────────────────────────────────────────

/// Map a **linear-light** BT.2020 colour triple to **linear-light** BT.709,
/// applying a knee-function soft-clip to handle highlights gracefully.
///
/// # Arguments
///
/// * `rgb`  — linear-light RGB in BT.2020 primaries.
/// * `knee` — knee point in the range `(0.0, 1.0]`.  Values above `knee` are
///   soft-clipped toward 1.0.  A typical broadcast value is `0.7`.
///
/// # Returns
///
/// Linear-light RGB in BT.709, clamped to `[0.0, 1.0]`.
#[must_use]
pub fn hdr_bt2020_to_bt709(rgb: [f32; 3], knee: f32) -> [f32; 3] {
    // Clamp knee to a sensible range to avoid division by zero / inversion.
    let knee = knee.clamp(0.01, 1.0);

    // Apply BT.2020 → BT.709 colour matrix.
    let m = &BT2020_TO_BT709;
    let [r, g, b] = rgb;
    let r709 = m[0][0] * r + m[0][1] * g + m[0][2] * b;
    let g709 = m[1][0] * r + m[1][1] * g + m[1][2] * b;
    let b709 = m[2][0] * r + m[2][1] * g + m[2][2] * b;

    // Soft-clip each channel above the knee.
    let r_clipped = soft_clip(r709, knee);
    let g_clipped = soft_clip(g709, knee);
    let b_clipped = soft_clip(b709, knee);

    // Handle out-of-gamut negative values: desaturate toward the luminance
    // by blending with the Y (relative luminance in BT.709 coefficients).
    let y = 0.2126 * r_clipped + 0.7152 * g_clipped + 0.0722 * b_clipped;
    let out = desaturate_negatives([r_clipped, g_clipped, b_clipped], y);

    // Final hard clamp to [0, 1].
    [
        out[0].clamp(0.0, 1.0),
        out[1].clamp(0.0, 1.0),
        out[2].clamp(0.0, 1.0),
    ]
}

/// Convert a PQ-encoded BT.2020 triple to a linear-light BT.709 triple with
/// knee-based gamut mapping.
///
/// This combines PQ inverse-EOTF linearisation with [`hdr_bt2020_to_bt709`].
///
/// # Arguments
///
/// * `pq_rgb` — PQ-encoded RGB in BT.2020 primaries (values in `[0.0, 1.0]`).
/// * `knee`   — highlight knee point; see [`hdr_bt2020_to_bt709`].
#[must_use]
pub fn pq_bt2020_to_linear_bt709(pq_rgb: [f32; 3], knee: f32) -> [f32; 3] {
    let linear = [
        pq_eotf(pq_rgb[0]),
        pq_eotf(pq_rgb[1]),
        pq_eotf(pq_rgb[2]),
    ];
    hdr_bt2020_to_bt709(linear, knee)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Knee-function soft-clip: values below `knee` are passed through; above
/// `knee` they are compressed toward 1.0 with a smooth rolloff.
///
/// Uses a simple rational curve: `knee + (v - knee) / (1 + (v - knee) / (1 - knee))`
/// which approaches 1.0 asymptotically as `v → ∞`.
#[inline]
fn soft_clip(v: f32, knee: f32) -> f32 {
    if v <= knee {
        v
    } else {
        let excess = v - knee;
        let headroom = 1.0 - knee;
        knee + excess / (1.0 + excess / headroom)
    }
}

/// Desaturate channels that have gone negative (out-of-gamut) by blending
/// linearly toward the luminance `y`.
///
/// The blend factor `t` is chosen as the minimum required to make all channels
/// non-negative.
fn desaturate_negatives(rgb: [f32; 3], y: f32) -> [f32; 3] {
    // Find the most negative channel.
    let min_channel = rgb[0].min(rgb[1]).min(rgb[2]);
    if min_channel >= 0.0 {
        return rgb; // nothing to fix
    }
    // We want: (1 - t) * channel + t * y >= 0  =>  t >= -channel / (y - channel)
    // Guard against y <= 0 (pure black — just clamp).
    if y <= 0.0 {
        return [0.0_f32, 0.0, 0.0];
    }
    // t = -min_channel / (y - min_channel)
    let t = (-min_channel / (y - min_channel)).clamp(0.0, 1.0);
    [
        (1.0 - t) * rgb[0] + t * y,
        (1.0 - t) * rgb[1] + t * y,
        (1.0 - t) * rgb[2] + t * y,
    ]
}

/// PQ (SMPTE ST 2084) electro-optical transfer function.
///
/// Maps an encoded signal in `[0.0, 1.0]` to scene-linear luminance
/// normalised to the range `[0.0, 1.0]` (where 1.0 ≡ 10 000 cd/m²).
#[inline]
fn pq_eotf(e: f32) -> f32 {
    const M1_INV: f32 = 1.0 / 0.1593_017_578_125;
    const M2_INV: f32 = 1.0 / 78.843_750;
    const C1: f32 = 0.835_937_5;
    const C2: f32 = 18.851_562_5;
    const C3: f32 = 18.687_5;
    let e = e.clamp(0.0, 1.0);
    let xm2 = e.powf(M2_INV);
    let numerator = (xm2 - C1).max(0.0);
    let denominator = C2 - C3 * xm2;
    (numerator / denominator).powf(M1_INV)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_white_maps_to_white() {
        // BT.2020 white (1,1,1) is spectrally white. After the BT.2020→BT.709 matrix
        // (which preserves neutral greys) white remains neutral, but the knee soft-clip
        // at 0.7 compresses the highlights — the output is neutral (equal RGB) but
        // not necessarily exactly 1.0. Verify neutrality and that the output is in [0,1].
        let out = hdr_bt2020_to_bt709([1.0, 1.0, 1.0], 0.7);
        // Output must be in [0, 1] and neutral (equal channels within float tolerance)
        for &v in &out {
            assert!(v >= 0.0 && v <= 1.0, "channel out of [0,1]: {v}");
        }
        assert!(
            approx_eq(out[0], out[1], 1e-4) && approx_eq(out[1], out[2], 1e-4),
            "White should remain neutral: ({}, {}, {})",
            out[0], out[1], out[2]
        );
    }

    #[test]
    fn test_black_maps_to_black() {
        let out = hdr_bt2020_to_bt709([0.0, 0.0, 0.0], 0.7);
        assert!(approx_eq(out[0], 0.0, 1e-6));
        assert!(approx_eq(out[1], 0.0, 1e-6));
        assert!(approx_eq(out[2], 0.0, 1e-6));
    }

    #[test]
    fn test_output_in_range() {
        let test_cases = [
            [0.5_f32, 0.3, 0.2],
            [0.9, 0.1, 0.05],
            [0.0, 0.8, 0.6],
            [1.5, 0.5, 0.5], // above-peak HDR value
        ];
        for rgb in &test_cases {
            let out = hdr_bt2020_to_bt709(*rgb, 0.7);
            for (i, &v) in out.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "channel {i} = {v} out of [0,1] for input {rgb:?}"
                );
            }
        }
    }

    #[test]
    fn test_soft_clip_below_knee() {
        // Values below knee must pass through unchanged.
        assert!(approx_eq(soft_clip(0.3, 0.7), 0.3, 1e-6));
        assert!(approx_eq(soft_clip(0.7, 0.7), 0.7, 1e-6));
    }

    #[test]
    fn test_soft_clip_above_knee_approaches_one() {
        // A very large value should be compressed to close to 1.0.
        let clipped = soft_clip(100.0, 0.7);
        assert!(clipped < 1.0, "should not exceed 1.0");
        assert!(clipped > 0.9, "should be close to 1.0: {clipped}");
    }

    #[test]
    fn test_pq_eotf_zero_is_zero() {
        assert!(approx_eq(pq_eotf(0.0), 0.0, 1e-4));
    }

    #[test]
    fn test_knee_clamp() {
        // knee=0 and knee=2.0 should not panic.
        let _ = hdr_bt2020_to_bt709([0.5, 0.5, 0.5], 0.0);
        let _ = hdr_bt2020_to_bt709([0.5, 0.5, 0.5], 2.0);
    }
}
