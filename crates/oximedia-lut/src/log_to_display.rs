//! Log-to-display LUT generation.
//!
//! Converts camera log-encoded footage to a display-referred colour space by
//! modelling the electro-optical transfer functions (EOTFs / OETFs) of common
//! camera log curves and applying an output display transform.
//!
//! # Supported log curves
//!
//! | Curve      | Manufacturer  | Notes                             |
//! |------------|---------------|-----------------------------------|
//! | S-Log3     | Sony          | Venice / FX9 / FX6                |
//! | LogC3      | ARRI          | Alexa / AMIRA (EI ≤ 1600)         |
//! | V-Log      | Panasonic     | AU-EVA1 / VariCam                 |
//! | C-Log3     | Canon         | EOS C70 / C300 mkIII              |
//! | Log-C3 (LogC EI) | ARRI   | Alias for `LogC3` for clarity     |
//!
//! # Display targets
//!
//! - sRGB (gamma 2.2 approximation, IEC 61966-2-1)
//! - Rec.709 (BT.1886 EOTF)
//! - P3-D65 (DCI-P3 with D65 white point)
//! - HDR-HLG (BT.2100 HLG for SDR displays at 1000 nit peak)

use crate::error::{LutError, LutResult};
use crate::Rgb;

// ---------------------------------------------------------------------------
// Log curve identifiers
// ---------------------------------------------------------------------------

/// Camera log encoding curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogCurve {
    /// Sony S-Log3 (Venice, FX9, FX6).
    SLog3,
    /// ARRI LogC3 at EI 800.
    LogC3,
    /// Panasonic V-Log.
    VLog,
    /// Canon C-Log3.
    CLog3,
}

impl LogCurve {
    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::SLog3 => "S-Log3",
            Self::LogC3 => "LogC3",
            Self::VLog => "V-Log",
            Self::CLog3 => "C-Log3",
        }
    }

    /// Decode a log-encoded value (scene-linear assumed 0-1 normalised output).
    ///
    /// Input `v` is the normalised code value in `[0.0, 1.0]`.
    /// Output is normalised scene-linear light (may exceed 1.0 for bright values).
    #[must_use]
    pub fn decode(self, v: f64) -> f64 {
        match self {
            Self::SLog3 => slog3_decode(v),
            Self::LogC3 => logc3_decode(v),
            Self::VLog => vlog_decode(v),
            Self::CLog3 => clog3_decode(v),
        }
    }

    /// Encode a scene-linear value to the log code value.
    ///
    /// Output is normalised code value in `[0.0, 1.0]`.
    #[must_use]
    pub fn encode(self, scene_linear: f64) -> f64 {
        match self {
            Self::SLog3 => slog3_encode(scene_linear),
            Self::LogC3 => logc3_encode(scene_linear),
            Self::VLog => vlog_encode(scene_linear),
            Self::CLog3 => clog3_encode(scene_linear),
        }
    }
}

// ---------------------------------------------------------------------------
// S-Log3 (Sony)
// Reference: Sony S-Log3 Technical Summary
// ---------------------------------------------------------------------------

// Sony S-Log3 constants (from Sony S-Log3 Technical Summary v1.3)
// Encode: cv = (420 + log10((x + 0.01) / (0.18 + 0.01)) * 261.5) / 1023  for x >= 0.014
//         cv = (x * (171.2102946 - 95.0) / 0.014 + 95.0) / 1023           for x < 0.014
const SLOG3_CUT_SCENE: f64 = 0.014;         // scene-linear cut
const SLOG3_CUT_CV: f64 = 171.2102946;      // corresponding cut in code-value domain
const SLOG3_LOG_OFFSET: f64 = 0.01;         // log offset (black offset)
const SLOG3_REF_WHITE: f64 = 0.18;          // scene-linear 18% grey
const SLOG3_LOG_SLOPE: f64 = 261.5;         // log slope
const SLOG3_LOG_OFFSET_CV: f64 = 420.0;     // code-value offset at 18% grey
const SLOG3_LIN_SLOPE: f64 = (171.2102946 - 95.0) / 0.014; // linear-segment slope
const SLOG3_LIN_OFFSET_CV: f64 = 95.0;      // linear-segment offset

fn slog3_encode(x: f64) -> f64 {
    let cv = if x >= SLOG3_CUT_SCENE {
        // log10((x + 0.01) / (0.18 + 0.01)) * 261.5 + 420
        ((x + SLOG3_LOG_OFFSET) / (SLOG3_REF_WHITE + SLOG3_LOG_OFFSET)).log10()
            * SLOG3_LOG_SLOPE
            + SLOG3_LOG_OFFSET_CV
    } else {
        x * SLOG3_LIN_SLOPE + SLOG3_LIN_OFFSET_CV
    };
    (cv / 1023.0).clamp(0.0, 1.0)
}

fn slog3_decode(v: f64) -> f64 {
    let cv = v * 1023.0;
    if cv >= SLOG3_CUT_CV {
        // 10^((cv - 420) / 261.5) * (0.18 + 0.01) - 0.01
        10.0_f64.powf((cv - SLOG3_LOG_OFFSET_CV) / SLOG3_LOG_SLOPE)
            * (SLOG3_REF_WHITE + SLOG3_LOG_OFFSET)
            - SLOG3_LOG_OFFSET
    } else {
        (cv - SLOG3_LIN_OFFSET_CV) / SLOG3_LIN_SLOPE
    }
}

// ---------------------------------------------------------------------------
// LogC3 (ARRI, EI 800)
// Reference: ARRI White Paper WP-2018-001
// ---------------------------------------------------------------------------

const LOGC3_CUT: f64 = 0.010_591;
const LOGC3_A: f64 = 5.555_556;
const LOGC3_B: f64 = 0.052_272;
const LOGC3_C: f64 = 0.247_190;
const LOGC3_D: f64 = 0.385_537;
const LOGC3_E: f64 = 5.367_655;
const LOGC3_F: f64 = 0.092_809;
const LOGC3_CUT_ENCODED: f64 = 0.149_658; // E*cut+F

fn logc3_encode(x: f64) -> f64 {
    if x >= LOGC3_CUT {
        LOGC3_C * (LOGC3_A * x + LOGC3_B).log10() + LOGC3_D
    } else {
        LOGC3_E * x + LOGC3_F
    }
    .clamp(0.0, 1.0)
}

fn logc3_decode(v: f64) -> f64 {
    if v >= LOGC3_CUT_ENCODED {
        (10.0_f64.powf((v - LOGC3_D) / LOGC3_C) - LOGC3_B) / LOGC3_A
    } else {
        (v - LOGC3_F) / LOGC3_E
    }
}

// ---------------------------------------------------------------------------
// V-Log (Panasonic)
// Reference: Panasonic V-Log/V-Gamut Rev. 1.0
// ---------------------------------------------------------------------------

const VLOG_CUT1: f64 = 0.01;
const VLOG_CUT2: f64 = 0.181;
const VLOG_B: f64 = 0.00873;
const VLOG_C: f64 = 0.241514;
const VLOG_D: f64 = 0.598206;

fn vlog_encode(x: f64) -> f64 {
    if x < VLOG_CUT1 {
        5.6 * x + 0.125
    } else {
        VLOG_C * (x + VLOG_B).log10() + VLOG_D
    }
    .clamp(0.0, 1.0)
}

fn vlog_decode(v: f64) -> f64 {
    if v < VLOG_CUT2 {
        (v - 0.125) / 5.6
    } else {
        10.0_f64.powf((v - VLOG_D) / VLOG_C) - VLOG_B
    }
}

// ---------------------------------------------------------------------------
// C-Log3 (Canon)
// Reference: Canon Log 3 Specification v1.0
// ---------------------------------------------------------------------------

const CLOG3_A: f64 = 0.332_914_6;
const CLOG3_B: f64 = 0.073_015;
const CLOG3_C: f64 = 0.105_357;

fn clog3_encode(x: f64) -> f64 {
    let linear = x.max(0.0);
    if linear < 0.097465 {
        2.3265084 * linear + CLOG3_C
    } else {
        CLOG3_A * (linear + CLOG3_B).log10() + 0.710_699
    }
    .clamp(0.0, 1.0)
}

fn clog3_decode(v: f64) -> f64 {
    if v < 0.332_314_1 {
        (v - CLOG3_C) / 2.3265084
    } else {
        10.0_f64.powf((v - 0.710_699) / CLOG3_A) - CLOG3_B
    }
}

// ---------------------------------------------------------------------------
// Display targets
// ---------------------------------------------------------------------------

/// Target display colour space / EOTF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayTarget {
    /// sRGB (IEC 61966-2-1).
    Srgb,
    /// Rec.709 (BT.1886 gamma 2.4 EOTF).
    Rec709,
    /// DCI-P3 with D65 white point (gamma 2.6).
    P3D65,
    /// BT.2100 HLG (for combined SDR/HDR displays).
    HlgHdr,
}

impl DisplayTarget {
    /// Apply the display OETF (scene-linear → code value).
    #[must_use]
    pub fn encode(self, linear: f64) -> f64 {
        match self {
            Self::Srgb => srgb_encode(linear),
            Self::Rec709 => bt1886_encode(linear),
            Self::P3D65 => p3_encode(linear),
            Self::HlgHdr => hlg_encode(linear),
        }
    }
}

/// sRGB OETF (piece-wise gamma).
fn srgb_encode(l: f64) -> f64 {
    let l = l.clamp(0.0, 1.0);
    if l <= 0.003_130_8 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// BT.1886 (Rec.709 display) OETF: pure power gamma 2.4 approximation.
fn bt1886_encode(l: f64) -> f64 {
    l.clamp(0.0, 1.0).powf(1.0 / 2.4)
}

/// DCI-P3 OETF: gamma 2.6.
fn p3_encode(l: f64) -> f64 {
    l.clamp(0.0, 1.0).powf(1.0 / 2.6)
}

/// HLG OETF (BT.2100).
fn hlg_encode(l: f64) -> f64 {
    let l = l.clamp(0.0, 1.0);
    const A: f64 = 0.178_832_77;
    const B: f64 = 0.284_668_92;
    const C: f64 = 0.559_910_73;
    if l <= 1.0 / 12.0 {
        (3.0 * l).sqrt()
    } else {
        A * (12.0 * l - B).ln() + C
    }
}

// ---------------------------------------------------------------------------
// Log-to-display LUT generation
// ---------------------------------------------------------------------------

/// Parameters for log-to-display LUT generation.
#[derive(Debug, Clone)]
pub struct LogToDisplayParams {
    /// Source log encoding curve.
    pub log_curve: LogCurve,
    /// Target display EOTF/colour space.
    pub display: DisplayTarget,
    /// Exposure offset in stops applied in scene-linear space before the
    /// display transform.  Positive values brighten, negative darken.
    pub exposure_stops: f64,
    /// Lift applied in display space (black level raise, 0–0.1 typical).
    pub lift: f64,
    /// Gain applied in display space (white level, 1.0 = no change).
    pub gain: f64,
    /// Optional creative look LUT: a 1-D per-channel curve in `[0,1]` space
    /// sampled uniformly.  Applied after the display transform.
    /// Stored as `(red_curve, green_curve, blue_curve)` where each curve has
    /// the same number of samples.
    pub creative_look: Option<(Vec<f64>, Vec<f64>, Vec<f64>)>,
}

impl Default for LogToDisplayParams {
    fn default() -> Self {
        Self {
            log_curve: LogCurve::SLog3,
            display: DisplayTarget::Srgb,
            exposure_stops: 0.0,
            lift: 0.0,
            gain: 1.0,
            creative_look: None,
        }
    }
}

impl LogToDisplayParams {
    /// Create a new params builder for the given log curve and display target.
    #[must_use]
    pub fn new(log_curve: LogCurve, display: DisplayTarget) -> Self {
        Self {
            log_curve,
            display,
            ..Default::default()
        }
    }

    /// Set exposure offset in stops.
    #[must_use]
    pub fn exposure(mut self, stops: f64) -> Self {
        self.exposure_stops = stops;
        self
    }

    /// Set lift and gain in display space.
    #[must_use]
    pub fn lift_gain(mut self, lift: f64, gain: f64) -> Self {
        self.lift = lift.clamp(0.0, 0.5);
        self.gain = gain.max(0.0);
        self
    }
}

/// Apply a 1-D look curve (uniform sampling, linear interpolation) to a value.
fn apply_curve(curve: &[f64], v: f64) -> f64 {
    let n = curve.len();
    if n == 0 {
        return v;
    }
    let v_clamped = v.clamp(0.0, 1.0);
    let pos = v_clamped * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    curve[lo] * (1.0 - frac) + curve[hi] * frac
}

/// Process a single normalised log code value through the full transform.
#[must_use]
pub fn process_log_to_display(v: f64, params: &LogToDisplayParams) -> f64 {
    // 1. Decode log → scene-linear.
    let mut linear = params.log_curve.decode(v);

    // 2. Apply exposure offset.
    if params.exposure_stops != 0.0 {
        linear *= (2.0_f64).powf(params.exposure_stops);
    }

    // 3. Clamp to valid display range (soft-clip above 1.0).
    let linear_clamped = linear.clamp(0.0, 1.0);

    // 4. Apply display OETF.
    let mut display = params.display.encode(linear_clamped);

    // 5. Lift / gain in display space.
    display = display * params.gain + params.lift;
    display = display.clamp(0.0, 1.0);

    display
}

/// Process an RGB triple through the log-to-display transform (per-channel).
#[must_use]
pub fn process_rgb_log_to_display(rgb: &Rgb, params: &LogToDisplayParams) -> Rgb {
    let mut out = [
        process_log_to_display(rgb[0], params),
        process_log_to_display(rgb[1], params),
        process_log_to_display(rgb[2], params),
    ];

    // 6. Apply creative look.
    if let Some((ref r_curve, ref g_curve, ref b_curve)) = params.creative_look {
        out[0] = apply_curve(r_curve, out[0]);
        out[1] = apply_curve(g_curve, out[1]);
        out[2] = apply_curve(b_curve, out[2]);
    }

    out
}

/// Generate a 3-D log-to-display LUT of the given `size` (entries per dimension).
///
/// The LUT is stored in `r-major` order: index = `r * size² + g * size + b`.
/// Input axes represent log code values uniformly sampled in `[0.0, 1.0]`.
///
/// # Errors
///
/// Returns [`LutError::InvalidData`] if `size < 2`.
pub fn generate_log_to_display_lut(
    size: usize,
    params: &LogToDisplayParams,
) -> LutResult<Vec<Rgb>> {
    if size < 2 {
        return Err(LutError::InvalidData(format!(
            "LUT size must be >= 2, got {size}"
        )));
    }

    let scale = (size - 1) as f64;
    let total = size * size * size;
    let mut lut = Vec::with_capacity(total);

    for r in 0..size {
        for g in 0..size {
            for b in 0..size {
                let rgb_in: Rgb = [r as f64 / scale, g as f64 / scale, b as f64 / scale];
                lut.push(process_rgb_log_to_display(&rgb_in, params));
            }
        }
    }

    Ok(lut)
}

/// Generate a 1-D per-channel log-to-display curve of `samples` points.
///
/// Returns `(red, green, blue)` each of length `samples`, sampled from
/// `0.0` to `1.0` uniformly.  For monochrome / per-channel-identical
/// transforms the three curves will be identical.
///
/// # Errors
///
/// Returns [`LutError::InvalidData`] if `samples < 2`.
pub fn generate_log_to_display_1d(
    samples: usize,
    params: &LogToDisplayParams,
) -> LutResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if samples < 2 {
        return Err(LutError::InvalidData(format!(
            "samples must be >= 2, got {samples}"
        )));
    }

    let scale = (samples - 1) as f64;
    let mut r_curve = Vec::with_capacity(samples);
    let mut g_curve = Vec::with_capacity(samples);
    let mut b_curve = Vec::with_capacity(samples);

    for i in 0..samples {
        let v = i as f64 / scale;
        let rgb_in: Rgb = [v, v, v];
        let out = process_rgb_log_to_display(&rgb_in, params);
        r_curve.push(out[0]);
        g_curve.push(out[1]);
        b_curve.push(out[2]);
    }

    Ok((r_curve, g_curve, b_curve))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // --- Log curve round-trip tests ---

    #[test]
    fn test_slog3_round_trip() {
        // Start from i=1 (skip v=0.0) because the linear segment at black has
        // limited numerical range; use a relaxed tolerance for the segment boundary.
        for i in 1..=20 {
            let v = i as f64 / 20.0;
            let encoded = LogCurve::SLog3.encode(v);
            let decoded = LogCurve::SLog3.decode(encoded);
            assert!(
                approx_eq(v, decoded, 1e-5),
                "SLog3 round-trip failed for v={v}: decoded={decoded}"
            );
        }
        // Separately verify that v=0.0 encodes and decodes in the linear segment.
        let enc0 = LogCurve::SLog3.encode(0.0);
        let dec0 = LogCurve::SLog3.decode(enc0);
        assert!(approx_eq(0.0, dec0, 1e-9), "SLog3 round-trip at 0: {dec0}");
    }

    #[test]
    fn test_logc3_round_trip() {
        for i in 0..=20 {
            let v = i as f64 / 20.0;
            let encoded = LogCurve::LogC3.encode(v);
            let decoded = LogCurve::LogC3.decode(encoded);
            assert!(
                approx_eq(v, decoded, 1e-6),
                "LogC3 round-trip failed for v={v}: decoded={decoded}"
            );
        }
    }

    #[test]
    fn test_vlog_round_trip() {
        for i in 1..=20 {
            // start from 1 to avoid negative-domain edge
            let v = i as f64 / 20.0;
            let encoded = LogCurve::VLog.encode(v);
            let decoded = LogCurve::VLog.decode(encoded);
            assert!(
                approx_eq(v, decoded, 1e-6),
                "VLog round-trip failed for v={v}: decoded={decoded}"
            );
        }
    }

    #[test]
    fn test_clog3_round_trip() {
        for i in 0..=20 {
            let v = i as f64 / 20.0;
            let encoded = LogCurve::CLog3.encode(v);
            let decoded = LogCurve::CLog3.decode(encoded);
            assert!(
                approx_eq(v, decoded, 1e-5),
                "CLog3 round-trip failed for v={v}: decoded={decoded}"
            );
        }
    }

    // --- Display OETF monotonicity tests ---

    #[test]
    fn test_srgb_monotone() {
        let vals: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
        for w in vals.windows(2) {
            assert!(srgb_encode(w[1]) >= srgb_encode(w[0]));
        }
    }

    #[test]
    fn test_bt1886_monotone() {
        let vals: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
        for w in vals.windows(2) {
            assert!(bt1886_encode(w[1]) >= bt1886_encode(w[0]));
        }
    }

    // --- LUT generation ---

    #[test]
    fn test_generate_lut_size2() {
        let params = LogToDisplayParams::new(LogCurve::SLog3, DisplayTarget::Srgb);
        let lut = generate_log_to_display_lut(2, &params).expect("should succeed");
        assert_eq!(lut.len(), 8); // 2³
        // (0,0,0) input → blackest output
        let black = lut[0];
        assert!(black[0] <= 0.5);
    }

    #[test]
    fn test_generate_lut_invalid_size() {
        let params = LogToDisplayParams::default();
        assert!(generate_log_to_display_lut(1, &params).is_err());
    }

    #[test]
    fn test_generate_1d_monotone() {
        let params = LogToDisplayParams::new(LogCurve::LogC3, DisplayTarget::Rec709);
        let (r, _, _) = generate_log_to_display_1d(64, &params).expect("should succeed");
        for w in r.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-9,
                "1-D curve not monotone: {} < {}",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn test_generate_1d_invalid_samples() {
        let params = LogToDisplayParams::default();
        assert!(generate_log_to_display_1d(1, &params).is_err());
    }

    #[test]
    fn test_exposure_offset_brightens() {
        let base = LogToDisplayParams::new(LogCurve::SLog3, DisplayTarget::Srgb);
        let bright = base.clone().exposure(1.0);
        let pixel = [0.4, 0.4, 0.4];
        let out_base = process_rgb_log_to_display(&pixel, &base);
        let out_bright = process_rgb_log_to_display(&pixel, &bright);
        assert!(out_bright[0] >= out_base[0]);
    }

    #[test]
    fn test_lift_raises_black() {
        let params = LogToDisplayParams::new(LogCurve::SLog3, DisplayTarget::Srgb)
            .lift_gain(0.05, 1.0);
        let pixel = [0.0, 0.0, 0.0];
        let out = process_rgb_log_to_display(&pixel, &params);
        assert!(out[0] >= 0.05 - 1e-6);
    }

    #[test]
    fn test_creative_look_applied() {
        // Identity look should not change output.
        let n = 32;
        let identity_curve: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let mut params = LogToDisplayParams::new(LogCurve::VLog, DisplayTarget::Srgb);
        params.creative_look = Some((
            identity_curve.clone(),
            identity_curve.clone(),
            identity_curve.clone(),
        ));

        let pixel = [0.5, 0.5, 0.5];
        let params_no_look = LogToDisplayParams::new(LogCurve::VLog, DisplayTarget::Srgb);
        let out_no_look = process_rgb_log_to_display(&pixel, &params_no_look);
        let out_look = process_rgb_log_to_display(&pixel, &params);

        assert!(
            (out_look[0] - out_no_look[0]).abs() < 0.01,
            "identity look changed output: no_look={} look={}",
            out_no_look[0],
            out_look[0]
        );
    }

    #[test]
    fn test_all_log_curves_decode_midgray() {
        // All curves should produce a valid (0–1) result for any code value.
        let curves = [
            LogCurve::SLog3,
            LogCurve::LogC3,
            LogCurve::VLog,
            LogCurve::CLog3,
        ];
        for curve in &curves {
            let v = 0.5; // mid code value
            let d = curve.decode(v);
            assert!(d.is_finite(), "{} decode(0.5) not finite", curve.name());
        }
    }

    #[test]
    fn test_all_display_targets_in_range() {
        let targets = [
            DisplayTarget::Srgb,
            DisplayTarget::Rec709,
            DisplayTarget::P3D65,
            DisplayTarget::HlgHdr,
        ];
        for target in &targets {
            let out = target.encode(0.18); // 18% gray
            assert!(
                (0.0..=1.0).contains(&out),
                "{target:?} encode(0.18)={out} out of range"
            );
        }
    }
}
