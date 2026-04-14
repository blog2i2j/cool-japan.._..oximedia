//! HDR-aware scaling module.
//!
//! Handles PQ (ST 2084) and HLG (ARIB STD-B67) tone mapping during resolution
//! changes. The pipeline is:
//!
//! 1. Linearise: apply inverse EOTF (PQ/HLG) to obtain scene-linear light.
//! 2. Optionally apply a reference white normalisation.
//! 3. Scale the linear image to the target resolution using Lanczos-3.
//! 4. Re-apply the forward EOTF so the output is in the same transfer encoding
//!    as the input.
//!
//! Both transfer functions are implemented from their normative specifications:
//! - PQ: ITU-R BT.2100 / SMPTE ST 2084
//! - HLG: ITU-R BT.2100 Annex 2

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::fmt;

use rayon::prelude::*;

use crate::lanczos::{LanczosResampler, LanczosWindowSize};

// ── PQ (SMPTE ST 2084) constants ─────────────────────────────────────────────

/// PQ: m1 exponent.
const PQ_M1: f64 = 0.159_301_758_125; // 2610 / 16384
/// PQ: m2 exponent.
const PQ_M2: f64 = 78.843_75; // 2523 / 32  (exact)
/// PQ: c1 constant.
const PQ_C1: f64 = 0.835_937_5; // 3424 / 4096
/// PQ: c2 constant.
const PQ_C2: f64 = 18.851_562_5; // 2413 / 128
/// PQ: c3 constant.
const PQ_C3: f64 = 18.687_5; // 2392 / 128

/// Peak luminance of PQ system (cd/m²) — the normalisation reference.
#[allow(dead_code)]
const PQ_PEAK_NITS: f64 = 10_000.0;

// ── HLG (ARIB STD-B67) constants ─────────────────────────────────────────────

/// HLG: a constant.
const HLG_A: f64 = 0.178_832_776_9;
/// HLG: b constant.
const HLG_B: f64 = 0.284_668_917_0;
/// HLG: c constant.
const HLG_C: f64 = 0.559_910_729_5;

// ── Transfer function enum ────────────────────────────────────────────────────

/// High-dynamic-range transfer function used for encoding the pixel values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrTransferFunction {
    /// Perceptual Quantiser (ITU-R BT.2100 / SMPTE ST 2084).
    Pq,
    /// Hybrid Log-Gamma (ITU-R BT.2100 Annex 2).
    Hlg,
}

impl fmt::Display for HdrTransferFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pq => write!(f, "PQ (ST 2084)"),
            Self::Hlg => write!(f, "HLG (BT.2100)"),
        }
    }
}

// ── Tone-mapping operator enum ────────────────────────────────────────────────

/// Operator applied in the linear-light domain when the output peak differs
/// from the input peak (e.g. when downscaling from HDR10 to SDR).
///
/// All operators map the normalised linear range `[0, 1]` where `1 = 10 000 nit`
/// for PQ and `1 = scene luminance of 1` for HLG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneMapOperator {
    /// No tone mapping — linear passthrough (clip values > 1).
    Clip,
    /// Simple Reinhard global operator: `y = x / (1 + x)`.
    Reinhard,
    /// ACES filmic approximation (Narkowicz 2015).
    AcesFilmic,
    /// Hable (Uncharted 2) filmic operator.
    HableFilmic,
}

impl fmt::Display for ToneMapOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clip => write!(f, "Clip"),
            Self::Reinhard => write!(f, "Reinhard"),
            Self::AcesFilmic => write!(f, "ACES Filmic"),
            Self::HableFilmic => write!(f, "Hable Filmic"),
        }
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for HDR-aware scaling.
#[derive(Debug, Clone)]
pub struct HdrScalingConfig {
    /// Transfer function of both the input and output.
    pub transfer: HdrTransferFunction,
    /// Tone-mapping applied in the linear domain (before downscale).
    pub tone_map: ToneMapOperator,
    /// Lanczos window size used for the resize step.
    pub lanczos_window: LanczosWindowSize,
    /// Target width in pixels.
    pub target_width: usize,
    /// Target height in pixels.
    pub target_height: usize,
}

impl HdrScalingConfig {
    /// Create a configuration targeting the given dimensions.
    pub fn new(target_width: usize, target_height: usize) -> Self {
        Self {
            transfer: HdrTransferFunction::Pq,
            tone_map: ToneMapOperator::Reinhard,
            lanczos_window: LanczosWindowSize::Tap3,
            target_width,
            target_height,
        }
    }

    /// Set the HDR transfer function.
    pub fn with_transfer(mut self, transfer: HdrTransferFunction) -> Self {
        self.transfer = transfer;
        self
    }

    /// Set the tone-mapping operator.
    pub fn with_tone_map(mut self, op: ToneMapOperator) -> Self {
        self.tone_map = op;
        self
    }

    /// Set the Lanczos window size.
    pub fn with_lanczos_window(mut self, window: LanczosWindowSize) -> Self {
        self.lanczos_window = window;
        self
    }
}

// ── PQ transfer functions ─────────────────────────────────────────────────────

/// Apply the **inverse** PQ EOTF: convert a normalised PQ code value `e` in
/// `[0, 1]` to linear scene luminance normalised to `[0, 1]` where
/// `1 ≡ 10 000 cd/m²`.
///
/// Reference: ITU-R BT.2100, Table 4.
#[must_use]
pub fn pq_eotf_inverse(e: f64) -> f64 {
    // The EOTF is: L = (max(e^(1/m2) − c1, 0) / (c2 − c3·e^(1/m2)))^(1/m1)
    // Inverse (OETF): E = ((c1 + c2·L^m1) / (1 + c3·L^m1))^m2
    let lm1 = e.max(0.0).powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * lm1;
    let den = 1.0 + PQ_C3 * lm1;
    (num / den).powf(PQ_M2)
}

/// Apply the **forward** PQ OETF: convert a normalised linear luminance `l`
/// (where `1 ≡ 10 000 cd/m²`) to a PQ code value in `[0, 1]`.
#[must_use]
pub fn pq_oetf(l: f64) -> f64 {
    // EOTF: L = (max(E^(1/m2) - c1, 0) / (c2 - c3·E^(1/m2)))^(1/m1)
    // We want OETF which is the inverse.
    let lm1 = l.max(0.0).powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * lm1;
    let den = 1.0 + PQ_C3 * lm1;
    (num / den).powf(PQ_M2)
}

// ── HLG transfer functions ────────────────────────────────────────────────────

/// Apply the **inverse** HLG OETF: convert a normalised HLG signal `e` in
/// `[0, 1]` to a scene-linear value in `[0, 1]`.
///
/// Reference: ITU-R BT.2100, Table 5.
#[must_use]
pub fn hlg_oetf_inverse(e: f64) -> f64 {
    let e = e.clamp(0.0, 1.0);
    if e <= 0.5 {
        (e * e) / 3.0
    } else {
        (((e - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    }
}

/// Apply the **forward** HLG OETF: convert a normalised scene-linear value
/// `s` to a normalised HLG signal in `[0, 1]`.
#[must_use]
pub fn hlg_oetf(s: f64) -> f64 {
    let s = s.max(0.0);
    if s <= 1.0 / 12.0 {
        (3.0 * s).sqrt()
    } else {
        HLG_A * (12.0 * s - HLG_B).ln() + HLG_C
    }
}

// ── Tone-mapping operators ────────────────────────────────────────────────────

/// Apply the selected `ToneMapOperator` to a normalised linear luminance.
///
/// Input `x` is normalised linear with `1 = 10 000 nit` (PQ) or `1 = max
/// scene luminance` (HLG). Returns a value in `[0, 1]`.
#[must_use]
pub fn apply_tone_map(x: f64, op: ToneMapOperator) -> f64 {
    match op {
        ToneMapOperator::Clip => x.clamp(0.0, 1.0),
        ToneMapOperator::Reinhard => x / (1.0 + x),
        ToneMapOperator::AcesFilmic => {
            // Narkowicz 2015 ACES approximation
            let a = 2.51;
            let b = 0.03;
            let c = 2.43;
            let d = 0.59;
            let e = 0.14;
            ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
        }
        ToneMapOperator::HableFilmic => {
            // Hable "Uncharted 2" operator
            let hable = |v: f64| -> f64 {
                let a = 0.15;
                let b = 0.50;
                let c = 0.10;
                let d = 0.20;
                let e_c = 0.02;
                let f = 0.30;
                (v * (a * v + c * b) + d * e_c) / (v * (a * v + b) + d * f) - e_c / f
            };
            let w = 11.2; // white point
            (hable(x * 2.0) / hable(w)).clamp(0.0, 1.0)
        }
    }
}

// ── HDR scaler ────────────────────────────────────────────────────────────────

/// HDR-aware image scaler.
///
/// Performs a three-stage pipeline:
/// 1. Inverse EOTF → linear light.
/// 2. Tone map in linear domain (optional, useful for SDR output).
/// 3. Resize with Lanczos resampling.
/// 4. Forward EOTF → re-encode.
///
/// All pixel values are treated as normalised `[0, 1]` f32 channels.
#[derive(Debug)]
pub struct HdrScaler {
    config: HdrScalingConfig,
}

impl HdrScaler {
    /// Create a new `HdrScaler` with the given configuration.
    pub fn new(config: HdrScalingConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &HdrScalingConfig {
        &self.config
    }

    /// Scale a single-channel (grayscale) HDR image stored as `f32` values in
    /// `[0, 1]`.
    ///
    /// The pipeline:
    /// 1. Convert from the encoded domain to linear light via the inverse EOTF.
    /// 2. Apply the configured tone-mapping operator.
    /// 3. Resize from `(src_w × src_h)` to `(target_w × target_h)` using
    ///    Lanczos resampling (rayon-parallel rows).
    /// 4. Re-encode to the same transfer function via the forward EOTF.
    ///
    /// Returns `None` if the pixel buffer is too small.
    pub fn scale_f32_gray(
        &self,
        pixels: &[f32],
        src_w: usize,
        src_h: usize,
    ) -> Option<(Vec<f32>, usize, usize)> {
        if pixels.len() < src_w * src_h {
            return None;
        }
        let dst_w = self.config.target_width;
        let dst_h = self.config.target_height;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }

        let transfer = self.config.transfer;
        let tone_map_op = self.config.tone_map;

        // 1. Inverse EOTF + tone mapping → linear light
        let linear: Vec<f32> = pixels
            .iter()
            .map(|&v| {
                let lin = inverse_eotf(v as f64, transfer);
                let mapped = apply_tone_map(lin, tone_map_op);
                mapped as f32
            })
            .collect();

        // 2. Lanczos resize using parallel rows
        let resampler = LanczosResampler::from_window_size(self.config.lanczos_window);
        let linear_u8: Vec<u8> = linear
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();

        let resized_u8 =
            lanczos_resize_parallel(&resampler, &linear_u8, src_w, src_h, dst_w, dst_h);

        // 3. Forward EOTF → encoded output
        let output: Vec<f32> = resized_u8
            .iter()
            .map(|&v| {
                let lin = v as f64 / 255.0;
                forward_eotf(lin, transfer) as f32
            })
            .collect();

        Some((output, dst_w, dst_h))
    }

    /// Scale a packed RGB HDR image stored as interleaved `[R, G, B]` `f32`
    /// triplets with values in `[0, 1]`.
    ///
    /// Returns `None` if the pixel buffer is too small (`< src_w * src_h * 3`).
    pub fn scale_f32_rgb(
        &self,
        pixels: &[f32],
        src_w: usize,
        src_h: usize,
    ) -> Option<(Vec<f32>, usize, usize)> {
        if pixels.len() < src_w * src_h * 3 {
            return None;
        }
        let dst_w = self.config.target_width;
        let dst_h = self.config.target_height;
        if dst_w == 0 || dst_h == 0 {
            return None;
        }

        let transfer = self.config.transfer;
        let tone_map_op = self.config.tone_map;

        // 1. Per-channel inverse EOTF + tone mapping
        let count = src_w * src_h;
        let mut linear_u8 = vec![0u8; count * 3];
        for i in 0..count {
            for c in 0..3 {
                let encoded = pixels[i * 3 + c] as f64;
                let lin = inverse_eotf(encoded, transfer);
                let mapped = apply_tone_map(lin, tone_map_op);
                linear_u8[i * 3 + c] = (mapped.clamp(0.0, 1.0) * 255.0) as u8;
            }
        }

        // 2. Lanczos resize with parallel rows per channel
        let resized_u8 = lanczos_resize_rgb_parallel(
            &LanczosResampler::from_window_size(self.config.lanczos_window),
            &linear_u8,
            src_w,
            src_h,
            dst_w,
            dst_h,
        );

        // 3. Forward EOTF → re-encode
        let out_count = dst_w * dst_h;
        let mut output = vec![0.0f32; out_count * 3];
        for i in 0..out_count {
            for c in 0..3 {
                let lin = resized_u8[i * 3 + c] as f64 / 255.0;
                output[i * 3 + c] = forward_eotf(lin, transfer) as f32;
            }
        }

        Some((output, dst_w, dst_h))
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Apply the inverse EOTF (encoded → linear light) for the given transfer function.
fn inverse_eotf(e: f64, tf: HdrTransferFunction) -> f64 {
    match tf {
        HdrTransferFunction::Pq => {
            // PQ EOTF: L = (max(E^(1/m2) - c1, 0) / (c2 - c3·E^(1/m2)))^(1/m1)
            // Normalise to [0, 1] where 1 = 10 000 nit.
            let e_m2 = e.max(0.0).powf(1.0 / PQ_M2);
            let num = (e_m2 - PQ_C1).max(0.0);
            let den = (PQ_C2 - PQ_C3 * e_m2).max(f64::EPSILON);
            (num / den).powf(1.0 / PQ_M1)
        }
        HdrTransferFunction::Hlg => hlg_oetf_inverse(e),
    }
}

/// Apply the forward EOTF (linear light → encoded) for the given transfer function.
fn forward_eotf(l: f64, tf: HdrTransferFunction) -> f64 {
    match tf {
        HdrTransferFunction::Pq => pq_oetf(l),
        HdrTransferFunction::Hlg => hlg_oetf(l),
    }
}

/// Resize a single-channel image using Lanczos with rayon-parallel horizontal rows.
///
/// Two-pass: horizontal resize first (parallel over rows), then vertical resize
/// (parallel over columns).
fn lanczos_resize_parallel(
    resampler: &LanczosResampler,
    pixels: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    if pixels.is_empty() || dst_w == 0 || dst_h == 0 {
        return Vec::new();
    }

    let src_f32: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();

    // Horizontal pass (parallel over rows)
    let h_pass: Vec<f32> = (0..src_h)
        .into_par_iter()
        .flat_map_iter(|row| {
            let src_row = &src_f32[row * src_w..(row + 1) * src_w];
            resampler.resample_1d(src_row, dst_w).into_iter()
        })
        .collect();

    // Vertical pass (parallel over columns)
    let mut result = vec![0u8; dst_w * dst_h];
    let col_results: Vec<Vec<u8>> = (0..dst_w)
        .into_par_iter()
        .map(|col| {
            let col_data: Vec<f32> = (0..src_h).map(|row| h_pass[row * dst_w + col]).collect();
            let resampled = resampler.resample_1d(&col_data, dst_h);
            resampled
                .iter()
                .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
                .collect()
        })
        .collect();

    for (col, col_data) in col_results.iter().enumerate() {
        for (row, &val) in col_data.iter().enumerate() {
            result[row * dst_w + col] = val;
        }
    }

    result
}

/// Resize a packed RGB image using Lanczos with rayon-parallel processing.
///
/// Splits into three independent channels, resizes each in parallel, then
/// interleaves back.
fn lanczos_resize_rgb_parallel(
    resampler: &LanczosResampler,
    pixels: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    if pixels.is_empty() || dst_w == 0 || dst_h == 0 {
        return Vec::new();
    }

    let count = src_w * src_h;
    let channels: Vec<Vec<u8>> = (0..3usize)
        .into_par_iter()
        .map(|c| {
            let channel: Vec<u8> = (0..count).map(|i| pixels[i * 3 + c]).collect();
            lanczos_resize_parallel(resampler, &channel, src_w, src_h, dst_w, dst_h)
        })
        .collect();

    let out_count = dst_w * dst_h;
    let mut result = vec![0u8; out_count * 3];
    for i in 0..out_count {
        for c in 0..3 {
            result[i * 3 + c] = channels[c][i];
        }
    }
    result
}

/// Convenience: scale a `u8` grayscale image through the HDR pipeline.
///
/// Pixel values are treated as normalised HDR-encoded values where `255 ≡ 1.0`.
pub fn hdr_scale_gray(
    pixels: &[u8],
    src_w: usize,
    src_h: usize,
    config: &HdrScalingConfig,
) -> Option<(Vec<u8>, usize, usize)> {
    let scaler = HdrScaler::new(config.clone());
    let f32_pixels: Vec<f32> = pixels.iter().map(|&v| v as f32 / 255.0).collect();
    let (output, w, h) = scaler.scale_f32_gray(&f32_pixels, src_w, src_h)?;
    let u8_output: Vec<u8> = output
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    Some((u8_output, w, h))
}

/// Convenience: scale a `u8` packed RGB image through the HDR pipeline.
pub fn hdr_scale_rgb(
    pixels: &[u8],
    src_w: usize,
    src_h: usize,
    config: &HdrScalingConfig,
) -> Option<(Vec<u8>, usize, usize)> {
    let scaler = HdrScaler::new(config.clone());
    let f32_pixels: Vec<f32> = pixels.iter().map(|&v| v as f32 / 255.0).collect();
    let (output, w, h) = scaler.scale_f32_rgb(&f32_pixels, src_w, src_h)?;
    let u8_output: Vec<u8> = output
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    Some((u8_output, w, h))
}

// ── Re-export kernel helpers used by tests ────────────────────────────────────
pub use crate::lanczos::LanczosKernel as LanczosKernelRef;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Transfer function round-trips ────────────────────────────────────────

    #[test]
    fn test_pq_round_trip_midpoint() {
        // Encode and decode a mid-grey value; error should be < 0.001
        for v in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let encoded = pq_oetf(v);
            let decoded = inverse_eotf(encoded, HdrTransferFunction::Pq);
            assert!(
                (decoded - v).abs() < 1e-3,
                "PQ round-trip failed at {v}: got {decoded}"
            );
        }
    }

    #[test]
    fn test_hlg_round_trip_midpoint() {
        for v in [0.05, 0.15, 0.3, 0.5, 0.8] {
            let encoded = hlg_oetf(v);
            let decoded = hlg_oetf_inverse(encoded);
            assert!(
                (decoded - v).abs() < 1e-6,
                "HLG round-trip failed at {v}: got {decoded}"
            );
        }
    }

    #[test]
    fn test_pq_eotf_inverse_at_zero() {
        let v = pq_eotf_inverse(0.0);
        assert!(v >= 0.0 && v <= 0.01, "PQ at 0 should be ~0, got {v}");
    }

    #[test]
    fn test_hlg_oetf_at_zero() {
        let v = hlg_oetf(0.0);
        assert!((v - 0.0).abs() < 1e-9, "HLG at 0 should be 0, got {v}");
    }

    #[test]
    fn test_hlg_oetf_continuity_at_transition() {
        // At s = 1/12, both branches should be continuous
        let s = 1.0 / 12.0;
        let below = hlg_oetf(s - 1e-7);
        let above = hlg_oetf(s + 1e-7);
        assert!(
            (below - above).abs() < 1e-4,
            "HLG discontinuity at 1/12: {below} vs {above}"
        );
    }

    // ── Tone-mapping operators ────────────────────────────────────────────────

    #[test]
    fn test_tone_map_clip() {
        assert!((apply_tone_map(0.5, ToneMapOperator::Clip) - 0.5).abs() < 1e-9);
        assert!((apply_tone_map(2.0, ToneMapOperator::Clip) - 1.0).abs() < 1e-9);
        assert!((apply_tone_map(-0.1, ToneMapOperator::Clip)).abs() < 1e-9);
    }

    #[test]
    fn test_tone_map_reinhard_at_one() {
        let v = apply_tone_map(1.0, ToneMapOperator::Reinhard);
        assert!((v - 0.5).abs() < 1e-9, "Reinhard(1) should be 0.5, got {v}");
    }

    #[test]
    fn test_tone_map_reinhard_zero() {
        let v = apply_tone_map(0.0, ToneMapOperator::Reinhard);
        assert!(v.abs() < 1e-9, "Reinhard(0) should be 0, got {v}");
    }

    #[test]
    fn test_tone_map_aces_range() {
        for x in [0.0, 0.25, 0.5, 1.0, 2.0, 10.0] {
            let v = apply_tone_map(x, ToneMapOperator::AcesFilmic);
            assert!(v >= 0.0 && v <= 1.0, "ACES({x}) = {v} out of [0, 1]");
        }
    }

    #[test]
    fn test_tone_map_hable_range() {
        for x in [0.0, 0.5, 1.0, 5.0] {
            let v = apply_tone_map(x, ToneMapOperator::HableFilmic);
            assert!(v >= 0.0 && v <= 1.0, "Hable({x}) = {v} out of [0, 1]");
        }
    }

    #[test]
    fn test_tone_map_monotonic() {
        // All operators should be monotonically increasing for positive inputs
        let xs = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0];
        for &op in &[
            ToneMapOperator::Clip,
            ToneMapOperator::Reinhard,
            ToneMapOperator::AcesFilmic,
            ToneMapOperator::HableFilmic,
        ] {
            let vals: Vec<f64> = xs.iter().map(|&x| apply_tone_map(x, op)).collect();
            for w in vals.windows(2) {
                assert!(
                    w[1] >= w[0] - 1e-9,
                    "{op} is not monotonic: {:.4} > {:.4}",
                    w[0],
                    w[1]
                );
            }
        }
    }

    // ── HDR scaling pipeline ──────────────────────────────────────────────────

    #[test]
    fn test_hdr_scaler_config_builder() {
        let cfg = HdrScalingConfig::new(1920, 1080)
            .with_transfer(HdrTransferFunction::Hlg)
            .with_tone_map(ToneMapOperator::AcesFilmic)
            .with_lanczos_window(LanczosWindowSize::Tap4);
        assert_eq!(cfg.target_width, 1920);
        assert_eq!(cfg.target_height, 1080);
        assert_eq!(cfg.transfer, HdrTransferFunction::Hlg);
        assert_eq!(cfg.tone_map, ToneMapOperator::AcesFilmic);
        assert_eq!(cfg.lanczos_window, LanczosWindowSize::Tap4);
    }

    #[test]
    fn test_scale_f32_gray_output_size() {
        let pixels = vec![0.5f32; 8 * 8];
        let cfg = HdrScalingConfig::new(4, 4);
        let scaler = HdrScaler::new(cfg);
        let result = scaler.scale_f32_gray(&pixels, 8, 8);
        assert!(result.is_some());
        let (out, w, h) = result.expect("scale should succeed");
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(out.len(), 4 * 4);
    }

    #[test]
    fn test_scale_f32_rgb_output_size() {
        let pixels = vec![0.5f32; 8 * 8 * 3];
        let cfg = HdrScalingConfig::new(4, 4);
        let scaler = HdrScaler::new(cfg);
        let result = scaler.scale_f32_rgb(&pixels, 8, 8);
        assert!(result.is_some());
        let (out, w, h) = result.expect("scale should succeed");
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(out.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_scale_f32_gray_values_in_range() {
        let pixels: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
        let cfg = HdrScalingConfig::new(4, 4);
        let scaler = HdrScaler::new(cfg);
        let (out, _, _) = scaler
            .scale_f32_gray(&pixels, 8, 8)
            .expect("should succeed");
        for &v in &out {
            assert!(v >= 0.0 && v <= 1.0, "output {v} out of [0, 1]");
        }
    }

    #[test]
    fn test_scale_f32_rgb_values_in_range() {
        let pixels: Vec<f32> = (0..192).map(|i| (i % 64) as f32 / 63.0).collect();
        let cfg = HdrScalingConfig::new(4, 4);
        let scaler = HdrScaler::new(cfg);
        let (out, _, _) = scaler.scale_f32_rgb(&pixels, 8, 8).expect("should succeed");
        for &v in &out {
            assert!(v >= 0.0 && v <= 1.0, "output {v} out of [0, 1]");
        }
    }

    #[test]
    fn test_scale_f32_gray_invalid_buffer() {
        let pixels = vec![0.5f32; 4]; // too small
        let cfg = HdrScalingConfig::new(4, 4);
        let scaler = HdrScaler::new(cfg);
        assert!(scaler.scale_f32_gray(&pixels, 8, 8).is_none());
    }

    #[test]
    fn test_scale_f32_rgb_invalid_buffer() {
        let pixels = vec![0.5f32; 10]; // too small for 4*4*3
        let cfg = HdrScalingConfig::new(4, 4);
        let scaler = HdrScaler::new(cfg);
        assert!(scaler.scale_f32_rgb(&pixels, 3, 3).is_none());
    }

    #[test]
    fn test_hdr_scale_gray_convenience() {
        let pixels = vec![128u8; 8 * 8];
        let cfg = HdrScalingConfig::new(4, 4);
        let result = hdr_scale_gray(&pixels, 8, 8, &cfg);
        assert!(result.is_some());
        let (out, w, h) = result.expect("should succeed");
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_hdr_scale_rgb_convenience() {
        let pixels = vec![128u8; 8 * 8 * 3];
        let cfg = HdrScalingConfig::new(4, 4);
        let result = hdr_scale_rgb(&pixels, 8, 8, &cfg);
        assert!(result.is_some());
        let (out, w, h) = result.expect("should succeed");
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(out.len(), 48);
    }

    #[test]
    fn test_transfer_function_display() {
        assert_eq!(HdrTransferFunction::Pq.to_string(), "PQ (ST 2084)");
        assert_eq!(HdrTransferFunction::Hlg.to_string(), "HLG (BT.2100)");
    }

    #[test]
    fn test_tone_map_operator_display() {
        assert_eq!(ToneMapOperator::Clip.to_string(), "Clip");
        assert_eq!(ToneMapOperator::Reinhard.to_string(), "Reinhard");
        assert_eq!(ToneMapOperator::AcesFilmic.to_string(), "ACES Filmic");
        assert_eq!(ToneMapOperator::HableFilmic.to_string(), "Hable Filmic");
    }

    #[test]
    fn test_pq_hlg_uniform_image_preserves_tone() {
        // A uniform image scaled through both PQ and HLG should stay roughly uniform.
        for &tf in &[HdrTransferFunction::Pq, HdrTransferFunction::Hlg] {
            let pixels = vec![0.6f32; 8 * 8];
            let cfg = HdrScalingConfig::new(4, 4)
                .with_transfer(tf)
                .with_tone_map(ToneMapOperator::Clip);
            let scaler = HdrScaler::new(cfg);
            let (out, _, _) = scaler
                .scale_f32_gray(&pixels, 8, 8)
                .expect("should succeed");
            let mean = out.iter().sum::<f32>() / out.len() as f32;
            assert!(
                (mean - 0.6).abs() < 0.05,
                "{tf}: uniform image mean {mean} deviates from 0.6"
            );
        }
    }

    #[test]
    fn test_hdr_scaling_upscale() {
        let pixels = vec![0.5f32; 4 * 4];
        let cfg = HdrScalingConfig::new(8, 8);
        let scaler = HdrScaler::new(cfg);
        let result = scaler.scale_f32_gray(&pixels, 4, 4);
        assert!(result.is_some());
        let (out, w, h) = result.expect("should succeed");
        assert_eq!(w, 8);
        assert_eq!(h, 8);
        assert_eq!(out.len(), 64);
    }
}
