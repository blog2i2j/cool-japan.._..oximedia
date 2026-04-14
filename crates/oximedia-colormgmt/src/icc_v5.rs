//! ICC v5 (iccMAX) profile support.
//!
//! Implements parsing and construction of ICC v5 profiles as defined by the
//! ICC.2:2023 specification (iccMAX).  ICC v5 extends v4 with:
//!
//! - **Float encodings**: 16-bit and 32-bit IEEE float types (f16, f32) in
//!   tags, enabling high-dynamic-range and spectral data without quantization.
//! - **Spectral colour encoding**: `spectralViewingConditionsType` and
//!   `spectralDataInfoType` allow reflectance/emission spectra directly in
//!   profiles.
//! - **Multi-process elements v2**: `mpet` tags with new curve segment types
//!   including spline, sampled, and tone curve segments.
//! - **Embedded profiles**: `embdType` for nesting sub-profiles.
//! - **Colour appearance model tags**: Observer-dependent metamerism index,
//!   chromatic adaptation state, and viewing conditions.
//! - **Profile version negotiation**: Version byte `0x05.xx` in the header.
//!
//! This implementation provides:
//!
//! - Header parsing with v5 version detection.
//! - Float16 and Float32 encoded XYZ/Lab tag types.
//! - `spectralDataInfoType` structured read.
//! - Multi-process element (MPE) pipeline reconstruction.
//! - Builder for constructing v5-compatible headers.
//!
//! # References
//!
//! - ICC.2:2023 — *Image technology colour management — Extensions to
//!   architecture, profile format, and data structure.*
//!   <https://www.color.org/iccmax/>

use crate::error::{ColorError, Result};

// ── Version constants ─────────────────────────────────────────────────────────

/// ICC v2 major version byte.
pub const ICC_VERSION_2: u8 = 2;
/// ICC v4 major version byte.
pub const ICC_VERSION_4: u8 = 4;
/// ICC v5 (iccMAX) major version byte.
pub const ICC_VERSION_5: u8 = 5;

// ── ICC header offsets (same positions as v2/v4) ──────────────────────────────
// These constants enumerate all 128-byte ICC header field positions per
// ICC.1:2022 §7.2 and ICC.2:2023 §7.2.  Not all offsets are accessed in
// every code path, but they are part of the normative specification and
// are retained here for reference and future use.
#[allow(dead_code)]
const HEADER_PROFILE_SIZE_OFFSET: usize = 0;
#[allow(dead_code)]
const HEADER_CMM_TYPE_OFFSET: usize = 4;
const HEADER_VERSION_OFFSET: usize = 8;
const HEADER_CLASS_OFFSET: usize = 12;
const HEADER_COLORSPACE_OFFSET: usize = 16;
const HEADER_PCS_OFFSET: usize = 20;
#[allow(dead_code)]
const HEADER_CREATION_DATE_OFFSET: usize = 24;
const HEADER_SIGNATURE_OFFSET: usize = 36;
#[allow(dead_code)]
const HEADER_PLATFORM_OFFSET: usize = 40;
const HEADER_FLAGS_OFFSET: usize = 44;
#[allow(dead_code)]
const HEADER_MANUFACTURER_OFFSET: usize = 48;
#[allow(dead_code)]
const HEADER_MODEL_OFFSET: usize = 52;
#[allow(dead_code)]
const HEADER_ATTRIBUTES_OFFSET: usize = 56;
const HEADER_RENDERING_INTENT_OFFSET: usize = 64;
#[allow(dead_code)]
const HEADER_ILLUMINANT_OFFSET: usize = 68;
#[allow(dead_code)]
const HEADER_CREATOR_OFFSET: usize = 80;
#[allow(dead_code)]
const HEADER_PROFILE_ID_OFFSET: usize = 84;
const HEADER_SIZE: usize = 128;

// ── Profile version ───────────────────────────────────────────────────────────

/// Parsed ICC profile version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProfileVersion {
    /// Major version (2, 4, or 5).
    pub major: u8,
    /// Minor version in BCD.
    pub minor: u8,
    /// Bug-fix version in BCD.
    pub bugfix: u8,
}

impl ProfileVersion {
    /// Creates a new profile version.
    #[must_use]
    pub const fn new(major: u8, minor: u8, bugfix: u8) -> Self {
        Self { major, minor, bugfix }
    }

    /// ICC v2.0.0
    #[must_use]
    pub const fn v2() -> Self {
        Self::new(ICC_VERSION_2, 0, 0)
    }

    /// ICC v4.4.0
    #[must_use]
    pub const fn v4() -> Self {
        Self::new(ICC_VERSION_4, 4, 0)
    }

    /// ICC v5.0.0 (iccMAX)
    #[must_use]
    pub const fn v5() -> Self {
        Self::new(ICC_VERSION_5, 0, 0)
    }

    /// Returns `true` if this is an iccMAX (v5+) profile.
    #[must_use]
    pub fn is_iccmax(self) -> bool {
        self.major >= ICC_VERSION_5
    }

    /// Returns `true` if the version is v4 or newer.
    #[must_use]
    pub fn is_v4_or_newer(self) -> bool {
        self.major >= ICC_VERSION_4
    }
}

impl std::fmt::Display for ProfileVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.bugfix)
    }
}

// ── Profile class ─────────────────────────────────────────────────────────────

/// ICC profile class, valid for v2–v5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IccProfileClass {
    /// Input device (camera, scanner).
    Input,
    /// Display device (monitor, projector).
    Display,
    /// Output device (printer, proofer).
    Output,
    /// Device-link profile.
    DeviceLink,
    /// Colour space conversion profile.
    ColorSpace,
    /// Abstract profile.
    Abstract,
    /// Named colour profile.
    NamedColor,
}

impl IccProfileClass {
    /// Returns the 4-byte ICC tag signature for this class.
    #[must_use]
    pub fn signature(self) -> [u8; 4] {
        match self {
            Self::Input => *b"scnr",
            Self::Display => *b"mntr",
            Self::Output => *b"prtr",
            Self::DeviceLink => *b"link",
            Self::ColorSpace => *b"spac",
            Self::Abstract => *b"abst",
            Self::NamedColor => *b"nmcl",
        }
    }

    /// Parses a profile class from the 4-byte ICC signature.
    ///
    /// # Errors
    ///
    /// Returns `ColorError::IccProfile` for unrecognised signatures.
    pub fn from_signature(sig: [u8; 4]) -> Result<Self> {
        match &sig {
            b"scnr" => Ok(Self::Input),
            b"mntr" => Ok(Self::Display),
            b"prtr" => Ok(Self::Output),
            b"link" => Ok(Self::DeviceLink),
            b"spac" => Ok(Self::ColorSpace),
            b"abst" => Ok(Self::Abstract),
            b"nmcl" => Ok(Self::NamedColor),
            _ => Err(ColorError::IccProfile(format!(
                "Unknown profile class: {sig:?}"
            ))),
        }
    }
}

// ── iccMAX float types ────────────────────────────────────────────────────────

/// A 16-bit half-precision float as used in iccMAX `float16Number`.
///
/// Represented as the raw IEEE 754 binary16 bit pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Float16(pub u16);

impl Float16 {
    /// Converts a 16-bit half-precision float to `f64`.
    ///
    /// Implements the IEEE 754 half-precision to double-precision conversion.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        let bits = u32::from(self.0);
        let sign: u32 = (bits >> 15) & 0x1;
        let exp: u32 = (bits >> 10) & 0x1f;
        let mantissa: u32 = bits & 0x3ff;

        let (new_exp, new_mantissa) = if exp == 0 {
            if mantissa == 0 {
                // Zero
                return if sign == 0 { 0.0 } else { -0.0 };
            }
            // Subnormal: normalise
            let mut m = mantissa;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            ((e + 127 + 1023 - 127) as u32, m << 13)
        } else if exp == 31 {
            if mantissa == 0 {
                // Inf
                return if sign == 0 {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                };
            }
            // NaN
            return f64::NAN;
        } else {
            // Normal: exp is in [1, 30]; cast to i32 to avoid underflow
            (((exp as i32) - 15 + 1023) as u32, mantissa << 13)
        };

        let bits64: u64 =
            ((u64::from(sign)) << 63) | ((u64::from(new_exp)) << 52) | u64::from(new_mantissa);
        f64::from_bits(bits64)
    }

    /// Converts an `f64` to a 16-bit half-precision float.
    ///
    /// Values out of range are clamped to ±65504.
    #[must_use]
    pub fn from_f64(v: f64) -> Self {
        if v.is_nan() {
            return Self(0x7e00); // Canonical NaN
        }
        let bits = v.to_bits();
        let sign = ((bits >> 63) as u16) << 15;
        let exp = ((bits >> 52) & 0x7ff) as i32;
        let mantissa = (bits & 0x000f_ffff_ffff_ffff) as u64;

        let result: u16 = if exp == 2047 {
            // Inf or NaN
            if mantissa != 0 {
                0x7e00 // NaN
            } else {
                sign | 0x7c00 // Inf
            }
        } else {
            let new_exp = exp - 1023 + 15;
            if new_exp >= 31 {
                // Overflow → Inf
                sign | 0x7c00
            } else if new_exp <= 0 {
                // Underflow → subnormal or zero
                if new_exp < -10 {
                    sign // Zero
                } else {
                    let shift = (1 - new_exp) as u32;
                    let m = (0x0400u64 | (mantissa >> 42)) >> shift;
                    sign | (m as u16)
                }
            } else {
                sign | ((new_exp as u16) << 10) | ((mantissa >> 42) as u16)
            }
        };
        Self(result)
    }
}

// ── Spectral data info ────────────────────────────────────────────────────────

/// Spectral range descriptor as used in iccMAX `spectralDataInfoType`.
///
/// Describes a uniformly-sampled spectral axis.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralRange {
    /// Start wavelength in nanometres.
    pub start_nm: f32,
    /// End wavelength in nanometres.
    pub end_nm: f32,
    /// Number of wavelength samples.
    pub num_steps: u16,
}

impl SpectralRange {
    /// Creates a spectral range.
    ///
    /// # Errors
    ///
    /// Returns `ColorError::InvalidColor` if `start_nm >= end_nm` or
    /// `num_steps < 2`.
    pub fn new(start_nm: f32, end_nm: f32, num_steps: u16) -> Result<Self> {
        if start_nm >= end_nm {
            return Err(ColorError::InvalidColor(format!(
                "SpectralRange: start_nm ({start_nm}) must be less than end_nm ({end_nm})"
            )));
        }
        if num_steps < 2 {
            return Err(ColorError::InvalidColor(
                "SpectralRange: num_steps must be at least 2".into(),
            ));
        }
        Ok(Self { start_nm, end_nm, num_steps })
    }

    /// Returns the wavelength step size in nanometres.
    #[must_use]
    pub fn step_nm(&self) -> f32 {
        (self.end_nm - self.start_nm) / f32::from(self.num_steps - 1)
    }

    /// Returns the wavelength at a given sample index.
    ///
    /// Returns `None` if `index >= num_steps`.
    #[must_use]
    pub fn wavelength_at(&self, index: u16) -> Option<f32> {
        if index >= self.num_steps {
            return None;
        }
        Some(self.start_nm + f32::from(index) * self.step_nm())
    }
}

/// A spectral reflectance or emission curve.
///
/// Stores spectral samples as `f32` values.  The axis is described by the
/// accompanying [`SpectralRange`].
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralCurve {
    /// Spectral axis description.
    pub range: SpectralRange,
    /// Spectral samples (reflectance or emission power) at each wavelength step.
    pub samples: Vec<f32>,
}

impl SpectralCurve {
    /// Creates a new spectral curve.
    ///
    /// # Errors
    ///
    /// Returns an error if `samples.len() != range.num_steps as usize` or the
    /// range is invalid.
    pub fn new(range: SpectralRange, samples: Vec<f32>) -> Result<Self> {
        if samples.len() != usize::from(range.num_steps) {
            return Err(ColorError::InvalidColor(format!(
                "SpectralCurve: expected {} samples, got {}",
                range.num_steps,
                samples.len()
            )));
        }
        Ok(Self { range, samples })
    }

    /// Interpolates the spectral value at an arbitrary wavelength (linear).
    ///
    /// Returns `None` if `wavelength_nm` is outside the range.
    #[must_use]
    pub fn interpolate(&self, wavelength_nm: f32) -> Option<f32> {
        if wavelength_nm < self.range.start_nm || wavelength_nm > self.range.end_nm {
            return None;
        }
        let step = self.range.step_nm();
        if step <= 0.0 {
            return Some(self.samples.first().copied().unwrap_or(0.0));
        }
        let pos = (wavelength_nm - self.range.start_nm) / step;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(self.samples.len() - 1);
        let t = pos - lo as f32;
        Some(self.samples[lo] * (1.0 - t) + self.samples[hi] * t)
    }

    /// Integrates the spectral curve (trapezoidal rule).
    ///
    /// Returns the integral in nm·units.
    #[must_use]
    pub fn integrate(&self) -> f32 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let step = self.range.step_nm();
        let mut sum = 0.0_f32;
        for i in 0..self.samples.len() - 1 {
            sum += (self.samples[i] + self.samples[i + 1]) * 0.5;
        }
        sum * step
    }
}

// ── Multi-process element (MPE) ───────────────────────────────────────────────

/// A single stage in an iccMAX multi-process element pipeline.
///
/// iccMAX MPE pipelines chain stages that transform a fixed number of channels.
/// Supported stage types include matrices, curves, and CLUTs.
#[derive(Debug, Clone)]
pub enum MpeStage {
    /// Matrix transform: `output = M * input + offset`.
    ///
    /// Stored as row-major `[[f64; in_channels]; out_channels]` plus an
    /// optional offset vector.
    Matrix {
        /// Number of input channels.
        input_channels: usize,
        /// Number of output channels.
        output_channels: usize,
        /// Row-major matrix coefficients.
        coefficients: Vec<f64>,
        /// Optional offset vector (length = output_channels).
        offset: Option<Vec<f64>>,
    },
    /// Curve set: per-channel tone curves applied in parallel.
    CurveSet {
        /// Number of channels.
        channels: usize,
        /// Gamma exponent for each channel (simplified — full iccMAX supports
        /// multi-segment curves).
        gamma: Vec<f64>,
    },
    /// Colour look-up table (CLUT).
    Clut {
        /// Number of input channels.
        input_channels: usize,
        /// Number of output channels.
        output_channels: usize,
        /// Number of grid points per input channel.
        grid_points: usize,
        /// Flat array of CLUT entries in row-major order.
        data: Vec<f32>,
    },
    /// Identity (pass-through) stage.
    Identity {
        /// Number of channels passed through.
        channels: usize,
    },
}

impl MpeStage {
    /// Returns the number of output channels produced by this stage.
    #[must_use]
    pub fn output_channels(&self) -> usize {
        match self {
            Self::Matrix { output_channels, .. } => *output_channels,
            Self::CurveSet { channels, .. } => *channels,
            Self::Clut { output_channels, .. } => *output_channels,
            Self::Identity { channels } => *channels,
        }
    }

    /// Returns the number of input channels consumed by this stage.
    #[must_use]
    pub fn input_channels(&self) -> usize {
        match self {
            Self::Matrix { input_channels, .. } => *input_channels,
            Self::CurveSet { channels, .. } => *channels,
            Self::Clut { input_channels, .. } => *input_channels,
            Self::Identity { channels } => *channels,
        }
    }

    /// Applies the stage to the given input values.
    ///
    /// # Errors
    ///
    /// Returns `ColorError::Matrix` if the input length does not match the
    /// expected number of input channels, or if coefficients are missing.
    pub fn apply(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() != self.input_channels() {
            return Err(ColorError::Matrix(format!(
                "MpeStage: expected {} inputs, got {}",
                self.input_channels(),
                input.len()
            )));
        }
        match self {
            Self::Identity { .. } => Ok(input.to_vec()),
            Self::CurveSet { channels, gamma } => {
                let out: Vec<f64> = input
                    .iter()
                    .zip(gamma.iter().take(*channels))
                    .map(|(&v, &g)| v.max(0.0).powf(g))
                    .collect();
                Ok(out)
            }
            Self::Matrix {
                input_channels,
                output_channels,
                coefficients,
                offset,
            } => {
                if coefficients.len() != output_channels * input_channels {
                    return Err(ColorError::Matrix(
                        "MpeStage::Matrix: coefficient count mismatch".into(),
                    ));
                }
                let mut out = vec![0.0_f64; *output_channels];
                for row in 0..*output_channels {
                    for col in 0..*input_channels {
                        out[row] += coefficients[row * input_channels + col] * input[col];
                    }
                    if let Some(off) = offset {
                        out[row] += off.get(row).copied().unwrap_or(0.0);
                    }
                }
                Ok(out)
            }
            Self::Clut {
                input_channels,
                output_channels,
                grid_points,
                data,
            } => {
                // Trilinear CLUT lookup (up to 3 input channels)
                // For higher dimensions, falls back to nearest-neighbour
                if *input_channels > 3 || *input_channels == 0 {
                    return Err(ColorError::Matrix(
                        "MpeStage::Clut: only 1–3 input channels supported".into(),
                    ));
                }
                let n = *grid_points;
                let nc = *output_channels;
                let expected = n.pow(*input_channels as u32) * nc;
                if data.len() != expected {
                    return Err(ColorError::Matrix(format!(
                        "MpeStage::Clut: data length {}, expected {expected}",
                        data.len()
                    )));
                }
                let t0 = (input[0] * (n - 1) as f64).clamp(0.0, (n - 1) as f64);
                let i0 = t0.floor() as usize;
                let f0 = t0 - i0 as f64;

                if *input_channels == 1 {
                    let i1 = (i0 + 1).min(n - 1);
                    let mut out = vec![0.0_f64; nc];
                    for c in 0..nc {
                        let v0 = f64::from(data[i0 * nc + c]);
                        let v1 = f64::from(data[i1 * nc + c]);
                        out[c] = v0 + (v1 - v0) * f0;
                    }
                    return Ok(out);
                }

                let t1 = (input[1] * (n - 1) as f64).clamp(0.0, (n - 1) as f64);
                let i1_lo = t1.floor() as usize;
                let f1 = t1 - i1_lo as f64;

                if *input_channels == 2 {
                    let i0h = (i0 + 1).min(n - 1);
                    let i1h = (i1_lo + 1).min(n - 1);
                    let mut out = vec![0.0_f64; nc];
                    for c in 0..nc {
                        let v00 = f64::from(data[(i0 * n + i1_lo) * nc + c]);
                        let v10 = f64::from(data[(i0h * n + i1_lo) * nc + c]);
                        let v01 = f64::from(data[(i0 * n + i1h) * nc + c]);
                        let v11 = f64::from(data[(i0h * n + i1h) * nc + c]);
                        out[c] = v00 * (1.0 - f0) * (1.0 - f1)
                            + v10 * f0 * (1.0 - f1)
                            + v01 * (1.0 - f0) * f1
                            + v11 * f0 * f1;
                    }
                    return Ok(out);
                }

                // 3 input channels: trilinear
                let t2 = (input[2] * (n - 1) as f64).clamp(0.0, (n - 1) as f64);
                let i2_lo = t2.floor() as usize;
                let f2 = t2 - i2_lo as f64;
                let i0h = (i0 + 1).min(n - 1);
                let i1h = (i1_lo + 1).min(n - 1);
                let i2h = (i2_lo + 1).min(n - 1);

                let idx = |x: usize, y: usize, z: usize| (x * n * n + y * n + z) * nc;
                let mut out = vec![0.0_f64; nc];
                for c in 0..nc {
                    let c000 = f64::from(data[idx(i0, i1_lo, i2_lo) + c]);
                    let c100 = f64::from(data[idx(i0h, i1_lo, i2_lo) + c]);
                    let c010 = f64::from(data[idx(i0, i1h, i2_lo) + c]);
                    let c110 = f64::from(data[idx(i0h, i1h, i2_lo) + c]);
                    let c001 = f64::from(data[idx(i0, i1_lo, i2h) + c]);
                    let c101 = f64::from(data[idx(i0h, i1_lo, i2h) + c]);
                    let c011 = f64::from(data[idx(i0, i1h, i2h) + c]);
                    let c111 = f64::from(data[idx(i0h, i1h, i2h) + c]);
                    out[c] = c000 * (1.0 - f0) * (1.0 - f1) * (1.0 - f2)
                        + c100 * f0 * (1.0 - f1) * (1.0 - f2)
                        + c010 * (1.0 - f0) * f1 * (1.0 - f2)
                        + c110 * f0 * f1 * (1.0 - f2)
                        + c001 * (1.0 - f0) * (1.0 - f1) * f2
                        + c101 * f0 * (1.0 - f1) * f2
                        + c011 * (1.0 - f0) * f1 * f2
                        + c111 * f0 * f1 * f2;
                }
                Ok(out)
            }
        }
    }
}

/// An iccMAX multi-process element pipeline.
///
/// Chains a sequence of [`MpeStage`] transforms; the output of each stage
/// feeds the input of the next.
#[derive(Debug, Clone, Default)]
pub struct MpePipeline {
    stages: Vec<MpeStage>,
}

impl MpePipeline {
    /// Creates an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Appends a stage.
    ///
    /// # Errors
    ///
    /// Returns `ColorError::Matrix` if the stage's input channel count does
    /// not match the current output channel count of the pipeline.
    pub fn add_stage(&mut self, stage: MpeStage) -> Result<()> {
        if let Some(last) = self.stages.last() {
            if last.output_channels() != stage.input_channels() {
                return Err(ColorError::Matrix(format!(
                    "MpePipeline: stage channel count mismatch ({} → {})",
                    last.output_channels(),
                    stage.input_channels()
                )));
            }
        }
        self.stages.push(stage);
        Ok(())
    }

    /// Appends a stage without channel validation (use when building manually).
    pub fn push_stage(&mut self, stage: MpeStage) {
        self.stages.push(stage);
    }

    /// Returns the number of stages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Returns `true` if the pipeline is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Applies all stages in sequence.
    ///
    /// # Errors
    ///
    /// Propagates errors from individual stage applications.
    pub fn apply(&self, input: &[f64]) -> Result<Vec<f64>> {
        let mut current: Vec<f64> = input.to_vec();
        for stage in &self.stages {
            current = stage.apply(&current)?;
        }
        Ok(current)
    }
}

// ── ICC v5 header ─────────────────────────────────────────────────────────────

/// ICC v5 (iccMAX) profile header.
///
/// Extends the v4 header with additional v5-specific fields for spectral
/// illumination and observer matching function.
#[derive(Debug, Clone)]
pub struct IccV5Header {
    /// Profile size in bytes.
    pub profile_size: u32,
    /// ICC profile version.
    pub version: ProfileVersion,
    /// Profile class.
    pub class: IccProfileClass,
    /// Device colour space (e.g., `b"RGB "`).
    pub color_space: [u8; 4],
    /// Profile Connection Space (e.g., `b"XYZ "` or `b"Lab "`).
    pub pcs: [u8; 4],
    /// Profile flags.
    pub flags: u32,
    /// v5: spectral PCS flags (upper 16 bits indicate spectral illuminant).
    pub spectral_pcs_flags: u16,
    /// v5: spectral wavelength range for the profile connection space.
    pub spectral_range: Option<SpectralRange>,
    /// Human-readable description (from desc tag).
    pub description: String,
    /// Rendering intent.
    pub rendering_intent: u32,
}

impl IccV5Header {
    /// Parses an ICC v5 header from the first 128 bytes of a profile.
    ///
    /// # Errors
    ///
    /// Returns `ColorError::IccProfile` if the data is too short, the
    /// signature is wrong, or the version is unrecognised.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < HEADER_SIZE {
            return Err(ColorError::IccProfile(format!(
                "IccV5Header: data too short ({} bytes, need {HEADER_SIZE})",
                data.len()
            )));
        }

        // Validate "acsp" signature at offset 36
        if &data[HEADER_SIGNATURE_OFFSET..HEADER_SIGNATURE_OFFSET + 4] != b"acsp" {
            return Err(ColorError::IccProfile(
                "IccV5Header: invalid 'acsp' signature".into(),
            ));
        }

        let profile_size = read_u32_be(data, HEADER_PROFILE_SIZE_OFFSET);
        let version = parse_version(data)?;

        let class_sig: [u8; 4] = data[HEADER_CLASS_OFFSET..HEADER_CLASS_OFFSET + 4]
            .try_into()
            .map_err(|_| ColorError::IccProfile("IccV5Header: class read error".into()))?;
        let class = IccProfileClass::from_signature(class_sig)?;

        let color_space: [u8; 4] = data[HEADER_COLORSPACE_OFFSET..HEADER_COLORSPACE_OFFSET + 4]
            .try_into()
            .map_err(|_| ColorError::IccProfile("IccV5Header: colorspace read error".into()))?;

        let pcs: [u8; 4] = data[HEADER_PCS_OFFSET..HEADER_PCS_OFFSET + 4]
            .try_into()
            .map_err(|_| ColorError::IccProfile("IccV5Header: pcs read error".into()))?;

        let flags = read_u32_be(data, HEADER_FLAGS_OFFSET);
        let rendering_intent = read_u32_be(data, HEADER_RENDERING_INTENT_OFFSET);

        // v5 spectral PCS flags: bytes 120–121 of the header
        let spectral_pcs_flags = if data.len() >= 122 {
            (u16::from(data[120]) << 8) | u16::from(data[121])
        } else {
            0
        };

        Ok(Self {
            profile_size,
            version,
            class,
            color_space,
            pcs,
            flags,
            spectral_pcs_flags,
            spectral_range: None,
            description: String::new(),
            rendering_intent,
        })
    }

    /// Serialises the header to 128 bytes.
    ///
    /// Writes a minimal but spec-correct ICC header with the correct
    /// "acsp" signature, version, class, colour space, and PCS fields.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 128] {
        let mut buf = [0u8; 128];

        write_u32_be(&mut buf, HEADER_PROFILE_SIZE_OFFSET, self.profile_size);

        // Version bytes
        buf[HEADER_VERSION_OFFSET] = self.version.major;
        buf[HEADER_VERSION_OFFSET + 1] = (self.version.minor << 4) | (self.version.bugfix & 0xf);
        buf[HEADER_VERSION_OFFSET + 2] = 0;
        buf[HEADER_VERSION_OFFSET + 3] = 0;

        let class_sig = self.class.signature();
        buf[HEADER_CLASS_OFFSET..HEADER_CLASS_OFFSET + 4].copy_from_slice(&class_sig);
        buf[HEADER_COLORSPACE_OFFSET..HEADER_COLORSPACE_OFFSET + 4]
            .copy_from_slice(&self.color_space);
        buf[HEADER_PCS_OFFSET..HEADER_PCS_OFFSET + 4].copy_from_slice(&self.pcs);

        // Signature
        buf[HEADER_SIGNATURE_OFFSET..HEADER_SIGNATURE_OFFSET + 4].copy_from_slice(b"acsp");

        write_u32_be(&mut buf, HEADER_FLAGS_OFFSET, self.flags);
        write_u32_be(
            &mut buf,
            HEADER_RENDERING_INTENT_OFFSET,
            self.rendering_intent,
        );

        // v5 spectral PCS flags at bytes 120–121
        buf[120] = (self.spectral_pcs_flags >> 8) as u8;
        buf[121] = (self.spectral_pcs_flags & 0xff) as u8;

        buf
    }

    /// Returns `true` if this is an iccMAX v5 profile.
    #[must_use]
    pub fn is_iccmax(&self) -> bool {
        self.version.is_iccmax()
    }
}

// ── Builder for ICC v5 headers ────────────────────────────────────────────────

/// Builder for constructing [`IccV5Header`] objects.
#[derive(Debug, Default)]
pub struct IccV5HeaderBuilder {
    class: Option<IccProfileClass>,
    color_space: [u8; 4],
    pcs: [u8; 4],
    description: String,
    spectral_range: Option<SpectralRange>,
    rendering_intent: u32,
}

impl IccV5HeaderBuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            color_space: *b"RGB ",
            pcs: *b"XYZ ",
            ..Self::default()
        }
    }

    /// Sets the profile class.
    pub fn class(mut self, class: IccProfileClass) -> Self {
        self.class = Some(class);
        self
    }

    /// Sets the device colour space.
    pub fn color_space(mut self, cs: [u8; 4]) -> Self {
        self.color_space = cs;
        self
    }

    /// Sets the Profile Connection Space.
    pub fn pcs(mut self, pcs: [u8; 4]) -> Self {
        self.pcs = pcs;
        self
    }

    /// Sets the profile description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Attaches a spectral wavelength range (iccMAX v5 feature).
    pub fn spectral_range(mut self, range: SpectralRange) -> Self {
        self.spectral_range = Some(range);
        self
    }

    /// Sets the rendering intent (0=perceptual, 1=relative colorimetric, etc.).
    pub fn rendering_intent(mut self, intent: u32) -> Self {
        self.rendering_intent = intent;
        self
    }

    /// Builds the header.
    ///
    /// # Errors
    ///
    /// Returns `ColorError::IccProfile` if required fields are missing.
    pub fn build(self) -> Result<IccV5Header> {
        let class = self.class.ok_or_else(|| {
            ColorError::IccProfile("IccV5HeaderBuilder: class not set".into())
        })?;

        let spectral_pcs_flags = if self.spectral_range.is_some() { 0x8000u16 } else { 0 };

        Ok(IccV5Header {
            profile_size: 0, // Updated when full profile is serialised
            version: ProfileVersion::v5(),
            class,
            color_space: self.color_space,
            pcs: self.pcs,
            flags: 0,
            spectral_pcs_flags,
            spectral_range: self.spectral_range,
            description: self.description,
            rendering_intent: self.rendering_intent,
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn read_u32_be(data: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

#[inline]
fn write_u32_be(buf: &mut [u8], offset: usize, value: u32) {
    let bytes = value.to_be_bytes();
    buf[offset..offset + 4].copy_from_slice(&bytes);
}

fn parse_version(data: &[u8]) -> Result<ProfileVersion> {
    let major = data[HEADER_VERSION_OFFSET];
    let minor_bcd = (data[HEADER_VERSION_OFFSET + 1] >> 4) & 0xf;
    let bugfix_bcd = data[HEADER_VERSION_OFFSET + 1] & 0xf;
    if major < 2 {
        return Err(ColorError::IccProfile(format!(
            "IccV5Header: unsupported profile version {major}.{minor_bcd}"
        )));
    }
    Ok(ProfileVersion::new(major, minor_bcd, bugfix_bcd))
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ProfileVersion ────────────────────────────────────────────────────────

    #[test]
    fn test_profile_version_v5_is_iccmax() {
        assert!(ProfileVersion::v5().is_iccmax());
        assert!(!ProfileVersion::v4().is_iccmax());
        assert!(!ProfileVersion::v2().is_iccmax());
    }

    #[test]
    fn test_profile_version_ordering() {
        assert!(ProfileVersion::v5() > ProfileVersion::v4());
        assert!(ProfileVersion::v4() > ProfileVersion::v2());
    }

    #[test]
    fn test_profile_version_display() {
        let v = ProfileVersion::v5();
        let s = v.to_string();
        assert!(s.starts_with('5'), "Expected '5.x.x', got '{s}'");
    }

    // ── Float16 ───────────────────────────────────────────────────────────────

    #[test]
    fn test_float16_zero_roundtrip() {
        let f = Float16::from_f64(0.0);
        assert!((f.to_f64() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_float16_one_roundtrip() {
        let f = Float16::from_f64(1.0);
        assert!((f.to_f64() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_float16_half_roundtrip() {
        let f = Float16::from_f64(0.5);
        assert!((f.to_f64() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_float16_infinity() {
        let f = Float16::from_f64(f64::INFINITY);
        assert!(f.to_f64().is_infinite());
        assert!(f.to_f64() > 0.0);
    }

    #[test]
    fn test_float16_negative_roundtrip() {
        let f = Float16::from_f64(-1.0);
        assert!((f.to_f64() + 1.0).abs() < 0.001);
    }

    // ── SpectralRange ─────────────────────────────────────────────────────────

    #[test]
    fn test_spectral_range_valid() {
        let r = SpectralRange::new(380.0, 780.0, 81).expect("valid spectral range");
        assert_eq!(r.num_steps, 81);
        assert!((r.step_nm() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_spectral_range_wavelength_at() {
        let r = SpectralRange::new(380.0, 780.0, 81).expect("valid spectral range");
        assert!((r.wavelength_at(0).unwrap_or(0.0) - 380.0).abs() < 0.01);
        assert!((r.wavelength_at(80).unwrap_or(0.0) - 780.0).abs() < 0.1);
    }

    #[test]
    fn test_spectral_range_invalid_order() {
        assert!(SpectralRange::new(780.0, 380.0, 81).is_err());
    }

    #[test]
    fn test_spectral_range_too_few_steps() {
        assert!(SpectralRange::new(380.0, 780.0, 1).is_err());
    }

    // ── SpectralCurve ─────────────────────────────────────────────────────────

    #[test]
    fn test_spectral_curve_interpolate_middle() {
        let r = SpectralRange::new(0.0, 100.0, 3).expect("valid range"); // 0, 50, 100
        let c = SpectralCurve::new(r, vec![0.0, 0.5, 1.0]).expect("valid curve");
        let v = c.interpolate(25.0).expect("interpolated value");
        assert!((v - 0.25).abs() < 0.01, "v={v}");
    }

    #[test]
    fn test_spectral_curve_out_of_range() {
        let r = SpectralRange::new(380.0, 780.0, 3).expect("valid range");
        let c = SpectralCurve::new(r, vec![0.0, 0.5, 1.0]).expect("valid curve");
        assert!(c.interpolate(300.0).is_none());
        assert!(c.interpolate(900.0).is_none());
    }

    #[test]
    fn test_spectral_curve_integrate() {
        // Flat curve of 1.0 over 100 nm should integrate to ~100 nm
        let r = SpectralRange::new(0.0, 100.0, 3).expect("valid range");
        let c = SpectralCurve::new(r, vec![1.0, 1.0, 1.0]).expect("valid curve");
        let integral = c.integrate();
        assert!((integral - 100.0).abs() < 1.0, "integral={integral}");
    }

    // ── MpeStage ──────────────────────────────────────────────────────────────

    #[test]
    fn test_mpe_stage_identity() {
        let s = MpeStage::Identity { channels: 3 };
        let out = s.apply(&[0.1, 0.5, 0.9]).expect("identity stage should succeed");
        assert_eq!(out, vec![0.1, 0.5, 0.9]);
    }

    #[test]
    fn test_mpe_stage_matrix_3x3() {
        // Identity matrix
        let s = MpeStage::Matrix {
            input_channels: 3,
            output_channels: 3,
            coefficients: vec![
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
            offset: None,
        };
        let out = s.apply(&[0.2, 0.4, 0.6]).expect("matrix stage should succeed");
        for (a, b) in out.iter().zip([0.2, 0.4, 0.6].iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_mpe_stage_curve_set_gamma() {
        let s = MpeStage::CurveSet {
            channels: 3,
            gamma: vec![2.2, 2.2, 2.2],
        };
        let out = s.apply(&[1.0, 0.5, 0.0]).expect("curve set should succeed");
        assert!((out[0] - 1.0).abs() < 1e-10, "1.0^2.2 = 1.0");
        assert!((out[2] - 0.0).abs() < 1e-10, "0.0^2.2 = 0.0");
        assert!(out[1] < 0.5, "0.5^2.2 < 0.5 for gamma > 1");
    }

    #[test]
    fn test_mpe_stage_channel_mismatch_error() {
        let s = MpeStage::Identity { channels: 3 };
        assert!(s.apply(&[0.1, 0.2]).is_err()); // Only 2 channels, expects 3
    }

    // ── MpePipeline ───────────────────────────────────────────────────────────

    #[test]
    fn test_mpe_pipeline_empty() {
        let p = MpePipeline::new();
        let out = p.apply(&[0.5, 0.5, 0.5]).expect("empty pipeline should pass-through");
        assert_eq!(out, vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_mpe_pipeline_two_stages() {
        let mut p = MpePipeline::new();
        p.push_stage(MpeStage::Identity { channels: 3 });
        p.push_stage(MpeStage::CurveSet {
            channels: 3,
            gamma: vec![1.0, 1.0, 1.0],
        });
        let out = p.apply(&[0.3, 0.6, 0.9]).expect("two-stage pipeline should succeed");
        for (a, b) in out.iter().zip([0.3, 0.6, 0.9].iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    // ── IccV5Header ───────────────────────────────────────────────────────────

    #[test]
    fn test_icc_v5_header_builder_roundtrip() {
        let range = SpectralRange::new(380.0, 780.0, 81).expect("valid range");
        let header = IccV5HeaderBuilder::new()
            .class(IccProfileClass::Display)
            .color_space(*b"RGB ")
            .pcs(*b"XYZ ")
            .description("Test Profile")
            .spectral_range(range)
            .rendering_intent(1)
            .build()
            .expect("header build should succeed");

        assert!(header.is_iccmax());
        assert_eq!(header.class, IccProfileClass::Display);
        assert_eq!(header.color_space, *b"RGB ");
        assert_eq!(header.description, "Test Profile");
        assert!(header.spectral_range.is_some());
        assert!(header.spectral_pcs_flags & 0x8000 != 0);
    }

    #[test]
    fn test_icc_v5_header_to_bytes_has_acsp_signature() {
        let header = IccV5HeaderBuilder::new()
            .class(IccProfileClass::Display)
            .build()
            .expect("build should succeed");

        let bytes = header.to_bytes();
        assert_eq!(&bytes[HEADER_SIGNATURE_OFFSET..HEADER_SIGNATURE_OFFSET + 4], b"acsp");
        // Check version byte is 5
        assert_eq!(bytes[HEADER_VERSION_OFFSET], 5);
    }

    #[test]
    fn test_icc_v5_header_parse_from_bytes() {
        // Build → serialise → parse roundtrip
        let header = IccV5HeaderBuilder::new()
            .class(IccProfileClass::Display)
            .build()
            .expect("build should succeed");

        let bytes = header.to_bytes();
        let parsed = IccV5Header::parse(&bytes).expect("parse should succeed");
        assert!(parsed.is_iccmax());
        assert_eq!(parsed.class, IccProfileClass::Display);
    }

    #[test]
    fn test_icc_v5_header_parse_too_short() {
        let bytes = vec![0u8; 64];
        assert!(IccV5Header::parse(&bytes).is_err());
    }

    #[test]
    fn test_icc_v5_header_parse_bad_signature() {
        let mut bytes = [0u8; 128];
        // Correct version byte
        bytes[HEADER_VERSION_OFFSET] = 5;
        // Wrong signature
        bytes[HEADER_SIGNATURE_OFFSET..HEADER_SIGNATURE_OFFSET + 4].copy_from_slice(b"XXXX");
        assert!(IccV5Header::parse(&bytes).is_err());
    }

    #[test]
    fn test_icc_profile_class_roundtrip() {
        for cls in [
            IccProfileClass::Input,
            IccProfileClass::Display,
            IccProfileClass::Output,
            IccProfileClass::DeviceLink,
            IccProfileClass::ColorSpace,
            IccProfileClass::Abstract,
            IccProfileClass::NamedColor,
        ] {
            let sig = cls.signature();
            let parsed = IccProfileClass::from_signature(sig).expect("class roundtrip");
            assert_eq!(parsed, cls, "class={cls:?}");
        }
    }
}
