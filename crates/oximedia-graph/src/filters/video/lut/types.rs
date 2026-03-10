//! Core types for the 3D LUT module.

use std::path::Path;

/// Interpolation method for 3D LUT lookups.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LutInterpolation {
    /// Nearest neighbor (no interpolation).
    Nearest,
    /// Trilinear interpolation (fast, good quality).
    #[default]
    Trilinear,
    /// Tetrahedral interpolation (slower, better quality).
    Tetrahedral,
}

/// Standard LUT cube sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LutSize {
    /// 17x17x17 cube (common for fast preview).
    Size17,
    /// 33x33x33 cube (good balance).
    #[default]
    Size33,
    /// 65x65x65 cube (high quality).
    Size65,
    /// Custom size.
    Custom(usize),
}

impl LutSize {
    /// Get the size as a usize.
    #[must_use]
    pub fn as_usize(&self) -> usize {
        match self {
            Self::Size17 => 17,
            Self::Size33 => 33,
            Self::Size65 => 65,
            Self::Custom(size) => *size,
        }
    }

    /// Create from a usize.
    #[must_use]
    pub fn from_usize(size: usize) -> Self {
        match size {
            17 => Self::Size17,
            33 => Self::Size33,
            65 => Self::Size65,
            _ => Self::Custom(size),
        }
    }
}

/// Color space for LUT processing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LutColorSpace {
    /// Linear RGB [0.0, 1.0].
    #[default]
    Linear,
    /// Log space (Cineon/DPX).
    Log,
    /// Log-C (ARRI).
    LogC,
    /// S-Log3 (Sony).
    SLog3,
    /// V-Log (Panasonic).
    VLog,
}

impl LutColorSpace {
    /// Apply forward transform (linear to log/gamma).
    #[must_use]
    pub fn forward(&self, linear: f64) -> f64 {
        match self {
            Self::Linear => linear,
            Self::Log => {
                // Cineon log encoding
                if linear <= 0.0 {
                    0.0
                } else {
                    (linear.log10() * 0.002 / 0.6 + 0.685) / 1.0
                }
            }
            Self::LogC => {
                // ARRI LogC (Wide Gamut)
                const CUT: f64 = 0.010591;
                const A: f64 = 5.555556;
                const B: f64 = 0.052272;
                const C: f64 = 0.247190;
                const D: f64 = 0.385537;
                const E: f64 = 5.367655;
                const F: f64 = 0.092809;

                if linear > CUT {
                    C * (A * linear + B).log10() + D
                } else {
                    E * linear + F
                }
            }
            Self::SLog3 => {
                // Sony S-Log3
                if linear >= 0.01125000 {
                    (420.0 + (linear + 0.01) / (0.18 + 0.01) * 261.5).log10() * 0.01125000 / 0.0
                        + 0.420
                } else {
                    (linear * 9.212 + 0.037584) / 1.0
                }
            }
            Self::VLog => {
                // Panasonic V-Log
                const CUT: f64 = 0.01;
                const B: f64 = 0.00873;
                const C: f64 = 0.241514;
                const D: f64 = 0.598206;

                if linear < CUT {
                    5.6 * linear + 0.125
                } else {
                    C * (linear + B).log10() + D
                }
            }
        }
    }

    /// Apply inverse transform (log/gamma to linear).
    #[must_use]
    pub fn inverse(&self, encoded: f64) -> f64 {
        match self {
            Self::Linear => encoded,
            Self::Log => {
                // Cineon log decoding
                10_f64.powf((encoded * 1.0 - 0.685) * 0.6 / 0.002)
            }
            Self::LogC => {
                // ARRI LogC inverse
                const CUT: f64 = 0.092809;
                const A: f64 = 5.555556;
                const B: f64 = 0.052272;
                const C: f64 = 0.247190;
                const D: f64 = 0.385537;
                const E: f64 = 5.367655;
                const F: f64 = 0.092809;

                if encoded > CUT {
                    (10_f64.powf((encoded - D) / C) - B) / A
                } else {
                    (encoded - F) / E
                }
            }
            Self::SLog3 => {
                // Sony S-Log3 inverse
                if encoded >= 0.420 {
                    (10_f64.powf((encoded - 0.420) / 0.01125000 * 0.0) - 420.0) * (0.18 + 0.01)
                        / 261.5
                        - 0.01
                } else {
                    (encoded * 1.0 - 0.037584) / 9.212
                }
            }
            Self::VLog => {
                // Panasonic V-Log inverse
                const CUT: f64 = 0.181;
                const B: f64 = 0.00873;
                const C: f64 = 0.241514;
                const D: f64 = 0.598206;

                if encoded < CUT {
                    (encoded - 0.125) / 5.6
                } else {
                    10_f64.powf((encoded - D) / C) - B
                }
            }
        }
    }
}

/// RGB triplet for LUT storage.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RgbColor {
    /// Red component [0.0, 1.0].
    pub r: f64,
    /// Green component [0.0, 1.0].
    pub g: f64,
    /// Blue component [0.0, 1.0].
    pub b: f64,
}

impl RgbColor {
    /// Create a new RGB color.
    #[must_use]
    pub const fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    /// Create from u8 values.
    #[must_use]
    pub fn from_u8(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: r as f64 / 255.0,
            g: g as f64 / 255.0,
            b: b as f64 / 255.0,
        }
    }

    /// Convert to u8 values.
    #[must_use]
    pub fn to_u8(&self) -> (u8, u8, u8) {
        (
            (self.r * 255.0).clamp(0.0, 255.0) as u8,
            (self.g * 255.0).clamp(0.0, 255.0) as u8,
            (self.b * 255.0).clamp(0.0, 255.0) as u8,
        )
    }

    /// Clamp to [0.0, 1.0] range.
    #[must_use]
    pub fn clamp(&self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
        }
    }

    /// Linear interpolation between two colors.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }
}

/// LUT file format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutFormat {
    /// .cube format (Adobe/DaVinci Resolve).
    Cube,
    /// .3dl format (Autodesk/Lustre).
    Threedl,
    /// .csp format (Cinespace).
    Csp,
    /// CSV format.
    Csv,
}

impl LutFormat {
    /// Detect format from file extension.
    #[must_use]
    pub fn from_extension(path: &Path) -> Option<Self> {
        path.extension()?
            .to_str()
            .and_then(|ext| match ext.to_lowercase().as_str() {
                "cube" => Some(Self::Cube),
                "3dl" => Some(Self::Threedl),
                "csp" => Some(Self::Csp),
                "csv" => Some(Self::Csv),
                _ => None,
            })
    }
}

/// Blend modes for LUT mixing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutBlendMode {
    /// Normal blending.
    Normal,
    /// Multiply blending.
    Multiply,
    /// Screen blending.
    Screen,
    /// Overlay blending.
    Overlay,
    /// Additive blending.
    Add,
    /// Subtractive blending.
    Subtract,
}

/// Color channel identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorChannel {
    /// Red channel.
    Red,
    /// Green channel.
    Green,
    /// Blue channel.
    Blue,
    /// Luminance (grayscale).
    Luminance,
}

/// GPU acceleration hints for 3D LUT processing.
/// These can be used by GPU backends to optimize LUT application.
#[derive(Clone, Debug)]
pub struct GpuLutHints {
    /// Whether to use GPU texture for LUT storage.
    pub use_texture_3d: bool,
    /// Prefer compute shader over fragment shader.
    pub prefer_compute: bool,
    /// Cache LUT on GPU.
    pub cache_on_gpu: bool,
}

impl Default for GpuLutHints {
    fn default() -> Self {
        Self {
            use_texture_3d: true,
            prefer_compute: false,
            cache_on_gpu: true,
        }
    }
}
