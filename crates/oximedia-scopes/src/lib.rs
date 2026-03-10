#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::many_single_char_names
)]
//! Professional video scopes for broadcast-quality video analysis.
//!
//! This crate provides industry-standard video scopes for analyzing video signals,
//! including waveform monitors, vectorscopes, histograms, and parade displays.
//! All scopes are ITU-R BT.709/BT.2020 compliant and suitable for broadcast workflows.
//!
//! # Features
//!
//! - **Waveform Monitor**: Luma, RGB parade, RGB overlay, YCbCr waveform with graticule
//! - **Vectorscope**: YUV vectorscope with SMPTE color bars, skin tone line, gamut warnings
//! - **Histogram**: RGB and luma histograms with statistical overlays
//! - **Parade**: RGB and YCbCr parade displays with component selection
//! - **High Precision**: 8-bit and 10-bit support
//! - **Real-time**: Optimized for real-time video analysis
//! - **Broadcast Quality**: ITU-R BT.709/BT.2020 compliant
//!
//! # Example
//!
//! ```
//! use oximedia_scopes::{VideoScopes, ScopeType, WaveformMode, ScopeConfig};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create video scopes analyzer
//! let mut scopes = VideoScopes::new(ScopeConfig::default());
//!
//! // Analyze frame with waveform
//! // let frame_data: &[u8] = /* your video frame */;
//! // let waveform = scopes.analyze(frame_data, 1920, 1080, ScopeType::WaveformLuma)?;
//!
//! // Render scope to image
//! // let image = scopes.render(&waveform)?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]

pub mod audio_scope;
pub mod bit_depth_scope;
pub mod cie;
pub mod clipping_detector;
pub mod color_temperature;
pub mod compliance;
pub mod exposure_meter;
pub mod false_color;
pub mod false_color_mapping;
pub mod focus;
pub mod focus_assist;
pub mod gamut_scope;
pub mod hdr;
pub mod histogram;
pub mod histogram_stats;
pub mod lissajous;
pub mod loudness_scope;
pub mod motion_vector_scope;
pub mod overlay;
pub mod parade;
pub mod peaking;
pub mod render;
pub mod rgb_balance;
pub mod scope_layout;
pub mod signal_stats;
pub mod stats;
pub mod vectorscope;
pub mod vectorscope_targets;
pub mod waveform;
pub mod waveform_analyzer;
pub mod zebra;

use oximedia_core::OxiResult;

/// Type of video scope to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeType {
    /// Luma waveform (Y channel only).
    WaveformLuma,

    /// RGB parade waveform (R|G|B side-by-side).
    WaveformRgbParade,

    /// RGB overlay waveform (all channels overlaid).
    WaveformRgbOverlay,

    /// YCbCr waveform (Y|Cb|Cr parade).
    WaveformYcbcr,

    /// YUV vectorscope (Cb/Cr circular display).
    Vectorscope,

    /// RGB histogram.
    HistogramRgb,

    /// Luma histogram (Y channel only).
    HistogramLuma,

    /// RGB parade (R|G|B side-by-side vertical bars).
    ParadeRgb,

    /// YCbCr parade (Y|Cb|Cr side-by-side).
    ParadeYcbcr,

    /// False color exposure visualization.
    FalseColor,

    /// CIE 1931 chromaticity diagram.
    CieDiagram,

    /// Focus assist with edge peaking.
    FocusAssist,

    /// HDR waveform with PQ/HLG/nits scale.
    HdrWaveform,
}

/// Waveform display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveformMode {
    /// Overlay all scanlines (brightest where most pixels).
    Overlay,

    /// Side-by-side parade (R|G|B or Y|Cb|Cr).
    Parade,

    /// Blended/averaged display.
    Blend,
}

/// Vectorscope display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorscopeMode {
    /// Circular display (traditional).
    Circular,

    /// Rectangular display.
    Rectangular,
}

/// Histogram display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistogramMode {
    /// Overlay all channels.
    Overlay,

    /// Stacked channels.
    Stacked,

    /// Logarithmic scale.
    Logarithmic,
}

/// Configuration for video scopes.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ScopeConfig {
    /// Width of the scope display in pixels.
    pub width: u32,

    /// Height of the scope display in pixels.
    pub height: u32,

    /// Whether to show graticule overlay.
    pub show_graticule: bool,

    /// Whether to show text labels.
    pub show_labels: bool,

    /// Whether to enable anti-aliasing.
    pub anti_alias: bool,

    /// Waveform display mode.
    pub waveform_mode: WaveformMode,

    /// Vectorscope display mode.
    pub vectorscope_mode: VectorscopeMode,

    /// Histogram display mode.
    pub histogram_mode: HistogramMode,

    /// Vectorscope gain (1.0 = normal, 2.0 = 2x zoom).
    pub vectorscope_gain: f32,

    /// Whether to highlight out-of-gamut colors.
    pub highlight_gamut: bool,

    /// Color space for gamut warnings (709, 2020, P3).
    pub gamut_colorspace: GamutColorspace,
}

/// Color space for gamut warnings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamutColorspace {
    /// Rec.709 (HD).
    Rec709,

    /// Rec.2020 (UHD/HDR).
    Rec2020,

    /// DCI-P3 (Digital Cinema).
    DciP3,
}

impl Default for ScopeConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            show_graticule: true,
            show_labels: true,
            anti_alias: true,
            waveform_mode: WaveformMode::Overlay,
            vectorscope_mode: VectorscopeMode::Circular,
            histogram_mode: HistogramMode::Overlay,
            vectorscope_gain: 1.0,
            highlight_gamut: false,
            gamut_colorspace: GamutColorspace::Rec709,
        }
    }
}

/// Scope data ready for rendering.
#[derive(Debug, Clone)]
pub struct ScopeData {
    /// Width of the scope.
    pub width: u32,

    /// Height of the scope.
    pub height: u32,

    /// Scope pixel data (RGBA, row-major).
    pub data: Vec<u8>,

    /// Type of scope.
    pub scope_type: ScopeType,
}

/// Main video scopes analyzer.
pub struct VideoScopes {
    config: ScopeConfig,
}

impl VideoScopes {
    /// Creates a new video scopes analyzer with the given configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_scopes::{VideoScopes, ScopeConfig};
    ///
    /// let scopes = VideoScopes::new(ScopeConfig::default());
    /// ```
    #[must_use]
    pub fn new(config: ScopeConfig) -> Self {
        Self { config }
    }

    /// Analyzes a video frame and generates the specified scope.
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame pixel data (RGB24 or `YUV420p`)
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `scope_type` - Type of scope to generate
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Frame dimensions are invalid
    /// - Frame data is insufficient
    /// - Scope generation fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oximedia_scopes::{VideoScopes, ScopeType, ScopeConfig};
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let scopes = VideoScopes::new(ScopeConfig::default());
    /// let frame_data = vec![0u8; 1920 * 1080 * 3]; // RGB24
    /// let scope = scopes.analyze(&frame_data, 1920, 1080, ScopeType::WaveformLuma)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn analyze(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
        scope_type: ScopeType,
    ) -> OxiResult<ScopeData> {
        match scope_type {
            ScopeType::WaveformLuma => {
                waveform::generate_luma_waveform(frame, width, height, &self.config)
            }
            ScopeType::WaveformRgbParade => {
                waveform::generate_rgb_parade(frame, width, height, &self.config)
            }
            ScopeType::WaveformRgbOverlay => {
                waveform::generate_rgb_overlay(frame, width, height, &self.config)
            }
            ScopeType::WaveformYcbcr => {
                waveform::generate_ycbcr_waveform(frame, width, height, &self.config)
            }
            ScopeType::Vectorscope => {
                vectorscope::generate_vectorscope(frame, width, height, &self.config)
            }
            ScopeType::HistogramRgb => {
                histogram::generate_rgb_histogram(frame, width, height, &self.config)
            }
            ScopeType::HistogramLuma => {
                histogram::generate_luma_histogram(frame, width, height, &self.config)
            }
            ScopeType::ParadeRgb => parade::generate_rgb_parade(frame, width, height, &self.config),
            ScopeType::ParadeYcbcr => {
                parade::generate_ycbcr_parade(frame, width, height, &self.config)
            }
            ScopeType::FalseColor => {
                let scale = false_color::FalseColorScale::default();
                false_color::generate_false_color(
                    frame,
                    width,
                    height,
                    false_color::FalseColorMode::Ire,
                    &scale,
                )
            }
            ScopeType::CieDiagram => cie::generate_cie_diagram(frame, width, height, &self.config),
            ScopeType::FocusAssist => {
                let config = focus::FocusAssistConfig::default();
                focus::generate_focus_assist(frame, width, height, &config)
            }
            ScopeType::HdrWaveform => {
                let config = hdr::HdrWaveformConfig::default();
                hdr::generate_hdr_waveform(frame, width, height, &config)
            }
        }
    }

    /// Renders scope data to an RGBA image.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn render(&self, scope: &ScopeData) -> OxiResult<Vec<u8>> {
        // Scope data is already rendered, just return a copy
        Ok(scope.data.clone())
    }

    /// Updates the configuration.
    pub fn set_config(&mut self, config: ScopeConfig) {
        self.config = config;
    }

    /// Gets the current configuration.
    #[must_use]
    pub const fn config(&self) -> &ScopeConfig {
        &self.config
    }
}

impl Default for VideoScopes {
    fn default() -> Self {
        Self::new(ScopeConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_config_default() {
        let config = ScopeConfig::default();
        assert_eq!(config.width, 512);
        assert_eq!(config.height, 512);
        assert!(config.show_graticule);
        assert!(config.show_labels);
    }

    #[test]
    fn test_video_scopes_new() {
        let scopes = VideoScopes::new(ScopeConfig::default());
        assert_eq!(scopes.config().width, 512);
    }

    #[test]
    fn test_video_scopes_default() {
        let scopes = VideoScopes::default();
        assert_eq!(scopes.config().width, 512);
    }
}
