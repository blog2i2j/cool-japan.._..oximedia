//! Color grading filter with lift/gamma/gain, curves, and color wheels.
//!
//! This filter provides professional color grading tools including:
//!
//! - **Lift/Gamma/Gain:** Primary color correction controls
//!   - Lift adjusts shadows (blacks)
//!   - Gamma adjusts midtones
//!   - Gain adjusts highlights (whites)
//!   - Per-channel RGB and master controls
//!   - ASC CDL (Color Decision List) implementation
//!
//! - **Curves:** Precise tonal and color control
//!   - RGB curves (per-channel)
//!   - Luma curve (brightness)
//!   - Hue vs. Saturation curves
//!   - Hue vs. Luminance curves
//!   - Cubic spline interpolation
//!   - Bezier curve support
//!   - Curve presets (S-curve, contrast, etc.)
//!
//! - **Color Wheels:** Intuitive color grading controls
//!   - Shadow/Midtone/Highlight wheels
//!   - Hue shift per region
//!   - Saturation adjustment
//!   - Log/Offset/Power controls
//!   - Temperature/Tint adjustment
//!
//! - **HSL Qualifiers:** Secondary color correction
//!   - Hue range selection
//!   - Saturation range selection
//!   - Luminance range selection
//!   - Soft edge feathering
//!   - Keying and masking
//!
//! # Example
//!
//! ```ignore
//! use oximedia_graph::filters::video::{ColorGradingFilter, ColorGradingConfig};
//! use oximedia_graph::node::NodeId;
//!
//! let config = ColorGradingConfig::new()
//!     .with_lift(0.1, 0.0, -0.1)
//!     .with_gamma(1.2, 1.0, 0.8)
//!     .with_gain(1.1, 1.0, 0.9);
//!
//! let filter = ColorGradingFilter::new(NodeId(0), "grading", config);
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::struct_excessive_bools)]
#![allow(dead_code)]
#![allow(unused_imports)]

mod curves;
mod primary;
mod qualifier;
mod scopes;
mod types;
mod utility;
mod wheels;

// Re-export public items
pub use curves::{Curve, CurveInterpolation, CurvePoint, HueVsLumCurve, HueVsSatCurve, RgbCurves};
pub use primary::{AscCdl, LiftGammaGain};
pub use qualifier::{HslQualifier, MultiQualifier, QualifierCombineMode};
pub use scopes::{ColorHistogram, VectorscopeAnalyzer, WaveformMonitor};
pub use types::{ColorChannel, HslColor, RgbColor};
pub use wheels::{ColorWheel, ColorWheels, LogOffsetPower, TemperatureTint};

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortId, PortType};
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

// ============================================================================
// Main Color Grading Configuration
// ============================================================================

/// Complete color grading configuration.
#[derive(Clone, Debug)]
pub struct ColorGradingConfig {
    /// Lift/Gamma/Gain correction.
    pub lgg: LiftGammaGain,
    /// ASC CDL correction.
    pub cdl: AscCdl,
    /// RGB curves.
    pub rgb_curves: RgbCurves,
    /// Luma curve.
    pub luma_curve: Curve,
    /// Hue vs. Saturation curve.
    pub hue_vs_sat: HueVsSatCurve,
    /// Hue vs. Luminance curve.
    pub hue_vs_lum: HueVsLumCurve,
    /// Color wheels.
    pub color_wheels: ColorWheels,
    /// Log/Offset/Power.
    pub lop: LogOffsetPower,
    /// Temperature/Tint.
    pub temp_tint: TemperatureTint,
    /// HSL qualifier for secondary correction.
    pub qualifier: HslQualifier,
    /// Enable lift/gamma/gain.
    pub enable_lgg: bool,
    /// Enable ASC CDL.
    pub enable_cdl: bool,
    /// Enable RGB curves.
    pub enable_rgb_curves: bool,
    /// Enable luma curve.
    pub enable_luma_curve: bool,
    /// Enable hue vs. sat curve.
    pub enable_hue_vs_sat: bool,
    /// Enable hue vs. lum curve.
    pub enable_hue_vs_lum: bool,
    /// Enable color wheels.
    pub enable_color_wheels: bool,
    /// Enable log/offset/power.
    pub enable_lop: bool,
    /// Enable temperature/tint.
    pub enable_temp_tint: bool,
    /// Enable HSL qualifier.
    pub enable_qualifier: bool,
}

impl Default for ColorGradingConfig {
    fn default() -> Self {
        Self {
            lgg: LiftGammaGain::default(),
            cdl: AscCdl::default(),
            rgb_curves: RgbCurves::default(),
            luma_curve: Curve::linear(),
            hue_vs_sat: HueVsSatCurve::default(),
            hue_vs_lum: HueVsLumCurve::default(),
            color_wheels: ColorWheels::default(),
            lop: LogOffsetPower::default(),
            temp_tint: TemperatureTint::default(),
            qualifier: HslQualifier::default(),
            enable_lgg: false,
            enable_cdl: false,
            enable_rgb_curves: false,
            enable_luma_curve: false,
            enable_hue_vs_sat: false,
            enable_hue_vs_lum: false,
            enable_color_wheels: false,
            enable_lop: false,
            enable_temp_tint: false,
            enable_qualifier: false,
        }
    }
}

impl ColorGradingConfig {
    /// Create a new color grading configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable and set lift values.
    #[must_use]
    pub fn with_lift(mut self, r: f64, g: f64, b: f64) -> Self {
        self.lgg.lift = RgbColor::new(r, g, b);
        self.enable_lgg = true;
        self
    }

    /// Enable and set gamma values.
    #[must_use]
    pub fn with_gamma(mut self, r: f64, g: f64, b: f64) -> Self {
        self.lgg.gamma = RgbColor::new(r, g, b);
        self.enable_lgg = true;
        self
    }

    /// Enable and set gain values.
    #[must_use]
    pub fn with_gain(mut self, r: f64, g: f64, b: f64) -> Self {
        self.lgg.gain = RgbColor::new(r, g, b);
        self.enable_lgg = true;
        self
    }

    /// Apply all color grading to a color.
    #[must_use]
    pub fn apply(&self, color: RgbColor) -> RgbColor {
        let mut result = color;

        // Apply temperature/tint first (white balance)
        if self.enable_temp_tint {
            result = self.temp_tint.apply(result);
        }

        // Apply lift/gamma/gain
        if self.enable_lgg {
            result = self.lgg.apply(result);
        }

        // Apply ASC CDL
        if self.enable_cdl {
            result = self.cdl.apply(result);
        }

        // Apply log/offset/power
        if self.enable_lop {
            result = self.lop.apply(result);
        }

        // Apply RGB curves
        if self.enable_rgb_curves {
            result = self.rgb_curves.apply(result);
        }

        // Apply luma curve
        if self.enable_luma_curve {
            let luma = 0.2126 * result.r + 0.7152 * result.g + 0.0722 * result.b;
            let new_luma = self.luma_curve.evaluate(luma);
            let scale = if luma > 0.0 { new_luma / luma } else { 1.0 };
            result = RgbColor::new(result.r * scale, result.g * scale, result.b * scale);
        }

        // Apply color wheels
        if self.enable_color_wheels {
            result = self.color_wheels.apply(result);
        }

        // Apply hue vs. saturation
        if self.enable_hue_vs_sat {
            result = self.hue_vs_sat.apply(result);
        }

        // Apply hue vs. luminance
        if self.enable_hue_vs_lum {
            result = self.hue_vs_lum.apply(result);
        }

        // Apply qualifier if enabled
        if self.enable_qualifier {
            let mask = self.qualifier.calculate_mask(color);
            result = color.lerp(&result, mask);
        }

        result
    }
}

// ============================================================================
// Color Grading Filter
// ============================================================================

/// Color grading filter node.
pub struct ColorGradingFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    input: InputPort,
    output: OutputPort,
    config: ColorGradingConfig,
}

impl ColorGradingFilter {
    /// Create a new color grading filter.
    #[must_use]
    pub fn new(id: NodeId, name: &str, config: ColorGradingConfig) -> Self {
        Self {
            id,
            name: name.to_string(),
            state: NodeState::Idle,
            input: InputPort::new(PortId(0), "input", PortType::Video),
            output: OutputPort::new(PortId(1), "output", PortType::Video),
            config,
        }
    }

    /// Process a video frame.
    fn process_frame(&self, frame: VideoFrame) -> GraphResult<VideoFrame> {
        // Only process RGB formats for now
        match frame.format {
            PixelFormat::Rgb24 | PixelFormat::Rgba32 => self.process_rgb_frame(frame),
            _ => {
                // Return frame unchanged for unsupported formats
                Ok(frame)
            }
        }
    }

    /// Process an RGB frame.
    fn process_rgb_frame(&self, mut frame: VideoFrame) -> GraphResult<VideoFrame> {
        let width = frame.width;
        let height = frame.height;
        let planes = &mut frame.planes;

        if planes.is_empty() {
            return Ok(frame);
        }

        let plane = &mut planes[0];
        let stride = plane.stride;
        let data = plane.data.as_mut_slice();

        let bytes_per_pixel = match frame.format {
            PixelFormat::Rgb24 => 3,
            PixelFormat::Rgba32 => 4,
            _ => return Ok(frame),
        };

        for y in 0..height as usize {
            for x in 0..width as usize {
                let offset = y * stride + x * bytes_per_pixel;

                // Read color
                let r = data[offset] as f64 / 255.0;
                let g = data[offset + 1] as f64 / 255.0;
                let b = data[offset + 2] as f64 / 255.0;

                let color = RgbColor::new(r, g, b);

                // Apply color grading
                let graded = self.config.apply(color).clamp();

                // Write back
                data[offset] = (graded.r * 255.0) as u8;
                data[offset + 1] = (graded.g * 255.0) as u8;
                data[offset + 2] = (graded.b * 255.0) as u8;
            }
        }

        Ok(frame)
    }
}

impl Node for ColorGradingFilter {
    fn id(&self) -> NodeId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn node_type(&self) -> NodeType {
        NodeType::Filter
    }

    fn state(&self) -> NodeState {
        self.state
    }

    fn set_state(&mut self, state: NodeState) -> GraphResult<()> {
        self.state = state;
        Ok(())
    }

    fn inputs(&self) -> &[InputPort] {
        std::slice::from_ref(&self.input)
    }

    fn outputs(&self) -> &[OutputPort] {
        std::slice::from_ref(&self.output)
    }

    fn process(&mut self, input: Option<FilterFrame>) -> GraphResult<Option<FilterFrame>> {
        match input {
            Some(FilterFrame::Video(video_frame)) => {
                let processed = self.process_frame(video_frame)?;
                Ok(Some(FilterFrame::Video(processed)))
            }
            Some(_) => Err(GraphError::ProcessingError {
                node: self.id,
                message: "Color grading filter expects video input".to_string(),
            }),
            None => Ok(None),
        }
    }
}

// ============================================================================
// Preset Color Grading Looks
// ============================================================================

/// Preset color grading configurations for common looks.
pub mod presets {
    use super::*;

    /// Cinematic look with teal shadows and orange highlights.
    #[must_use]
    pub fn cinematic_teal_orange() -> ColorGradingConfig {
        let mut config = ColorGradingConfig::new();

        // Teal shadows, orange highlights
        config.color_wheels.shadows.hue = 180.0; // Teal
        config.color_wheels.shadows.saturation = 1.3;
        config.color_wheels.highlights.hue = 30.0; // Orange
        config.color_wheels.highlights.saturation = 1.2;

        config.enable_color_wheels = true;

        // Add some contrast
        config.lgg.contrast = 1.15;
        config.enable_lgg = true;

        config
    }

    /// Vintage film look with faded blacks and warm tones.
    #[must_use]
    pub fn vintage_film() -> ColorGradingConfig {
        let mut config = ColorGradingConfig::new();

        // Lift blacks (faded look)
        config.lgg.lift = RgbColor::new(0.1, 0.08, 0.05);
        config.lgg.gamma = RgbColor::new(1.1, 1.0, 0.95);
        config.enable_lgg = true;

        // Warm tones
        config.temp_tint.temperature = 500.0;
        config.enable_temp_tint = true;

        // Reduce saturation slightly
        config.color_wheels.global_saturation = 0.85;
        config.enable_color_wheels = true;

        // Film response curve
        config.rgb_curves.master = Curve::film_response();
        config.enable_rgb_curves = true;

        config
    }

    /// High-contrast black and white.
    #[must_use]
    pub fn black_and_white_contrast() -> ColorGradingConfig {
        let mut config = ColorGradingConfig::new();

        // Remove all color
        config.color_wheels.global_saturation = 0.0;
        config.enable_color_wheels = true;

        // High contrast
        config.lgg.contrast = 1.4;
        config.lgg.pivot = 0.45;
        config.enable_lgg = true;

        // Crushing blacks slightly
        config.lgg.lift = RgbColor::new(-0.05, -0.05, -0.05);

        config
    }

    /// Bleach bypass look (desaturated with crushed highlights).
    #[must_use]
    pub fn bleach_bypass() -> ColorGradingConfig {
        let mut config = ColorGradingConfig::new();

        // Reduced saturation
        config.color_wheels.global_saturation = 0.6;
        config.enable_color_wheels = true;

        // High contrast
        config.lgg.contrast = 1.35;
        config.enable_lgg = true;

        // Crush highlights
        config.rgb_curves.master = Curve::highlights(-0.3);
        config.enable_rgb_curves = true;

        config
    }

    /// Warm sunset look.
    #[must_use]
    pub fn warm_sunset() -> ColorGradingConfig {
        let mut config = ColorGradingConfig::new();

        // Warm temperature
        config.temp_tint.temperature = 2000.0;
        config.enable_temp_tint = true;

        // Orange/red highlights
        config.color_wheels.highlights.hue = 15.0;
        config.color_wheels.highlights.saturation = 1.4;
        config.enable_color_wheels = true;

        // Lift shadows slightly
        config.lgg.master_lift = 0.05;
        config.enable_lgg = true;

        config
    }

    /// Cool moonlight look.
    #[must_use]
    pub fn cool_moonlight() -> ColorGradingConfig {
        let mut config = ColorGradingConfig::new();

        // Cool temperature
        config.temp_tint.temperature = -3000.0;
        config.enable_temp_tint = true;

        // Blue shadows
        config.color_wheels.shadows.hue = 220.0; // Blue
        config.color_wheels.shadows.saturation = 1.3;
        config.enable_color_wheels = true;

        // Reduce overall saturation
        config.color_wheels.global_saturation = 0.75;

        // Darken slightly
        config.lgg.master_lift = -0.1;
        config.enable_lgg = true;

        config
    }

    /// Cross-processed look (shifted colors).
    #[must_use]
    pub fn cross_process() -> ColorGradingConfig {
        let mut config = ColorGradingConfig::new();

        // Shifted color curves
        config.rgb_curves.red = Curve::cross_process(0.8);
        config.rgb_curves.green = Curve::cross_process(0.6);
        config.rgb_curves.blue = Curve::cross_process(0.7);
        config.enable_rgb_curves = true;

        // Increased saturation
        config.color_wheels.global_saturation = 1.3;
        config.enable_color_wheels = true;

        config
    }

    /// HDR look (enhanced dynamic range appearance).
    #[must_use]
    pub fn hdr_look() -> ColorGradingConfig {
        let mut config = ColorGradingConfig::new();

        // Lift shadows significantly
        config.lgg.lift = RgbColor::new(0.15, 0.15, 0.15);
        config.enable_lgg = true;

        // Compress highlights
        config.rgb_curves.master = Curve::highlights(-0.4);
        config.enable_rgb_curves = true;

        // Boost saturation
        config.color_wheels.global_saturation = 1.4;
        config.enable_color_wheels = true;

        config
    }

    /// Day for night conversion.
    #[must_use]
    pub fn day_for_night() -> ColorGradingConfig {
        let mut config = ColorGradingConfig::new();

        // Darken significantly
        config.lgg.master_gain = 0.4;
        config.enable_lgg = true;

        // Cool blue tones
        config.temp_tint.temperature = -4000.0;
        config.enable_temp_tint = true;

        // Blue everything
        config.color_wheels.midtones.hue = 220.0;
        config.color_wheels.midtones.saturation = 1.2;
        config.color_wheels.offset_mode = true;
        config.enable_color_wheels = true;

        // Reduce overall saturation
        config.color_wheels.global_saturation = 0.6;

        config
    }

    /// Sepia tone (classic brown/yellow tinted look).
    #[must_use]
    pub fn sepia() -> ColorGradingConfig {
        let mut config = ColorGradingConfig::new();

        // Remove color first
        config.color_wheels.global_saturation = 0.0;
        config.enable_color_wheels = true;

        // Add sepia tones through offset mode
        config.color_wheels.offset_mode = true;
        config.color_wheels.midtones.hue = 40.0; // Yellow-orange
        config.color_wheels.midtones.saturation = 1.5;
        config.color_wheels.midtones.luminance = 0.0;

        // Lift blacks for faded look
        config.lgg.lift = RgbColor::new(0.08, 0.06, 0.03);
        config.enable_lgg = true;

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_hsl_and_back() {
        let rgb = RgbColor::new(0.5, 0.3, 0.7);
        let hsl = rgb.to_hsl();
        let rgb2 = hsl.to_rgb();

        assert!((rgb.r - rgb2.r).abs() < 0.01);
        assert!((rgb.g - rgb2.g).abs() < 0.01);
        assert!((rgb.b - rgb2.b).abs() < 0.01);
    }

    #[test]
    fn test_lift_gamma_gain() {
        let lgg = LiftGammaGain::new()
            .with_lift(0.1, 0.0, 0.0)
            .with_gamma(1.2, 1.0, 1.0)
            .with_gain(1.1, 1.0, 1.0);

        let color = RgbColor::new(0.5, 0.5, 0.5);
        let result = lgg.apply(color);

        // Lift should increase shadows
        assert!(result.r > color.r);
    }

    #[test]
    fn test_curve_evaluation() {
        let curve = Curve::linear();
        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.01);
        assert!((curve.evaluate(0.5) - 0.5).abs() < 0.01);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hsl_qualifier() {
        let qualifier = HslQualifier::new()
            .with_hue_range(0.0, 60.0)
            .with_feather(10.0);

        // Red color (hue ~0)
        let red = RgbColor::new(1.0, 0.0, 0.0);
        let mask_red = qualifier.calculate_mask(red);
        assert!(mask_red > 0.5);

        // Blue color (hue ~240)
        let blue = RgbColor::new(0.0, 0.0, 1.0);
        let mask_blue = qualifier.calculate_mask(blue);
        assert!(mask_blue < 0.5);
    }
}
