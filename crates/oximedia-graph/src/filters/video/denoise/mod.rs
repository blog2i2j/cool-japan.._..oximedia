//! Video noise reduction filter.
//!
//! This filter provides comprehensive spatial and temporal noise reduction
//! for video frames using various advanced techniques.
//!
//! # Features
//!
//! ## Spatial Noise Reduction:
//! - Bilateral filtering (edge-preserving)
//! - Non-local means (NLM) denoising
//! - Gaussian filtering
//! - Median filtering
//! - Adaptive filtering based on local variance
//!
//! ## Temporal Noise Reduction:
//! - Motion-compensated temporal filtering
//! - Recursive temporal averaging
//! - 3D block matching (BM3D-inspired)
//! - Weighted temporal averaging
//!
//! ## Advanced Features:
//! - Separate chroma and luma processing
//! - Edge-preserving noise reduction
//! - Adaptive strength based on content
//! - Multi-frame temporal coherence
//! - Configurable strength, radius, and temporal depth
//!
//! # Example
//!
//! ```ignore
//! use oximedia_graph::filters::video::{DenoiseFilter, DenoiseConfig, DenoiseMethod};
//! use oximedia_graph::node::NodeId;
//!
//! // Create a denoise filter with bilateral filtering
//! let config = DenoiseConfig::new()
//!     .with_method(DenoiseMethod::Bilateral)
//!     .with_strength(0.8)
//!     .with_spatial_radius(5)
//!     .with_temporal_depth(3);
//!
//! let filter = DenoiseFilter::new(NodeId(0), "denoise", config);
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::bool_to_int_with_if)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::no_effect_underscore_binding)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use std::collections::VecDeque;

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortFormat, PortId, PortType, VideoPortFormat};
use oximedia_codec::{Plane, VideoFrame};

use noise_stats::NoiseStatistics;

pub mod advanced;
pub mod analysis;
pub mod bench;
pub mod color_space;
pub mod edge;
pub mod gpu;
pub mod metrics;
pub(super) mod noise_stats;
pub mod presets;
pub mod temporal;

/// Noise reduction method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DenoiseMethod {
    /// Bilateral filtering (edge-preserving spatial filter).
    #[default]
    Bilateral,
    /// Non-local means denoising.
    NonLocalMeans,
    /// Gaussian blur.
    Gaussian,
    /// Median filtering.
    Median,
    /// Adaptive filtering based on local variance.
    Adaptive,
    /// Temporal averaging.
    Temporal,
    /// Motion-compensated temporal filtering.
    MotionCompensated,
    /// 3D block matching (BM3D-inspired).
    BlockMatching3D,
    /// Combined spatial and temporal.
    Combined,
}

/// Temporal filter mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TemporalMode {
    /// Simple temporal averaging.
    #[default]
    Average,
    /// Recursive temporal filtering.
    Recursive,
    /// Motion-compensated.
    MotionCompensated,
    /// Weighted averaging based on similarity.
    WeightedAverage,
}

/// Motion estimation quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MotionQuality {
    /// Fast motion estimation (larger block size, limited search).
    Fast,
    /// Balanced quality and speed.
    #[default]
    Medium,
    /// High quality (smaller blocks, larger search).
    High,
}

/// Configuration for the denoise filter.
#[derive(Clone, Debug)]
pub struct DenoiseConfig {
    /// Primary denoising method.
    pub method: DenoiseMethod,
    /// Overall noise reduction strength (0.0-1.0).
    pub strength: f32,
    /// Luma (Y) plane strength multiplier.
    pub luma_strength: f32,
    /// Chroma (U/V) plane strength multiplier.
    pub chroma_strength: f32,
    /// Spatial filter radius (pixels).
    pub spatial_radius: u32,
    /// Temporal depth (number of frames to use).
    pub temporal_depth: usize,
    /// Temporal filter mode.
    pub temporal_mode: TemporalMode,
    /// Motion estimation quality.
    pub motion_quality: MotionQuality,
    /// Sigma for color/range in bilateral filter.
    pub sigma_color: f32,
    /// Sigma for space/distance in bilateral filter.
    pub sigma_space: f32,
    /// Search window size for NLM and motion estimation.
    pub search_window: u32,
    /// Patch size for NLM and block matching.
    pub patch_size: u32,
    /// NLM filtering strength parameter (h).
    pub nlm_h: f32,
    /// Enable edge preservation.
    pub preserve_edges: bool,
    /// Edge threshold for adaptive filtering.
    pub edge_threshold: f32,
    /// Enable adaptive strength based on noise level.
    pub adaptive_strength: bool,
    /// Recursive filter alpha (for temporal recursive mode).
    pub recursive_alpha: f32,
}

impl DenoiseConfig {
    /// Create a new denoise configuration with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            method: DenoiseMethod::Bilateral,
            strength: 0.7,
            luma_strength: 1.0,
            chroma_strength: 1.2,
            spatial_radius: 5,
            temporal_depth: 3,
            temporal_mode: TemporalMode::Average,
            motion_quality: MotionQuality::Medium,
            sigma_color: 50.0,
            sigma_space: 10.0,
            search_window: 21,
            patch_size: 7,
            nlm_h: 10.0,
            preserve_edges: true,
            edge_threshold: 30.0,
            adaptive_strength: true,
            recursive_alpha: 0.3,
        }
    }

    /// Set the denoising method.
    #[must_use]
    pub fn with_method(mut self, method: DenoiseMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the overall strength.
    #[must_use]
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set luma strength multiplier.
    #[must_use]
    pub fn with_luma_strength(mut self, strength: f32) -> Self {
        self.luma_strength = strength.clamp(0.0, 2.0);
        self
    }

    /// Set chroma strength multiplier.
    #[must_use]
    pub fn with_chroma_strength(mut self, strength: f32) -> Self {
        self.chroma_strength = strength.clamp(0.0, 2.0);
        self
    }

    /// Set spatial filter radius.
    #[must_use]
    pub fn with_spatial_radius(mut self, radius: u32) -> Self {
        self.spatial_radius = radius.clamp(1, 31);
        self
    }

    /// Set temporal depth.
    #[must_use]
    pub fn with_temporal_depth(mut self, depth: usize) -> Self {
        self.temporal_depth = depth.clamp(1, 10);
        self
    }

    /// Set temporal mode.
    #[must_use]
    pub fn with_temporal_mode(mut self, mode: TemporalMode) -> Self {
        self.temporal_mode = mode;
        self
    }

    /// Set motion estimation quality.
    #[must_use]
    pub fn with_motion_quality(mut self, quality: MotionQuality) -> Self {
        self.motion_quality = quality;
        self
    }

    /// Set bilateral filter parameters.
    #[must_use]
    pub fn with_bilateral_params(mut self, sigma_color: f32, sigma_space: f32) -> Self {
        self.sigma_color = sigma_color;
        self.sigma_space = sigma_space;
        self
    }

    /// Set NLM parameters.
    #[must_use]
    pub fn with_nlm_params(mut self, h: f32, search_window: u32, patch_size: u32) -> Self {
        self.nlm_h = h;
        self.search_window = search_window;
        self.patch_size = patch_size;
        self
    }

    /// Enable/disable edge preservation.
    #[must_use]
    pub fn with_edge_preservation(mut self, enabled: bool) -> Self {
        self.preserve_edges = enabled;
        self
    }

    /// Set edge threshold.
    #[must_use]
    pub fn with_edge_threshold(mut self, threshold: f32) -> Self {
        self.edge_threshold = threshold;
        self
    }

    /// Enable/disable adaptive strength.
    #[must_use]
    pub fn with_adaptive_strength(mut self, enabled: bool) -> Self {
        self.adaptive_strength = enabled;
        self
    }

    /// Set recursive filter alpha.
    #[must_use]
    pub fn with_recursive_alpha(mut self, alpha: f32) -> Self {
        self.recursive_alpha = alpha.clamp(0.0, 1.0);
        self
    }
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Video denoise filter.
///
/// Removes noise from video frames using spatial and/or temporal filtering
/// techniques. Supports various methods and can process luma and chroma
/// planes with different strengths.
pub struct DenoiseFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
    config: DenoiseConfig,
    /// Frame buffer for temporal processing.
    frame_buffer: VecDeque<VideoFrame>,
    /// Previous frame for recursive temporal filtering.
    prev_frame: Option<VideoFrame>,
    /// Noise statistics for adaptive processing.
    noise_stats: Option<NoiseStatistics>,
}

impl DenoiseFilter {
    /// Create a new denoise filter.
    #[must_use]
    pub fn new(id: NodeId, name: impl Into<String>, config: DenoiseConfig) -> Self {
        Self {
            id,
            name: name.into(),
            state: NodeState::Idle,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Video)
                .with_format(PortFormat::Video(VideoPortFormat::any()))],
            outputs: vec![OutputPort::new(PortId(0), "output", PortType::Video)
                .with_format(PortFormat::Video(VideoPortFormat::any()))],
            config,
            frame_buffer: VecDeque::new(),
            prev_frame: None,
            noise_stats: None,
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &DenoiseConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: DenoiseConfig) {
        self.config = config;
        self.frame_buffer.clear();
        self.prev_frame = None;
        self.noise_stats = None;
    }

    /// Process a single frame.
    fn process_frame(&mut self, mut frame: VideoFrame) -> GraphResult<VideoFrame> {
        // Update temporal buffer
        if self.needs_temporal_processing() {
            self.frame_buffer.push_back(frame.clone());
            if self.frame_buffer.len() > self.config.temporal_depth * 2 + 1 {
                self.frame_buffer.pop_front();
            }
        }

        // Estimate noise if adaptive strength is enabled
        if self.config.adaptive_strength && self.noise_stats.is_none() {
            self.noise_stats = Some(NoiseStatistics::estimate(&frame));
        }

        // Process frame based on method
        match self.config.method {
            DenoiseMethod::Bilateral => {
                self.apply_bilateral(&mut frame)?;
            }
            DenoiseMethod::NonLocalMeans => {
                self.apply_nlm(&mut frame)?;
            }
            DenoiseMethod::Gaussian => {
                self.apply_gaussian(&mut frame)?;
            }
            DenoiseMethod::Median => {
                self.apply_median(&mut frame)?;
            }
            DenoiseMethod::Adaptive => {
                self.apply_adaptive(&mut frame)?;
            }
            DenoiseMethod::Temporal => {
                self.apply_temporal(&mut frame)?;
            }
            DenoiseMethod::MotionCompensated => {
                self.apply_motion_compensated(&mut frame)?;
            }
            DenoiseMethod::BlockMatching3D => {
                self.apply_bm3d(&mut frame)?;
            }
            DenoiseMethod::Combined => {
                self.apply_combined(&mut frame)?;
            }
        }

        // Update previous frame for recursive temporal
        if self.config.temporal_mode == TemporalMode::Recursive {
            self.prev_frame = Some(frame.clone());
        }

        Ok(frame)
    }

    /// Check if temporal processing is needed.
    fn needs_temporal_processing(&self) -> bool {
        matches!(
            self.config.method,
            DenoiseMethod::Temporal
                | DenoiseMethod::MotionCompensated
                | DenoiseMethod::BlockMatching3D
                | DenoiseMethod::Combined
        )
    }

    /// Apply bilateral filtering.
    fn apply_bilateral(&self, frame: &mut VideoFrame) -> GraphResult<()> {
        let (h_sub, v_sub) = frame.format.chroma_subsampling();

        for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
            let (width, height) = if plane_idx == 0 {
                (frame.width, frame.height)
            } else {
                (frame.width / h_sub, frame.height / v_sub)
            };

            let strength = self.get_plane_strength(plane_idx);
            let sigma_color = self.config.sigma_color * strength;
            let sigma_space = self.config.sigma_space;
            let radius = self.config.spatial_radius;

            self.bilateral_filter_plane(plane, width, height, radius, sigma_color, sigma_space)?;
        }

        Ok(())
    }

    /// Apply bilateral filter to a single plane.
    fn bilateral_filter_plane(
        &self,
        plane: &mut Plane,
        width: u32,
        height: u32,
        radius: u32,
        sigma_color: f32,
        sigma_space: f32,
    ) -> GraphResult<()> {
        let mut data = plane.data.to_vec();
        let original = data.clone();

        let color_coeff = -0.5 / (sigma_color * sigma_color);
        let space_coeff = -0.5 / (sigma_space * sigma_space);

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let center_val = original.get(idx).copied().unwrap_or(128) as f32;

                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                let y_min = y.saturating_sub(radius);
                let y_max = (y + radius + 1).min(height);
                let x_min = x.saturating_sub(radius);
                let x_max = (x + radius + 1).min(width);

                for ny in y_min..y_max {
                    for nx in x_min..x_max {
                        let nidx = (ny * width + nx) as usize;
                        let neighbor_val = original.get(nidx).copied().unwrap_or(128) as f32;

                        let color_dist = neighbor_val - center_val;
                        let space_dist =
                            ((nx as i32 - x as i32).pow(2) + (ny as i32 - y as i32).pow(2)) as f32;

                        let color_weight = (color_dist * color_dist * color_coeff).exp();
                        let space_weight = (space_dist * space_coeff).exp();
                        let weight = color_weight * space_weight;

                        sum += neighbor_val * weight;
                        weight_sum += weight;
                    }
                }

                if weight_sum > 0.0 {
                    data[idx] = (sum / weight_sum).round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        *plane = Plane::new(data, plane.stride);
        Ok(())
    }

    /// Apply non-local means denoising.
    fn apply_nlm(&self, frame: &mut VideoFrame) -> GraphResult<()> {
        let (h_sub, v_sub) = frame.format.chroma_subsampling();

        for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
            let (width, height) = if plane_idx == 0 {
                (frame.width, frame.height)
            } else {
                (frame.width / h_sub, frame.height / v_sub)
            };

            let strength = self.get_plane_strength(plane_idx);
            let h = self.config.nlm_h * strength;

            self.nlm_filter_plane(plane, width, height, h)?;
        }

        Ok(())
    }

    /// Apply NLM filter to a single plane.
    fn nlm_filter_plane(
        &self,
        plane: &mut Plane,
        width: u32,
        height: u32,
        h: f32,
    ) -> GraphResult<()> {
        let mut data = plane.data.to_vec();
        let original = data.clone();

        let search_radius = self.config.search_window / 2;
        let patch_radius = self.config.patch_size / 2;
        let h_sq = h * h;

        for y in patch_radius..(height - patch_radius) {
            for x in patch_radius..(width - patch_radius) {
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                let y_min = y.saturating_sub(search_radius);
                let y_max = (y + search_radius + 1).min(height - patch_radius);
                let x_min = x.saturating_sub(search_radius);
                let x_max = (x + search_radius + 1).min(width - patch_radius);

                for sy in y_min..y_max {
                    for sx in x_min..x_max {
                        let dist =
                            self.patch_distance(&original, x, y, sx, sy, width, patch_radius);

                        let weight = (-dist / h_sq).exp();
                        let sidx = (sy * width + sx) as usize;
                        sum += original.get(sidx).copied().unwrap_or(128) as f32 * weight;
                        weight_sum += weight;
                    }
                }

                let idx = (y * width + x) as usize;
                if weight_sum > 0.0 {
                    data[idx] = (sum / weight_sum).round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        *plane = Plane::new(data, plane.stride);
        Ok(())
    }

    /// Compute patch distance.
    fn patch_distance(
        &self,
        data: &[u8],
        x1: u32,
        y1: u32,
        x2: u32,
        y2: u32,
        width: u32,
        radius: u32,
    ) -> f32 {
        let mut dist = 0.0f32;
        let mut count = 0;

        for dy in 0..=radius * 2 {
            for dx in 0..=radius * 2 {
                let px1 = x1 + dx - radius;
                let py1 = y1 + dy - radius;
                let px2 = x2 + dx - radius;
                let py2 = y2 + dy - radius;

                let idx1 = (py1 * width + px1) as usize;
                let idx2 = (py2 * width + px2) as usize;

                let v1 = data.get(idx1).copied().unwrap_or(128) as f32;
                let v2 = data.get(idx2).copied().unwrap_or(128) as f32;

                let diff = v1 - v2;
                dist += diff * diff;
                count += 1;
            }
        }

        if count > 0 {
            dist / count as f32
        } else {
            0.0
        }
    }

    /// Apply Gaussian filtering.
    fn apply_gaussian(&self, frame: &mut VideoFrame) -> GraphResult<()> {
        let (h_sub, v_sub) = frame.format.chroma_subsampling();

        for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
            let (width, height) = if plane_idx == 0 {
                (frame.width, frame.height)
            } else {
                (frame.width / h_sub, frame.height / v_sub)
            };

            let strength = self.get_plane_strength(plane_idx);
            let sigma = self.config.sigma_space * strength;
            let radius = self.config.spatial_radius;

            self.gaussian_filter_plane(plane, width, height, radius, sigma)?;
        }

        Ok(())
    }

    /// Apply Gaussian filter to a single plane.
    fn gaussian_filter_plane(
        &self,
        plane: &mut Plane,
        width: u32,
        height: u32,
        radius: u32,
        sigma: f32,
    ) -> GraphResult<()> {
        let kernel = create_gaussian_kernel(radius as usize, sigma);
        let mut data = plane.data.to_vec();
        let original = data.clone();

        for y in 0..height {
            for x in 0..width {
                let value = self.apply_kernel(&original, x, y, width, height, &kernel, radius);
                let idx = (y * width + x) as usize;
                data[idx] = value;
            }
        }

        *plane = Plane::new(data, plane.stride);
        Ok(())
    }

    /// Apply convolution kernel.
    fn apply_kernel(
        &self,
        data: &[u8],
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        kernel: &[f32],
        radius: u32,
    ) -> u8 {
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;
        let ksize = (radius * 2 + 1) as usize;

        for ky in 0..ksize {
            let py = y as i32 + ky as i32 - radius as i32;
            if py < 0 || py >= height as i32 {
                continue;
            }

            for kx in 0..ksize {
                let px = x as i32 + kx as i32 - radius as i32;
                if px < 0 || px >= width as i32 {
                    continue;
                }

                let idx = (py as u32 * width + px as u32) as usize;
                let weight = kernel[ky * ksize + kx];
                sum += data.get(idx).copied().unwrap_or(128) as f32 * weight;
                weight_sum += weight;
            }
        }

        if weight_sum > 0.0 {
            (sum / weight_sum).round().clamp(0.0, 255.0) as u8
        } else {
            128
        }
    }

    /// Apply median filtering.
    fn apply_median(&self, frame: &mut VideoFrame) -> GraphResult<()> {
        let (h_sub, v_sub) = frame.format.chroma_subsampling();

        for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
            let (width, height) = if plane_idx == 0 {
                (frame.width, frame.height)
            } else {
                (frame.width / h_sub, frame.height / v_sub)
            };

            let radius = self.config.spatial_radius;
            self.median_filter_plane(plane, width, height, radius)?;
        }

        Ok(())
    }

    /// Apply median filter to a single plane.
    fn median_filter_plane(
        &self,
        plane: &mut Plane,
        width: u32,
        height: u32,
        radius: u32,
    ) -> GraphResult<()> {
        let mut data = plane.data.to_vec();
        let original = data.clone();

        for y in 0..height {
            for x in 0..width {
                let mut values = Vec::new();

                let y_min = y.saturating_sub(radius);
                let y_max = (y + radius + 1).min(height);
                let x_min = x.saturating_sub(radius);
                let x_max = (x + radius + 1).min(width);

                for ny in y_min..y_max {
                    for nx in x_min..x_max {
                        let nidx = (ny * width + nx) as usize;
                        values.push(original.get(nidx).copied().unwrap_or(128));
                    }
                }

                values.sort_unstable();
                let median = if values.is_empty() {
                    128
                } else {
                    values[values.len() / 2]
                };

                let idx = (y * width + x) as usize;
                data[idx] = median;
            }
        }

        *plane = Plane::new(data, plane.stride);
        Ok(())
    }

    /// Apply adaptive filtering based on local variance.
    fn apply_adaptive(&self, frame: &mut VideoFrame) -> GraphResult<()> {
        let (h_sub, v_sub) = frame.format.chroma_subsampling();

        for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
            let (width, height) = if plane_idx == 0 {
                (frame.width, frame.height)
            } else {
                (frame.width / h_sub, frame.height / v_sub)
            };

            self.adaptive_filter_plane(plane, width, height)?;
        }

        Ok(())
    }

    /// Apply adaptive filter to a single plane.
    fn adaptive_filter_plane(&self, plane: &mut Plane, width: u32, height: u32) -> GraphResult<()> {
        let mut data = plane.data.to_vec();
        let original = data.clone();

        let radius = self.config.spatial_radius;

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let center_val = original.get(idx).copied().unwrap_or(128);

                let (_mean, variance) =
                    self.compute_local_statistics(&original, x, y, width, height, radius);

                let adaptive_strength = if variance < self.config.edge_threshold {
                    self.config.strength
                } else {
                    self.config.strength * 0.3
                };

                let filtered = self.bilateral_filter_pixel(
                    &original,
                    x,
                    y,
                    width,
                    height,
                    radius,
                    self.config.sigma_color,
                    self.config.sigma_space,
                );

                let blended = center_val as f32 * (1.0 - adaptive_strength)
                    + filtered as f32 * adaptive_strength;
                data[idx] = blended.round().clamp(0.0, 255.0) as u8;
            }
        }

        *plane = Plane::new(data, plane.stride);
        Ok(())
    }

    /// Compute local statistics (mean and variance).
    fn compute_local_statistics(
        &self,
        data: &[u8],
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        radius: u32,
    ) -> (f32, f32) {
        let mut sum = 0.0f32;
        let mut sq_sum = 0.0f32;
        let mut count = 0;

        let y_min = y.saturating_sub(radius);
        let y_max = (y + radius + 1).min(height);
        let x_min = x.saturating_sub(radius);
        let x_max = (x + radius + 1).min(width);

        for ny in y_min..y_max {
            for nx in x_min..x_max {
                let nidx = (ny * width + nx) as usize;
                let val = data.get(nidx).copied().unwrap_or(128) as f32;
                sum += val;
                sq_sum += val * val;
                count += 1;
            }
        }

        if count > 0 {
            let mean = sum / count as f32;
            let variance = (sq_sum / count as f32) - (mean * mean);
            (mean, variance)
        } else {
            (128.0, 0.0)
        }
    }

    /// Bilateral filter for a single pixel.
    fn bilateral_filter_pixel(
        &self,
        data: &[u8],
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        radius: u32,
        sigma_color: f32,
        sigma_space: f32,
    ) -> u8 {
        let center_val = data.get((y * width + x) as usize).copied().unwrap_or(128) as f32;

        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;

        let color_coeff = -0.5 / (sigma_color * sigma_color);
        let space_coeff = -0.5 / (sigma_space * sigma_space);

        let y_min = y.saturating_sub(radius);
        let y_max = (y + radius + 1).min(height);
        let x_min = x.saturating_sub(radius);
        let x_max = (x + radius + 1).min(width);

        for ny in y_min..y_max {
            for nx in x_min..x_max {
                let nidx = (ny * width + nx) as usize;
                let neighbor_val = data.get(nidx).copied().unwrap_or(128) as f32;

                let color_dist = neighbor_val - center_val;
                let space_dist =
                    ((nx as i32 - x as i32).pow(2) + (ny as i32 - y as i32).pow(2)) as f32;

                let color_weight = (color_dist * color_dist * color_coeff).exp();
                let space_weight = (space_dist * space_coeff).exp();
                let weight = color_weight * space_weight;

                sum += neighbor_val * weight;
                weight_sum += weight;
            }
        }

        if weight_sum > 0.0 {
            (sum / weight_sum).round().clamp(0.0, 255.0) as u8
        } else {
            center_val as u8
        }
    }

    /// Apply temporal filtering.
    fn apply_temporal(&mut self, frame: &mut VideoFrame) -> GraphResult<()> {
        if self.frame_buffer.is_empty() {
            return Ok(());
        }

        match self.config.temporal_mode {
            TemporalMode::Average => self.temporal_average(frame),
            TemporalMode::Recursive => self.temporal_recursive(frame),
            TemporalMode::MotionCompensated => self.temporal_motion_compensated(frame),
            TemporalMode::WeightedAverage => self.temporal_weighted(frame),
        }
    }

    /// Simple temporal averaging.
    fn temporal_average(&self, frame: &mut VideoFrame) -> GraphResult<()> {
        let (h_sub, v_sub) = frame.format.chroma_subsampling();

        for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
            let (width, height) = if plane_idx == 0 {
                (frame.width, frame.height)
            } else {
                (frame.width / h_sub, frame.height / v_sub)
            };

            let mut data = plane.data.to_vec();

            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) as usize;
                    let mut sum = data.get(idx).copied().unwrap_or(128) as f32;
                    let mut count = 1;

                    for buffered_frame in &self.frame_buffer {
                        if let Some(buffered_plane) = buffered_frame.planes.get(plane_idx) {
                            sum += buffered_plane.data.get(idx).copied().unwrap_or(128) as f32;
                            count += 1;
                        }
                    }

                    data[idx] = (sum / count as f32).round().clamp(0.0, 255.0) as u8;
                }
            }

            *plane = Plane::new(data, plane.stride);
        }

        Ok(())
    }

    /// Recursive temporal filtering.
    fn temporal_recursive(&self, frame: &mut VideoFrame) -> GraphResult<()> {
        if let Some(ref prev) = self.prev_frame {
            let (h_sub, v_sub) = frame.format.chroma_subsampling();
            let alpha = self.config.recursive_alpha;

            for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
                let (width, height) = if plane_idx == 0 {
                    (frame.width, frame.height)
                } else {
                    (frame.width / h_sub, frame.height / v_sub)
                };

                let mut data = plane.data.to_vec();

                if let Some(prev_plane) = prev.planes.get(plane_idx) {
                    for idx in 0..(width * height) as usize {
                        let current = data.get(idx).copied().unwrap_or(128) as f32;
                        let previous = prev_plane.data.get(idx).copied().unwrap_or(128) as f32;

                        let filtered = current * alpha + previous * (1.0 - alpha);
                        data[idx] = filtered.round().clamp(0.0, 255.0) as u8;
                    }
                }

                *plane = Plane::new(data, plane.stride);
            }
        }

        Ok(())
    }

    /// Motion-compensated temporal filtering.
    fn temporal_motion_compensated(&self, frame: &mut VideoFrame) -> GraphResult<()> {
        if self.frame_buffer.is_empty() {
            return Ok(());
        }

        let (h_sub, v_sub) = frame.format.chroma_subsampling();

        for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
            let (width, height) = if plane_idx == 0 {
                (frame.width, frame.height)
            } else {
                (frame.width / h_sub, frame.height / v_sub)
            };

            let mut data = plane.data.to_vec();
            let current_data = data.clone();

            let block_size = match self.config.motion_quality {
                MotionQuality::Fast => 16,
                MotionQuality::Medium => 8,
                MotionQuality::High => 4,
            };

            for by in (0..height).step_by(block_size as usize) {
                for bx in (0..width).step_by(block_size as usize) {
                    let bw = block_size.min(width - bx);
                    let bh = block_size.min(height - by);

                    let mut sum_block = vec![0.0f32; (bw * bh) as usize];
                    let mut count = 1;

                    for py in 0..bh {
                        for px in 0..bw {
                            let idx = ((by + py) * width + (bx + px)) as usize;
                            sum_block[(py * bw + px) as usize] =
                                current_data.get(idx).copied().unwrap_or(128) as f32;
                        }
                    }

                    for buffered_frame in &self.frame_buffer {
                        if let Some(buffered_plane) = buffered_frame.planes.get(plane_idx) {
                            let (mv_x, mv_y) = self.estimate_motion(
                                &current_data,
                                &buffered_plane.data,
                                bx,
                                by,
                                bw,
                                bh,
                                width,
                                height,
                            );

                            for py in 0..bh {
                                for px in 0..bw {
                                    let src_x = (bx as i32 + px as i32 + mv_x)
                                        .clamp(0, width as i32 - 1)
                                        as u32;
                                    let src_y = (by as i32 + py as i32 + mv_y)
                                        .clamp(0, height as i32 - 1)
                                        as u32;
                                    let src_idx = (src_y * width + src_x) as usize;

                                    sum_block[(py * bw + px) as usize] +=
                                        buffered_plane.data.get(src_idx).copied().unwrap_or(128)
                                            as f32;
                                }
                            }
                            count += 1;
                        }
                    }

                    for py in 0..bh {
                        for px in 0..bw {
                            let idx = ((by + py) * width + (bx + px)) as usize;
                            let avg = sum_block[(py * bw + px) as usize] / count as f32;
                            data[idx] = avg.round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }

            *plane = Plane::new(data, plane.stride);
        }

        Ok(())
    }

    /// Estimate motion vector for a block.
    fn estimate_motion(
        &self,
        current: &[u8],
        reference: &[u8],
        bx: u32,
        by: u32,
        bw: u32,
        bh: u32,
        width: u32,
        height: u32,
    ) -> (i32, i32) {
        let search_range = match self.config.motion_quality {
            MotionQuality::Fast => 8,
            MotionQuality::Medium => 16,
            MotionQuality::High => 32,
        };

        let mut best_mv = (0i32, 0i32);
        let mut best_sad = f32::INFINITY;

        for mv_y in -search_range..=search_range {
            for mv_x in -search_range..=search_range {
                let ref_x = bx as i32 + mv_x;
                let ref_y = by as i32 + mv_y;

                if ref_x < 0
                    || ref_y < 0
                    || ref_x + bw as i32 > width as i32
                    || ref_y + bh as i32 > height as i32
                {
                    continue;
                }

                let sad = self.compute_sad(
                    current,
                    reference,
                    bx,
                    by,
                    ref_x as u32,
                    ref_y as u32,
                    bw,
                    bh,
                    width,
                );

                if sad < best_sad {
                    best_sad = sad;
                    best_mv = (mv_x, mv_y);
                }
            }
        }

        best_mv
    }

    /// Compute sum of absolute differences (SAD).
    fn compute_sad(
        &self,
        current: &[u8],
        reference: &[u8],
        cur_x: u32,
        cur_y: u32,
        ref_x: u32,
        ref_y: u32,
        bw: u32,
        bh: u32,
        width: u32,
    ) -> f32 {
        let mut sad = 0.0f32;

        for y in 0..bh {
            for x in 0..bw {
                let cur_idx = ((cur_y + y) * width + (cur_x + x)) as usize;
                let ref_idx = ((ref_y + y) * width + (ref_x + x)) as usize;

                let cur_val = current.get(cur_idx).copied().unwrap_or(128) as f32;
                let ref_val = reference.get(ref_idx).copied().unwrap_or(128) as f32;

                sad += (cur_val - ref_val).abs();
            }
        }

        sad
    }

    /// Weighted temporal averaging based on similarity.
    fn temporal_weighted(&self, frame: &mut VideoFrame) -> GraphResult<()> {
        if self.frame_buffer.is_empty() {
            return Ok(());
        }

        let (h_sub, v_sub) = frame.format.chroma_subsampling();

        for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
            let (width, height) = if plane_idx == 0 {
                (frame.width, frame.height)
            } else {
                (frame.width / h_sub, frame.height / v_sub)
            };

            let mut data = plane.data.to_vec();
            let current_data = data.clone();

            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) as usize;
                    let current_val = current_data.get(idx).copied().unwrap_or(128) as f32;

                    let mut sum = current_val;
                    let mut weight_sum = 1.0f32;

                    for buffered_frame in &self.frame_buffer {
                        if let Some(buffered_plane) = buffered_frame.planes.get(plane_idx) {
                            let buffered_val =
                                buffered_plane.data.get(idx).copied().unwrap_or(128) as f32;

                            let diff = (current_val - buffered_val).abs();
                            let weight = (-diff / (self.config.sigma_color * 0.5)).exp();

                            sum += buffered_val * weight;
                            weight_sum += weight;
                        }
                    }

                    data[idx] = (sum / weight_sum).round().clamp(0.0, 255.0) as u8;
                }
            }

            *plane = Plane::new(data, plane.stride);
        }

        Ok(())
    }

    /// Apply motion-compensated temporal filtering (same as in temporal mode).
    fn apply_motion_compensated(&mut self, frame: &mut VideoFrame) -> GraphResult<()> {
        self.temporal_motion_compensated(frame)
    }

    /// Apply BM3D-inspired denoising.
    fn apply_bm3d(&self, frame: &mut VideoFrame) -> GraphResult<()> {
        let (h_sub, v_sub) = frame.format.chroma_subsampling();

        for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
            let (width, height) = if plane_idx == 0 {
                (frame.width, frame.height)
            } else {
                (frame.width / h_sub, frame.height / v_sub)
            };

            self.bm3d_filter_plane(plane, width, height)?;
        }

        Ok(())
    }

    /// Apply BM3D filter to a single plane.
    fn bm3d_filter_plane(&self, plane: &mut Plane, width: u32, height: u32) -> GraphResult<()> {
        let mut data = plane.data.to_vec();
        let original = data.clone();

        let block_size = self.config.patch_size;
        let search_window = self.config.search_window;
        let max_similar_blocks = 16;

        for by in (0..height).step_by(block_size as usize) {
            for bx in (0..width).step_by(block_size as usize) {
                let bw = block_size.min(width - bx);
                let bh = block_size.min(height - by);

                let similar_blocks = self.find_similar_blocks(
                    &original,
                    bx,
                    by,
                    bw,
                    bh,
                    width,
                    height,
                    search_window,
                    max_similar_blocks,
                );

                for py in 0..bh {
                    for px in 0..bw {
                        let idx = ((by + py) * width + (bx + px)) as usize;
                        let mut sum = 0.0f32;
                        let mut weight_sum = 0.0f32;

                        for (block_x, block_y, weight) in &similar_blocks {
                            let src_idx = ((block_y + py) * width + (block_x + px)) as usize;
                            sum += original.get(src_idx).copied().unwrap_or(128) as f32 * weight;
                            weight_sum += weight;
                        }

                        if weight_sum > 0.0 {
                            data[idx] = (sum / weight_sum).round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }

        *plane = Plane::new(data, plane.stride);
        Ok(())
    }

    /// Find similar blocks for BM3D.
    fn find_similar_blocks(
        &self,
        data: &[u8],
        bx: u32,
        by: u32,
        bw: u32,
        bh: u32,
        width: u32,
        height: u32,
        search_window: u32,
        max_blocks: usize,
    ) -> Vec<(u32, u32, f32)> {
        let mut candidates = Vec::new();

        let search_radius = search_window / 2;
        let y_min = by.saturating_sub(search_radius);
        let y_max = (by + search_radius).min(height - bh);
        let x_min = bx.saturating_sub(search_radius);
        let x_max = (bx + search_radius).min(width - bw);

        for sy in (y_min..=y_max).step_by(bw as usize) {
            for sx in (x_min..=x_max).step_by(bw as usize) {
                let dist = self.block_distance(data, bx, by, sx, sy, bw, bh, width);
                let weight = (-dist / (self.config.nlm_h * self.config.nlm_h)).exp();
                candidates.push((sx, sy, weight));
            }
        }

        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(max_blocks);
        candidates
    }

    /// Compute block distance.
    fn block_distance(
        &self,
        data: &[u8],
        x1: u32,
        y1: u32,
        x2: u32,
        y2: u32,
        bw: u32,
        bh: u32,
        width: u32,
    ) -> f32 {
        let mut dist = 0.0f32;
        let mut count = 0;

        for y in 0..bh {
            for x in 0..bw {
                let idx1 = ((y1 + y) * width + (x1 + x)) as usize;
                let idx2 = ((y2 + y) * width + (x2 + x)) as usize;

                let v1 = data.get(idx1).copied().unwrap_or(128) as f32;
                let v2 = data.get(idx2).copied().unwrap_or(128) as f32;

                let diff = v1 - v2;
                dist += diff * diff;
                count += 1;
            }
        }

        if count > 0 {
            dist / count as f32
        } else {
            0.0
        }
    }

    /// Apply combined spatial and temporal filtering.
    fn apply_combined(&mut self, frame: &mut VideoFrame) -> GraphResult<()> {
        self.apply_bilateral(frame)?;

        if !self.frame_buffer.is_empty() {
            self.apply_temporal(frame)?;
        }

        Ok(())
    }

    /// Get strength for a specific plane (luma vs chroma).
    fn get_plane_strength(&self, plane_idx: usize) -> f32 {
        let multiplier = if plane_idx == 0 {
            self.config.luma_strength
        } else {
            self.config.chroma_strength
        };
        self.config.strength * multiplier
    }
}

impl Node for DenoiseFilter {
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
        if !self.state.can_transition_to(state) {
            return Err(GraphError::InvalidStateTransition {
                node: self.id,
                from: self.state.to_string(),
                to: state.to_string(),
            });
        }
        self.state = state;
        Ok(())
    }

    fn inputs(&self) -> &[InputPort] {
        &self.inputs
    }

    fn outputs(&self) -> &[OutputPort] {
        &self.outputs
    }

    fn process(&mut self, input: Option<FilterFrame>) -> GraphResult<Option<FilterFrame>> {
        match input {
            Some(FilterFrame::Video(frame)) => {
                let processed = self.process_frame(frame)?;
                Ok(Some(FilterFrame::Video(processed)))
            }
            Some(_) => Err(GraphError::PortTypeMismatch {
                expected: "Video".to_string(),
                actual: "Audio".to_string(),
            }),
            None => Ok(None),
        }
    }

    fn reset(&mut self) -> GraphResult<()> {
        self.frame_buffer.clear();
        self.prev_frame = None;
        self.noise_stats = None;
        self.set_state(NodeState::Idle)
    }
}

/// Create a Gaussian kernel for filtering.
fn create_gaussian_kernel(radius: usize, sigma: f32) -> Vec<f32> {
    let size = radius * 2 + 1;
    let mut kernel = vec![0.0f32; size * size];
    let mut sum = 0.0f32;

    let sigma_sq = sigma * sigma;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - radius as f32;
            let dy = y as f32 - radius as f32;
            let dist_sq = dx * dx + dy * dy;

            let val = (-dist_sq / (2.0 * sigma_sq)).exp();
            kernel[y * size + x] = val;
            sum += val;
        }
    }

    for val in &mut kernel {
        *val /= sum;
    }

    kernel
}
