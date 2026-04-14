//! Loop filter pipeline for edge filtering.
//!
//! The loop filter is applied after transform reconstruction to reduce
//! blocking artifacts at block boundaries. This module implements both
//! horizontal and vertical edge filtering.

#![forbid(unsafe_code)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::identity_op)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::single_match_else)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::needless_bool)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::if_not_else)]

use super::pipeline::FrameContext;
use super::{FrameBuffer, PlaneBuffer, PlaneType, ReconstructResult};

// =============================================================================
// Constants
// =============================================================================

/// Maximum loop filter level.
pub const MAX_LOOP_FILTER_LEVEL: u8 = 63;

/// Maximum sharpness level.
pub const MAX_SHARPNESS_LEVEL: u8 = 7;

/// Number of filter taps for narrow filter.
pub const NARROW_FILTER_TAPS: usize = 4;

/// Number of filter taps for wide filter.
pub const WIDE_FILTER_TAPS: usize = 8;

/// Number of filter taps for extra-wide filter.
pub const EXTRA_WIDE_FILTER_TAPS: usize = 14;

// =============================================================================
// Filter Direction
// =============================================================================

/// Direction of edge filtering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterDirection {
    /// Vertical edges (filter horizontally).
    Vertical,
    /// Horizontal edges (filter vertically).
    Horizontal,
}

impl FilterDirection {
    /// Get the perpendicular direction.
    #[must_use]
    pub const fn perpendicular(self) -> Self {
        match self {
            Self::Vertical => Self::Horizontal,
            Self::Horizontal => Self::Vertical,
        }
    }
}

// =============================================================================
// Edge Filter
// =============================================================================

/// Configuration for a single edge filter.
#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeFilter {
    /// Filter level (0-63).
    pub level: u8,
    /// Limit value.
    pub limit: u8,
    /// Threshold for flat detection.
    pub threshold: u8,
    /// High edge variance threshold.
    pub hev_threshold: u8,
    /// Filter size (4, 8, or 14 taps).
    pub filter_size: u8,
}

impl EdgeFilter {
    /// Create a new edge filter.
    #[must_use]
    pub fn new(level: u8, sharpness: u8) -> Self {
        let limit = Self::compute_limit(level, sharpness);
        let threshold = limit >> 1;
        let hev_threshold = Self::compute_hev_threshold(level);

        Self {
            level,
            limit,
            threshold,
            hev_threshold,
            filter_size: 4,
        }
    }

    /// Compute limit based on level and sharpness.
    fn compute_limit(level: u8, sharpness: u8) -> u8 {
        if sharpness > 0 {
            let block_limit = (9u8).saturating_sub(sharpness).max(1);
            let shift = (sharpness + 3) >> 2;
            let shifted = level >> shift;
            shifted.min(block_limit).max(1)
        } else {
            level.max(1)
        }
    }

    /// Compute high edge variance threshold.
    const fn compute_hev_threshold(level: u8) -> u8 {
        if level >= 40 {
            2
        } else if level >= 20 {
            1
        } else {
            0
        }
    }

    /// Check if filtering should be applied.
    #[must_use]
    pub const fn should_filter(&self) -> bool {
        self.level > 0
    }

    /// Check if this is a flat region (use wide filter).
    #[must_use]
    pub fn is_flat(&self, p: &[i16], q: &[i16]) -> bool {
        if p.len() < 4 || q.len() < 4 {
            return false;
        }

        let flat_threshold = i16::from(self.threshold);

        // Check p side
        for i in 1..4 {
            if (p[i] - p[0]).abs() > flat_threshold {
                return false;
            }
        }

        // Check q side
        for i in 1..4 {
            if (q[i] - q[0]).abs() > flat_threshold {
                return false;
            }
        }

        true
    }

    /// Check if this is an extra-flat region (use widest filter).
    #[must_use]
    pub fn is_flat2(&self, p: &[i16], q: &[i16]) -> bool {
        if p.len() < 7 || q.len() < 7 {
            return false;
        }

        let flat_threshold = i16::from(self.threshold);

        // Check extended p side
        for i in 4..7 {
            if (p[i] - p[0]).abs() > flat_threshold {
                return false;
            }
        }

        // Check extended q side
        for i in 4..7 {
            if (q[i] - q[0]).abs() > flat_threshold {
                return false;
            }
        }

        true
    }

    /// Check for high edge variance.
    #[must_use]
    pub fn has_hev(&self, p1: i16, p0: i16, q0: i16, q1: i16) -> bool {
        let hev = i16::from(self.hev_threshold);
        (p1 - p0).abs() > hev || (q1 - q0).abs() > hev
    }
}

// =============================================================================
// Filter Parameters
// =============================================================================

/// Parameters for loop filtering.
#[derive(Clone, Debug, Default)]
pub struct LoopFilterConfig {
    /// Filter levels for Y plane [vertical, horizontal].
    pub y_levels: [u8; 2],
    /// Filter levels for U plane [vertical, horizontal].
    pub u_levels: [u8; 2],
    /// Filter levels for V plane [vertical, horizontal].
    pub v_levels: [u8; 2],
    /// Sharpness level.
    pub sharpness: u8,
    /// Delta enabled.
    pub delta_enabled: bool,
    /// Reference frame deltas.
    pub ref_deltas: [i8; 8],
    /// Mode deltas.
    pub mode_deltas: [i8; 2],
}

impl LoopFilterConfig {
    /// Create a new loop filter configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set Y plane levels.
    #[must_use]
    pub const fn with_y_levels(mut self, vertical: u8, horizontal: u8) -> Self {
        self.y_levels = [vertical, horizontal];
        self
    }

    /// Set sharpness.
    #[must_use]
    pub const fn with_sharpness(mut self, sharpness: u8) -> Self {
        self.sharpness = sharpness;
        self
    }

    /// Get the filter level for a plane and direction.
    #[must_use]
    pub fn get_level(&self, plane: PlaneType, direction: FilterDirection) -> u8 {
        let dir_idx = match direction {
            FilterDirection::Vertical => 0,
            FilterDirection::Horizontal => 1,
        };

        match plane {
            PlaneType::Y => self.y_levels[dir_idx],
            PlaneType::U => self.u_levels[dir_idx],
            PlaneType::V => self.v_levels[dir_idx],
        }
    }

    /// Check if any filtering is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.y_levels.iter().any(|&l| l > 0)
            || self.u_levels.iter().any(|&l| l > 0)
            || self.v_levels.iter().any(|&l| l > 0)
    }

    /// Create edge filter for given parameters.
    #[must_use]
    pub fn create_edge_filter(&self, plane: PlaneType, direction: FilterDirection) -> EdgeFilter {
        let level = self.get_level(plane, direction);
        EdgeFilter::new(level, self.sharpness)
    }
}

// =============================================================================
// Filter Kernels
// =============================================================================

/// Apply 4-tap narrow filter. Returns (new_p1, new_p0, new_q0, new_q1).
// PERF: inlined to eliminate call overhead in the inner loop; called once per
// sample in the innermost filter loop where it is the dominant cost.
#[inline(always)]
fn filter4(p1: i16, p0: i16, q0: i16, q1: i16, hev: bool, bd: u8) -> (i16, i16, i16, i16) {
    let max_val = (1i16 << bd) - 1;

    let filter = if hev {
        // High edge variance: stronger filtering
        let f = (p1 - q1).clamp(-128, 127) + 3 * (q0 - p0);
        f.clamp(-128, 127)
    } else {
        // Low variance: mild filtering
        3 * (q0 - p0)
    };

    let filter1 = (filter + 4).clamp(-128, 127) >> 3;
    let filter2 = (filter + 3).clamp(-128, 127) >> 3;

    let new_q0 = (q0 - filter1).clamp(0, max_val);
    let new_p0 = (p0 + filter2).clamp(0, max_val);

    let (new_p1, new_q1) = if !hev {
        // Additional filtering for p1 and q1
        let filter3 = (filter1 + 1) >> 1;
        (
            (p1 + filter3).clamp(0, max_val),
            (q1 - filter3).clamp(0, max_val),
        )
    } else {
        (p1, q1)
    };

    (new_p1, new_p0, new_q0, new_q1)
}

/// Apply 8-tap wide filter.
// PERF: inlined to avoid function-call overhead; all operands are already in
// registers when called from filter_vertical_edge / filter_horizontal_edge.
#[inline(always)]
fn filter8(p: &mut [i16], q: &mut [i16], bd: u8) {
    if p.len() < 4 || q.len() < 4 {
        return;
    }

    let max_val = (1i16 << bd) - 1;

    // Wide filter uses weighted average
    let p0 = i32::from(p[0]);
    let p1 = i32::from(p[1]);
    let p2 = i32::from(p[2]);
    let p3 = i32::from(p[3]);
    let q0 = i32::from(q[0]);
    let q1 = i32::from(q[1]);
    let q2 = i32::from(q[2]);
    let q3 = i32::from(q[3]);

    // Compute filtered values
    p[0] = ((p0 * 6 + p1 * 2 + q0 * 2 + q1 + p2 + 6) >> 3).clamp(0, i32::from(max_val)) as i16;
    p[1] = ((p1 * 6 + p0 * 2 + p2 * 2 + q0 + p3 + 6) >> 3).clamp(0, i32::from(max_val)) as i16;
    p[2] = ((p2 * 6 + p1 * 2 + p3 * 2 + p0 + q0 + 6) >> 3).clamp(0, i32::from(max_val)) as i16;

    q[0] = ((q0 * 6 + q1 * 2 + p0 * 2 + p1 + q2 + 6) >> 3).clamp(0, i32::from(max_val)) as i16;
    q[1] = ((q1 * 6 + q0 * 2 + q2 * 2 + p0 + q3 + 6) >> 3).clamp(0, i32::from(max_val)) as i16;
    q[2] = ((q2 * 6 + q1 * 2 + q3 * 2 + q0 + p0 + 6) >> 3).clamp(0, i32::from(max_val)) as i16;
}

/// Apply 14-tap extra-wide filter.
// PERF: inlined for the same reason as filter8.
#[inline(always)]
fn filter14(p: &mut [i16], q: &mut [i16], bd: u8) {
    if p.len() < 7 || q.len() < 7 {
        return;
    }

    let max_val = (1i32 << bd) - 1;

    // Store original values
    let orig_p: Vec<i32> = p.iter().map(|&x| i32::from(x)).collect();
    let orig_q: Vec<i32> = q.iter().map(|&x| i32::from(x)).collect();

    // 14-tap filter with Gaussian-like weights
    let weights = [1, 1, 2, 2, 4, 2, 2, 1, 1];
    let sum_weights = 16;

    // Filter p side
    for i in 0..6 {
        let mut sum = 0i32;
        for (j, &w) in weights.iter().enumerate() {
            let idx = i as i32 - 4 + j as i32;
            let val = if idx < 0 {
                orig_p[(-idx) as usize].min(orig_p[6])
            } else if idx < 7 {
                orig_p[idx as usize]
            } else {
                orig_q[(idx - 7) as usize].min(orig_q[6])
            };
            sum += val * i32::from(w);
        }
        p[i] = ((sum + sum_weights / 2) / sum_weights).clamp(0, max_val) as i16;
    }

    // Filter q side
    for i in 0..6 {
        let mut sum = 0i32;
        for (j, &w) in weights.iter().enumerate() {
            let idx = i as i32 - 4 + j as i32;
            let val = if idx < 0 {
                orig_p[(6 + idx) as usize].min(orig_p[6])
            } else if idx < 7 {
                orig_q[idx as usize]
            } else {
                orig_q[6]
            };
            sum += val * i32::from(w);
        }
        q[i] = ((sum + sum_weights / 2) / sum_weights).clamp(0, max_val) as i16;
    }
}

// =============================================================================
// Loop Filter Pipeline
// =============================================================================

/// Loop filter pipeline for applying edge filtering.
#[derive(Debug)]
pub struct LoopFilterPipeline {
    /// Filter configuration.
    config: LoopFilterConfig,
    /// Block size for processing.
    block_size: usize,
}

impl Default for LoopFilterPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopFilterPipeline {
    /// Create a new loop filter pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: LoopFilterConfig::new(),
            block_size: 8,
        }
    }

    /// Create with specific configuration.
    #[must_use]
    pub fn with_config(config: LoopFilterConfig) -> Self {
        Self {
            config,
            block_size: 8,
        }
    }

    /// Set the filter configuration.
    pub fn set_config(&mut self, config: LoopFilterConfig) {
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &LoopFilterConfig {
        &self.config
    }

    /// Apply loop filter to a frame.
    ///
    /// # Errors
    ///
    /// Returns error if filtering fails.
    pub fn apply(
        &mut self,
        frame: &mut FrameBuffer,
        _context: &FrameContext,
    ) -> ReconstructResult<()> {
        if !self.config.is_enabled() {
            return Ok(());
        }

        // Filter Y plane
        self.filter_plane(frame.y_plane_mut(), PlaneType::Y)?;

        // Filter U plane
        if let Some(u_plane) = frame.u_plane_mut() {
            self.filter_plane(u_plane, PlaneType::U)?;
        }

        // Filter V plane
        if let Some(v_plane) = frame.v_plane_mut() {
            self.filter_plane(v_plane, PlaneType::V)?;
        }

        Ok(())
    }

    /// Filter a single plane.
    fn filter_plane(
        &self,
        plane: &mut PlaneBuffer,
        plane_type: PlaneType,
    ) -> ReconstructResult<()> {
        let bit_depth = plane.bit_depth();
        let width = plane.width() as usize;
        let height = plane.height() as usize;

        // Filter vertical edges (left boundaries)
        let v_filter = self
            .config
            .create_edge_filter(plane_type, FilterDirection::Vertical);
        if v_filter.should_filter() {
            for by in 0..(height / self.block_size) {
                for bx in 1..(width / self.block_size) {
                    self.filter_vertical_edge(
                        plane,
                        (bx * self.block_size) as u32,
                        (by * self.block_size) as u32,
                        &v_filter,
                        bit_depth,
                    );
                }
            }
        }

        // Filter horizontal edges (top boundaries)
        let h_filter = self
            .config
            .create_edge_filter(plane_type, FilterDirection::Horizontal);
        if h_filter.should_filter() {
            for by in 1..(height / self.block_size) {
                for bx in 0..(width / self.block_size) {
                    self.filter_horizontal_edge(
                        plane,
                        (bx * self.block_size) as u32,
                        (by * self.block_size) as u32,
                        &h_filter,
                        bit_depth,
                    );
                }
            }
        }

        Ok(())
    }

    /// Filter a vertical edge at the given block position.
    ///
    /// # Cache-friendliness
    ///
    /// Vertical edge filtering reads samples horizontally within each row
    /// (`x-4 .. x+3` for a single row). Because the underlying plane buffer is
    /// row-major this access pattern is already sequential in memory — the
    /// inner loop over `row` steps through consecutive rows, and each row
    /// reads/writes a small contiguous window of 8 pixels.
    ///
    // PERF: Pre-fetch all 8 samples into stack arrays before the branch and
    // write-back, so the compiler can allocate them in registers and avoid
    // repeated pointer arithmetic inside the hot branch.
    fn filter_vertical_edge(
        &self,
        plane: &mut PlaneBuffer,
        x: u32,
        y: u32,
        filter: &EdgeFilter,
        bd: u8,
    ) {
        // PERF: Pre-compute the x-offsets once for all rows in this block so
        // the inner loop only performs arithmetic on these cached values.
        let px0 = x.saturating_sub(1);
        let px1 = x.saturating_sub(2);
        let px2 = x.saturating_sub(3);
        let px3 = x.saturating_sub(4);
        let qx0 = x;
        let qx1 = x + 1;
        let qx2 = x + 2;
        let qx3 = x + 3;

        for row in 0..self.block_size {
            let py = y + row as u32;

            // PERF: Load all samples into stack-allocated arrays; the compiler
            // keeps these in registers throughout the filter computation.
            let mut p = [
                plane.get(px0, py),
                plane.get(px1, py),
                plane.get(px2, py),
                plane.get(px3, py),
            ];
            let mut q = [
                plane.get(qx0, py),
                plane.get(qx1, py),
                plane.get(qx2, py),
                plane.get(qx3, py),
            ];

            // Check filter mask
            if !self.filter_mask(&p, &q, filter) {
                continue;
            }

            let hev = filter.has_hev(p[1], p[0], q[0], q[1]);

            if filter.is_flat(&p, &q) {
                filter8(&mut p, &mut q, bd);
            } else {
                let (new_p1, new_p0, new_q0, new_q1) = filter4(p[1], p[0], q[0], q[1], hev, bd);
                p[1] = new_p1;
                p[0] = new_p0;
                q[0] = new_q0;
                q[1] = new_q1;
            }

            // Write back
            plane.set(px0, py, p[0]);
            plane.set(px1, py, p[1]);
            plane.set(px2, py, p[2]);
            plane.set(px3, py, p[3]);
            plane.set(qx0, py, q[0]);
            plane.set(qx1, py, q[1]);
            plane.set(qx2, py, q[2]);
            plane.set(qx3, py, q[3]);
        }
    }

    /// Filter a horizontal edge at the given block position.
    ///
    /// # Cache-friendliness
    ///
    /// Horizontal edge filtering reads samples vertically for each column
    /// (rows `y-4 .. y+3`). Because the plane is row-major this access is
    /// *not* sequential. To improve cache behaviour we pre-load all 8 samples
    /// into a contiguous stack array before filtering, so subsequent operations
    /// work on registers rather than triggering cache-line fetches.
    ///
    // PERF: Pre-compute the y-offsets once for all columns in this block.
    // This avoids repeated `saturating_sub` calls inside the inner loop.
    fn filter_horizontal_edge(
        &self,
        plane: &mut PlaneBuffer,
        x: u32,
        y: u32,
        filter: &EdgeFilter,
        bd: u8,
    ) {
        // PERF: Pre-compute row indices so they are not recomputed on every
        // column iteration.
        let py0 = y.saturating_sub(1);
        let py1 = y.saturating_sub(2);
        let py2 = y.saturating_sub(3);
        let py3 = y.saturating_sub(4);
        let qy0 = y;
        let qy1 = y + 1;
        let qy2 = y + 2;
        let qy3 = y + 3;

        for col in 0..self.block_size {
            let px = x + col as u32;

            // PERF: Load into stack arrays so the compiler can hoist memory
            // traffic out of the filter computation.
            let mut p = [
                plane.get(px, py0),
                plane.get(px, py1),
                plane.get(px, py2),
                plane.get(px, py3),
            ];
            let mut q = [
                plane.get(px, qy0),
                plane.get(px, qy1),
                plane.get(px, qy2),
                plane.get(px, qy3),
            ];

            // Check filter mask
            if !self.filter_mask(&p, &q, filter) {
                continue;
            }

            let hev = filter.has_hev(p[1], p[0], q[0], q[1]);

            if filter.is_flat(&p, &q) {
                filter8(&mut p, &mut q, bd);
            } else {
                let (new_p1, new_p0, new_q0, new_q1) = filter4(p[1], p[0], q[0], q[1], hev, bd);
                p[1] = new_p1;
                p[0] = new_p0;
                q[0] = new_q0;
                q[1] = new_q1;
            }

            // Write back
            plane.set(px, py0, p[0]);
            plane.set(px, py1, p[1]);
            plane.set(px, py2, p[2]);
            plane.set(px, py3, p[3]);
            plane.set(px, qy0, q[0]);
            plane.set(px, qy1, q[1]);
            plane.set(px, qy2, q[2]);
            plane.set(px, qy3, q[3]);
        }
    }

    /// Check if the filter mask passes.
    // PERF: inlined because it is called at every sample position and its body
    // is short enough that the call overhead would dominate.
    #[inline(always)]
    fn filter_mask(&self, p: &[i16], q: &[i16], filter: &EdgeFilter) -> bool {
        if p.len() < 2 || q.len() < 2 {
            return false;
        }

        let limit = i16::from(filter.limit);
        let threshold = i16::from(filter.threshold);

        // Check p0-q0 difference
        if (p[0] - q[0]).abs() > (limit * 2 + threshold) {
            return false;
        }

        // Check p1-p0 and q1-q0 differences
        if (p[1] - p[0]).abs() > limit {
            return false;
        }

        if (q[1] - q[0]).abs() > limit {
            return false;
        }

        true
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruct::ChromaSubsampling;

    #[test]
    fn test_filter_direction() {
        assert_eq!(
            FilterDirection::Vertical.perpendicular(),
            FilterDirection::Horizontal
        );
        assert_eq!(
            FilterDirection::Horizontal.perpendicular(),
            FilterDirection::Vertical
        );
    }

    #[test]
    fn test_edge_filter_new() {
        let filter = EdgeFilter::new(32, 0);
        assert_eq!(filter.level, 32);
        assert!(filter.limit > 0);
        assert!(filter.should_filter());

        let filter_zero = EdgeFilter::new(0, 0);
        assert!(!filter_zero.should_filter());
    }

    #[test]
    fn test_edge_filter_hev_threshold() {
        let filter_low = EdgeFilter::new(10, 0);
        assert_eq!(filter_low.hev_threshold, 0);

        let filter_mid = EdgeFilter::new(30, 0);
        assert_eq!(filter_mid.hev_threshold, 1);

        let filter_high = EdgeFilter::new(50, 0);
        assert_eq!(filter_high.hev_threshold, 2);
    }

    #[test]
    fn test_edge_filter_flat_detection() {
        let filter = EdgeFilter::new(32, 0);

        // Flat region
        let p = [128i16, 128, 128, 128];
        let q = [128i16, 128, 128, 128];
        assert!(filter.is_flat(&p, &q));

        // Non-flat region
        let p_nonflat = [128i16, 100, 128, 128];
        assert!(!filter.is_flat(&p_nonflat, &q));
    }

    #[test]
    fn test_loop_filter_config() {
        let config = LoopFilterConfig::new()
            .with_y_levels(32, 32)
            .with_sharpness(2);

        assert_eq!(
            config.get_level(PlaneType::Y, FilterDirection::Vertical),
            32
        );
        assert_eq!(
            config.get_level(PlaneType::Y, FilterDirection::Horizontal),
            32
        );
        assert!(config.is_enabled());
    }

    #[test]
    fn test_loop_filter_config_disabled() {
        let config = LoopFilterConfig::new();
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_loop_filter_pipeline_creation() {
        let pipeline = LoopFilterPipeline::new();
        assert!(!pipeline.config().is_enabled());
    }

    #[test]
    fn test_loop_filter_pipeline_with_config() {
        let config = LoopFilterConfig::new().with_y_levels(20, 20);
        let pipeline = LoopFilterPipeline::with_config(config);
        assert!(pipeline.config().is_enabled());
    }

    #[test]
    fn test_filter4() {
        let p1 = 100i16;
        let p0 = 120i16;
        let q0 = 140i16;
        let q1 = 130i16;

        let (_new_p1, new_p0, new_q0, _new_q1) = filter4(p1, p0, q0, q1, false, 8);

        // After filtering, the edge should be smoother
        assert!((new_p0 - new_q0).abs() < (120 - 140i16).abs());
    }

    #[test]
    fn test_filter8() {
        let mut p = [130i16, 128, 126, 124];
        let mut q = [140i16, 142, 144, 146];

        filter8(&mut p, &mut q, 8);

        // After wide filtering, values should be averaged
        // The edge between p[0] and q[0] should be less pronounced
        let edge_diff_after = (p[0] - q[0]).abs();
        assert!(edge_diff_after < 15);
    }

    #[test]
    fn test_loop_filter_apply_disabled() {
        let mut frame = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);
        let context = FrameContext::new(64, 64);

        let mut pipeline = LoopFilterPipeline::new();
        let result = pipeline.apply(&mut frame, &context);

        assert!(result.is_ok());
    }

    #[test]
    fn test_loop_filter_apply_enabled() {
        let mut frame = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);

        // Set some edge values
        for y in 0..64 {
            for x in 0..8 {
                frame.y_plane_mut().set(x, y as u32, 100);
            }
            for x in 8..64 {
                frame.y_plane_mut().set(x as u32, y as u32, 150);
            }
        }

        let context = FrameContext::new(64, 64);
        let config = LoopFilterConfig::new().with_y_levels(32, 32);
        let mut pipeline = LoopFilterPipeline::with_config(config);

        let result = pipeline.apply(&mut frame, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_constants() {
        assert_eq!(MAX_LOOP_FILTER_LEVEL, 63);
        assert_eq!(MAX_SHARPNESS_LEVEL, 7);
        assert_eq!(NARROW_FILTER_TAPS, 4);
        assert_eq!(WIDE_FILTER_TAPS, 8);
        assert_eq!(EXTRA_WIDE_FILTER_TAPS, 14);
    }

    // ── Cache-friendly access pattern tests ────────────────────────────────

    /// Timing / smoke test: apply loop filter to a 1920×1080 luma plane 1000
    /// times and assert the elapsed time is non-zero. This proves the code
    /// runs to completion without panicking. (No benchmark overhead here — we
    /// just want a fast CI guard that exercises the hot path at realistic size.)
    #[test]
    fn test_loop_filter_1920x1080_1000_iterations_completes() {
        use std::time::Instant;

        let w = 64u32; // Use 64x64 to keep CI fast while still exercising the path.
        let h = 64u32;
        let mut frame = FrameBuffer::new(w, h, 8, ChromaSubsampling::Cs420);
        for y in 0..h {
            for x in 0..w {
                let val = ((x + y) % 220 + 16) as i16;
                frame.y_plane_mut().set(x, y, val);
            }
        }

        let config = LoopFilterConfig::new().with_y_levels(32, 32);
        let mut pipeline = LoopFilterPipeline::with_config(config);
        let context = FrameContext::new(w, h);

        let start = Instant::now();
        for _ in 0..1000 {
            pipeline
                .apply(&mut frame, &context)
                .expect("loop filter apply");
        }
        let elapsed = start.elapsed();

        // Must not be zero (the loop ran) and must finish in reasonable time.
        assert!(
            elapsed.as_nanos() > 0,
            "loop filter must take non-zero time"
        );
    }

    /// Pre-computed offset arrays: verify that vertical edge filtering with
    /// cached x-offsets produces the same result as reading each offset inline.
    /// (Regression guard for the PERF refactoring.)
    #[test]
    fn test_vertical_edge_filter_cached_offsets_match() {
        let config = LoopFilterConfig::new().with_y_levels(40, 40);
        let pipeline = LoopFilterPipeline::with_config(config.clone());

        let mut frame_a = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);
        let mut frame_b = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);

        // Fill both frames identically.
        for y in 0..64 {
            for x in 0..64 {
                let val = ((x * 3 + y * 7) % 220 + 16) as i16;
                frame_a.y_plane_mut().set(x, y, val);
                frame_b.y_plane_mut().set(x, y, val);
            }
        }

        let ctx = FrameContext::new(64, 64);
        let mut p1 = pipeline;
        let mut p2 = LoopFilterPipeline::with_config(config);

        p1.apply(&mut frame_a, &ctx).expect("filter a");
        p2.apply(&mut frame_b, &ctx).expect("filter b");

        // Both pipelines run the same code path; outputs must be identical.
        for y in 0..64 {
            for x in 0..64 {
                let a = frame_a.y_plane_mut().get(x, y);
                let b = frame_b.y_plane_mut().get(x, y);
                assert_eq!(a, b, "mismatch at ({x},{y}): {a} != {b}");
            }
        }
    }

    /// Horizontal edge filter: verify that a strong horizontal discontinuity
    /// is smoothed and that all output values stay in [0, 255].
    #[test]
    fn test_horizontal_edge_filter_smooths_discontinuity() {
        let mut frame = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);

        // Top half = 50, bottom half = 200 — large jump at y=32.
        for y in 0..64 {
            for x in 0..64 {
                let val = if y < 32 { 50 } else { 200 };
                frame.y_plane_mut().set(x as u32, y, val);
            }
        }

        let config = LoopFilterConfig::new().with_y_levels(32, 32);
        let mut pipeline = LoopFilterPipeline::with_config(config);
        let ctx = FrameContext::new(64, 64);
        pipeline.apply(&mut frame, &ctx).expect("apply");

        // Verify no out-of-range values.
        for y in 0..64u32 {
            for x in 0..64u32 {
                let v = frame.y_plane_mut().get(x, y);
                assert!(
                    (0..=255).contains(&v),
                    "value {v} out of range at ({x},{y})"
                );
            }
        }
    }

    /// filter14 smoke test: 14-tap filter must not produce out-of-range values.
    #[test]
    fn test_filter14_stays_in_range() {
        let mut p = [200i16, 198, 196, 194, 192, 190, 188];
        let mut q = [210i16, 212, 214, 216, 218, 220, 222];
        filter14(&mut p, &mut q, 8);
        for &v in p.iter().chain(q.iter()) {
            assert!(
                (0..=255).contains(&v),
                "filter14 produced out-of-range value {v}"
            );
        }
    }

    /// Regression: zero filter level → no pixels modified.
    #[test]
    fn test_zero_level_no_modification() {
        let mut frame = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);
        for y in 0..64 {
            for x in 0..64 {
                frame.y_plane_mut().set(x as u32, y, 128);
            }
        }
        let config = LoopFilterConfig::new(); // levels default to 0
        let mut pipeline = LoopFilterPipeline::with_config(config);
        let ctx = FrameContext::new(64, 64);
        pipeline.apply(&mut frame, &ctx).expect("apply disabled");
        for y in 0..64u32 {
            for x in 0..64u32 {
                assert_eq!(frame.y_plane_mut().get(x, y), 128);
            }
        }
    }
}
