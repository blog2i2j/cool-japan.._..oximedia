//! Tile-based quality selector for adaptive 360° video streaming.
//!
//! Modern 360° streaming platforms (YouTube, Facebook 360, etc.) divide the
//! equirectangular frame into a regular grid of **tiles** and encode each tile
//! at multiple quality levels.  At playback time, tiles that overlap the
//! viewer's current viewport should be downloaded at the highest quality,
//! while tiles at the periphery or behind the viewer can use lower quality or
//! be skipped entirely.
//!
//! ## Overview
//!
//! A [`TileGrid`] defines the row×column layout of tiles over the
//! equirectangular frame.  [`TileSelector`] takes a viewport description and
//! assigns each tile a [`TilePriority`]:
//!
//! | Priority | Description |
//! |----------|-------------|
//! | `Critical` | Tile overlaps the viewport; must be at highest quality. |
//! | `Near`     | Tile is adjacent to the viewport; prefetch at high quality. |
//! | `Far`      | Tile is outside the near zone; low quality or skip. |
//! | `Skip`     | Tile is entirely outside the streaming budget; omit. |
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_360::tile_selector::{TileGrid, TileSelector, TilePriority};
//! use oximedia_360::viewport_predictor::ViewportRegion;
//! use std::f32::consts::PI;
//!
//! let grid = TileGrid::new(4, 8).expect("valid grid");
//! let selector = TileSelector::new(grid);
//!
//! // User looking straight ahead, 90° × 60° FOV
//! let viewport = ViewportRegion::from_fov(0.0, 0.0, PI / 2.0, PI / 3.0)
//!     .expect("valid region");
//! let priorities = selector.select(&viewport);
//!
//! let critical: Vec<_> = priorities.iter()
//!     .filter(|p| p.priority == TilePriority::Critical)
//!     .collect();
//! assert!(!critical.is_empty());
//! ```

use crate::{viewport_predictor::ViewportRegion, VrError};
use std::f32::consts::PI;

// ─── TileIndex ────────────────────────────────────────────────────────────────

/// Row/column index of a tile within a [`TileGrid`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileIndex {
    /// Zero-based row (0 = top).
    pub row: u32,
    /// Zero-based column (0 = left).
    pub col: u32,
}

// ─── TilePriority ─────────────────────────────────────────────────────────────

/// Playback priority assigned to a tile by the selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TilePriority {
    /// Skip this tile entirely (outside the streaming budget).
    Skip = 0,
    /// Low-quality background tile, outside the near-viewport zone.
    Far = 1,
    /// Adjacent to the viewport; prefetch at elevated quality.
    Near = 2,
    /// Overlaps the viewport; must be delivered at highest quality.
    Critical = 3,
}

// ─── TileAssignment ───────────────────────────────────────────────────────────

/// A tile plus the priority assigned to it by the selector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileAssignment {
    /// Identity of this tile.
    pub index: TileIndex,
    /// Quality / download priority.
    pub priority: TilePriority,
    /// Intersection-over-union of the tile region with the viewport.
    pub iou: f32,
}

// ─── TileGrid ─────────────────────────────────────────────────────────────────

/// A regular rectangular tile layout over the equirectangular sphere.
///
/// The equirectangular sphere occupies `[−π, +π]` in yaw and `[−π/2, +π/2]`
/// in pitch.  `rows` tiles divide the pitch axis uniformly; `cols` tiles
/// divide the yaw axis uniformly.
#[derive(Debug, Clone, PartialEq)]
pub struct TileGrid {
    /// Number of tile rows (pitch axis).
    pub rows: u32,
    /// Number of tile columns (yaw axis).
    pub cols: u32,
    /// Angular height of each tile (radians).
    pub tile_pitch_rad: f32,
    /// Angular width of each tile (radians).
    pub tile_yaw_rad: f32,
}

impl TileGrid {
    /// Create a new tile grid with the given dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidDimensions`] if `rows` or `cols` is zero.
    pub fn new(rows: u32, cols: u32) -> Result<Self, VrError> {
        if rows == 0 || cols == 0 {
            return Err(VrError::InvalidDimensions(
                "TileGrid rows and cols must be > 0".into(),
            ));
        }
        let tile_pitch_rad = PI / rows as f32; // total pitch span = π
        let tile_yaw_rad = 2.0 * PI / cols as f32; // total yaw span = 2π
        Ok(Self {
            rows,
            cols,
            tile_pitch_rad,
            tile_yaw_rad,
        })
    }

    /// Return the total number of tiles.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        (self.rows * self.cols) as usize
    }

    /// Compute the spherical bounding region for tile `(row, col)`.
    ///
    /// The returned region is centred at the middle of the tile and has half-
    /// extents equal to half of the tile's angular dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidCoordinate`] if the indices are out of range.
    pub fn tile_region(&self, row: u32, col: u32) -> Result<ViewportRegion, VrError> {
        if row >= self.rows || col >= self.cols {
            return Err(VrError::InvalidCoordinate);
        }
        // Centre pitch: top row = near north pole (+π/2), rows go down.
        let pitch_centre = PI / 2.0 - (row as f32 + 0.5) * self.tile_pitch_rad;
        // Centre yaw: columns go left → right, starting at −π.
        let yaw_centre = -PI + (col as f32 + 0.5) * self.tile_yaw_rad;

        ViewportRegion::from_fov(
            yaw_centre,
            pitch_centre,
            self.tile_yaw_rad,
            self.tile_pitch_rad,
        )
        .map_err(|_| VrError::InvalidCoordinate)
    }

    /// Iterate over all `(row, col)` tile indices.
    pub fn indices(&self) -> impl Iterator<Item = TileIndex> + '_ {
        (0..self.rows).flat_map(move |r| (0..self.cols).map(move |c| TileIndex { row: r, col: c }))
    }
}

// ─── TileSelector ─────────────────────────────────────────────────────────────

/// Assigns a [`TilePriority`] to each tile in a [`TileGrid`] based on overlap
/// with a viewport region.
///
/// ## Priority rules
///
/// 1. **Critical** — IoU with viewport > `critical_iou_threshold` (default 0.0,
///    meaning any overlap at all triggers *Critical*).
/// 2. **Near** — Tile centre is within `near_margin_rad` of the viewport boundary.
/// 3. **Far** — Everything else.
/// 4. **Skip** — Enabled only when `max_critical` or `max_near` limits are
///    exceeded (budget enforcement); otherwise, *Far* is used instead.
#[derive(Debug, Clone)]
pub struct TileSelector {
    grid: TileGrid,
    /// Minimum IoU to be classified as `Critical`.  0.0 means any overlap.
    critical_iou_threshold: f32,
    /// Angular margin (radians) around the viewport for `Near` classification.
    near_margin_rad: f32,
}

impl TileSelector {
    /// Create a selector with default thresholds:
    /// * `critical_iou_threshold = 0.0` (any overlap → critical)
    /// * `near_margin_rad = π / 6` (~30°)
    #[must_use]
    pub fn new(grid: TileGrid) -> Self {
        Self {
            grid,
            critical_iou_threshold: 0.0,
            near_margin_rad: PI / 6.0,
        }
    }

    /// Override the minimum IoU threshold for `Critical` classification.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidCoordinate`] if `threshold` is not in
    /// `[0.0, 1.0]`.
    pub fn with_critical_threshold(mut self, threshold: f32) -> Result<Self, VrError> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(VrError::InvalidCoordinate);
        }
        self.critical_iou_threshold = threshold;
        Ok(self)
    }

    /// Override the near-zone margin in radians.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidDimensions`] if `margin_rad` is negative.
    pub fn with_near_margin(mut self, margin_rad: f32) -> Result<Self, VrError> {
        if margin_rad < 0.0 {
            return Err(VrError::InvalidDimensions(
                "near_margin_rad must be ≥ 0".into(),
            ));
        }
        self.near_margin_rad = margin_rad;
        Ok(self)
    }

    /// Compute tile assignments for the given viewport.
    ///
    /// Returns one [`TileAssignment`] per tile in row-major order
    /// (row 0 col 0, row 0 col 1, …, row n-1 col m-1).
    #[must_use]
    pub fn select(&self, viewport: &ViewportRegion) -> Vec<TileAssignment> {
        let mut assignments = Vec::with_capacity(self.grid.tile_count());

        for idx in self.grid.indices() {
            let tile_region = match self.grid.tile_region(idx.row, idx.col) {
                Ok(r) => r,
                Err(_) => {
                    assignments.push(TileAssignment {
                        index: idx,
                        priority: TilePriority::Skip,
                        iou: 0.0,
                    });
                    continue;
                }
            };

            let iou = viewport.iou(&tile_region);
            let priority = self.classify(iou, viewport, &tile_region);

            assignments.push(TileAssignment {
                index: idx,
                priority,
                iou,
            });
        }

        assignments
    }

    /// Classify a single tile given its IoU and the expanded near-margin region.
    fn classify(&self, iou: f32, viewport: &ViewportRegion, tile: &ViewportRegion) -> TilePriority {
        if iou > self.critical_iou_threshold {
            return TilePriority::Critical;
        }
        // Check if tile centre falls within the near-margin expanded viewport.
        let expanded = ViewportRegion {
            yaw_rad: viewport.yaw_rad,
            pitch_rad: viewport.pitch_rad,
            half_yaw_rad: viewport.half_yaw_rad + self.near_margin_rad,
            half_pitch_rad: viewport.half_pitch_rad + self.near_margin_rad,
        };
        if expanded.contains(tile.yaw_rad, tile.pitch_rad) {
            return TilePriority::Near;
        }
        TilePriority::Far
    }

    /// Return tiles sorted by priority (highest first) and then by IoU.
    ///
    /// Useful for greedy download scheduling: iterate the result and fetch
    /// tiles until the bandwidth budget is exhausted.
    #[must_use]
    pub fn sorted_by_priority(&self, viewport: &ViewportRegion) -> Vec<TileAssignment> {
        let mut assignments = self.select(viewport);
        assignments.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then_with(|| {
                b.iou
                    .partial_cmp(&a.iou)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        assignments
    }

    /// Return the grid used by this selector.
    #[must_use]
    pub fn grid(&self) -> &TileGrid {
        &self.grid
    }
}

// ─── QualityLadder ────────────────────────────────────────────────────────────

/// Maps a [`TilePriority`] to a target quality level index.
///
/// Quality levels are zero-based integers where 0 = lowest quality and
/// `max_level` = highest quality.  Callers use this to look up which segment
/// URL to request for a given tile.
#[derive(Debug, Clone)]
pub struct QualityLadder {
    /// Highest quality level index (inclusive).
    pub max_level: u8,
}

impl QualityLadder {
    /// Create a quality ladder with `levels` distinct quality tiers.
    ///
    /// `levels` must be at least 1.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::InvalidDimensions`] if `levels` is 0.
    pub fn new(levels: u8) -> Result<Self, VrError> {
        if levels == 0 {
            return Err(VrError::InvalidDimensions(
                "QualityLadder levels must be > 0".into(),
            ));
        }
        Ok(Self {
            max_level: levels - 1,
        })
    }

    /// Return the quality level index for the given tile priority.
    ///
    /// | Priority   | Level (4-rung ladder example) |
    /// |------------|-------------------------------|
    /// | `Critical` | max_level                     |
    /// | `Near`     | max_level − 1 (clamped to 0)  |
    /// | `Far`      | 1 (or 0 if max_level < 2)     |
    /// | `Skip`     | 0                             |
    #[must_use]
    pub fn level_for(&self, priority: TilePriority) -> u8 {
        match priority {
            TilePriority::Critical => self.max_level,
            TilePriority::Near => self.max_level.saturating_sub(1),
            TilePriority::Far => self.max_level.min(1),
            TilePriority::Skip => 0,
        }
    }
}

// ─── FovQualityAllocator ──────────────────────────────────────────────────────

/// FOV-dependent quality allocation for 360° adaptive streaming.
///
/// Rather than using a fixed threshold for critical/near/far classification,
/// `FovQualityAllocator` scales quality based on **angular distance from the
/// viewport centre**: tiles at the centre of the viewer's gaze receive the
/// highest quality, quality decreases smoothly as tiles move toward the
/// periphery, and tiles beyond a configurable angular cutoff are skipped.
///
/// This models the human visual system's eccentricity-dependent acuity — we
/// resolve much more detail in the fovea (centre of gaze) than in the
/// periphery.
///
/// ## Quality scaling
///
/// For each tile, the quality level is computed as:
///
/// ```text
/// q = round(max_level × f(angle / fovea_radius_rad))
/// ```
///
/// where `f` is a fall-off function.  Three fall-off modes are provided:
///
/// | Mode | Formula |
/// |------|---------|
/// | `Linear`   | `max(0, 1 − x)` |
/// | `Cosine`   | `max(0, cos(x · π/2))` — smooth, perceptually uniform |
/// | `Gaussian` | `exp(−(x²) / (2 · σ²))` where σ = `fovea_radius_rad / 2` |
#[derive(Debug, Clone)]
pub struct FovQualityAllocator {
    grid: TileGrid,
    /// Angular radius of the fovea / viewport centre (radians).
    /// Tiles within this angle get full quality; beyond it quality falls off.
    pub fovea_radius_rad: f32,
    /// Angular cutoff: tiles beyond this angle get quality 0 (skip).
    pub cutoff_radius_rad: f32,
    /// Highest available quality level index (0-based).
    pub max_level: u8,
    /// Fall-off mode.
    pub falloff: FovFalloff,
}

/// Fall-off function for [`FovQualityAllocator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FovFalloff {
    /// Quality decreases linearly with angular distance.
    Linear,
    /// Quality decreases as `cos(angle × π / (2 × fovea_radius))`.
    #[default]
    Cosine,
    /// Quality decreases as a Gaussian with σ = `fovea_radius / 2`.
    Gaussian,
}

/// Quality assignment from [`FovQualityAllocator`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FovTileQuality {
    /// Tile index.
    pub index: TileIndex,
    /// Assigned quality level (0 = skip, `max_level` = highest).
    pub level: u8,
    /// Angular distance from the viewport centre to the tile centre (radians).
    pub angle_rad: f32,
}

impl FovQualityAllocator {
    /// Create a new allocator with the given grid and gaze parameters.
    ///
    /// # Errors
    /// Returns [`VrError::InvalidDimensions`] if `fovea_radius_rad` ≤ 0 or
    /// `cutoff_radius_rad` < `fovea_radius_rad`.
    pub fn new(
        grid: TileGrid,
        fovea_radius_rad: f32,
        cutoff_radius_rad: f32,
        max_level: u8,
        falloff: FovFalloff,
    ) -> Result<Self, VrError> {
        if fovea_radius_rad <= 0.0 {
            return Err(VrError::InvalidDimensions(
                "fovea_radius_rad must be > 0".into(),
            ));
        }
        if cutoff_radius_rad < fovea_radius_rad {
            return Err(VrError::InvalidDimensions(
                "cutoff_radius_rad must be ≥ fovea_radius_rad".into(),
            ));
        }
        Ok(Self {
            grid,
            fovea_radius_rad,
            cutoff_radius_rad,
            max_level,
            falloff,
        })
    }

    /// Compute per-tile quality levels for the given gaze direction.
    ///
    /// * `gaze_yaw_rad`   — yaw of the viewer's gaze (radians, −π .. +π)
    /// * `gaze_pitch_rad` — pitch of the viewer's gaze (radians, −π/2 .. +π/2)
    ///
    /// Returns one [`FovTileQuality`] per tile in row-major order.
    #[must_use]
    pub fn allocate(&self, gaze_yaw_rad: f32, gaze_pitch_rad: f32) -> Vec<FovTileQuality> {
        let mut result = Vec::with_capacity(self.grid.tile_count());

        for idx in self.grid.indices() {
            // Tile centre in spherical coordinates
            let tile_pitch = PI / 2.0 - (idx.row as f32 + 0.5) * self.grid.tile_pitch_rad;
            let tile_yaw = -PI + (idx.col as f32 + 0.5) * self.grid.tile_yaw_rad;

            // Great-circle angular distance from gaze to tile centre
            let angle_rad = great_circle_angle(gaze_yaw_rad, gaze_pitch_rad, tile_yaw, tile_pitch);

            let level = self.quality_level(angle_rad);
            result.push(FovTileQuality {
                index: idx,
                level,
                angle_rad,
            });
        }
        result
    }

    /// Compute quality level for a tile at `angle_rad` from gaze centre.
    fn quality_level(&self, angle_rad: f32) -> u8 {
        if angle_rad >= self.cutoff_radius_rad {
            return 0;
        }
        // Normalised distance: 0 at gaze centre, 1 at fovea radius
        let x = (angle_rad / self.fovea_radius_rad).min(1.0);
        let weight = match self.falloff {
            FovFalloff::Linear => (1.0 - x).max(0.0),
            FovFalloff::Cosine => {
                let angle = x * std::f32::consts::FRAC_PI_2;
                angle.cos().max(0.0)
            }
            FovFalloff::Gaussian => {
                let sigma = self.fovea_radius_rad * 0.5;
                let sigma_norm = sigma / self.fovea_radius_rad; // = 0.5
                let x_norm = angle_rad / self.fovea_radius_rad;
                (-x_norm * x_norm / (2.0 * sigma_norm * sigma_norm)).exp()
            }
        };
        let level_f = weight * self.max_level as f32;
        // At minimum 1 if within the cutoff (not skipped)
        (level_f.round() as u8).max(1)
    }

    /// Return the grid used by this allocator.
    #[must_use]
    pub fn grid(&self) -> &TileGrid {
        &self.grid
    }
}

/// Compute the great-circle angular distance between two points on the sphere
/// expressed in (yaw, pitch) = (azimuth, elevation) in radians.
fn great_circle_angle(yaw1: f32, pitch1: f32, yaw2: f32, pitch2: f32) -> f32 {
    let dpitch = pitch2 - pitch1;
    let dyaw = yaw2 - yaw1;
    let h = (dpitch * 0.5).sin().powi(2) + pitch1.cos() * pitch2.cos() * (dyaw * 0.5).sin().powi(2);
    2.0 * h.sqrt().clamp(0.0, 1.0).asin()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fov_region(yaw: f32, pitch: f32, h_deg: f32, v_deg: f32) -> ViewportRegion {
        ViewportRegion::from_fov(yaw, pitch, h_deg.to_radians(), v_deg.to_radians()).unwrap()
    }

    // ── TileGrid ──────────────────────────────────────────────────────────────

    #[test]
    fn tile_grid_rejects_zero_dimensions() {
        assert!(TileGrid::new(0, 4).is_err());
        assert!(TileGrid::new(4, 0).is_err());
    }

    #[test]
    fn tile_grid_correct_tile_count() {
        let g = TileGrid::new(4, 8).unwrap();
        assert_eq!(g.tile_count(), 32);
    }

    #[test]
    fn tile_grid_region_out_of_bounds() {
        let g = TileGrid::new(4, 8).unwrap();
        assert!(g.tile_region(4, 0).is_err());
        assert!(g.tile_region(0, 8).is_err());
    }

    #[test]
    fn tile_grid_angular_widths_cover_sphere() {
        let rows = 6u32;
        let cols = 12u32;
        let g = TileGrid::new(rows, cols).unwrap();
        // Total angular coverage must span the full sphere
        let total_pitch = g.tile_pitch_rad * rows as f32;
        let total_yaw = g.tile_yaw_rad * cols as f32;
        assert!((total_pitch - PI).abs() < 1e-5, "pitch span={total_pitch}");
        assert!((total_yaw - 2.0 * PI).abs() < 1e-5, "yaw span={total_yaw}");
    }

    #[test]
    fn tile_grid_indices_count_matches_tile_count() {
        let g = TileGrid::new(3, 5).unwrap();
        let count = g.indices().count();
        assert_eq!(count, 15);
    }

    // ── TileSelector ─────────────────────────────────────────────────────────

    #[test]
    fn selector_critical_tiles_in_viewport() {
        let grid = TileGrid::new(4, 8).unwrap();
        let selector = TileSelector::new(grid);
        // Looking straight ahead, 90° × 60° FOV
        let viewport = fov_region(0.0, 0.0, 90.0, 60.0);
        let assignments = selector.select(&viewport);
        let critical_count = assignments
            .iter()
            .filter(|a| a.priority == TilePriority::Critical)
            .count();
        assert!(critical_count > 0, "should have critical tiles in viewport");
    }

    #[test]
    fn selector_all_tiles_assigned() {
        let grid = TileGrid::new(4, 8).unwrap();
        let n = grid.tile_count();
        let selector = TileSelector::new(grid);
        let viewport = fov_region(0.0, 0.0, 90.0, 60.0);
        let assignments = selector.select(&viewport);
        assert_eq!(assignments.len(), n);
    }

    #[test]
    fn sorted_assignment_is_critical_first() {
        let grid = TileGrid::new(4, 8).unwrap();
        let selector = TileSelector::new(grid);
        let viewport = fov_region(0.0, 0.0, 60.0, 45.0);
        let sorted = selector.sorted_by_priority(&viewport);
        // First element must be Critical or Near (never Skip before Critical)
        let first_priority = sorted.first().map(|a| a.priority);
        assert!(
            matches!(
                first_priority,
                Some(TilePriority::Critical) | Some(TilePriority::Near)
            ),
            "first priority: {first_priority:?}"
        );
    }

    #[test]
    fn selector_with_zero_near_margin_has_no_near_tiles_outside_viewport() {
        let grid = TileGrid::new(4, 8).unwrap();
        let selector = TileSelector::new(grid).with_near_margin(0.0).unwrap();
        let viewport = fov_region(0.0, 0.0, 45.0, 30.0);
        let assignments = selector.select(&viewport);
        // With zero near margin, tiles with iou == 0 should be Far, not Near
        for a in &assignments {
            if a.iou == 0.0 {
                assert_ne!(
                    a.priority,
                    TilePriority::Near,
                    "tile {:?} iou=0 but Near",
                    a.index
                );
            }
        }
    }

    // ── QualityLadder ─────────────────────────────────────────────────────────

    #[test]
    fn quality_ladder_critical_gets_max_level() {
        let ladder = QualityLadder::new(4).unwrap();
        assert_eq!(ladder.level_for(TilePriority::Critical), 3);
    }

    #[test]
    fn quality_ladder_skip_always_zero() {
        let ladder = QualityLadder::new(4).unwrap();
        assert_eq!(ladder.level_for(TilePriority::Skip), 0);
    }

    #[test]
    fn quality_ladder_rejects_zero_levels() {
        assert!(QualityLadder::new(0).is_err());
    }

    // ── FovQualityAllocator ───────────────────────────────────────────────────

    fn make_allocator(falloff: FovFalloff) -> FovQualityAllocator {
        let grid = TileGrid::new(4, 8).expect("valid 4x8 grid");
        FovQualityAllocator::new(
            grid,
            PI / 4.0, // fovea radius = 45°
            PI * 0.8, // cutoff at 144°
            4,        // max_level = 4 (levels 0..=4)
            falloff,
        )
        .expect("valid allocator params")
    }

    #[test]
    fn fov_allocator_invalid_fovea_radius() {
        let grid = TileGrid::new(4, 8).expect("valid 4x8 grid");
        assert!(FovQualityAllocator::new(grid, 0.0, PI, 4, FovFalloff::Linear).is_err());
    }

    #[test]
    fn fov_allocator_cutoff_less_than_fovea_error() {
        let grid = TileGrid::new(4, 8).expect("valid 4x8 grid");
        assert!(FovQualityAllocator::new(grid, PI / 2.0, PI / 4.0, 4, FovFalloff::Linear).is_err());
    }

    #[test]
    fn fov_allocator_correct_tile_count() {
        let alloc = make_allocator(FovFalloff::Linear);
        let result = alloc.allocate(0.0, 0.0);
        assert_eq!(result.len(), 4 * 8);
    }

    #[test]
    fn fov_allocator_centre_tile_gets_max_level() {
        // Use a fine 8×16 grid so a tile centre falls very close to (yaw=0, pitch=0).
        // The closest tile should receive the highest quality among all tiles.
        let grid = TileGrid::new(8, 16).expect("valid 8x16 grid");
        let alloc = FovQualityAllocator::new(
            grid,
            PI / 2.0, // fovea radius = 90°
            PI * 0.9, // cutoff at 162°
            4,
            FovFalloff::Linear,
        )
        .expect("valid allocator params");
        let result = alloc.allocate(0.0, 0.0);
        // The tile with the smallest angle should have the highest quality
        let best = result
            .iter()
            .min_by(|a, b| {
                a.angle_rad
                    .partial_cmp(&b.angle_rad)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty");
        let max_level = result.iter().map(|t| t.level).max().expect("non-empty");
        assert_eq!(
            best.level, max_level,
            "closest tile should have highest quality"
        );
    }

    #[test]
    fn fov_allocator_tiles_beyond_cutoff_are_zero() {
        let grid = TileGrid::new(4, 8).expect("valid 4x8 grid");
        // Tiny fovea (15°) + cutoff at 30°: most tiles should be zero
        let alloc = FovQualityAllocator::new(
            grid,
            PI / 12.0, // 15°
            PI / 6.0,  // 30°
            4,
            FovFalloff::Cosine,
        )
        .expect("valid allocator params");
        let result = alloc.allocate(0.0, 0.0);
        let zero_count = result.iter().filter(|t| t.level == 0).count();
        // With a 30° cutoff on a 4×8 grid over 180°×360°, many tiles should be skipped
        assert!(zero_count > 0, "expected some skipped tiles, got none");
    }

    #[test]
    fn fov_allocator_quality_decreases_with_angle() {
        // The closest tile should have higher quality than the furthest tile
        let alloc = make_allocator(FovFalloff::Cosine);
        let result = alloc.allocate(0.0, 0.0);
        let min_angle = result
            .iter()
            .min_by(|a, b| {
                a.angle_rad
                    .partial_cmp(&b.angle_rad)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty");
        let max_angle = result
            .iter()
            .filter(|t| t.level > 0)
            .max_by(|a, b| {
                a.angle_rad
                    .partial_cmp(&b.angle_rad)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty");
        assert!(
            min_angle.level >= max_angle.level,
            "closer tile {} should have >= quality than far tile {}",
            min_angle.level,
            max_angle.level
        );
    }

    #[test]
    fn fov_allocator_angle_is_non_negative() {
        let alloc = make_allocator(FovFalloff::Gaussian);
        let result = alloc.allocate(PI / 4.0, PI / 8.0);
        for t in &result {
            assert!(
                t.angle_rad >= 0.0,
                "angle should be non-negative, got {}",
                t.angle_rad
            );
        }
    }

    #[test]
    fn fov_allocator_all_falloff_modes_produce_valid_levels() {
        for &falloff in &[FovFalloff::Linear, FovFalloff::Cosine, FovFalloff::Gaussian] {
            let alloc = make_allocator(falloff);
            let result = alloc.allocate(0.0, 0.0);
            for t in &result {
                assert!(
                    t.level <= alloc.max_level,
                    "level {} > max_level {} for {falloff:?}",
                    t.level,
                    alloc.max_level
                );
            }
        }
    }

    #[test]
    fn fov_allocator_grid_accessor() {
        let alloc = make_allocator(FovFalloff::Linear);
        assert_eq!(alloc.grid().rows, 4);
        assert_eq!(alloc.grid().cols, 8);
    }
}
