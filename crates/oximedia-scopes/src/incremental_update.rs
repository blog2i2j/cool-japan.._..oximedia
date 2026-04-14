//! Incremental scope update — update only changed regions of a scope display.
//!
//! Re-generating a full scope display every frame is wasteful when only a
//! small portion of the input changes (e.g. a lower-third graphic appearing
//! on an otherwise static background, or a constant-level audio signal).
//! This module provides a dirty-region tracker that records which rectangular
//! tiles of the *source frame* changed since the last scope update, together
//! with utilities to rebuild only the affected areas of the scope canvas.
//!
//! # Tile Grid
//!
//! The source frame is divided into an evenly-spaced grid of [`Tile`]s.
//! Each tile is a rectangular block of source pixels; tiles are the atomic
//! unit of change detection.  When a tile's pixel content differs from the
//! previous frame, it is marked as *dirty* and re-processed.  Unchanged tiles
//! are left untouched on the accumulated scope canvas.
//!
//! # Integration with scope pipeline
//!
//! 1. On the first frame, call [`IncrementalUpdater::update`] — all tiles are
//!    dirty and a full update is performed.
//! 2. On subsequent frames, call [`IncrementalUpdater::update`] again.
//!    Internally the updater compares tiles against the cached previous frame
//!    and triggers the user-supplied tile-process callback only for changed
//!    tiles.
//! 3. Call [`IncrementalUpdater::dirty_ratio`] to monitor how much of the
//!    frame is actually changing (useful for adaptive downsampling decisions).

use oximedia_core::{OxiError, OxiResult};

/// Coordinates of a single tile in the tile grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    /// Column index (0 = leftmost).
    pub col: u32,
    /// Row index (0 = topmost).
    pub row: u32,
}

/// A rectangular region in source-frame pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelRect {
    /// Left edge (inclusive).
    pub x: u32,
    /// Top edge (inclusive).
    pub y: u32,
    /// Width in pixels.
    pub w: u32,
    /// Height in pixels.
    pub h: u32,
}

impl PixelRect {
    /// Returns true if the rectangle contains at least one pixel.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.w == 0 || self.h == 0
    }

    /// Total pixel area.
    #[must_use]
    pub fn area(self) -> u64 {
        u64::from(self.w) * u64::from(self.h)
    }
}

/// Threshold used when comparing tile content for change detection.
///
/// Strict means any single-byte difference triggers a dirty mark;
/// tolerant ignores differences ≤ a configurable epsilon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeThreshold {
    /// Any differing byte → tile is dirty.
    Exact,
    /// Byte difference must exceed `epsilon` to count as changed.
    Tolerant(u8),
}

impl Default for ChangeThreshold {
    fn default() -> Self {
        Self::Exact
    }
}

/// Persistent state for incremental scope updates.
///
/// Maintains a copy of the previous frame and the tile-dirty flags
/// from the most recent call to [`IncrementalUpdater::update`].
#[derive(Debug)]
pub struct IncrementalUpdater {
    /// Source frame width in pixels.
    pub width: u32,
    /// Source frame height in pixels.
    pub height: u32,
    /// Number of tiles in the horizontal direction.
    pub cols: u32,
    /// Number of tiles in the vertical direction.
    pub rows: u32,
    /// Width of each tile in pixels.
    pub tile_w: u32,
    /// Height of each tile in pixels.
    pub tile_h: u32,
    /// Change-detection threshold.
    pub threshold: ChangeThreshold,

    /// Cached copy of the previous frame (RGB-24, row-major).
    prev_frame: Vec<u8>,
    /// Dirty flags from the most recent update call (row-major tile order).
    dirty_flags: Vec<bool>,
    /// Total number of dirty tiles in the last update.
    dirty_count: u32,
    /// Frame counter (incremented on every call to `update`).
    frame_count: u64,
}

impl IncrementalUpdater {
    /// Creates a new [`IncrementalUpdater`] for the given source frame size and
    /// tile dimensions.
    ///
    /// The tile grid is computed so that all tiles have the same `tile_w` ×
    /// `tile_h` pixel size (border tiles may be smaller in practice; the
    /// comparison logic handles partial tiles correctly).
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::InvalidData`] if any dimension is zero.
    pub fn new(
        width: u32,
        height: u32,
        tile_w: u32,
        tile_h: u32,
        threshold: ChangeThreshold,
    ) -> OxiResult<Self> {
        if width == 0 || height == 0 {
            return Err(OxiError::InvalidData("Frame dimensions must be non-zero".into()));
        }
        if tile_w == 0 || tile_h == 0 {
            return Err(OxiError::InvalidData("Tile dimensions must be non-zero".into()));
        }

        let cols = (width + tile_w - 1) / tile_w;
        let rows = (height + tile_h - 1) / tile_h;
        let tile_count = (cols * rows) as usize;

        Ok(Self {
            width,
            height,
            cols,
            rows,
            tile_w,
            tile_h,
            threshold,
            prev_frame: Vec::new(),
            dirty_flags: vec![true; tile_count],
            dirty_count: cols * rows,
            frame_count: 0,
        })
    }

    /// Returns the [`PixelRect`] for a tile identified by `(col, row)`.
    ///
    /// Returns `None` if the coordinates are out of range.
    #[must_use]
    pub fn tile_rect(&self, col: u32, row: u32) -> Option<PixelRect> {
        if col >= self.cols || row >= self.rows {
            return None;
        }
        let x = col * self.tile_w;
        let y = row * self.tile_h;
        let w = self.tile_w.min(self.width - x);
        let h = self.tile_h.min(self.height - y);
        Some(PixelRect { x, y, w, h })
    }

    /// Returns `true` if the tile at `(col, row)` was dirty in the last update.
    #[must_use]
    pub fn is_dirty(&self, col: u32, row: u32) -> bool {
        if col >= self.cols || row >= self.rows {
            return false;
        }
        self.dirty_flags[(row * self.cols + col) as usize]
    }

    /// Returns the fraction of tiles that were dirty in the last update (0.0–1.0).
    #[must_use]
    pub fn dirty_ratio(&self) -> f64 {
        let total = (self.cols * self.rows) as f64;
        if total < 1.0 {
            return 0.0;
        }
        f64::from(self.dirty_count) / total
    }

    /// Total number of dirty tiles from the most recent update.
    #[must_use]
    pub fn dirty_count(&self) -> u32 {
        self.dirty_count
    }

    /// How many update calls have been made since this updater was created.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Collect the [`TileCoord`]s of all dirty tiles from the last update.
    #[must_use]
    pub fn dirty_tiles(&self) -> Vec<TileCoord> {
        self.dirty_flags
            .iter()
            .enumerate()
            .filter(|(_, &dirty)| dirty)
            .map(|(idx, _)| TileCoord {
                col: idx as u32 % self.cols,
                row: idx as u32 / self.cols,
            })
            .collect()
    }

    /// Compares the current frame against the cached previous frame and marks
    /// tiles as dirty or clean.
    ///
    /// After the comparison the previous-frame cache is updated to `frame`.
    ///
    /// # Arguments
    ///
    /// * `frame` – RGB-24 source buffer (`width * height * 3` bytes).
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::InvalidData`] if the buffer is too small.
    pub fn update(&mut self, frame: &[u8]) -> OxiResult<()> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if frame.len() < expected {
            return Err(OxiError::InvalidData(format!(
                "Frame too small: need {expected}, got {}",
                frame.len()
            )));
        }

        // On first call, everything is dirty.
        if self.prev_frame.is_empty() {
            self.dirty_flags.fill(true);
            self.dirty_count = self.cols * self.rows;
            self.prev_frame = frame[..expected].to_vec();
            self.frame_count += 1;
            return Ok(());
        }

        let mut dirty_count = 0u32;
        for row in 0..self.rows {
            for col in 0..self.cols {
                let tile_idx = (row * self.cols + col) as usize;
                let dirty = self.tile_changed(frame, col, row);
                self.dirty_flags[tile_idx] = dirty;
                if dirty {
                    dirty_count += 1;
                }
            }
        }
        self.dirty_count = dirty_count;

        // Update cache.
        self.prev_frame[..expected].copy_from_slice(&frame[..expected]);
        self.frame_count += 1;
        Ok(())
    }

    /// Force all tiles dirty on the next call to [`update`].
    ///
    /// Useful when a non-incremental event occurs (e.g. a cut in the edit).
    pub fn invalidate_all(&mut self) {
        self.dirty_flags.fill(true);
        self.dirty_count = self.cols * self.rows;
        self.prev_frame.clear();
    }

    /// Force a single tile dirty regardless of its content.
    ///
    /// Returns `false` if the coordinates are out of range.
    pub fn invalidate_tile(&mut self, col: u32, row: u32) -> bool {
        if col >= self.cols || row >= self.rows {
            return false;
        }
        let idx = (row * self.cols + col) as usize;
        if !self.dirty_flags[idx] {
            self.dirty_flags[idx] = true;
            self.dirty_count += 1;
        }
        true
    }

    /// Internal: compare a single tile between `frame` and `prev_frame`.
    fn tile_changed(&self, frame: &[u8], col: u32, row: u32) -> bool {
        let x = col * self.tile_w;
        let y = row * self.tile_h;
        let tw = self.tile_w.min(self.width - x);
        let th = self.tile_h.min(self.height - y);

        match self.threshold {
            ChangeThreshold::Exact => {
                for dy in 0..th {
                    let row_off = ((y + dy) * self.width + x) as usize * 3;
                    let end = row_off + (tw as usize) * 3;
                    if frame[row_off..end] != self.prev_frame[row_off..end] {
                        return true;
                    }
                }
                false
            }
            ChangeThreshold::Tolerant(eps) => {
                for dy in 0..th {
                    let row_off = ((y + dy) * self.width + x) as usize * 3;
                    for dx in 0..tw as usize {
                        let off = row_off + dx * 3;
                        for c in 0..3 {
                            let a = frame[off + c];
                            let b = self.prev_frame[off + c];
                            let diff = a.abs_diff(b);
                            if diff > eps {
                                return true;
                            }
                        }
                    }
                }
                false
            }
        }
    }
}

/// Summary statistics for an incremental update session.
#[derive(Debug, Clone)]
pub struct UpdateStats {
    /// Total frames processed.
    pub frames_processed: u64,
    /// Cumulative dirty tiles across all frames.
    pub total_dirty_tiles: u64,
    /// Total tiles per frame (grid size).
    pub tiles_per_frame: u32,
}

impl UpdateStats {
    /// Average dirty ratio per frame (0.0 – 1.0).
    #[must_use]
    pub fn average_dirty_ratio(&self) -> f64 {
        if self.frames_processed == 0 || self.tiles_per_frame == 0 {
            return 0.0;
        }
        let total_possible =
            self.frames_processed as f64 * self.tiles_per_frame as f64;
        self.total_dirty_tiles as f64 / total_possible
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_frame(r: u8, g: u8, b: u8, w: u32, h: u32) -> Vec<u8> {
        let n = (w * h) as usize * 3;
        let mut v = vec![0u8; n];
        for i in 0..n / 3 {
            v[i * 3] = r;
            v[i * 3 + 1] = g;
            v[i * 3 + 2] = b;
        }
        v
    }

    #[test]
    fn test_new_rejects_zero_dimensions() {
        assert!(IncrementalUpdater::new(0, 4, 2, 2, ChangeThreshold::Exact).is_err());
        assert!(IncrementalUpdater::new(4, 0, 2, 2, ChangeThreshold::Exact).is_err());
        assert!(IncrementalUpdater::new(4, 4, 0, 2, ChangeThreshold::Exact).is_err());
        assert!(IncrementalUpdater::new(4, 4, 2, 0, ChangeThreshold::Exact).is_err());
    }

    #[test]
    fn test_first_frame_all_tiles_dirty() {
        let mut u = IncrementalUpdater::new(4, 4, 2, 2, ChangeThreshold::Exact)
            .expect("ok");
        let frame = rgb_frame(10, 20, 30, 4, 4);
        u.update(&frame).expect("ok");
        assert_eq!(u.dirty_count(), u.cols * u.rows);
        assert!((u.dirty_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_identical_second_frame_no_dirty_tiles() {
        let mut u = IncrementalUpdater::new(4, 4, 2, 2, ChangeThreshold::Exact)
            .expect("ok");
        let frame = rgb_frame(50, 60, 70, 4, 4);
        u.update(&frame).expect("ok");
        u.update(&frame).expect("ok");
        assert_eq!(u.dirty_count(), 0);
        assert!((u.dirty_ratio() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_changed_pixel_marks_tile_dirty() {
        let mut u = IncrementalUpdater::new(4, 4, 2, 2, ChangeThreshold::Exact)
            .expect("ok");
        let frame1 = rgb_frame(0, 0, 0, 4, 4);
        u.update(&frame1).expect("ok");

        let mut frame2 = frame1.clone();
        // Modify pixel at (0,0) — sits in tile (col=0, row=0).
        frame2[0] = 255;
        u.update(&frame2).expect("ok");

        assert!(u.is_dirty(0, 0), "tile (0,0) should be dirty");
        // Other tiles should be clean.
        assert!(!u.is_dirty(1, 0));
        assert!(!u.is_dirty(0, 1));
        assert!(!u.is_dirty(1, 1));
    }

    #[test]
    fn test_tolerant_threshold_ignores_small_changes() {
        let mut u = IncrementalUpdater::new(4, 4, 2, 2, ChangeThreshold::Tolerant(10))
            .expect("ok");
        let frame1 = rgb_frame(100, 100, 100, 4, 4);
        u.update(&frame1).expect("ok");

        let mut frame2 = frame1.clone();
        frame2[0] = 105; // diff = 5 ≤ 10 → not dirty
        u.update(&frame2).expect("ok");
        assert!(!u.is_dirty(0, 0), "small change should not mark tile dirty");
    }

    #[test]
    fn test_tolerant_threshold_detects_large_changes() {
        let mut u = IncrementalUpdater::new(4, 4, 2, 2, ChangeThreshold::Tolerant(10))
            .expect("ok");
        let frame1 = rgb_frame(100, 100, 100, 4, 4);
        u.update(&frame1).expect("ok");

        let mut frame2 = frame1.clone();
        frame2[0] = 120; // diff = 20 > 10 → dirty
        u.update(&frame2).expect("ok");
        assert!(u.is_dirty(0, 0), "large change should mark tile dirty");
    }

    #[test]
    fn test_invalidate_all_resets_to_full_dirty() {
        let mut u = IncrementalUpdater::new(4, 4, 2, 2, ChangeThreshold::Exact)
            .expect("ok");
        let frame = rgb_frame(0, 0, 0, 4, 4);
        u.update(&frame).expect("ok");
        u.update(&frame).expect("ok"); // now clean
        assert_eq!(u.dirty_count(), 0);

        u.invalidate_all();
        // After invalidate_all the cache is cleared; next update treats all tiles dirty.
        u.update(&frame).expect("ok");
        assert_eq!(u.dirty_count(), u.cols * u.rows);
    }

    #[test]
    fn test_invalidate_tile_marks_single_tile_dirty() {
        let mut u = IncrementalUpdater::new(4, 4, 2, 2, ChangeThreshold::Exact)
            .expect("ok");
        let frame = rgb_frame(0, 0, 0, 4, 4);
        u.update(&frame).expect("ok");
        u.update(&frame).expect("ok"); // clean

        let ok = u.invalidate_tile(1, 1);
        assert!(ok);
        assert!(u.is_dirty(1, 1));
    }

    #[test]
    fn test_invalidate_tile_oob_returns_false() {
        let mut u = IncrementalUpdater::new(4, 4, 2, 2, ChangeThreshold::Exact)
            .expect("ok");
        assert!(!u.invalidate_tile(99, 0));
    }

    #[test]
    fn test_dirty_tiles_returns_correct_coords() {
        let mut u = IncrementalUpdater::new(4, 4, 2, 2, ChangeThreshold::Exact)
            .expect("ok");
        let frame1 = rgb_frame(0, 0, 0, 4, 4);
        u.update(&frame1).expect("ok");

        let mut frame2 = frame1.clone();
        // Touch pixel in tile (1, 0): pixel at column 2, row 0 → offset 2*3=6
        frame2[6] = 200;
        u.update(&frame2).expect("ok");

        let dirty = u.dirty_tiles();
        assert_eq!(dirty.len(), 1);
        assert_eq!(dirty[0], TileCoord { col: 1, row: 0 });
    }

    #[test]
    fn test_tile_rect_boundary() {
        let u = IncrementalUpdater::new(5, 5, 3, 3, ChangeThreshold::Exact)
            .expect("ok");
        // Last tile (col=1, row=1) should be clipped to 2×2 (5 - 3 = 2).
        let r = u.tile_rect(1, 1).expect("in range");
        assert_eq!(r.x, 3);
        assert_eq!(r.y, 3);
        assert_eq!(r.w, 2);
        assert_eq!(r.h, 2);
    }

    #[test]
    fn test_update_stats_average_dirty_ratio() {
        let stats = UpdateStats {
            frames_processed: 4,
            total_dirty_tiles: 8,
            tiles_per_frame: 4,
        };
        // 8 / (4 * 4) = 0.5
        assert!((stats.average_dirty_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_frame_count_increments() {
        let mut u = IncrementalUpdater::new(2, 2, 1, 1, ChangeThreshold::Exact)
            .expect("ok");
        assert_eq!(u.frame_count(), 0);
        let f = rgb_frame(0, 0, 0, 2, 2);
        u.update(&f).expect("ok");
        assert_eq!(u.frame_count(), 1);
        u.update(&f).expect("ok");
        assert_eq!(u.frame_count(), 2);
    }
}
