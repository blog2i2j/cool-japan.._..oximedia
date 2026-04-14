//! Ring-buffer row caching for memory-efficient vertical filter passes.
//!
//! During vertical (column-wise) resampling, a filter of radius `r` needs
//! access to `2r` rows surrounding each output position.  Holding the entire
//! source image in memory wastes space when only a sliding window of rows is
//! required.
//!
//! `RowRingBuffer` implements a fixed-capacity circular buffer of image rows
//! (each row stored as `Vec<f32>`) so that the vertical filter pass can work
//! in a streaming fashion, evicting old rows as new ones are pushed.
//!
//! # Example
//!
//! ```
//! use oximedia_scaling::ring_buffer_cache::RowRingBuffer;
//!
//! let mut ring = RowRingBuffer::new(3, 8);
//! ring.push_row(&[1.0; 8]);
//! ring.push_row(&[2.0; 8]);
//! ring.push_row(&[3.0; 8]);
//! assert_eq!(ring.len(), 3);
//! assert_eq!(ring.get(0), Some(&[1.0f32; 8][..]));
//! // Pushing a 4th row evicts the oldest.
//! ring.push_row(&[4.0; 8]);
//! assert_eq!(ring.get(0), Some(&[2.0f32; 8][..]));
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

// ---------------------------------------------------------------------------
// RowRingBuffer
// ---------------------------------------------------------------------------

/// A fixed-capacity ring buffer of image rows.
///
/// Each row has a uniform width (`row_width` values).  When `capacity` rows
/// have been pushed, subsequent pushes overwrite the oldest row, maintaining
/// a sliding window of the most recently pushed rows.
#[derive(Debug, Clone)]
pub struct RowRingBuffer {
    /// Internal storage: `capacity` rows, each of `row_width` f32 values.
    rows: Vec<Vec<f32>>,
    /// Maximum number of rows that can be held simultaneously.
    capacity: usize,
    /// Width of each row (number of f32 values).
    row_width: usize,
    /// Index of the next slot to write into.
    write_idx: usize,
    /// Number of rows currently stored (<= capacity).
    count: usize,
}

impl RowRingBuffer {
    /// Create a new ring buffer that holds up to `capacity` rows, each of
    /// `row_width` values.
    ///
    /// If `capacity` is zero, the buffer will never store any rows.
    pub fn new(capacity: usize, row_width: usize) -> Self {
        let rows = (0..capacity).map(|_| vec![0.0f32; row_width]).collect();
        Self {
            rows,
            capacity,
            row_width,
            write_idx: 0,
            count: 0,
        }
    }

    /// Push a new row into the buffer.
    ///
    /// If the buffer is at capacity, the oldest row is overwritten.
    /// Only the first `row_width` values of `row` are copied; if `row` is
    /// shorter, the remaining values are zeroed.
    pub fn push_row(&mut self, row: &[f32]) {
        if self.capacity == 0 {
            return;
        }
        let dst = &mut self.rows[self.write_idx];
        let copy_len = row.len().min(self.row_width);
        dst[..copy_len].copy_from_slice(&row[..copy_len]);
        // Zero-fill the remainder if the source row is narrower.
        for v in dst[copy_len..].iter_mut() {
            *v = 0.0;
        }
        self.write_idx = (self.write_idx + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Number of rows currently stored.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if no rows are stored.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Maximum number of rows the buffer can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Width of each row (number of `f32` values).
    pub fn row_width(&self) -> usize {
        self.row_width
    }

    /// Get the `i`-th stored row (0 = oldest, `len()-1` = newest).
    ///
    /// Returns `None` if `i >= len()`.
    pub fn get(&self, i: usize) -> Option<&[f32]> {
        if i >= self.count {
            return None;
        }
        // The oldest row is at `(write_idx - count) mod capacity` (wrapping).
        let start = (self.write_idx + self.capacity - self.count) % self.capacity;
        let actual = (start + i) % self.capacity;
        Some(&self.rows[actual])
    }

    /// Get a mutable reference to the `i`-th stored row.
    pub fn get_mut(&mut self, i: usize) -> Option<&mut [f32]> {
        if i >= self.count {
            return None;
        }
        let start = (self.write_idx + self.capacity - self.count) % self.capacity;
        let actual = (start + i) % self.capacity;
        Some(&mut self.rows[actual])
    }

    /// Clear all stored rows (count drops to 0, capacity unchanged).
    pub fn clear(&mut self) {
        self.count = 0;
        self.write_idx = 0;
    }

    /// Compute the weighted sum of all stored rows.
    ///
    /// `weights` must have at least `len()` elements; extra weights are ignored.
    /// Returns a row of `row_width` values, or `None` if the buffer is empty or
    /// `weights` is too short.
    pub fn weighted_sum(&self, weights: &[f32]) -> Option<Vec<f32>> {
        if self.count == 0 || weights.len() < self.count {
            return None;
        }
        let mut result = vec![0.0f32; self.row_width];
        for i in 0..self.count {
            if let Some(row) = self.get(i) {
                let w = weights[i];
                for (r, &v) in result.iter_mut().zip(row.iter()) {
                    *r += v * w;
                }
            }
        }
        Some(result)
    }

    /// Apply a vertical Lanczos-like filter using the stored rows and a set of
    /// kernel weights centred at the buffer.
    ///
    /// This is a convenience wrapper over [`weighted_sum`](Self::weighted_sum)
    /// that normalises the weights so they sum to 1.0 before applying.
    pub fn apply_vertical_filter(&self, weights: &[f32]) -> Option<Vec<f32>> {
        if self.count == 0 || weights.len() < self.count {
            return None;
        }
        let sum: f32 = weights[..self.count].iter().sum();
        if sum.abs() < f32::EPSILON {
            return self.weighted_sum(weights);
        }
        let normalised: Vec<f32> = weights[..self.count].iter().map(|&w| w / sum).collect();
        self.weighted_sum(&normalised)
    }
}

/// Apply a vertical filter to an entire image using a ring buffer, producing
/// output rows one at a time.
///
/// This is a demonstration entry point that shows how to use `RowRingBuffer`
/// in a vertical resampling pass. The filter kernel is applied centred at
/// each output row position.
///
/// # Parameters
///
/// * `src`        — row-major f32 image, `src_w × src_h` values.
/// * `src_w`      — source width.
/// * `src_h`      — source height.
/// * `kernel`     — symmetric vertical filter kernel (length = filter diameter).
///
/// # Returns
///
/// A filtered image of the same dimensions, or `None` if inputs are invalid.
pub fn vertical_filter_with_ring(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    kernel: &[f32],
) -> Option<Vec<f32>> {
    if src_w == 0 || src_h == 0 || kernel.is_empty() {
        return None;
    }
    if src.len() < src_w * src_h {
        return None;
    }

    let k_len = kernel.len();
    let half = k_len / 2;
    let cap = k_len;
    let mut ring = RowRingBuffer::new(cap, src_w);
    let mut output = vec![0.0f32; src_w * src_h];

    // Pre-fill with mirrored border rows.
    for i in 0..half {
        let mirror_row = half.saturating_sub(i + 1).min(src_h - 1);
        ring.push_row(&src[mirror_row * src_w..(mirror_row + 1) * src_w]);
    }

    // Push initial rows.
    let initial_push = (half + 1).min(src_h);
    for y in 0..initial_push {
        ring.push_row(&src[y * src_w..(y + 1) * src_w]);
    }

    // Process each output row.
    for y in 0..src_h {
        // Ensure we have the right rows in the ring.
        let next_needed = y + half + 1;
        if next_needed < src_h && ring.len() == cap {
            ring.push_row(&src[next_needed * src_w..(next_needed + 1) * src_w]);
        } else if next_needed >= src_h && ring.len() == cap {
            // Mirror the border.
            let mirror = (2 * src_h).saturating_sub(next_needed + 2).min(src_h - 1);
            ring.push_row(&src[mirror * src_w..(mirror + 1) * src_w]);
        }

        if let Some(filtered) = ring.apply_vertical_filter(kernel) {
            output[y * src_w..(y + 1) * src_w].copy_from_slice(&filtered);
        }
    }

    Some(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_buffer_is_empty() {
        let ring = RowRingBuffer::new(4, 8);
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.capacity(), 4);
        assert_eq!(ring.row_width(), 8);
    }

    #[test]
    fn test_push_and_get() {
        let mut ring = RowRingBuffer::new(3, 4);
        ring.push_row(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ring.len(), 1);
        assert_eq!(ring.get(0), Some(&[1.0, 2.0, 3.0, 4.0][..]));
    }

    #[test]
    fn test_push_fills_to_capacity() {
        let mut ring = RowRingBuffer::new(3, 2);
        ring.push_row(&[1.0, 1.0]);
        ring.push_row(&[2.0, 2.0]);
        ring.push_row(&[3.0, 3.0]);
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.get(0), Some(&[1.0, 1.0][..]));
        assert_eq!(ring.get(1), Some(&[2.0, 2.0][..]));
        assert_eq!(ring.get(2), Some(&[3.0, 3.0][..]));
    }

    #[test]
    fn test_push_evicts_oldest() {
        let mut ring = RowRingBuffer::new(3, 2);
        ring.push_row(&[1.0, 1.0]);
        ring.push_row(&[2.0, 2.0]);
        ring.push_row(&[3.0, 3.0]);
        ring.push_row(&[4.0, 4.0]); // evicts row 1
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.get(0), Some(&[2.0, 2.0][..]));
        assert_eq!(ring.get(1), Some(&[3.0, 3.0][..]));
        assert_eq!(ring.get(2), Some(&[4.0, 4.0][..]));
    }

    #[test]
    fn test_get_out_of_bounds() {
        let mut ring = RowRingBuffer::new(4, 2);
        ring.push_row(&[1.0, 1.0]);
        assert_eq!(ring.get(1), None);
        assert_eq!(ring.get(100), None);
    }

    #[test]
    fn test_clear() {
        let mut ring = RowRingBuffer::new(3, 2);
        ring.push_row(&[1.0, 1.0]);
        ring.push_row(&[2.0, 2.0]);
        ring.clear();
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.capacity(), 3);
    }

    #[test]
    fn test_short_row_zero_padded() {
        let mut ring = RowRingBuffer::new(2, 4);
        ring.push_row(&[5.0, 6.0]); // only 2 of 4
        let row = ring.get(0).expect("should have row");
        assert_eq!(row, &[5.0, 6.0, 0.0, 0.0]);
    }

    #[test]
    fn test_zero_capacity_never_stores() {
        let mut ring = RowRingBuffer::new(0, 4);
        ring.push_row(&[1.0, 2.0, 3.0, 4.0]);
        assert!(ring.is_empty());
    }

    #[test]
    fn test_weighted_sum_basic() {
        let mut ring = RowRingBuffer::new(3, 2);
        ring.push_row(&[1.0, 2.0]);
        ring.push_row(&[3.0, 4.0]);
        ring.push_row(&[5.0, 6.0]);
        let result = ring.weighted_sum(&[1.0, 0.0, 0.0]).expect("sum ok");
        assert_eq!(result, vec![1.0, 2.0]);
    }

    #[test]
    fn test_weighted_sum_mixed_weights() {
        let mut ring = RowRingBuffer::new(2, 3);
        ring.push_row(&[2.0, 4.0, 6.0]);
        ring.push_row(&[1.0, 3.0, 5.0]);
        let result = ring.weighted_sum(&[0.5, 0.5]).expect("sum ok");
        assert_eq!(result, vec![1.5, 3.5, 5.5]);
    }

    #[test]
    fn test_weighted_sum_insufficient_weights() {
        let mut ring = RowRingBuffer::new(3, 2);
        ring.push_row(&[1.0, 2.0]);
        ring.push_row(&[3.0, 4.0]);
        let result = ring.weighted_sum(&[1.0]); // need 2 weights
        assert!(result.is_none());
    }

    #[test]
    fn test_apply_vertical_filter_normalised() {
        let mut ring = RowRingBuffer::new(3, 2);
        ring.push_row(&[10.0, 20.0]);
        ring.push_row(&[10.0, 20.0]);
        ring.push_row(&[10.0, 20.0]);
        // Uniform weights: normalised to 1/3 each → average = same values.
        let result = ring.apply_vertical_filter(&[1.0, 1.0, 1.0]).expect("ok");
        for &v in &result {
            assert!((v - result[0]).abs() < 1e-6 || (v - result[1]).abs() < 1e-6);
        }
        assert!((result[0] - 10.0).abs() < 1e-4);
        assert!((result[1] - 20.0).abs() < 1e-4);
    }

    #[test]
    fn test_get_mut_modifies_row() {
        let mut ring = RowRingBuffer::new(2, 3);
        ring.push_row(&[1.0, 2.0, 3.0]);
        if let Some(row) = ring.get_mut(0) {
            row[0] = 99.0;
        }
        assert_eq!(ring.get(0).expect("row")[0], 99.0);
    }

    #[test]
    fn test_wrap_around_multiple_times() {
        let mut ring = RowRingBuffer::new(2, 1);
        for i in 0..10 {
            ring.push_row(&[i as f32]);
        }
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.get(0), Some(&[8.0f32][..]));
        assert_eq!(ring.get(1), Some(&[9.0f32][..]));
    }

    #[test]
    fn test_vertical_filter_uniform_image() {
        // A uniform image filtered with any kernel should remain uniform.
        let src: Vec<f32> = vec![42.0; 4 * 4];
        let kernel = vec![0.25, 0.5, 0.25];
        let result = vertical_filter_with_ring(&src, 4, 4, &kernel).expect("ok");
        assert_eq!(result.len(), 16);
        for &v in &result {
            assert!(
                (v - 42.0).abs() < 1.0,
                "uniform value should be preserved, got {v}"
            );
        }
    }

    #[test]
    fn test_vertical_filter_empty_inputs() {
        assert!(vertical_filter_with_ring(&[], 0, 0, &[1.0]).is_none());
        assert!(vertical_filter_with_ring(&[1.0], 1, 1, &[]).is_none());
    }

    #[test]
    fn test_vertical_filter_output_size() {
        let src: Vec<f32> = vec![1.0; 8 * 6];
        let kernel = vec![0.5, 0.5];
        let result = vertical_filter_with_ring(&src, 8, 6, &kernel).expect("ok");
        assert_eq!(result.len(), 8 * 6);
    }

    #[test]
    fn test_debug_format() {
        let ring = RowRingBuffer::new(2, 3);
        let s = format!("{ring:?}");
        assert!(s.contains("RowRingBuffer"));
    }
}
