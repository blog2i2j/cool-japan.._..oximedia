//! Frame-differencing background subtraction.
//!
//! [`FrameDiffSubtractor`] maintains a reference background frame and computes
//! per-pixel foreground masks by thresholding the absolute pixel difference
//! between consecutive frames.  It is intentionally simple and allocation-lean:
//! only the previous frame is retained, and the output `Vec<bool>` can be
//! reused by callers via repeated calls.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::motion::bg_sub::FrameDiffSubtractor;
//!
//! let w = 4u32;
//! let h = 4u32;
//! let frame = vec![128u8; (w * h) as usize];
//!
//! let mut sub = FrameDiffSubtractor::new(20);
//! // First call seeds the background — all pixels are background.
//! let mask = sub.update(&frame, w, h);
//! assert_eq!(mask.len(), (w * h) as usize);
//! assert!(mask.iter().all(|&fg| !fg));
//!
//! // A frame with one bright pixel becomes foreground.
//! let mut frame2 = frame.clone();
//! frame2[0] = 255;
//! let mask2 = sub.update(&frame2, w, h);
//! assert!(mask2[0]); // pixel 0 changed by 127 > threshold 20
//! ```

/// Frame-differencing background subtractor.
///
/// Compares each new frame against the most-recently seen frame and labels
/// pixels as **foreground** (`true`) when `|current − previous| > threshold`.
///
/// The first call to [`update`](FrameDiffSubtractor::update) always initialises
/// the internal background frame and returns an all-`false` mask (nothing has
/// moved yet).
#[derive(Debug, Clone)]
pub struct FrameDiffSubtractor {
    /// Absolute-difference threshold in \[0, 255\].
    threshold: u8,
    /// The previous frame stored as a flat grayscale byte slice.
    previous: Option<Vec<u8>>,
}

impl FrameDiffSubtractor {
    /// Create a new subtractor with the given per-pixel difference `threshold`.
    ///
    /// A pixel is classified as foreground when
    /// `|current_pixel − previous_pixel| > threshold`.
    ///
    /// * `threshold = 0` marks every pixel as foreground (except the first
    ///   frame).
    /// * `threshold = 255` marks no pixel as foreground.
    #[must_use]
    pub fn new(threshold: u8) -> Self {
        Self {
            threshold,
            previous: None,
        }
    }

    /// Process a new frame and return a foreground mask.
    ///
    /// The mask has one `bool` per pixel in row-major order.
    /// `true` means the pixel is classified as foreground (moving / changed).
    ///
    /// `frame` must be a flat grayscale image of exactly `w × h` bytes.
    /// If the frame dimensions change between calls the internal background is
    /// reset and the new frame is used as the seed (all pixels → `false`).
    ///
    /// # Panics
    ///
    /// Does **not** panic — if `frame.len() != w * h as usize` the excess or
    /// missing bytes are handled gracefully (iteration stops at the shorter
    /// length).
    pub fn update(&mut self, frame: &[u8], w: u32, h: u32) -> Vec<bool> {
        let n = (w as usize).saturating_mul(h as usize);

        match &self.previous {
            None => {
                // First frame: seed background, no foreground.
                self.previous = Some(frame[..frame.len().min(n)].to_vec());
                vec![false; n]
            }
            Some(prev) if prev.len() != n => {
                // Dimensions changed: reset.
                self.previous = Some(frame[..frame.len().min(n)].to_vec());
                vec![false; n]
            }
            Some(prev) => {
                let len = n.min(frame.len()).min(prev.len());
                let mut mask = vec![false; n];
                for i in 0..len {
                    let diff = (frame[i] as i16 - prev[i] as i16).unsigned_abs() as u8;
                    mask[i] = diff > self.threshold;
                }
                // Update background to current frame.
                self.previous = Some(frame[..frame.len().min(n)].to_vec());
                mask
            }
        }
    }

    /// Reset the internal background, forcing the next [`update`] call to
    /// re-seed from the supplied frame.
    pub fn reset(&mut self) {
        self.previous = None;
    }

    /// Return the current threshold value.
    #[must_use]
    pub fn threshold(&self) -> u8 {
        self.threshold
    }

    /// Return `true` if a background frame has been established.
    #[must_use]
    pub fn has_background(&self) -> bool {
        self.previous.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_frame_all_background() {
        let mut sub = FrameDiffSubtractor::new(10);
        let frame = vec![100u8; 16];
        let mask = sub.update(&frame, 4, 4);
        assert_eq!(mask.len(), 16);
        assert!(mask.iter().all(|&fg| !fg));
    }

    #[test]
    fn test_identical_frames_all_background() {
        let frame = vec![50u8; 9];
        let mut sub = FrameDiffSubtractor::new(5);
        sub.update(&frame, 3, 3);
        let mask = sub.update(&frame, 3, 3);
        assert!(mask.iter().all(|&fg| !fg));
    }

    #[test]
    fn test_changed_pixel_becomes_foreground() {
        let frame1 = vec![0u8; 4];
        let mut frame2 = vec![0u8; 4];
        frame2[2] = 100; // pixel 2 changes by 100

        let mut sub = FrameDiffSubtractor::new(20);
        sub.update(&frame1, 2, 2);
        let mask = sub.update(&frame2, 2, 2);

        assert!(!mask[0]);
        assert!(!mask[1]);
        assert!(mask[2]); // diff=100 > threshold=20
        assert!(!mask[3]);
    }

    #[test]
    fn test_threshold_boundary() {
        let frame1 = vec![0u8; 4];
        let mut frame2 = vec![0u8; 4];
        frame2[0] = 30; // diff = 30

        let mut sub_below = FrameDiffSubtractor::new(30); // diff NOT > 30
        sub_below.update(&frame1, 2, 2);
        let mask_below = sub_below.update(&frame2, 2, 2);
        assert!(!mask_below[0]);

        let mut sub_above = FrameDiffSubtractor::new(29); // diff > 29
        sub_above.update(&frame1, 2, 2);
        let mask_above = sub_above.update(&frame2, 2, 2);
        assert!(mask_above[0]);
    }

    #[test]
    fn test_dimension_change_reseeds() {
        let frame_4x4 = vec![128u8; 16];
        let frame_2x2 = vec![255u8; 4];

        let mut sub = FrameDiffSubtractor::new(10);
        sub.update(&frame_4x4, 4, 4);
        // Change dimensions — should reseed without panicking
        let mask = sub.update(&frame_2x2, 2, 2);
        assert_eq!(mask.len(), 4);
        assert!(mask.iter().all(|&fg| !fg));
    }

    #[test]
    fn test_reset_reseeds() {
        let frame1 = vec![0u8; 4];
        let frame2 = vec![255u8; 4];

        let mut sub = FrameDiffSubtractor::new(10);
        sub.update(&frame1, 2, 2);
        sub.reset();
        // After reset, next frame re-seeds → all background
        let mask = sub.update(&frame2, 2, 2);
        assert!(mask.iter().all(|&fg| !fg));
    }

    #[test]
    fn test_has_background() {
        let mut sub = FrameDiffSubtractor::new(5);
        assert!(!sub.has_background());
        sub.update(&[0u8; 4], 2, 2);
        assert!(sub.has_background());
        sub.reset();
        assert!(!sub.has_background());
    }
}
