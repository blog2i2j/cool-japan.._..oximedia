// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Key/fill separation for external keying workflows.
//!
//! In professional video production, keying often uses two separate signals:
//! - **Key** (alpha matte): a greyscale signal where luminance defines
//!   transparency (white = opaque, black = transparent).
//! - **Fill**: the actual image to be composited over the background.
//!
//! `KeyFillSeparator` derives both from a single RGBA or luma+RGB frame.

/// Separator that derives a key matte and fill signal from a raw frame.
///
/// The input frame is assumed to be 8-bit RGBA interleaved (4 bytes per pixel).
/// If the frame does not satisfy that invariant, extraction returns empty vectors.
pub struct KeyFillSeparator;

impl KeyFillSeparator {
    /// Extract a key matte and fill from a raw RGBA frame.
    ///
    /// **Key channel**: each pixel's alpha value is compared against `threshold`
    /// (0.0 = fully transparent, 1.0 = fully opaque, range `[0.0, 1.0]`).
    /// - Pixels with `alpha / 255 >= threshold` produce a white key pixel (255).
    /// - Pixels below threshold produce a black key pixel (0).
    ///
    /// **Fill channel**: the RGB channels of every pixel are copied verbatim;
    /// the alpha byte in the fill output is set to 255 (fully opaque).
    ///
    /// Both returned `Vec<u8>` are RGBA-ordered and have the same length as
    /// `frame`.  An empty pair is returned when `frame.len() % 4 != 0`.
    ///
    /// # Arguments
    ///
    /// * `frame`     – Interleaved RGBA bytes.
    /// * `threshold` – Alpha threshold clamped to `[0.0, 1.0]`.
    ///
    /// # Returns
    ///
    /// `(key, fill)` where both have the same length as `frame`.
    pub fn extract_key(frame: &[u8], threshold: f32) -> (Vec<u8>, Vec<u8>) {
        if frame.len() % 4 != 0 {
            return (Vec::new(), Vec::new());
        }

        let threshold_u8 = (threshold.clamp(0.0, 1.0) * 255.0) as u8;
        let num_pixels = frame.len() / 4;
        let mut key = Vec::with_capacity(frame.len());
        let mut fill = Vec::with_capacity(frame.len());

        for p in 0..num_pixels {
            let base = p * 4;
            let r = frame[base];
            let g = frame[base + 1];
            let b = frame[base + 2];
            let a = frame[base + 3];

            // Key: binary threshold on alpha
            let key_val: u8 = if a >= threshold_u8 { 255 } else { 0 };
            key.push(key_val);
            key.push(key_val);
            key.push(key_val);
            key.push(255);

            // Fill: original RGB, alpha forced to 255
            fill.push(r);
            fill.push(g);
            fill.push(b);
            fill.push(255);
        }

        (key, fill)
    }

    /// Extract a *soft* key matte — the key pixel luminance directly reflects
    /// the original alpha value (anti-aliased / feathered keying).
    ///
    /// Unlike [`extract_key`][Self::extract_key], no thresholding is applied;
    /// the alpha channel value is used directly as the greyscale key intensity.
    pub fn extract_soft_key(frame: &[u8]) -> (Vec<u8>, Vec<u8>) {
        if frame.len() % 4 != 0 {
            return (Vec::new(), Vec::new());
        }

        let num_pixels = frame.len() / 4;
        let mut key = Vec::with_capacity(frame.len());
        let mut fill = Vec::with_capacity(frame.len());

        for p in 0..num_pixels {
            let base = p * 4;
            let r = frame[base];
            let g = frame[base + 1];
            let b = frame[base + 2];
            let a = frame[base + 3];

            // Key: replicate alpha to RGB
            key.push(a);
            key.push(a);
            key.push(a);
            key.push(255);

            // Fill: original RGB
            fill.push(r);
            fill.push(g);
            fill.push(b);
            fill.push(255);
        }

        (key, fill)
    }

    /// Composite fill over background using the key matte.
    ///
    /// Performs `output = fill * (key/255) + background * (1 - key/255)` per
    /// channel.  All inputs must be RGBA interleaved with the same length.
    /// Returns an empty `Vec` if lengths differ or are not a multiple of 4.
    pub fn composite(key: &[u8], fill: &[u8], background: &[u8]) -> Vec<u8> {
        if key.len() != fill.len() || fill.len() != background.len() || key.len() % 4 != 0 {
            return Vec::new();
        }

        let num_pixels = key.len() / 4;
        let mut out = Vec::with_capacity(key.len());

        for p in 0..num_pixels {
            let base = p * 4;
            for ch in 0..3 {
                let k = key[base + ch] as f32 / 255.0;
                let f = fill[base + ch] as f32;
                let bg = background[base + ch] as f32;
                out.push((f * k + bg * (1.0 - k)).round() as u8);
            }
            out.push(255); // alpha output is fully opaque
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgba(r: u8, g: u8, b: u8, a: u8, count: usize) -> Vec<u8> {
        let pixel = [r, g, b, a];
        pixel.iter().copied().cycle().take(count * 4).collect()
    }

    #[test]
    fn extract_key_output_length() {
        let frame = make_rgba(200, 100, 50, 255, 10);
        let (key, fill) = KeyFillSeparator::extract_key(&frame, 0.5);
        assert_eq!(key.len(), frame.len());
        assert_eq!(fill.len(), frame.len());
    }

    #[test]
    fn extract_key_above_threshold_is_white() {
        let frame = make_rgba(10, 20, 30, 200, 1); // alpha=200, threshold=0.5→128
        let (key, _fill) = KeyFillSeparator::extract_key(&frame, 0.5);
        assert_eq!(key[0], 255);
    }

    #[test]
    fn extract_key_below_threshold_is_black() {
        let frame = make_rgba(10, 20, 30, 50, 1); // alpha=50, threshold=0.5→128
        let (key, _fill) = KeyFillSeparator::extract_key(&frame, 0.5);
        assert_eq!(key[0], 0);
    }

    #[test]
    fn fill_rgb_preserved() {
        let frame = make_rgba(100, 150, 200, 255, 1);
        let (_key, fill) = KeyFillSeparator::extract_key(&frame, 0.5);
        assert_eq!(&fill[0..3], &[100, 150, 200]);
    }

    #[test]
    fn fill_alpha_is_255() {
        let frame = make_rgba(10, 20, 30, 100, 1);
        let (_key, fill) = KeyFillSeparator::extract_key(&frame, 0.0);
        assert_eq!(fill[3], 255);
    }

    #[test]
    fn invalid_frame_length_returns_empty() {
        let bad = vec![0u8; 7];
        let (key, fill) = KeyFillSeparator::extract_key(&bad, 0.5);
        assert!(key.is_empty());
        assert!(fill.is_empty());
    }

    #[test]
    fn soft_key_uses_alpha_directly() {
        let frame = make_rgba(0, 0, 0, 128, 1);
        let (key, _fill) = KeyFillSeparator::extract_soft_key(&frame);
        assert_eq!(key[0], 128);
        assert_eq!(key[1], 128);
        assert_eq!(key[2], 128);
    }

    #[test]
    fn composite_full_key_returns_fill() {
        let key = make_rgba(255, 255, 255, 255, 1);
        let fill = make_rgba(200, 100, 50, 255, 1);
        let bg = make_rgba(0, 0, 0, 255, 1);
        let out = KeyFillSeparator::composite(&key, &fill, &bg);
        assert_eq!(out[0], 200);
        assert_eq!(out[1], 100);
        assert_eq!(out[2], 50);
    }

    #[test]
    fn composite_zero_key_returns_background() {
        let key = make_rgba(0, 0, 0, 255, 1);
        let fill = make_rgba(200, 100, 50, 255, 1);
        let bg = make_rgba(10, 20, 30, 255, 1);
        let out = KeyFillSeparator::composite(&key, &fill, &bg);
        assert_eq!(out[0], 10);
        assert_eq!(out[1], 20);
        assert_eq!(out[2], 30);
    }
}
