//! GPU readback utilities.
//!
//! Provides facilities for reading GPU framebuffer / texture data back to
//! CPU-accessible memory.  On the CPU-stub backend this is a direct copy
//! (identity transform); on a real GPU backend the same interface wraps
//! `wgpu::Buffer::map_read`.
//!
//! # Example
//!
//! ```rust
//! use oximedia_gpu::readback::GpuReadback;
//!
//! let data = vec![0xFFu8; 4 * 4 * 4]; // 4×4 RGBA
//! let out = GpuReadback::download(4, 4, &data);
//! assert_eq!(out, data);
//! ```

#![allow(dead_code)]

// ── GpuReadback ───────────────────────────────────────────────────────────────

/// GPU → CPU readback helper.
///
/// The current implementation operates as a CPU stub (identity copy).
/// Replace the body of [`GpuReadback::download`] and [`GpuReadback::download_region`] with actual
/// WGPU buffer-mapping logic when a live device is available.
pub struct GpuReadback;

impl GpuReadback {
    /// Download a full frame from GPU-accessible memory to a CPU `Vec<u8>`.
    ///
    /// On the CPU stub backend this is an identity copy of `gpu_data`.
    ///
    /// # Arguments
    ///
    /// * `width`    – Frame width in pixels.
    /// * `height`   – Frame height in pixels.
    /// * `gpu_data` – GPU-accessible source buffer (RGBA packed, row-major).
    ///
    /// # Returns
    ///
    /// A `Vec<u8>` containing a copy of the readback data.
    #[must_use]
    pub fn download(width: u32, height: u32, gpu_data: &[u8]) -> Vec<u8> {
        let expected = (width as usize) * (height as usize) * 4;
        // Clamp to the actual data length in case the caller passes a
        // smaller slice (e.g. partial rows during streaming readback).
        let len = expected.min(gpu_data.len());
        gpu_data[..len].to_vec()
    }

    /// Download a sub-region of a frame.
    ///
    /// * `src_width`  – Width of the full source frame in pixels.
    /// * `x`, `y`    – Top-left corner of the sub-region.
    /// * `w`, `h`    – Width/height of the sub-region in pixels.
    /// * `gpu_data`  – Full source frame data (RGBA, row-major).
    ///
    /// Returns an empty `Vec` if the region is fully out of bounds.
    #[must_use]
    pub fn download_region(
        src_width: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        gpu_data: &[u8],
    ) -> Vec<u8> {
        if w == 0 || h == 0 {
            return Vec::new();
        }
        let stride = (src_width as usize) * 4;
        let mut out = Vec::with_capacity((w as usize) * (h as usize) * 4);
        for row in 0..h {
            let src_y = (y + row) as usize;
            let src_x = x as usize;
            let row_start = src_y * stride + src_x * 4;
            let row_end = row_start + (w as usize) * 4;
            if row_end > gpu_data.len() {
                break;
            }
            out.extend_from_slice(&gpu_data[row_start..row_end]);
        }
        out
    }

    /// Compute the expected byte length for a frame of `width × height` pixels
    /// in RGBA format.
    #[must_use]
    pub fn expected_len(width: u32, height: u32) -> usize {
        (width as usize) * (height as usize) * 4
    }

    /// Verify that `gpu_data` has exactly the expected length for a
    /// `width × height` RGBA frame.
    #[must_use]
    pub fn validate_size(width: u32, height: u32, gpu_data: &[u8]) -> bool {
        gpu_data.len() == Self::expected_len(width, height)
    }

    /// Split RGBA packed data into separate R, G, B, A channel planes.
    ///
    /// Returns `(r, g, b, a)` each of length `width * height`.
    #[must_use]
    pub fn split_channels(gpu_data: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let pixels = gpu_data.len() / 4;
        let mut r = Vec::with_capacity(pixels);
        let mut g = Vec::with_capacity(pixels);
        let mut b = Vec::with_capacity(pixels);
        let mut a = Vec::with_capacity(pixels);
        for chunk in gpu_data.chunks_exact(4) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
            a.push(chunk[3]);
        }
        (r, g, b, a)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_identity() {
        let data: Vec<u8> = (0..64).collect();
        let result = GpuReadback::download(4, 4, &data);
        assert_eq!(result, data);
    }

    #[test]
    fn test_download_truncates_to_expected_len() {
        // 2×2 RGBA = 16 bytes; supply 20 bytes → only 16 returned
        let data = vec![0xAAu8; 20];
        let result = GpuReadback::download(2, 2, &data);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_download_short_slice_returns_available_bytes() {
        // Supply only 8 bytes for a 2×2 frame (expected 16)
        let data = vec![0xBBu8; 8];
        let result = GpuReadback::download(2, 2, &data);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_download_region_basic() {
        // 4×4 RGBA frame, each pixel is its row index repeated 4 times.
        let mut frame = vec![0u8; 4 * 4 * 4];
        for row in 0..4usize {
            for col in 0..4usize {
                let idx = (row * 4 + col) * 4;
                frame[idx] = row as u8;
                frame[idx + 1] = row as u8;
                frame[idx + 2] = row as u8;
                frame[idx + 3] = 0xFF;
            }
        }
        // Download row 1, columns 0-1 (2×1 region)
        let region = GpuReadback::download_region(4, 0, 1, 2, 1, &frame);
        assert_eq!(region.len(), 8); // 2 pixels × 4 channels
        assert_eq!(region[0], 1u8); // row 1
    }

    #[test]
    fn test_download_region_zero_dimensions() {
        let frame = vec![0u8; 64];
        assert!(GpuReadback::download_region(4, 0, 0, 0, 4, &frame).is_empty());
        assert!(GpuReadback::download_region(4, 0, 0, 4, 0, &frame).is_empty());
    }

    #[test]
    fn test_expected_len() {
        assert_eq!(GpuReadback::expected_len(1920, 1080), 1920 * 1080 * 4);
    }

    #[test]
    fn test_validate_size_correct() {
        let data = vec![0u8; 4 * 4 * 4];
        assert!(GpuReadback::validate_size(4, 4, &data));
    }

    #[test]
    fn test_validate_size_wrong() {
        let data = vec![0u8; 10];
        assert!(!GpuReadback::validate_size(4, 4, &data));
    }

    #[test]
    fn test_split_channels_correctness() {
        // Single pixel: R=1, G=2, B=3, A=255
        let data = vec![1u8, 2, 3, 255];
        let (r, g, b, a) = GpuReadback::split_channels(&data);
        assert_eq!(r, [1]);
        assert_eq!(g, [2]);
        assert_eq!(b, [3]);
        assert_eq!(a, [255]);
    }

    #[test]
    fn test_split_channels_multiple_pixels() {
        let mut data = Vec::new();
        for i in 0u8..4 {
            data.extend_from_slice(&[i, i + 10, i + 20, 0xFF]);
        }
        let (r, g, b, a) = GpuReadback::split_channels(&data);
        assert_eq!(r, [0, 1, 2, 3]);
        assert_eq!(g, [10, 11, 12, 13]);
        assert_eq!(b, [20, 21, 22, 23]);
        assert_eq!(a, [0xFF, 0xFF, 0xFF, 0xFF]);
    }
}
