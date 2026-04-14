#![allow(dead_code)]
//! Mipmap generation utilities for GPU textures.
//!
//! This module computes mip-chain metadata and performs CPU-side
//! mipmap generation using box-filter downsampling. It can be used
//! as a reference implementation or for CPU fallback paths when
//! GPU mipmap generation is unavailable.

use std::fmt;

/// Describes a single mip level within a mip chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MipLevel {
    /// Mip level index (0 = full resolution).
    pub level: u32,
    /// Width of this mip level in pixels.
    pub width: u32,
    /// Height of this mip level in pixels.
    pub height: u32,
    /// Byte offset into the mip chain buffer.
    pub offset: usize,
    /// Size in bytes of this mip level.
    pub size: usize,
}

impl MipLevel {
    /// Total number of pixels at this level.
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

impl fmt::Display for MipLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mip[{}]: {}x{} (offset={}, size={})",
            self.level, self.width, self.height, self.offset, self.size
        )
    }
}

/// Describes the full mip chain for a texture.
#[derive(Debug, Clone)]
pub struct MipChain {
    /// Base width of the texture (level 0).
    pub base_width: u32,
    /// Base height of the texture (level 0).
    pub base_height: u32,
    /// Number of channels per pixel.
    pub channels: u32,
    /// Individual mip level descriptors.
    pub levels: Vec<MipLevel>,
}

impl MipChain {
    /// Compute the full mip chain for a texture with given dimensions.
    ///
    /// Generates levels down to 1x1 unless `max_levels` limits the count.
    pub fn compute(
        base_width: u32,
        base_height: u32,
        channels: u32,
        max_levels: Option<u32>,
    ) -> Self {
        let full_count = compute_mip_count(base_width, base_height);
        let level_count = max_levels.map_or(full_count, |m| m.min(full_count));

        let mut levels = Vec::with_capacity(level_count as usize);
        let mut w = base_width;
        let mut h = base_height;
        let mut offset = 0usize;

        for i in 0..level_count {
            let size = (w as usize) * (h as usize) * (channels as usize);
            levels.push(MipLevel {
                level: i,
                width: w,
                height: h,
                offset,
                size,
            });
            offset += size;
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }

        Self {
            base_width,
            base_height,
            channels,
            levels,
        }
    }

    /// Total number of mip levels in the chain.
    pub fn level_count(&self) -> u32 {
        self.levels.len() as u32
    }

    /// Total size in bytes for the entire mip chain.
    pub fn total_size(&self) -> usize {
        self.levels.iter().map(|l| l.size).sum()
    }

    /// Get a specific mip level by index.
    pub fn level(&self, index: u32) -> Option<&MipLevel> {
        self.levels.get(index as usize)
    }

    /// Check whether the chain includes only a single level (no mips).
    pub fn is_single_level(&self) -> bool {
        self.levels.len() <= 1
    }
}

/// Compute the maximum number of mip levels for given dimensions.
///
/// This is `floor(log2(max(width, height))) + 1`.
#[allow(clippy::cast_precision_loss)]
pub fn compute_mip_count(width: u32, height: u32) -> u32 {
    if width == 0 || height == 0 {
        return 0;
    }
    let max_dim = width.max(height);
    (max_dim as f64).log2().floor() as u32 + 1
}

/// Compute the dimension of a specific mip level.
pub fn mip_dimension(base: u32, level: u32) -> u32 {
    (base >> level).max(1)
}

/// Downsample a single-channel u8 image by 2x using a box filter (CPU path).
///
/// The source dimensions must be at least 2x2.
pub fn downsample_box_u8(src: &[u8], src_width: u32, src_height: u32, channels: u32) -> Vec<u8> {
    let dst_width = (src_width / 2).max(1);
    let dst_height = (src_height / 2).max(1);
    let ch = channels as usize;
    let mut dst = vec![0u8; dst_width as usize * dst_height as usize * ch];

    let sw = src_width as usize;

    for dy in 0..dst_height as usize {
        for dx in 0..dst_width as usize {
            let sx = dx * 2;
            let sy = dy * 2;

            // Clamp secondary samples
            let sx1 = (sx + 1).min(src_width as usize - 1);
            let sy1 = (sy + 1).min(src_height as usize - 1);

            for c in 0..ch {
                let s00 = u16::from(src[(sy * sw + sx) * ch + c]);
                let s10 = u16::from(src[(sy * sw + sx1) * ch + c]);
                let s01 = u16::from(src[(sy1 * sw + sx) * ch + c]);
                let s11 = u16::from(src[(sy1 * sw + sx1) * ch + c]);
                let avg = ((s00 + s10 + s01 + s11 + 2) / 4) as u8;
                dst[(dy * dst_width as usize + dx) * ch + c] = avg;
            }
        }
    }

    dst
}

/// Generate the full mip chain data from a base level image.
///
/// Returns a flat buffer containing all mip levels concatenated,
/// along with the mip chain descriptor.
pub fn generate_mip_chain_u8(
    base_data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
) -> (Vec<u8>, MipChain) {
    let chain = MipChain::compute(width, height, channels, None);
    let mut output = Vec::with_capacity(chain.total_size());

    // Level 0 is the original image
    output.extend_from_slice(base_data);

    let mut prev_data = base_data.to_vec();
    let mut prev_w = width;
    let mut prev_h = height;

    for level in chain.levels.iter().skip(1) {
        let down = downsample_box_u8(&prev_data, prev_w, prev_h, channels);
        output.extend_from_slice(&down);
        prev_w = level.width;
        prev_h = level.height;
        prev_data = down;
    }

    (output, chain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mip_count_square() {
        assert_eq!(compute_mip_count(256, 256), 9); // log2(256)+1 = 9
    }

    #[test]
    fn test_compute_mip_count_nonsquare() {
        assert_eq!(compute_mip_count(1920, 1080), 11); // log2(1920)+1 = 11
    }

    #[test]
    fn test_compute_mip_count_one() {
        assert_eq!(compute_mip_count(1, 1), 1);
    }

    #[test]
    fn test_compute_mip_count_zero() {
        assert_eq!(compute_mip_count(0, 100), 0);
        assert_eq!(compute_mip_count(100, 0), 0);
    }

    #[test]
    fn test_mip_dimension() {
        assert_eq!(mip_dimension(256, 0), 256);
        assert_eq!(mip_dimension(256, 1), 128);
        assert_eq!(mip_dimension(256, 8), 1);
        assert_eq!(mip_dimension(256, 20), 1); // Clamped to 1
    }

    #[test]
    fn test_mip_chain_compute() {
        let chain = MipChain::compute(64, 64, 4, None);
        assert_eq!(chain.level_count(), 7); // log2(64)+1 = 7
        assert_eq!(chain.levels[0].width, 64);
        assert_eq!(chain.levels[0].height, 64);
        assert_eq!(chain.levels[1].width, 32);
        assert_eq!(chain.levels[6].width, 1);
    }

    #[test]
    fn test_mip_chain_max_levels() {
        let chain = MipChain::compute(256, 256, 4, Some(3));
        assert_eq!(chain.level_count(), 3);
        assert_eq!(chain.levels[2].width, 64);
    }

    #[test]
    fn test_mip_chain_total_size() {
        let chain = MipChain::compute(4, 4, 1, None);
        // 4x4=16 + 2x2=4 + 1x1=1 = 21
        assert_eq!(chain.total_size(), 21);
    }

    #[test]
    fn test_mip_chain_offsets() {
        let chain = MipChain::compute(8, 8, 1, None);
        assert_eq!(chain.levels[0].offset, 0);
        assert_eq!(chain.levels[1].offset, 64); // 8*8*1
        assert_eq!(chain.levels[2].offset, 80); // 64 + 4*4*1
    }

    #[test]
    fn test_mip_level_display() {
        let level = MipLevel {
            level: 2,
            width: 64,
            height: 32,
            offset: 1000,
            size: 2048,
        };
        assert_eq!(format!("{level}"), "Mip[2]: 64x32 (offset=1000, size=2048)");
    }

    #[test]
    fn test_downsample_box_u8_simple() {
        // 4x4 image, 1 channel, all 200
        let src = vec![200u8; 16];
        let dst = downsample_box_u8(&src, 4, 4, 1);
        assert_eq!(dst.len(), 4); // 2x2
        for &v in &dst {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn test_downsample_box_u8_gradient() {
        // 2x2 image with values [0, 100, 200, 56]
        let src = vec![0u8, 100, 200, 56];
        let dst = downsample_box_u8(&src, 2, 2, 1);
        assert_eq!(dst.len(), 1);
        // Average: (0+100+200+56+2)/4 = 89 (integer)
        assert_eq!(dst[0], 89);
    }

    #[test]
    fn test_generate_mip_chain_u8() {
        let base = vec![128u8; 4 * 4]; // 4x4, 1 channel
        let (data, chain) = generate_mip_chain_u8(&base, 4, 4, 1);
        assert_eq!(chain.level_count(), 3);
        assert_eq!(data.len(), chain.total_size());
        // Level 0 should be the original
        assert_eq!(&data[0..16], &base[..]);
    }

    #[test]
    fn test_mip_chain_is_single_level() {
        let chain = MipChain::compute(1, 1, 4, None);
        assert!(chain.is_single_level());
        let chain2 = MipChain::compute(4, 4, 4, None);
        assert!(!chain2.is_single_level());
    }

    #[test]
    fn test_mip_level_pixel_count() {
        let level = MipLevel {
            level: 0,
            width: 100,
            height: 50,
            offset: 0,
            size: 0,
        };
        assert_eq!(level.pixel_count(), 5000);
    }
}
