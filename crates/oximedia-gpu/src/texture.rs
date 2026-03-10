//! GPU texture management
//!
//! Describes texture formats, computes memory requirements, and provides a
//! simple pooled allocator for GPU textures.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Supported GPU texture formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    /// 8-bit RGBA (4 bytes / pixel)
    Rgba8,
    /// 16-bit half-float RGBA (8 bytes / pixel)
    Rgba16f,
    /// 10-bit RGB + 2-bit alpha packed (4 bytes / pixel)
    Rgb10A2,
    /// Single 8-bit red channel (1 byte / pixel)
    R8,
    /// Dual 8-bit RG channels (2 bytes / pixel)
    Rg8,
    /// Planar YUV 4:2:0 (1.5 bytes / pixel)
    Yuv420,
    /// Semi-planar NV12 YUV 4:2:0 (1.5 bytes / pixel)
    Nv12,
}

impl TextureFormat {
    /// Average bytes per pixel (may be fractional for planar formats)
    #[must_use]
    pub fn bytes_per_pixel(&self) -> f32 {
        match self {
            Self::Rgba8 | Self::Rgb10A2 => 4.0,
            Self::Rgba16f => 8.0,
            Self::R8 => 1.0,
            Self::Rg8 => 2.0,
            Self::Yuv420 | Self::Nv12 => 1.5,
        }
    }

    /// Whether this format is a YUV variant
    #[must_use]
    pub fn is_yuv(&self) -> bool {
        matches!(self, Self::Yuv420 | Self::Nv12)
    }

    /// Logical channel count (Y, Cb, Cr are each counted separately)
    #[must_use]
    pub fn channel_count(&self) -> u8 {
        match self {
            Self::R8 => 1,
            Self::Rg8 => 2,
            Self::Rgba8 | Self::Rgba16f | Self::Rgb10A2 => 4,
            Self::Yuv420 | Self::Nv12 => 3,
        }
    }
}

/// Descriptor for a single GPU texture
#[derive(Debug, Clone)]
pub struct TextureDescriptor {
    /// Width in texels
    pub width: u32,
    /// Height in texels
    pub height: u32,
    /// Texel format
    pub format: TextureFormat,
    /// Number of mip levels (1 = base level only)
    pub mip_levels: u8,
    /// Number of array layers (1 = single texture)
    pub array_layers: u16,
}

impl TextureDescriptor {
    /// Create a simple 2-D texture with no mip chain and a single layer
    #[must_use]
    pub fn new(width: u32, height: u32, format: TextureFormat) -> Self {
        Self {
            width,
            height,
            format,
            mip_levels: 1,
            array_layers: 1,
        }
    }

    /// Total size in bytes for the full mip chain and all array layers
    ///
    /// Each successive mip level has half the dimensions of the previous.
    /// Minimum mip size is 1×1.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let bpp = self.format.bytes_per_pixel();
        let layers = self.array_layers as usize;
        let mut total_pixels: f64 = 0.0;
        let (mut w, mut h) = (f64::from(self.width), f64::from(self.height));
        for _ in 0..self.mip_levels {
            total_pixels += w * h;
            w = (w / 2.0).max(1.0);
            h = (h / 2.0).max(1.0);
        }
        (total_pixels * f64::from(bpp) * layers as f64) as usize
    }

    /// Total number of texels in the base mip level (ignoring arrays / mips)
    #[must_use]
    pub fn total_pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// A pool of GPU textures backed by a fixed memory budget
pub struct TexturePool {
    /// All allocated descriptors (index acts as texture handle)
    descriptors: Vec<Option<TextureDescriptor>>,
    /// Currently allocated bytes
    allocated_bytes: usize,
    /// Maximum bytes the pool may use
    max_bytes: usize,
}

impl TexturePool {
    /// Create a pool with a budget of `max_gb` gigabytes
    #[must_use]
    pub fn new(max_gb: f64) -> Self {
        Self {
            descriptors: Vec::new(),
            allocated_bytes: 0,
            max_bytes: (max_gb * 1024.0 * 1024.0 * 1024.0) as usize,
        }
    }

    /// Allocate a texture in the pool
    ///
    /// Returns `Some(handle)` on success, or `None` if the budget is
    /// exceeded.
    pub fn allocate(&mut self, desc: TextureDescriptor) -> Option<usize> {
        let bytes = desc.size_bytes();
        if self.allocated_bytes + bytes > self.max_bytes {
            return None;
        }
        // Reuse a freed slot if possible
        if let Some(idx) = self
            .descriptors
            .iter()
            .position(std::option::Option::is_none)
        {
            self.descriptors[idx] = Some(desc);
            self.allocated_bytes += bytes;
            return Some(idx);
        }
        let idx = self.descriptors.len();
        self.descriptors.push(Some(desc));
        self.allocated_bytes += bytes;
        Some(idx)
    }

    /// Free a texture by handle
    pub fn free(&mut self, id: usize) {
        if let Some(slot) = self.descriptors.get_mut(id) {
            if let Some(desc) = slot.take() {
                let bytes = desc.size_bytes();
                self.allocated_bytes = self.allocated_bytes.saturating_sub(bytes);
            }
        }
    }

    /// Utilisation in [0.0, 1.0]
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.max_bytes == 0 {
            return 0.0;
        }
        self.allocated_bytes as f64 / self.max_bytes as f64
    }

    /// Number of live (allocated) textures
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.descriptors.iter().filter(|s| s.is_some()).count()
    }

    /// Currently allocated bytes
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Pool capacity in bytes
    #[must_use]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }
}

// ============================================================
// Unit tests
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba8_bytes_per_pixel() {
        assert!((TextureFormat::Rgba8.bytes_per_pixel() - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_yuv_formats_are_yuv() {
        assert!(TextureFormat::Yuv420.is_yuv());
        assert!(TextureFormat::Nv12.is_yuv());
        assert!(!TextureFormat::Rgba8.is_yuv());
    }

    #[test]
    fn test_channel_counts() {
        assert_eq!(TextureFormat::R8.channel_count(), 1);
        assert_eq!(TextureFormat::Rg8.channel_count(), 2);
        assert_eq!(TextureFormat::Rgba8.channel_count(), 4);
        assert_eq!(TextureFormat::Yuv420.channel_count(), 3);
    }

    #[test]
    fn test_descriptor_new_defaults() {
        let d = TextureDescriptor::new(1920, 1080, TextureFormat::Rgba8);
        assert_eq!(d.mip_levels, 1);
        assert_eq!(d.array_layers, 1);
    }

    #[test]
    fn test_descriptor_total_pixels() {
        let d = TextureDescriptor::new(100, 200, TextureFormat::R8);
        assert_eq!(d.total_pixels(), 20_000);
    }

    #[test]
    fn test_descriptor_size_bytes_rgba8() {
        let d = TextureDescriptor::new(4, 4, TextureFormat::Rgba8);
        // 4*4 = 16 pixels × 4 bytes = 64 bytes
        assert_eq!(d.size_bytes(), 64);
    }

    #[test]
    fn test_descriptor_size_bytes_with_mips() {
        // 4×4 + 2×2 + 1×1 = 16 + 4 + 1 = 21 pixels × 4 bytes = 84
        let mut d = TextureDescriptor::new(4, 4, TextureFormat::Rgba8);
        d.mip_levels = 3;
        assert_eq!(d.size_bytes(), 84);
    }

    #[test]
    fn test_pool_basic_allocation() {
        let mut pool = TexturePool::new(1.0);
        let desc = TextureDescriptor::new(64, 64, TextureFormat::Rgba8);
        let handle = pool.allocate(desc);
        assert!(handle.is_some());
        assert_eq!(pool.live_count(), 1);
    }

    #[test]
    fn test_pool_free_reduces_bytes() {
        let mut pool = TexturePool::new(1.0);
        let desc = TextureDescriptor::new(4, 4, TextureFormat::Rgba8);
        let handle = pool.allocate(desc).expect("allocation should succeed");
        let before = pool.allocated_bytes();
        pool.free(handle);
        assert!(pool.allocated_bytes() < before);
        assert_eq!(pool.live_count(), 0);
    }

    #[test]
    fn test_pool_reuses_freed_slot() {
        let mut pool = TexturePool::new(1.0);
        let d1 = TextureDescriptor::new(4, 4, TextureFormat::R8);
        let h1 = pool.allocate(d1).expect("allocation should succeed");
        pool.free(h1);
        let d2 = TextureDescriptor::new(4, 4, TextureFormat::R8);
        let h2 = pool.allocate(d2).expect("allocation should succeed");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_pool_budget_exceeded_returns_none() {
        // Pool with a tiny 1-byte budget
        let mut pool = TexturePool::new(0.0);
        pool.max_bytes = 1;
        let desc = TextureDescriptor::new(1920, 1080, TextureFormat::Rgba8);
        assert!(pool.allocate(desc).is_none());
    }

    #[test]
    fn test_pool_utilization_after_alloc() {
        let mut pool = TexturePool::new(0.0);
        // Set a precise budget
        let desc = TextureDescriptor::new(4, 4, TextureFormat::Rgba8); // 64 bytes
        pool.max_bytes = 128;
        pool.allocate(desc).expect("allocation should succeed");
        let util = pool.utilization();
        assert!((util - 0.5).abs() < 1e-6, "expected 0.5, got {util}");
    }
}
