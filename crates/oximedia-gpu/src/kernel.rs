//! GPU kernel management — kernel types, specs, and caching.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Type of GPU kernel, determining workgroup shape and shared-memory usage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelType {
    /// Video scaling kernel.
    VideoScale,
    /// Color space conversion kernel.
    ColorConvert,
    /// Histogram accumulation kernel.
    Histogram,
    /// Motion estimation kernel.
    MotionEstimate,
    /// Denoising kernel.
    Denoise,
    /// Sharpening kernel.
    Sharpen,
}

impl KernelType {
    /// Returns the preferred (x, y) workgroup size for this kernel type.
    #[must_use]
    pub fn workgroup_size(&self) -> (u32, u32) {
        match self {
            Self::VideoScale => (16, 16),
            Self::ColorConvert => (32, 8),
            Self::Histogram => (256, 1),
            Self::MotionEstimate => (8, 8),
            Self::Denoise => (16, 16),
            Self::Sharpen => (16, 16),
        }
    }

    /// Returns whether this kernel type requires shared memory.
    #[must_use]
    pub fn requires_shared_memory(&self) -> bool {
        match self {
            Self::Histogram | Self::MotionEstimate | Self::Denoise => true,
            Self::VideoScale | Self::ColorConvert | Self::Sharpen => false,
        }
    }
}

/// Specification of a single GPU kernel invocation.
#[derive(Debug, Clone)]
pub struct KernelSpec {
    /// The kernel type.
    pub kernel_type: KernelType,
    /// Number of input channels.
    pub input_channels: u8,
    /// Number of output channels.
    pub output_channels: u8,
    /// Image / buffer width in elements.
    pub width: u32,
    /// Image / buffer height in elements.
    pub height: u32,
}

impl KernelSpec {
    /// Creates a new `KernelSpec`.
    #[must_use]
    pub fn new(
        kernel_type: KernelType,
        input_channels: u8,
        output_channels: u8,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            kernel_type,
            input_channels,
            output_channels,
            width,
            height,
        }
    }

    /// Total number of elements processed (width × height).
    #[must_use]
    pub fn total_elements(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Rough estimate of floating-point operations for this kernel.
    ///
    /// Uses heuristic multipliers per kernel type and channel count.
    #[must_use]
    pub fn estimated_flops(&self) -> u64 {
        let elements = self.total_elements();
        let channels = u64::from(self.input_channels.max(self.output_channels));
        let per_element: u64 = match self.kernel_type {
            KernelType::VideoScale => 8,
            KernelType::ColorConvert => 12,
            KernelType::Histogram => 2,
            KernelType::MotionEstimate => 32,
            KernelType::Denoise => 64,
            KernelType::Sharpen => 16,
        };
        elements * channels * per_element
    }
}

/// A cache that stores multiple [`KernelSpec`] entries and provides lookup.
#[derive(Debug, Default)]
pub struct KernelCache {
    specs: Vec<KernelSpec>,
}

impl KernelCache {
    /// Creates an empty `KernelCache`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a kernel spec to the cache.
    pub fn add(&mut self, spec: KernelSpec) {
        self.specs.push(spec);
    }

    /// Finds the first spec whose `kernel_type` matches `kt`.
    #[must_use]
    pub fn find(&self, kt: &KernelType) -> Option<&KernelSpec> {
        self.specs.iter().find(|s| &s.kernel_type == kt)
    }

    /// Estimates total GPU memory required for all cached specs (bytes).
    ///
    /// Each element occupies 4 bytes (f32), multiplied by channel count.
    #[must_use]
    pub fn total_memory_estimate_bytes(&self) -> u64 {
        self.specs.iter().fold(0u64, |acc, s| {
            let channels = u64::from(s.input_channels) + u64::from(s.output_channels);
            acc + s.total_elements() * channels * 4
        })
    }

    /// Returns the number of specs in the cache.
    #[must_use]
    pub fn kernel_count(&self) -> usize {
        self.specs.len()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_scale_workgroup() {
        assert_eq!(KernelType::VideoScale.workgroup_size(), (16, 16));
    }

    #[test]
    fn test_histogram_workgroup() {
        assert_eq!(KernelType::Histogram.workgroup_size(), (256, 1));
    }

    #[test]
    fn test_motion_estimate_workgroup() {
        assert_eq!(KernelType::MotionEstimate.workgroup_size(), (8, 8));
    }

    #[test]
    fn test_color_convert_workgroup() {
        assert_eq!(KernelType::ColorConvert.workgroup_size(), (32, 8));
    }

    #[test]
    fn test_requires_shared_memory_histogram() {
        assert!(KernelType::Histogram.requires_shared_memory());
    }

    #[test]
    fn test_requires_shared_memory_motion() {
        assert!(KernelType::MotionEstimate.requires_shared_memory());
    }

    #[test]
    fn test_no_shared_memory_video_scale() {
        assert!(!KernelType::VideoScale.requires_shared_memory());
    }

    #[test]
    fn test_no_shared_memory_sharpen() {
        assert!(!KernelType::Sharpen.requires_shared_memory());
    }

    #[test]
    fn test_total_elements() {
        let spec = KernelSpec::new(KernelType::VideoScale, 3, 3, 1920, 1080);
        assert_eq!(spec.total_elements(), 1920 * 1080);
    }

    #[test]
    fn test_total_elements_zero_width() {
        let spec = KernelSpec::new(KernelType::Sharpen, 4, 4, 0, 1080);
        assert_eq!(spec.total_elements(), 0);
    }

    #[test]
    fn test_estimated_flops_positive() {
        let spec = KernelSpec::new(KernelType::Denoise, 4, 4, 256, 256);
        assert!(spec.estimated_flops() > 0);
    }

    #[test]
    fn test_estimated_flops_denoise_greater_than_histogram() {
        let base = KernelSpec::new(KernelType::Denoise, 4, 4, 256, 256);
        let hist = KernelSpec::new(KernelType::Histogram, 4, 4, 256, 256);
        assert!(base.estimated_flops() > hist.estimated_flops());
    }

    #[test]
    fn test_cache_add_and_count() {
        let mut cache = KernelCache::new();
        cache.add(KernelSpec::new(KernelType::VideoScale, 4, 4, 1920, 1080));
        cache.add(KernelSpec::new(KernelType::Histogram, 3, 3, 1920, 1080));
        assert_eq!(cache.kernel_count(), 2);
    }

    #[test]
    fn test_cache_find_existing() {
        let mut cache = KernelCache::new();
        cache.add(KernelSpec::new(KernelType::Sharpen, 4, 4, 640, 480));
        let found = cache.find(&KernelType::Sharpen);
        assert!(found.is_some());
        assert_eq!(found.expect("operation should succeed in test").width, 640);
    }

    #[test]
    fn test_cache_find_missing() {
        let cache = KernelCache::new();
        assert!(cache.find(&KernelType::Denoise).is_none());
    }

    #[test]
    fn test_cache_memory_estimate_nonzero() {
        let mut cache = KernelCache::new();
        cache.add(KernelSpec::new(KernelType::ColorConvert, 4, 4, 1920, 1080));
        assert!(cache.total_memory_estimate_bytes() > 0);
    }

    #[test]
    fn test_cache_empty_memory_estimate() {
        let cache = KernelCache::new();
        assert_eq!(cache.total_memory_estimate_bytes(), 0);
    }

    #[test]
    fn test_cache_find_first_match() {
        let mut cache = KernelCache::new();
        cache.add(KernelSpec::new(KernelType::VideoScale, 3, 3, 100, 100));
        cache.add(KernelSpec::new(KernelType::VideoScale, 4, 4, 200, 200));
        // Must return the first inserted spec.
        let found = cache
            .find(&KernelType::VideoScale)
            .expect("find should return a result");
        assert_eq!(found.width, 100);
    }

    #[test]
    fn test_kernel_spec_new() {
        let spec = KernelSpec::new(KernelType::MotionEstimate, 1, 2, 320, 240);
        assert_eq!(spec.input_channels, 1);
        assert_eq!(spec.output_channels, 2);
        assert_eq!(spec.height, 240);
    }
}
