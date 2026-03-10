#![allow(dead_code)]
//! Indirect dispatch support for GPU compute kernels.
//!
//! Indirect dispatch allows the GPU to determine workgroup counts from
//! a buffer rather than from CPU-side values. This is essential for
//! data-dependent workloads where the number of items is computed by
//! a previous GPU pass (e.g., compaction, prefix-sum driven dispatch).

use std::fmt;

/// Arguments for an indirect compute dispatch, matching the layout
/// expected by `wgpu::RenderPass::dispatch_workgroups_indirect`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct IndirectDispatchArgs {
    /// Number of workgroups in the X dimension.
    pub x: u32,
    /// Number of workgroups in the Y dimension.
    pub y: u32,
    /// Number of workgroups in the Z dimension.
    pub z: u32,
}

impl IndirectDispatchArgs {
    /// Create new dispatch arguments.
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Create a 1D dispatch with the given workgroup count.
    pub fn one_d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Create a 2D dispatch with the given workgroup counts.
    pub fn two_d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Total number of workgroups dispatched.
    pub fn total_workgroups(&self) -> u64 {
        u64::from(self.x) * u64::from(self.y) * u64::from(self.z)
    }

    /// Serialize to a byte array (12 bytes, little-endian).
    pub fn to_bytes(&self) -> [u8; 12] {
        let mut buf = [0u8; 12];
        buf[0..4].copy_from_slice(&self.x.to_le_bytes());
        buf[4..8].copy_from_slice(&self.y.to_le_bytes());
        buf[8..12].copy_from_slice(&self.z.to_le_bytes());
        buf
    }

    /// Deserialize from a byte slice (must be at least 12 bytes).
    ///
    /// Returns `None` if the slice is too short.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 12 {
            return None;
        }
        let x = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let y = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let z = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        Some(Self { x, y, z })
    }

    /// Check whether all dimensions are non-zero.
    pub fn is_valid(&self) -> bool {
        self.x > 0 && self.y > 0 && self.z > 0
    }
}

impl Default for IndirectDispatchArgs {
    fn default() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }
}

impl fmt::Display for IndirectDispatchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dispatch({}x{}x{})", self.x, self.y, self.z)
    }
}

/// Strategy for computing indirect dispatch arguments from an element count
/// and a workgroup size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchStrategy {
    /// Simple 1D linear dispatch: ceil(elements / workgroup_size).
    Linear,
    /// 2D tiled dispatch for image-like workloads.
    Tiled2D {
        /// Tile width in workgroup units.
        tile_w: u32,
        /// Tile height in workgroup units.
        tile_h: u32,
    },
    /// 3D volumetric dispatch.
    Volumetric {
        /// Volume width in workgroup units.
        vol_w: u32,
        /// Volume height in workgroup units.
        vol_h: u32,
        /// Volume depth in workgroup units.
        vol_d: u32,
    },
}

/// Compute dispatch arguments from an element count and strategy.
#[allow(clippy::cast_precision_loss)]
pub fn compute_dispatch(element_count: u32, workgroup_size: u32, strategy: DispatchStrategy) -> IndirectDispatchArgs {
    match strategy {
        DispatchStrategy::Linear => {
            let groups = (element_count + workgroup_size - 1) / workgroup_size;
            IndirectDispatchArgs::one_d(groups)
        }
        DispatchStrategy::Tiled2D { tile_w, tile_h } => {
            let gx = (tile_w + workgroup_size - 1) / workgroup_size;
            let gy = (tile_h + workgroup_size - 1) / workgroup_size;
            IndirectDispatchArgs::two_d(gx, gy)
        }
        DispatchStrategy::Volumetric { vol_w, vol_h, vol_d } => {
            let gx = (vol_w + workgroup_size - 1) / workgroup_size;
            let gy = (vol_h + workgroup_size - 1) / workgroup_size;
            let gz = (vol_d + workgroup_size - 1) / workgroup_size;
            IndirectDispatchArgs::new(gx, gy, gz)
        }
    }
}

/// A buffer that holds indirect dispatch arguments.
///
/// This represents the GPU-side buffer that would be used for
/// `dispatch_workgroups_indirect`. In CPU simulation mode it
/// stores the arguments in-memory for testing.
pub struct IndirectBuffer {
    /// The current dispatch arguments.
    args: IndirectDispatchArgs,
    /// Label for debugging.
    label: String,
    /// Generation counter for change tracking.
    generation: u64,
}

impl IndirectBuffer {
    /// Create a new indirect buffer with default (1,1,1) dispatch.
    pub fn new(label: &str) -> Self {
        Self {
            args: IndirectDispatchArgs::default(),
            label: label.to_string(),
            generation: 0,
        }
    }

    /// Create an indirect buffer with specific initial arguments.
    pub fn with_args(label: &str, args: IndirectDispatchArgs) -> Self {
        Self {
            args,
            label: label.to_string(),
            generation: 0,
        }
    }

    /// Update the dispatch arguments.
    pub fn update(&mut self, args: IndirectDispatchArgs) {
        self.args = args;
        self.generation += 1;
    }

    /// Get the current dispatch arguments.
    pub fn args(&self) -> IndirectDispatchArgs {
        self.args
    }

    /// Get the buffer label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Get the generation counter.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Get the buffer size in bytes (always 12 bytes for dispatch args).
    pub fn size_bytes(&self) -> usize {
        12
    }

    /// Serialize the current arguments to bytes.
    pub fn to_bytes(&self) -> [u8; 12] {
        self.args.to_bytes()
    }
}

/// Validates that dispatch arguments do not exceed device limits.
pub fn validate_dispatch_limits(args: &IndirectDispatchArgs, max_per_dimension: u32) -> Result<(), String> {
    if args.x > max_per_dimension {
        return Err(format!("X workgroup count {} exceeds limit {}", args.x, max_per_dimension));
    }
    if args.y > max_per_dimension {
        return Err(format!("Y workgroup count {} exceeds limit {}", args.y, max_per_dimension));
    }
    if args.z > max_per_dimension {
        return Err(format!("Z workgroup count {} exceeds limit {}", args.z, max_per_dimension));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_args_new() {
        let args = IndirectDispatchArgs::new(4, 8, 2);
        assert_eq!(args.x, 4);
        assert_eq!(args.y, 8);
        assert_eq!(args.z, 2);
    }

    #[test]
    fn test_dispatch_args_one_d() {
        let args = IndirectDispatchArgs::one_d(16);
        assert_eq!(args.x, 16);
        assert_eq!(args.y, 1);
        assert_eq!(args.z, 1);
    }

    #[test]
    fn test_total_workgroups() {
        let args = IndirectDispatchArgs::new(4, 8, 2);
        assert_eq!(args.total_workgroups(), 64);
    }

    #[test]
    fn test_to_from_bytes_roundtrip() {
        let original = IndirectDispatchArgs::new(123, 456, 789);
        let bytes = original.to_bytes();
        let restored = IndirectDispatchArgs::from_bytes(&bytes).expect("deserialization from bytes should succeed");
        assert_eq!(original, restored);
    }

    #[test]
    fn test_from_bytes_too_short() {
        assert!(IndirectDispatchArgs::from_bytes(&[0u8; 8]).is_none());
    }

    #[test]
    fn test_is_valid() {
        assert!(IndirectDispatchArgs::new(1, 1, 1).is_valid());
        assert!(!IndirectDispatchArgs::new(0, 1, 1).is_valid());
        assert!(!IndirectDispatchArgs::new(1, 0, 1).is_valid());
        assert!(!IndirectDispatchArgs::new(1, 1, 0).is_valid());
    }

    #[test]
    fn test_display() {
        let args = IndirectDispatchArgs::new(4, 8, 2);
        assert_eq!(format!("{args}"), "Dispatch(4x8x2)");
    }

    #[test]
    fn test_compute_dispatch_linear() {
        let args = compute_dispatch(1000, 64, DispatchStrategy::Linear);
        // ceil(1000/64) = 16
        assert_eq!(args.x, 16);
        assert_eq!(args.y, 1);
        assert_eq!(args.z, 1);
    }

    #[test]
    fn test_compute_dispatch_tiled() {
        let args = compute_dispatch(0, 16, DispatchStrategy::Tiled2D { tile_w: 1920, tile_h: 1080 });
        assert_eq!(args.x, 120); // ceil(1920/16)
        assert_eq!(args.y, 68);  // ceil(1080/16)
        assert_eq!(args.z, 1);
    }

    #[test]
    fn test_compute_dispatch_volumetric() {
        let args = compute_dispatch(0, 8, DispatchStrategy::Volumetric { vol_w: 64, vol_h: 64, vol_d: 32 });
        assert_eq!(args.x, 8);
        assert_eq!(args.y, 8);
        assert_eq!(args.z, 4);
    }

    #[test]
    fn test_indirect_buffer_new() {
        let buf = IndirectBuffer::new("test_buf");
        assert_eq!(buf.label(), "test_buf");
        assert_eq!(buf.args(), IndirectDispatchArgs::default());
        assert_eq!(buf.generation(), 0);
        assert_eq!(buf.size_bytes(), 12);
    }

    #[test]
    fn test_indirect_buffer_update() {
        let mut buf = IndirectBuffer::new("buf");
        buf.update(IndirectDispatchArgs::new(10, 20, 30));
        assert_eq!(buf.args().x, 10);
        assert_eq!(buf.generation(), 1);
        buf.update(IndirectDispatchArgs::one_d(5));
        assert_eq!(buf.generation(), 2);
    }

    #[test]
    fn test_validate_dispatch_limits_ok() {
        let args = IndirectDispatchArgs::new(100, 100, 100);
        assert!(validate_dispatch_limits(&args, 65535).is_ok());
    }

    #[test]
    fn test_validate_dispatch_limits_exceeded() {
        let args = IndirectDispatchArgs::new(70000, 1, 1);
        assert!(validate_dispatch_limits(&args, 65535).is_err());
    }

    #[test]
    fn test_default_dispatch_args() {
        let args = IndirectDispatchArgs::default();
        assert_eq!(args.x, 1);
        assert_eq!(args.y, 1);
        assert_eq!(args.z, 1);
        assert!(args.is_valid());
    }
}
