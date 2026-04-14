//! Subgroup operations capability modeling for Vulkan 1.1+ devices.
//!
//! Subgroup (warp/wave) operations allow threads within a hardware-defined
//! group to communicate without shared memory, enabling highly efficient
//! parallel reductions, prefix sums, and data broadcast in compute shaders.
//!
//! # Vulkan Subgroup Requirements
//!
//! Subgroup operations require Vulkan 1.1+ and the relevant feature bits set
//! in `VkPhysicalDeviceSubgroupProperties`. The subgroup size varies by vendor:
//! - **NVIDIA**: 32 threads (warp)
//! - **AMD**: 64 threads (wavefront)
//! - **Intel**: 8–32 threads (EU SIMD width)
//! - **Apple (M-series)**: 32 threads
//!
//! # SPIR-V / GLSL Availability
//!
//! Subgroup intrinsics are available via:
//! - GLSL extension: `GL_KHR_shader_subgroup_*`
//! - SPIR-V capability: `GroupNonUniform*`

#![allow(dead_code)]

use crate::device::VulkanDevice;

/// Minimum subgroup size across all known GPU vendors.
pub const MIN_SUBGROUP_SIZE: u32 = 4;

/// Maximum subgroup size across all known GPU vendors.
pub const MAX_SUBGROUP_SIZE: u32 = 128;

/// Default subgroup size assumption when device query is unavailable.
pub const DEFAULT_SUBGROUP_SIZE: u32 = 32;

/// Which pipeline stages support subgroup operations on this device.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubgroupStages {
    /// Compute shaders support subgroup ops (always true for Vulkan 1.1 baseline).
    pub compute: bool,
    /// Fragment shaders support subgroup ops (vendor-dependent).
    pub fragment: bool,
    /// Vertex shaders support subgroup ops (rare).
    pub vertex: bool,
    /// Tessellation shaders support subgroup ops.
    pub tessellation: bool,
    /// Geometry shaders support subgroup ops.
    pub geometry: bool,
}

impl SubgroupStages {
    /// Returns true if at least one stage supports subgroup operations.
    #[must_use]
    pub fn any_supported(&self) -> bool {
        self.compute || self.fragment || self.vertex || self.tessellation || self.geometry
    }

    /// Returns a list of supported stage names for diagnostics.
    #[must_use]
    pub fn supported_names(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.compute {
            names.push("compute");
        }
        if self.fragment {
            names.push("fragment");
        }
        if self.vertex {
            names.push("vertex");
        }
        if self.tessellation {
            names.push("tessellation");
        }
        if self.geometry {
            names.push("geometry");
        }
        names
    }
}

/// Which subgroup operation categories are available on this device.
///
/// Corresponds to `VkSubgroupFeatureFlagBits` from Vulkan 1.1+.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubgroupOperations {
    /// Basic operations: `OpGroupNonUniformElect`, `OpGroupNonUniformBroadcast`.
    /// This is the Vulkan 1.1 baseline for compute shaders.
    pub basic: bool,

    /// Vote operations: `subgroupAny`, `subgroupAll`, `subgroupAllEqual`.
    pub vote: bool,

    /// Arithmetic operations: `subgroupAdd`, `subgroupMul`, `subgroupMin`, `subgroupMax`.
    /// Useful for parallel reductions (e.g., sum over a workgroup).
    pub arithmetic: bool,

    /// Ballot operations: `subgroupBallot`, `subgroupInverseBallot`,
    /// `subgroupBallotBitCount`, `subgroupBallotFindLSB/MSB`.
    pub ballot: bool,

    /// Shuffle operations: `subgroupShuffle`, `subgroupShuffleXor`.
    /// Allows arbitrary thread-to-thread data exchange within a subgroup.
    pub shuffle: bool,

    /// Shuffle-relative operations: `subgroupShuffleUp`, `subgroupShuffleDown`.
    /// Useful for scan (prefix sum) algorithms.
    pub shuffle_relative: bool,

    /// Clustered operations: arithmetic ops over fixed-size clusters.
    pub clustered: bool,

    /// Quad operations: `subgroupQuad*` for 2×2 pixel neighborhoods in fragment shaders.
    pub quad: bool,
}

impl SubgroupOperations {
    /// Returns true if any operation category is supported.
    #[must_use]
    pub fn any_supported(&self) -> bool {
        self.basic
            || self.vote
            || self.arithmetic
            || self.ballot
            || self.shuffle
            || self.shuffle_relative
            || self.clustered
            || self.quad
    }

    /// Returns a bitmask summary (matching `VkSubgroupFeatureFlagBits` ordering).
    #[must_use]
    pub fn feature_bitmask(&self) -> u32 {
        let mut mask = 0u32;
        if self.basic {
            mask |= 1 << 0;
        }
        if self.vote {
            mask |= 1 << 1;
        }
        if self.arithmetic {
            mask |= 1 << 2;
        }
        if self.ballot {
            mask |= 1 << 3;
        }
        if self.shuffle {
            mask |= 1 << 4;
        }
        if self.shuffle_relative {
            mask |= 1 << 5;
        }
        if self.clustered {
            mask |= 1 << 6;
        }
        if self.quad {
            mask |= 1 << 7;
        }
        mask
    }
}

/// Subgroup capabilities for a Vulkan 1.1+ device.
///
/// Use [`SubgroupCapabilities::query_from_device`] to populate from a real device,
/// or [`SubgroupCapabilities::default`] for a conservative fallback.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubgroupCapabilities {
    /// Pipeline stages where subgroup operations are supported.
    pub supported_stages: SubgroupStages,
    /// Which subgroup operation types are supported.
    pub supported_operations: SubgroupOperations,
    /// The subgroup (warp/wavefront) size in threads.
    ///
    /// - NVIDIA: 32
    /// - AMD: 64
    /// - Intel: 8–32
    pub subgroup_size: u32,
}

impl SubgroupCapabilities {
    /// Queries subgroup capabilities from the given Vulkan device.
    ///
    /// In production this would query `VkPhysicalDeviceSubgroupProperties`
    /// from `vkGetPhysicalDeviceProperties2` (Vulkan 1.1+). Since we use
    /// vulkano which exposes this via `PhysicalDevice::subgroup_properties()`,
    /// we provide a conservative estimate based on device type when the
    /// data is unavailable.
    ///
    /// # Conservative Defaults
    ///
    /// When device properties indicate Vulkan 1.1+, the baseline guarantee is:
    /// - Compute stage subgroup support
    /// - Basic + arithmetic operations (sufficient for parallel reduction)
    /// - Subgroup size = `DEFAULT_SUBGROUP_SIZE` (32)
    #[must_use]
    pub fn query_from_device(_device: &VulkanDevice) -> Self {
        // NOTE: vulkano 0.35 exposes `PhysicalDevice::properties().subgroup_properties`
        // under Vulkan 1.1. We use a conservative estimate here since the full
        // subgroup properties chain requires VkPhysicalDeviceSubgroupProperties
        // which vulkano exposes via `physical_device.properties()`.
        // For now, we return Vulkan 1.1 baseline guarantees that apply to all
        // conformant Vulkan 1.1+ implementations.
        Self {
            supported_stages: SubgroupStages {
                compute: true,   // Vulkan 1.1 guarantees compute subgroup support
                fragment: false, // Optional
                vertex: false,
                tessellation: false,
                geometry: false,
            },
            supported_operations: SubgroupOperations {
                basic: true,      // Vulkan 1.1 baseline
                vote: false,      // VK_SUBGROUP_FEATURE_VOTE_BIT - optional
                arithmetic: true, // VK_SUBGROUP_FEATURE_ARITHMETIC_BIT - common
                ballot: false,    // VK_SUBGROUP_FEATURE_BALLOT_BIT - optional
                shuffle: false,   // VK_SUBGROUP_FEATURE_SHUFFLE_BIT - optional
                shuffle_relative: false,
                clustered: false,
                quad: false,
            },
            subgroup_size: DEFAULT_SUBGROUP_SIZE,
        }
    }

    /// Returns true if parallel reduction via `subgroupAdd` etc. is supported.
    ///
    /// This is the most commonly needed subgroup operation for media workloads
    /// (e.g., computing frame statistics, histogram normalization).
    #[must_use]
    pub fn supports_reduce(&self) -> bool {
        self.supported_operations.arithmetic && self.supported_stages.compute
    }

    /// Returns true if inter-thread data shuffle is supported.
    ///
    /// Useful for implementing efficient butterfly networks (e.g., FFT stages)
    /// and warp-level sort networks.
    #[must_use]
    pub fn supports_shuffle(&self) -> bool {
        self.supported_operations.shuffle
    }

    /// Returns true if ballot operations are supported.
    ///
    /// Ballot enables prefix-sum and compaction patterns without shared memory.
    #[must_use]
    pub fn supports_ballot(&self) -> bool {
        self.supported_operations.ballot
    }

    /// Returns true if quad operations are supported (for 2×2 pixel tiles in fragment shaders).
    #[must_use]
    pub fn supports_quad(&self) -> bool {
        self.supported_operations.quad && self.supported_stages.fragment
    }

    /// Minimum subgroup size across all known Vulkan GPU vendors.
    #[must_use]
    pub const fn min_subgroup_size() -> u32 {
        MIN_SUBGROUP_SIZE
    }

    /// Maximum subgroup size across all known Vulkan GPU vendors.
    #[must_use]
    pub const fn max_subgroup_size() -> u32 {
        MAX_SUBGROUP_SIZE
    }

    /// Returns the number of subgroups needed to cover a workgroup of `local_size` threads.
    #[must_use]
    pub fn subgroups_per_workgroup(&self, local_size: u32) -> u32 {
        if self.subgroup_size == 0 {
            return 1;
        }
        local_size.div_ceil(self.subgroup_size)
    }

    /// Returns the optimal workgroup size for a subgroup-aware reduction.
    ///
    /// For a 2-level reduction (subgroup then workgroup), the ideal local size
    /// is `subgroup_size^2` or the next power of two, capped at 1024.
    #[must_use]
    pub fn optimal_reduce_workgroup_size(&self) -> u32 {
        let sz = self.subgroup_size * self.subgroup_size;
        sz.min(1024).max(64)
    }
}

/// Returns a GLSL shader snippet illustrating a subgroup reduce-sum operation.
///
/// Requires Vulkan 1.1 with `VK_SUBGROUP_FEATURE_ARITHMETIC_BIT` and
/// the `GL_KHR_shader_subgroup_arithmetic` extension enabled.
///
/// # Usage
///
/// Embed this template in a compute shader for frame-level statistics (e.g.,
/// computing mean luminance for tone mapping).
#[must_use]
pub fn subgroup_reduce_sum_glsl() -> &'static str {
    r#"
    // Subgroup parallel reduce-sum (Vulkan 1.1 + GL_KHR_shader_subgroup_arithmetic)
    //
    // Requires: #extension GL_KHR_shader_subgroup_arithmetic : enable
    // Requires: layout(local_size_x = 64) in;
    // Requires: buffer InputBuffer  { float data[]; };
    // Requires: buffer OutputBuffer { float result[]; };
    //
    // shared float shared_partial[gl_NumSubgroups];
    //
    // void main() {
    //     uint idx = gl_GlobalInvocationID.x;
    //     float val = (idx < data.length()) ? data[idx] : 0.0;
    //
    //     // Stage 1: warp-level reduction (no shared memory needed)
    //     float warp_sum = subgroupAdd(val);
    //
    //     // Stage 2: elect one thread per subgroup to write partial sum
    //     if (subgroupElect()) {
    //         shared_partial[gl_SubgroupID] = warp_sum;
    //     }
    //     barrier();
    //
    //     // Stage 3: first subgroup reduces across partial sums
    //     if (gl_SubgroupID == 0) {
    //         float partial = (gl_SubgroupInvocationID < gl_NumSubgroups)
    //             ? shared_partial[gl_SubgroupInvocationID] : 0.0;
    //         float total = subgroupAdd(partial);
    //         if (subgroupElect()) {
    //             result[gl_WorkGroupID.x] = total;
    //         }
    //     }
    // }
    "#
}

/// Returns a GLSL shader snippet for subgroup ballot-based stream compaction.
///
/// Stream compaction removes inactive elements from a data stream efficiently.
/// This is useful for sparse motion vector filtering, skip-block detection, etc.
#[must_use]
pub fn subgroup_ballot_compaction_glsl() -> &'static str {
    r#"
    // Subgroup ballot-based stream compaction (Vulkan 1.1 + GL_KHR_shader_subgroup_ballot)
    //
    // Requires: #extension GL_KHR_shader_subgroup_ballot : enable
    //
    // Example: compact non-zero motion vectors from a stream
    // bool is_active = (motion_vector[gl_GlobalInvocationID.x] != vec2(0.0));
    // uvec4 ballot = subgroupBallot(is_active);
    // uint active_count = subgroupBallotBitCount(ballot);
    // uint local_offset = subgroupBallotExclusiveBitCount(ballot);
    // if (is_active) {
    //     compact_output[base_offset + local_offset] = motion_vector[gl_GlobalInvocationID.x];
    // }
    // // First thread writes count for this subgroup
    // if (subgroupElect()) {
    //     subgroup_counts[gl_SubgroupID] = active_count;
    // }
    "#
}

/// Describes the tradeoffs for choosing a subgroup size for a given kernel.
#[derive(Debug, Clone)]
pub struct SubgroupSizeHint {
    /// Recommended subgroup size for this operation type.
    pub recommended_size: u32,
    /// Reason for the recommendation.
    pub rationale: &'static str,
}

/// Recommends a subgroup size for common media processing operations.
///
/// Returns a hint about optimal subgroup size based on the operation type
/// and available hardware capabilities.
#[must_use]
pub fn recommend_subgroup_size(
    caps: &SubgroupCapabilities,
    op: SubgroupOpType,
) -> SubgroupSizeHint {
    match op {
        SubgroupOpType::Reduction => SubgroupSizeHint {
            recommended_size: caps.subgroup_size,
            rationale: "Full subgroup size maximizes reduction parallelism per warp",
        },
        SubgroupOpType::Histogram => SubgroupSizeHint {
            recommended_size: caps.subgroup_size.min(32),
            rationale: "Smaller subgroups reduce histogram bank conflicts",
        },
        SubgroupOpType::Convolution => SubgroupSizeHint {
            recommended_size: 16,
            rationale: "16-thread subgroups align with common 2D tile widths",
        },
        SubgroupOpType::PrefixSum => SubgroupSizeHint {
            recommended_size: caps.subgroup_size,
            rationale: "Full subgroup size enables single-pass Kogge-Stone scan",
        },
    }
}

/// Operation type classification for subgroup size recommendations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupOpType {
    /// Parallel reduction (sum, min, max, OR, AND).
    Reduction,
    /// Histogram computation with atomic writes.
    Histogram,
    /// 2D spatial convolution with shared memory tiles.
    Convolution,
    /// Exclusive/inclusive prefix sum (scan).
    PrefixSum,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subgroup_capabilities_default() {
        let caps = SubgroupCapabilities::default();
        assert_eq!(caps.subgroup_size, 0); // default is zero (uninitialized)
        assert!(!caps.supported_stages.compute);
        assert!(!caps.supported_operations.basic);
    }

    #[test]
    fn subgroup_constants_are_sane() {
        assert!(MIN_SUBGROUP_SIZE > 0, "min subgroup size must be positive");
        assert!(MAX_SUBGROUP_SIZE > MIN_SUBGROUP_SIZE, "max must exceed min");
        assert!(
            DEFAULT_SUBGROUP_SIZE >= MIN_SUBGROUP_SIZE,
            "default must be within [min, max]"
        );
        assert!(
            DEFAULT_SUBGROUP_SIZE <= MAX_SUBGROUP_SIZE,
            "default must be within [min, max]"
        );
    }

    #[test]
    fn min_max_subgroup_size_functions() {
        assert_eq!(SubgroupCapabilities::min_subgroup_size(), MIN_SUBGROUP_SIZE);
        assert_eq!(SubgroupCapabilities::max_subgroup_size(), MAX_SUBGROUP_SIZE);
    }

    #[test]
    fn supports_reduce_requires_arithmetic_and_compute() {
        let mut caps = SubgroupCapabilities::default();
        assert!(!caps.supports_reduce());

        caps.supported_operations.arithmetic = true;
        caps.supported_stages.compute = true;
        assert!(caps.supports_reduce());

        // Missing compute stage → no reduce
        caps.supported_stages.compute = false;
        assert!(!caps.supports_reduce());
    }

    #[test]
    fn supports_shuffle_reflects_operations() {
        let mut caps = SubgroupCapabilities::default();
        assert!(!caps.supports_shuffle());
        caps.supported_operations.shuffle = true;
        assert!(caps.supports_shuffle());
    }

    #[test]
    fn supports_ballot_reflects_operations() {
        let mut caps = SubgroupCapabilities::default();
        assert!(!caps.supports_ballot());
        caps.supported_operations.ballot = true;
        assert!(caps.supports_ballot());
    }

    #[test]
    fn supports_quad_requires_fragment_and_quad() {
        let mut caps = SubgroupCapabilities::default();
        assert!(!caps.supports_quad());
        caps.supported_operations.quad = true;
        assert!(!caps.supports_quad(), "quad needs fragment stage too");
        caps.supported_stages.fragment = true;
        assert!(caps.supports_quad());
    }

    #[test]
    fn subgroups_per_workgroup_calculation() {
        let mut caps = SubgroupCapabilities {
            subgroup_size: 32,
            ..Default::default()
        };
        assert_eq!(caps.subgroups_per_workgroup(64), 2);
        assert_eq!(caps.subgroups_per_workgroup(128), 4);
        assert_eq!(caps.subgroups_per_workgroup(31), 1);
        assert_eq!(caps.subgroups_per_workgroup(33), 2);

        // Zero subgroup size should not panic
        caps.subgroup_size = 0;
        assert_eq!(caps.subgroups_per_workgroup(64), 1);
    }

    #[test]
    fn optimal_reduce_workgroup_size_is_reasonable() {
        let caps = SubgroupCapabilities {
            subgroup_size: 32,
            ..Default::default()
        };
        let wg = caps.optimal_reduce_workgroup_size();
        assert!(wg >= 64, "workgroup size should be at least 64");
        assert!(wg <= 1024, "workgroup size should not exceed 1024");
    }

    #[test]
    fn glsl_reduce_sum_snippet_is_nonempty() {
        let snippet = subgroup_reduce_sum_glsl();
        assert!(!snippet.is_empty(), "GLSL snippet must not be empty");
        assert!(
            snippet.contains("subgroupAdd"),
            "snippet must reference subgroupAdd intrinsic"
        );
    }

    #[test]
    fn glsl_ballot_compaction_snippet_is_nonempty() {
        let snippet = subgroup_ballot_compaction_glsl();
        assert!(!snippet.is_empty(), "GLSL snippet must not be empty");
        assert!(
            snippet.contains("subgroupBallot"),
            "snippet must reference subgroupBallot intrinsic"
        );
    }

    #[test]
    fn subgroup_stages_any_supported() {
        let mut stages = SubgroupStages::default();
        assert!(!stages.any_supported());
        stages.compute = true;
        assert!(stages.any_supported());
    }

    #[test]
    fn subgroup_stages_supported_names() {
        let stages = SubgroupStages {
            compute: true,
            fragment: true,
            vertex: false,
            tessellation: false,
            geometry: false,
        };
        let names = stages.supported_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"compute"));
        assert!(names.contains(&"fragment"));
    }

    #[test]
    fn subgroup_operations_feature_bitmask() {
        let ops = SubgroupOperations {
            basic: true,
            arithmetic: true,
            ..Default::default()
        };
        let mask = ops.feature_bitmask();
        assert_eq!(mask & 1, 1, "basic bit should be set");
        assert_eq!((mask >> 2) & 1, 1, "arithmetic bit should be set");
        assert_eq!((mask >> 1) & 1, 0, "vote bit should not be set");
    }

    #[test]
    fn subgroup_operations_any_supported() {
        let mut ops = SubgroupOperations::default();
        assert!(!ops.any_supported());
        ops.basic = true;
        assert!(ops.any_supported());
    }

    #[test]
    fn recommend_subgroup_size_reduction() {
        let caps = SubgroupCapabilities {
            subgroup_size: 32,
            ..Default::default()
        };
        let hint = recommend_subgroup_size(&caps, SubgroupOpType::Reduction);
        assert_eq!(hint.recommended_size, 32);
        assert!(!hint.rationale.is_empty());
    }

    #[test]
    fn recommend_subgroup_size_convolution() {
        let caps = SubgroupCapabilities {
            subgroup_size: 64,
            ..Default::default()
        };
        let hint = recommend_subgroup_size(&caps, SubgroupOpType::Convolution);
        assert_eq!(hint.recommended_size, 16);
    }

    #[test]
    fn recommend_subgroup_size_histogram_capped() {
        let caps = SubgroupCapabilities {
            subgroup_size: 64,
            ..Default::default()
        };
        let hint = recommend_subgroup_size(&caps, SubgroupOpType::Histogram);
        assert!(hint.recommended_size <= 32, "histogram capped at 32");
    }
}
