//! Pipeline acceleration helpers.
//!
//! Provides parallelism hints, vectorisation helpers, and loop-unrolling
//! strategies to improve throughput in media-processing pipelines.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ──────────────────────────────────────────────────────────────────────────────
// Parallelism hints
// ──────────────────────────────────────────────────────────────────────────────

/// The degree of parallelism recommended for a workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismHint {
    /// Single-threaded only (data dependencies prevent parallelism).
    Serial,
    /// Two threads (mild data-parallel gain).
    Dual,
    /// One thread per logical CPU core (high data-parallel gain).
    FullCores,
    /// Custom thread count.
    Custom(usize),
}

impl ParallelismHint {
    /// Resolve the hint to a concrete thread count.
    ///
    /// Requires the number of logical cores available.
    #[must_use]
    pub fn thread_count(self, logical_cores: usize) -> usize {
        match self {
            Self::Serial => 1,
            Self::Dual => 2,
            Self::FullCores => logical_cores.max(1),
            Self::Custom(n) => n.max(1),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Vectorisation helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Recommended SIMD vector width (element count) for a data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorWidth {
    /// No vectorisation – scalar processing.
    Scalar,
    /// 128-bit register (SSE / NEON): 4 × f32 or 16 × u8.
    Width128,
    /// 256-bit register (AVX / AVX2): 8 × f32 or 32 × u8.
    Width256,
    /// Custom element count.
    Custom(usize),
}

impl VectorWidth {
    /// Return the number of `f32` elements per vector register.
    #[must_use]
    pub fn f32_lanes(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Width128 => 4,
            Self::Width256 => 8,
            Self::Custom(n) => n,
        }
    }

    /// Return the number of `u8` elements per vector register.
    #[must_use]
    pub fn u8_lanes(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Width128 => 16,
            Self::Width256 => 32,
            Self::Custom(n) => n,
        }
    }

    /// Return `true` if this is a SIMD width (not scalar).
    #[must_use]
    pub fn is_simd(self) -> bool {
        !matches!(self, Self::Scalar)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Loop unroll strategy
// ──────────────────────────────────────────────────────────────────────────────

/// How aggressively to unroll inner loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnrollStrategy {
    /// No explicit unrolling (compiler decides).
    Auto,
    /// Unroll by a factor of 2.
    Unroll2,
    /// Unroll by a factor of 4.
    Unroll4,
    /// Unroll by a factor of 8.
    Unroll8,
}

impl UnrollStrategy {
    /// Return the unroll factor as a `usize`.
    #[must_use]
    pub fn factor(self) -> usize {
        match self {
            Self::Auto => 1,
            Self::Unroll2 => 2,
            Self::Unroll4 => 4,
            Self::Unroll8 => 8,
        }
    }

    /// Return `true` if the strategy is compiler-controlled.
    #[must_use]
    pub fn is_auto(self) -> bool {
        matches!(self, Self::Auto)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Pipeline stage descriptor
// ──────────────────────────────────────────────────────────────────────────────

/// Describes a single processing stage in an accelerated pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Human-readable name.
    pub name: String,
    /// Parallelism hint.
    pub parallelism: ParallelismHint,
    /// Vectorisation width.
    pub vector_width: VectorWidth,
    /// Loop unroll strategy.
    pub unroll: UnrollStrategy,
    /// Whether this stage benefits from software prefetching.
    pub prefetch: bool,
}

impl PipelineStage {
    /// Create a stage with conservative defaults.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parallelism: ParallelismHint::Serial,
            vector_width: VectorWidth::Scalar,
            unroll: UnrollStrategy::Auto,
            prefetch: false,
        }
    }

    /// Set the parallelism hint.
    #[must_use]
    pub fn with_parallelism(mut self, hint: ParallelismHint) -> Self {
        self.parallelism = hint;
        self
    }

    /// Set the vector width.
    #[must_use]
    pub fn with_vector_width(mut self, width: VectorWidth) -> Self {
        self.vector_width = width;
        self
    }

    /// Set the unroll strategy.
    #[must_use]
    pub fn with_unroll(mut self, strategy: UnrollStrategy) -> Self {
        self.unroll = strategy;
        self
    }

    /// Enable software prefetching.
    #[must_use]
    pub fn with_prefetch(mut self) -> Self {
        self.prefetch = true;
        self
    }

    /// Compute an estimated throughput multiplier relative to the scalar
    /// serial baseline (1.0).
    ///
    /// This is a rough heuristic: SIMD width × unroll factor × sqrt(threads).
    #[must_use]
    pub fn throughput_estimate(&self, logical_cores: usize) -> f64 {
        let simd = self.vector_width.f32_lanes() as f64;
        let unroll = self.unroll.factor() as f64;
        let threads = self.parallelism.thread_count(logical_cores) as f64;
        simd * unroll * threads.sqrt()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Pipeline descriptor
// ──────────────────────────────────────────────────────────────────────────────

/// A full accelerated pipeline composed of named stages.
#[derive(Debug, Clone, Default)]
pub struct AccelPipeline {
    stages: Vec<PipelineStage>,
}

impl AccelPipeline {
    /// Create an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a stage.
    pub fn push(&mut self, stage: PipelineStage) {
        self.stages.push(stage);
    }

    /// Return the number of stages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Return `true` if the pipeline has no stages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Return the stage at `index`, or `None`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&PipelineStage> {
        self.stages.get(index)
    }

    /// Return stages that use SIMD vectorisation.
    #[must_use]
    pub fn simd_stages(&self) -> Vec<&PipelineStage> {
        self.stages
            .iter()
            .filter(|s| s.vector_width.is_simd())
            .collect()
    }

    /// Return stages that use software prefetching.
    #[must_use]
    pub fn prefetch_stages(&self) -> Vec<&PipelineStage> {
        self.stages.iter().filter(|s| s.prefetch).collect()
    }

    /// Return the estimated combined throughput multiplier.
    ///
    /// Assumes stages execute sequentially, so the bottleneck (minimum
    /// throughput) is the limiting factor.
    #[must_use]
    pub fn bottleneck_throughput(&self, logical_cores: usize) -> f64 {
        self.stages
            .iter()
            .map(|s| s.throughput_estimate(logical_cores))
            .fold(f64::INFINITY, f64::min)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper: pad slice length to a SIMD-friendly boundary
// ──────────────────────────────────────────────────────────────────────────────

/// Return the smallest multiple of `lanes` that is ≥ `len`.
#[must_use]
pub fn align_to_lanes(len: usize, lanes: usize) -> usize {
    let lanes = lanes.max(1);
    len.div_ceil(lanes) * lanes
}

/// Return the number of full SIMD chunks in a slice of length `len`.
#[must_use]
pub fn full_chunks(len: usize, lanes: usize) -> usize {
    len / lanes.max(1)
}

/// Return the scalar tail length (elements that don't fill a full chunk).
#[must_use]
pub fn tail_len(len: usize, lanes: usize) -> usize {
    len % lanes.max(1)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallelism_hint_serial() {
        assert_eq!(ParallelismHint::Serial.thread_count(8), 1);
    }

    #[test]
    fn test_parallelism_hint_dual() {
        assert_eq!(ParallelismHint::Dual.thread_count(8), 2);
    }

    #[test]
    fn test_parallelism_hint_full_cores() {
        assert_eq!(ParallelismHint::FullCores.thread_count(16), 16);
    }

    #[test]
    fn test_parallelism_hint_full_cores_zero_guard() {
        assert_eq!(ParallelismHint::FullCores.thread_count(0), 1);
    }

    #[test]
    fn test_parallelism_hint_custom() {
        assert_eq!(ParallelismHint::Custom(3).thread_count(8), 3);
    }

    #[test]
    fn test_vector_width_f32_lanes() {
        assert_eq!(VectorWidth::Scalar.f32_lanes(), 1);
        assert_eq!(VectorWidth::Width128.f32_lanes(), 4);
        assert_eq!(VectorWidth::Width256.f32_lanes(), 8);
    }

    #[test]
    fn test_vector_width_u8_lanes() {
        assert_eq!(VectorWidth::Width128.u8_lanes(), 16);
        assert_eq!(VectorWidth::Width256.u8_lanes(), 32);
    }

    #[test]
    fn test_vector_width_is_simd() {
        assert!(!VectorWidth::Scalar.is_simd());
        assert!(VectorWidth::Width128.is_simd());
        assert!(VectorWidth::Width256.is_simd());
    }

    #[test]
    fn test_unroll_strategy_factor() {
        assert_eq!(UnrollStrategy::Auto.factor(), 1);
        assert_eq!(UnrollStrategy::Unroll4.factor(), 4);
        assert_eq!(UnrollStrategy::Unroll8.factor(), 8);
    }

    #[test]
    fn test_unroll_strategy_is_auto() {
        assert!(UnrollStrategy::Auto.is_auto());
        assert!(!UnrollStrategy::Unroll2.is_auto());
    }

    #[test]
    fn test_pipeline_stage_throughput_scalar_serial() {
        let stage = PipelineStage::new("scalar");
        // 1 lane × 1 unroll × sqrt(1) thread = 1.0
        assert!((stage.throughput_estimate(1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pipeline_stage_throughput_simd() {
        let stage = PipelineStage::new("simd")
            .with_vector_width(VectorWidth::Width256)
            .with_unroll(UnrollStrategy::Unroll4)
            .with_parallelism(ParallelismHint::Serial);
        // 8 × 4 × 1.0 = 32.0
        assert!((stage.throughput_estimate(1) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_accel_pipeline_empty() {
        let p = AccelPipeline::new();
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn test_accel_pipeline_push_and_get() {
        let mut p = AccelPipeline::new();
        p.push(PipelineStage::new("A"));
        p.push(PipelineStage::new("B"));
        assert_eq!(p.len(), 2);
        assert_eq!(p.get(0).expect("get should succeed").name, "A");
        assert!(p.get(99).is_none());
    }

    #[test]
    fn test_accel_pipeline_simd_stages() {
        let mut p = AccelPipeline::new();
        p.push(PipelineStage::new("scalar"));
        p.push(PipelineStage::new("simd").with_vector_width(VectorWidth::Width128));
        assert_eq!(p.simd_stages().len(), 1);
    }

    #[test]
    fn test_accel_pipeline_prefetch_stages() {
        let mut p = AccelPipeline::new();
        p.push(PipelineStage::new("no_prefetch"));
        p.push(PipelineStage::new("with_prefetch").with_prefetch());
        assert_eq!(p.prefetch_stages().len(), 1);
    }

    #[test]
    fn test_align_to_lanes() {
        assert_eq!(align_to_lanes(10, 4), 12);
        assert_eq!(align_to_lanes(8, 4), 8);
        assert_eq!(align_to_lanes(0, 4), 0);
    }

    #[test]
    fn test_full_chunks_and_tail() {
        assert_eq!(full_chunks(10, 4), 2);
        assert_eq!(tail_len(10, 4), 2);
        assert_eq!(full_chunks(8, 4), 2);
        assert_eq!(tail_len(8, 4), 0);
    }

    #[test]
    fn test_bottleneck_throughput_single_stage() {
        let mut p = AccelPipeline::new();
        p.push(PipelineStage::new("only").with_vector_width(VectorWidth::Width128));
        // 4 × 1 × 1.0 = 4.0
        assert!((p.bottleneck_throughput(1) - 4.0).abs() < 1e-6);
    }
}
