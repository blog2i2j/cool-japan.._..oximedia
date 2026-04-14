//! SIMD-aware node scheduling for data-parallel pipeline filters.
//!
//! The [`SimdScheduler`] analyses an [`ExecutionPlan`] and annotates each
//! [`ExecutionStage`] with a [`SimdAnnotation`]: whether the nodes in that
//! stage are candidates for vectorised execution and which SIMD tier (SSE2,
//! AVX2, AVX-512, NEON) they should target at runtime.
//!
//! # Design
//!
//! OxiMedia deliberately avoids `unsafe` blocks and direct SIMD intrinsics in
//! this crate.  The scheduler therefore operates purely at the *planning* level:
//! it consults the [`FilterConfig`] of each node (via its relative
//! [`cost_estimate`](crate::node::FilterConfig::cost_estimate)) and the
//! abstract [`CpuCapabilities`] description to decide which vectorisation
//! tier is worth targeting.  The actual SIMD kernels live in the dedicated
//! `oximedia-simd` crate; the scheduler just records the intent.
//!
//! # Runtime SIMD tier detection
//!
//! [`CpuCapabilities::detect`] reads `std::env::var("OXIMEDIA_SIMD_TIER")` as a
//! build-time or environment override (useful for testing / cross-compilation)
//! and falls back to a compile-time feature probe:
//!
//! * `avx512f` feature → [`SimdTier::Avx512`]
//! * `avx2` feature → [`SimdTier::Avx2`]
//! * `sse2` / `x86_64` → [`SimdTier::Sse2`]
//! * `neon` (aarch64) → [`SimdTier::Neon`]
//! * Otherwise → [`SimdTier::Scalar`]
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::builder::PipelineBuilder;
//! use oximedia_pipeline::node::{SourceConfig, SinkConfig};
//! use oximedia_pipeline::execution_plan::ExecutionPlanner;
//! use oximedia_pipeline::simd_scheduler::{CpuCapabilities, SimdScheduler, SimdTier};
//!
//! let graph = PipelineBuilder::new()
//!     .source("in", SourceConfig::File("v.mp4".into()))
//!     .scale(1280, 720)
//!     .hflip()
//!     .sink("out", SinkConfig::Null)
//!     .build()
//!     .expect("valid");
//!
//! let plan = ExecutionPlanner::plan(&graph).expect("plan ok");
//! let caps = CpuCapabilities::detect();
//! let annotated = SimdScheduler::annotate(&plan, &caps);
//! assert_eq!(annotated.len(), plan.stage_count());
//! ```

use std::collections::HashMap;

use crate::execution_plan::{ExecutionPlan, ExecutionStage};
use crate::graph::PipelineGraph;
use crate::node::{FilterConfig, NodeId, NodeSpec, NodeType};

// ── SimdTier ──────────────────────────────────────────────────────────────────

/// Abstracts the available SIMD width on the executing CPU.
///
/// Ordered from weakest to strongest: `Scalar < Sse2 < Avx2 < Avx512`.
/// `Neon` is parallel to the x86 tiers and applies only on AArch64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdTier {
    /// No vectorisation; pure scalar code path.
    Scalar,
    /// 128-bit SSE2 (x86/x86-64 baseline since ~2001).
    Sse2,
    /// 256-bit AVX2 (Haswell and later).
    Avx2,
    /// 512-bit AVX-512 (Skylake-X and later).
    Avx512,
    /// 128-bit ARM NEON (AArch64 baseline).
    Neon,
}

impl SimdTier {
    /// Human-readable name for the tier.
    pub fn name(self) -> &'static str {
        match self {
            SimdTier::Scalar => "scalar",
            SimdTier::Sse2 => "sse2",
            SimdTier::Avx2 => "avx2",
            SimdTier::Avx512 => "avx512",
            SimdTier::Neon => "neon",
        }
    }

    /// Register width in bits.
    pub fn register_bits(self) -> u32 {
        match self {
            SimdTier::Scalar => 0,
            SimdTier::Sse2 | SimdTier::Neon => 128,
            SimdTier::Avx2 => 256,
            SimdTier::Avx512 => 512,
        }
    }

    /// Whether this tier provides vectorised execution (not scalar).
    pub fn is_vectorised(self) -> bool {
        self != SimdTier::Scalar
    }
}

impl std::fmt::Display for SimdTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ── CpuCapabilities ───────────────────────────────────────────────────────────

/// Describes the SIMD capabilities of the executing CPU.
///
/// Callers may construct this directly for testing, or use
/// [`Self::detect`] to probe the current environment.
#[derive(Debug, Clone)]
pub struct CpuCapabilities {
    /// Best SIMD tier available at runtime.
    pub best_tier: SimdTier,
    /// Number of logical cores (used for parallelism estimates).
    pub logical_cores: usize,
    /// Whether the CPU supports fused multiply-add instructions.
    pub has_fma: bool,
    /// Whether the CPU supports half-precision (FP16) arithmetic.
    pub has_fp16: bool,
}

impl CpuCapabilities {
    /// Detect capabilities at runtime.
    ///
    /// Reads `OXIMEDIA_SIMD_TIER` environment variable for overrides
    /// (`scalar`, `sse2`, `avx2`, `avx512`, `neon`).  Falls back to
    /// compile-time feature flags.
    pub fn detect() -> Self {
        let tier = Self::detect_tier_from_env().unwrap_or_else(Self::detect_tier_compile_time);

        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Conservatively assume FMA and FP16 when AVX2 or better is present.
        let has_fma = matches!(tier, SimdTier::Avx2 | SimdTier::Avx512);
        let has_fp16 = matches!(tier, SimdTier::Avx512 | SimdTier::Neon);

        Self {
            best_tier: tier,
            logical_cores,
            has_fma,
            has_fp16,
        }
    }

    /// Override all fields manually.  Useful for unit tests.
    pub fn with_tier(tier: SimdTier, logical_cores: usize) -> Self {
        let has_fma = matches!(tier, SimdTier::Avx2 | SimdTier::Avx512);
        let has_fp16 = matches!(tier, SimdTier::Avx512 | SimdTier::Neon);
        Self {
            best_tier: tier,
            logical_cores,
            has_fma,
            has_fp16,
        }
    }

    fn detect_tier_from_env() -> Option<SimdTier> {
        std::env::var("OXIMEDIA_SIMD_TIER").ok().and_then(|v| {
            match v.to_ascii_lowercase().as_str() {
                "scalar" => Some(SimdTier::Scalar),
                "sse2" => Some(SimdTier::Sse2),
                "avx2" => Some(SimdTier::Avx2),
                "avx512" => Some(SimdTier::Avx512),
                "neon" => Some(SimdTier::Neon),
                _ => None,
            }
        })
    }

    fn detect_tier_compile_time() -> SimdTier {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        return SimdTier::Avx512;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        return SimdTier::Avx2;

        #[cfg(target_arch = "x86_64")]
        return SimdTier::Sse2;

        #[cfg(target_arch = "aarch64")]
        return SimdTier::Neon;

        #[allow(unreachable_code)]
        SimdTier::Scalar
    }
}

impl Default for CpuCapabilities {
    fn default() -> Self {
        Self::detect()
    }
}

// ── SimdFilterClass ───────────────────────────────────────────────────────────

/// Classifies a filter into a SIMD workload category.
///
/// The category drives which SIMD tier is *required* to see a meaningful
/// speedup.  Cheap scalar operations (e.g. `Hflip`) only need SSE2; heavy
/// convolutions (e.g. `Scale` with Lanczos) benefit from AVX2+.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdFilterClass {
    /// The operation is control-flow heavy or memory-bound; SIMD gives
    /// negligible benefit.  Run on any tier including `Scalar`.
    NotSuitableForSimd,
    /// Byte-level permutation or simple arithmetic; SSE2 / NEON suffice.
    BasicVectorOp,
    /// Floating-point arithmetic on YUV/RGB planes; AVX2 or NEON desirable.
    FloatIntensive,
    /// Matrix multiply or convolution kernels; AVX2 minimum, AVX-512 ideal.
    ConvolutionHeavy,
}

impl SimdFilterClass {
    /// Minimum [`SimdTier`] that provides meaningful acceleration.
    pub fn minimum_tier(self) -> SimdTier {
        match self {
            SimdFilterClass::NotSuitableForSimd => SimdTier::Scalar,
            SimdFilterClass::BasicVectorOp => SimdTier::Sse2,
            SimdFilterClass::FloatIntensive => SimdTier::Avx2,
            SimdFilterClass::ConvolutionHeavy => SimdTier::Avx2,
        }
    }
}

/// Classify a [`FilterConfig`] into a [`SimdFilterClass`].
fn classify_filter(cfg: &FilterConfig) -> SimdFilterClass {
    match cfg {
        FilterConfig::Hflip | FilterConfig::Vflip | FilterConfig::Transpose(_) => {
            SimdFilterClass::BasicVectorOp
        }
        FilterConfig::Format(_) => SimdFilterClass::BasicVectorOp,
        FilterConfig::Volume { .. } | FilterConfig::Fps { .. } | FilterConfig::Trim { .. } => {
            SimdFilterClass::NotSuitableForSimd
        }
        FilterConfig::Scale { .. } | FilterConfig::Crop { .. } | FilterConfig::Pad { .. } => {
            SimdFilterClass::ConvolutionHeavy
        }
        FilterConfig::Overlay | FilterConfig::Concat { .. } => SimdFilterClass::FloatIntensive,
        FilterConfig::Custom { .. } => SimdFilterClass::FloatIntensive,
        FilterConfig::Parametric { base, .. } => classify_filter(base),
    }
}

// ── SimdAnnotation ────────────────────────────────────────────────────────────

/// SIMD scheduling annotation attached to an [`ExecutionStage`].
#[derive(Debug, Clone)]
pub struct SimdAnnotation {
    /// Stage index this annotation corresponds to.
    pub stage_id: usize,
    /// Whether *any* node in the stage is a SIMD candidate.
    pub has_simd_candidates: bool,
    /// Recommended SIMD tier for the dominant workload in this stage.
    pub recommended_tier: SimdTier,
    /// Whether the current CPU actually supports `recommended_tier`.
    pub tier_supported_by_cpu: bool,
    /// Per-node classification map (`NodeId` → `SimdFilterClass`).
    pub node_classes: HashMap<NodeId, SimdFilterClass>,
    /// Estimated speedup factor when running at `recommended_tier` vs scalar.
    ///
    /// This is a rough heuristic based on register width:
    /// `max(1.0, register_bits / 64)`.
    pub estimated_speedup: f32,
}

impl SimdAnnotation {
    /// Returns `true` when the CPU can execute the recommended tier.
    pub fn is_acceleratable(&self) -> bool {
        self.has_simd_candidates && self.tier_supported_by_cpu
    }
}

// ── SimdScheduler ─────────────────────────────────────────────────────────────

/// Annotates an [`ExecutionPlan`] with SIMD scheduling hints.
///
/// The scheduler is stateless — create one per invocation or reuse it for
/// multiple plans.
pub struct SimdScheduler;

impl SimdScheduler {
    /// Annotate every stage of `plan` with SIMD scheduling hints.
    ///
    /// Returns a `Vec<SimdAnnotation>` with one entry per stage, in stage
    /// order.  The function does not modify the plan itself; annotations are
    /// advisory metadata for the executor.
    pub fn annotate(plan: &ExecutionPlan, caps: &CpuCapabilities) -> Vec<SimdAnnotation> {
        plan.stages
            .iter()
            .map(|stage| Self::annotate_stage(stage, caps))
            .collect()
    }

    fn annotate_stage(stage: &ExecutionStage, caps: &CpuCapabilities) -> SimdAnnotation {
        // We only have NodeIds here; SIMD class requires FilterConfig.
        // Build a synthetic classification based on the stage's CPU weight as
        // a proxy (the execution plan stores resource estimates, not node
        // types, so we derive classification from the aggregated cost weight).
        let cpu_weight = stage.resource_estimate.cpu_weight;

        let dominant_class = if cpu_weight <= 0.0 {
            SimdFilterClass::NotSuitableForSimd
        } else if cpu_weight < 3.0 {
            SimdFilterClass::BasicVectorOp
        } else if cpu_weight < 8.0 {
            SimdFilterClass::FloatIntensive
        } else {
            SimdFilterClass::ConvolutionHeavy
        };

        let recommended_tier = dominant_class.minimum_tier();

        // Check whether the CPU supports the recommended tier.
        let tier_supported = Self::tier_supported(recommended_tier, caps);

        // Compute speedup heuristic: register_bits / 64, minimum 1.0.
        let effective_tier = if tier_supported {
            recommended_tier
        } else {
            caps.best_tier
        };
        let speedup = {
            let bits = effective_tier.register_bits();
            if bits == 0 {
                1.0f32
            } else {
                (bits as f32 / 64.0_f32).max(1.0)
            }
        };

        let has_candidates = dominant_class != SimdFilterClass::NotSuitableForSimd;

        // Build per-node class map using the same weight heuristic per node.
        // (A real executor would inspect NodeType; we use a cost proxy here.)
        let node_classes: HashMap<NodeId, SimdFilterClass> = stage
            .nodes
            .iter()
            .map(|&id| (id, dominant_class))
            .collect();

        SimdAnnotation {
            stage_id: stage.stage_id,
            has_simd_candidates: has_candidates,
            recommended_tier,
            tier_supported_by_cpu: tier_supported,
            node_classes,
            estimated_speedup: speedup,
        }
    }

    /// Returns `true` when `tier` is supported by `caps`.
    fn tier_supported(tier: SimdTier, caps: &CpuCapabilities) -> bool {
        match tier {
            SimdTier::Scalar => true,
            SimdTier::Neon => caps.best_tier == SimdTier::Neon,
            other => caps.best_tier >= other && caps.best_tier != SimdTier::Neon,
        }
    }

    /// Classify a [`FilterConfig`] for use outside the scheduler (public API).
    pub fn classify(filter: &FilterConfig) -> SimdFilterClass {
        classify_filter(filter)
    }

    /// Annotate a graph directly, bypassing the execution plan abstraction.
    ///
    /// Each node is classified individually and returned as a `HashMap` from
    /// `NodeId` to `SimdFilterClass`.
    pub fn classify_graph(
        graph: &PipelineGraph,
        caps: &CpuCapabilities,
    ) -> HashMap<NodeId, SimdNodeSchedule> {
        graph
            .nodes
            .iter()
            .map(|(&id, spec)| {
                let sched = Self::classify_node(spec, caps);
                (id, sched)
            })
            .collect()
    }

    fn classify_node(spec: &NodeSpec, caps: &CpuCapabilities) -> SimdNodeSchedule {
        let class = match &spec.node_type {
            NodeType::Filter(cfg) => classify_filter(cfg),
            NodeType::Source(_) | NodeType::Sink(_) => SimdFilterClass::NotSuitableForSimd,
            NodeType::Split | NodeType::Merge => SimdFilterClass::BasicVectorOp,
            NodeType::Null | NodeType::Conditional(_) => SimdFilterClass::NotSuitableForSimd,
        };

        let min_tier = class.minimum_tier();
        let acceleratable = Self::tier_supported(min_tier, caps) && class != SimdFilterClass::NotSuitableForSimd;

        SimdNodeSchedule {
            node_id: spec.id,
            node_name: spec.name.clone(),
            filter_class: class,
            minimum_tier: min_tier,
            acceleratable,
        }
    }
}

// ── SimdNodeSchedule ──────────────────────────────────────────────────────────

/// Per-node SIMD scheduling decision.
#[derive(Debug, Clone)]
pub struct SimdNodeSchedule {
    /// Node identifier.
    pub node_id: NodeId,
    /// Human-readable node name.
    pub node_name: String,
    /// SIMD workload classification.
    pub filter_class: SimdFilterClass,
    /// Minimum tier required for meaningful acceleration.
    pub minimum_tier: SimdTier,
    /// Whether the current CPU can accelerate this node.
    pub acceleratable: bool,
}

// ── SimdScheduleReport ────────────────────────────────────────────────────────

/// Summary of SIMD scheduling decisions for an entire pipeline.
#[derive(Debug, Clone)]
pub struct SimdScheduleReport {
    /// CPU capabilities used for this report.
    pub cpu_capabilities: CpuCapabilities,
    /// Total number of pipeline stages.
    pub total_stages: usize,
    /// Number of stages with at least one SIMD candidate.
    pub simd_stages: usize,
    /// Number of stages where the CPU can fully support the recommended tier.
    pub acceleratable_stages: usize,
    /// Per-stage annotations, in stage order.
    pub annotations: Vec<SimdAnnotation>,
}

impl SimdScheduleReport {
    /// Build a complete report from an execution plan.
    pub fn from_plan(plan: &ExecutionPlan, caps: &CpuCapabilities) -> Self {
        let annotations = SimdScheduler::annotate(plan, caps);
        let simd_stages = annotations.iter().filter(|a| a.has_simd_candidates).count();
        let acceleratable_stages = annotations.iter().filter(|a| a.is_acceleratable()).count();
        Self {
            cpu_capabilities: caps.clone(),
            total_stages: plan.stage_count(),
            simd_stages,
            acceleratable_stages,
            annotations,
        }
    }

    /// Fraction of stages that are SIMD-acceleratable on the current CPU.
    pub fn acceleration_coverage(&self) -> f32 {
        if self.total_stages == 0 {
            return 0.0;
        }
        self.acceleratable_stages as f32 / self.total_stages as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::PipelineBuilder;
    use crate::execution_plan::ExecutionPlanner;
    use crate::node::{FilterConfig, FrameFormat, SinkConfig, SourceConfig};

    fn make_caps(tier: SimdTier) -> CpuCapabilities {
        CpuCapabilities::with_tier(tier, 4)
    }

    fn scale_hflip_graph() -> crate::graph::PipelineGraph {
        PipelineBuilder::new()
            .source("in", SourceConfig::File("v.mp4".into()))
            .scale(1280, 720)
            .hflip()
            .sink("out", SinkConfig::Null)
            .build()
            .expect("valid graph")
    }

    // ── SimdTier tests ────────────────────────────────────────────────────────

    #[test]
    fn simd_tier_ordering() {
        assert!(SimdTier::Scalar < SimdTier::Sse2);
        assert!(SimdTier::Sse2 < SimdTier::Avx2);
        assert!(SimdTier::Avx2 < SimdTier::Avx512);
    }

    #[test]
    fn simd_tier_register_bits() {
        assert_eq!(SimdTier::Scalar.register_bits(), 0);
        assert_eq!(SimdTier::Sse2.register_bits(), 128);
        assert_eq!(SimdTier::Avx2.register_bits(), 256);
        assert_eq!(SimdTier::Avx512.register_bits(), 512);
        assert_eq!(SimdTier::Neon.register_bits(), 128);
    }

    #[test]
    fn simd_tier_is_vectorised() {
        assert!(!SimdTier::Scalar.is_vectorised());
        assert!(SimdTier::Sse2.is_vectorised());
        assert!(SimdTier::Avx2.is_vectorised());
    }

    // ── CpuCapabilities tests ─────────────────────────────────────────────────

    #[test]
    fn cpu_capabilities_with_tier() {
        let caps = CpuCapabilities::with_tier(SimdTier::Avx2, 8);
        assert_eq!(caps.best_tier, SimdTier::Avx2);
        assert_eq!(caps.logical_cores, 8);
        assert!(caps.has_fma); // AVX2 → FMA
    }

    #[test]
    fn cpu_capabilities_scalar_no_fma() {
        let caps = CpuCapabilities::with_tier(SimdTier::Scalar, 1);
        assert!(!caps.has_fma);
        assert!(!caps.has_fp16);
    }

    // ── SimdFilterClass tests ─────────────────────────────────────────────────

    #[test]
    fn classify_hflip_basic_vector_op() {
        let class = SimdScheduler::classify(&FilterConfig::Hflip);
        assert_eq!(class, SimdFilterClass::BasicVectorOp);
    }

    #[test]
    fn classify_scale_convolution_heavy() {
        let class = SimdScheduler::classify(&FilterConfig::Scale { width: 1920, height: 1080 });
        assert_eq!(class, SimdFilterClass::ConvolutionHeavy);
    }

    #[test]
    fn classify_volume_not_suitable() {
        let class = SimdScheduler::classify(&FilterConfig::Volume { gain_db: 3.0 });
        assert_eq!(class, SimdFilterClass::NotSuitableForSimd);
    }

    #[test]
    fn classify_parametric_delegates_to_base() {
        let base = FilterConfig::Scale { width: 640, height: 360 };
        let param = FilterConfig::parametric(base, Default::default());
        let class = SimdScheduler::classify(&param);
        assert_eq!(class, SimdFilterClass::ConvolutionHeavy);
    }

    // ── SimdScheduler::annotate tests ─────────────────────────────────────────

    #[test]
    fn annotate_returns_one_entry_per_stage() {
        let graph = scale_hflip_graph();
        let plan = ExecutionPlanner::plan(&graph).expect("plan ok");
        let caps = make_caps(SimdTier::Avx2);
        let annotations = SimdScheduler::annotate(&plan, &caps);
        assert_eq!(annotations.len(), plan.stage_count());
    }

    #[test]
    fn annotate_stage_ids_match_plan() {
        let graph = scale_hflip_graph();
        let plan = ExecutionPlanner::plan(&graph).expect("plan ok");
        let caps = make_caps(SimdTier::Avx2);
        let annotations = SimdScheduler::annotate(&plan, &caps);
        for (i, ann) in annotations.iter().enumerate() {
            assert_eq!(ann.stage_id, i);
        }
    }

    #[test]
    fn annotate_speedup_at_least_one() {
        let graph = scale_hflip_graph();
        let plan = ExecutionPlanner::plan(&graph).expect("plan ok");
        let caps = make_caps(SimdTier::Avx2);
        let annotations = SimdScheduler::annotate(&plan, &caps);
        for ann in &annotations {
            assert!(ann.estimated_speedup >= 1.0, "speedup must be ≥ 1.0");
        }
    }

    // ── SimdScheduleReport tests ──────────────────────────────────────────────

    #[test]
    fn report_totals_match_plan() {
        let graph = scale_hflip_graph();
        let plan = ExecutionPlanner::plan(&graph).expect("plan ok");
        let caps = make_caps(SimdTier::Avx2);
        let report = SimdScheduleReport::from_plan(&plan, &caps);
        assert_eq!(report.total_stages, plan.stage_count());
        assert!(report.simd_stages <= report.total_stages);
        assert!(report.acceleratable_stages <= report.simd_stages);
    }

    #[test]
    fn report_acceleration_coverage_in_range() {
        let graph = scale_hflip_graph();
        let plan = ExecutionPlanner::plan(&graph).expect("plan ok");
        let caps = make_caps(SimdTier::Avx2);
        let report = SimdScheduleReport::from_plan(&plan, &caps);
        let cov = report.acceleration_coverage();
        assert!((0.0..=1.0).contains(&cov));
    }

    #[test]
    fn report_empty_plan_coverage_zero() {
        let plan = crate::execution_plan::ExecutionPlan::new(vec![]);
        let caps = make_caps(SimdTier::Scalar);
        let report = SimdScheduleReport::from_plan(&plan, &caps);
        assert_eq!(report.acceleration_coverage(), 0.0);
        assert_eq!(report.total_stages, 0);
    }

    // ── classify_graph tests ──────────────────────────────────────────────────

    #[test]
    fn classify_graph_covers_all_nodes() {
        let graph = scale_hflip_graph();
        let caps = make_caps(SimdTier::Avx2);
        let schedule = SimdScheduler::classify_graph(&graph, &caps);
        assert_eq!(schedule.len(), graph.node_count());
    }

    #[test]
    fn classify_graph_source_not_suitable() {
        let graph = scale_hflip_graph();
        let caps = make_caps(SimdTier::Avx2);
        let schedule = SimdScheduler::classify_graph(&graph, &caps);
        // At least one node should be NotSuitableForSimd (source node).
        let has_non_simd = schedule.values().any(|s| s.filter_class == SimdFilterClass::NotSuitableForSimd);
        assert!(has_non_simd);
    }
}
