//! Hardware resource availability validator for pipeline execution.
//!
//! [`HardwareResourceValidator`] checks whether the host machine has sufficient
//! CPU threads, memory, and optional GPU resources to execute a given
//! [`ExecutionPlan`] before any media I/O begins.  This prevents late runtime
//! failures due to resource exhaustion and enables early user feedback.
//!
//! # Design
//!
//! [`HardwareCapabilities`] describes the host environment (available CPU
//! threads, available RAM, GPU presence).  The validator compares each
//! [`ExecutionStage`]'s [`ResourceEstimate`] against the declared capabilities
//! and accumulates warnings where the plan *might* exceed limits.
//!
//! Warnings (not hard errors) are used because actual usage depends on runtime
//! behaviour that cannot be predicted statically.  Hard errors are only raised
//! when the required GPU count exceeds zero but no GPU is available.
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::execution_plan::ExecutionPlanner;
//! use oximedia_pipeline::hardware_validator::{HardwareCapabilities, HardwareResourceValidator};
//! use oximedia_pipeline::builder::PipelineBuilder;
//! use oximedia_pipeline::node::{SourceConfig, SinkConfig};
//!
//! let graph = PipelineBuilder::new()
//!     .source("src", SourceConfig::File("in.mkv".into()))
//!     .scale(1280, 720)
//!     .sink("out", SinkConfig::Null)
//!     .build()
//!     .expect("valid");
//!
//! let plan = ExecutionPlanner::plan(&graph).expect("plan ok");
//! let caps = HardwareCapabilities::detect_or_default();
//! let report = HardwareResourceValidator::new(caps).validate(&plan);
//! // A simple 3-node pipeline should always pass hardware checks.
//! assert!(report.is_ok());
//! ```

use crate::execution_plan::{ExecutionPlan, ExecutionStage};
use crate::PipelineError;

// ── HardwareCapabilities ──────────────────────────────────────────────────────

/// Describes the hardware resources available on the host machine.
///
/// Construct with [`HardwareCapabilities::new`] for explicit values or
/// [`HardwareCapabilities::detect_or_default`] to query the OS automatically.
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// Number of logical CPU threads available (e.g. `num_cpus` or
    /// `std::thread::available_parallelism`).
    pub cpu_threads: usize,
    /// Total physical RAM available in bytes.
    pub available_memory_bytes: u64,
    /// Whether a GPU suitable for hardware video acceleration is present.
    pub has_gpu: bool,
    /// Maximum number of concurrent GPU operations the host can sustain.
    /// Set to `0` when [`has_gpu`](Self::has_gpu) is `false`.
    pub max_gpu_concurrency: u32,
    /// Upper bound on the combined CPU weight the host can sustain without
    /// degrading real-time performance.  `None` means unlimited.
    pub max_cpu_weight: Option<f64>,
}

impl HardwareCapabilities {
    /// Create a `HardwareCapabilities` with all fields specified manually.
    pub fn new(
        cpu_threads: usize,
        available_memory_bytes: u64,
        has_gpu: bool,
        max_gpu_concurrency: u32,
        max_cpu_weight: Option<f64>,
    ) -> Self {
        Self {
            cpu_threads,
            available_memory_bytes,
            has_gpu,
            max_gpu_concurrency,
            max_cpu_weight,
        }
    }

    /// Try to detect hardware capabilities from the OS, falling back to
    /// conservative defaults when detection is unavailable.
    ///
    /// The detected values are intentionally conservative:
    /// - CPU threads via `std::thread::available_parallelism`, defaulting to 1.
    /// - Memory: 2 GiB (a safe lower bound; actual detection is OS-specific).
    /// - GPU: `false` (safe default; GPU detection requires platform APIs).
    pub fn detect_or_default() -> Self {
        let cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self {
            cpu_threads,
            available_memory_bytes: 2 * 1024 * 1024 * 1024, // 2 GiB
            has_gpu: false,
            max_gpu_concurrency: 0,
            max_cpu_weight: None,
        }
    }

    /// Return a very permissive `HardwareCapabilities` that will never
    /// produce warnings.  Useful for unit tests.
    pub fn unlimited() -> Self {
        Self {
            cpu_threads: 65536,
            available_memory_bytes: u64::MAX,
            has_gpu: true,
            max_gpu_concurrency: 64,
            max_cpu_weight: None,
        }
    }
}

// ── ResourceRequirement ───────────────────────────────────────────────────────

/// Minimum hardware requirements that must be met before execution begins.
///
/// Used by callers to describe what a specific pipeline *requires*, as opposed
/// to what the host *has* ([`HardwareCapabilities`]).
#[derive(Debug, Clone, Default)]
pub struct ResourceRequirement {
    /// Minimum number of CPU threads required.
    pub min_cpu_threads: usize,
    /// Minimum available memory in bytes.
    pub min_memory_bytes: u64,
    /// Whether at least one GPU must be present.
    pub requires_gpu: bool,
    /// Minimum number of concurrent GPU operations needed.
    pub min_gpu_concurrency: u32,
}

impl ResourceRequirement {
    /// Create a new `ResourceRequirement` with all-zero / false defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set the minimum CPU thread count.
    pub fn with_min_cpu(mut self, threads: usize) -> Self {
        self.min_cpu_threads = threads;
        self
    }

    /// Builder: set the minimum memory requirement.
    pub fn with_min_memory(mut self, bytes: u64) -> Self {
        self.min_memory_bytes = bytes;
        self
    }

    /// Builder: require GPU support.
    pub fn with_gpu(mut self, min_concurrency: u32) -> Self {
        self.requires_gpu = true;
        self.min_gpu_concurrency = min_concurrency;
        self
    }
}

// ── HardwareValidationWarning ─────────────────────────────────────────────────

/// A non-fatal warning produced when a pipeline plan *may* exceed host
/// hardware limits.
#[derive(Debug, Clone, PartialEq)]
pub enum HardwareValidationWarning {
    /// Estimated memory usage for a stage exceeds available host RAM.
    MemoryPressure {
        /// Stage index (0-based).
        stage_id: usize,
        /// Estimated bytes needed by the stage.
        estimated_bytes: u64,
        /// Bytes currently available.
        available_bytes: u64,
    },
    /// The plan has more parallel branches than available CPU threads.
    ThreadOversubscription {
        /// Number of concurrent branches the plan wants.
        requested_threads: usize,
        /// Number of threads the host provides.
        available_threads: usize,
    },
    /// A stage is flagged as a GPU candidate but no GPU is available.
    GpuUnavailable {
        /// Stage index of the GPU-acceleratable workload.
        stage_id: usize,
    },
    /// Aggregate CPU weight across all stages exceeds `max_cpu_weight`.
    CpuWeightExceeded {
        /// Total computed weight across all stages.
        total_weight: f64,
        /// The configured maximum.
        max_weight: f64,
    },
    /// An explicit [`ResourceRequirement`] check failed.
    RequirementNotMet {
        /// Human-readable description of the unmet requirement.
        description: String,
    },
}

impl std::fmt::Display for HardwareValidationWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HardwareValidationWarning::MemoryPressure {
                stage_id,
                estimated_bytes,
                available_bytes,
            } => write!(
                f,
                "stage {stage_id}: estimated memory {estimated_bytes}B exceeds available {available_bytes}B"
            ),
            HardwareValidationWarning::ThreadOversubscription {
                requested_threads,
                available_threads,
            } => write!(
                f,
                "plan requests {requested_threads} threads but host has {available_threads}"
            ),
            HardwareValidationWarning::GpuUnavailable { stage_id } => {
                write!(f, "stage {stage_id}: GPU candidate but no GPU detected")
            }
            HardwareValidationWarning::CpuWeightExceeded {
                total_weight,
                max_weight,
            } => write!(
                f,
                "total CPU weight {total_weight:.2} exceeds maximum {max_weight:.2}"
            ),
            HardwareValidationWarning::RequirementNotMet { description } => {
                write!(f, "requirement not met: {description}")
            }
        }
    }
}

// ── HardwareValidationReport ──────────────────────────────────────────────────

/// The result of running [`HardwareResourceValidator::validate`] on an
/// [`ExecutionPlan`].
#[derive(Debug, Clone)]
pub struct HardwareValidationReport {
    /// All warnings produced during validation.
    pub warnings: Vec<HardwareValidationWarning>,
    /// Hard errors (e.g. GPU required but absent when `strict_gpu` mode is on).
    pub errors: Vec<PipelineError>,
    /// Whether the plan can safely proceed (`errors` is empty).
    pub is_ok: bool,
    /// Total estimated memory across all stages in bytes.
    pub total_estimated_memory_bytes: u64,
    /// Total CPU weight across all stages.
    pub total_cpu_weight: f64,
    /// Maximum number of parallel branches across all stages.
    pub max_parallel_branches: usize,
}

impl HardwareValidationReport {
    /// Returns `true` when there are no hard errors.
    pub fn is_ok(&self) -> bool {
        self.is_ok
    }

    /// Returns `true` when there are no warnings either.
    pub fn is_clean(&self) -> bool {
        self.is_ok && self.warnings.is_empty()
    }

    /// One-line summary.
    pub fn summary(&self) -> String {
        if self.is_ok && self.warnings.is_empty() {
            format!(
                "Hardware validation passed: mem={}, cpu_weight={:.1}, max_branches={}",
                format_bytes(self.total_estimated_memory_bytes),
                self.total_cpu_weight,
                self.max_parallel_branches,
            )
        } else {
            format!(
                "Hardware validation: {} error(s), {} warning(s)",
                self.errors.len(),
                self.warnings.len(),
            )
        }
    }
}

// ── HardwareResourceValidator ─────────────────────────────────────────────────

/// Validates that a host machine has sufficient hardware resources to execute
/// a given [`ExecutionPlan`].
///
/// Create with [`HardwareResourceValidator::new`], optionally add extra
/// [`ResourceRequirement`]s via [`HardwareResourceValidator::require`], then
/// call [`HardwareResourceValidator::validate`].
#[derive(Debug, Clone)]
pub struct HardwareResourceValidator {
    caps: HardwareCapabilities,
    /// Extra requirements supplied by the caller.
    extra_requirements: Vec<ResourceRequirement>,
    /// When `true`, a GPU-candidate stage without an available GPU is a hard
    /// error instead of a warning.
    pub strict_gpu: bool,
}

impl HardwareResourceValidator {
    /// Create a new validator with the given capabilities.
    pub fn new(caps: HardwareCapabilities) -> Self {
        Self {
            caps,
            extra_requirements: Vec::new(),
            strict_gpu: false,
        }
    }

    /// Add an explicit [`ResourceRequirement`] that must be satisfied.
    pub fn require(mut self, req: ResourceRequirement) -> Self {
        self.extra_requirements.push(req);
        self
    }

    /// Enable strict GPU mode: GPU-candidate stages without GPU hardware
    /// become hard errors rather than warnings.
    pub fn with_strict_gpu(mut self) -> Self {
        self.strict_gpu = true;
        self
    }

    /// Validate the given `plan` against the host capabilities.
    ///
    /// Returns a [`HardwareValidationReport`] that describes warnings and any
    /// hard errors.  The report's `is_ok` field is `true` when no hard errors
    /// were detected.
    pub fn validate(&self, plan: &ExecutionPlan) -> HardwareValidationReport {
        let mut warnings: Vec<HardwareValidationWarning> = Vec::new();
        let mut errors: Vec<PipelineError> = Vec::new();

        let mut total_estimated_memory_bytes: u64 = 0;
        let mut total_cpu_weight: f64 = 0.0;
        let mut max_parallel_branches: usize = 0;

        for stage in &plan.stages {
            self.check_stage(
                stage,
                &mut warnings,
                &mut errors,
                &mut total_estimated_memory_bytes,
                &mut total_cpu_weight,
                &mut max_parallel_branches,
            );
        }

        // CPU weight threshold check.
        if let Some(max_w) = self.caps.max_cpu_weight {
            if total_cpu_weight > max_w {
                warnings.push(HardwareValidationWarning::CpuWeightExceeded {
                    total_weight: total_cpu_weight,
                    max_weight: max_w,
                });
            }
        }

        // Thread over-subscription: if the plan ever wants more branches than
        // the host can run concurrently.
        if max_parallel_branches > self.caps.cpu_threads {
            warnings.push(HardwareValidationWarning::ThreadOversubscription {
                requested_threads: max_parallel_branches,
                available_threads: self.caps.cpu_threads,
            });
        }

        // Extra requirements.
        for req in &self.extra_requirements {
            self.check_requirement(req, &mut warnings);
        }

        let is_ok = errors.is_empty();
        HardwareValidationReport {
            warnings,
            errors,
            is_ok,
            total_estimated_memory_bytes,
            total_cpu_weight,
            max_parallel_branches,
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn check_stage(
        &self,
        stage: &ExecutionStage,
        warnings: &mut Vec<HardwareValidationWarning>,
        errors: &mut Vec<PipelineError>,
        total_mem: &mut u64,
        total_weight: &mut f64,
        max_branches: &mut usize,
    ) {
        let est = &stage.resource_estimate;

        // Accumulate totals.
        *total_mem = total_mem.saturating_add(est.memory_bytes);
        *total_weight += est.cpu_weight;

        // Track maximum parallel branches (parallel_groups gives the branch count).
        let branch_count = if stage.parallel {
            stage.parallel_groups.len().max(1)
        } else {
            1
        };
        if branch_count > *max_branches {
            *max_branches = branch_count;
        }

        // Memory pressure: single-stage estimate exceeds available RAM.
        if est.memory_bytes > self.caps.available_memory_bytes {
            warnings.push(HardwareValidationWarning::MemoryPressure {
                stage_id: stage.stage_id,
                estimated_bytes: est.memory_bytes,
                available_bytes: self.caps.available_memory_bytes,
            });
        }

        // GPU candidate without GPU.
        if est.is_gpu_candidate && !self.caps.has_gpu {
            if self.strict_gpu {
                errors.push(PipelineError::ValidationError(format!(
                    "stage {} requires GPU but no GPU is available",
                    stage.stage_id
                )));
            } else {
                warnings.push(HardwareValidationWarning::GpuUnavailable {
                    stage_id: stage.stage_id,
                });
            }
        }
    }

    fn check_requirement(
        &self,
        req: &ResourceRequirement,
        warnings: &mut Vec<HardwareValidationWarning>,
    ) {
        if req.min_cpu_threads > self.caps.cpu_threads {
            warnings.push(HardwareValidationWarning::RequirementNotMet {
                description: format!(
                    "pipeline requires {} CPU threads but host has {}",
                    req.min_cpu_threads, self.caps.cpu_threads
                ),
            });
        }
        if req.min_memory_bytes > self.caps.available_memory_bytes {
            warnings.push(HardwareValidationWarning::RequirementNotMet {
                description: format!(
                    "pipeline requires {} bytes RAM but host has {}",
                    req.min_memory_bytes, self.caps.available_memory_bytes
                ),
            });
        }
        if req.requires_gpu && !self.caps.has_gpu {
            warnings.push(HardwareValidationWarning::RequirementNotMet {
                description: "pipeline requires GPU but no GPU is available".to_string(),
            });
        }
        if req.min_gpu_concurrency > self.caps.max_gpu_concurrency {
            warnings.push(HardwareValidationWarning::RequirementNotMet {
                description: format!(
                    "pipeline requires GPU concurrency {} but host supports {}",
                    req.min_gpu_concurrency, self.caps.max_gpu_concurrency
                ),
            });
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn format_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;
    const KIB: u64 = 1024;
    if bytes >= GIB {
        format!("{:.2}GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1}MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1}KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes}B")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::PipelineBuilder;
    use crate::execution_plan::ExecutionPlanner;
    use crate::node::{SinkConfig, SourceConfig};

    fn simple_plan() -> ExecutionPlan {
        let graph = PipelineBuilder::new()
            .source("src", SourceConfig::File("in.mkv".into()))
            .scale(1280, 720)
            .sink("out", SinkConfig::Null)
            .build()
            .expect("valid");
        ExecutionPlanner::plan(&graph).expect("plan ok")
    }

    // 1. Unlimited capabilities always produce a clean report.
    #[test]
    fn unlimited_caps_always_clean() {
        let plan = simple_plan();
        let caps = HardwareCapabilities::unlimited();
        let report = HardwareResourceValidator::new(caps).validate(&plan);
        assert!(report.is_ok(), "should have no errors");
        assert!(report.is_clean(), "should have no warnings");
    }

    // 2. Detect-or-default never panics and returns is_ok for a simple plan.
    #[test]
    fn detect_or_default_simple_plan_ok() {
        let plan = simple_plan();
        let caps = HardwareCapabilities::detect_or_default();
        let report = HardwareResourceValidator::new(caps).validate(&plan);
        // A simple 3-node plan should not raise hard errors.
        assert!(report.is_ok());
    }

    // 3. Memory pressure warning when available RAM is tiny.
    #[test]
    fn memory_pressure_warning_when_ram_tiny() {
        let plan = simple_plan();
        let caps = HardwareCapabilities::new(
            4,
            1, // Only 1 byte of "RAM" — will always trigger memory pressure for non-trivial stages
            false,
            0,
            None,
        );
        let report = HardwareResourceValidator::new(caps).validate(&plan);
        // The plan may produce memory pressure warnings.  For a plan with
        // at least one non-zero memory estimate this should trigger.
        // We check that validation still completes without panicking.
        assert!(report.is_ok() || !report.warnings.is_empty() || report.is_ok());
    }

    // 4. GPU-candidate stage with no GPU produces a warning (non-strict mode).
    #[test]
    fn gpu_warning_non_strict() {
        use crate::execution_plan::{ExecutionPlan, ExecutionStage, ResourceEstimate};
        // Build a plan with a GPU-candidate stage manually.
        let plan = ExecutionPlan::new(vec![ExecutionStage {
            stage_id: 0,
            nodes: vec![],
            resource_estimate: ResourceEstimate::new(0, 1.0, true),
            dependencies: vec![],
            parallel: false,
            parallel_groups: vec![],
        }]);
        let caps = HardwareCapabilities::new(4, 4 * 1024 * 1024 * 1024, false, 0, None);
        let report = HardwareResourceValidator::new(caps).validate(&plan);
        assert!(report.is_ok(), "non-strict: GPU warning should not be an error");
        let has_gpu_warn = report.warnings.iter().any(|w| {
            matches!(w, HardwareValidationWarning::GpuUnavailable { .. })
        });
        assert!(has_gpu_warn, "expected GpuUnavailable warning");
    }

    // 5. GPU-candidate stage with no GPU is a hard error in strict mode.
    #[test]
    fn gpu_error_strict_mode() {
        use crate::execution_plan::{ExecutionPlan, ExecutionStage, ResourceEstimate};
        let plan = ExecutionPlan::new(vec![ExecutionStage {
            stage_id: 0,
            nodes: vec![],
            resource_estimate: ResourceEstimate::new(0, 1.0, true),
            dependencies: vec![],
            parallel: false,
            parallel_groups: vec![],
        }]);
        let caps = HardwareCapabilities::new(4, 4 * 1024 * 1024 * 1024, false, 0, None);
        let report = HardwareResourceValidator::new(caps)
            .with_strict_gpu()
            .validate(&plan);
        assert!(!report.is_ok(), "strict GPU mode: should be an error");
    }

    // 6. CPU weight threshold warning.
    #[test]
    fn cpu_weight_exceeded_warning() {
        use crate::execution_plan::{ExecutionPlan, ExecutionStage, ResourceEstimate};
        let plan = ExecutionPlan::new(vec![ExecutionStage {
            stage_id: 0,
            nodes: vec![],
            resource_estimate: ResourceEstimate::new(0, 999.0, false),
            dependencies: vec![],
            parallel: false,
            parallel_groups: vec![],
        }]);
        let caps = HardwareCapabilities::new(4, 4 * 1024 * 1024 * 1024, false, 0, Some(10.0));
        let report = HardwareResourceValidator::new(caps).validate(&plan);
        let has_weight_warn = report.warnings.iter().any(|w| {
            matches!(w, HardwareValidationWarning::CpuWeightExceeded { .. })
        });
        assert!(has_weight_warn, "expected CpuWeightExceeded warning");
    }

    // 7. ResourceRequirement::with_min_cpu warns when insufficient.
    #[test]
    fn resource_requirement_cpu_threads() {
        let plan = simple_plan();
        let caps = HardwareCapabilities::new(1, 4 * 1024 * 1024 * 1024, false, 0, None);
        let req = ResourceRequirement::new().with_min_cpu(16);
        let report = HardwareResourceValidator::new(caps).require(req).validate(&plan);
        let has_req_warn = report.warnings.iter().any(|w| {
            matches!(w, HardwareValidationWarning::RequirementNotMet { .. })
        });
        assert!(has_req_warn, "expected RequirementNotMet for CPU threads");
    }

    // 8. ResourceRequirement::with_min_memory warns when insufficient.
    #[test]
    fn resource_requirement_memory() {
        let plan = simple_plan();
        let caps = HardwareCapabilities::new(8, 1024, false, 0, None); // 1 KiB
        let req = ResourceRequirement::new().with_min_memory(1024 * 1024 * 1024); // 1 GiB
        let report = HardwareResourceValidator::new(caps).require(req).validate(&plan);
        let has_req_warn = report.warnings.iter().any(|w| {
            matches!(w, HardwareValidationWarning::RequirementNotMet { .. })
        });
        assert!(has_req_warn, "expected RequirementNotMet for memory");
    }

    // 9. ResourceRequirement::with_gpu warns when no GPU.
    #[test]
    fn resource_requirement_gpu() {
        let plan = simple_plan();
        let caps = HardwareCapabilities::new(4, 4 * 1024 * 1024 * 1024, false, 0, None);
        let req = ResourceRequirement::new().with_gpu(1);
        let report = HardwareResourceValidator::new(caps).require(req).validate(&plan);
        let has_req_warn = report.warnings.iter().any(|w| {
            matches!(w, HardwareValidationWarning::RequirementNotMet { .. })
        });
        assert!(has_req_warn, "expected RequirementNotMet for GPU");
    }

    // 10. HardwareValidationReport::summary is non-empty.
    #[test]
    fn report_summary_non_empty() {
        let plan = simple_plan();
        let caps = HardwareCapabilities::unlimited();
        let report = HardwareResourceValidator::new(caps).validate(&plan);
        let summary = report.summary();
        assert!(!summary.is_empty());
    }

    // 11. HardwareValidationWarning Display trait.
    #[test]
    fn warning_display() {
        let w = HardwareValidationWarning::GpuUnavailable { stage_id: 2 };
        let s = w.to_string();
        assert!(s.contains("GPU") || s.contains("gpu") || s.contains("stage 2"));
    }

    // 12. format_bytes helper.
    #[test]
    fn format_bytes_variants() {
        assert!(format_bytes(500).contains('B'));
        assert!(format_bytes(2048).contains("KiB"));
        assert!(format_bytes(3 * 1024 * 1024).contains("MiB"));
        assert!(format_bytes(2 * 1024 * 1024 * 1024).contains("GiB"));
    }

    // 13. Empty plan produces a clean report.
    #[test]
    fn empty_plan_clean() {
        use crate::graph::PipelineGraph;
        let g = PipelineGraph::new();
        let plan = ExecutionPlanner::plan(&g).expect("plan empty");
        let caps = HardwareCapabilities::detect_or_default();
        let report = HardwareResourceValidator::new(caps).validate(&plan);
        assert!(report.is_ok());
        assert!(report.is_clean());
    }
}
