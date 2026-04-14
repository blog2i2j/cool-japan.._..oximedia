//! Task-to-worker affinity rules and placement constraints.
//!
//! This module provides a rich affinity system that goes beyond simple capability
//! matching.  Rules are split into two classes:
//!
//! - **Hard rules** (`Require*`, `ExcludeWorker`) — all must be satisfied or the
//!   worker is rejected outright.
//! - **Soft rules** (`Prefer*`, `CollocateWith`) — satisfaction increments the
//!   placement score; unsatisfied soft rules do *not* disqualify a worker.
//!
//! [`AffinityMatcher`] is stateless: it evaluates rules against a snapshot of
//! [`WorkerCapabilities`] at scheduling time.  The caller should re-run matching
//! whenever the candidate set or rules change.
//!
//! # Example
//!
//! ```
//! use oximedia_farm::task_affinity::{
//!     AffinityMatcher, AffinityPolicy, AffinityRule, WorkerCapabilities,
//! };
//!
//! let caps = WorkerCapabilities {
//!     worker_id: "gpu-node-1".into(),
//!     tags: vec!["gpu".into(), "high-mem".into()],
//!     memory_gb: 64.0,
//!     cpu_cores: 32,
//!     has_gpu: true,
//! };
//!
//! let policy = AffinityPolicy {
//!     rules: vec![
//!         AffinityRule::RequireGpu,
//!         AffinityRule::RequireMinMemoryGb(32.0),
//!         AffinityRule::PreferTag("high-mem".into()),
//!     ],
//!     fallback_any: false,
//! };
//!
//! assert!(AffinityMatcher::matches_hard(&caps, &policy.rules));
//! let score = AffinityMatcher::score(&caps, &policy.rules);
//! assert!(score > 0.0);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AffinityRule
// ---------------------------------------------------------------------------

/// A single placement rule that constrains or influences task-to-worker assignment.
///
/// Rules prefixed with `Require` are **hard** constraints: a worker that does not
/// satisfy every hard rule in a policy is excluded from candidacy.
///
/// Rules prefixed with `Prefer` and `CollocateWith` are **soft** constraints: they
/// contribute to the affinity score but never exclude a worker.
#[derive(Debug, Clone, PartialEq)]
pub enum AffinityRule {
    // ---- Hard rules --------------------------------------------------------
    /// Worker must carry the given tag in its [`WorkerCapabilities::tags`] list.
    RequireTag(String),
    /// Worker must have at least the specified amount of memory (GiB).
    RequireMinMemoryGb(f32),
    /// Worker must have at least the specified number of CPU cores.
    RequireMinCpuCores(u8),
    /// Worker must have GPU acceleration (`has_gpu == true`).
    RequireGpu,
    /// This specific worker ID must **not** be used.
    ExcludeWorker(String),

    // ---- Soft rules --------------------------------------------------------
    /// Boost the score of workers that carry the given tag.
    PreferTag(String),
    /// Boost the score of the worker that is running (or last ran) the named
    /// sibling task — useful for keeping related tasks on the same machine to
    /// reduce data transfer.  The caller is responsible for resolving which
    /// worker ID the named task was assigned to and inserting an
    /// [`AffinityRule::PreferTag`] or adjusting the score externally if needed.
    /// In this implementation the rule stores the sibling task name as a tag
    /// match hint.
    CollocateWith(String),
}

impl AffinityRule {
    /// Return `true` when the rule is a hard (must-satisfy) constraint.
    #[must_use]
    pub fn is_hard(&self) -> bool {
        matches!(
            self,
            Self::RequireTag(_)
                | Self::RequireMinMemoryGb(_)
                | Self::RequireMinCpuCores(_)
                | Self::RequireGpu
                | Self::ExcludeWorker(_)
        )
    }

    /// Return `true` when the rule is a soft (preference) constraint.
    #[must_use]
    pub fn is_soft(&self) -> bool {
        !self.is_hard()
    }
}

// ---------------------------------------------------------------------------
// WorkerCapabilities
// ---------------------------------------------------------------------------

/// Static capability advertisement for a single worker node.
///
/// Populated from the worker's registration heartbeat and refreshed as
/// resources change.
#[derive(Debug, Clone)]
pub struct WorkerCapabilities {
    /// Stable identifier for this worker.
    pub worker_id: String,
    /// Arbitrary string labels (e.g. `"gpu"`, `"high-mem"`, `"datacenter-a"`).
    pub tags: Vec<String>,
    /// Total available memory in gibibytes.
    pub memory_gb: f32,
    /// Number of logical CPU cores available for task execution.
    pub cpu_cores: u8,
    /// Whether the worker has at least one GPU available.
    pub has_gpu: bool,
}

impl WorkerCapabilities {
    /// Return `true` when `tag` is present in `self.tags` (case-insensitive).
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        let lower = tag.to_lowercase();
        self.tags.iter().any(|t| t.to_lowercase() == lower)
    }
}

// ---------------------------------------------------------------------------
// AffinityPolicy
// ---------------------------------------------------------------------------

/// A collection of affinity rules that governs task placement for a specific job
/// or task type.
#[derive(Debug, Clone, Default)]
pub struct AffinityPolicy {
    /// Ordered list of affinity rules.  Hard rules are evaluated before soft rules.
    pub rules: Vec<AffinityRule>,
    /// When `true`, if *no* worker satisfies the hard rules the scheduler may
    /// fall back to any available worker.  When `false` the job is held until a
    /// qualifying worker becomes available.
    pub fallback_any: bool,
}

impl AffinityPolicy {
    /// Create an empty policy (accept any worker, no preferences).
    #[must_use]
    pub fn permissive() -> Self {
        Self {
            rules: Vec::new(),
            fallback_any: true,
        }
    }

    /// Create a policy from a list of rules without fallback.
    #[must_use]
    pub fn strict(rules: Vec<AffinityRule>) -> Self {
        Self {
            rules,
            fallback_any: false,
        }
    }
}

// ---------------------------------------------------------------------------
// AffinityMatcher
// ---------------------------------------------------------------------------

/// Stateless helper that evaluates [`AffinityRule`]s against [`WorkerCapabilities`].
///
/// All methods are pure functions; no state is stored between calls.
pub struct AffinityMatcher;

impl AffinityMatcher {
    /// Return `true` when `worker` satisfies **all hard rules** in `rules`.
    ///
    /// Soft rules (`PreferTag`, `CollocateWith`) are ignored by this check.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_farm::task_affinity::{AffinityMatcher, AffinityRule, WorkerCapabilities};
    ///
    /// let worker = WorkerCapabilities {
    ///     worker_id: "w1".into(),
    ///     tags: vec!["gpu".into()],
    ///     memory_gb: 16.0,
    ///     cpu_cores: 8,
    ///     has_gpu: true,
    /// };
    ///
    /// let rules = vec![AffinityRule::RequireGpu, AffinityRule::RequireMinMemoryGb(8.0)];
    /// assert!(AffinityMatcher::matches_hard(&worker, &rules));
    /// ```
    #[must_use]
    pub fn matches_hard(worker: &WorkerCapabilities, rules: &[AffinityRule]) -> bool {
        for rule in rules {
            match rule {
                AffinityRule::RequireTag(tag) => {
                    if !worker.has_tag(tag) {
                        return false;
                    }
                }
                AffinityRule::RequireMinMemoryGb(min_gb) => {
                    if worker.memory_gb < *min_gb {
                        return false;
                    }
                }
                AffinityRule::RequireMinCpuCores(min_cores) => {
                    if worker.cpu_cores < *min_cores {
                        return false;
                    }
                }
                AffinityRule::RequireGpu => {
                    if !worker.has_gpu {
                        return false;
                    }
                }
                AffinityRule::ExcludeWorker(excluded_id) => {
                    if worker.worker_id == *excluded_id {
                        return false;
                    }
                }
                // Soft rules do not contribute to hard matching.
                AffinityRule::PreferTag(_) | AffinityRule::CollocateWith(_) => {}
            }
        }
        true
    }

    /// Compute a placement preference score in `[0.0, 1.0]` for `worker`.
    ///
    /// The score is the fraction of soft rules satisfied.  A score of `1.0`
    /// means all soft rules match; `0.0` means none match (or there are no soft
    /// rules).  Hard rules are ignored by this method.
    ///
    /// When there are no soft rules at all the method returns `0.5` so that
    /// workers are treated as equally preferred when no preferences are
    /// expressed.
    #[must_use]
    pub fn score(worker: &WorkerCapabilities, rules: &[AffinityRule]) -> f32 {
        let soft_rules: Vec<&AffinityRule> = rules.iter().filter(|r| r.is_soft()).collect();

        if soft_rules.is_empty() {
            // No preferences → neutral score.
            return 0.5;
        }

        let mut satisfied: u32 = 0;
        for rule in &soft_rules {
            match rule {
                AffinityRule::PreferTag(tag) => {
                    if worker.has_tag(tag) {
                        satisfied += 1;
                    }
                }
                AffinityRule::CollocateWith(sibling_tag) => {
                    // Treat the sibling name as a tag hint — the scheduler is
                    // expected to have tagged the target worker accordingly.
                    if worker.has_tag(sibling_tag) {
                        satisfied += 1;
                    }
                }
                // Hard rules are filtered out above.
                _ => {}
            }
        }

        satisfied as f32 / soft_rules.len() as f32
    }

    /// Filter `workers` to those that satisfy all hard rules, then sort the
    /// survivors by affinity score descending (best match first).
    ///
    /// Workers with equal scores retain their original relative order (stable
    /// sort).
    ///
    /// Returns an empty slice when no worker satisfies the hard constraints.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_farm::task_affinity::{AffinityMatcher, AffinityRule, WorkerCapabilities};
    ///
    /// let workers = vec![
    ///     WorkerCapabilities { worker_id: "w1".into(), tags: vec![], memory_gb: 8.0, cpu_cores: 4, has_gpu: false },
    ///     WorkerCapabilities { worker_id: "w2".into(), tags: vec!["fast".into()], memory_gb: 8.0, cpu_cores: 4, has_gpu: false },
    /// ];
    /// let rules = vec![AffinityRule::PreferTag("fast".into())];
    /// let ranked = AffinityMatcher::rank_workers(&workers, &rules);
    /// assert_eq!(ranked[0].worker_id, "w2");
    /// ```
    #[must_use]
    pub fn rank_workers<'a>(
        workers: &'a [WorkerCapabilities],
        rules: &[AffinityRule],
    ) -> Vec<&'a WorkerCapabilities> {
        let mut candidates: Vec<(f32, &'a WorkerCapabilities)> = workers
            .iter()
            .filter(|w| Self::matches_hard(w, rules))
            .map(|w| (Self::score(w, rules), w))
            .collect();

        // Stable descending sort by score.
        candidates.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        candidates.into_iter().map(|(_, w)| w).collect()
    }

    /// Build a score map keyed by worker ID, useful for diagnostics.
    ///
    /// Only workers that satisfy the hard rules are included.
    #[must_use]
    pub fn score_map(
        workers: &[WorkerCapabilities],
        rules: &[AffinityRule],
    ) -> HashMap<String, f32> {
        workers
            .iter()
            .filter(|w| Self::matches_hard(w, rules))
            .map(|w| (w.worker_id.clone(), Self::score(w, rules)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: &str, tags: &[&str], memory_gb: f32, cpu_cores: u8, has_gpu: bool) -> WorkerCapabilities {
        WorkerCapabilities {
            worker_id: id.into(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
            memory_gb,
            cpu_cores,
            has_gpu,
        }
    }

    // ---- Hard rule tests ---------------------------------------------------

    #[test]
    fn test_require_tag_satisfied() {
        let w = worker("w1", &["gpu", "fast"], 32.0, 16, true);
        let rules = vec![AffinityRule::RequireTag("gpu".into())];
        assert!(AffinityMatcher::matches_hard(&w, &rules));
    }

    #[test]
    fn test_require_tag_not_satisfied() {
        let w = worker("w1", &["cpu-only"], 32.0, 16, false);
        let rules = vec![AffinityRule::RequireTag("gpu".into())];
        assert!(!AffinityMatcher::matches_hard(&w, &rules));
    }

    #[test]
    fn test_require_tag_case_insensitive() {
        let w = worker("w1", &["GPU"], 32.0, 8, true);
        let rules = vec![AffinityRule::RequireTag("gpu".into())];
        assert!(AffinityMatcher::matches_hard(&w, &rules));
    }

    #[test]
    fn test_require_gpu_satisfied() {
        let w = worker("w1", &[], 16.0, 8, true);
        let rules = vec![AffinityRule::RequireGpu];
        assert!(AffinityMatcher::matches_hard(&w, &rules));
    }

    #[test]
    fn test_require_gpu_not_satisfied() {
        let w = worker("w1", &[], 16.0, 8, false);
        let rules = vec![AffinityRule::RequireGpu];
        assert!(!AffinityMatcher::matches_hard(&w, &rules));
    }

    #[test]
    fn test_require_min_memory_satisfied() {
        let w = worker("w1", &[], 64.0, 8, false);
        let rules = vec![AffinityRule::RequireMinMemoryGb(32.0)];
        assert!(AffinityMatcher::matches_hard(&w, &rules));
    }

    #[test]
    fn test_require_min_memory_not_satisfied() {
        let w = worker("w1", &[], 16.0, 8, false);
        let rules = vec![AffinityRule::RequireMinMemoryGb(32.0)];
        assert!(!AffinityMatcher::matches_hard(&w, &rules));
    }

    #[test]
    fn test_require_min_cpu_cores_satisfied() {
        let w = worker("w1", &[], 16.0, 16, false);
        let rules = vec![AffinityRule::RequireMinCpuCores(8)];
        assert!(AffinityMatcher::matches_hard(&w, &rules));
    }

    #[test]
    fn test_require_min_cpu_cores_not_satisfied() {
        let w = worker("w1", &[], 16.0, 4, false);
        let rules = vec![AffinityRule::RequireMinCpuCores(8)];
        assert!(!AffinityMatcher::matches_hard(&w, &rules));
    }

    #[test]
    fn test_exclude_worker_rejects_specific_id() {
        let w = worker("bad-node", &["gpu"], 64.0, 32, true);
        let rules = vec![AffinityRule::ExcludeWorker("bad-node".into())];
        assert!(!AffinityMatcher::matches_hard(&w, &rules));
    }

    #[test]
    fn test_exclude_worker_allows_other_ids() {
        let w = worker("good-node", &["gpu"], 64.0, 32, true);
        let rules = vec![AffinityRule::ExcludeWorker("bad-node".into())];
        assert!(AffinityMatcher::matches_hard(&w, &rules));
    }

    #[test]
    fn test_multiple_hard_rules_all_must_pass() {
        let w = worker("w1", &["gpu"], 64.0, 8, true);
        let rules = vec![
            AffinityRule::RequireGpu,
            AffinityRule::RequireMinMemoryGb(32.0),
            AffinityRule::RequireTag("gpu".into()),
        ];
        assert!(AffinityMatcher::matches_hard(&w, &rules));

        let rules_fail = vec![
            AffinityRule::RequireGpu,
            AffinityRule::RequireMinMemoryGb(128.0), // insufficient
        ];
        assert!(!AffinityMatcher::matches_hard(&w, &rules_fail));
    }

    #[test]
    fn test_no_rules_always_matches_hard() {
        let w = worker("w1", &[], 0.0, 0, false);
        assert!(AffinityMatcher::matches_hard(&w, &[]));
    }

    // ---- Soft rule / score tests -------------------------------------------

    #[test]
    fn test_prefer_tag_score_full() {
        let w = worker("w1", &["fast", "nvme"], 32.0, 8, false);
        let rules = vec![
            AffinityRule::PreferTag("fast".into()),
            AffinityRule::PreferTag("nvme".into()),
        ];
        let s = AffinityMatcher::score(&w, &rules);
        assert!((s - 1.0).abs() < f32::EPSILON, "expected 1.0, got {s}");
    }

    #[test]
    fn test_prefer_tag_score_partial() {
        let w = worker("w1", &["fast"], 32.0, 8, false);
        let rules = vec![
            AffinityRule::PreferTag("fast".into()),
            AffinityRule::PreferTag("nvme".into()),
        ];
        let s = AffinityMatcher::score(&w, &rules);
        assert!((s - 0.5).abs() < f32::EPSILON, "expected 0.5, got {s}");
    }

    #[test]
    fn test_prefer_tag_score_none() {
        let w = worker("w1", &[], 32.0, 8, false);
        let rules = vec![AffinityRule::PreferTag("fast".into())];
        let s = AffinityMatcher::score(&w, &rules);
        assert!((s - 0.0).abs() < f32::EPSILON, "expected 0.0, got {s}");
    }

    #[test]
    fn test_no_soft_rules_neutral_score() {
        let w = worker("w1", &[], 32.0, 8, true);
        let rules = vec![AffinityRule::RequireGpu];
        let s = AffinityMatcher::score(&w, &rules);
        assert!((s - 0.5).abs() < f32::EPSILON, "expected 0.5, got {s}");
    }

    #[test]
    fn test_collocate_with_scored_as_tag() {
        let w = worker("w1", &["job-abc"], 32.0, 8, false);
        let rules = vec![AffinityRule::CollocateWith("job-abc".into())];
        let s = AffinityMatcher::score(&w, &rules);
        assert!((s - 1.0).abs() < f32::EPSILON, "expected 1.0, got {s}");
    }

    // ---- Ranking tests -----------------------------------------------------

    #[test]
    fn test_rank_workers_hard_filter_first() {
        let workers = vec![
            worker("w1", &["gpu"], 64.0, 16, true),
            worker("w2", &[], 16.0, 4, false), // no GPU
        ];
        let rules = vec![AffinityRule::RequireGpu];
        let ranked = AffinityMatcher::rank_workers(&workers, &rules);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].worker_id, "w1");
    }

    #[test]
    fn test_rank_workers_sorted_by_score_desc() {
        let workers = vec![
            worker("low", &[], 32.0, 8, false),
            worker("high", &["fast", "nvme"], 32.0, 8, false),
            worker("mid", &["fast"], 32.0, 8, false),
        ];
        let rules = vec![
            AffinityRule::PreferTag("fast".into()),
            AffinityRule::PreferTag("nvme".into()),
        ];
        let ranked = AffinityMatcher::rank_workers(&workers, &rules);
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].worker_id, "high");
        assert_eq!(ranked[1].worker_id, "mid");
        assert_eq!(ranked[2].worker_id, "low");
    }

    #[test]
    fn test_rank_workers_empty_when_no_match() {
        let workers = vec![
            worker("w1", &[], 8.0, 4, false),
        ];
        let rules = vec![AffinityRule::RequireGpu];
        let ranked = AffinityMatcher::rank_workers(&workers, &rules);
        assert!(ranked.is_empty());
    }

    #[test]
    fn test_score_map_keys() {
        let workers = vec![
            worker("w1", &["gpu"], 64.0, 16, true),
            worker("w2", &[], 8.0, 4, false),
        ];
        let rules = vec![AffinityRule::RequireGpu];
        let map = AffinityMatcher::score_map(&workers, &rules);
        assert!(map.contains_key("w1"));
        assert!(!map.contains_key("w2"));
    }

    #[test]
    fn test_affinity_policy_permissive() {
        let policy = AffinityPolicy::permissive();
        assert!(policy.rules.is_empty());
        assert!(policy.fallback_any);
    }

    #[test]
    fn test_affinity_policy_strict() {
        let rules = vec![AffinityRule::RequireGpu];
        let policy = AffinityPolicy::strict(rules.clone());
        assert_eq!(policy.rules.len(), 1);
        assert!(!policy.fallback_any);
    }

    #[test]
    fn test_affinity_rule_classification() {
        assert!(AffinityRule::RequireTag("x".into()).is_hard());
        assert!(AffinityRule::RequireMinMemoryGb(1.0).is_hard());
        assert!(AffinityRule::RequireMinCpuCores(1).is_hard());
        assert!(AffinityRule::RequireGpu.is_hard());
        assert!(AffinityRule::ExcludeWorker("x".into()).is_hard());
        assert!(AffinityRule::PreferTag("x".into()).is_soft());
        assert!(AffinityRule::CollocateWith("x".into()).is_soft());
    }
}
