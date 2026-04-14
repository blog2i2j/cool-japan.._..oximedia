//! Job-to-worker affinity rules.
//!
//! [`JobAffinityRule`] specifies a single capability that a worker must
//! possess before a job can be assigned to it.  Multiple rules can be combined
//! by the scheduler to express complex requirements such as "needs GPU *and*
//! AV1 codec support".
//!
//! # Example
//!
//! ```
//! use oximedia_farm::affinity::JobAffinityRule;
//! use oximedia_farm::capabilities::WorkerCapabilities;
//!
//! let rule = JobAffinityRule::new("av1");
//!
//! let mut worker = WorkerCapabilities::new(1);
//! worker.add_codec("av1");
//!
//! assert!(rule.matches(&worker));
//!
//! let mut other_worker = WorkerCapabilities::new(2);
//! other_worker.add_codec("h264");
//! assert!(!rule.matches(&other_worker));
//! ```

use crate::capabilities::WorkerCapabilities;

/// A single capability requirement that constrains job-to-worker placement.
///
/// A rule is satisfied when the candidate worker's [`WorkerCapabilities`]
/// contains the required capability (case-insensitive comparison).
#[derive(Debug, Clone)]
pub struct JobAffinityRule {
    /// The capability string that must be present on a compatible worker.
    required_capability: String,
}

impl JobAffinityRule {
    /// Create a new affinity rule requiring `required_capability`.
    ///
    /// The capability string is normalised to lowercase so matching is
    /// case-insensitive.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_farm::affinity::JobAffinityRule;
    ///
    /// let rule = JobAffinityRule::new("NVIDIA-A100");
    /// assert_eq!(rule.required_capability(), "nvidia-a100");
    /// ```
    #[must_use]
    pub fn new(required_capability: &str) -> Self {
        Self {
            required_capability: required_capability.to_lowercase(),
        }
    }

    /// Return the required capability string (normalised lowercase).
    #[must_use]
    pub fn required_capability(&self) -> &str {
        &self.required_capability
    }

    /// Test whether `worker` satisfies this affinity rule.
    ///
    /// Returns `true` when `worker` reports support for the required capability
    /// via [`WorkerCapabilities::supports`].
    #[must_use]
    pub fn matches(&self, worker: &WorkerCapabilities) -> bool {
        worker.supports(&self.required_capability)
    }

    /// Test whether ALL of the given rules are satisfied by `worker`.
    ///
    /// Convenience method for combining multiple rules with AND semantics.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_farm::affinity::JobAffinityRule;
    /// use oximedia_farm::capabilities::WorkerCapabilities;
    ///
    /// let rules = vec![
    ///     JobAffinityRule::new("av1"),
    ///     JobAffinityRule::new("nvidia-a100"),
    /// ];
    ///
    /// let mut w = WorkerCapabilities::new(1);
    /// w.add_codec("av1");
    /// w.add_gpu("nvidia-a100");
    ///
    /// assert!(JobAffinityRule::all_match(&rules, &w));
    /// ```
    #[must_use]
    pub fn all_match(rules: &[Self], worker: &WorkerCapabilities) -> bool {
        rules.iter().all(|r| r.matches(worker))
    }

    /// Test whether ANY of the given rules is satisfied by `worker`.
    ///
    /// Convenience method for combining multiple rules with OR semantics.
    #[must_use]
    pub fn any_match(rules: &[Self], worker: &WorkerCapabilities) -> bool {
        rules.iter().any(|r| r.matches(worker))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_worker(worker_id: u64, codec: &str) -> WorkerCapabilities {
        let mut w = WorkerCapabilities::new(worker_id);
        w.add_codec(codec);
        w
    }

    #[test]
    fn test_matches_when_capability_present() {
        let rule = JobAffinityRule::new("av1");
        let worker = make_worker(1, "av1");
        assert!(rule.matches(&worker));
    }

    #[test]
    fn test_no_match_when_capability_absent() {
        let rule = JobAffinityRule::new("vp9");
        let worker = make_worker(1, "h264");
        assert!(!rule.matches(&worker));
    }

    #[test]
    fn test_case_insensitive_rule_and_worker() {
        let rule = JobAffinityRule::new("AV1");
        let mut worker = WorkerCapabilities::new(1);
        worker.add_codec("av1"); // stored as lowercase
        assert!(rule.matches(&worker));
    }

    #[test]
    fn test_required_capability_normalised() {
        let rule = JobAffinityRule::new("NVENC");
        assert_eq!(rule.required_capability(), "nvenc");
    }

    #[test]
    fn test_all_match_true() {
        let rules = vec![JobAffinityRule::new("av1"), JobAffinityRule::new("h265")];
        let mut w = WorkerCapabilities::new(1);
        w.add_codec("av1");
        w.add_codec("h265");
        assert!(JobAffinityRule::all_match(&rules, &w));
    }

    #[test]
    fn test_all_match_false_missing_one() {
        let rules = vec![JobAffinityRule::new("av1"), JobAffinityRule::new("hevc-10bit")];
        let w = make_worker(1, "av1");
        assert!(!JobAffinityRule::all_match(&rules, &w));
    }

    #[test]
    fn test_any_match_true() {
        let rules = vec![JobAffinityRule::new("vp9"), JobAffinityRule::new("h264")];
        let w = make_worker(1, "h264");
        assert!(JobAffinityRule::any_match(&rules, &w));
    }

    #[test]
    fn test_any_match_false_all_missing() {
        let rules = vec![JobAffinityRule::new("vp9"), JobAffinityRule::new("av1")];
        let w = make_worker(1, "h264");
        assert!(!JobAffinityRule::any_match(&rules, &w));
    }

    #[test]
    fn test_empty_rules_all_match_true() {
        let rules: Vec<JobAffinityRule> = vec![];
        let w = WorkerCapabilities::new(1);
        // vacuously true
        assert!(JobAffinityRule::all_match(&rules, &w));
    }

    #[test]
    fn test_empty_rules_any_match_false() {
        let rules: Vec<JobAffinityRule> = vec![];
        let w = WorkerCapabilities::new(1);
        assert!(!JobAffinityRule::any_match(&rules, &w));
    }
}
