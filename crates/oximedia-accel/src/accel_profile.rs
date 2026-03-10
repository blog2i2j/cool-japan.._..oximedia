#![allow(dead_code)]
//! Acceleration target profiles for `oximedia-accel`.
//!
//! Defines the available compute targets (CPU, GPU, NPU) and a
//! `ProfileSelector` that chooses the best target given available hardware
//! capabilities and workload characteristics.

/// The hardware target for a compute operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccelTarget {
    /// Scalar CPU computation (always available).
    Cpu,
    /// GPU compute via Vulkan / CUDA / Metal.
    Gpu,
    /// Neural Processing Unit for ML-based operations.
    Npu,
}

impl AccelTarget {
    /// Returns a human-readable name for the target.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
            Self::Npu => "NPU",
        }
    }

    /// Returns `true` if this target performs massively-parallel work.
    #[must_use]
    pub fn is_parallel(&self) -> bool {
        matches!(self, Self::Gpu | Self::Npu)
    }
}

/// A profile describing the capabilities and preferences for a compute target.
#[derive(Debug, Clone)]
pub struct AccelProfile {
    /// The target hardware unit.
    pub target: AccelTarget,
    /// Estimated throughput in operations per second (normalised, `[0.0, 1.0]`).
    pub throughput_score: f32,
    /// Estimated memory bandwidth in GB/s.
    pub memory_bandwidth_gbs: f32,
    /// Power consumption estimate in watts.
    pub power_watts: f32,
    /// Whether this target is currently available on the host.
    pub available: bool,
}

impl AccelProfile {
    /// Create a new acceleration profile.
    #[must_use]
    pub fn new(
        target: AccelTarget,
        throughput_score: f32,
        memory_bandwidth_gbs: f32,
        power_watts: f32,
        available: bool,
    ) -> Self {
        Self {
            target,
            throughput_score: throughput_score.clamp(0.0, 1.0),
            memory_bandwidth_gbs,
            power_watts,
            available,
        }
    }

    /// Returns a default CPU profile (always available).
    #[must_use]
    pub fn cpu() -> Self {
        Self::new(AccelTarget::Cpu, 0.3, 50.0, 65.0, true)
    }

    /// Returns a typical mid-range GPU profile.
    #[must_use]
    pub fn gpu(available: bool) -> Self {
        Self::new(AccelTarget::Gpu, 0.9, 400.0, 150.0, available)
    }

    /// Returns a typical NPU profile.
    #[must_use]
    pub fn npu(available: bool) -> Self {
        Self::new(AccelTarget::Npu, 0.75, 120.0, 15.0, available)
    }

    /// Returns the efficiency score (throughput per watt).
    #[must_use]
    pub fn efficiency(&self) -> f32 {
        if self.power_watts > 0.0 {
            self.throughput_score / self.power_watts
        } else {
            0.0
        }
    }
}

/// Criteria used when selecting the best acceleration profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionCriteria {
    /// Maximise raw throughput.
    MaxThroughput,
    /// Minimise power consumption while meeting throughput requirements.
    PowerEfficient,
    /// Prefer the CPU (e.g. for deterministic unit tests).
    ForceCpu,
}

/// Selects the best `AccelProfile` from a pool of candidates.
#[derive(Debug)]
pub struct ProfileSelector {
    profiles: Vec<AccelProfile>,
    criteria: SelectionCriteria,
}

impl ProfileSelector {
    /// Create a selector with the given profiles and selection criteria.
    #[must_use]
    pub fn new(profiles: Vec<AccelProfile>, criteria: SelectionCriteria) -> Self {
        Self { profiles, criteria }
    }

    /// Build a selector with the default CPU + optional GPU/NPU profiles.
    #[must_use]
    pub fn auto(gpu_available: bool, npu_available: bool) -> Self {
        let profiles = vec![
            AccelProfile::cpu(),
            AccelProfile::gpu(gpu_available),
            AccelProfile::npu(npu_available),
        ];
        Self::new(profiles, SelectionCriteria::MaxThroughput)
    }

    /// Select the best available profile according to the configured criteria.
    ///
    /// Always falls back to CPU if no other target is available.
    #[must_use]
    pub fn select(&self) -> &AccelProfile {
        let available: Vec<&AccelProfile> = self.profiles.iter().filter(|p| p.available).collect();

        if available.is_empty() {
            // Safety net — CPU profile is always available.
            return self
                .profiles
                .iter()
                .find(|p| p.target == AccelTarget::Cpu)
                .unwrap_or(&self.profiles[0]);
        }

        match self.criteria {
            SelectionCriteria::MaxThroughput => available
                .iter()
                .copied()
                .max_by(|a, b| {
                    a.throughput_score
                        .partial_cmp(&b.throughput_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(available[0]),
            SelectionCriteria::PowerEfficient => available
                .iter()
                .copied()
                .max_by(|a, b| {
                    a.efficiency()
                        .partial_cmp(&b.efficiency())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(available[0]),
            SelectionCriteria::ForceCpu => available
                .iter()
                .copied()
                .find(|p| p.target == AccelTarget::Cpu)
                .unwrap_or(available[0]),
        }
    }

    /// Returns the number of registered profiles.
    #[must_use]
    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }

    /// Returns the number of currently available profiles.
    #[must_use]
    pub fn available_count(&self) -> usize {
        self.profiles.iter().filter(|p| p.available).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accel_target_name_cpu() {
        assert_eq!(AccelTarget::Cpu.name(), "CPU");
    }

    #[test]
    fn test_accel_target_name_gpu() {
        assert_eq!(AccelTarget::Gpu.name(), "GPU");
    }

    #[test]
    fn test_accel_target_name_npu() {
        assert_eq!(AccelTarget::Npu.name(), "NPU");
    }

    #[test]
    fn test_accel_target_is_parallel_gpu() {
        assert!(AccelTarget::Gpu.is_parallel());
    }

    #[test]
    fn test_accel_target_is_parallel_npu() {
        assert!(AccelTarget::Npu.is_parallel());
    }

    #[test]
    fn test_accel_target_cpu_not_parallel() {
        assert!(!AccelTarget::Cpu.is_parallel());
    }

    #[test]
    fn test_accel_profile_throughput_clamped() {
        let p = AccelProfile::new(AccelTarget::Cpu, 5.0, 50.0, 65.0, true);
        assert!(p.throughput_score <= 1.0);
    }

    #[test]
    fn test_accel_profile_cpu_always_available() {
        let p = AccelProfile::cpu();
        assert!(p.available);
        assert_eq!(p.target, AccelTarget::Cpu);
    }

    #[test]
    fn test_accel_profile_gpu_availability() {
        let p = AccelProfile::gpu(false);
        assert!(!p.available);
    }

    #[test]
    fn test_accel_profile_efficiency() {
        let p = AccelProfile::new(AccelTarget::Gpu, 0.9, 400.0, 150.0, true);
        let expected = 0.9 / 150.0;
        assert!((p.efficiency() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_profile_selector_max_throughput_picks_gpu() {
        let sel = ProfileSelector::auto(true, false);
        let best = sel.select();
        assert_eq!(best.target, AccelTarget::Gpu);
    }

    #[test]
    fn test_profile_selector_force_cpu() {
        let profiles = vec![AccelProfile::cpu(), AccelProfile::gpu(true)];
        let sel = ProfileSelector::new(profiles, SelectionCriteria::ForceCpu);
        assert_eq!(sel.select().target, AccelTarget::Cpu);
    }

    #[test]
    fn test_profile_selector_power_efficient_prefers_npu() {
        // NPU has lower power (15W) and decent throughput → higher efficiency.
        let profiles = vec![AccelProfile::cpu(), AccelProfile::npu(true)];
        let sel = ProfileSelector::new(profiles, SelectionCriteria::PowerEfficient);
        assert_eq!(sel.select().target, AccelTarget::Npu);
    }

    #[test]
    fn test_profile_selector_available_count() {
        let sel = ProfileSelector::auto(true, true);
        assert_eq!(sel.available_count(), 3);
    }

    #[test]
    fn test_profile_selector_unavailable_gpu_falls_back() {
        let sel = ProfileSelector::auto(false, false);
        assert_eq!(sel.select().target, AccelTarget::Cpu);
    }

    #[test]
    fn test_profile_count() {
        let sel = ProfileSelector::auto(true, false);
        assert_eq!(sel.profile_count(), 3);
    }

    #[test]
    fn test_selection_criteria_force_cpu_with_no_cpu_profile() {
        // Should fall back to first available.
        let profiles = vec![AccelProfile::gpu(true)];
        let sel = ProfileSelector::new(profiles, SelectionCriteria::ForceCpu);
        let result = sel.select();
        assert_eq!(result.target, AccelTarget::Gpu);
    }
}
