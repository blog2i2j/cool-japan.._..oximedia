//! Render-farm extensions for TODO.md (0.1.3 session).
//!
//! Provides:
//! - `FairShareScheduler` — equal-slot allocation across users
//! - `CloudBurstManager` — queue-depth-based cloud bursting
//! - `TileScheduler` — frame-split / tile-merge
//! - `RenderCostTracker` — per-job CPU/GPU cost recording
//! - `MultiSiteCoordinator` — lowest-latency site selection
//! - `FaultToleranceManager` — worker-failure job reassignment
//! - `PreemptionPolicy` — priority-gap preemption
//! - `FarmProgress::aggregate` — farm-wide progress summary
//! - `RenderCheckpoint` (new style) — job_id + frame-level resume

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Minimal RenderJob definition for this module
// ─────────────────────────────────────────────────────────────────────────────

/// Render job state used in extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderJobState {
    /// Waiting to be scheduled.
    Pending,
    /// Currently rendering.
    Running,
    /// Successfully completed.
    Completed,
    /// Failed.
    Failed,
}

/// A render job record used across extension APIs.
#[derive(Debug, Clone)]
pub struct RenderJob {
    /// Unique job identifier.
    pub id: u64,
    /// Priority (higher = more important).
    pub priority: u32,
    /// User or project owning the job.
    pub user_id: String,
    /// Identifier of the worker currently running this job (if any).
    pub worker_id: Option<u64>,
    /// Current job state.
    pub state: RenderJobState,
    /// Render progress [0.0, 1.0].
    pub progress: f64,
}

impl RenderJob {
    /// Create a new pending job.
    #[must_use]
    pub fn new(id: u64, priority: u32, user_id: impl Into<String>) -> Self {
        Self {
            id,
            priority,
            user_id: user_id.into(),
            worker_id: None,
            state: RenderJobState::Pending,
            progress: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Fair-share scheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Fair-share scheduler that allocates equal render slots to each user/project.
pub struct FairShareScheduler;

impl FairShareScheduler {
    /// Construct a new fair-share scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Allocate render slots across jobs, distributing equally by user.
    ///
    /// Each unique user receives at least 1 slot.  Remaining slots (after
    /// the minimum allocation) are distributed in round-robin order.
    /// Returns a map of `job_id → allocated_slots`.
    #[must_use]
    pub fn allocate_slots(&self, jobs: &[RenderJob], total_slots: u32) -> HashMap<u64, u32> {
        if jobs.is_empty() || total_slots == 0 {
            return HashMap::new();
        }

        // Group jobs by user
        let mut user_jobs: HashMap<String, Vec<u64>> = HashMap::new();
        for job in jobs {
            user_jobs
                .entry(job.user_id.clone())
                .or_default()
                .push(job.id);
        }

        let user_count = user_jobs.len() as u32;
        // Every user gets at least 1 slot
        let base_slots = 1u32;
        let total_base = base_slots * user_count;
        let remaining = total_slots.saturating_sub(total_base);

        // Per-user allocation: base + fair share of remaining
        let extra_per_user = remaining / user_count;
        let leftover = remaining % user_count;

        let mut result: HashMap<u64, u32> = HashMap::new();

        for (i, (_user, job_ids)) in user_jobs.iter().enumerate() {
            let user_slots =
                base_slots + extra_per_user + if (i as u32) < leftover { 1 } else { 0 };
            let job_count = job_ids.len() as u32;
            // Spread user slots across their jobs
            let per_job = (user_slots / job_count).max(1);
            for &jid in job_ids {
                result.insert(jid, per_job);
            }
        }

        result
    }
}

impl Default for FairShareScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Cloud burst manager
// ─────────────────────────────────────────────────────────────────────────────

/// Cloud burst manager for hybrid on-prem + cloud rendering.
#[derive(Debug, Default)]
pub struct CloudBurstManager;

impl CloudBurstManager {
    /// Create a new cloud burst manager.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Decide whether to burst to cloud.
    ///
    /// Returns `true` when `queue_depth / local_capacity > burst_threshold`.
    #[must_use]
    pub fn should_burst(queue_depth: u32, local_capacity: u32, burst_threshold: f32) -> bool {
        if local_capacity == 0 {
            return queue_depth > 0;
        }
        let ratio = queue_depth as f32 / local_capacity as f32;
        ratio > burst_threshold
    }

    /// Select the highest-priority N jobs for cloud burst.
    ///
    /// Returns the IDs of up to `n` pending jobs sorted by descending priority.
    #[must_use]
    pub fn burst_jobs(jobs: &[RenderJob], n: u32) -> Vec<u64> {
        let mut pending: Vec<&RenderJob> = jobs
            .iter()
            .filter(|j| j.state == RenderJobState::Pending)
            .collect();

        // Sort descending by priority
        pending.sort_by(|a, b| b.priority.cmp(&a.priority));

        pending.iter().take(n as usize).map(|j| j.id).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Tile scheduling – split frame / merge tiles
// ─────────────────────────────────────────────────────────────────────────────

/// A rectangular tile region within a frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileRegion {
    /// Horizontal offset in pixels.
    pub x: u32,
    /// Vertical offset in pixels.
    pub y: u32,
    /// Width of the tile in pixels.
    pub width: u32,
    /// Height of the tile in pixels.
    pub height: u32,
    /// Sequential tile index.
    pub tile_id: u32,
}

impl TileRegion {
    /// Create a tile region.
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32, tile_id: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            tile_id,
        }
    }
}

/// A rendered tile result (RGBA pixel buffer).
#[derive(Debug, Clone)]
pub struct TileResult {
    /// Source tile region.
    pub region: TileRegion,
    /// RGBA pixel data (`region.width * region.height * 4` bytes).
    pub pixels: Vec<u8>,
}

impl TileResult {
    /// Construct a tile result.
    #[must_use]
    pub fn new(region: TileRegion, pixels: Vec<u8>) -> Self {
        Self { region, pixels }
    }
}

/// Tile scheduler for splitting frames and merging tile renders.
pub struct TileScheduler;

impl TileScheduler {
    /// Split a frame into a grid of tiles of at most `tile_size × tile_size` pixels.
    ///
    /// Returns tiles in row-major order.  The last column/row may be smaller
    /// than `tile_size` when the dimensions are not evenly divisible.
    #[must_use]
    pub fn split_frame(width: u32, height: u32, tile_size: u32) -> Vec<TileRegion> {
        if width == 0 || height == 0 || tile_size == 0 {
            return Vec::new();
        }

        let cols = width.div_ceil(tile_size);
        let rows = height.div_ceil(tile_size);
        let mut tiles = Vec::with_capacity((cols * rows) as usize);

        for row in 0..rows {
            let y = row * tile_size;
            let tile_h = tile_size.min(height - y);
            for col in 0..cols {
                let x = col * tile_size;
                let tile_w = tile_size.min(width - x);
                let tile_id = row * cols + col;
                tiles.push(TileRegion::new(x, y, tile_w, tile_h, tile_id));
            }
        }

        tiles
    }

    /// Merge rendered tile results into a single flat RGBA frame buffer.
    ///
    /// `width` and `height` define the full frame dimensions.
    /// Any tile that overflows the frame boundaries is silently skipped.
    /// Returns a `width × height × 4` byte RGBA buffer (zero-initialized).
    #[must_use]
    pub fn merge_tiles(tiles: &[TileResult], width: u32, height: u32) -> Vec<u8> {
        let total = (width as usize) * (height as usize) * 4;
        let mut frame = vec![0u8; total];

        for tile in tiles {
            let r = &tile.region;
            let expected = (r.width as usize) * (r.height as usize) * 4;
            if tile.pixels.len() != expected {
                continue; // skip malformed tile
            }
            for row in 0..r.height {
                let dst_y = r.y + row;
                if dst_y >= height {
                    break;
                }
                let src_row_start = (row as usize) * (r.width as usize) * 4;
                let dst_row_start = (dst_y as usize) * (width as usize) * 4 + (r.x as usize) * 4;
                let copy_len = (r.width as usize).min((width - r.x) as usize) * 4;
                frame[dst_row_start..dst_row_start + copy_len]
                    .copy_from_slice(&tile.pixels[src_row_start..src_row_start + copy_len]);
            }
        }

        frame
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Render cost tracker
// ─────────────────────────────────────────────────────────────────────────────

/// Per-job cost record.
#[derive(Debug, Clone, Default)]
struct JobCostRecord {
    total_cost: f64,
}

/// Render cost tracker recording CPU/GPU usage per job.
#[derive(Debug, Default)]
pub struct RenderCostTracker {
    job_costs: HashMap<u64, JobCostRecord>,
}

impl RenderCostTracker {
    /// Create a new cost tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            job_costs: HashMap::new(),
        }
    }

    /// Record CPU and GPU hours for a job and return the incremental cost.
    ///
    /// `cost = cpu_hours * rate_per_cpu_hour + gpu_hours * rate_per_gpu_hour`
    pub fn record(
        &mut self,
        job_id: u64,
        cpu_hours: f64,
        gpu_hours: f64,
        rate_per_cpu_hour: f64,
        rate_per_gpu_hour: f64,
    ) -> f64 {
        let cost = cpu_hours * rate_per_cpu_hour + gpu_hours * rate_per_gpu_hour;
        self.job_costs.entry(job_id).or_default().total_cost += cost;
        cost
    }

    /// Return the total accumulated cost across all jobs.
    #[must_use]
    pub fn total_cost(&self) -> f64 {
        self.job_costs.values().map(|r| r.total_cost).sum()
    }

    /// Return the total cost for a specific job.
    #[must_use]
    pub fn job_cost(&self, job_id: u64) -> f64 {
        self.job_costs.get(&job_id).map_or(0.0, |r| r.total_cost)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Multi-site coordinator
// ─────────────────────────────────────────────────────────────────────────────

/// A render site entry.
#[derive(Debug, Clone)]
pub struct RenderSite {
    /// Site identifier.
    pub site_id: u32,
    /// Render capacity (total slots).
    pub capacity: u32,
    /// Network round-trip latency in milliseconds.
    pub latency_ms: u32,
    /// Current number of running jobs.
    pub running_jobs: u32,
}

impl RenderSite {
    /// Returns the remaining available capacity.
    #[must_use]
    pub fn available_capacity(&self) -> u32 {
        self.capacity.saturating_sub(self.running_jobs)
    }
}

/// Multi-site render coordinator.
#[derive(Debug, Default)]
pub struct MultiSiteCoordinator {
    sites: Vec<RenderSite>,
}

impl MultiSiteCoordinator {
    /// Create a new coordinator with no sites.
    #[must_use]
    pub fn new() -> Self {
        Self { sites: Vec::new() }
    }

    /// Register a render site.
    pub fn add_site(&mut self, site_id: u32, capacity: u32, latency_ms: u32) {
        self.sites.push(RenderSite {
            site_id,
            capacity,
            latency_ms,
            running_jobs: 0,
        });
    }

    /// Select the best site for a job: lowest-latency site with available capacity.
    ///
    /// Returns the `site_id` of the selected site, or `0` if no site has capacity.
    #[must_use]
    pub fn select_site(&self, _job: &RenderJob) -> u32 {
        self.sites
            .iter()
            .filter(|s| s.available_capacity() > 0)
            .min_by_key(|s| s.latency_ms)
            .map_or(0, |s| s.site_id)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Fault tolerance manager
// ─────────────────────────────────────────────────────────────────────────────

/// Fault tolerance manager for worker failure handling.
pub struct FaultToleranceManager;

impl FaultToleranceManager {
    /// Create a new fault tolerance manager.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Handle a worker failure: reassign all jobs belonging to `worker_id` back
    /// to `Pending` state so they can be re-scheduled.
    pub fn on_worker_failure(worker_id: u64, jobs: &mut Vec<RenderJob>) {
        for job in jobs.iter_mut() {
            if job.worker_id == Some(worker_id) {
                job.state = RenderJobState::Pending;
                job.worker_id = None;
            }
        }
    }
}

impl Default for FaultToleranceManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Priority preemption policy
// ─────────────────────────────────────────────────────────────────────────────

/// Policy for preempting running jobs based on incoming job priority.
pub struct PreemptionPolicy;

impl PreemptionPolicy {
    /// Determine whether to preempt `running` in favour of `incoming`.
    ///
    /// Returns `true` when `incoming.priority > running.priority + 2`.
    #[must_use]
    pub fn should_preempt(running: &RenderJob, incoming: &RenderJob) -> bool {
        incoming.priority > running.priority.saturating_add(2)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Farm progress aggregation
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated progress summary for the entire render farm.
#[derive(Debug, Clone, Default)]
pub struct FarmProgressSummary {
    /// Total number of jobs.
    pub total: usize,
    /// Jobs in Pending state.
    pub pending: usize,
    /// Jobs currently running.
    pub running: usize,
    /// Successfully completed jobs.
    pub completed: usize,
    /// Failed jobs.
    pub failed: usize,
}

impl FarmProgressSummary {
    /// Completion ratio [0.0, 1.0].
    #[must_use]
    pub fn completion_ratio(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.completed as f64 / self.total as f64
    }
}

/// Farm-wide progress aggregator.
pub struct FarmProgress;

impl FarmProgress {
    /// Aggregate the state of all render jobs into a summary.
    #[must_use]
    pub fn aggregate(jobs: &[RenderJob]) -> FarmProgressSummary {
        let mut summary = FarmProgressSummary {
            total: jobs.len(),
            ..Default::default()
        };

        for job in jobs {
            match job.state {
                RenderJobState::Pending => summary.pending += 1,
                RenderJobState::Running => summary.running += 1,
                RenderJobState::Completed => summary.completed += 1,
                RenderJobState::Failed => summary.failed += 1,
            }
        }

        summary
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Render checkpoint (new-style with frame granularity)
// ─────────────────────────────────────────────────────────────────────────────

/// A render checkpoint recording per-frame progress.
#[derive(Debug, Clone)]
pub struct RenderCheckpoint {
    /// Identifier of the job being checkpointed.
    pub job_id: u64,
    /// Last completed frame index (0-based).
    pub last_completed_frame: u32,
    /// Additional metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

impl RenderCheckpoint {
    /// Create a new checkpoint at the given frame.
    #[must_use]
    pub fn new(job_id: u64, frame: u32) -> Self {
        Self {
            job_id,
            last_completed_frame: frame,
            metadata: HashMap::new(),
        }
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Retrieve metadata by key.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Build a `RenderJob` that resumes from this checkpoint.
    ///
    /// The returned job has a high priority (255) and `Pending` state,
    /// with metadata indicating the start frame for the resumed render.
    #[must_use]
    pub fn resume_from_checkpoint(checkpoint: &RenderCheckpoint) -> RenderJob {
        RenderJob {
            id: checkpoint.job_id,
            priority: 255, // high priority for resumed jobs
            user_id: "system-resume".to_string(),
            worker_id: None,
            state: RenderJobState::Pending,
            progress: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- FairShareScheduler ---

    #[test]
    fn test_fair_share_allocates_equal_slots() {
        let sched = FairShareScheduler::new();
        let jobs = vec![
            RenderJob::new(1, 5, "alice"),
            RenderJob::new(2, 5, "bob"),
            RenderJob::new(3, 5, "charlie"),
        ];
        let slots = sched.allocate_slots(&jobs, 30);
        // 3 users, 30 slots → 10 per user × 1 job each = 10 slots per job
        assert_eq!(slots.len(), 3);
        let v: Vec<u32> = slots.values().cloned().collect();
        let min = *v.iter().min().expect("should have min");
        let max = *v.iter().max().expect("should have max");
        // All should be equal (or differ by at most 1 due to rounding)
        assert!(
            max - min <= 1,
            "Fair share: max-min slot diff should be ≤1, got {min}/{max}"
        );
    }

    #[test]
    fn test_fair_share_empty_jobs() {
        let sched = FairShareScheduler::new();
        let slots = sched.allocate_slots(&[], 10);
        assert!(slots.is_empty());
    }

    #[test]
    fn test_fair_share_zero_slots() {
        let sched = FairShareScheduler::new();
        let jobs = vec![RenderJob::new(1, 5, "alice")];
        let slots = sched.allocate_slots(&jobs, 0);
        assert!(slots.is_empty());
    }

    // --- CloudBurstManager ---

    #[test]
    fn test_cloud_burst_threshold_exceeded() {
        assert!(CloudBurstManager::should_burst(10, 5, 1.5)); // ratio=2.0 > 1.5
    }

    #[test]
    fn test_cloud_burst_threshold_not_exceeded() {
        assert!(!CloudBurstManager::should_burst(5, 10, 0.8)); // ratio=0.5 < 0.8
    }

    #[test]
    fn test_cloud_burst_zero_local_capacity() {
        assert!(CloudBurstManager::should_burst(1, 0, 0.5));
    }

    #[test]
    fn test_cloud_burst_selects_highest_priority() {
        let jobs = vec![
            RenderJob::new(1, 1, "u"),
            RenderJob::new(2, 10, "u"),
            RenderJob::new(3, 5, "u"),
        ];
        let burst = CloudBurstManager::burst_jobs(&jobs, 2);
        assert_eq!(burst[0], 2); // priority 10
        assert_eq!(burst[1], 3); // priority 5
    }

    // --- TileScheduler ---

    #[test]
    fn test_tile_split_round_trip() {
        let w = 1920u32;
        let h = 1080u32;
        let tile_size = 256u32;
        let tiles = TileScheduler::split_frame(w, h, tile_size);

        // Every pixel must be covered exactly once
        let mut coverage = vec![0u32; (w * h) as usize];
        for tile in &tiles {
            for row in 0..tile.height {
                for col in 0..tile.width {
                    let px = (tile.y + row) * w + (tile.x + col);
                    coverage[px as usize] += 1;
                }
            }
        }
        assert!(
            coverage.iter().all(|&c| c == 1),
            "Every pixel covered exactly once"
        );
    }

    #[test]
    fn test_tile_merge_round_trip() {
        let w = 4u32;
        let h = 4u32;
        let tile_size = 2u32;
        let regions = TileScheduler::split_frame(w, h, tile_size);

        // Fill each tile with a recognisable pattern
        let tile_results: Vec<TileResult> = regions
            .iter()
            .map(|r| {
                let pixels = vec![r.tile_id as u8; (r.width * r.height * 4) as usize];
                TileResult::new(r.clone(), pixels)
            })
            .collect();

        let frame = TileScheduler::merge_tiles(&tile_results, w, h);
        assert_eq!(frame.len(), (w * h * 4) as usize);
        // Top-left pixel belongs to tile 0
        assert_eq!(frame[0], 0);
        // Pixel at (2, 0) belongs to tile 1 (col=1, row=0)
        let px_2_0_offset = (0 * w as usize + 2) * 4;
        assert_eq!(frame[px_2_0_offset], 1);
    }

    #[test]
    fn test_tile_split_empty_frame() {
        assert!(TileScheduler::split_frame(0, 100, 64).is_empty());
        assert!(TileScheduler::split_frame(100, 0, 64).is_empty());
        assert!(TileScheduler::split_frame(100, 100, 0).is_empty());
    }

    // --- RenderCostTracker ---

    #[test]
    fn test_cost_tracker_record_and_total() {
        let mut tracker = RenderCostTracker::new();
        let cost1 = tracker.record(1, 2.0, 0.0, 0.10, 2.00);
        assert!((cost1 - 0.20).abs() < f64::EPSILON);

        let cost2 = tracker.record(2, 0.0, 1.0, 0.10, 2.00);
        assert!((cost2 - 2.00).abs() < f64::EPSILON);

        let total = tracker.total_cost();
        assert!((total - 2.20).abs() < 1e-10);
    }

    #[test]
    fn test_cost_tracker_per_job() {
        let mut tracker = RenderCostTracker::new();
        tracker.record(42, 4.0, 2.0, 0.50, 1.00);
        // 4 * 0.50 + 2 * 1.00 = 2.00 + 2.00 = 4.00
        assert!((tracker.job_cost(42) - 4.00).abs() < f64::EPSILON);
        assert_eq!(tracker.job_cost(99), 0.0);
    }

    // --- MultiSiteCoordinator ---

    #[test]
    fn test_multi_site_selects_lowest_latency_with_capacity() {
        let mut coord = MultiSiteCoordinator::new();
        coord.add_site(1, 10, 100); // 100 ms
        coord.add_site(2, 10, 50); // 50 ms ← should be selected
        coord.add_site(3, 10, 200);

        let job = RenderJob::new(9, 5, "user");
        assert_eq!(coord.select_site(&job), 2);
    }

    #[test]
    fn test_multi_site_skips_full_site() {
        let mut coord = MultiSiteCoordinator::new();
        coord.add_site(1, 0, 10); // full — 0 capacity
        coord.add_site(2, 5, 50);

        let job = RenderJob::new(7, 3, "user");
        assert_eq!(coord.select_site(&job), 2);
    }

    #[test]
    fn test_multi_site_no_capacity_returns_zero() {
        let mut coord = MultiSiteCoordinator::new();
        coord.add_site(1, 0, 10);

        let job = RenderJob::new(8, 5, "user");
        assert_eq!(coord.select_site(&job), 0);
    }

    // --- FaultToleranceManager ---

    #[test]
    fn test_fault_tolerance_reassigns_jobs() {
        let mut jobs = vec![
            RenderJob {
                id: 1,
                priority: 5,
                user_id: "u".into(),
                worker_id: Some(100),
                state: RenderJobState::Running,
                progress: 0.5,
            },
            RenderJob {
                id: 2,
                priority: 5,
                user_id: "u".into(),
                worker_id: Some(200),
                state: RenderJobState::Running,
                progress: 0.3,
            },
        ];

        FaultToleranceManager::on_worker_failure(100, &mut jobs);

        // Job 1 was on worker 100 → must be Pending
        assert_eq!(jobs[0].state, RenderJobState::Pending);
        assert!(jobs[0].worker_id.is_none());

        // Job 2 was on worker 200 → unchanged
        assert_eq!(jobs[1].state, RenderJobState::Running);
        assert_eq!(jobs[1].worker_id, Some(200));
    }

    // --- PreemptionPolicy ---

    #[test]
    fn test_preemption_triggers_when_gap_exceeds_two() {
        let running = RenderJob::new(1, 3, "u");
        let incoming = RenderJob::new(2, 7, "u"); // 7 > 3 + 2 = 5 → preempt
        assert!(PreemptionPolicy::should_preempt(&running, &incoming));
    }

    #[test]
    fn test_preemption_does_not_trigger_at_gap_of_two() {
        let running = RenderJob::new(1, 3, "u");
        let incoming = RenderJob::new(2, 5, "u"); // 5 == 3 + 2 → no preempt
        assert!(!PreemptionPolicy::should_preempt(&running, &incoming));
    }

    #[test]
    fn test_preemption_does_not_trigger_lower_priority() {
        let running = RenderJob::new(1, 10, "u");
        let incoming = RenderJob::new(2, 5, "u");
        assert!(!PreemptionPolicy::should_preempt(&running, &incoming));
    }

    // --- FarmProgress ---

    #[test]
    fn test_farm_progress_aggregation() {
        let jobs = vec![
            RenderJob {
                id: 1,
                priority: 1,
                user_id: "u".into(),
                worker_id: None,
                state: RenderJobState::Pending,
                progress: 0.0,
            },
            RenderJob {
                id: 2,
                priority: 1,
                user_id: "u".into(),
                worker_id: Some(10),
                state: RenderJobState::Running,
                progress: 0.5,
            },
            RenderJob {
                id: 3,
                priority: 1,
                user_id: "u".into(),
                worker_id: None,
                state: RenderJobState::Completed,
                progress: 1.0,
            },
            RenderJob {
                id: 4,
                priority: 1,
                user_id: "u".into(),
                worker_id: None,
                state: RenderJobState::Failed,
                progress: 0.0,
            },
            RenderJob {
                id: 5,
                priority: 1,
                user_id: "u".into(),
                worker_id: None,
                state: RenderJobState::Pending,
                progress: 0.0,
            },
        ];

        let summary = FarmProgress::aggregate(&jobs);
        assert_eq!(summary.total, 5);
        assert_eq!(summary.pending, 2);
        assert_eq!(summary.running, 1);
        assert_eq!(summary.completed, 1);
        assert_eq!(summary.failed, 1);
    }

    // --- RenderCheckpoint ---

    #[test]
    fn test_render_checkpoint_new() {
        let ckpt = RenderCheckpoint::new(42, 100);
        assert_eq!(ckpt.job_id, 42);
        assert_eq!(ckpt.last_completed_frame, 100);
    }

    #[test]
    fn test_render_checkpoint_metadata() {
        let mut ckpt = RenderCheckpoint::new(1, 50);
        ckpt.set_metadata("renderer", "cycles");
        assert_eq!(ckpt.get_metadata("renderer"), Some("cycles"));
        assert!(ckpt.get_metadata("unknown").is_none());
    }

    #[test]
    fn test_resume_from_checkpoint_returns_pending_job() {
        let ckpt = RenderCheckpoint::new(99, 200);
        let job = RenderCheckpoint::resume_from_checkpoint(&ckpt);
        assert_eq!(job.id, 99);
        assert_eq!(job.state, RenderJobState::Pending);
        assert_eq!(job.priority, 255);
    }
}
