//! GPU compute pass management — pass types, buffer bindings, and pass queues.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Category of work that a compute pass performs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PassType {
    /// Video processing (real-time).
    Video,
    /// Audio processing (real-time).
    Audio,
    /// Still-image processing.
    Image,
    /// Post-processing effects.
    PostProcess,
}

impl PassType {
    /// Returns `true` for pass types that operate in real-time context.
    #[must_use]
    pub fn is_real_time(&self) -> bool {
        matches!(self, Self::Video | Self::Audio)
    }
}

/// A binding between a GPU buffer slot and a logical buffer.
#[derive(Debug, Clone)]
pub struct BufferBinding {
    /// The shader binding slot index.
    pub slot: u8,
    /// Size of the buffer in bytes.
    pub size_bytes: u32,
    /// Whether the binding is read-only (i.e. an input buffer).
    pub read_only: bool,
}

impl BufferBinding {
    /// Creates a new `BufferBinding`.
    #[must_use]
    pub fn new(slot: u8, size_bytes: u32, read_only: bool) -> Self {
        Self {
            slot,
            size_bytes,
            read_only,
        }
    }

    /// Returns `true` if this binding is an input (read-only) binding.
    #[must_use]
    pub fn is_input(&self) -> bool {
        self.read_only
    }

    /// Returns `true` if this binding is an output (writable) binding.
    #[must_use]
    pub fn is_output(&self) -> bool {
        !self.read_only
    }
}

/// A single compute pass with a name, type, buffer bindings, and dispatch dimensions.
#[derive(Debug)]
pub struct ComputePass {
    /// Human-readable name for debugging.
    pub name: String,
    /// The category of this pass.
    pub pass_type: PassType,
    /// Buffer bindings used by this pass.
    pub bindings: Vec<BufferBinding>,
    /// Workgroup dispatch dimensions (x, y, z).
    pub workgroups: (u32, u32, u32),
}

impl ComputePass {
    /// Creates a new `ComputePass` with no bindings and a default dispatch of (1, 1, 1).
    #[must_use]
    pub fn new(name: impl Into<String>, pt: PassType) -> Self {
        Self {
            name: name.into(),
            pass_type: pt,
            bindings: Vec::new(),
            workgroups: (1, 1, 1),
        }
    }

    /// Adds a read-only (input) buffer binding on the given slot.
    pub fn add_input_binding(&mut self, slot: u8, size: u32) {
        self.bindings.push(BufferBinding::new(slot, size, true));
    }

    /// Adds a writable (output) buffer binding on the given slot.
    pub fn add_output_binding(&mut self, slot: u8, size: u32) {
        self.bindings.push(BufferBinding::new(slot, size, false));
    }

    /// Total work items = workgroups.x × workgroups.y × workgroups.z.
    #[must_use]
    pub fn total_work_items(&self) -> u64 {
        u64::from(self.workgroups.0) * u64::from(self.workgroups.1) * u64::from(self.workgroups.2)
    }

    /// Returns the number of bindings attached to this pass.
    #[must_use]
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }
}

/// An ordered queue of [`ComputePass`] entries.
#[derive(Debug, Default)]
pub struct PassQueue {
    passes: Vec<ComputePass>,
}

impl PassQueue {
    /// Creates an empty `PassQueue`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a pass to the queue.
    pub fn add(&mut self, pass: ComputePass) {
        self.passes.push(pass);
    }

    /// Returns references to all passes whose type matches `pt`.
    #[must_use]
    pub fn passes_of_type(&self, pt: &PassType) -> Vec<&ComputePass> {
        self.passes.iter().filter(|p| &p.pass_type == pt).collect()
    }

    /// Total number of bindings across all passes.
    #[must_use]
    pub fn total_bindings(&self) -> usize {
        self.passes.iter().map(ComputePass::binding_count).sum()
    }

    /// Returns the number of passes in the queue.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }
}

// ---------------------------------------------------------------------------
// BatchedComputePass — batched GPU dispatch management
// ---------------------------------------------------------------------------

/// A single queued GPU compute dispatch command.
#[derive(Debug, Clone)]
pub struct DispatchCommand {
    /// Identifier of the compute pipeline this dispatch uses.
    pub pipeline_id: u64,
    /// Bind group index to set before dispatching.
    pub bind_group: u32,
    /// Number of workgroups along the X axis.
    pub dispatch_x: u32,
    /// Number of workgroups along the Y axis.
    pub dispatch_y: u32,
    /// Number of workgroups along the Z axis.
    pub dispatch_z: u32,
}

impl DispatchCommand {
    /// Create a new `DispatchCommand`.
    #[must_use]
    pub fn new(
        pipeline_id: u64,
        bind_group: u32,
        dispatch_x: u32,
        dispatch_y: u32,
        dispatch_z: u32,
    ) -> Self {
        Self {
            pipeline_id,
            bind_group,
            dispatch_x,
            dispatch_y,
            dispatch_z,
        }
    }
}

/// Accumulates compute dispatch commands and issues them in batches.
///
/// Batching reduces command-encoder overhead by coalescing small dispatches
/// and sorting them by pipeline so that pipeline switches are minimised.
///
/// When the number of pending commands reaches `max_batch_size`, an automatic
/// flush is triggered and the commands are sorted by `pipeline_id` before
/// being returned.
pub struct BatchedComputePass {
    pending: Vec<DispatchCommand>,
    max_batch_size: usize,
    /// Total number of commands that have been flushed across all batches.
    total_flushed: u64,
}

impl BatchedComputePass {
    /// Create a `BatchedComputePass` with the given `max_batch_size`.
    ///
    /// A `max_batch_size` of 0 is treated as 1 (each submit is auto-flushed).
    #[must_use]
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            pending: Vec::new(),
            max_batch_size: max_batch_size.max(1),
            total_flushed: 0,
        }
    }

    /// Submit a dispatch command.
    ///
    /// Returns `true` if an automatic flush was triggered (i.e., the pending
    /// queue reached `max_batch_size`).  The caller should retrieve the flushed
    /// batch via [`flush`][Self::flush] when this returns `true`.
    pub fn submit(&mut self, cmd: DispatchCommand) -> bool {
        self.pending.push(cmd);
        if self.pending.len() >= self.max_batch_size {
            // Auto-flush triggered; drain will happen on next flush() call.
            true
        } else {
            false
        }
    }

    /// Drain all pending commands, sorted by `pipeline_id` (ascending) to
    /// minimise pipeline state switches on the GPU.
    ///
    /// Returns the sorted batch.  If the queue is empty, returns an empty `Vec`.
    pub fn flush(&mut self) -> Vec<DispatchCommand> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let mut batch = std::mem::take(&mut self.pending);
        // Sort by pipeline_id so similar pipelines are adjacent.
        batch.sort_by_key(|c| c.pipeline_id);
        self.total_flushed += batch.len() as u64;
        batch
    }

    /// Number of commands currently pending.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Total commands flushed across all batches.
    #[must_use]
    pub fn total_flushed(&self) -> u64 {
        self.total_flushed
    }

    /// Maximum batch size before an auto-flush is triggered.
    #[must_use]
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}

impl std::fmt::Debug for BatchedComputePass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedComputePass")
            .field("pending", &self.pending.len())
            .field("max_batch_size", &self.max_batch_size)
            .field("total_flushed", &self.total_flushed)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass_type_video_is_real_time() {
        assert!(PassType::Video.is_real_time());
    }

    #[test]
    fn test_pass_type_audio_is_real_time() {
        assert!(PassType::Audio.is_real_time());
    }

    #[test]
    fn test_pass_type_image_not_real_time() {
        assert!(!PassType::Image.is_real_time());
    }

    #[test]
    fn test_pass_type_post_process_not_real_time() {
        assert!(!PassType::PostProcess.is_real_time());
    }

    #[test]
    fn test_buffer_binding_input() {
        let b = BufferBinding::new(0, 1024, true);
        assert!(b.is_input());
        assert!(!b.is_output());
    }

    #[test]
    fn test_buffer_binding_output() {
        let b = BufferBinding::new(1, 2048, false);
        assert!(b.is_output());
        assert!(!b.is_input());
    }

    #[test]
    fn test_compute_pass_new_defaults() {
        let pass = ComputePass::new("test", PassType::Image);
        assert_eq!(pass.name, "test");
        assert_eq!(pass.workgroups, (1, 1, 1));
        assert_eq!(pass.binding_count(), 0);
    }

    #[test]
    fn test_compute_pass_add_input_binding() {
        let mut pass = ComputePass::new("p", PassType::Video);
        pass.add_input_binding(0, 512);
        assert_eq!(pass.binding_count(), 1);
        assert!(pass.bindings[0].is_input());
    }

    #[test]
    fn test_compute_pass_add_output_binding() {
        let mut pass = ComputePass::new("p", PassType::Video);
        pass.add_output_binding(1, 512);
        assert_eq!(pass.binding_count(), 1);
        assert!(pass.bindings[0].is_output());
    }

    #[test]
    fn test_total_work_items_1x1x1() {
        let pass = ComputePass::new("p", PassType::Audio);
        assert_eq!(pass.total_work_items(), 1);
    }

    #[test]
    fn test_total_work_items_custom() {
        let mut pass = ComputePass::new("p", PassType::Image);
        pass.workgroups = (4, 8, 2);
        assert_eq!(pass.total_work_items(), 64);
    }

    #[test]
    fn test_pass_queue_add_and_count() {
        let mut q = PassQueue::new();
        q.add(ComputePass::new("a", PassType::Video));
        q.add(ComputePass::new("b", PassType::Image));
        assert_eq!(q.pass_count(), 2);
    }

    #[test]
    fn test_pass_queue_passes_of_type() {
        let mut q = PassQueue::new();
        q.add(ComputePass::new("v1", PassType::Video));
        q.add(ComputePass::new("i1", PassType::Image));
        q.add(ComputePass::new("v2", PassType::Video));
        let videos = q.passes_of_type(&PassType::Video);
        assert_eq!(videos.len(), 2);
    }

    #[test]
    fn test_pass_queue_passes_of_type_empty_result() {
        let mut q = PassQueue::new();
        q.add(ComputePass::new("a", PassType::Audio));
        let results = q.passes_of_type(&PassType::PostProcess);
        assert!(results.is_empty());
    }

    #[test]
    fn test_pass_queue_total_bindings() {
        let mut q = PassQueue::new();
        let mut p1 = ComputePass::new("p1", PassType::Video);
        p1.add_input_binding(0, 256);
        p1.add_output_binding(1, 256);
        let mut p2 = ComputePass::new("p2", PassType::Image);
        p2.add_input_binding(0, 128);
        q.add(p1);
        q.add(p2);
        assert_eq!(q.total_bindings(), 3);
    }

    #[test]
    fn test_pass_queue_empty() {
        let q = PassQueue::new();
        assert_eq!(q.pass_count(), 0);
        assert_eq!(q.total_bindings(), 0);
    }

    // ── BatchedComputePass tests ──────────────────────────────────────────────

    fn make_cmd(pipeline_id: u64, x: u32) -> DispatchCommand {
        DispatchCommand::new(pipeline_id, 0, x, 1, 1)
    }

    #[test]
    fn test_batched_submit_no_auto_flush_below_limit() {
        let mut b = BatchedComputePass::new(5);
        for i in 0..4u32 {
            let flushed = b.submit(make_cmd(1, i));
            assert!(!flushed, "should not auto-flush below max_batch_size");
        }
        assert_eq!(b.pending_count(), 4);
    }

    #[test]
    fn test_batched_submit_auto_flush_at_limit() {
        let mut b = BatchedComputePass::new(5);
        for i in 0..4u32 {
            b.submit(make_cmd(1, i));
        }
        let triggered = b.submit(make_cmd(1, 4));
        assert!(triggered, "5th submit should signal auto-flush");
    }

    #[test]
    fn test_batched_flush_returns_all_pending() {
        let mut b = BatchedComputePass::new(10);
        b.submit(make_cmd(3, 1));
        b.submit(make_cmd(1, 2));
        b.submit(make_cmd(2, 3));
        let batch = b.flush();
        assert_eq!(batch.len(), 3);
        assert_eq!(b.pending_count(), 0);
    }

    #[test]
    fn test_batched_flush_sorts_by_pipeline_id() {
        let mut b = BatchedComputePass::new(100);
        b.submit(make_cmd(5, 0));
        b.submit(make_cmd(1, 0));
        b.submit(make_cmd(3, 0));
        b.submit(make_cmd(2, 0));
        let batch = b.flush();
        let ids: Vec<u64> = batch.iter().map(|c| c.pipeline_id).collect();
        assert_eq!(ids, vec![1, 2, 3, 5], "batch must be sorted by pipeline_id");
    }

    #[test]
    fn test_batched_flush_empty_returns_empty() {
        let mut b = BatchedComputePass::new(5);
        let batch = b.flush();
        assert!(
            batch.is_empty(),
            "flushing an empty batcher returns empty vec"
        );
    }

    #[test]
    fn test_batched_total_flushed_accumulates() {
        let mut b = BatchedComputePass::new(3);
        for i in 0..6u32 {
            b.submit(make_cmd(1, i));
        }
        b.flush(); // flush remaining
        assert_eq!(
            b.total_flushed(),
            6,
            "total flushed must equal total submitted"
        );
    }

    #[test]
    fn test_batched_similar_pipeline_ids_adjacent() {
        let mut b = BatchedComputePass::new(100);
        // Mix of pipeline IDs 10 and 20
        b.submit(make_cmd(20, 0));
        b.submit(make_cmd(10, 0));
        b.submit(make_cmd(20, 1));
        b.submit(make_cmd(10, 1));
        let batch = b.flush();
        // Expect: 10, 10, 20, 20
        let ids: Vec<u64> = batch.iter().map(|c| c.pipeline_id).collect();
        assert_eq!(ids[0], 10);
        assert_eq!(ids[1], 10);
        assert_eq!(ids[2], 20);
        assert_eq!(ids[3], 20);
    }

    #[test]
    fn test_batched_max_batch_size_accessor() {
        let b = BatchedComputePass::new(8);
        assert_eq!(b.max_batch_size(), 8);
    }

    #[test]
    fn test_batched_debug_fmt() {
        let b = BatchedComputePass::new(4);
        let s = format!("{b:?}");
        assert!(s.contains("BatchedComputePass"));
    }
}
