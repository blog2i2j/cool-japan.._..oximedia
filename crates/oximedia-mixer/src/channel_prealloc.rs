//! Pre-allocated channel output buffer management.
//!
//! Allocating a fresh `Vec<f32>` for each channel on every `process()` call
//! produces garbage-collector pressure and latency jitter on the audio thread.
//! This module provides a [`ChannelBufferSet`] that owns a contiguous slab of
//! `f32` memory divided into fixed-size blocks — one per channel slot.  The
//! mixer allocates the slab once, then borrows mutable sub-slices during each
//! processing block with zero heap activity.
//!
//! # Design
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │  ChannelBufferSet (heap slab: max_channels × block)  │
//! │  ┌────────┬────────┬────────┬─ ─ ─┬────────┐        │
//! │  │  ch 0  │  ch 1  │  ch 2  │     │ ch N-1 │        │
//! │  └────────┴────────┴────────┴─ ─ ─┴────────┘        │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! Channels are addressed by a 0-based `slot` index.  The mixer assigns each
//! logical channel a stable slot at creation time and releases the slot when
//! the channel is removed.  A simple free-list (`SlotAllocator`) tracks which
//! slots are available.
//!
//! # Thread Safety
//!
//! [`ChannelBufferSet`] is intentionally **not** `Sync` — it is designed for
//! single-threaded use on the audio callback.  For parallel channel processing
//! (e.g. rayon) see `parallel_mix`.
//!
//! # Example
//!
//! ```rust
//! use oximedia_mixer::channel_prealloc::{ChannelBufferSet, SlotAllocator};
//!
//! let mut set = ChannelBufferSet::new(32, 512); // 32 channels × 512 samples
//! let mut alloc = SlotAllocator::new(32);
//!
//! // Acquire a slot for a new channel.
//! let slot = alloc.acquire().unwrap();
//!
//! // Zero the buffer for this channel.
//! set.zero(slot);
//!
//! // Write into the channel buffer.
//! {
//!     let buf = set.get_mut(slot).unwrap();
//!     buf[0] = 1.0;
//! }
//!
//! // Release the slot when the channel is removed.
//! alloc.release(slot);
//! ```

// ---------------------------------------------------------------------------
// SlotAllocator
// ---------------------------------------------------------------------------

/// Manages a fixed set of numbered slots, handing them out and reclaiming them.
///
/// Backed by a simple free-list.  Allocation and release are both O(1) amortised.
#[derive(Debug, Clone)]
pub struct SlotAllocator {
    capacity: usize,
    /// Stack of free slot indices.
    free: Vec<usize>,
    /// Bitmask tracking live (allocated) slots for O(1) `is_allocated` checks.
    allocated: Vec<bool>,
}

impl SlotAllocator {
    /// Create a new allocator with `capacity` slots (all initially free).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let free: Vec<usize> = (0..capacity).rev().collect();
        Self {
            capacity,
            free,
            allocated: vec![false; capacity],
        }
    }

    /// Acquire the next free slot.
    ///
    /// Returns `None` if all slots are in use.
    pub fn acquire(&mut self) -> Option<usize> {
        let slot = self.free.pop()?;
        self.allocated[slot] = true;
        Some(slot)
    }

    /// Release `slot`, making it available again.
    ///
    /// Returns `false` (no-op) if the slot was not allocated.
    pub fn release(&mut self, slot: usize) -> bool {
        if slot >= self.capacity || !self.allocated[slot] {
            return false;
        }
        self.allocated[slot] = false;
        self.free.push(slot);
        true
    }

    /// Returns `true` if `slot` is currently allocated.
    #[must_use]
    pub fn is_allocated(&self, slot: usize) -> bool {
        slot < self.capacity && self.allocated[slot]
    }

    /// Number of free slots remaining.
    #[must_use]
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Maximum number of slots (including allocated ones).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of currently allocated slots.
    #[must_use]
    pub fn allocated_count(&self) -> usize {
        self.capacity - self.free.len()
    }
}

// ---------------------------------------------------------------------------
// ChannelBufferSet
// ---------------------------------------------------------------------------

/// A slab of pre-allocated `f32` audio buffers, one per channel slot.
///
/// Create once with `max_channels` and `block_size`, then call [`get`] /
/// [`get_mut`] per channel during each audio callback without any heap
/// allocation.
pub struct ChannelBufferSet {
    /// Contiguous storage: `max_channels × block_size` samples.
    data: Vec<f32>,
    /// Number of channel slots.
    max_channels: usize,
    /// Samples per channel buffer.
    block_size: usize,
}

impl ChannelBufferSet {
    /// Allocate the slab.  All samples are initialised to `0.0`.
    #[must_use]
    pub fn new(max_channels: usize, block_size: usize) -> Self {
        Self {
            data: vec![0.0_f32; max_channels * block_size],
            max_channels,
            block_size,
        }
    }

    /// Immutable view of the buffer for `slot`.
    ///
    /// Returns `None` if `slot >= max_channels`.
    #[must_use]
    pub fn get(&self, slot: usize) -> Option<&[f32]> {
        if slot >= self.max_channels {
            return None;
        }
        let start = slot * self.block_size;
        Some(&self.data[start..start + self.block_size])
    }

    /// Mutable view of the buffer for `slot`.
    ///
    /// Returns `None` if `slot >= max_channels`.
    pub fn get_mut(&mut self, slot: usize) -> Option<&mut [f32]> {
        if slot >= self.max_channels {
            return None;
        }
        let start = slot * self.block_size;
        Some(&mut self.data[start..start + self.block_size])
    }

    /// Fill the buffer for `slot` with zeros.
    ///
    /// Does nothing if `slot >= max_channels`.
    pub fn zero(&mut self, slot: usize) {
        if let Some(buf) = self.get_mut(slot) {
            buf.iter_mut().for_each(|s| *s = 0.0);
        }
    }

    /// Zero all buffers in the slab at once.
    ///
    /// Call this at the start of each processing block to clear stale data.
    pub fn zero_all(&mut self) {
        self.data.iter_mut().for_each(|s| *s = 0.0);
    }

    /// Copy `src` into the buffer for `slot`.
    ///
    /// Copies `min(block_size, src.len())` samples.  Returns `false` if `slot`
    /// is out of range.
    pub fn copy_into(&mut self, slot: usize, src: &[f32]) -> bool {
        match self.get_mut(slot) {
            None => false,
            Some(buf) => {
                let n = buf.len().min(src.len());
                buf[..n].copy_from_slice(&src[..n]);
                true
            }
        }
    }

    /// Accumulate (add) `src` into the buffer for `slot`.
    ///
    /// Useful for summing multiple contributors into a single bus buffer.
    /// Returns `false` if `slot` is out of range.
    pub fn accumulate(&mut self, slot: usize, src: &[f32]) -> bool {
        match self.get_mut(slot) {
            None => false,
            Some(buf) => {
                let n = buf.len().min(src.len());
                for (b, &s) in buf[..n].iter_mut().zip(src[..n].iter()) {
                    *b += s;
                }
                true
            }
        }
    }

    /// Apply a scalar gain to all samples in the buffer for `slot`.
    ///
    /// Returns `false` if `slot` is out of range.
    pub fn apply_gain(&mut self, slot: usize, gain: f32) -> bool {
        match self.get_mut(slot) {
            None => false,
            Some(buf) => {
                buf.iter_mut().for_each(|s| *s *= gain);
                true
            }
        }
    }

    /// Peak sample magnitude in the buffer for `slot`.
    ///
    /// Returns `0.0` if `slot` is out of range or the buffer is empty.
    #[must_use]
    pub fn peak(&self, slot: usize) -> f32 {
        self.get(slot)
            .map(|buf| buf.iter().fold(0.0_f32, |acc, &s| acc.max(s.abs())))
            .unwrap_or(0.0)
    }

    /// RMS level of the buffer for `slot` (linear).
    ///
    /// Returns `0.0` if `slot` is out of range or block_size is 0.
    #[must_use]
    pub fn rms(&self, slot: usize) -> f32 {
        match self.get(slot) {
            None => 0.0,
            Some([]) => 0.0,
            Some(buf) => {
                let sum_sq: f32 = buf.iter().map(|&s| s * s).sum();
                (sum_sq / buf.len() as f32).sqrt()
            }
        }
    }

    /// Maximum channel slots.
    #[must_use]
    pub fn max_channels(&self) -> usize {
        self.max_channels
    }

    /// Samples per buffer (block size).
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Total `f32` words allocated in the slab.
    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.data.len()
    }
}

impl std::fmt::Debug for ChannelBufferSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelBufferSet")
            .field("max_channels", &self.max_channels)
            .field("block_size", &self.block_size)
            .field("total_samples", &self.data.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ChannelSlab — convenience struct combining SlotAllocator + ChannelBufferSet
// ---------------------------------------------------------------------------

/// Convenience struct that ties together a [`ChannelBufferSet`] and its
/// associated [`SlotAllocator`].
///
/// Use this as the single owner of all pre-allocated audio state in a mixer.
#[derive(Debug)]
pub struct ChannelSlab {
    buffers: ChannelBufferSet,
    slots: SlotAllocator,
}

impl ChannelSlab {
    /// Create a new slab for `max_channels` channels, each `block_size` samples.
    #[must_use]
    pub fn new(max_channels: usize, block_size: usize) -> Self {
        Self {
            buffers: ChannelBufferSet::new(max_channels, block_size),
            slots: SlotAllocator::new(max_channels),
        }
    }

    /// Add a new channel, returning its slot index.
    ///
    /// Returns `None` if the slab is full.
    pub fn add_channel(&mut self) -> Option<usize> {
        let slot = self.slots.acquire()?;
        self.buffers.zero(slot);
        Some(slot)
    }

    /// Remove a channel by slot, returning `true` if it was allocated.
    pub fn remove_channel(&mut self, slot: usize) -> bool {
        if self.slots.release(slot) {
            self.buffers.zero(slot);
            true
        } else {
            false
        }
    }

    /// Immutable view of the channel buffer at `slot`.
    #[must_use]
    pub fn buffer(&self, slot: usize) -> Option<&[f32]> {
        if self.slots.is_allocated(slot) {
            self.buffers.get(slot)
        } else {
            None
        }
    }

    /// Mutable view of the channel buffer at `slot`.
    pub fn buffer_mut(&mut self, slot: usize) -> Option<&mut [f32]> {
        if self.slots.is_allocated(slot) {
            self.buffers.get_mut(slot)
        } else {
            None
        }
    }

    /// Zero all channel buffers.
    pub fn zero_all(&mut self) {
        self.buffers.zero_all();
    }

    /// Number of active (allocated) channels.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.slots.allocated_count()
    }

    /// Maximum number of channels the slab can hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// Samples per channel buffer.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.buffers.block_size()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_allocator_basic() {
        let mut alloc = SlotAllocator::new(4);
        assert_eq!(alloc.free_count(), 4);
        let s0 = alloc.acquire().unwrap();
        assert_eq!(alloc.free_count(), 3);
        assert!(alloc.is_allocated(s0));
        alloc.release(s0);
        assert_eq!(alloc.free_count(), 4);
        assert!(!alloc.is_allocated(s0));
    }

    #[test]
    fn test_slot_allocator_exhaustion() {
        let mut alloc = SlotAllocator::new(2);
        let _s0 = alloc.acquire().unwrap();
        let _s1 = alloc.acquire().unwrap();
        assert!(alloc.acquire().is_none(), "should return None when full");
    }

    #[test]
    fn test_slot_allocator_release_invalid_is_noop() {
        let mut alloc = SlotAllocator::new(4);
        let released = alloc.release(0); // slot 0 was never allocated
        assert!(!released, "releasing unallocated slot should return false");
    }

    #[test]
    fn test_channel_buffer_set_get_correct_slice() {
        let set = ChannelBufferSet::new(4, 128);
        let buf = set.get(0).unwrap();
        assert_eq!(buf.len(), 128);
    }

    #[test]
    fn test_channel_buffer_set_out_of_range() {
        let set = ChannelBufferSet::new(4, 128);
        assert!(set.get(4).is_none());
        assert!(set.get(99).is_none());
    }

    #[test]
    fn test_zero_and_write() {
        let mut set = ChannelBufferSet::new(2, 8);
        let ok = set.copy_into(0, &[1.0_f32; 8]);
        assert!(ok);
        set.zero(0);
        let buf = set.get(0).unwrap();
        assert!(buf.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_accumulate() {
        let mut set = ChannelBufferSet::new(2, 4);
        set.copy_into(0, &[0.5_f32; 4]);
        set.accumulate(0, &[0.5_f32; 4]);
        let buf = set.get(0).unwrap();
        assert!(buf.iter().all(|&s| (s - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_apply_gain() {
        let mut set = ChannelBufferSet::new(2, 4);
        set.copy_into(0, &[1.0_f32; 4]);
        set.apply_gain(0, 0.5);
        let buf = set.get(0).unwrap();
        assert!(buf.iter().all(|&s| (s - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_peak_and_rms() {
        let mut set = ChannelBufferSet::new(1, 4);
        set.copy_into(0, &[0.0, 0.5, -1.0, 0.25]);
        assert!((set.peak(0) - 1.0).abs() < 1e-6);
        // RMS of [0, 0.5, -1.0, 0.25]: sum_sq = 0+0.25+1.0+0.0625=1.3125
        let expected_rms = (1.3125_f32 / 4.0).sqrt();
        assert!((set.rms(0) - expected_rms).abs() < 1e-5);
    }

    #[test]
    fn test_channel_slab_add_remove() {
        let mut slab = ChannelSlab::new(8, 256);
        let s0 = slab.add_channel().unwrap();
        let s1 = slab.add_channel().unwrap();
        assert_eq!(slab.channel_count(), 2);

        // Write to channel 0.
        if let Some(buf) = slab.buffer_mut(s0) {
            buf[0] = 3.14;
        }
        assert!((slab.buffer(s0).unwrap()[0] - 3.14).abs() < 1e-5);

        // Remove channel 0; its buffer should be inaccessible.
        slab.remove_channel(s0);
        assert!(slab.buffer(s0).is_none(), "removed channel should be inaccessible");
        assert_eq!(slab.channel_count(), 1);
        drop(s1);
    }

    #[test]
    fn test_channel_slab_capacity_enforced() {
        let mut slab = ChannelSlab::new(2, 64);
        let _s0 = slab.add_channel().unwrap();
        let _s1 = slab.add_channel().unwrap();
        assert!(
            slab.add_channel().is_none(),
            "should not be able to add more than capacity"
        );
    }

    #[test]
    fn test_zero_all() {
        let mut set = ChannelBufferSet::new(3, 4);
        for slot in 0..3 {
            set.copy_into(slot, &[1.0_f32; 4]);
        }
        set.zero_all();
        for slot in 0..3 {
            assert!(set.get(slot).unwrap().iter().all(|&s| s == 0.0));
        }
    }

    #[test]
    fn test_total_samples() {
        let set = ChannelBufferSet::new(16, 512);
        assert_eq!(set.total_samples(), 16 * 512);
    }

    #[test]
    fn test_slab_block_size_accessor() {
        let slab = ChannelSlab::new(4, 128);
        assert_eq!(slab.block_size(), 128);
        assert_eq!(slab.capacity(), 4);
    }
}
