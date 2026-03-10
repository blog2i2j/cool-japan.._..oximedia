//! Resource handle tracking for media pipeline objects.
//!
//! Provides lightweight handle types and a tracker that enforces reference
//! counting semantics without `Arc`/`Rc`, making it easy to audit GPU and
//! CPU resource lifetimes in hot paths.

#![allow(dead_code)]

use std::collections::HashMap;

/// The kind of resource represented by a [`ResourceHandle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// A GPU texture or buffer allocation.
    Gpu,
    /// A CPU heap allocation (e.g. a frame buffer pool entry).
    CpuBuffer,
    /// An open file descriptor.
    FileDescriptor,
    /// A network socket.
    Socket,
    /// Any other resource kind.
    Other,
}

impl ResourceKind {
    /// Returns `true` if this resource resides on the GPU.
    #[must_use]
    pub fn is_gpu(self) -> bool {
        matches!(self, ResourceKind::Gpu)
    }
}

/// Lifecycle state of a tracked resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceState {
    /// The resource has been allocated and is in use.
    Active,
    /// The resource has been released and its handle is invalid.
    Released,
    /// The resource is temporarily unavailable (e.g. being flushed).
    Suspended,
}

impl ResourceState {
    /// Returns `true` if the resource is in the [`ResourceState::Active`] state.
    #[must_use]
    pub fn is_active(self) -> bool {
        matches!(self, ResourceState::Active)
    }
}

/// A lightweight, copy-able handle that identifies a tracked resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle {
    id: u64,
    kind: ResourceKind,
}

impl ResourceHandle {
    /// Create a handle with a specific id and kind (used internally by
    /// [`ResourceTracker`]).
    fn new(id: u64, kind: ResourceKind) -> Self {
        Self { id, kind }
    }

    /// Returns `true` if this handle has a non-zero id (the zero id is
    /// reserved as the *null handle*).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.id != 0
    }

    /// The numeric id of this handle.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The kind of resource this handle refers to.
    #[must_use]
    pub fn kind(&self) -> ResourceKind {
        self.kind
    }
}

/// Entry stored inside [`ResourceTracker`] for each active handle.
#[derive(Debug)]
struct ResourceEntry {
    state: ResourceState,
    kind: ResourceKind,
    ref_count: u32,
}

/// Tracks [`ResourceHandle`] allocations and reference counts.
///
/// Designed for single-threaded or externally-synchronised use.
#[derive(Debug, Default)]
pub struct ResourceTracker {
    entries: HashMap<u64, ResourceEntry>,
    next_id: u64,
}

impl ResourceTracker {
    /// Create a new, empty [`ResourceTracker`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            next_id: 1, // 0 is the null handle
        }
    }

    /// Acquire a new resource handle of the given kind.
    ///
    /// The returned handle is immediately in the [`ResourceState::Active`] state
    /// with a reference count of 1.
    pub fn acquire(&mut self, kind: ResourceKind) -> ResourceHandle {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(
            id,
            ResourceEntry {
                state: ResourceState::Active,
                kind,
                ref_count: 1,
            },
        );
        ResourceHandle::new(id, kind)
    }

    /// Release a resource handle, decrementing its reference count.
    ///
    /// When the reference count reaches zero the entry is removed.
    /// Returns `true` if the handle was found and released.
    pub fn release(&mut self, handle: ResourceHandle) -> bool {
        if let Some(entry) = self.entries.get_mut(&handle.id) {
            if entry.ref_count > 1 {
                entry.ref_count -= 1;
            } else {
                entry.state = ResourceState::Released;
                self.entries.remove(&handle.id);
            }
            true
        } else {
            false
        }
    }

    /// Returns the number of currently active (non-released) resources.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.entries
            .values()
            .filter(|e| e.state.is_active())
            .count()
    }

    /// Returns the [`ResourceState`] for the given handle, or `None` if not
    /// tracked.
    #[must_use]
    pub fn state(&self, handle: ResourceHandle) -> Option<ResourceState> {
        self.entries.get(&handle.id).map(|e| e.state)
    }

    /// Suspend a resource (e.g. while flushing a pipeline stage).
    ///
    /// Returns `false` if the handle is not found.
    pub fn suspend(&mut self, handle: ResourceHandle) -> bool {
        if let Some(entry) = self.entries.get_mut(&handle.id) {
            entry.state = ResourceState::Suspended;
            true
        } else {
            false
        }
    }

    /// Resume a previously suspended resource.
    ///
    /// Returns `false` if the handle is not found.
    pub fn resume(&mut self, handle: ResourceHandle) -> bool {
        if let Some(entry) = self.entries.get_mut(&handle.id) {
            entry.state = ResourceState::Active;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_kind_is_gpu() {
        assert!(ResourceKind::Gpu.is_gpu());
        assert!(!ResourceKind::CpuBuffer.is_gpu());
        assert!(!ResourceKind::FileDescriptor.is_gpu());
    }

    #[test]
    fn test_resource_state_is_active() {
        assert!(ResourceState::Active.is_active());
        assert!(!ResourceState::Released.is_active());
        assert!(!ResourceState::Suspended.is_active());
    }

    #[test]
    fn test_handle_is_valid() {
        let h = ResourceHandle::new(1, ResourceKind::Gpu);
        assert!(h.is_valid());
        let null = ResourceHandle::new(0, ResourceKind::Other);
        assert!(!null.is_valid());
    }

    #[test]
    fn test_tracker_acquire() {
        let mut tracker = ResourceTracker::new();
        let h = tracker.acquire(ResourceKind::Gpu);
        assert!(h.is_valid());
        assert_eq!(tracker.active_count(), 1);
    }

    #[test]
    fn test_tracker_release() {
        let mut tracker = ResourceTracker::new();
        let h = tracker.acquire(ResourceKind::CpuBuffer);
        assert!(tracker.release(h));
        assert_eq!(tracker.active_count(), 0);
    }

    #[test]
    fn test_tracker_release_unknown_handle() {
        let mut tracker = ResourceTracker::new();
        let fake = ResourceHandle::new(9999, ResourceKind::Other);
        assert!(!tracker.release(fake));
    }

    #[test]
    fn test_tracker_active_count_multiple() {
        let mut tracker = ResourceTracker::new();
        let _h1 = tracker.acquire(ResourceKind::Gpu);
        let _h2 = tracker.acquire(ResourceKind::CpuBuffer);
        let h3 = tracker.acquire(ResourceKind::Socket);
        assert_eq!(tracker.active_count(), 3);
        tracker.release(h3);
        assert_eq!(tracker.active_count(), 2);
    }

    #[test]
    fn test_tracker_state_active() {
        let mut tracker = ResourceTracker::new();
        let h = tracker.acquire(ResourceKind::Gpu);
        assert_eq!(tracker.state(h), Some(ResourceState::Active));
    }

    #[test]
    fn test_tracker_state_none_after_release() {
        let mut tracker = ResourceTracker::new();
        let h = tracker.acquire(ResourceKind::Gpu);
        tracker.release(h);
        assert_eq!(tracker.state(h), None);
    }

    #[test]
    fn test_tracker_suspend_resume() {
        let mut tracker = ResourceTracker::new();
        let h = tracker.acquire(ResourceKind::Gpu);
        assert!(tracker.suspend(h));
        assert_eq!(tracker.state(h), Some(ResourceState::Suspended));
        assert!(!tracker.state(h).expect("state should exist").is_active());
        assert!(tracker.resume(h));
        assert_eq!(tracker.state(h), Some(ResourceState::Active));
    }

    #[test]
    fn test_handle_kind_accessor() {
        let mut tracker = ResourceTracker::new();
        let h = tracker.acquire(ResourceKind::FileDescriptor);
        assert_eq!(h.kind(), ResourceKind::FileDescriptor);
    }

    #[test]
    fn test_tracker_ids_are_unique() {
        let mut tracker = ResourceTracker::new();
        let h1 = tracker.acquire(ResourceKind::Gpu);
        let h2 = tracker.acquire(ResourceKind::Gpu);
        assert_ne!(h1.id(), h2.id());
    }
}
