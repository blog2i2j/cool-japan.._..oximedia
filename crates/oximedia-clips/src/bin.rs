//! Hierarchical clip bin management with nested child bins.
//!
//! Provides [`ClipBin`] — a named container that can hold direct clip IDs
//! **and** nested child bins identified by a numeric index.  This mirrors the
//! bin/folder hierarchy found in professional NLEs (Avid, Resolve, Premiere).
//!
//! # Example
//!
//! ```
//! use oximedia_clips::bin::ClipBin;
//!
//! let mut root = ClipBin::new("Root");
//! let child_idx = root.add_child_bin("Day 1");
//! root.add_clip(100);
//! root.add_clip(101);
//!
//! assert_eq!(root.list_clips(), &[100, 101]);
//! assert_eq!(root.child_bin_count(), 1);
//! let child = root.get_child_bin(child_idx).unwrap();
//! assert_eq!(child.name(), "Day 1");
//! ```

#![allow(dead_code)]

// ─────────────────────────────────────────────────────────────────────────────
// ClipBin
// ─────────────────────────────────────────────────────────────────────────────

/// A named container that holds clip IDs and nested child bins.
///
/// Child bins are stored in insertion order; the index returned by
/// [`add_child_bin`](Self::add_child_bin) is a stable zero-based position into
/// that ordered list.
#[derive(Debug, Clone)]
pub struct ClipBin {
    name: String,
    clips: Vec<u64>,
    children: Vec<ClipBin>,
}

impl ClipBin {
    /// Create a new, empty `ClipBin` with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            clips: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Return the bin name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Rename the bin.
    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    /// Add a new child bin with the given name.
    ///
    /// Returns the zero-based index of the newly created child bin.
    pub fn add_child_bin(&mut self, name: &str) -> usize {
        let idx = self.children.len();
        self.children.push(ClipBin::new(name));
        idx
    }

    /// Add a clip ID to this bin.
    pub fn add_clip(&mut self, id: u64) {
        self.clips.push(id);
    }

    /// Remove a clip ID from this bin.  Returns `true` if the clip was present.
    pub fn remove_clip(&mut self, id: u64) -> bool {
        if let Some(pos) = self.clips.iter().position(|&c| c == id) {
            self.clips.remove(pos);
            true
        } else {
            false
        }
    }

    /// Return a slice of all clip IDs contained directly in this bin.
    pub fn list_clips(&self) -> &[u64] {
        &self.clips
    }

    /// Number of clips directly in this bin.
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Number of direct child bins.
    pub fn child_bin_count(&self) -> usize {
        self.children.len()
    }

    /// Get a reference to the child bin at `index`.
    pub fn get_child_bin(&self, index: usize) -> Option<&ClipBin> {
        self.children.get(index)
    }

    /// Get a mutable reference to the child bin at `index`.
    pub fn get_child_bin_mut(&mut self, index: usize) -> Option<&mut ClipBin> {
        self.children.get_mut(index)
    }

    /// Return an iterator over all direct child bins.
    pub fn child_bins(&self) -> impl Iterator<Item = &ClipBin> {
        self.children.iter()
    }

    /// Recursively count all clips (including clips in child bins).
    pub fn total_clip_count(&self) -> usize {
        let own = self.clips.len();
        let child_total: usize = self.children.iter().map(|c| c.total_clip_count()).sum();
        own + child_total
    }

    /// Whether this bin contains `id` directly (not in children).
    pub fn contains_clip(&self, id: u64) -> bool {
        self.clips.contains(&id)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_bin_is_empty() {
        let bin = ClipBin::new("Root");
        assert_eq!(bin.name(), "Root");
        assert_eq!(bin.clip_count(), 0);
        assert_eq!(bin.child_bin_count(), 0);
    }

    #[test]
    fn test_add_clip() {
        let mut bin = ClipBin::new("B");
        bin.add_clip(42);
        bin.add_clip(99);
        assert_eq!(bin.list_clips(), &[42, 99]);
    }

    #[test]
    fn test_add_child_bin_returns_index() {
        let mut bin = ClipBin::new("Root");
        let idx0 = bin.add_child_bin("Child A");
        let idx1 = bin.add_child_bin("Child B");
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(bin.child_bin_count(), 2);
    }

    #[test]
    fn test_get_child_bin() {
        let mut bin = ClipBin::new("Root");
        let idx = bin.add_child_bin("Day 1");
        let child = bin.get_child_bin(idx).expect("child should exist");
        assert_eq!(child.name(), "Day 1");
    }

    #[test]
    fn test_get_child_bin_out_of_bounds() {
        let bin = ClipBin::new("Root");
        assert!(bin.get_child_bin(99).is_none());
    }

    #[test]
    fn test_list_clips_empty() {
        let bin = ClipBin::new("Empty");
        assert!(bin.list_clips().is_empty());
    }

    #[test]
    fn test_remove_clip_present() {
        let mut bin = ClipBin::new("B");
        bin.add_clip(10);
        bin.add_clip(20);
        assert!(bin.remove_clip(10));
        assert_eq!(bin.list_clips(), &[20]);
    }

    #[test]
    fn test_remove_clip_absent() {
        let mut bin = ClipBin::new("B");
        bin.add_clip(10);
        assert!(!bin.remove_clip(99));
        assert_eq!(bin.clip_count(), 1);
    }

    #[test]
    fn test_contains_clip() {
        let mut bin = ClipBin::new("B");
        bin.add_clip(5);
        assert!(bin.contains_clip(5));
        assert!(!bin.contains_clip(6));
    }

    #[test]
    fn test_total_clip_count_includes_children() {
        let mut root = ClipBin::new("Root");
        root.add_clip(1);
        let child_idx = root.add_child_bin("Sub");
        root.get_child_bin_mut(child_idx)
            .expect("child")
            .add_clip(2);
        root.get_child_bin_mut(child_idx)
            .expect("child")
            .add_clip(3);
        assert_eq!(root.total_clip_count(), 3);
    }

    #[test]
    fn test_set_name() {
        let mut bin = ClipBin::new("Old");
        bin.set_name("New");
        assert_eq!(bin.name(), "New");
    }

    #[test]
    fn test_child_bins_iterator() {
        let mut root = ClipBin::new("Root");
        root.add_child_bin("A");
        root.add_child_bin("B");
        let names: Vec<_> = root.child_bins().map(|c| c.name()).collect();
        assert_eq!(names, &["A", "B"]);
    }
}
