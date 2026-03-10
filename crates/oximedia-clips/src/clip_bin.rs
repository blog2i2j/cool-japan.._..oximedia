//! Clip bin management for organizing clips into named containers.
//!
//! Provides `BinItem`, `ClipBin`, and `BinManager` for hierarchical clip organization.

#![allow(dead_code)]

use std::collections::HashMap;

/// Unique identifier for a bin.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BinId(pub String);

impl BinId {
    /// Create a new `BinId` from a string.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// A single item stored inside a `ClipBin`.
#[derive(Debug, Clone)]
pub struct BinItem {
    /// Identifier of the clip or folder this item references.
    pub item_id: String,
    /// Display name of the item.
    pub name: String,
    /// Whether this item represents a sub-folder rather than a clip.
    pub folder: bool,
    /// Optional description.
    pub description: Option<String>,
}

impl BinItem {
    /// Create a new clip item (not a folder).
    pub fn new_clip(item_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            name: name.into(),
            folder: false,
            description: None,
        }
    }

    /// Create a new folder item.
    pub fn new_folder(item_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            name: name.into(),
            folder: true,
            description: None,
        }
    }

    /// Returns `true` when this item represents a folder rather than a clip.
    pub fn is_folder(&self) -> bool {
        self.folder
    }

    /// Set an optional description on this item.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// A named bin that holds an ordered list of `BinItem`s.
#[derive(Debug, Clone)]
pub struct ClipBin {
    /// Unique identifier.
    pub id: BinId,
    /// Human-readable name.
    pub name: String,
    /// Ordered collection of items.
    pub items: Vec<BinItem>,
    /// Optional color label (hex string).
    pub color: Option<String>,
}

impl ClipBin {
    /// Create an empty `ClipBin` with the given name.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: BinId::new(id),
            name: name.into(),
            items: Vec::new(),
            color: None,
        }
    }

    /// Append a `BinItem` to this bin.
    pub fn add_item(&mut self, item: BinItem) {
        self.items.push(item);
    }

    /// Move the item at `from_index` to `to_index`.
    ///
    /// Returns `false` if either index is out of range.
    pub fn move_item(&mut self, from_index: usize, to_index: usize) -> bool {
        let len = self.items.len();
        if from_index >= len || to_index >= len {
            return false;
        }
        let item = self.items.remove(from_index);
        self.items.insert(to_index, item);
        true
    }

    /// Remove the item with the given `item_id`.
    ///
    /// Returns the removed item if found.
    pub fn remove(&mut self, item_id: &str) -> Option<BinItem> {
        if let Some(pos) = self.items.iter().position(|i| i.item_id == item_id) {
            Some(self.items.remove(pos))
        } else {
            None
        }
    }

    /// Number of items currently in this bin.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Assign a color label to this bin.
    pub fn set_color(&mut self, color: impl Into<String>) {
        self.color = Some(color.into());
    }
}

/// Manages a collection of `ClipBin`s.
#[derive(Debug, Default)]
pub struct BinManager {
    bins: HashMap<BinId, ClipBin>,
}

impl BinManager {
    /// Create a new, empty `BinManager`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a bin with the given id and name and register it.
    ///
    /// Returns a reference to the newly created bin.
    pub fn create_bin(&mut self, id: impl Into<String>, name: impl Into<String>) -> &ClipBin {
        let bin = ClipBin::new(id, name);
        let bin_id = bin.id.clone();
        self.bins.insert(bin_id.clone(), bin);
        self.bins
            .get(&bin_id)
            .expect("bin was just inserted so it must be present")
    }

    /// Find a bin by its id, returning an immutable reference if present.
    pub fn find_bin(&self, id: &str) -> Option<&ClipBin> {
        self.bins.get(&BinId::new(id))
    }

    /// Find a bin by its id, returning a mutable reference if present.
    pub fn find_bin_mut(&mut self, id: &str) -> Option<&mut ClipBin> {
        self.bins.get_mut(&BinId::new(id))
    }

    /// Total number of bins.
    pub fn bin_count(&self) -> usize {
        self.bins.len()
    }

    /// Remove a bin by id, returning it if it existed.
    pub fn remove_bin(&mut self, id: &str) -> Option<ClipBin> {
        self.bins.remove(&BinId::new(id))
    }

    /// Iterate over all bins.
    pub fn bins(&self) -> impl Iterator<Item = &ClipBin> {
        self.bins.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(id: &str, name: &str) -> BinItem {
        BinItem::new_clip(id, name)
    }

    #[test]
    fn bin_item_is_folder_false_for_clip() {
        let item = BinItem::new_clip("clip-1", "Interview");
        assert!(!item.is_folder());
    }

    #[test]
    fn bin_item_is_folder_true_for_folder() {
        let item = BinItem::new_folder("folder-1", "Raw Footage");
        assert!(item.is_folder());
    }

    #[test]
    fn bin_item_description() {
        let item = BinItem::new_clip("c1", "Take 1").with_description("Primary angle");
        assert_eq!(item.description.as_deref(), Some("Primary angle"));
    }

    #[test]
    fn clip_bin_add_item_increments_count() {
        let mut bin = ClipBin::new("b1", "Day 1");
        assert_eq!(bin.item_count(), 0);
        bin.add_item(make_item("c1", "Clip A"));
        bin.add_item(make_item("c2", "Clip B"));
        assert_eq!(bin.item_count(), 2);
    }

    #[test]
    fn clip_bin_remove_returns_item() {
        let mut bin = ClipBin::new("b1", "Bin");
        bin.add_item(make_item("c1", "Alpha"));
        let removed = bin.remove("c1");
        assert!(removed.is_some());
        assert_eq!(removed.expect("value should be valid").name, "Alpha");
        assert_eq!(bin.item_count(), 0);
    }

    #[test]
    fn clip_bin_remove_missing_returns_none() {
        let mut bin = ClipBin::new("b1", "Bin");
        assert!(bin.remove("nonexistent").is_none());
    }

    #[test]
    fn clip_bin_move_item_reorders() {
        let mut bin = ClipBin::new("b1", "Bin");
        bin.add_item(make_item("c1", "First"));
        bin.add_item(make_item("c2", "Second"));
        bin.add_item(make_item("c3", "Third"));
        assert!(bin.move_item(0, 2));
        assert_eq!(bin.items[0].item_id, "c2");
        assert_eq!(bin.items[2].item_id, "c1");
    }

    #[test]
    fn clip_bin_move_item_out_of_range() {
        let mut bin = ClipBin::new("b1", "Bin");
        bin.add_item(make_item("c1", "Only"));
        assert!(!bin.move_item(0, 5));
    }

    #[test]
    fn bin_manager_create_and_find() {
        let mut mgr = BinManager::new();
        mgr.create_bin("bin-a", "Scene 1");
        let found = mgr.find_bin("bin-a");
        assert!(found.is_some());
        assert_eq!(found.expect("value should be valid").name, "Scene 1");
    }

    #[test]
    fn bin_manager_find_missing_returns_none() {
        let mgr = BinManager::new();
        assert!(mgr.find_bin("no-such-bin").is_none());
    }

    #[test]
    fn bin_manager_bin_count() {
        let mut mgr = BinManager::new();
        assert_eq!(mgr.bin_count(), 0);
        mgr.create_bin("b1", "A");
        mgr.create_bin("b2", "B");
        assert_eq!(mgr.bin_count(), 2);
    }

    #[test]
    fn bin_manager_remove_bin() {
        let mut mgr = BinManager::new();
        mgr.create_bin("x", "X");
        let removed = mgr.remove_bin("x");
        assert!(removed.is_some());
        assert_eq!(mgr.bin_count(), 0);
    }

    #[test]
    fn bin_manager_find_bin_mut_allows_modification() {
        let mut mgr = BinManager::new();
        mgr.create_bin("b1", "Editable");
        if let Some(bin) = mgr.find_bin_mut("b1") {
            bin.add_item(make_item("c1", "New Clip"));
        }
        assert_eq!(
            mgr.find_bin("b1")
                .expect("find_bin should succeed")
                .item_count(),
            1
        );
    }

    #[test]
    fn bin_color_label() {
        let mut bin = ClipBin::new("b1", "Colored");
        bin.set_color("#FF0000");
        assert_eq!(bin.color.as_deref(), Some("#FF0000"));
    }
}
