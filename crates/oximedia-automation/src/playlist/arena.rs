//! Arena allocator for [`ExecutableItem`] storage.
//!
//! Allocates playlist items in large contiguous blocks rather than individually,
//! which reduces per-item allocation overhead during playout.  Items are stored
//! inside `Vec<Vec<ExecutableItem>>`, so the total heap allocation count is
//! `ceil(len / block_capacity)` instead of `len`.
//!
//! The arena only supports push and indexed read — it deliberately does not
//! expose mutation or deletion in order to keep the implementation simple and
//! safe.  Calling [`PlaylistArena::clear`] resets the logical length to zero
//! *without* deallocating the underlying blocks, so the memory is immediately
//! reusable for the next playlist.

use crate::playlist::executor::ExecutableItem;

/// Arena allocator for [`ExecutableItem`] storage.
///
/// # Block layout
///
/// Items are packed into contiguous *blocks*, each of capacity `block_capacity`.
/// A new block is allocated only when the current one is full, keeping the total
/// number of heap allocations proportional to `len / block_capacity`.
///
/// ```text
/// blocks[0]:  [item0 | item1 | … | item_{B-1}]   ← B = block_capacity
/// blocks[1]:  [item_B | item_{B+1} | …]
/// …
/// ```
///
/// # Indexing
///
/// Given a flat index `i`:
/// - block index   = `i / block_capacity`
/// - intra-block   = `i % block_capacity`
pub struct PlaylistArena {
    /// Contiguous storage blocks.
    blocks: Vec<Vec<ExecutableItem>>,
    /// Capacity of each individual block.
    block_capacity: usize,
    /// Total number of items currently stored.
    len: usize,
}

impl PlaylistArena {
    /// Create a new arena with the given block capacity.
    ///
    /// # Panics
    ///
    /// Panics if `block_capacity` is zero.
    pub fn new(block_capacity: usize) -> Self {
        assert!(
            block_capacity > 0,
            "block_capacity must be greater than zero"
        );
        Self {
            blocks: Vec::new(),
            block_capacity,
            len: 0,
        }
    }

    /// Append `item` to the arena and return its flat index.
    ///
    /// A new block is allocated if the current block is full.
    pub fn push(&mut self, item: ExecutableItem) -> usize {
        let block_idx = self.len / self.block_capacity;
        let intra = self.len % self.block_capacity;

        if block_idx >= self.blocks.len() {
            // Allocate a new block pre-sized to `block_capacity`.
            self.blocks.push(Vec::with_capacity(self.block_capacity));
        }

        // The block exists — push into it.
        // Safety: we just ensured `block_idx < self.blocks.len()`.
        self.blocks[block_idx].push(item);
        let _ = intra; // intra is implicitly used via blocks[block_idx].len()

        let idx = self.len;
        self.len += 1;
        idx
    }

    /// Return a reference to the item at `index`, or `None` if out of range.
    pub fn get(&self, index: usize) -> Option<&ExecutableItem> {
        if index >= self.len {
            return None;
        }
        let block_idx = index / self.block_capacity;
        let intra = index % self.block_capacity;
        self.blocks.get(block_idx)?.get(intra)
    }

    /// Return the total number of items stored in the arena.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return `true` if no items have been stored.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Reset the arena without deallocating underlying blocks.
    ///
    /// The logical length drops to zero and all previously allocated blocks are
    /// cleared.  The block `Vec`s themselves are retained so their memory can be
    /// reused immediately.
    pub fn clear(&mut self) {
        for block in &mut self.blocks {
            block.clear();
        }
        self.len = 0;
    }

    /// Return the number of blocks currently allocated (including partially
    /// filled ones).
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Return the total item capacity across all allocated blocks.
    ///
    /// This is `block_count() × block_capacity` and is always `≥ len()`.
    pub fn capacity(&self) -> usize {
        self.blocks.len() * self.block_capacity
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Construct a minimal `ExecutableItem` for testing.
    fn make_item(id: &str) -> ExecutableItem {
        ExecutableItem {
            id: id.to_string(),
            file_path: format!("/media/{id}.mxf"),
            duration_frames: 1000,
            scheduled_start: None,
            preroll_frames: 0,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn push_1000_items_yields_correct_len() {
        let mut arena = PlaylistArena::new(64);
        for i in 0..1000 {
            arena.push(make_item(&format!("item_{i:04}")));
        }
        assert_eq!(arena.len(), 1000);
    }

    #[test]
    fn get_first_and_last_return_correct_items() {
        let mut arena = PlaylistArena::new(64);
        for i in 0..1000 {
            arena.push(make_item(&format!("item_{i:04}")));
        }

        let first = arena.get(0).expect("item 0 should exist");
        assert_eq!(first.id, "item_0000");

        let last = arena.get(999).expect("item 999 should exist");
        assert_eq!(last.id, "item_0999");
    }

    #[test]
    fn get_out_of_range_returns_none() {
        let mut arena = PlaylistArena::new(64);
        arena.push(make_item("only"));
        assert!(arena.get(1).is_none());
        assert!(arena.get(1000).is_none());
    }

    #[test]
    fn clear_resets_len_but_preserves_block_count() {
        let mut arena = PlaylistArena::new(16);
        for i in 0..64 {
            arena.push(make_item(&format!("i{i}")));
        }
        let block_count_before = arena.block_count();
        assert!(block_count_before > 0);

        arena.clear();
        assert_eq!(arena.len(), 0, "len must be 0 after clear");
        assert!(arena.is_empty());
        assert_eq!(
            arena.block_count(),
            block_count_before,
            "clear must not deallocate blocks"
        );
    }

    #[test]
    fn block_count_grows_as_items_exceed_block_capacity() {
        let block_capacity = 10;
        let mut arena = PlaylistArena::new(block_capacity);

        // Push exactly one block's worth — block_count should be 1.
        for i in 0..block_capacity {
            arena.push(make_item(&format!("a{i}")));
        }
        assert_eq!(arena.block_count(), 1);

        // Push one more item — triggers a second block.
        arena.push(make_item("overflow"));
        assert_eq!(arena.block_count(), 2);
    }

    #[test]
    fn arena_uses_fewer_allocations_than_individual_items() {
        // For 1000 items with block_size = 64 the arena only allocates
        // ceil(1000/64) = 16 blocks, vs 1000 separate allocations.
        let mut arena = PlaylistArena::new(64);
        for i in 0..1000 {
            arena.push(make_item(&format!("it{i}")));
        }
        let expected_blocks = (1000 + 63) / 64; // = 16
        assert_eq!(arena.block_count(), expected_blocks);
        assert!(
            arena.block_count() < 1000,
            "block_count ({}) must be far less than 1000 individual allocations",
            arena.block_count()
        );
    }

    #[test]
    fn capacity_always_gte_len() {
        let mut arena = PlaylistArena::new(32);
        for i in 0..100 {
            arena.push(make_item(&format!("c{i}")));
            assert!(arena.capacity() >= arena.len());
        }
    }

    #[test]
    fn is_empty_correct_before_and_after_push() {
        let mut arena = PlaylistArena::new(8);
        assert!(arena.is_empty());
        arena.push(make_item("first"));
        assert!(!arena.is_empty());
    }

    #[test]
    fn clear_then_push_reuses_blocks() {
        let mut arena = PlaylistArena::new(8);
        for i in 0..32 {
            arena.push(make_item(&format!("r{i}")));
        }
        let blocks_after_first_fill = arena.block_count();
        arena.clear();

        // Push fewer items than the first fill — should not allocate new blocks.
        for i in 0..16 {
            arena.push(make_item(&format!("s{i}")));
        }
        assert_eq!(arena.len(), 16);
        assert_eq!(
            arena.block_count(),
            blocks_after_first_fill,
            "refill should reuse existing blocks"
        );

        let item = arena.get(15).expect("item 15 should exist");
        assert_eq!(item.id, "s15");
    }

    #[test]
    fn push_returns_sequential_indices() {
        let mut arena = PlaylistArena::new(4);
        for i in 0..12 {
            let idx = arena.push(make_item(&format!("x{i}")));
            assert_eq!(idx, i, "push should return sequential index {i}");
        }
    }
}
