//! Rundown editor: insert, delete, reorder, and splice rundown items.
//!
//! This module extends the basic [`crate::rundown`] types with a rich editing
//! API that supports:
//!
//! - **Insert** — at any position, with automatic ID allocation
//! - **Delete** — by ID or by position
//! - **Reorder** — move an item up/down or to an explicit index
//! - **Swap** — exchange two items by their IDs
//! - **Split** — divide one item into two at a given timecode offset
//! - **Merge** — combine two adjacent items into one
//! - **Duration recalculation** — gap / over-run analysis after each edit
//! - **History stack** — linear undo / redo (up to configurable depth)
//!
//! All mutating operations produce an [`EditOp`] value that is pushed onto
//! the undo stack so that changes can be reversed.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::rundown::{Rundown, RundownItem};

// ── Edit operations ────────────────────────────────────────────────────────────

/// A reversible edit operation applied to a [`RundownEditor`].
#[derive(Debug, Clone)]
pub enum EditOp {
    /// Insert an item at the given index.
    Insert { index: usize, item: RundownItem },
    /// Remove the item at the given index.
    Remove { index: usize, item: RundownItem },
    /// Move item from `from` index to `to` index.
    Move { from: usize, to: usize },
    /// Swap items at the two indices.
    Swap { a: usize, b: usize },
    /// Replace item at `index` with a new value (captures prior state).
    Replace { index: usize, before: RundownItem, after: RundownItem },
    /// Batch of sub-operations committed together (e.g. merge / split).
    Batch(Vec<EditOp>),
}

// ── Errors ─────────────────────────────────────────────────────────────────────

/// Errors returned by the rundown editor.
#[derive(Debug, Clone, PartialEq)]
pub enum EditorError {
    /// The supplied item ID does not exist in the rundown.
    IdNotFound(u32),
    /// The supplied position index is out of bounds.
    IndexOutOfBounds(usize),
    /// The two items specified for a merge are not adjacent.
    NotAdjacent(u32, u32),
    /// The split offset exceeds the item's planned duration.
    SplitOffsetExceedsDuration { offset_secs: f32, duration_secs: f32 },
    /// An operation was applied to an empty rundown.
    EmptyRundown,
    /// The undo/redo stack is empty.
    NothingToUndo,
    /// Duplicate item ID detected during insertion.
    DuplicateId(u32),
    /// Duration underflow — resulting duration would be negative.
    DurationUnderflow,
}

impl std::fmt::Display for EditorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IdNotFound(id) => write!(f, "item ID {id} not found in rundown"),
            Self::IndexOutOfBounds(i) => write!(f, "index {i} is out of bounds"),
            Self::NotAdjacent(a, b) => write!(f, "items {a} and {b} are not adjacent"),
            Self::SplitOffsetExceedsDuration { offset_secs, duration_secs } => write!(
                f,
                "split offset {offset_secs}s exceeds item duration {duration_secs}s"
            ),
            Self::EmptyRundown => write!(f, "operation requires a non-empty rundown"),
            Self::NothingToUndo => write!(f, "undo/redo stack is empty"),
            Self::DuplicateId(id) => write!(f, "item ID {id} already exists in rundown"),
            Self::DurationUnderflow => write!(f, "resulting duration would be negative"),
        }
    }
}

impl std::error::Error for EditorError {}

// ── Duration analysis ──────────────────────────────────────────────────────────

/// Per-item timing report produced after an edit.
#[derive(Debug, Clone)]
pub struct ItemTimingReport {
    /// Item ID this report covers.
    pub item_id: u32,
    /// Planned offset from programme start (accumulated from previous items).
    pub planned_offset_secs: f32,
    /// Over-run / under-run from planned duration (positive = over, negative = under).
    pub deviation_secs: f32,
}

/// Programme-level timing summary.
#[derive(Debug, Clone)]
pub struct TimingSummary {
    /// Total planned duration (sum of all items).
    pub total_planned_secs: f32,
    /// Total actual duration (items that have played; unplayed items counted at planned).
    pub total_actual_secs: f32,
    /// Net deviation in seconds (positive = programme running long).
    pub net_deviation_secs: f32,
    /// Per-item timing details.
    pub items: Vec<ItemTimingReport>,
}

// ── ID allocator ───────────────────────────────────────────────────────────────

/// Simple monotonically-increasing ID allocator.
#[derive(Debug, Clone, Default)]
struct IdAllocator {
    next: u32,
}

impl IdAllocator {
    fn new(start: u32) -> Self {
        Self { next: start }
    }

    fn alloc(&mut self) -> u32 {
        let id = self.next;
        self.next = self.next.saturating_add(1);
        id
    }
}

// ── Rundown editor ─────────────────────────────────────────────────────────────

/// Maximum default undo history depth.
const DEFAULT_HISTORY_DEPTH: usize = 64;

/// Full-featured rundown editor with undo/redo support.
///
/// Wraps a [`Rundown`] and provides edit operations that are automatically
/// recorded in a history stack.
#[derive(Debug, Clone)]
pub struct RundownEditor {
    /// The rundown being edited.
    rundown: Rundown,
    /// Undo stack — most recent operation is at the back.
    undo_stack: Vec<EditOp>,
    /// Redo stack — cleared on every new mutating operation.
    redo_stack: Vec<EditOp>,
    /// Maximum undo stack depth.
    history_depth: usize,
    /// ID allocator for items inserted by the editor.
    id_alloc: IdAllocator,
}

impl RundownEditor {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Wrap an existing [`Rundown`] in an editor.
    ///
    /// The first auto-allocated ID will be one more than the maximum existing
    /// item ID, or 1 if the rundown is empty.
    pub fn new(rundown: Rundown) -> Self {
        let next_id = rundown
            .items
            .iter()
            .map(|i| i.id)
            .max()
            .map(|m| m.saturating_add(1))
            .unwrap_or(1);
        Self {
            rundown,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            history_depth: DEFAULT_HISTORY_DEPTH,
            id_alloc: IdAllocator::new(next_id),
        }
    }

    /// Set the maximum undo history depth (number of operations retained).
    pub fn with_history_depth(mut self, depth: usize) -> Self {
        self.history_depth = depth;
        self
    }

    // ── Read-only accessors ───────────────────────────────────────────────────

    /// Borrow the underlying rundown.
    pub fn rundown(&self) -> &Rundown {
        &self.rundown
    }

    /// Number of items in the rundown.
    pub fn len(&self) -> usize {
        self.rundown.items.len()
    }

    /// Returns `true` if the rundown contains no items.
    pub fn is_empty(&self) -> bool {
        self.rundown.items.is_empty()
    }

    /// Return the number of undoable operations available.
    pub fn undo_depth(&self) -> usize {
        self.undo_stack.len()
    }

    /// Return the number of redoable operations available.
    pub fn redo_depth(&self) -> usize {
        self.redo_stack.len()
    }

    /// Find the position index of an item by its ID.
    pub fn position_of(&self, id: u32) -> Option<usize> {
        self.rundown.items.iter().position(|i| i.id == id)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Push an operation onto the undo stack, trimming to `history_depth`.
    fn push_undo(&mut self, op: EditOp) {
        if self.undo_stack.len() >= self.history_depth {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(op);
        self.redo_stack.clear();
    }

    /// Check whether `index` is a valid insertion point (0..=len).
    fn check_insert_index(&self, index: usize) -> Result<(), EditorError> {
        if index > self.rundown.items.len() {
            Err(EditorError::IndexOutOfBounds(index))
        } else {
            Ok(())
        }
    }

    /// Check whether `index` is a valid item index (0..len).
    fn check_item_index(&self, index: usize) -> Result<(), EditorError> {
        if index >= self.rundown.items.len() {
            Err(EditorError::IndexOutOfBounds(index))
        } else {
            Ok(())
        }
    }

    /// Check that ID `id` does not already appear in the rundown.
    fn check_no_duplicate_id(&self, id: u32) -> Result<(), EditorError> {
        if self.rundown.items.iter().any(|i| i.id == id) {
            Err(EditorError::DuplicateId(id))
        } else {
            Ok(())
        }
    }

    // ── Insert ────────────────────────────────────────────────────────────────

    /// Insert `item` at position `index`.
    ///
    /// `index == 0` prepends; `index == len()` appends.
    pub fn insert_at(&mut self, index: usize, item: RundownItem) -> Result<(), EditorError> {
        self.check_insert_index(index)?;
        self.check_no_duplicate_id(item.id)?;
        let op = EditOp::Insert { index, item: item.clone() };
        self.rundown.items.insert(index, item);
        self.push_undo(op);
        Ok(())
    }

    /// Append an item to the end of the rundown.
    pub fn append(&mut self, item: RundownItem) -> Result<(), EditorError> {
        let idx = self.rundown.items.len();
        self.insert_at(idx, item)
    }

    /// Prepend an item to the front of the rundown.
    pub fn prepend(&mut self, item: RundownItem) -> Result<(), EditorError> {
        self.insert_at(0, item)
    }

    /// Allocate a new unique item ID and return it.
    ///
    /// Useful when callers want the editor to manage IDs.
    pub fn alloc_id(&mut self) -> u32 {
        self.id_alloc.alloc()
    }

    // ── Delete ────────────────────────────────────────────────────────────────

    /// Remove the item at `index` and return a copy of it.
    pub fn remove_at(&mut self, index: usize) -> Result<RundownItem, EditorError> {
        self.check_item_index(index)?;
        let item = self.rundown.items.remove(index);
        self.push_undo(EditOp::Remove { index, item: item.clone() });
        Ok(item)
    }

    /// Remove the item with the given `id`.
    pub fn remove_by_id(&mut self, id: u32) -> Result<RundownItem, EditorError> {
        let index = self
            .position_of(id)
            .ok_or(EditorError::IdNotFound(id))?;
        self.remove_at(index)
    }

    // ── Reorder ───────────────────────────────────────────────────────────────

    /// Move the item at position `from` to position `to`.
    ///
    /// Both indices must be valid item positions. The item is first removed
    /// from `from` then inserted at the (potentially shifted) `to`.
    pub fn move_item(&mut self, from: usize, to: usize) -> Result<(), EditorError> {
        self.check_item_index(from)?;
        // `to` is the desired final index — allowed up to len()-1 because we're
        // moving an existing item, not inserting a new one.
        let last = self.rundown.items.len().saturating_sub(1);
        if to > last {
            return Err(EditorError::IndexOutOfBounds(to));
        }
        if from == to {
            return Ok(()); // no-op — nothing to record
        }
        let item = self.rundown.items.remove(from);
        // After removing `from`, all indices > from shift down by 1.
        // When moving forward (to > from), the desired final index `to` in the
        // *original* list corresponds to index `to` in the shortened list,
        // because we insert *after* what was at `to-1` (now at `to-1`).
        // When moving backward (to < from), the desired index is unchanged.
        self.rundown.items.insert(to, item);
        self.push_undo(EditOp::Move { from, to });
        Ok(())
    }

    /// Move item with `id` one position toward the front of the rundown.
    pub fn move_up(&mut self, id: u32) -> Result<(), EditorError> {
        let index = self.position_of(id).ok_or(EditorError::IdNotFound(id))?;
        if index == 0 {
            return Ok(()); // already first
        }
        self.move_item(index, index - 1)
    }

    /// Move item with `id` one position toward the end of the rundown.
    pub fn move_down(&mut self, id: u32) -> Result<(), EditorError> {
        let index = self.position_of(id).ok_or(EditorError::IdNotFound(id))?;
        let last = self.rundown.items.len().saturating_sub(1);
        if index >= last {
            return Ok(()); // already last
        }
        self.move_item(index, index + 1)
    }

    // ── Swap ──────────────────────────────────────────────────────────────────

    /// Swap the positions of the two items identified by `id_a` and `id_b`.
    pub fn swap_by_id(&mut self, id_a: u32, id_b: u32) -> Result<(), EditorError> {
        let a = self.position_of(id_a).ok_or(EditorError::IdNotFound(id_a))?;
        let b = self.position_of(id_b).ok_or(EditorError::IdNotFound(id_b))?;
        if a == b {
            return Ok(());
        }
        self.rundown.items.swap(a, b);
        self.push_undo(EditOp::Swap { a, b });
        Ok(())
    }

    // ── Split ─────────────────────────────────────────────────────────────────

    /// Split the item with `id` into two items at `offset_secs` from the start.
    ///
    /// The first item retains the original ID and gets `offset_secs` duration.
    /// The second item is allocated a new ID and gets the remaining duration.
    /// Both items inherit the original item's type and status.
    ///
    /// Returns `(first_id, second_id)`.
    pub fn split(&mut self, id: u32, offset_secs: f32) -> Result<(u32, u32), EditorError> {
        let index = self.position_of(id).ok_or(EditorError::IdNotFound(id))?;
        let original = self.rundown.items[index].clone();

        if offset_secs <= 0.0 || offset_secs >= original.duration_secs {
            return Err(EditorError::SplitOffsetExceedsDuration {
                offset_secs,
                duration_secs: original.duration_secs,
            });
        }

        let second_id = self.id_alloc.alloc();
        let remainder = original.duration_secs - offset_secs;

        let mut first = original.clone();
        first.duration_secs = offset_secs;
        first.actual_duration_secs = None;

        let second = RundownItem {
            id: second_id,
            title: format!("{} (part 2)", original.title),
            item_type: original.item_type.clone(),
            duration_secs: remainder,
            actual_duration_secs: None,
            status: original.status.clone(),
        };

        // Record as a Batch so undo restores the original single item.
        let before = original.clone();
        let after_first = first.clone();
        let after_second = second.clone();

        self.rundown.items[index] = first;
        self.rundown.items.insert(index + 1, second);

        self.push_undo(EditOp::Batch(vec![
            EditOp::Replace { index, before, after: after_first },
            EditOp::Insert { index: index + 1, item: after_second },
        ]));

        Ok((id, second_id))
    }

    // ── Merge ─────────────────────────────────────────────────────────────────

    /// Merge two adjacent items into a single item.
    ///
    /// `id_a` must immediately precede `id_b` in the rundown. The merged item
    /// keeps `id_a`, a concatenated title, and the summed planned duration.
    /// Actual durations are summed if both are present; otherwise the merged
    /// item's actual duration is cleared.
    pub fn merge(&mut self, id_a: u32, id_b: u32) -> Result<u32, EditorError> {
        let a = self.position_of(id_a).ok_or(EditorError::IdNotFound(id_a))?;
        let b = self.position_of(id_b).ok_or(EditorError::IdNotFound(id_b))?;

        if b != a + 1 {
            return Err(EditorError::NotAdjacent(id_a, id_b));
        }

        let item_a = self.rundown.items[a].clone();
        let item_b = self.rundown.items[b].clone();

        let merged_actual = match (item_a.actual_duration_secs, item_b.actual_duration_secs) {
            (Some(x), Some(y)) => Some(x + y),
            _ => None,
        };

        let merged = RundownItem {
            id: item_a.id,
            title: format!("{} + {}", item_a.title, item_b.title),
            item_type: item_a.item_type.clone(),
            duration_secs: item_a.duration_secs + item_b.duration_secs,
            actual_duration_secs: merged_actual,
            status: item_a.status.clone(),
        };

        let before_a = item_a.clone();
        let before_b = item_b;
        let after_merged = merged.clone();

        // Remove b first (higher index) then replace a.
        self.rundown.items.remove(b);
        self.rundown.items[a] = merged;

        self.push_undo(EditOp::Batch(vec![
            EditOp::Replace { index: a, before: before_a, after: after_merged },
            EditOp::Remove { index: b, item: before_b },
        ]));

        Ok(id_a)
    }

    // ── Duration analysis ─────────────────────────────────────────────────────

    /// Build a [`TimingSummary`] for the current rundown state.
    ///
    /// Items that have not been played contribute their planned duration to the
    /// total actual count (assumes they will run to time).
    pub fn timing_summary(&self) -> TimingSummary {
        let mut offset = 0.0f32;
        let mut total_actual = 0.0f32;
        let mut item_reports = Vec::with_capacity(self.rundown.items.len());

        for item in &self.rundown.items {
            let actual = item.actual_duration_secs.unwrap_or(item.duration_secs);
            let deviation = item.actual_duration_secs
                .map(|a| a - item.duration_secs)
                .unwrap_or(0.0);

            item_reports.push(ItemTimingReport {
                item_id: item.id,
                planned_offset_secs: offset,
                deviation_secs: deviation,
            });

            offset += item.duration_secs;
            total_actual += actual;
        }

        let total_planned = self.rundown.items.iter().map(|i| i.duration_secs).sum();
        TimingSummary {
            total_planned_secs: total_planned,
            total_actual_secs: total_actual,
            net_deviation_secs: total_actual - total_planned,
            items: item_reports,
        }
    }

    /// Return `true` if the programme is currently running long by more than
    /// `threshold_secs`.
    pub fn is_over_run(&self, threshold_secs: f32) -> bool {
        self.timing_summary().net_deviation_secs > threshold_secs
    }

    /// Compute the remaining programme time assuming all unplayed items run to
    /// their planned durations.
    ///
    /// `elapsed_secs` is the number of seconds already transmitted.
    pub fn remaining_secs(&self, elapsed_secs: f32) -> f32 {
        let total = self.rundown.items.iter().map(|i| i.duration_secs).sum::<f32>();
        (total - elapsed_secs).max(0.0)
    }

    // ── Undo / Redo ───────────────────────────────────────────────────────────

    /// Undo the most recent edit operation.
    pub fn undo(&mut self) -> Result<(), EditorError> {
        let op = self.undo_stack.pop().ok_or(EditorError::NothingToUndo)?;
        self.apply_inverse(&op)?;
        self.redo_stack.push(op);
        Ok(())
    }

    /// Redo the most recently undone operation.
    pub fn redo(&mut self) -> Result<(), EditorError> {
        let op = self.redo_stack.pop().ok_or(EditorError::NothingToUndo)?;
        self.apply_forward(&op)?;
        self.undo_stack.push(op);
        Ok(())
    }

    /// Apply `op` in the forward (original) direction without recording history.
    fn apply_forward(&mut self, op: &EditOp) -> Result<(), EditorError> {
        match op {
            EditOp::Insert { index, item } => {
                self.check_insert_index(*index)?;
                self.rundown.items.insert(*index, item.clone());
            }
            EditOp::Remove { index, .. } => {
                self.check_item_index(*index)?;
                self.rundown.items.remove(*index);
            }
            EditOp::Move { from, to } => {
                self.check_item_index(*from)?;
                let last = self.rundown.items.len().saturating_sub(1);
                if *to > last {
                    return Err(EditorError::IndexOutOfBounds(*to));
                }
                let item = self.rundown.items.remove(*from);
                self.rundown.items.insert(*to, item);
            }
            EditOp::Swap { a, b } => {
                self.check_item_index(*a)?;
                self.check_item_index(*b)?;
                self.rundown.items.swap(*a, *b);
            }
            EditOp::Replace { index, after, .. } => {
                self.check_item_index(*index)?;
                self.rundown.items[*index] = after.clone();
            }
            EditOp::Batch(ops) => {
                for sub in ops {
                    self.apply_forward(sub)?;
                }
            }
        }
        Ok(())
    }

    /// Apply the inverse of `op` to undo it.
    fn apply_inverse(&mut self, op: &EditOp) -> Result<(), EditorError> {
        match op {
            // Undo an insert → remove
            EditOp::Insert { index, .. } => {
                self.check_item_index(*index)?;
                self.rundown.items.remove(*index);
            }
            // Undo a remove → re-insert
            EditOp::Remove { index, item } => {
                let insert_idx = (*index).min(self.rundown.items.len());
                self.rundown.items.insert(insert_idx, item.clone());
            }
            // Undo a move → remove from `to`, insert at `from`.
            EditOp::Move { from, to } => {
                // After the forward move the item is at `to`.
                // Remove it and reinsert at `from`.
                if *to < self.rundown.items.len() {
                    let item = self.rundown.items.remove(*to);
                    let ins = (*from).min(self.rundown.items.len());
                    self.rundown.items.insert(ins, item);
                }
            }
            // Undo a swap → swap back (swap is self-inverse)
            EditOp::Swap { a, b } => {
                self.check_item_index(*a)?;
                self.check_item_index(*b)?;
                self.rundown.items.swap(*a, *b);
            }
            // Undo a replace → restore before
            EditOp::Replace { index, before, .. } => {
                self.check_item_index(*index)?;
                self.rundown.items[*index] = before.clone();
            }
            // Undo a batch in reverse order
            EditOp::Batch(ops) => {
                for sub in ops.iter().rev() {
                    self.apply_inverse(sub)?;
                }
            }
        }
        Ok(())
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rundown::ItemType;

    fn make_rundown() -> Rundown {
        let mut rd = Rundown::new("News at Six", 1800.0);
        rd.add_item(RundownItem::new(1, "Intro", ItemType::Story, 60.0));
        rd.add_item(RundownItem::new(2, "Headline 1", ItemType::Story, 120.0));
        rd.add_item(RundownItem::new(3, "Ad Break", ItemType::Advert, 90.0));
        rd.add_item(RundownItem::new(4, "Story A", ItemType::Story, 180.0));
        rd
    }

    // ── Insert / append / prepend ─────────────────────────────────────────────

    #[test]
    fn test_append_increases_len() {
        let mut ed = RundownEditor::new(make_rundown());
        let initial = ed.len();
        let item = RundownItem::new(99, "Filler", ItemType::Filler, 30.0);
        ed.append(item).expect("append should succeed");
        assert_eq!(ed.len(), initial + 1);
    }

    #[test]
    fn test_prepend_places_item_first() {
        let mut ed = RundownEditor::new(make_rundown());
        let item = RundownItem::new(99, "Cold Open", ItemType::Story, 15.0);
        ed.prepend(item).expect("prepend should succeed");
        assert_eq!(ed.rundown().items[0].id, 99);
    }

    #[test]
    fn test_insert_at_middle() {
        let mut ed = RundownEditor::new(make_rundown());
        let item = RundownItem::new(50, "Inserted", ItemType::Story, 45.0);
        ed.insert_at(2, item).expect("insert should succeed");
        assert_eq!(ed.rundown().items[2].id, 50);
    }

    #[test]
    fn test_insert_out_of_bounds_returns_error() {
        let mut ed = RundownEditor::new(make_rundown());
        let item = RundownItem::new(99, "X", ItemType::Story, 10.0);
        let result = ed.insert_at(999, item);
        assert!(matches!(result, Err(EditorError::IndexOutOfBounds(_))));
    }

    #[test]
    fn test_insert_duplicate_id_returns_error() {
        let mut ed = RundownEditor::new(make_rundown());
        let dup = RundownItem::new(1, "Dup", ItemType::Story, 10.0);
        let result = ed.append(dup);
        assert!(matches!(result, Err(EditorError::DuplicateId(1))));
    }

    // ── Delete ────────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_by_id_decreases_len() {
        let mut ed = RundownEditor::new(make_rundown());
        let initial = ed.len();
        ed.remove_by_id(2).expect("remove should succeed");
        assert_eq!(ed.len(), initial - 1);
    }

    #[test]
    fn test_remove_by_id_not_found_returns_error() {
        let mut ed = RundownEditor::new(make_rundown());
        let result = ed.remove_by_id(999);
        assert!(matches!(result, Err(EditorError::IdNotFound(999))));
    }

    // ── Reorder / move ────────────────────────────────────────────────────────

    #[test]
    fn test_move_up_changes_order() {
        let mut ed = RundownEditor::new(make_rundown());
        // Item 3 is initially at index 2; move it up to index 1.
        ed.move_up(3).expect("move_up should succeed");
        assert_eq!(ed.rundown().items[1].id, 3);
    }

    #[test]
    fn test_move_down_changes_order() {
        let mut ed = RundownEditor::new(make_rundown());
        // Item 1 is at index 0; move it down to index 1.
        ed.move_down(1).expect("move_down should succeed");
        assert_eq!(ed.rundown().items[1].id, 1);
    }

    #[test]
    fn test_move_up_at_front_is_noop() {
        let mut ed = RundownEditor::new(make_rundown());
        // Item 1 is already first — moving up should be a no-op.
        ed.move_up(1).expect("should succeed even at front");
        assert_eq!(ed.rundown().items[0].id, 1);
        // No undo entry because it was a no-op.
        assert_eq!(ed.undo_depth(), 0);
    }

    // ── Swap ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_swap_exchanges_positions() {
        let mut ed = RundownEditor::new(make_rundown());
        // Items at indices 0 and 3
        ed.swap_by_id(1, 4).expect("swap should succeed");
        assert_eq!(ed.rundown().items[0].id, 4);
        assert_eq!(ed.rundown().items[3].id, 1);
    }

    // ── Split ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_split_creates_two_items() {
        let mut ed = RundownEditor::new(make_rundown());
        let initial_len = ed.len();
        let (id_a, id_b) = ed.split(4, 90.0).expect("split should succeed");
        assert_eq!(ed.len(), initial_len + 1);
        // First part keeps original ID
        assert_eq!(id_a, 4);
        // Durations should sum to original
        let a = ed.rundown().items.iter().find(|i| i.id == id_a).expect("first part");
        let b = ed.rundown().items.iter().find(|i| i.id == id_b).expect("second part");
        assert!((a.duration_secs - 90.0).abs() < 0.001);
        assert!((b.duration_secs - 90.0).abs() < 0.001);
    }

    #[test]
    fn test_split_offset_at_boundary_returns_error() {
        let mut ed = RundownEditor::new(make_rundown());
        // offset == duration → error
        let result = ed.split(4, 180.0);
        assert!(matches!(result, Err(EditorError::SplitOffsetExceedsDuration { .. })));
    }

    // ── Merge ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_merge_adjacent_items() {
        let mut ed = RundownEditor::new(make_rundown());
        // items at indices 0 and 1 (IDs 1 and 2)
        let initial_len = ed.len();
        ed.merge(1, 2).expect("merge should succeed");
        assert_eq!(ed.len(), initial_len - 1);
        // Merged item keeps id 1 and has summed duration (60 + 120 = 180)
        let merged = &ed.rundown().items[0];
        assert_eq!(merged.id, 1);
        assert!((merged.duration_secs - 180.0).abs() < 0.001);
    }

    #[test]
    fn test_merge_non_adjacent_returns_error() {
        let mut ed = RundownEditor::new(make_rundown());
        let result = ed.merge(1, 3); // IDs 1 and 3 are not adjacent
        assert!(matches!(result, Err(EditorError::NotAdjacent(1, 3))));
    }

    // ── Timing summary ────────────────────────────────────────────────────────

    #[test]
    fn test_timing_summary_total_planned() {
        let ed = RundownEditor::new(make_rundown());
        let summary = ed.timing_summary();
        // 60 + 120 + 90 + 180 = 450
        assert!((summary.total_planned_secs - 450.0).abs() < 0.001);
    }

    #[test]
    fn test_timing_summary_no_deviation_when_unplayed() {
        let ed = RundownEditor::new(make_rundown());
        let summary = ed.timing_summary();
        assert!((summary.net_deviation_secs).abs() < 0.001);
    }

    #[test]
    fn test_remaining_secs_calculation() {
        let ed = RundownEditor::new(make_rundown());
        // Total planned = 450; if 100s elapsed, remaining = 350
        assert!((ed.remaining_secs(100.0) - 350.0).abs() < 0.001);
    }

    // ── Undo / Redo ───────────────────────────────────────────────────────────

    #[test]
    fn test_undo_insert() {
        let mut ed = RundownEditor::new(make_rundown());
        let initial_len = ed.len();
        let item = RundownItem::new(99, "Extra", ItemType::Story, 30.0);
        ed.append(item).expect("append");
        assert_eq!(ed.len(), initial_len + 1);
        ed.undo().expect("undo");
        assert_eq!(ed.len(), initial_len);
    }

    #[test]
    fn test_redo_restores_insert() {
        let mut ed = RundownEditor::new(make_rundown());
        let initial_len = ed.len();
        let item = RundownItem::new(99, "Extra", ItemType::Story, 30.0);
        ed.append(item).expect("append");
        ed.undo().expect("undo");
        ed.redo().expect("redo");
        assert_eq!(ed.len(), initial_len + 1);
    }

    #[test]
    fn test_undo_remove() {
        let mut ed = RundownEditor::new(make_rundown());
        let initial_len = ed.len();
        ed.remove_by_id(2).expect("remove");
        ed.undo().expect("undo remove");
        assert_eq!(ed.len(), initial_len);
    }

    #[test]
    fn test_undo_swap() {
        let mut ed = RundownEditor::new(make_rundown());
        let id0_before = ed.rundown().items[0].id;
        let id3_before = ed.rundown().items[3].id;
        ed.swap_by_id(id0_before, id3_before).expect("swap");
        ed.undo().expect("undo swap");
        assert_eq!(ed.rundown().items[0].id, id0_before);
        assert_eq!(ed.rundown().items[3].id, id3_before);
    }

    #[test]
    fn test_undo_empty_stack_returns_error() {
        let mut ed = RundownEditor::new(make_rundown());
        let result = ed.undo();
        assert!(matches!(result, Err(EditorError::NothingToUndo)));
    }

    #[test]
    fn test_new_operation_clears_redo_stack() {
        let mut ed = RundownEditor::new(make_rundown());
        let item = RundownItem::new(99, "X", ItemType::Story, 10.0);
        ed.append(item).expect("append");
        ed.undo().expect("undo");
        assert_eq!(ed.redo_depth(), 1);
        // New operation should clear redo stack
        let item2 = RundownItem::new(98, "Y", ItemType::Story, 10.0);
        ed.append(item2).expect("append again");
        assert_eq!(ed.redo_depth(), 0);
    }

    #[test]
    fn test_alloc_id_unique() {
        let mut ed = RundownEditor::new(make_rundown());
        let id_a = ed.alloc_id();
        let id_b = ed.alloc_id();
        assert_ne!(id_a, id_b);
        // Neither should collide with existing IDs 1-4
        assert!(!ed.rundown().items.iter().any(|i| i.id == id_a));
        assert!(!ed.rundown().items.iter().any(|i| i.id == id_b));
    }

    // ── Position helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_position_of_known_id() {
        let ed = RundownEditor::new(make_rundown());
        assert_eq!(ed.position_of(1), Some(0));
        assert_eq!(ed.position_of(4), Some(3));
    }

    #[test]
    fn test_position_of_unknown_id() {
        let ed = RundownEditor::new(make_rundown());
        assert_eq!(ed.position_of(999), None);
    }
}
