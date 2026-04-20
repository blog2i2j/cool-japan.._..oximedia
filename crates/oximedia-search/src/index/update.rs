//! Incremental index updates.

use crate::error::SearchResult;
use crate::index::builder::IndexDocument;
use uuid::Uuid;

/// Index updater for incremental updates
pub struct IndexUpdater {
    index_path: String,
    pending_updates: Vec<IndexUpdate>,
}

/// Type of index update
#[derive(Debug, Clone)]
pub enum IndexUpdate {
    /// Add a new document
    Add(IndexDocument),
    /// Update an existing document
    Update(IndexDocument),
    /// Delete a document
    Delete(Uuid),
}

impl IndexUpdater {
    /// Create a new index updater
    #[must_use]
    pub fn new(index_path: &str) -> Self {
        Self {
            index_path: index_path.to_string(),
            pending_updates: Vec::new(),
        }
    }

    /// Queue an add operation
    pub fn add(&mut self, doc: IndexDocument) {
        self.pending_updates.push(IndexUpdate::Add(doc));
    }

    /// Queue an update operation
    pub fn update(&mut self, doc: IndexDocument) {
        self.pending_updates.push(IndexUpdate::Update(doc));
    }

    /// Queue a delete operation
    pub fn delete(&mut self, asset_id: Uuid) {
        self.pending_updates.push(IndexUpdate::Delete(asset_id));
    }

    /// Apply all pending updates
    ///
    /// # Errors
    ///
    /// Returns an error if updates cannot be applied
    pub fn apply(&mut self) -> SearchResult<usize> {
        let count = self.pending_updates.len();

        for update in &self.pending_updates {
            self.apply_update(update)?;
        }

        self.pending_updates.clear();

        Ok(count)
    }

    /// Apply a single update
    fn apply_update(&self, update: &IndexUpdate) -> SearchResult<()> {
        match update {
            IndexUpdate::Add(doc) => self.apply_add(doc),
            IndexUpdate::Update(doc) => self.apply_update_doc(doc),
            IndexUpdate::Delete(id) => self.apply_delete(*id),
        }
    }

    /// Apply add operation
    fn apply_add(&self, _doc: &IndexDocument) -> SearchResult<()> {
        // Implementation would add document to all indices
        Ok(())
    }

    /// Apply update operation
    fn apply_update_doc(&self, _doc: &IndexDocument) -> SearchResult<()> {
        // Implementation would update document in all indices
        // This typically involves delete + add
        Ok(())
    }

    /// Apply delete operation
    fn apply_delete(&self, _asset_id: Uuid) -> SearchResult<()> {
        // Implementation would delete document from all indices
        Ok(())
    }

    /// Get number of pending updates
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending_updates.len()
    }

    /// Clear all pending updates
    pub fn clear(&mut self) {
        self.pending_updates.clear();
    }

    /// Batch update multiple documents efficiently
    ///
    /// # Errors
    ///
    /// Returns an error if batch update fails
    pub fn batch_update(&mut self, updates: Vec<IndexUpdate>) -> SearchResult<()> {
        for update in updates {
            self.pending_updates.push(update);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_index() -> String {
        std::env::temp_dir()
            .join("oximedia-search-index-update-test_index")
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_index_updater_new() {
        let updater = IndexUpdater::new(&tmp_index());
        assert_eq!(updater.pending_count(), 0);
    }

    #[test]
    fn test_add_delete() {
        let mut updater = IndexUpdater::new(&tmp_index());
        let id = Uuid::new_v4();

        updater.delete(id);
        assert_eq!(updater.pending_count(), 1);

        updater.clear();
        assert_eq!(updater.pending_count(), 0);
    }
}
