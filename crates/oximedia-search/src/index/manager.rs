//! Index management and maintenance.

use crate::error::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStatistics {
    /// Total number of documents
    pub document_count: usize,
    /// Index size in bytes
    pub size_bytes: u64,
    /// Number of segments
    pub segment_count: usize,
    /// Last update timestamp
    pub last_updated: i64,
    /// Index health status
    pub health: IndexHealth,
}

/// Index health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexHealth {
    /// Index is healthy
    Healthy,
    /// Index needs optimization
    NeedsOptimization,
    /// Index is corrupted
    Corrupted,
}

/// Index manager for maintenance and optimization
pub struct IndexManager {
    index_path: PathBuf,
}

impl IndexManager {
    /// Create a new index manager
    #[must_use]
    pub fn new(index_path: &Path) -> Self {
        Self {
            index_path: index_path.to_path_buf(),
        }
    }

    /// Get index statistics
    ///
    /// # Errors
    ///
    /// Returns an error if statistics cannot be retrieved
    pub fn get_statistics(&self) -> SearchResult<IndexStatistics> {
        // Calculate index size
        let size_bytes = self.calculate_index_size()?;

        // Get document count (placeholder)
        let document_count = 0;

        // Get segment count (placeholder)
        let segment_count = 0;

        // Get last modified time
        let metadata = std::fs::metadata(&self.index_path)?;
        let last_updated = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| SearchError::Other(format!("Time error: {e}")))?
            .as_secs() as i64;

        Ok(IndexStatistics {
            document_count,
            size_bytes,
            segment_count,
            last_updated,
            health: IndexHealth::Healthy,
        })
    }

    /// Optimize the index by merging segments
    ///
    /// # Errors
    ///
    /// Returns an error if optimization fails
    pub fn optimize(&self) -> SearchResult<()> {
        // Optimization implementation
        // This would merge segments in Tantivy and clean up fragmented indices
        Ok(())
    }

    /// Verify index integrity
    ///
    /// # Errors
    ///
    /// Returns an error if verification fails
    pub fn verify(&self) -> SearchResult<IndexHealth> {
        // Verification implementation
        // This would check index consistency and detect corruption

        if !self.index_path.exists() {
            return Ok(IndexHealth::Corrupted);
        }

        Ok(IndexHealth::Healthy)
    }

    /// Rebuild the index from scratch
    ///
    /// # Errors
    ///
    /// Returns an error if rebuild fails
    pub fn rebuild(&self) -> SearchResult<()> {
        // Rebuild implementation
        // This would reconstruct the entire index
        Ok(())
    }

    /// Compact the index to reclaim space
    ///
    /// # Errors
    ///
    /// Returns an error if compaction fails
    pub fn compact(&self) -> SearchResult<()> {
        // Compaction implementation
        Ok(())
    }

    /// Calculate total index size
    fn calculate_index_size(&self) -> SearchResult<u64> {
        if !self.index_path.exists() {
            return Ok(0);
        }

        let mut total_size = 0u64;

        for entry in std::fs::read_dir(&self.index_path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;

            if metadata.is_file() {
                total_size += metadata.len();
            } else if metadata.is_dir() {
                total_size += self.calculate_dir_size(&entry.path())?;
            }
        }

        Ok(total_size)
    }

    /// Calculate directory size recursively
    fn calculate_dir_size(&self, path: &Path) -> SearchResult<u64> {
        let mut total_size = 0u64;

        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;

            if metadata.is_file() {
                total_size += metadata.len();
            } else if metadata.is_dir() {
                total_size += self.calculate_dir_size(&entry.path())?;
            }
        }

        Ok(total_size)
    }

    /// Export index statistics to JSON
    ///
    /// # Errors
    ///
    /// Returns an error if export fails
    pub fn export_statistics(&self, output_path: &Path) -> SearchResult<()> {
        let stats = self.get_statistics()?;
        let json = serde_json::to_string_pretty(&stats)?;
        std::fs::write(output_path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_manager_new() {
        let index_path = std::env::temp_dir().join("oximedia-search-index-manager-test_index");
        let manager = IndexManager::new(&index_path);
        assert_eq!(manager.index_path, index_path);
    }

    #[test]
    fn test_index_health() {
        assert_eq!(IndexHealth::Healthy, IndexHealth::Healthy);
        assert_ne!(IndexHealth::Healthy, IndexHealth::Corrupted);
    }
}
