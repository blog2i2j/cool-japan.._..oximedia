//! Face-based search implementation.

use crate::error::SearchResult;
use crate::index::builder::FaceDescriptor;
use std::path::Path;
use uuid::Uuid;

/// Face index
pub struct FaceIndex {
    index_path: std::path::PathBuf,
    faces: Vec<(Uuid, Vec<FaceDescriptor>)>,
}

impl FaceIndex {
    /// Create a new face index
    ///
    /// # Errors
    ///
    /// Returns an error if index creation fails
    pub fn new(index_path: &Path) -> SearchResult<Self> {
        if !index_path.exists() {
            std::fs::create_dir_all(index_path)?;
        }

        Ok(Self {
            index_path: index_path.to_path_buf(),
            faces: Vec::new(),
        })
    }

    /// Add faces for an asset
    ///
    /// # Errors
    ///
    /// Returns an error if indexing fails
    pub fn add_faces(&mut self, asset_id: Uuid, faces: &[FaceDescriptor]) -> SearchResult<()> {
        self.faces.push((asset_id, faces.to_vec()));
        Ok(())
    }

    /// Search for similar faces using Euclidean distance on face embeddings.
    ///
    /// Returns asset IDs that contain at least one face within distance
    /// threshold 1.0 of the query embedding, sorted by closest match.
    ///
    /// # Errors
    ///
    /// Returns an error if search fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn search_similar(&self, embedding: &[f32]) -> SearchResult<Vec<Uuid>> {
        if embedding.is_empty() {
            return Ok(Vec::new());
        }

        let threshold = 1.0_f32;
        let mut matches: Vec<(Uuid, f32)> = Vec::new();

        for (asset_id, descriptors) in &self.faces {
            let min_dist = descriptors
                .iter()
                .map(|desc| {
                    // Euclidean distance between query and stored embedding.
                    desc.embedding
                        .iter()
                        .zip(embedding.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt()
                })
                .fold(f32::MAX, f32::min);

            if min_dist <= threshold {
                matches.push((*asset_id, min_dist));
            }
        }

        // Sort by distance ascending.
        matches.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(matches.into_iter().map(|(id, _)| id).collect())
    }

    /// Commit changes
    ///
    /// # Errors
    ///
    /// Returns an error if commit fails
    pub fn commit(&self) -> SearchResult<()> {
        Ok(())
    }

    /// Delete faces for an asset
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails
    pub fn delete(&mut self, asset_id: Uuid) -> SearchResult<()> {
        self.faces.retain(|(id, _)| *id != asset_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_index() {
        let temp_dir = std::env::temp_dir().join("face_index_test");
        let index = FaceIndex::new(&temp_dir).expect("should succeed in test");
        assert!(index.faces.is_empty());
        std::fs::remove_dir_all(temp_dir).ok();
    }
}
