//! Reverse image search.
//!
//! Uses perceptual hashing (aHash) to find visually similar images in
//! an indexed database.  The search computes the perceptual hash of
//! the query image and compares it against stored hashes using Hamming
//! distance, returning results sorted by similarity.

use crate::error::SearchResult;
use crate::visual::features::FeatureExtractor;
use uuid::Uuid;

/// Reverse image search result.
#[derive(Debug, Clone)]
pub struct ReverseImageResult {
    /// Asset ID.
    pub asset_id: Uuid,
    /// Similarity score (0.0 to 1.0, higher = more similar).
    pub similarity: f32,
}

/// Reverse image search engine using perceptual hashing.
pub struct ReverseImageSearch {
    /// Indexed perceptual hashes: `(asset_id, phash_bytes)`.
    entries: Vec<(Uuid, Vec<u8>)>,
    /// Feature extractor for computing perceptual hashes.
    extractor: FeatureExtractor,
}

impl ReverseImageSearch {
    /// Create a new reverse image search engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            extractor: FeatureExtractor::new(),
        }
    }

    /// Add an image to the search index.
    ///
    /// # Errors
    ///
    /// Returns an error if perceptual hash computation fails.
    pub fn add_image(&mut self, asset_id: Uuid, image_data: &[u8]) -> SearchResult<()> {
        let phash = self.extractor.compute_phash(image_data)?;
        self.entries.push((asset_id, phash));
        Ok(())
    }

    /// Add a pre-computed perceptual hash to the index.
    pub fn add_hash(&mut self, asset_id: Uuid, phash: Vec<u8>) {
        self.entries.push((asset_id, phash));
    }

    /// Find similar images by computing the perceptual hash of
    /// `image_data` and comparing against all indexed hashes.
    ///
    /// Returns results sorted by similarity (descending).
    ///
    /// # Errors
    ///
    /// Returns an error if search fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn search(&self, image_data: &[u8]) -> SearchResult<Vec<ReverseImageResult>> {
        let query_hash = self.extractor.compute_phash(image_data)?;
        let max_bits = (query_hash.len() * 8) as f32;

        if max_bits < 1.0 {
            return Ok(Vec::new());
        }

        let mut results: Vec<ReverseImageResult> = self
            .entries
            .iter()
            .map(|(id, stored_hash)| {
                let distance = FeatureExtractor::phash_distance(&query_hash, stored_hash);
                let similarity = 1.0 - (distance as f32 / max_bits);
                ReverseImageResult {
                    asset_id: *id,
                    similarity,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Return the number of indexed images.
    #[must_use]
    pub fn index_size(&self) -> usize {
        self.entries.len()
    }
}

impl Default for ReverseImageSearch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_search_empty_index() {
        let search = ReverseImageSearch::new();
        let results = search.search(&[]).expect("should succeed in test");
        assert!(results.is_empty());
    }

    #[test]
    fn test_reverse_search_identical_image() {
        let mut search = ReverseImageSearch::new();
        let image = vec![128u8; 16 * 16 * 3];
        let id = Uuid::new_v4();
        search
            .add_image(id, &image)
            .expect("should succeed in test");

        let results = search.search(&image).expect("should succeed in test");
        assert!(!results.is_empty());
        assert_eq!(results[0].asset_id, id);
        assert!((results[0].similarity - 1.0).abs() < f32::EPSILON);
    }
}
