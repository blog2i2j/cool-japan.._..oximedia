//! Reverse video search.
//!
//! Indexes video key frames by perceptual hash and searches for the
//! source video given a query frame.

use crate::error::SearchResult;
use crate::visual::features::FeatureExtractor;
use uuid::Uuid;

/// Reverse video search result.
#[derive(Debug, Clone)]
pub struct ReverseVideoResult {
    /// Asset ID.
    pub asset_id: Uuid,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
    /// Timestamp in source video (ms).
    pub timestamp_ms: i64,
}

/// A stored keyframe entry.
#[derive(Debug, Clone)]
struct KeyframeEntry {
    asset_id: Uuid,
    timestamp_ms: i64,
    phash: Vec<u8>,
}

/// Reverse video search engine.
///
/// Indexes video keyframes by perceptual hash and finds the source
/// video + timestamp for a given query frame.
pub struct ReverseVideoSearch {
    entries: Vec<KeyframeEntry>,
    extractor: FeatureExtractor,
}

impl ReverseVideoSearch {
    /// Create a new reverse video search engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            extractor: FeatureExtractor::new(),
        }
    }

    /// Add a keyframe to the index.
    ///
    /// # Errors
    ///
    /// Returns an error if perceptual hash computation fails.
    pub fn add_keyframe(
        &mut self,
        asset_id: Uuid,
        timestamp_ms: i64,
        frame_data: &[u8],
    ) -> SearchResult<()> {
        let phash = self.extractor.compute_phash(frame_data)?;
        self.entries.push(KeyframeEntry {
            asset_id,
            timestamp_ms,
            phash,
        });
        Ok(())
    }

    /// Find source video from a sample frame.
    ///
    /// Returns results sorted by confidence descending.
    ///
    /// # Errors
    ///
    /// Returns an error if search fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn search_frame(&self, frame_data: &[u8]) -> SearchResult<Vec<ReverseVideoResult>> {
        let query_hash = self.extractor.compute_phash(frame_data)?;
        let max_bits = (query_hash.len() * 8) as f32;

        if max_bits < 1.0 {
            return Ok(Vec::new());
        }

        let mut results: Vec<ReverseVideoResult> = self
            .entries
            .iter()
            .map(|entry| {
                let distance = FeatureExtractor::phash_distance(&query_hash, &entry.phash);
                let confidence = 1.0 - (distance as f32 / max_bits);
                ReverseVideoResult {
                    asset_id: entry.asset_id,
                    confidence,
                    timestamp_ms: entry.timestamp_ms,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Return the number of indexed keyframes.
    #[must_use]
    pub fn index_size(&self) -> usize {
        self.entries.len()
    }
}

impl Default for ReverseVideoSearch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_video_empty() {
        let search = ReverseVideoSearch::new();
        let results = search.search_frame(&[]).expect("should succeed in test");
        assert!(results.is_empty());
    }

    #[test]
    fn test_reverse_video_add_and_search() {
        let mut search = ReverseVideoSearch::new();
        let frame = vec![100u8; 16 * 16 * 3];
        let id = Uuid::new_v4();
        search
            .add_keyframe(id, 5000, &frame)
            .expect("should succeed in test");

        let results = search.search_frame(&frame).expect("should succeed in test");
        assert!(!results.is_empty());
        assert_eq!(results[0].asset_id, id);
        assert_eq!(results[0].timestamp_ms, 5000);
        assert!((results[0].confidence - 1.0).abs() < f32::EPSILON);
    }
}
