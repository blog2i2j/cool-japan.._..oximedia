//! Content-based filtering utilities.
//!
//! Lightweight similarity functions operating directly on content metadata
//! (tags, categories, text features) without requiring user interaction data.
//!
//! These primitives complement the richer `content::similarity` module by
//! providing quick overlap-based similarity for cold-start scenarios or when
//! full embeddings are unavailable.

#![allow(dead_code)]

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// ContentSimilarity
// ---------------------------------------------------------------------------

/// Stateless helper for computing set-based content similarity.
///
/// All methods are pure functions, so the struct carries no state and may be
/// constructed freely.
pub struct ContentSimilarity;

impl ContentSimilarity {
    /// Creates a new [`ContentSimilarity`] helper.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Jaccard similarity between two tag lists.
    ///
    /// Returns the size of the intersection divided by the size of the union
    /// of the two tag sets.  Returns `0.0` when both lists are empty.
    ///
    /// Duplicate tags within a single list are deduplicated before comparison.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_recommend::content_filter::ContentSimilarity;
    ///
    /// let sim = ContentSimilarity::jaccard(&["action", "sci-fi"], &["action", "horror"]);
    /// // intersection = {"action"}, union = {"action","sci-fi","horror"} → 1/3
    /// assert!((sim - 1.0/3.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn jaccard(tags_a: &[&str], tags_b: &[&str]) -> f32 {
        let set_a: HashSet<&str> = tags_a.iter().copied().collect();
        let set_b: HashSet<&str> = tags_b.iter().copied().collect();

        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        if union == 0 {
            return 0.0;
        }

        intersection as f32 / union as f32
    }

    /// Dice similarity coefficient between two tag lists.
    ///
    /// `2 * |A ∩ B| / (|A| + |B|)` — tends to weight similarity higher than
    /// Jaccard for sets with near-equal sizes.
    ///
    /// Returns `0.0` when both lists are empty.
    #[must_use]
    pub fn dice(tags_a: &[&str], tags_b: &[&str]) -> f32 {
        let set_a: HashSet<&str> = tags_a.iter().copied().collect();
        let set_b: HashSet<&str> = tags_b.iter().copied().collect();

        let intersection = set_a.intersection(&set_b).count();
        let total = set_a.len() + set_b.len();

        if total == 0 {
            return 0.0;
        }

        2.0 * intersection as f32 / total as f32
    }

    /// Overlap coefficient (Szymkiewicz–Simpson) between two tag lists.
    ///
    /// `|A ∩ B| / min(|A|, |B|)` — returns `1.0` when the smaller set is a
    /// perfect subset of the larger.  Returns `0.0` when either set is empty.
    #[must_use]
    pub fn overlap(tags_a: &[&str], tags_b: &[&str]) -> f32 {
        let set_a: HashSet<&str> = tags_a.iter().copied().collect();
        let set_b: HashSet<&str> = tags_b.iter().copied().collect();

        let min_size = set_a.len().min(set_b.len());
        if min_size == 0 {
            return 0.0;
        }

        let intersection = set_a.intersection(&set_b).count();
        intersection as f32 / min_size as f32
    }

    /// Cosine similarity over a binary tag-vector representation.
    ///
    /// Builds the union vocabulary, encodes each tag list as a binary vector,
    /// then computes the standard cosine similarity.  Returns `0.0` when
    /// either list is empty.
    #[must_use]
    pub fn cosine_binary(tags_a: &[&str], tags_b: &[&str]) -> f32 {
        let set_a: HashSet<&str> = tags_a.iter().copied().collect();
        let set_b: HashSet<&str> = tags_b.iter().copied().collect();

        if set_a.is_empty() || set_b.is_empty() {
            return 0.0;
        }

        let dot = set_a.intersection(&set_b).count() as f32;
        let norm_a = (set_a.len() as f32).sqrt();
        let norm_b = (set_b.len() as f32).sqrt();

        dot / (norm_a * norm_b)
    }
}

impl Default for ContentSimilarity {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_identical() {
        let sim = ContentSimilarity::jaccard(&["a", "b", "c"], &["a", "b", "c"]);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical sets: expected 1.0, got {sim}"
        );
    }

    #[test]
    fn test_jaccard_disjoint() {
        let sim = ContentSimilarity::jaccard(&["a", "b"], &["c", "d"]);
        assert!(sim.abs() < 1e-6, "disjoint sets: expected 0.0, got {sim}");
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let sim = ContentSimilarity::jaccard(&["action", "sci-fi"], &["action", "horror"]);
        let expected = 1.0_f32 / 3.0;
        assert!(
            (sim - expected).abs() < 1e-6,
            "expected {expected}, got {sim}"
        );
    }

    #[test]
    fn test_jaccard_empty_both() {
        let sim = ContentSimilarity::jaccard(&[], &[]);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_one_empty() {
        let sim = ContentSimilarity::jaccard(&["a"], &[]);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_deduplicates() {
        // Duplicate "a" in both lists shouldn't inflate the score
        let sim_dup = ContentSimilarity::jaccard(&["a", "a", "b"], &["a", "b"]);
        let sim_clean = ContentSimilarity::jaccard(&["a", "b"], &["a", "b"]);
        assert!(
            (sim_dup - sim_clean).abs() < 1e-6,
            "duplicates should be ignored"
        );
    }

    #[test]
    fn test_dice_identical() {
        let sim = ContentSimilarity::dice(&["x", "y"], &["x", "y"]);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dice_disjoint() {
        let sim = ContentSimilarity::dice(&["a"], &["b"]);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_overlap_subset() {
        // {"a"} ⊆ {"a","b","c"} → overlap = 1.0
        let sim = ContentSimilarity::overlap(&["a"], &["a", "b", "c"]);
        assert!((sim - 1.0).abs() < 1e-6, "subset: expected 1.0, got {sim}");
    }

    #[test]
    fn test_cosine_binary_identical() {
        let sim = ContentSimilarity::cosine_binary(&["a", "b"], &["a", "b"]);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_binary_orthogonal() {
        let sim = ContentSimilarity::cosine_binary(&["a"], &["b"]);
        assert!(sim.abs() < 1e-6);
    }
}
