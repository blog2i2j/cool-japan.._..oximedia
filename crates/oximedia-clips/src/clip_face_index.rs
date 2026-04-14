//! Face-based clip search and grouping.
//!
//! This module builds an in-memory face index that maps detected face
//! *identities* to the clips they appear in.  Face embeddings are
//! represented as fixed-length float vectors; similarity is measured with
//! cosine distance.  A nearest-neighbour search identifies the closest known
//! identity for an unknown probe embedding.
//!
//! # Example
//!
//! ```rust
//! use oximedia_clips::clip_face_index::{ClipFaceIndex, FaceEntry};
//!
//! let mut index = ClipFaceIndex::new(128);
//!
//! // Register a known identity.
//! index.add_identity("alice".to_string(), vec![0.1_f32; 128]);
//!
//! // Index a clip that contains Alice.
//! let entry = FaceEntry {
//!     clip_id: "clip-01".to_string(),
//!     frame_number: 120,
//!     bounding_box: (10, 20, 80, 90),
//!     embedding: vec![0.1_f32; 128],
//!     confidence: 0.97,
//! };
//! index.add_face(entry);
//!
//! // Search for clips containing Alice.
//! let clips = index.clips_by_identity("alice", 0.9);
//! assert_eq!(clips.len(), 1);
//! assert_eq!(clips[0], "clip-01");
//! ```

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: cosine similarity
// ─────────────────────────────────────────────────────────────────────────────

/// Compute cosine similarity between two equal-length vectors.
/// Returns a value in `[0.0, 1.0]` (1.0 = identical direction).
/// Returns `0.0` if either vector has zero norm.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

/// A detected face occurrence within a clip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceEntry {
    /// ID of the clip this face was detected in.
    pub clip_id: String,
    /// Frame number (0-based) where the face was detected.
    pub frame_number: u64,
    /// Bounding box `(x, y, width, height)` in pixels.
    pub bounding_box: (u32, u32, u32, u32),
    /// Face embedding vector of length `dim` (normalised or raw).
    pub embedding: Vec<f32>,
    /// Detection confidence `[0.0, 1.0]`.
    pub confidence: f32,
}

/// Match result from a face identity search.
#[derive(Debug, Clone)]
pub struct FaceMatch {
    /// Matched identity label (or `"unknown"` if unrecognised).
    pub identity: String,
    /// Similarity score `[0.0, 1.0]`.
    pub similarity: f32,
    /// Clip ID where the face was found.
    pub clip_id: String,
    /// Frame number.
    pub frame_number: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// ClipFaceIndex
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory face index for clip search and grouping.
///
/// The index stores:
/// - A gallery of *known identities*, each represented by a centroid embedding.
/// - A list of *face entries* observed in clips, each with its own embedding.
///
/// Identity matching uses cosine similarity with a configurable threshold.
pub struct ClipFaceIndex {
    /// Embedding dimensionality.
    dim: usize,
    /// Gallery: identity label → centroid embedding.
    gallery: HashMap<String, Vec<f32>>,
    /// All indexed face entries.
    entries: Vec<FaceEntry>,
}

impl ClipFaceIndex {
    /// Create a new index for embeddings of the given dimension.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            gallery: HashMap::new(),
            entries: Vec::new(),
        }
    }

    /// Register a known identity with a reference embedding.
    ///
    /// If the identity already exists the embedding is **updated** to the new
    /// value (not averaged).  For a running mean, use [`update_identity`].
    pub fn add_identity(&mut self, label: String, embedding: Vec<f32>) {
        self.gallery.insert(label, embedding);
    }

    /// Update an existing identity by averaging its current centroid with a
    /// new sample (exponential moving average with α = 0.1).
    pub fn update_identity(&mut self, label: &str, new_sample: &[f32]) {
        if new_sample.len() != self.dim {
            return;
        }
        if let Some(centroid) = self.gallery.get_mut(label) {
            for (c, s) in centroid.iter_mut().zip(new_sample.iter()) {
                *c = *c * 0.9 + s * 0.1;
            }
        } else {
            self.gallery.insert(label.to_string(), new_sample.to_vec());
        }
    }

    /// Remove an identity from the gallery.
    pub fn remove_identity(&mut self, label: &str) {
        self.gallery.remove(label);
    }

    /// List all known identity labels.
    #[must_use]
    pub fn identity_labels(&self) -> Vec<&str> {
        self.gallery.keys().map(String::as_str).collect()
    }

    /// Index a detected face entry.
    pub fn add_face(&mut self, entry: FaceEntry) {
        self.entries.push(entry);
    }

    /// Index multiple face entries.
    pub fn add_faces(&mut self, entries: impl IntoIterator<Item = FaceEntry>) {
        self.entries.extend(entries);
    }

    /// Remove all entries for a specific clip.
    pub fn remove_clip(&mut self, clip_id: &str) {
        self.entries.retain(|e| e.clip_id != clip_id);
    }

    /// Total number of indexed face entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    // ── Nearest-identity lookup ──────────────────────────────────────────────

    /// Find the nearest known identity to a probe embedding.
    ///
    /// Returns `(identity_label, similarity)` or `None` if the gallery is
    /// empty or the best similarity is below `min_similarity`.
    #[must_use]
    pub fn nearest_identity(&self, probe: &[f32], min_similarity: f32) -> Option<(&str, f32)> {
        if probe.len() != self.dim {
            return None;
        }
        let mut best_label: Option<&str> = None;
        let mut best_sim = min_similarity;
        for (label, centroid) in &self.gallery {
            let sim = cosine_similarity(probe, centroid);
            if sim > best_sim {
                best_sim = sim;
                best_label = Some(label);
            }
        }
        best_label.map(|l| (l, best_sim))
    }

    // ── Clip queries ──────────────────────────────────────────────────────────

    /// Return all clip IDs that contain the given identity.
    ///
    /// Each entry's embedding is compared against the gallery centroid for
    /// `identity`. Entries with `similarity >= min_similarity` are included.
    /// The returned clip IDs are deduplicated.
    #[must_use]
    pub fn clips_by_identity(&self, identity: &str, min_similarity: f32) -> Vec<String> {
        let centroid = match self.gallery.get(identity) {
            Some(c) => c,
            None => return Vec::new(),
        };
        let mut clips: Vec<String> = self
            .entries
            .iter()
            .filter(|e| cosine_similarity(&e.embedding, centroid) >= min_similarity)
            .map(|e| e.clip_id.clone())
            .collect();
        clips.sort_unstable();
        clips.dedup();
        clips
    }

    /// Return all face entries for a specific clip.
    #[must_use]
    pub fn entries_for_clip(&self, clip_id: &str) -> Vec<&FaceEntry> {
        self.entries
            .iter()
            .filter(|e| e.clip_id == clip_id)
            .collect()
    }

    /// Search for faces similar to `probe` across all clips.
    ///
    /// Returns `FaceMatch` records sorted by descending similarity.
    #[must_use]
    pub fn search_similar(&self, probe: &[f32], min_similarity: f32) -> Vec<FaceMatch> {
        if probe.len() != self.dim {
            return Vec::new();
        }
        let mut matches: Vec<FaceMatch> = self
            .entries
            .iter()
            .filter_map(|e| {
                let sim = cosine_similarity(probe, &e.embedding);
                if sim < min_similarity {
                    return None;
                }
                // Try to identify the face
                let identity = self
                    .nearest_identity(probe, min_similarity)
                    .map(|(l, _)| l.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                Some(FaceMatch {
                    identity,
                    similarity: sim,
                    clip_id: e.clip_id.clone(),
                    frame_number: e.frame_number,
                })
            })
            .collect();
        matches.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        matches
    }

    /// Group all indexed clips by their dominant identity.
    ///
    /// For each clip, the identity with the highest average similarity across
    /// all entries is selected. Unrecognised clips (no identity above threshold)
    /// are grouped under `"unknown"`.
    #[must_use]
    pub fn group_clips_by_identity(&self, min_similarity: f32) -> HashMap<String, Vec<String>> {
        let mut clip_identity_sums: HashMap<String, HashMap<String, (f32, u32)>> = HashMap::new();

        for entry in &self.entries {
            let cell = clip_identity_sums
                .entry(entry.clip_id.clone())
                .or_default();
            for (label, centroid) in &self.gallery {
                let sim = cosine_similarity(&entry.embedding, centroid);
                if sim >= min_similarity {
                    let acc = cell.entry(label.clone()).or_insert((0.0, 0));
                    acc.0 += sim;
                    acc.1 += 1;
                }
            }
        }

        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        for (clip_id, identity_sums) in &clip_identity_sums {
            let best = identity_sums
                .iter()
                .max_by(|(_, a), (_, b)| {
                    let avg_a = a.0 / a.1 as f32;
                    let avg_b = b.0 / b.1 as f32;
                    avg_a.total_cmp(&avg_b)
                })
                .map(|(label, _)| label.clone())
                .unwrap_or_else(|| "unknown".to_string());
            groups.entry(best).or_default().push(clip_id.clone());
        }
        // Clips with no matching identity
        for entry in &self.entries {
            if !clip_identity_sums.contains_key(&entry.clip_id) {
                groups
                    .entry("unknown".to_string())
                    .or_default()
                    .push(entry.clip_id.clone());
            }
        }
        // Deduplicate
        for clips in groups.values_mut() {
            clips.sort_unstable();
            clips.dedup();
        }
        groups
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dim: usize, val: f32) -> Vec<f32> {
        vec![val; dim]
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0_f32, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert!(cosine_similarity(&a, &b) < 1e-6);
    }

    #[test]
    fn test_add_identity_and_nearest() {
        let mut idx = ClipFaceIndex::new(4);
        idx.add_identity("alice".to_string(), vec![1.0, 0.0, 0.0, 0.0]);
        idx.add_identity("bob".to_string(), vec![0.0, 1.0, 0.0, 0.0]);

        let probe_alice = vec![0.99_f32, 0.01, 0.0, 0.0];
        let (label, sim) = idx.nearest_identity(&probe_alice, 0.5).expect("should match");
        assert_eq!(label, "alice");
        assert!(sim > 0.9);
    }

    #[test]
    fn test_clips_by_identity() {
        let mut idx = ClipFaceIndex::new(2);
        idx.add_identity("alice".to_string(), vec![1.0, 0.0]);

        idx.add_face(FaceEntry {
            clip_id: "clip-01".to_string(),
            frame_number: 10,
            bounding_box: (0, 0, 100, 100),
            embedding: vec![1.0, 0.0],
            confidence: 0.99,
        });
        idx.add_face(FaceEntry {
            clip_id: "clip-02".to_string(),
            frame_number: 5,
            bounding_box: (0, 0, 100, 100),
            embedding: vec![0.0, 1.0], // different person
            confidence: 0.95,
        });

        let clips = idx.clips_by_identity("alice", 0.9);
        assert_eq!(clips, vec!["clip-01"]);
    }

    #[test]
    fn test_remove_clip() {
        let mut idx = ClipFaceIndex::new(2);
        idx.add_face(FaceEntry {
            clip_id: "clip-01".to_string(),
            frame_number: 0,
            bounding_box: (0, 0, 10, 10),
            embedding: vec![1.0, 0.0],
            confidence: 0.9,
        });
        assert_eq!(idx.entry_count(), 1);
        idx.remove_clip("clip-01");
        assert_eq!(idx.entry_count(), 0);
    }

    #[test]
    fn test_search_similar() {
        let mut idx = ClipFaceIndex::new(2);
        idx.add_face(FaceEntry {
            clip_id: "clip-a".to_string(),
            frame_number: 0,
            bounding_box: (0, 0, 10, 10),
            embedding: vec![1.0, 0.0],
            confidence: 0.95,
        });
        idx.add_face(FaceEntry {
            clip_id: "clip-b".to_string(),
            frame_number: 0,
            bounding_box: (0, 0, 10, 10),
            embedding: vec![0.0, 1.0],
            confidence: 0.95,
        });

        let probe = vec![1.0_f32, 0.0];
        let results = idx.search_similar(&probe, 0.9);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].clip_id, "clip-a");
    }

    #[test]
    fn test_update_identity_ema() {
        let mut idx = ClipFaceIndex::new(2);
        idx.add_identity("charlie".to_string(), vec![1.0, 0.0]);
        idx.update_identity("charlie", &[0.0, 1.0]);
        let centroid = idx.gallery.get("charlie").expect("should exist");
        // After one EMA step: [0.9, 0.1]
        assert!((centroid[0] - 0.9).abs() < 1e-5);
        assert!((centroid[1] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_group_clips_by_identity() {
        let mut idx = ClipFaceIndex::new(2);
        idx.add_identity("alice".to_string(), vec![1.0, 0.0]);
        idx.add_identity("bob".to_string(), vec![0.0, 1.0]);
        idx.add_face(FaceEntry {
            clip_id: "c1".to_string(), frame_number: 0,
            bounding_box: (0,0,0,0), embedding: vec![1.0, 0.0], confidence: 0.9,
        });
        idx.add_face(FaceEntry {
            clip_id: "c2".to_string(), frame_number: 0,
            bounding_box: (0,0,0,0), embedding: vec![0.0, 1.0], confidence: 0.9,
        });
        let groups = idx.group_clips_by_identity(0.9);
        let alice_clips = groups.get("alice").cloned().unwrap_or_default();
        let bob_clips = groups.get("bob").cloned().unwrap_or_default();
        assert!(alice_clips.contains(&"c1".to_string()));
        assert!(bob_clips.contains(&"c2".to_string()));
    }

    #[test]
    fn test_entries_for_clip() {
        let mut idx = ClipFaceIndex::new(2);
        idx.add_face(FaceEntry {
            clip_id: "clip-x".to_string(), frame_number: 1,
            bounding_box: (0,0,0,0), embedding: vec![1.0, 0.0], confidence: 0.8,
        });
        idx.add_face(FaceEntry {
            clip_id: "clip-x".to_string(), frame_number: 2,
            bounding_box: (0,0,0,0), embedding: vec![0.9, 0.1], confidence: 0.8,
        });
        idx.add_face(FaceEntry {
            clip_id: "clip-y".to_string(), frame_number: 0,
            bounding_box: (0,0,0,0), embedding: vec![0.0, 1.0], confidence: 0.8,
        });
        assert_eq!(idx.entries_for_clip("clip-x").len(), 2);
        assert_eq!(idx.entries_for_clip("clip-y").len(), 1);
    }
}
