//! Content vector representation for similarity calculations.

use super::features::ContentFeatures;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Content vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentVector {
    /// Dense numerical features
    pub dense_features: Vec<f32>,
    /// Sparse categorical features (feature -> weight)
    pub sparse_features: HashMap<String, f32>,
    /// Dimension of dense vector
    pub dimension: usize,
}

impl ContentVector {
    /// Create a new content vector
    #[must_use]
    pub fn new(dense_features: Vec<f32>) -> Self {
        let dimension = dense_features.len();
        Self {
            dense_features,
            sparse_features: HashMap::new(),
            dimension,
        }
    }

    /// Create from features
    #[must_use]
    pub fn from_features(features: &ContentFeatures) -> Self {
        let dense = features.numerical_features();
        let categorical = features.categorical_features();

        let mut sparse = HashMap::new();
        for category in categorical {
            sparse.insert(category, 1.0);
        }

        let dimension = dense.len();
        Self {
            dense_features: dense,
            sparse_features: sparse,
            dimension,
        }
    }

    /// Get dense features as a cloned Vec
    #[must_use]
    pub fn as_vec(&self) -> Vec<f32> {
        self.dense_features.clone()
    }

    /// Get dense features as a slice
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.dense_features
    }

    /// Calculate L2 norm
    #[must_use]
    pub fn norm(&self) -> f32 {
        self.dense_features
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
    }

    /// Normalize vector to unit length
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > f32::EPSILON {
            for feature in &mut self.dense_features {
                *feature /= norm;
            }
        }
    }

    /// Get Jaccard similarity for categorical features
    #[must_use]
    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        if self.sparse_features.is_empty() && other.sparse_features.is_empty() {
            return 0.0;
        }

        let self_features: std::collections::HashSet<_> = self.sparse_features.keys().collect();
        let other_features: std::collections::HashSet<_> = other.sparse_features.keys().collect();

        let intersection = self_features.intersection(&other_features).count();
        let union = self_features.union(&other_features).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Combine multiple vectors with weights
    #[must_use]
    pub fn weighted_average(vectors: &[(Self, f32)]) -> Self {
        if vectors.is_empty() {
            return Self::new(Vec::new());
        }

        let dimension = vectors[0].0.dimension;
        let mut combined_dense = vec![0.0; dimension];
        let mut combined_sparse = HashMap::new();
        let mut total_weight = 0.0;

        for (vector, weight) in vectors {
            total_weight += weight;

            // Combine dense features
            for (i, &value) in vector.dense_features.iter().enumerate() {
                if i < combined_dense.len() {
                    combined_dense[i] += value * weight;
                }
            }

            // Combine sparse features
            for (feature, &value) in &vector.sparse_features {
                *combined_sparse.entry(feature.clone()).or_insert(0.0) += value * weight;
            }
        }

        // Normalize by total weight
        if total_weight > f32::EPSILON {
            for value in &mut combined_dense {
                *value /= total_weight;
            }
            for value in combined_sparse.values_mut() {
                *value /= total_weight;
            }
        }

        Self {
            dense_features: combined_dense,
            sparse_features: combined_sparse,
            dimension,
        }
    }
}

impl Default for ContentVector {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_vector_creation() {
        let vector = ContentVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(vector.dimension, 3);
        assert_eq!(vector.dense_features.len(), 3);
    }

    #[test]
    fn test_vector_norm() {
        let vector = ContentVector::new(vec![3.0, 4.0]);
        assert!((vector.norm() - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vector_normalize() {
        let mut vector = ContentVector::new(vec![3.0, 4.0]);
        vector.normalize();
        assert!((vector.norm() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_similarity() {
        let mut vec1 = ContentVector::new(vec![]);
        vec1.sparse_features.insert(String::from("a"), 1.0);
        vec1.sparse_features.insert(String::from("b"), 1.0);

        let mut vec2 = ContentVector::new(vec![]);
        vec2.sparse_features.insert(String::from("b"), 1.0);
        vec2.sparse_features.insert(String::from("c"), 1.0);

        let similarity = vec1.jaccard_similarity(&vec2);
        assert!((similarity - 0.333_333_34).abs() < 0.001);
    }

    #[test]
    fn test_weighted_average() {
        let vec1 = ContentVector::new(vec![1.0, 2.0]);
        let vec2 = ContentVector::new(vec![3.0, 4.0]);

        let combined = ContentVector::weighted_average(&[(vec1, 0.5), (vec2, 0.5)]);
        assert_eq!(combined.dense_features.len(), 2);
        assert!((combined.dense_features[0] - 2.0).abs() < f32::EPSILON);
        assert!((combined.dense_features[1] - 3.0).abs() < f32::EPSILON);
    }
}
