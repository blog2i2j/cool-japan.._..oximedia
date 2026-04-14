//! Audio similarity metrics for Music Information Retrieval.
//!
//! Provides low-level similarity measures that operate directly on feature
//! vectors and probability distributions, complementing the higher-level
//! [`similarity`](crate::similarity) module which works on structured
//! [`AudioFeatures`](crate::similarity::AudioFeatures) objects.
//!
//! # Algorithms
//!
//! | Algorithm | Use case |
//! |-----------|----------|
//! | [`CosineSimilarity`] | Feature-vector dot-product similarity (tempo, spectral, timbral vectors) |
//! | [`EarthMoverDistance`] | Comparing probability distributions (rhythm patterns, chroma histograms) |
//! | [`SimilarityMatrix`] | All-pairs pairwise similarity for a track collection |
//! | [`AudioSimilaritySearch`] | Ranked nearest-neighbour search against a corpus |
//!
//! # Example
//!
//! ```
//! use oximedia_mir::audio_similarity::{FeatureVector, CosineSimilarity};
//!
//! let a = FeatureVector::new(vec![1.0, 0.0, 0.0]).unwrap();
//! let b = FeatureVector::new(vec![0.0, 1.0, 0.0]).unwrap();
//! let cos = CosineSimilarity::default();
//! assert!((cos.compute(&a, &b).unwrap() - 0.0).abs() < 1e-10);
//! ```

#![allow(dead_code)]

use rayon::prelude::*;

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors produced by audio similarity computations.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum SimilarityError {
    /// The two feature vectors have different lengths.
    DimensionMismatch { left: usize, right: usize },
    /// One or both vectors are empty.
    EmptyVector,
    /// Division by zero: one or both vectors have zero norm.
    ZeroNorm,
    /// The two histograms have different numbers of bins.
    HistogramLengthMismatch { left: usize, right: usize },
    /// A histogram is empty.
    EmptyHistogram,
    /// A histogram contains negative values.
    NegativeHistogramValue,
    /// The corpus contains fewer than 1 track.
    EmptyCorpus,
    /// Top-K requested is zero.
    ZeroTopK,
}

impl std::fmt::Display for SimilarityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { left, right } => {
                write!(f, "dimension mismatch: {left} vs {right}")
            }
            Self::EmptyVector => write!(f, "feature vector is empty"),
            Self::ZeroNorm => write!(f, "cannot compute cosine similarity with zero-norm vector"),
            Self::HistogramLengthMismatch { left, right } => {
                write!(f, "histogram length mismatch: {left} vs {right}")
            }
            Self::EmptyHistogram => write!(f, "histogram is empty"),
            Self::NegativeHistogramValue => write!(f, "histogram contains negative values"),
            Self::EmptyCorpus => write!(f, "corpus is empty"),
            Self::ZeroTopK => write!(f, "top-K must be at least 1"),
        }
    }
}

impl std::error::Error for SimilarityError {}

/// Alias for `Result<T, SimilarityError>`.
pub type SimilarityResult<T> = Result<T, SimilarityError>;

// ── FeatureVector ─────────────────────────────────────────────────────────────

/// A dense real-valued feature vector.
///
/// Constructed via [`FeatureVector::new`] which validates that the input is
/// non-empty and contains only finite values.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureVector {
    data: Vec<f64>,
}

impl FeatureVector {
    /// Create a new feature vector.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityError::EmptyVector`] if `data` is empty.
    pub fn new(data: Vec<f64>) -> SimilarityResult<Self> {
        if data.is_empty() {
            return Err(SimilarityError::EmptyVector);
        }
        Ok(Self { data })
    }

    /// Number of dimensions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the vector has no dimensions (always `false` for a constructed
    /// `FeatureVector` — the constructor rejects empty vectors).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Raw slice of the feature values.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// L2 (Euclidean) norm.
    #[must_use]
    pub fn l2_norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// L1 (Manhattan) norm.
    #[must_use]
    pub fn l1_norm(&self) -> f64 {
        self.data.iter().map(|&x| x.abs()).sum()
    }

    /// Return a new vector normalized to unit L2 norm.
    ///
    /// Returns `None` if the vector has zero norm.
    #[must_use]
    pub fn l2_normalized(&self) -> Option<Self> {
        let norm = self.l2_norm();
        if norm < 1e-15 {
            return None;
        }
        Some(Self {
            data: self.data.iter().map(|&x| x / norm).collect(),
        })
    }

    /// Return a new vector normalized by z-score: subtract mean, divide by
    /// standard deviation.
    ///
    /// Returns `None` if the standard deviation is zero (constant vector).
    #[must_use]
    pub fn z_score_normalized(&self) -> Option<Self> {
        let n = self.data.len() as f64;
        let mean = self.data.iter().sum::<f64>() / n;
        let variance = self.data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        if std < 1e-15 {
            return None;
        }
        Some(Self {
            data: self.data.iter().map(|&x| (x - mean) / std).collect(),
        })
    }
}

// ── CosineSimilarity ──────────────────────────────────────────────────────────

/// Computes cosine similarity between two [`FeatureVector`]s.
///
/// `cosine(a, b) = dot(a, b) / (||a|| * ||b||)`
///
/// Result is in [-1, 1]:
/// - `1.0` → identical direction
/// - `0.0` → orthogonal
/// - `-1.0` → opposite direction
#[derive(Debug, Clone, Default)]
pub struct CosineSimilarity;

impl CosineSimilarity {
    /// Compute cosine similarity.
    ///
    /// # Errors
    ///
    /// - [`SimilarityError::DimensionMismatch`] if vectors differ in length.
    /// - [`SimilarityError::ZeroNorm`] if either vector has zero L2 norm.
    pub fn compute(&self, a: &FeatureVector, b: &FeatureVector) -> SimilarityResult<f64> {
        if a.len() != b.len() {
            return Err(SimilarityError::DimensionMismatch {
                left: a.len(),
                right: b.len(),
            });
        }
        let norm_a = a.l2_norm();
        let norm_b = b.l2_norm();
        if norm_a < 1e-15 || norm_b < 1e-15 {
            return Err(SimilarityError::ZeroNorm);
        }
        let dot: f64 = a
            .as_slice()
            .iter()
            .zip(b.as_slice().iter())
            .map(|(&x, &y)| x * y)
            .sum();
        Ok((dot / (norm_a * norm_b)).clamp(-1.0, 1.0))
    }

    /// Compute cosine distance: `1 - cosine_similarity`.
    ///
    /// Range is [0, 2].
    ///
    /// # Errors
    ///
    /// Same conditions as [`compute`](Self::compute).
    pub fn distance(&self, a: &FeatureVector, b: &FeatureVector) -> SimilarityResult<f64> {
        Ok(1.0 - self.compute(a, b)?)
    }
}

// ── EarthMoverDistance ────────────────────────────────────────────────────────

/// Computes the Earth Mover's Distance (Wasserstein-1 distance) between two
/// 1-D distributions represented as histograms over a uniform grid.
///
/// The linear-time algorithm for equal-length 1-D distributions is used:
/// normalize both histograms to unit mass, then the EMD equals the sum of
/// absolute values of the difference of their cumulative distribution
/// functions.
///
/// ```text
/// EMD = Σ |CDF_a(i) - CDF_b(i)|
/// ```
#[derive(Debug, Clone, Default)]
pub struct EarthMoverDistance;

impl EarthMoverDistance {
    /// Compute the Earth Mover's Distance between two histograms.
    ///
    /// # Arguments
    ///
    /// * `a`, `b` — slices of non-negative values representing bin counts or
    ///   densities. They need not be normalized; the function normalizes them
    ///   internally.
    ///
    /// # Errors
    ///
    /// - [`SimilarityError::EmptyHistogram`] if either histogram is empty.
    /// - [`SimilarityError::HistogramLengthMismatch`] if they have different
    ///   lengths.
    /// - [`SimilarityError::NegativeHistogramValue`] if any value is negative.
    pub fn compute(&self, a: &[f64], b: &[f64]) -> SimilarityResult<f64> {
        if a.is_empty() || b.is_empty() {
            return Err(SimilarityError::EmptyHistogram);
        }
        if a.len() != b.len() {
            return Err(SimilarityError::HistogramLengthMismatch {
                left: a.len(),
                right: b.len(),
            });
        }
        if a.iter().any(|&v| v < 0.0) || b.iter().any(|&v| v < 0.0) {
            return Err(SimilarityError::NegativeHistogramValue);
        }

        let sum_a: f64 = a.iter().sum();
        let sum_b: f64 = b.iter().sum();

        // If both are zero, distance is 0
        if sum_a < 1e-15 && sum_b < 1e-15 {
            return Ok(0.0);
        }
        // Normalize (use 1.0 if sum is zero to avoid NaN; the other histogram
        // handles it correctly through the CDF comparison)
        let norm_a = if sum_a < 1e-15 { 1.0 } else { sum_a };
        let norm_b = if sum_b < 1e-15 { 1.0 } else { sum_b };

        // Compute CDF difference accumulation
        let mut cdf_a = 0.0_f64;
        let mut cdf_b = 0.0_f64;
        let mut emd = 0.0_f64;

        for (&va, &vb) in a.iter().zip(b.iter()) {
            cdf_a += va / norm_a;
            cdf_b += vb / norm_b;
            emd += (cdf_a - cdf_b).abs();
        }

        Ok(emd)
    }

    /// Compute the normalized EMD (divide by number of bins), returning a
    /// value in [0, 1].
    ///
    /// # Errors
    ///
    /// Same conditions as [`compute`](Self::compute).
    pub fn normalized(&self, a: &[f64], b: &[f64]) -> SimilarityResult<f64> {
        let emd = self.compute(a, b)?;
        Ok(emd / a.len() as f64)
    }
}

// ── SimilarityMatrix ──────────────────────────────────────────────────────────

/// Mode for pairwise similarity computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMode {
    /// Use cosine similarity (for feature vectors).
    Cosine,
    /// Use normalized Earth Mover's Distance converted to similarity:
    /// `1 - normalized_emd`.
    EmdSimilarity,
}

/// Pairwise similarity matrix for a collection of feature vectors.
///
/// The matrix is symmetric with 1.0 on the diagonal (each item is maximally
/// similar to itself).
#[derive(Debug, Clone)]
pub struct SimilarityMatrix {
    /// Row-major symmetric matrix.  `matrix[i][j]` = similarity of item i and
    /// item j.
    pub matrix: Vec<Vec<f64>>,
    /// Number of items.
    pub n: usize,
}

impl SimilarityMatrix {
    /// Compute the pairwise similarity matrix for `vectors`.
    ///
    /// The computation is parallelized row-by-row using rayon.
    ///
    /// # Errors
    ///
    /// - [`SimilarityError::EmptyCorpus`] if `vectors` is empty.
    /// - Propagates [`SimilarityError`] from the chosen similarity metric if
    ///   any pair fails (e.g. dimension mismatch, zero norm).
    pub fn compute(
        vectors: &[FeatureVector],
        mode: SimilarityMode,
    ) -> SimilarityResult<Self> {
        if vectors.is_empty() {
            return Err(SimilarityError::EmptyCorpus);
        }
        let n = vectors.len();
        let cos = CosineSimilarity;
        let emd = EarthMoverDistance;

        // Compute upper triangle in parallel, then mirror
        let upper: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0_f64; n];
                for j in i..n {
                    if i == j {
                        row[j] = 1.0;
                    } else {
                        let sim = match mode {
                            SimilarityMode::Cosine => cos
                                .compute(&vectors[i], &vectors[j])
                                .unwrap_or(0.0),
                            SimilarityMode::EmdSimilarity => {
                                let d = emd
                                    .normalized(vectors[i].as_slice(), vectors[j].as_slice())
                                    .unwrap_or(1.0);
                                (1.0 - d).clamp(0.0, 1.0)
                            }
                        };
                        row[j] = sim;
                    }
                }
                row
            })
            .collect();

        // Build full symmetric matrix
        let mut matrix = upper;
        for i in 0..n {
            for j in 0..i {
                let val = matrix[j][i];
                matrix[i][j] = val;
            }
        }

        Ok(Self { matrix, n })
    }

    /// Return the similarity between items `i` and `j`.
    ///
    /// Returns `None` if either index is out of bounds.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> Option<f64> {
        self.matrix.get(i).and_then(|row| row.get(j)).copied()
    }
}

// ── AudioSimilaritySearch ─────────────────────────────────────────────────────

/// A single ranked search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityHit {
    /// Index into the corpus.
    pub index: usize,
    /// Similarity score (higher = more similar).
    pub score: f64,
}

/// Ranked nearest-neighbour search for a query feature vector against a corpus.
#[derive(Debug)]
pub struct AudioSimilaritySearch {
    corpus: Vec<FeatureVector>,
    mode: SimilarityMode,
}

impl AudioSimilaritySearch {
    /// Create a search index from a corpus of feature vectors.
    ///
    /// # Errors
    ///
    /// Returns [`SimilarityError::EmptyCorpus`] if `corpus` is empty.
    pub fn new(corpus: Vec<FeatureVector>, mode: SimilarityMode) -> SimilarityResult<Self> {
        if corpus.is_empty() {
            return Err(SimilarityError::EmptyCorpus);
        }
        Ok(Self { corpus, mode })
    }

    /// Search for the top-`k` most similar items to `query`.
    ///
    /// # Arguments
    ///
    /// * `query` — the query feature vector.
    /// * `k` — number of results to return. Clamped to the corpus size.
    /// * `min_score` — optional minimum similarity threshold; items below this
    ///   value are excluded.
    ///
    /// # Returns
    ///
    /// A vector of [`SimilarityHit`] sorted by descending score, length ≤ `k`.
    ///
    /// # Errors
    ///
    /// - [`SimilarityError::ZeroTopK`] if `k == 0`.
    /// - Propagates metric errors (dimension mismatch, zero norm).
    pub fn search(
        &self,
        query: &FeatureVector,
        k: usize,
        min_score: Option<f64>,
    ) -> SimilarityResult<Vec<SimilarityHit>> {
        if k == 0 {
            return Err(SimilarityError::ZeroTopK);
        }

        let cos = CosineSimilarity;
        let emd = EarthMoverDistance;
        let threshold = min_score.unwrap_or(f64::NEG_INFINITY);

        // Compute scores in parallel
        let scores: Vec<SimilarityResult<f64>> = self
            .corpus
            .par_iter()
            .map(|item| match self.mode {
                SimilarityMode::Cosine => cos.compute(query, item),
                SimilarityMode::EmdSimilarity => {
                    let d = emd.normalized(query.as_slice(), item.as_slice())?;
                    Ok((1.0 - d).clamp(0.0, 1.0))
                }
            })
            .collect();

        // Propagate first error
        let mut hits: Vec<SimilarityHit> = Vec::with_capacity(self.corpus.len());
        for (idx, score_res) in scores.into_iter().enumerate() {
            let score = score_res?;
            if score >= threshold {
                hits.push(SimilarityHit { index: idx, score });
            }
        }

        // Sort descending by score
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(k);
        Ok(hits)
    }

    /// Number of items in the corpus.
    #[must_use]
    pub fn len(&self) -> usize {
        self.corpus.len()
    }

    /// Whether the corpus is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.corpus.is_empty()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fv(data: Vec<f64>) -> FeatureVector {
        FeatureVector::new(data).expect("test vector must not be empty")
    }

    // ── FeatureVector ─────────────────────────────────────────────────────────

    #[test]
    fn test_feature_vector_l2_norm() {
        let v = fv(vec![3.0, 4.0]);
        assert!((v.l2_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_vector_l1_norm() {
        let v = fv(vec![-1.0, 2.0, -3.0]);
        assert!((v.l1_norm() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_vector_l2_normalized() {
        let v = fv(vec![3.0, 4.0]);
        let n = v.l2_normalized().expect("should not be zero");
        assert!((n.l2_norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_vector_empty_rejected() {
        let res = FeatureVector::new(vec![]);
        assert!(matches!(res, Err(SimilarityError::EmptyVector)));
    }

    // ── CosineSimilarity ──────────────────────────────────────────────────────

    #[test]
    fn test_cosine_identical_vectors() {
        let a = fv(vec![1.0, 2.0, 3.0]);
        let cos = CosineSimilarity;
        let sim = cos.compute(&a, &a).expect("should succeed");
        assert!((sim - 1.0).abs() < 1e-10, "identical vectors have sim=1.0");
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = fv(vec![1.0, 0.0, 0.0]);
        let b = fv(vec![0.0, 1.0, 0.0]);
        let cos = CosineSimilarity;
        let sim = cos.compute(&a, &b).expect("should succeed");
        assert!((sim - 0.0).abs() < 1e-10, "orthogonal vectors have sim=0.0");
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = fv(vec![1.0, 0.0]);
        let b = fv(vec![-1.0, 0.0]);
        let cos = CosineSimilarity;
        let sim = cos.compute(&a, &b).expect("should succeed");
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_dimension_mismatch() {
        let a = fv(vec![1.0, 2.0]);
        let b = fv(vec![1.0, 2.0, 3.0]);
        let cos = CosineSimilarity;
        let err = cos.compute(&a, &b).unwrap_err();
        assert!(matches!(err, SimilarityError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_cosine_zero_norm_error() {
        let a = fv(vec![0.0, 0.0]);
        let b = fv(vec![1.0, 0.0]);
        let cos = CosineSimilarity;
        let err = cos.compute(&a, &b).unwrap_err();
        assert!(matches!(err, SimilarityError::ZeroNorm));
    }

    // ── EarthMoverDistance ────────────────────────────────────────────────────

    #[test]
    fn test_emd_identical_distributions() {
        let hist = vec![0.1, 0.4, 0.3, 0.2];
        let emd_calc = EarthMoverDistance;
        let d = emd_calc.compute(&hist, &hist).expect("should succeed");
        assert!(d.abs() < 1e-10, "identical distributions have EMD=0");
    }

    #[test]
    fn test_emd_completely_different() {
        // All mass at bin 0 vs all mass at last bin
        let n = 10;
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        a[0] = 1.0;
        b[n - 1] = 1.0;
        let emd_calc = EarthMoverDistance;
        let d = emd_calc.compute(&a, &b).expect("should succeed");
        // CDF difference accumulates from 0 to n-1
        assert!(d > 0.0, "different distributions should have EMD > 0");
    }

    #[test]
    fn test_emd_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let emd_calc = EarthMoverDistance;
        let err = emd_calc.compute(&a, &b).unwrap_err();
        assert!(matches!(err, SimilarityError::HistogramLengthMismatch { .. }));
    }

    // ── SimilarityMatrix ──────────────────────────────────────────────────────

    #[test]
    fn test_similarity_matrix_diagonal_is_one() {
        let vectors = vec![
            fv(vec![1.0, 0.0, 0.0]),
            fv(vec![0.0, 1.0, 0.0]),
            fv(vec![0.0, 0.0, 1.0]),
        ];
        let mat = SimilarityMatrix::compute(&vectors, SimilarityMode::Cosine)
            .expect("should succeed");
        for i in 0..3 {
            assert!(
                (mat.get(i, i).expect("in bounds") - 1.0).abs() < 1e-10,
                "diagonal should be 1.0"
            );
        }
    }

    #[test]
    fn test_similarity_matrix_is_symmetric() {
        let vectors = vec![
            fv(vec![1.0, 2.0, 3.0]),
            fv(vec![4.0, 5.0, 6.0]),
            fv(vec![7.0, 8.0, 9.0]),
        ];
        let mat = SimilarityMatrix::compute(&vectors, SimilarityMode::Cosine)
            .expect("should succeed");
        for i in 0..3 {
            for j in 0..3 {
                let a = mat.get(i, j).expect("in bounds");
                let b = mat.get(j, i).expect("in bounds");
                assert!(
                    (a - b).abs() < 1e-10,
                    "matrix should be symmetric: [{i}][{j}]={a:.6} != [{j}][{i}]={b:.6}"
                );
            }
        }
    }

    // ── AudioSimilaritySearch ─────────────────────────────────────────────────

    #[test]
    fn test_search_returns_top_k() {
        let corpus = vec![
            fv(vec![1.0, 0.0]),
            fv(vec![0.0, 1.0]),
            fv(vec![1.0, 1.0]),
            fv(vec![-1.0, 0.0]),
        ];
        let search = AudioSimilaritySearch::new(corpus, SimilarityMode::Cosine)
            .expect("should succeed");
        let query = fv(vec![1.0, 0.0]);
        let hits = search.search(&query, 2, None).expect("should succeed");
        assert_eq!(hits.len(), 2, "should return exactly 2 results");
        // First hit should be the identical vector (index 0)
        assert_eq!(hits[0].index, 0, "most similar should be index 0");
    }

    #[test]
    fn test_search_with_min_score_filters() {
        let corpus = vec![
            fv(vec![1.0, 0.0]),
            fv(vec![0.0, 1.0]), // orthogonal to query
        ];
        let search = AudioSimilaritySearch::new(corpus, SimilarityMode::Cosine)
            .expect("should succeed");
        let query = fv(vec![1.0, 0.0]);
        // Only accept similarity > 0.5
        let hits = search
            .search(&query, 10, Some(0.5))
            .expect("should succeed");
        assert_eq!(hits.len(), 1, "only the identical vector should pass the threshold");
        assert_eq!(hits[0].index, 0);
    }

    #[test]
    fn test_search_empty_corpus_error() {
        let err = AudioSimilaritySearch::new(vec![], SimilarityMode::Cosine).unwrap_err();
        assert!(matches!(err, SimilarityError::EmptyCorpus));
    }

    #[test]
    fn test_search_zero_k_error() {
        let corpus = vec![fv(vec![1.0, 2.0])];
        let search = AudioSimilaritySearch::new(corpus, SimilarityMode::Cosine)
            .expect("should succeed");
        let query = fv(vec![1.0, 2.0]);
        let err = search.search(&query, 0, None).unwrap_err();
        assert!(matches!(err, SimilarityError::ZeroTopK));
    }
}
