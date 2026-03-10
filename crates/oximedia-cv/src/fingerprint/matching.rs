//! Fingerprint matching and similarity scoring.
//!
//! This module provides algorithms for comparing and matching fingerprints:
//!
//! - **Similarity scoring**: Computes similarity between fingerprints
//! - **Threshold-based detection**: Identifies matches above threshold
//! - **Near-duplicate detection**: Finds near-duplicates in databases
//! - **Partial matching**: Matches partial video/audio segments
//! - **False positive reduction**: Statistical validation of matches
//! - **Confidence scoring**: Computes match confidence levels

use super::{chromaprint, phash, temporal, VideoFingerprint};
use crate::error::{CvError, CvResult};
use rayon::prelude::*;

/// Match result with confidence score.
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// Index of matched item in database.
    pub index: usize,
    /// Overall similarity score (0.0-1.0).
    pub similarity: f64,
    /// Visual similarity component.
    pub visual_similarity: f64,
    /// Temporal similarity component.
    pub temporal_similarity: f64,
    /// Audio similarity component (if available).
    pub audio_similarity: Option<f64>,
    /// Confidence level (0.0-1.0).
    pub confidence: f64,
    /// Time offset (in seconds) if partial match.
    pub time_offset: Option<f64>,
}

impl MatchResult {
    /// Creates a new match result.
    #[must_use]
    pub fn new(
        index: usize,
        similarity: f64,
        visual_similarity: f64,
        temporal_similarity: f64,
        audio_similarity: Option<f64>,
        confidence: f64,
    ) -> Self {
        Self {
            index,
            similarity,
            visual_similarity,
            temporal_similarity,
            audio_similarity,
            confidence,
            time_offset: None,
        }
    }

    /// Sets the time offset.
    #[must_use]
    pub fn with_offset(mut self, offset: f64) -> Self {
        self.time_offset = Some(offset);
        self
    }

    /// Returns true if this is a strong match.
    #[must_use]
    pub fn is_strong_match(&self, threshold: f64) -> bool {
        self.similarity >= threshold && self.confidence >= 0.8
    }

    /// Returns true if this is a partial match.
    #[must_use]
    pub fn is_partial_match(&self) -> bool {
        self.time_offset.is_some()
    }
}

/// Matching configuration.
#[derive(Debug, Clone)]
pub struct MatchingConfig {
    /// Minimum similarity threshold (0.0-1.0).
    pub similarity_threshold: f64,
    /// Weight for visual similarity (0.0-1.0).
    pub visual_weight: f64,
    /// Weight for temporal similarity (0.0-1.0).
    pub temporal_weight: f64,
    /// Weight for audio similarity (0.0-1.0).
    pub audio_weight: f64,
    /// Enable partial matching.
    pub enable_partial_matching: bool,
    /// Maximum time offset for partial matching (seconds).
    pub max_time_offset: f64,
    /// Enable parallel search.
    pub parallel_search: bool,
}

impl Default for MatchingConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            visual_weight: 0.5,
            temporal_weight: 0.3,
            audio_weight: 0.2,
            enable_partial_matching: false,
            max_time_offset: 30.0,
            parallel_search: true,
        }
    }
}

impl MatchingConfig {
    /// Validates the configuration.
    pub fn validate(&self) -> CvResult<()> {
        if !(0.0..=1.0).contains(&self.similarity_threshold) {
            return Err(CvError::invalid_parameter(
                "similarity_threshold",
                format!("{}", self.similarity_threshold),
            ));
        }

        let total_weight = self.visual_weight + self.temporal_weight + self.audio_weight;
        if (total_weight - 1.0).abs() > 0.01 {
            return Err(CvError::invalid_parameter(
                "weights",
                format!("sum = {total_weight} (must equal 1.0)"),
            ));
        }

        Ok(())
    }
}

/// Compares two video fingerprints.
///
/// Returns overall similarity score in [0.0, 1.0].
#[must_use]
pub fn compare_fingerprints(fp1: &VideoFingerprint, fp2: &VideoFingerprint) -> f64 {
    let config = MatchingConfig::default();
    compare_fingerprints_with_config(fp1, fp2, &config)
        .map(|result| result.similarity)
        .unwrap_or(0.0)
}

/// Compares two fingerprints with custom configuration.
///
/// # Errors
///
/// Returns an error if configuration is invalid.
pub fn compare_fingerprints_with_config(
    fp1: &VideoFingerprint,
    fp2: &VideoFingerprint,
    config: &MatchingConfig,
) -> CvResult<MatchResult> {
    config.validate()?;

    // Compute visual similarity (perceptual hash)
    let visual_sim = compute_visual_similarity(&fp1.perceptual_hash, &fp2.perceptual_hash);

    // Compute temporal similarity
    let temporal_sim =
        temporal::compute_temporal_correlation(&fp1.temporal_signature, &fp2.temporal_signature);

    // Compute audio similarity (if available)
    let audio_sim = match (&fp1.audio_fingerprint, &fp2.audio_fingerprint) {
        (Some(a1), Some(a2)) => Some(chromaprint::compare_fingerprints(a1, a2)),
        _ => None,
    };

    // Weighted combination
    let mut similarity = visual_sim * config.visual_weight + temporal_sim * config.temporal_weight;

    if let Some(audio_score) = audio_sim {
        similarity += audio_score * config.audio_weight;
    } else {
        // Redistribute audio weight to visual and temporal
        let redistribution = config.audio_weight / 2.0;
        similarity += visual_sim * redistribution + temporal_sim * redistribution;
    }

    // Compute confidence
    let confidence = compute_confidence(visual_sim, temporal_sim, audio_sim);

    Ok(MatchResult::new(
        0,
        similarity,
        visual_sim,
        temporal_sim,
        audio_sim,
        confidence,
    ))
}

/// Computes visual similarity from perceptual hashes.
fn compute_visual_similarity(hashes1: &[u64], hashes2: &[u64]) -> f64 {
    if hashes1.is_empty() || hashes2.is_empty() {
        return 0.0;
    }

    let len = hashes1.len().min(hashes2.len());
    let mut total_similarity = 0.0;

    for i in 0..len {
        total_similarity += phash::hash_similarity(hashes1[i], hashes2[i]);
    }

    total_similarity / len as f64
}

/// Computes match confidence based on component scores.
fn compute_confidence(visual: f64, temporal: f64, audio: Option<f64>) -> f64 {
    let mut scores = vec![visual, temporal];
    if let Some(audio_score) = audio {
        scores.push(audio_score);
    }

    // Confidence is higher when all components agree
    let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance: f64 =
        scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;

    // High mean and low variance = high confidence
    mean * (1.0 - variance.sqrt())
}

/// Finds matches in a database.
///
/// Returns all matches above the similarity threshold.
///
/// # Errors
///
/// Returns an error if matching fails.
pub fn find_matches(
    query: &VideoFingerprint,
    database: &[VideoFingerprint],
    config: &MatchingConfig,
) -> CvResult<Vec<MatchResult>> {
    config.validate()?;

    let matches = if config.parallel_search {
        find_matches_parallel(query, database, config)?
    } else {
        find_matches_sequential(query, database, config)?
    };

    // Filter by threshold
    let filtered: Vec<_> = matches
        .into_iter()
        .filter(|m| m.similarity >= config.similarity_threshold)
        .collect();

    Ok(filtered)
}

/// Sequential matching.
fn find_matches_sequential(
    query: &VideoFingerprint,
    database: &[VideoFingerprint],
    config: &MatchingConfig,
) -> CvResult<Vec<MatchResult>> {
    let mut results = Vec::new();

    for (idx, db_fp) in database.iter().enumerate() {
        let mut match_result = compare_fingerprints_with_config(query, db_fp, config)?;
        match_result.index = idx;
        results.push(match_result);
    }

    Ok(results)
}

/// Parallel matching.
fn find_matches_parallel(
    query: &VideoFingerprint,
    database: &[VideoFingerprint],
    config: &MatchingConfig,
) -> CvResult<Vec<MatchResult>> {
    let results: Vec<MatchResult> = database
        .par_iter()
        .enumerate()
        .filter_map(|(idx, db_fp)| {
            let mut match_result = compare_fingerprints_with_config(query, db_fp, config).ok()?;
            match_result.index = idx;
            Some(match_result)
        })
        .collect();

    Ok(results)
}

/// Finds the best match in a database.
///
/// Returns the best matching fingerprint, or `None` if no match above threshold.
///
/// # Errors
///
/// Returns an error if matching fails.
pub fn find_best_match(
    query: &VideoFingerprint,
    database: &[VideoFingerprint],
    config: &MatchingConfig,
) -> CvResult<Option<MatchResult>> {
    let matches = find_matches(query, database, config)?;

    Ok(matches.into_iter().max_by(|a, b| {
        a.similarity
            .partial_cmp(&b.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    }))
}

/// Detects near-duplicates in a database.
///
/// Returns pairs of indices that are near-duplicates.
///
/// # Errors
///
/// Returns an error if detection fails.
pub fn detect_near_duplicates(
    database: &[VideoFingerprint],
    threshold: f64,
) -> CvResult<Vec<(usize, usize, f64)>> {
    let mut duplicates = Vec::new();

    for i in 0..database.len() {
        for j in (i + 1)..database.len() {
            let similarity = compare_fingerprints(&database[i], &database[j]);
            if similarity >= threshold {
                duplicates.push((i, j, similarity));
            }
        }
    }

    Ok(duplicates)
}

/// Detects near-duplicates using parallel processing.
///
/// # Errors
///
/// Returns an error if detection fails.
pub fn detect_near_duplicates_parallel(
    database: &[VideoFingerprint],
    threshold: f64,
) -> CvResult<Vec<(usize, usize, f64)>> {
    let pairs: Vec<(usize, usize)> = (0..database.len())
        .flat_map(|i| ((i + 1)..database.len()).map(move |j| (i, j)))
        .collect();

    let duplicates: Vec<_> = pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let similarity = compare_fingerprints(&database[i], &database[j]);
            if similarity >= threshold {
                Some((i, j, similarity))
            } else {
                None
            }
        })
        .collect();

    Ok(duplicates)
}

/// Finds partial matches in database.
///
/// Attempts to match query against segments of database fingerprints.
///
/// # Errors
///
/// Returns an error if matching fails.
#[allow(clippy::too_many_arguments)]
pub fn find_partial_matches(
    query: &VideoFingerprint,
    database: &[VideoFingerprint],
    config: &MatchingConfig,
    segment_duration: f64,
) -> CvResult<Vec<MatchResult>> {
    if !config.enable_partial_matching {
        return Ok(Vec::new());
    }

    let mut matches = Vec::new();

    for (idx, db_fp) in database.iter().enumerate() {
        if let Some(result) = find_partial_match(query, db_fp, config, segment_duration)? {
            let mut match_result = result;
            match_result.index = idx;
            matches.push(match_result);
        }
    }

    Ok(matches)
}

/// Finds partial match between query and database fingerprint.
fn find_partial_match(
    query: &VideoFingerprint,
    db_fp: &VideoFingerprint,
    config: &MatchingConfig,
    segment_duration: f64,
) -> CvResult<Option<MatchResult>> {
    // Try different time offsets
    let max_offset_frames = (config.max_time_offset / segment_duration) as usize;
    let mut best_match: Option<MatchResult> = None;

    for offset in 0..=max_offset_frames {
        // Compare with offset
        let similarity = compare_with_offset(query, db_fp, offset)?;

        if similarity >= config.similarity_threshold {
            let confidence = compute_confidence(similarity, similarity, None);
            let match_result =
                MatchResult::new(0, similarity, similarity, similarity, None, confidence)
                    .with_offset(offset as f64 * segment_duration);

            if best_match
                .as_ref()
                .map_or(true, |m| similarity > m.similarity)
            {
                best_match = Some(match_result);
            }
        }
    }

    Ok(best_match)
}

/// Compares fingerprints with temporal offset.
fn compare_with_offset(
    fp1: &VideoFingerprint,
    fp2: &VideoFingerprint,
    offset: usize,
) -> CvResult<f64> {
    let len1 = fp1.perceptual_hash.len();
    let len2 = fp2.perceptual_hash.len();

    if offset >= len2 {
        return Ok(0.0);
    }

    let compare_len = (len1).min(len2 - offset);
    if compare_len == 0 {
        return Ok(0.0);
    }

    let mut total_similarity = 0.0;
    for i in 0..compare_len {
        let sim = phash::hash_similarity(fp1.perceptual_hash[i], fp2.perceptual_hash[i + offset]);
        total_similarity += sim;
    }

    Ok(total_similarity / compare_len as f64)
}

/// Validates match using statistical tests.
///
/// Returns true if match passes validation (not a false positive).
#[must_use]
pub fn validate_match(
    result: &MatchResult,
    fp1: &VideoFingerprint,
    fp2: &VideoFingerprint,
) -> bool {
    // Check duration similarity
    let duration_ratio = fp1.duration / fp2.duration.max(0.001);
    if !(0.8..=1.2).contains(&duration_ratio) {
        return false;
    }

    // Check resolution similarity
    let res1 = (fp1.resolution.0 as f64, fp1.resolution.1 as f64);
    let res2 = (fp2.resolution.0 as f64, fp2.resolution.1 as f64);

    let aspect1 = res1.0 / res1.1.max(1.0);
    let aspect2 = res2.0 / res2.1.max(1.0);

    if (aspect1 - aspect2).abs() > 0.2 {
        return false;
    }

    // Check confidence
    if result.confidence < 0.7 {
        return false;
    }

    // Check component consistency
    let visual_temporal_diff = (result.visual_similarity - result.temporal_similarity).abs();
    if visual_temporal_diff > 0.3 {
        return false;
    }

    true
}

/// Computes match quality score.
///
/// Returns a quality score in [0.0, 1.0] based on various factors.
#[must_use]
pub fn compute_match_quality(
    result: &MatchResult,
    fp1: &VideoFingerprint,
    fp2: &VideoFingerprint,
) -> f64 {
    let mut quality = result.similarity * 0.4 + result.confidence * 0.3;

    // Duration similarity bonus
    let duration_ratio = fp1.duration / fp2.duration.max(0.001);
    let duration_score = if (0.9..=1.1).contains(&duration_ratio) {
        1.0
    } else {
        0.5
    };
    quality += duration_score * 0.15;

    // Frame count similarity bonus
    let frame_ratio = fp1.frame_count as f64 / fp2.frame_count.max(1) as f64;
    let frame_score = if (0.9..=1.1).contains(&frame_ratio) {
        1.0
    } else {
        0.5
    };
    quality += frame_score * 0.15;

    quality.min(1.0)
}

/// Filters false positives from match results.
#[must_use]
pub fn filter_false_positives(
    results: Vec<MatchResult>,
    query: &VideoFingerprint,
    database: &[VideoFingerprint],
) -> Vec<MatchResult> {
    results
        .into_iter()
        .filter(|result| {
            if result.index >= database.len() {
                return false;
            }
            validate_match(result, query, &database[result.index])
        })
        .collect()
}

/// Groups similar matches (likely same content).
#[must_use]
pub fn group_similar_matches(
    mut results: Vec<MatchResult>,
    threshold: f64,
) -> Vec<Vec<MatchResult>> {
    if results.is_empty() {
        return Vec::new();
    }

    // Sort by similarity (descending)
    results.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut groups: Vec<Vec<MatchResult>> = Vec::new();

    for result in results {
        let mut added = false;

        for group in &mut groups {
            if let Some(first) = group.first() {
                let sim_diff = (first.similarity - result.similarity).abs();
                if sim_diff < (1.0 - threshold) {
                    group.push(result.clone());
                    added = true;
                    break;
                }
            }
        }

        if !added {
            groups.push(vec![result]);
        }
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_fingerprint() -> VideoFingerprint {
        VideoFingerprint::new(
            vec![0x1234567890ABCDEF; 10],
            vec![0.5; 50],
            Some(vec![100; 100]),
            60.0,
            1500,
            (1920, 1080),
        )
    }

    #[test]
    fn test_matching_config_default() {
        let config = MatchingConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_matching_config_validation() {
        let mut config = MatchingConfig::default();
        config.similarity_threshold = 1.5;
        assert!(config.validate().is_err());

        config.similarity_threshold = 0.8;
        config.visual_weight = 0.5;
        config.temporal_weight = 0.5;
        config.audio_weight = 0.5;
        assert!(config.validate().is_err()); // Sum > 1.0
    }

    #[test]
    fn test_compare_fingerprints() {
        let fp1 = create_test_fingerprint();
        let fp2 = create_test_fingerprint();

        let similarity = compare_fingerprints(&fp1, &fp2);
        assert!(similarity > 0.9);
    }

    #[test]
    fn test_compare_fingerprints_with_config() {
        let fp1 = create_test_fingerprint();
        let fp2 = create_test_fingerprint();
        let config = MatchingConfig::default();

        let result = compare_fingerprints_with_config(&fp1, &fp2, &config)
            .expect("compare_fingerprints_with_config should succeed");
        assert!(result.similarity > 0.8);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_match_result_creation() {
        let result = MatchResult::new(0, 0.95, 0.9, 0.85, Some(0.92), 0.88);
        assert_eq!(result.index, 0);
        assert_eq!(result.similarity, 0.95);
        assert!(result.is_strong_match(0.9));
    }

    #[test]
    fn test_match_result_with_offset() {
        let result = MatchResult::new(0, 0.9, 0.9, 0.9, None, 0.85).with_offset(10.0);
        assert!(result.is_partial_match());
        assert_eq!(result.time_offset, Some(10.0));
    }

    #[test]
    fn test_find_matches() {
        let query = create_test_fingerprint();
        let database = vec![create_test_fingerprint(); 5];
        let config = MatchingConfig::default();

        let matches =
            find_matches(&query, &database, &config).expect("find_matches should succeed");
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_find_best_match() {
        let query = create_test_fingerprint();
        let database = vec![create_test_fingerprint(); 3];
        let config = MatchingConfig::default();

        let best =
            find_best_match(&query, &database, &config).expect("find_best_match should succeed");
        assert!(best.is_some());
    }

    #[test]
    fn test_detect_near_duplicates() {
        let database = vec![create_test_fingerprint(); 5];
        let duplicates =
            detect_near_duplicates(&database, 0.9).expect("detect_near_duplicates should succeed");
        assert!(!duplicates.is_empty());
    }

    #[test]
    fn test_detect_near_duplicates_parallel() {
        let database = vec![create_test_fingerprint(); 5];
        let duplicates = detect_near_duplicates_parallel(&database, 0.9)
            .expect("detect_near_duplicates_parallel should succeed");
        assert!(!duplicates.is_empty());
    }

    #[test]
    fn test_validate_match() {
        let fp1 = create_test_fingerprint();
        let fp2 = create_test_fingerprint();
        let result = MatchResult::new(0, 0.95, 0.9, 0.92, Some(0.93), 0.88);

        assert!(validate_match(&result, &fp1, &fp2));
    }

    #[test]
    fn test_compute_match_quality() {
        let fp1 = create_test_fingerprint();
        let fp2 = create_test_fingerprint();
        let result = MatchResult::new(0, 0.95, 0.9, 0.92, Some(0.93), 0.88);

        let quality = compute_match_quality(&result, &fp1, &fp2);
        assert!(quality > 0.5 && quality <= 1.0);
    }

    #[test]
    fn test_filter_false_positives() {
        let fp = create_test_fingerprint();
        let database = vec![fp.clone(); 3];
        let results = vec![
            MatchResult::new(0, 0.95, 0.9, 0.92, None, 0.88),
            MatchResult::new(1, 0.85, 0.8, 0.82, None, 0.78),
        ];

        let filtered = filter_false_positives(results, &fp, &database);
        assert!(!filtered.is_empty());
    }

    #[test]
    fn test_group_similar_matches() {
        let results = vec![
            MatchResult::new(0, 0.95, 0.9, 0.92, None, 0.88),
            MatchResult::new(1, 0.94, 0.89, 0.91, None, 0.87),
            MatchResult::new(2, 0.75, 0.7, 0.72, None, 0.68),
        ];

        let groups = group_similar_matches(results, 0.85);
        assert!(!groups.is_empty());
    }

    #[test]
    fn test_compute_confidence() {
        let conf1 = compute_confidence(0.9, 0.9, Some(0.9));
        assert!(conf1 > 0.8);

        let conf2 = compute_confidence(0.9, 0.5, Some(0.3));
        assert!(conf2 < conf1);
    }

    #[test]
    fn test_visual_similarity() {
        let hashes1 = vec![0x1234567890ABCDEF; 5];
        let hashes2 = vec![0x1234567890ABCDEF; 5];
        let sim = compute_visual_similarity(&hashes1, &hashes2);
        assert_eq!(sim, 1.0);
    }

    #[test]
    fn test_compare_with_offset() {
        let fp1 = create_test_fingerprint();
        let fp2 = create_test_fingerprint();

        let sim = compare_with_offset(&fp1, &fp2, 0).expect("compare_with_offset should succeed");
        assert!(sim > 0.9);

        let sim_offset =
            compare_with_offset(&fp1, &fp2, 5).expect("compare_with_offset should succeed");
        assert!(sim_offset >= 0.0);
    }
}
