//! Content-based matching strategies (checksums, duration, etc.).

use crate::config::ConformConfig;
use crate::error::{ConformError, ConformResult};
use crate::types::{ClipMatch, ClipReference, MatchMethod, MediaFile};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use xxhash_rust::xxh3::Xxh3;

/// Match media files by MD5 checksum.
#[must_use]
pub fn md5_match(
    clip: &ClipReference,
    candidates: &[MediaFile],
    expected_md5: &str,
) -> Vec<ClipMatch> {
    let mut matches = Vec::new();

    for media in candidates {
        if let Some(md5) = &media.md5 {
            if md5 == expected_md5 {
                matches.push(ClipMatch {
                    clip: clip.clone(),
                    media: media.clone(),
                    score: 1.0,
                    method: MatchMethod::ContentHash,
                    details: format!("MD5 match: {md5}"),
                });
            }
        }
    }

    matches
}

/// Match media files by duration.
#[must_use]
pub fn duration_match(
    clip: &ClipReference,
    candidates: &[MediaFile],
    config: &ConformConfig,
) -> Vec<ClipMatch> {
    let mut matches = Vec::new();

    let clip_duration_frames =
        clip.source_out.to_frames(clip.fps) - clip.source_in.to_frames(clip.fps);
    let clip_duration_secs = clip_duration_frames as f64 / clip.fps.as_f64();

    for media in candidates {
        if let Some(media_duration) = media.duration {
            let diff = (media_duration - clip_duration_secs).abs();
            if diff <= config.duration_tolerance {
                let score = 1.0 - (diff / config.duration_tolerance).min(1.0);
                if score >= config.match_threshold {
                    matches.push(ClipMatch {
                        clip: clip.clone(),
                        media: media.clone(),
                        score,
                        method: MatchMethod::Duration,
                        details: format!(
                            "Duration match: clip {clip_duration_secs:.2}s, media {media_duration:.2}s (diff: {diff:.2}s)"
                        ),
                    });
                }
            }
        }
    }

    matches
}

/// Match media files by file size.
#[must_use]
pub fn file_size_match(
    clip: &ClipReference,
    candidates: &[MediaFile],
    expected_size: u64,
    tolerance_bytes: u64,
) -> Vec<ClipMatch> {
    let mut matches = Vec::new();

    for media in candidates {
        if let Some(size) = media.size {
            let diff = size.abs_diff(expected_size);

            if diff <= tolerance_bytes {
                let score = 1.0 - (diff as f64 / tolerance_bytes as f64);
                matches.push(ClipMatch {
                    clip: clip.clone(),
                    media: media.clone(),
                    score,
                    method: MatchMethod::ContentHash,
                    details: format!(
                        "File size match: expected {expected_size}, found {size} (diff: {diff} bytes)"
                    ),
                });
            }
        }
    }

    matches
}

/// Calculate MD5 checksum for a file.
///
/// # Errors
///
/// Returns an error if the file cannot be read.
pub fn calculate_md5<P: AsRef<Path>>(path: P) -> ConformResult<String> {
    let mut file = File::open(path)?;
    let mut buffer = vec![0; 8192];
    let mut context = md5::Context::new();

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        context.consume(&buffer[..n]);
    }

    Ok(hex::encode(*context.finalize()))
}

/// Calculate `XXHash` for a file.
///
/// # Errors
///
/// Returns an error if the file cannot be read.
pub fn calculate_xxhash<P: AsRef<Path>>(path: P) -> ConformResult<String> {
    let mut file = File::open(path)?;
    let mut hasher = Xxh3::new();
    let mut buffer = vec![0; 8192];

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    Ok(format!("{:x}", hasher.digest()))
}

/// Verify media file checksum.
///
/// # Errors
///
/// Returns an error if the checksum doesn't match or file cannot be read.
pub fn verify_checksum(media: &MediaFile) -> ConformResult<bool> {
    if let Some(expected_md5) = &media.md5 {
        let actual_md5 = calculate_md5(&media.path)?;
        if &actual_md5 != expected_md5 {
            return Err(ConformError::ChecksumMismatch {
                path: media.path.clone(),
                expected: expected_md5.clone(),
                found: actual_md5,
            });
        }
        Ok(true)
    } else if let Some(expected_xxhash) = &media.xxhash {
        let actual_xxhash = calculate_xxhash(&media.path)?;
        if &actual_xxhash != expected_xxhash {
            return Err(ConformError::ChecksumMismatch {
                path: media.path.clone(),
                expected: expected_xxhash.clone(),
                found: actual_xxhash,
            });
        }
        Ok(true)
    } else {
        Ok(false) // No checksum to verify
    }
}

// ── Perceptual hash matching ──────────────────────────────────────────────────

/// Compute the Hamming-distance similarity between two perceptual hashes.
///
/// Each hash is treated as a 64-bit binary string.  The similarity is:
///
/// ```text
/// similarity = 1.0 - (hamming_distance / 64.0)
/// ```
///
/// Returns a value in \[0.0, 1.0\] where 1.0 means identical hashes.
#[must_use]
pub fn perceptual_hash_match(hash_a: u64, hash_b: u64) -> f32 {
    let hamming = (hash_a ^ hash_b).count_ones() as f32;
    1.0 - (hamming / 64.0)
}

/// A matcher that uses perceptual-hash Hamming distance to detect re-encoded
/// or visually similar source files.
pub struct PerceptualHashMatcher {
    /// Minimum similarity threshold to accept a match.
    threshold: f32,
}

impl PerceptualHashMatcher {
    /// Create a new `PerceptualHashMatcher` with the given similarity threshold.
    ///
    /// `threshold` should be in \[0.0, 1.0\]; typical values are 0.8–0.95.
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Find all `MediaFile` candidates whose stored perceptual hash (embedded
    /// in `metadata` as `{"phash": <u64>}`) matches `query_hash` above the
    /// configured threshold.
    ///
    /// Returns `(media, similarity)` pairs sorted by descending similarity.
    #[must_use]
    pub fn find_similar<'a>(
        &self,
        query_hash: u64,
        candidates: &'a [MediaFile],
    ) -> Vec<(&'a MediaFile, f32)> {
        let mut results: Vec<(&MediaFile, f32)> = candidates
            .iter()
            .filter_map(|m| {
                let stored_hash = extract_phash_from_metadata(m.metadata.as_deref()?)?;
                let sim = perceptual_hash_match(query_hash, stored_hash);
                if sim >= self.threshold {
                    Some((m, sim))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Build [`ClipMatch`] results for `clip` against `candidates` using
    /// perceptual hash similarity.
    #[must_use]
    pub fn match_clip(
        &self,
        clip: &ClipReference,
        query_hash: u64,
        candidates: &[MediaFile],
    ) -> Vec<ClipMatch> {
        self.find_similar(query_hash, candidates)
            .into_iter()
            .map(|(media, sim)| ClipMatch {
                clip: clip.clone(),
                media: media.clone(),
                score: f64::from(sim),
                method: crate::types::MatchMethod::ContentHash,
                details: format!("Perceptual hash similarity: {sim:.4}"),
            })
            .collect()
    }
}

/// Extract a perceptual hash integer from a JSON metadata string.
///
/// Expected format: `{"phash": 12345678901234567890}`.
fn extract_phash_from_metadata(meta: &str) -> Option<u64> {
    // Fast path: look for `"phash"` key and parse the following number.
    let key_pos = meta.find("\"phash\"")?;
    let after_key = &meta[key_pos + 7..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();

    // Read digits
    let digit_end = after_colon
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_colon.len());
    after_colon[..digit_end].parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FrameRate, Timecode, TrackType};
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    fn create_test_clip(source_in: Timecode, source_out: Timecode) -> ClipReference {
        ClipReference {
            id: "test".to_string(),
            source_file: Some("test.mov".to_string()),
            source_in,
            source_out,
            record_in: Timecode::new(1, 0, 0, 0),
            record_out: Timecode::new(1, 0, 10, 0),
            track: TrackType::Video,
            fps: FrameRate::Fps25,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_duration_match() {
        let clip = create_test_clip(Timecode::new(1, 0, 0, 0), Timecode::new(1, 0, 10, 0));

        let mut media = MediaFile::new(PathBuf::from("/path/test.mov"));
        media.duration = Some(10.0);
        media.fps = Some(FrameRate::Fps25);

        let config = ConformConfig::default();
        let matches = duration_match(&clip, &[media], &config);
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_md5_match() {
        let clip = create_test_clip(Timecode::new(1, 0, 0, 0), Timecode::new(1, 0, 10, 0));

        let mut media = MediaFile::new(PathBuf::from("/path/test.mov"));
        media.md5 = Some("abc123".to_string());

        let matches = md5_match(&clip, &[media], "abc123");
        assert_eq!(matches.len(), 1);
        assert!((matches[0].score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_md5() {
        let mut temp_file = NamedTempFile::new().expect("test expectation failed");
        temp_file
            .write_all(b"test content")
            .expect("write_all should succeed");
        temp_file.flush().expect("flush should succeed");

        let md5 = calculate_md5(temp_file.path()).expect("md5 should be valid");
        assert!(!md5.is_empty());
        assert_eq!(md5.len(), 32); // MD5 is 128 bits = 32 hex chars
    }

    #[test]
    fn test_calculate_xxhash() {
        let mut temp_file = NamedTempFile::new().expect("test expectation failed");
        temp_file
            .write_all(b"test content")
            .expect("write_all should succeed");
        temp_file.flush().expect("flush should succeed");

        let xxhash = calculate_xxhash(temp_file.path()).expect("xxhash should be valid");
        assert!(!xxhash.is_empty());
    }

    #[test]
    fn test_file_size_match() {
        let clip = create_test_clip(Timecode::new(1, 0, 0, 0), Timecode::new(1, 0, 10, 0));

        let mut media = MediaFile::new(PathBuf::from("/path/test.mov"));
        media.size = Some(1000);

        let matches = file_size_match(&clip, &[media], 1000, 100);
        assert_eq!(matches.len(), 1);
        assert!((matches[0].score - 1.0).abs() < f64::EPSILON);
    }

    // ── Perceptual hash tests ─────────────────────────────────────────────────

    #[test]
    fn test_perceptual_hash_match_identical() {
        let hash: u64 = 0xDEAD_BEEF_CAFE_1234;
        let sim = perceptual_hash_match(hash, hash);
        assert!(
            (sim - 1.0).abs() < f32::EPSILON,
            "identical hashes must yield similarity 1.0, got {sim}"
        );
    }

    #[test]
    fn test_perceptual_hash_match_all_bits_different() {
        let sim = perceptual_hash_match(u64::MAX, 0u64);
        assert!(
            sim.abs() < f32::EPSILON,
            "fully inverted hashes must yield similarity 0.0, got {sim}"
        );
    }

    #[test]
    fn test_perceptual_hash_match_half_bits() {
        // a XOR b has exactly 32 bits set: upper 32 bits are all 1, lower 32 are 0.
        // a = 0xFFFF_FFFF_0000_0000
        // b = 0x0000_0000_0000_0000
        // XOR = 0xFFFF_FFFF_0000_0000 → 32 bits set → similarity = 1 - 32/64 = 0.5
        let a: u64 = 0xFFFF_FFFF_0000_0000;
        let b: u64 = 0x0000_0000_0000_0000;
        let xor = a ^ b;
        assert_eq!(
            xor.count_ones(),
            32,
            "sanity: XOR must have exactly 32 bits set"
        );
        let sim = perceptual_hash_match(a, b);
        assert!(
            (sim - 0.5).abs() < f32::EPSILON,
            "half-bit difference must yield 0.5, got {sim}"
        );
    }

    #[test]
    fn test_perceptual_hash_match_in_range() {
        let a: u64 = 0xAAAA_AAAA_AAAA_AAAA;
        let b: u64 = 0xBBBB_BBBB_BBBB_BBBB;
        let sim = perceptual_hash_match(a, b);
        assert!(
            (0.0..=1.0).contains(&sim),
            "similarity must be in [0, 1], got {sim}"
        );
    }

    #[test]
    fn test_extract_phash_from_metadata_valid() {
        let meta = r#"{"phash": 12345678901234567}"#;
        let result = extract_phash_from_metadata(meta);
        assert_eq!(result, Some(12_345_678_901_234_567u64));
    }

    #[test]
    fn test_extract_phash_from_metadata_missing_key() {
        let meta = r#"{"md5": "abc123"}"#;
        assert!(extract_phash_from_metadata(meta).is_none());
    }

    #[test]
    fn test_perceptual_hash_matcher_find_similar() {
        let query: u64 = 0xFFFF_FFFF_FFFF_0000;
        // Very close: 4 bits different
        let close_hash: u64 = 0xFFFF_FFFF_FFFF_000F;
        // Far: 32 bits different
        let far_hash: u64 = 0x0000_0000_FFFF_0000;

        let mut media_close = MediaFile::new(PathBuf::from("/path/close.mov"));
        media_close.metadata = Some(format!("{{\"phash\": {close_hash}}}"));

        let mut media_far = MediaFile::new(PathBuf::from("/path/far.mov"));
        media_far.metadata = Some(format!("{{\"phash\": {far_hash}}}"));

        let matcher = PerceptualHashMatcher::new(0.8);
        let candidates = [media_close, media_far];
        let results = matcher.find_similar(query, &candidates);

        // Only the close one should pass the 0.8 threshold
        assert_eq!(results.len(), 1);
        assert!(results[0].1 >= 0.8);
    }

    #[test]
    fn test_perceptual_hash_matcher_match_clip() {
        let clip = create_test_clip(Timecode::new(1, 0, 0, 0), Timecode::new(1, 0, 10, 0));
        let query: u64 = 0xFFFF_FFFF_FFFF_FFFF;
        let same_hash: u64 = 0xFFFF_FFFF_FFFF_FFFF;

        let mut media = MediaFile::new(PathBuf::from("/path/same.mov"));
        media.metadata = Some(format!("{{\"phash\": {same_hash}}}"));

        let matcher = PerceptualHashMatcher::new(0.9);
        let matches = matcher.match_clip(&clip, query, &[media]);
        assert_eq!(matches.len(), 1);
        assert!((matches[0].score - 1.0).abs() < f64::EPSILON);
    }
}
