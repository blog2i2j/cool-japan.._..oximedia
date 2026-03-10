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

    Ok(format!("{:x}", context.finalize()))
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
}
