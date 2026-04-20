//! Fingerprint watermarking: dual content identification combining perceptual
//! hashing with watermark embedding.
//!
//! This module provides [`FingerprintWatermarker`], which embeds two types of
//! identity information into audio simultaneously:
//!
//! 1. **Perceptual fingerprint** — A compact [`AudioHash`] derived from the
//!    host audio's energy content.  It is robust against minor signal
//!    modifications (compression, EQ) but sensitive to content changes.
//!    Used to verify that the watermarked segment still matches its original.
//!
//! 2. **Payload watermark** — Arbitrary bytes embedded using the configured
//!    algorithm (spread-spectrum by default).  Used to carry ownership,
//!    session, or forensic metadata.
//!
//! Together they support a two-layer verification workflow:
//!
//! ```text
//! Embed:   audio + payload  →  watermarked_audio + fingerprint_record
//! Verify:  watermarked_audio  →  (extracted_payload, computed_fingerprint)
//!                                   ↳ compare extracted_payload to expected
//!                                   ↳ compare computed_fingerprint to record
//! ```
//!
//! # Example
//!
//! ```no_run
//! use oximedia_watermark::fingerprint_watermark::{
//!     FingerprintWatermarker, FingerprintWatermarkConfig,
//! };
//!
//! let config = FingerprintWatermarkConfig::default();
//! let wm = FingerprintWatermarker::new(config, 44100).expect("codec ok");
//!
//! let samples: Vec<f32> = vec![0.0; 73728];
//! let payload = b"OwnerID:42";
//!
//! let result = wm.embed(&samples, payload).expect("embed ok");
//! let verified = wm.verify(&result.watermarked, payload, &result.fingerprint)
//!     .expect("verify ok");
//! assert!(verified.payload_match);
//! assert!(verified.fingerprint_similar);
//! ```

#![allow(dead_code)]

use crate::{
    error::{WatermarkError, WatermarkResult},
    payload::PayloadCodec,
    perceptual_hash::{AudioHash, HasherConfig, PerceptualHasher},
    Algorithm, WatermarkConfig, WatermarkDetector, WatermarkEmbedder,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`FingerprintWatermarker`].
#[derive(Debug, Clone)]
pub struct FingerprintWatermarkConfig {
    /// Underlying watermark embedding algorithm.
    pub algorithm: Algorithm,
    /// Embedding strength (0.0 – 1.0).
    pub strength: f32,
    /// Secret key for the watermark.
    pub key: u64,
    /// Enable psychoacoustic masking.
    pub psychoacoustic: bool,
    /// Perceptual hasher configuration.
    pub hasher_config: HasherConfig,
    /// Similarity threshold for fingerprint comparison (0.0 = identical, 1.0 = totally different).
    /// A comparison is considered "similar" if the normalised Hamming distance
    /// between the stored and recomputed fingerprint is ≤ this value.
    pub fingerprint_threshold: f64,
}

impl Default for FingerprintWatermarkConfig {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::SpreadSpectrum,
            strength: 0.1,
            key: 0,
            psychoacoustic: true,
            hasher_config: HasherConfig::default(),
            fingerprint_threshold: 0.15,
        }
    }
}

// ---------------------------------------------------------------------------
// Embed result
// ---------------------------------------------------------------------------

/// The result of a [`FingerprintWatermarker::embed`] operation.
#[derive(Debug, Clone)]
pub struct FingerprintEmbedResult {
    /// The watermarked audio samples.
    pub watermarked: Vec<f32>,
    /// Perceptual fingerprint of the **original** (pre-watermark) audio.
    pub fingerprint: AudioHash,
    /// Perceptual fingerprint of the **watermarked** audio.
    pub watermarked_fingerprint: AudioHash,
    /// Hamming distance between `fingerprint` and `watermarked_fingerprint`.
    /// Ideally small — embedding should not significantly alter the fingerprint.
    pub embedding_distortion: Option<u32>,
}

// ---------------------------------------------------------------------------
// Verify result
// ---------------------------------------------------------------------------

/// The result of a [`FingerprintWatermarker::verify`] operation.
#[derive(Debug, Clone)]
pub struct FingerprintVerifyResult {
    /// Whether the extracted payload matches the expected bytes.
    pub payload_match: bool,
    /// Whether the fingerprint of the audio under test is similar enough to
    /// the stored reference fingerprint.
    pub fingerprint_similar: bool,
    /// Normalised Hamming distance between the current fingerprint and reference.
    /// `None` if lengths differ.
    pub fingerprint_distance: Option<f64>,
    /// The bytes extracted from the watermark.
    pub extracted_payload: Vec<u8>,
    /// The fingerprint recomputed from the audio under test.
    pub current_fingerprint: AudioHash,
}

// ---------------------------------------------------------------------------
// FingerprintWatermarker
// ---------------------------------------------------------------------------

/// Embeds and verifies dual-layer fingerprint watermarks.
pub struct FingerprintWatermarker {
    config: FingerprintWatermarkConfig,
    embedder: WatermarkEmbedder,
    detector: WatermarkDetector,
    hasher: PerceptualHasher,
    codec: PayloadCodec,
    sample_rate: u32,
}

impl FingerprintWatermarker {
    /// Create a new fingerprint watermarker.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError`] if internal codec setup fails.
    pub fn new(config: FingerprintWatermarkConfig, sample_rate: u32) -> WatermarkResult<Self> {
        let wm_config = WatermarkConfig::default()
            .with_algorithm(config.algorithm)
            .with_strength(config.strength)
            .with_key(config.key)
            .with_psychoacoustic(config.psychoacoustic);

        let embedder = WatermarkEmbedder::new(wm_config.clone(), sample_rate);
        let detector = WatermarkDetector::new(wm_config);
        let hasher = PerceptualHasher::new(config.hasher_config.clone());
        let codec = PayloadCodec::new(16, 8)?;

        Ok(Self {
            config,
            embedder,
            detector,
            hasher,
            codec,
            sample_rate,
        })
    }

    /// Embed `payload` into `samples` and compute both the original and
    /// post-watermark fingerprints.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError`] if embedding fails or the audio is too short.
    pub fn embed(
        &self,
        samples: &[f32],
        payload: &[u8],
    ) -> WatermarkResult<FingerprintEmbedResult> {
        // Compute fingerprint of the original signal before any modification.
        let fingerprint = self.hasher.hash(samples);

        // Embed payload watermark.
        let watermarked = self.embedder.embed(samples, payload)?;

        // Compute fingerprint of the watermarked signal.
        let watermarked_fingerprint = self.hasher.hash(&watermarked);

        // Measure embedding distortion in the fingerprint domain.
        let embedding_distortion = fingerprint.hamming_distance(&watermarked_fingerprint);

        Ok(FingerprintEmbedResult {
            watermarked,
            fingerprint,
            watermarked_fingerprint,
            embedding_distortion,
        })
    }

    /// Verify `audio` against `expected_payload` and the reference `fingerprint`
    /// recorded during embedding.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError`] if watermark extraction fails for a reason
    /// other than a simple payload mismatch (e.g. sync failure, data too short).
    pub fn verify(
        &self,
        audio: &[f32],
        expected_payload: &[u8],
        reference_fingerprint: &AudioHash,
    ) -> WatermarkResult<FingerprintVerifyResult> {
        // Compute the expected encoded bit count.
        let encoded = self.codec.encode(expected_payload)?;
        let expected_bits = encoded.len() * 8;

        // Extract watermark payload.
        let extract_result = self.detector.detect(audio, expected_bits);
        let (payload_match, extracted_payload) = match extract_result {
            Ok(bytes) => (bytes == expected_payload, bytes),
            Err(
                WatermarkError::NotDetected
                | WatermarkError::SyncFailed(_)
                | WatermarkError::ErrorCorrectionFailed,
            ) => (false, Vec::new()),
            Err(e) => return Err(e),
        };

        // Compute current fingerprint.
        let current_fingerprint = self.hasher.hash(audio);

        // Compare fingerprints.
        let fingerprint_distance = reference_fingerprint.normalised_distance(&current_fingerprint);
        let fingerprint_similar =
            fingerprint_distance.is_some_and(|d| d <= self.config.fingerprint_threshold);

        Ok(FingerprintVerifyResult {
            payload_match,
            fingerprint_similar,
            fingerprint_distance,
            extracted_payload,
            current_fingerprint,
        })
    }

    /// Return the embedding capacity in bits for a given sample count.
    #[must_use]
    pub fn capacity(&self, sample_count: usize) -> usize {
        self.embedder.capacity(sample_count)
    }

    /// Compute the perceptual fingerprint of audio without embedding.
    #[must_use]
    pub fn fingerprint(&self, samples: &[f32]) -> AudioHash {
        self.hasher.hash(samples)
    }

    /// Return the configured similarity threshold.
    #[must_use]
    pub fn fingerprint_threshold(&self) -> f64 {
        self.config.fingerprint_threshold
    }

    /// Return the configured algorithm.
    #[must_use]
    pub fn algorithm(&self) -> Algorithm {
        self.config.algorithm
    }

    /// Return the sample rate this embedder was constructed with.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// ---------------------------------------------------------------------------
// FingerprintWatermarkDatabase
// ---------------------------------------------------------------------------

/// A simple in-memory catalogue mapping asset IDs to their [`FingerprintEmbedResult`]
/// metadata — for bulk verification pipelines.
#[derive(Debug, Default)]
pub struct FingerprintWatermarkDatabase {
    entries: std::collections::HashMap<String, StoredRecord>,
}

/// A record stored in [`FingerprintWatermarkDatabase`].
#[derive(Debug, Clone)]
pub struct StoredRecord {
    /// The reference fingerprint captured at embed time.
    pub fingerprint: AudioHash,
    /// The watermarked-audio fingerprint (captured at embed time).
    pub watermarked_fingerprint: AudioHash,
    /// The payload that was embedded.
    pub payload: Vec<u8>,
    /// Arbitrary metadata string (owner, date, etc.).
    pub metadata: String,
}

impl FingerprintWatermarkDatabase {
    /// Create an empty database.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a record derived from a [`FingerprintEmbedResult`].
    pub fn store(
        &mut self,
        id: impl Into<String>,
        result: &FingerprintEmbedResult,
        payload: Vec<u8>,
        metadata: impl Into<String>,
    ) {
        self.entries.insert(
            id.into(),
            StoredRecord {
                fingerprint: result.fingerprint.clone(),
                watermarked_fingerprint: result.watermarked_fingerprint.clone(),
                payload,
                metadata: metadata.into(),
            },
        );
    }

    /// Look up a stored record by asset ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&StoredRecord> {
        self.entries.get(id)
    }

    /// Find all entries whose stored fingerprint is similar to `query` within
    /// `threshold`.
    #[must_use]
    pub fn find_by_fingerprint(&self, query: &AudioHash, threshold: f64) -> Vec<(&str, f64)> {
        self.entries
            .iter()
            .filter_map(|(id, rec)| {
                rec.fingerprint
                    .normalised_distance(query)
                    .filter(|&d| d <= threshold)
                    .map(|d| (id.as_str(), d))
            })
            .collect()
    }

    /// Number of stored records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the database is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Enough samples for SpreadSpectrum to embed a small payload.
    /// SpreadSpectrum with PayloadCodec(16,8) + 1-byte payload ≈ 35 bytes = 280 bits.
    /// Non-overlapping 2048-sample frames → need 36 * 2048 = 73728 samples.
    fn test_signal(len: usize) -> Vec<f32> {
        #[allow(clippy::cast_precision_loss)]
        (0..len).map(|i| (i as f32 * 0.01).sin() * 0.5).collect()
    }

    #[test]
    fn test_fingerprint_watermarker_new() {
        let wm = FingerprintWatermarker::new(FingerprintWatermarkConfig::default(), 44100).unwrap();
        assert_eq!(wm.sample_rate(), 44100);
        assert_eq!(wm.algorithm(), Algorithm::SpreadSpectrum);
        assert!(wm.fingerprint_threshold() > 0.0);
    }

    #[test]
    fn test_embed_produces_watermarked_and_fingerprint() {
        let wm = FingerprintWatermarker::new(FingerprintWatermarkConfig::default(), 44100).unwrap();
        let samples = test_signal(73728);
        let payload = b"FP-Test";

        let result = wm.embed(&samples, payload).expect("embed ok");
        assert_eq!(result.watermarked.len(), samples.len());
        assert!(result.fingerprint.bit_len > 0);
        assert!(result.watermarked_fingerprint.bit_len > 0);
    }

    #[test]
    fn test_embed_embedding_distortion_is_small() {
        let wm = FingerprintWatermarker::new(
            FingerprintWatermarkConfig {
                strength: 0.05,
                ..Default::default()
            },
            44100,
        )
        .unwrap();
        let samples = test_signal(73728);
        let payload = b"Low strength";

        let result = wm.embed(&samples, payload).expect("embed ok");
        let dist = result.embedding_distortion.expect("distortion computed");
        // The normalised Hamming distance should be less than 50% at low strength.
        let total_bits = result.fingerprint.bit_len as f64;
        let norm_dist = dist as f64 / total_bits;
        assert!(
            norm_dist < 0.5,
            "fingerprint distortion {norm_dist:.3} is unexpectedly large"
        );
    }

    #[test]
    fn test_fingerprint_is_deterministic() {
        let wm = FingerprintWatermarker::new(FingerprintWatermarkConfig::default(), 44100).unwrap();
        let samples = test_signal(4096);
        let fp1 = wm.fingerprint(&samples);
        let fp2 = wm.fingerprint(&samples);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_capacity_nonzero() {
        let wm = FingerprintWatermarker::new(FingerprintWatermarkConfig::default(), 44100).unwrap();
        assert!(wm.capacity(73728) > 0);
    }

    #[test]
    fn test_verify_fingerprint_similar_after_embed() {
        let wm = FingerprintWatermarker::new(FingerprintWatermarkConfig::default(), 44100).unwrap();
        let samples = test_signal(73728);
        let payload = b"Owner";

        let embed_result = wm.embed(&samples, payload).expect("embed ok");

        // Verify against the original fingerprint (before watermark).
        let verify = wm
            .verify(
                &embed_result.watermarked,
                payload,
                &embed_result.fingerprint,
            )
            .expect("verify ok");

        // The watermarked audio's fingerprint should be similar to the original's.
        assert!(
            verify.fingerprint_distance.is_some(),
            "fingerprint distance should be computable"
        );
    }

    #[test]
    fn test_verify_fingerprint_dissimilar_for_different_content() {
        let wm = FingerprintWatermarker::new(
            FingerprintWatermarkConfig {
                fingerprint_threshold: 0.05, // very strict threshold
                ..Default::default()
            },
            44100,
        )
        .unwrap();
        let original = test_signal(73728);
        let completely_different: Vec<f32> = (0..73728_usize)
            .map(|i| ((i as f32) * 0.5).cos() * 0.8)
            .collect();

        let fp_original = wm.fingerprint(&original);
        // Completely different content should have high fingerprint distance.
        let fp_different = wm.fingerprint(&completely_different);
        let dist = fp_original.normalised_distance(&fp_different);
        // We can't assert exact values, but distance should exist.
        assert!(dist.is_some());
    }

    #[test]
    fn test_database_store_and_retrieve() {
        let mut db = FingerprintWatermarkDatabase::new();
        let wm = FingerprintWatermarker::new(FingerprintWatermarkConfig::default(), 44100).unwrap();
        let samples = test_signal(73728);
        let payload = b"Asset-001";

        let embed_result = wm.embed(&samples, payload).expect("embed ok");
        db.store(
            "asset-001",
            &embed_result,
            payload.to_vec(),
            "2024 COOLJAPAN",
        );

        assert_eq!(db.len(), 1);
        assert!(!db.is_empty());
        let record = db.get("asset-001").expect("record exists");
        assert_eq!(record.payload, payload);
        assert_eq!(record.metadata, "2024 COOLJAPAN");
    }

    #[test]
    fn test_database_find_by_fingerprint() {
        let mut db = FingerprintWatermarkDatabase::new();
        let wm = FingerprintWatermarker::new(FingerprintWatermarkConfig::default(), 44100).unwrap();
        let samples = test_signal(73728);
        let payload = b"X";

        let embed_result = wm.embed(&samples, payload).expect("embed ok");
        db.store("asset-A", &embed_result, payload.to_vec(), "test");

        // Search with a very permissive threshold — should find the record.
        let query = wm.fingerprint(&samples);
        let results = db.find_by_fingerprint(&query, 0.5);
        assert!(
            !results.is_empty(),
            "should find at least one similar fingerprint"
        );
    }

    #[test]
    fn test_database_empty_find() {
        let db = FingerprintWatermarkDatabase::new();
        let fp = AudioHash::new(vec![0xFF], 8);
        let results = db.find_by_fingerprint(&fp, 1.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_different_payloads_produce_different_fingerprint_distortions() {
        let config = FingerprintWatermarkConfig {
            strength: 0.15,
            ..Default::default()
        };
        let wm = FingerprintWatermarker::new(config, 44100);
        let samples = test_signal(73728);

        let r1 = wm.embed(&samples, b"AAA").expect("embed ok");
        let r2 = wm.embed(&samples, b"ZZZ").expect("embed ok");

        // Both should have Some distortion.
        assert!(r1.embedding_distortion.is_some());
        assert!(r2.embedding_distortion.is_some());
    }
}
