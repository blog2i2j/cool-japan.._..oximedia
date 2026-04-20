//! Patchwork statistical watermarking.
//!
//! This module implements the patchwork algorithm, which embeds
//! watermarks using statistical properties of sample pairs.

use crate::error::{WatermarkError, WatermarkResult};
use crate::payload::{pack_bits, unpack_bits, PayloadCodec};

/// Patchwork configuration.
#[derive(Debug, Clone)]
pub struct PatchworkConfig {
    /// Number of sample pairs per bit.
    pub pairs_per_bit: usize,
    /// Embedding strength.
    pub strength: f32,
    /// Distance between samples in a pair.
    pub pair_distance: usize,
    /// Secret key for pseudorandom pair selection.
    pub key: u64,
}

impl Default for PatchworkConfig {
    fn default() -> Self {
        Self {
            pairs_per_bit: 100,
            strength: 0.01,
            pair_distance: 10,
            key: 0,
        }
    }
}

/// Patchwork watermark embedder.
pub struct PatchworkEmbedder {
    config: PatchworkConfig,
    codec: PayloadCodec,
}

impl PatchworkEmbedder {
    /// Create a new patchwork embedder.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon codec cannot be initialised.
    pub fn new(config: PatchworkConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Embed watermark using patchwork algorithm.
    ///
    /// # Errors
    ///
    /// Returns error if audio is too short or encoding fails.
    pub fn embed(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        // Encode payload
        let encoded = self.codec.encode(payload)?;
        let bits = unpack_bits(&encoded, encoded.len() * 8);

        // Check capacity
        let capacity = self.capacity(samples.len());
        if bits.len() > capacity {
            return Err(WatermarkError::InsufficientCapacity {
                needed: bits.len(),
                have: capacity,
            });
        }

        let mut watermarked = samples.to_vec();

        for (bit_idx, &bit) in bits.iter().enumerate() {
            // Generate pseudorandom sample pairs for this bit
            let pairs = self.generate_pairs(samples.len(), bit_idx);

            // Embed bit by modifying pair differences
            for (a_idx, b_idx) in pairs {
                if bit {
                    // Increase difference: a > b
                    watermarked[a_idx] += self.config.strength;
                    watermarked[b_idx] -= self.config.strength;
                } else {
                    // Decrease difference: a < b
                    watermarked[a_idx] -= self.config.strength;
                    watermarked[b_idx] += self.config.strength;
                }
            }
        }

        Ok(watermarked)
    }

    /// Detect and extract watermark.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    pub fn detect(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        let mut bits = Vec::new();

        for bit_idx in 0..expected_bits {
            // Generate same pseudorandom pairs
            let pairs = self.generate_pairs(samples.len(), bit_idx);

            // Calculate statistical measure
            let mut sum_diff = 0.0f32;

            for (a_idx, b_idx) in pairs {
                sum_diff += samples[a_idx] - samples[b_idx];
            }

            // Bit is 1 if sum is positive (a > b on average)
            bits.push(sum_diff > 0.0);
        }

        let bytes = pack_bits(&bits);
        self.codec.decode(&bytes)
    }

    /// Generate pseudorandom sample pairs for a bit.
    fn generate_pairs(&self, sample_count: usize, bit_idx: usize) -> Vec<(usize, usize)> {
        let mut rng =
            scirs2_core::random::Random::seed(self.config.key.wrapping_add(bit_idx as u64));

        let mut pairs = Vec::new();
        let max_idx = sample_count - self.config.pair_distance;

        for _ in 0..self.config.pairs_per_bit {
            let a_idx = (rng.random_f64() * max_idx as f64) as usize;
            let a_idx = a_idx.min(max_idx.saturating_sub(1));
            let b_idx = a_idx + self.config.pair_distance;

            if b_idx < sample_count {
                pairs.push((a_idx, b_idx));
            }
        }

        pairs
    }

    /// Calculate capacity in bits.
    #[must_use]
    pub fn capacity(&self, sample_count: usize) -> usize {
        // Conservative estimate
        sample_count / (self.config.pairs_per_bit * 2)
    }
}

/// Advanced patchwork with block-based embedding.
pub struct BlockPatchworkEmbedder {
    #[allow(dead_code)]
    block_size: usize,
    pairs_per_block: usize,
    strength: f32,
    key: u64,
}

impl BlockPatchworkEmbedder {
    /// Create a new block-based patchwork embedder.
    #[must_use]
    pub fn new(block_size: usize, pairs_per_block: usize, strength: f32, key: u64) -> Self {
        Self {
            block_size,
            pairs_per_block,
            strength,
            key,
        }
    }

    /// Embed bit in a block.
    #[must_use]
    pub fn embed_in_block(&self, block: &[f32], bit: bool) -> Vec<f32> {
        let mut watermarked = block.to_vec();
        let pairs = self.generate_block_pairs(block.len());

        for (a, b) in pairs {
            if bit {
                watermarked[a] += self.strength;
                watermarked[b] -= self.strength;
            } else {
                watermarked[a] -= self.strength;
                watermarked[b] += self.strength;
            }
        }

        watermarked
    }

    /// Detect bit from block.
    #[must_use]
    pub fn detect_from_block(&self, block: &[f32]) -> bool {
        let pairs = self.generate_block_pairs(block.len());
        let mut sum = 0.0f32;

        for (a, b) in pairs {
            sum += block[a] - block[b];
        }

        sum > 0.0
    }

    /// Generate pairs within a block.
    fn generate_block_pairs(&self, block_len: usize) -> Vec<(usize, usize)> {
        let mut rng = scirs2_core::random::Random::seed(self.key);

        let mut pairs = Vec::new();

        for _ in 0..self.pairs_per_block {
            let a = (rng.random_f64() * block_len as f64) as usize;
            let a = a.min(block_len.saturating_sub(1));
            let b = (rng.random_f64() * block_len as f64) as usize;
            let b = b.min(block_len.saturating_sub(1));

            if a != b {
                pairs.push((a.min(b), a.max(b)));
            }
        }

        pairs
    }
}

/// Improved patchwork with correlation-based detection.
pub struct CorrelationPatchwork {
    patch_size: usize,
    strength: f32,
    key: u64,
}

impl CorrelationPatchwork {
    /// Create a new correlation-based patchwork embedder.
    #[must_use]
    pub fn new(patch_size: usize, strength: f32, key: u64) -> Self {
        Self {
            patch_size,
            strength,
            key,
        }
    }

    /// Embed watermark using correlation patches.
    #[must_use]
    pub fn embed(&self, samples: &[f32], watermark_sequence: &[f32]) -> Vec<f32> {
        let mut watermarked = samples.to_vec();
        let patch_indices = self.generate_patch_indices(samples.len());

        for (i, &idx) in patch_indices.iter().enumerate() {
            if idx < watermarked.len() && i < watermark_sequence.len() {
                watermarked[idx] += watermark_sequence[i] * self.strength;
            }
        }

        watermarked
    }

    /// Detect watermark using correlation.
    #[must_use]
    pub fn detect(&self, samples: &[f32], watermark_sequence: &[f32]) -> f32 {
        let patch_indices = self.generate_patch_indices(samples.len());
        let mut correlation = 0.0f32;
        let mut count = 0;

        for (i, &idx) in patch_indices.iter().enumerate() {
            if idx < samples.len() && i < watermark_sequence.len() {
                correlation += samples[idx] * watermark_sequence[i];
                count += 1;
            }
        }

        if count > 0 {
            #[allow(clippy::cast_precision_loss)]
            let count_f32 = count as f32;
            correlation / count_f32
        } else {
            0.0
        }
    }

    /// Generate pseudorandom patch indices.
    fn generate_patch_indices(&self, max_len: usize) -> Vec<usize> {
        let mut rng = scirs2_core::random::Random::seed(self.key);

        (0..self.patch_size)
            .map(|_| {
                let idx = (rng.random_f64() * max_len as f64) as usize;
                idx.min(max_len.saturating_sub(1))
            })
            .collect()
    }
}

/// Calculate statistical detection metric.
#[must_use]
pub fn calculate_detection_statistic(samples_a: &[f32], samples_b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let n = samples_a.len().min(samples_b.len());

    for i in 0..n {
        sum += samples_a[i] - samples_b[i];
    }

    #[allow(clippy::cast_precision_loss)]
    let result = sum / n as f32;
    result
}

/// Calculate normalized correlation.
#[must_use]
pub fn calculate_normalized_correlation(signal: &[f32], reference: &[f32]) -> f32 {
    if signal.len() != reference.len() {
        return 0.0;
    }

    let mut sum = 0.0f32;
    let mut sig_energy = 0.0f32;
    let mut ref_energy = 0.0f32;

    for (s, r) in signal.iter().zip(reference.iter()) {
        sum += s * r;
        sig_energy += s * s;
        ref_energy += r * r;
    }

    if sig_energy == 0.0 || ref_energy == 0.0 {
        return 0.0;
    }

    sum / (sig_energy.sqrt() * ref_energy.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patchwork_embedding() {
        let config = PatchworkConfig {
            pairs_per_bit: 50,
            strength: 0.02,
            pair_distance: 5,
            key: 42,
        };

        let embedder = PatchworkEmbedder::new(config).unwrap();

        let samples: Vec<f32> = vec![0.0; 100000];
        let payload = b"Test";

        let watermarked = embedder
            .embed(&samples, payload)
            .expect("should succeed in test");

        let encoded = embedder
            .codec
            .encode(payload)
            .expect("should succeed in test");
        let expected_bits = encoded.len() * 8;

        let extracted = embedder
            .detect(&watermarked, expected_bits)
            .expect("should succeed in test");
        assert_eq!(payload.as_slice(), extracted.as_slice());
    }

    #[test]
    fn test_block_patchwork() {
        let embedder = BlockPatchworkEmbedder::new(1024, 50, 0.01, 12345);

        let block: Vec<f32> = vec![0.0; 1024];

        let wm_0 = embedder.embed_in_block(&block, false);
        let wm_1 = embedder.embed_in_block(&block, true);

        let det_0 = embedder.detect_from_block(&wm_0);
        let det_1 = embedder.detect_from_block(&wm_1);

        assert!(!det_0);
        assert!(det_1);
    }

    #[test]
    fn test_correlation_patchwork() {
        let embedder = CorrelationPatchwork::new(1000, 0.05, 54321);

        let samples: Vec<f32> = vec![0.1; 10000];
        let watermark: Vec<f32> = (0..1000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let watermarked = embedder.embed(&samples, &watermark);
        let correlation = embedder.detect(&watermarked, &watermark);

        assert!(correlation > 0.0);
    }

    #[test]
    fn test_detection_statistic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.9, 1.9, 2.9, 3.9];

        let stat = calculate_detection_statistic(&a, &b);
        assert!((stat - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_normalized_correlation() {
        let signal = vec![1.0, 0.0, -1.0, 0.0];
        let reference = vec![1.0, 0.0, -1.0, 0.0];

        let corr = calculate_normalized_correlation(&signal, &reference);
        assert!((corr - 1.0).abs() < 0.01);
    }
}
