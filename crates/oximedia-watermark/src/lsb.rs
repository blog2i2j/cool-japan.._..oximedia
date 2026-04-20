//! Least Significant Bit (LSB) steganography.
//!
//! This module implements simple LSB-based audio steganography.
//! While not robust to lossy compression, it provides high capacity.

use crate::error::{WatermarkError, WatermarkResult};
use crate::payload::{pack_bits, unpack_bits, PayloadCodec};

/// LSB steganography configuration.
#[derive(Debug, Clone)]
pub struct LsbConfig {
    /// Number of LSBs to use (1-8).
    pub bits_per_sample: usize,
    /// Apply dithering to reduce artifacts.
    pub dithering: bool,
    /// Randomize embedding positions.
    pub randomize: bool,
    /// Secret key for position randomization.
    pub key: u64,
}

impl Default for LsbConfig {
    fn default() -> Self {
        Self {
            bits_per_sample: 1,
            dithering: true,
            randomize: false,
            key: 0,
        }
    }
}

/// LSB steganography embedder.
pub struct LsbEmbedder {
    config: LsbConfig,
    codec: PayloadCodec,
}

impl LsbEmbedder {
    /// Create a new LSB embedder.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon codec cannot be initialised.
    pub fn new(config: LsbConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Embed data in LSBs of audio samples.
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

        // Convert to 16-bit PCM for LSB manipulation
        let mut pcm: Vec<i16> = samples.iter().map(|&s| self.float_to_i16(s)).collect();

        // Generate embedding positions
        let positions = if self.config.randomize {
            self.generate_random_positions(pcm.len(), bits.len())
        } else {
            (0..bits.len()).collect()
        };

        // Embed bits
        for (i, &bit) in bits.iter().enumerate() {
            let pos = positions[i];
            if pos >= pcm.len() {
                break;
            }

            pcm[pos] = self.embed_bit(pcm[pos], bit);
        }

        // Apply dithering if enabled
        if self.config.dithering {
            self.apply_dithering(&mut pcm);
        }

        // Convert back to float
        Ok(pcm.iter().map(|&s| self.i16_to_float(s)).collect())
    }

    /// Extract data from LSBs.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    pub fn extract(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        // Convert to 16-bit PCM
        let pcm: Vec<i16> = samples.iter().map(|&s| self.float_to_i16(s)).collect();

        // Generate extraction positions
        let positions = if self.config.randomize {
            self.generate_random_positions(pcm.len(), expected_bits)
        } else {
            (0..expected_bits).collect()
        };

        // Extract bits
        let mut bits = Vec::new();
        for &pos in &positions {
            if pos >= pcm.len() {
                break;
            }
            bits.push(self.extract_bit(pcm[pos]));
        }

        let bytes = pack_bits(&bits);
        self.codec.decode(&bytes)
    }

    /// Embed bit in sample LSB.
    fn embed_bit(&self, sample: i16, bit: bool) -> i16 {
        let mask = !((1 << self.config.bits_per_sample) - 1);
        let mut result = sample & mask;

        if bit {
            result |= 1 << (self.config.bits_per_sample - 1);
        }

        result
    }

    /// Extract bit from sample LSB.
    fn extract_bit(&self, sample: i16) -> bool {
        (sample >> (self.config.bits_per_sample - 1)) & 1 == 1
    }

    /// Generate random embedding positions.
    fn generate_random_positions(&self, max_pos: usize, count: usize) -> Vec<usize> {
        let mut rng = scirs2_core::random::Random::seed(self.config.key);

        let mut positions = Vec::new();
        let mut used = vec![false; max_pos];

        while positions.len() < count && positions.len() < max_pos {
            let pos = (rng.random_f64() * max_pos as f64) as usize;
            let pos = pos.min(max_pos.saturating_sub(1));
            if !used[pos] {
                positions.push(pos);
                used[pos] = true;
            }
        }

        positions
    }

    /// Apply dithering to reduce quantization artifacts.
    fn apply_dithering(&self, samples: &mut [i16]) {
        let mut rng = scirs2_core::random::Random::seed(self.config.key.wrapping_add(1));

        let dither_range = 1 << self.config.bits_per_sample;

        for sample in samples {
            let dither =
                (rng.random_f64() * dither_range as f64) as i16 - (dither_range / 2) as i16;
            *sample = sample.saturating_add(dither);
        }
    }

    /// Convert float to 16-bit PCM.
    fn float_to_i16(&self, sample: f32) -> i16 {
        // Use round() instead of truncation to ensure symmetric roundtrip
        // through float->i16->float->i16 for reliable LSB extraction.
        (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16
    }

    /// Convert 16-bit PCM to float.
    fn i16_to_float(&self, sample: i16) -> f32 {
        // Use the same 32767.0 scale as float_to_i16 for a symmetric roundtrip.
        #[allow(clippy::cast_precision_loss)]
        let result = f32::from(sample) / 32767.0;
        result
    }

    /// Calculate capacity in bits.
    #[must_use]
    pub fn capacity(&self, sample_count: usize) -> usize {
        sample_count * self.config.bits_per_sample
    }
}

/// Multi-bit LSB embedding using gray coding.
pub struct MultiLsbEmbedder {
    bits_per_sample: usize,
}

impl MultiLsbEmbedder {
    /// Create a new multi-bit LSB embedder.
    #[must_use]
    pub fn new(bits_per_sample: usize) -> Self {
        Self { bits_per_sample }
    }

    /// Embed multiple bits per sample.
    #[must_use]
    pub fn embed(&self, sample: i16, bits: u8) -> i16 {
        let mask = !((1 << self.bits_per_sample) - 1);
        let lsb_value = bits & ((1 << self.bits_per_sample) - 1);
        (sample & mask) | i16::from(lsb_value)
    }

    /// Extract multiple bits per sample.
    #[must_use]
    pub fn extract(&self, sample: i16) -> u8 {
        (sample & ((1 << self.bits_per_sample) - 1)) as u8
    }

    /// Convert to Gray code for better error resilience.
    #[must_use]
    pub fn to_gray_code(value: u8) -> u8 {
        value ^ (value >> 1)
    }

    /// Convert from Gray code.
    #[must_use]
    pub fn from_gray_code(gray: u8) -> u8 {
        let mut value = gray;
        let mut mask = gray >> 1;
        while mask != 0 {
            value ^= mask;
            mask >>= 1;
        }
        value
    }
}

/// Adaptive LSB embedding based on signal energy.
pub struct AdaptiveLsbEmbedder {
    energy_threshold: f32,
    bits_low_energy: usize,
    bits_high_energy: usize,
}

impl AdaptiveLsbEmbedder {
    /// Create a new adaptive LSB embedder.
    #[must_use]
    pub fn new(energy_threshold: f32, bits_low: usize, bits_high: usize) -> Self {
        Self {
            energy_threshold,
            bits_low_energy: bits_low,
            bits_high_energy: bits_high,
        }
    }

    /// Calculate local energy.
    fn calculate_energy(&self, samples: &[i16], center: usize, window: usize) -> f32 {
        let start = center.saturating_sub(window / 2);
        let end = (center + window / 2).min(samples.len());

        let mut energy = 0.0f32;
        for &s in &samples[start..end] {
            energy += f32::from(s) * f32::from(s);
        }

        energy / (end - start) as f32
    }

    /// Embed adaptively based on local energy.
    #[must_use]
    pub fn embed(&self, samples: &[i16], pos: usize, bits: u8) -> i16 {
        let energy = self.calculate_energy(samples, pos, 32);
        let bits_to_use = if energy > self.energy_threshold {
            self.bits_high_energy
        } else {
            self.bits_low_energy
        };

        let mask = !((1 << bits_to_use) - 1);
        let lsb_value = bits & ((1 << bits_to_use) - 1);
        (samples[pos] & mask) | i16::from(lsb_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsb_embedding() {
        let config = LsbConfig {
            bits_per_sample: 1,
            dithering: false,
            randomize: false,
            key: 0,
        };

        let embedder = LsbEmbedder::new(config).unwrap();

        let samples: Vec<f32> = vec![0.5; 100000];
        let payload = b"Secret";

        let watermarked = embedder
            .embed(&samples, payload)
            .expect("should succeed in test");

        let encoded = embedder
            .codec
            .encode(payload)
            .expect("should succeed in test");
        let expected_bits = encoded.len() * 8;

        let extracted = embedder
            .extract(&watermarked, expected_bits)
            .expect("should succeed in test");
        assert_eq!(payload.as_slice(), extracted.as_slice());
    }

    #[test]
    fn test_multi_bit_lsb() {
        let embedder = MultiLsbEmbedder::new(3);

        let sample = 1000i16;
        let bits = 0b101u8;

        let embedded = embedder.embed(sample, bits);
        let extracted = embedder.extract(embedded);

        assert_eq!(bits, extracted);
    }

    #[test]
    fn test_gray_code() {
        for i in 0..=255u8 {
            let gray = MultiLsbEmbedder::to_gray_code(i);
            let decoded = MultiLsbEmbedder::from_gray_code(gray);
            assert_eq!(i, decoded);
        }
    }

    #[test]
    fn test_capacity() {
        let config = LsbConfig {
            bits_per_sample: 2,
            ..Default::default()
        };

        let embedder = LsbEmbedder::new(config).unwrap();
        let capacity = embedder.capacity(44100);

        assert_eq!(capacity, 44100 * 2);
    }

    #[test]
    fn test_randomized_positions() {
        let config = LsbConfig {
            bits_per_sample: 1,
            dithering: false,
            randomize: true,
            key: 12345,
        };

        let embedder = LsbEmbedder::new(config).unwrap();
        let pos1 = embedder.generate_random_positions(1000, 100);
        let pos2 = embedder.generate_random_positions(1000, 100);

        assert_eq!(pos1, pos2); // Same key = same positions
        assert_eq!(pos1.len(), 100);
    }
}
