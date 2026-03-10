//! Audio concealment.
//!
//! This module provides functions for concealing corrupt audio samples.

use crate::Result;

/// Conceal corrupt audio by inserting silence.
pub fn insert_silence(samples: &mut [u8]) {
    samples.fill(0);
}

/// Conceal corrupt audio by fading.
pub fn conceal_with_fade(corrupt_samples: &mut [u8], previous_samples: &[u8]) -> Result<()> {
    let len = corrupt_samples.len().min(previous_samples.len());

    for i in 0..len {
        let t = i as f32 / len as f32;
        corrupt_samples[i] = (previous_samples[i] as f32 * (1.0 - t)) as u8;
    }

    Ok(())
}

/// Conceal corrupt audio by interpolation.
pub fn conceal_audio_interpolation(
    corrupt_samples: &mut [u8],
    previous_samples: &[u8],
    next_samples: &[u8],
) -> Result<()> {
    let len = corrupt_samples
        .len()
        .min(previous_samples.len())
        .min(next_samples.len());

    for i in 0..len {
        let t = i as f32 / len as f32;
        let value = previous_samples[i] as f32 * (1.0 - t) + next_samples[i] as f32 * t;
        corrupt_samples[i] = value as u8;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_silence() {
        let mut samples = vec![100; 10];
        insert_silence(&mut samples);
        assert!(samples.iter().all(|&s| s == 0));
    }

    #[test]
    fn test_conceal_audio_interpolation() {
        let mut corrupt = vec![0; 4];
        let prev = vec![0, 0, 0, 0];
        let next = vec![100, 100, 100, 100];

        conceal_audio_interpolation(&mut corrupt, &prev, &next)
            .expect("audio concealment should succeed");
        assert!(corrupt[0] < corrupt[3]);
    }
}
