//! LTC audio signal encoder.
//!
//! Encodes timecode values into audio-rate biphase-mark modulated samples
//! suitable for embedding in an audio track. The encoder produces `f32`
//! samples at a configurable sample rate and can generate a continuous
//! stream of LTC audio across multiple frames.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::Timecode;

// -- LtcSignalParams ---------------------------------------------------------

/// Parameters controlling the LTC audio signal generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LtcSignalParams {
    /// Audio sample rate in Hz (e.g. 48000).
    pub sample_rate: u32,
    /// Peak amplitude of the generated signal (0.0 .. 1.0).
    pub amplitude: f32,
    /// Frame rate (fps) of the timecode being encoded.
    pub fps: u8,
}

impl LtcSignalParams {
    /// Create default params: 48 kHz, amplitude 0.5, 25 fps.
    pub fn default_25fps() -> Self {
        Self {
            sample_rate: 48000,
            amplitude: 0.5,
            fps: 25,
        }
    }

    /// Create params for 30fps NTSC at 48 kHz.
    pub fn default_30fps() -> Self {
        Self {
            sample_rate: 48000,
            amplitude: 0.5,
            fps: 30,
        }
    }

    /// Samples per LTC bit at this sample rate and frame rate.
    ///
    /// LTC has 80 bits per frame, so samples_per_frame / 80.
    pub fn samples_per_bit(&self) -> f64 {
        self.sample_rate as f64 / (self.fps as f64 * 80.0)
    }

    /// Total audio samples per timecode frame.
    pub fn samples_per_frame(&self) -> u32 {
        (self.sample_rate as f64 / self.fps as f64).round() as u32
    }
}

// -- LtcBitEncoder -----------------------------------------------------------

/// Converts timecode fields into an 80-bit LTC word (as a `[u8; 80]` of 0/1).
///
/// This mirrors the SMPTE 12M standard bit layout.
#[derive(Debug, Clone)]
pub struct LtcBitEncoder;

impl LtcBitEncoder {
    /// The 16-bit sync word (bits 64..79 in an LTC word, LS bit first).
    const SYNC_WORD: [u8; 16] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1];

    /// Encode a `Timecode` into an 80-element array of logical bits (0 or 1).
    pub fn encode(tc: &Timecode) -> [u8; 80] {
        let mut word = [0u8; 80];

        // Frame units (bits 0-3)
        let fu = tc.frames % 10;
        for i in 0..4u8 {
            word[i as usize] = (fu >> i) & 1;
        }

        // Frame tens (bits 8-9), drop-frame (bit 10)
        let ft = tc.frames / 10;
        word[8] = ft & 1;
        word[9] = (ft >> 1) & 1;
        word[10] = u8::from(tc.frame_rate.drop_frame);

        // Seconds units (bits 16-19)
        let su = tc.seconds % 10;
        for i in 0..4u8 {
            word[16 + i as usize] = (su >> i) & 1;
        }

        // Seconds tens (bits 24-26)
        let st = tc.seconds / 10;
        for i in 0..3u8 {
            word[24 + i as usize] = (st >> i) & 1;
        }

        // Minutes units (bits 32-35)
        let mu = tc.minutes % 10;
        for i in 0..4u8 {
            word[32 + i as usize] = (mu >> i) & 1;
        }

        // Minutes tens (bits 40-42)
        let mt = tc.minutes / 10;
        for i in 0..3u8 {
            word[40 + i as usize] = (mt >> i) & 1;
        }

        // Hours units (bits 48-51)
        let hu = tc.hours % 10;
        for i in 0..4u8 {
            word[48 + i as usize] = (hu >> i) & 1;
        }

        // Hours tens (bits 56-57)
        let ht = tc.hours / 10;
        for i in 0..2u8 {
            word[56 + i as usize] = (ht >> i) & 1;
        }

        // Sync word (bits 64-79)
        word[64..80].copy_from_slice(&Self::SYNC_WORD);

        word
    }

    /// Decode an 80-bit word back to (hours, minutes, seconds, frames, drop_frame).
    pub fn decode(word: &[u8; 80]) -> (u8, u8, u8, u8, bool) {
        let nibble = |positions: &[usize]| -> u8 {
            positions
                .iter()
                .enumerate()
                .map(|(shift, &pos)| word[pos] << shift)
                .sum()
        };

        let frame_units = nibble(&[0, 1, 2, 3]);
        let frame_tens = nibble(&[8, 9]) & 0x03;
        let drop_frame = word[10] != 0;

        let sec_units = nibble(&[16, 17, 18, 19]);
        let sec_tens = nibble(&[24, 25, 26]) & 0x07;

        let min_units = nibble(&[32, 33, 34, 35]);
        let min_tens = nibble(&[40, 41, 42]) & 0x07;

        let hr_units = nibble(&[48, 49, 50, 51]);
        let hr_tens = nibble(&[56, 57]) & 0x03;

        let frames = frame_tens * 10 + frame_units;
        let seconds = sec_tens * 10 + sec_units;
        let minutes = min_tens * 10 + min_units;
        let hours = hr_tens * 10 + hr_units;

        (hours, minutes, seconds, frames, drop_frame)
    }
}

// -- LtcAudioEncoder ---------------------------------------------------------

/// Generates biphase-mark modulated audio samples from LTC bit words.
///
/// Biphase-mark encoding: every bit cell starts with a transition.
/// A '1' bit has an additional mid-cell transition; a '0' bit does not.
///
/// # Example
/// ```
/// use oximedia_timecode::ltc_encoder::{LtcAudioEncoder, LtcSignalParams, LtcBitEncoder};
/// use oximedia_timecode::{Timecode, FrameRate};
///
/// let params = LtcSignalParams::default_25fps();
/// let mut encoder = LtcAudioEncoder::new(params);
/// let tc = Timecode::from_raw_fields(1, 0, 0, 0, 25, false, 0);
/// let bits = LtcBitEncoder::encode(&tc);
/// let samples = encoder.encode_frame(&bits);
/// assert!(!samples.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct LtcAudioEncoder {
    /// Signal parameters.
    params: LtcSignalParams,
    /// Current polarity (+1 or -1).
    polarity: f32,
}

impl LtcAudioEncoder {
    /// Create a new audio encoder.
    pub fn new(params: LtcSignalParams) -> Self {
        Self {
            params,
            polarity: 1.0,
        }
    }

    /// Encode a single 80-bit LTC word into audio samples.
    ///
    /// Returns a `Vec<f32>` of biphase-mark modulated audio.
    pub fn encode_frame(&mut self, bits: &[u8; 80]) -> Vec<f32> {
        let spb = self.params.samples_per_bit();
        let amplitude = self.params.amplitude;
        let mut samples = Vec::with_capacity(self.params.samples_per_frame() as usize);

        for &bit in bits.iter() {
            let num_samples = spb.round() as usize;
            let half = num_samples / 2;

            if bit == 1 {
                // '1' bit: transition at start and mid-cell
                for _ in 0..half {
                    samples.push(self.polarity * amplitude);
                }
                self.polarity = -self.polarity;
                for _ in half..num_samples {
                    samples.push(self.polarity * amplitude);
                }
                self.polarity = -self.polarity;
            } else {
                // '0' bit: transition at start only
                for _ in 0..num_samples {
                    samples.push(self.polarity * amplitude);
                }
                self.polarity = -self.polarity;
            }
        }

        samples
    }

    /// Encode multiple consecutive timecodes into a continuous audio stream.
    pub fn encode_sequence(&mut self, timecodes: &[Timecode]) -> Vec<f32> {
        let mut all_samples = Vec::new();
        for tc in timecodes {
            let bits = LtcBitEncoder::encode(tc);
            let frame_samples = self.encode_frame(&bits);
            all_samples.extend_from_slice(&frame_samples);
        }
        all_samples
    }

    /// Return the current polarity state.
    pub fn polarity(&self) -> f32 {
        self.polarity
    }

    /// Reset the polarity to +1.
    pub fn reset_polarity(&mut self) {
        self.polarity = 1.0;
    }

    /// Return the signal parameters.
    pub fn params(&self) -> &LtcSignalParams {
        &self.params
    }
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tc(h: u8, m: u8, s: u8, f: u8, fps: u8, df: bool) -> Timecode {
        Timecode::from_raw_fields(h, m, s, f, fps, df, 0)
    }

    fn default_params() -> LtcSignalParams {
        LtcSignalParams::default_25fps()
    }

    #[test]
    fn test_signal_params_samples_per_bit_25fps() {
        let p = default_params();
        let spb = p.samples_per_bit();
        // 48000 / (25 * 80) = 24.0
        assert!((spb - 24.0).abs() < 1e-6);
    }

    #[test]
    fn test_signal_params_samples_per_frame_25fps() {
        let p = default_params();
        assert_eq!(p.samples_per_frame(), 1920); // 48000 / 25
    }

    #[test]
    fn test_signal_params_30fps() {
        let p = LtcSignalParams::default_30fps();
        assert_eq!(p.fps, 30);
        assert_eq!(p.samples_per_frame(), 1600); // 48000 / 30
    }

    #[test]
    fn test_bit_encoder_roundtrip() {
        let tc = make_tc(12, 34, 56, 7, 25, false);
        let bits = LtcBitEncoder::encode(&tc);
        let (h, m, s, f, df) = LtcBitEncoder::decode(&bits);
        assert_eq!(h, 12);
        assert_eq!(m, 34);
        assert_eq!(s, 56);
        assert_eq!(f, 7);
        assert!(!df);
    }

    #[test]
    fn test_bit_encoder_drop_frame_flag() {
        let tc = make_tc(0, 0, 0, 2, 30, true);
        let bits = LtcBitEncoder::encode(&tc);
        let (_, _, _, _, df) = LtcBitEncoder::decode(&bits);
        assert!(df);
    }

    #[test]
    fn test_bit_encoder_midnight() {
        let tc = make_tc(0, 0, 0, 0, 25, false);
        let bits = LtcBitEncoder::encode(&tc);
        let (h, m, s, f, _) = LtcBitEncoder::decode(&bits);
        assert_eq!((h, m, s, f), (0, 0, 0, 0));
    }

    #[test]
    fn test_bit_encoder_max_values() {
        let tc = make_tc(23, 59, 59, 24, 25, false);
        let bits = LtcBitEncoder::encode(&tc);
        let (h, m, s, f, _) = LtcBitEncoder::decode(&bits);
        assert_eq!(h, 23);
        assert_eq!(m, 59);
        assert_eq!(s, 59);
        assert_eq!(f, 24);
    }

    #[test]
    fn test_bit_encoder_sync_word_present() {
        let tc = make_tc(0, 0, 0, 0, 25, false);
        let bits = LtcBitEncoder::encode(&tc);
        assert_eq!(&bits[64..80], &LtcBitEncoder::SYNC_WORD);
    }

    #[test]
    fn test_audio_encoder_output_length() {
        let params = default_params();
        let mut enc = LtcAudioEncoder::new(params);
        let tc = make_tc(0, 0, 0, 0, 25, false);
        let bits = LtcBitEncoder::encode(&tc);
        let samples = enc.encode_frame(&bits);
        // Each bit produces ~24 samples at 48kHz/25fps → ~1920 total
        assert!(!samples.is_empty());
        assert!(samples.len() >= 1900 && samples.len() <= 1940);
    }

    #[test]
    fn test_audio_encoder_amplitude_bounds() {
        let params = LtcSignalParams {
            sample_rate: 48000,
            amplitude: 0.8,
            fps: 25,
        };
        let mut enc = LtcAudioEncoder::new(params);
        let tc = make_tc(1, 0, 0, 0, 25, false);
        let bits = LtcBitEncoder::encode(&tc);
        let samples = enc.encode_frame(&bits);
        for &s in &samples {
            assert!(s.abs() <= 0.8 + 1e-6);
        }
    }

    #[test]
    fn test_audio_encoder_polarity_reset() {
        let mut enc = LtcAudioEncoder::new(default_params());
        assert!((enc.polarity() - 1.0).abs() < 1e-6);
        enc.polarity = -1.0;
        enc.reset_polarity();
        assert!((enc.polarity() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_audio_encoder_sequence() {
        let mut enc = LtcAudioEncoder::new(default_params());
        let tcs = vec![
            make_tc(0, 0, 0, 0, 25, false),
            make_tc(0, 0, 0, 1, 25, false),
        ];
        let samples = enc.encode_sequence(&tcs);
        // Should be approximately 2 * 1920 samples
        assert!(samples.len() >= 3800);
    }

    #[test]
    fn test_audio_encoder_params_accessor() {
        let params = default_params();
        let enc = LtcAudioEncoder::new(params);
        assert_eq!(enc.params().fps, 25);
        assert_eq!(enc.params().sample_rate, 48000);
    }
}
