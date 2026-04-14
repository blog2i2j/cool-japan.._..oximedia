//! Surround sound upmixing algorithms.
//!
//! Provides stereo-to-5.1 and 5.1-to-7.1 upmixing with multiple algorithms:
//! - **Passive**: Frequency-based steering of spectral content to channels
//! - **MatrixDecode**: Dolby Pro Logic II style L-R / L+R extraction for center and surround
//! - **AmbientExtract**: Decorrelated ambient field extraction routed to surround channels
//!
//! # Example
//!
//! ```
//! use oximedia_audiopost::surround_upmix::{SurroundUpmixer, UpmixAlgorithm, UpmixConfig};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = UpmixConfig::default();
//! let mut upmixer = SurroundUpmixer::new(48000, UpmixAlgorithm::MatrixDecode, config)?;
//!
//! let left = vec![0.5_f32; 1024];
//! let right = vec![0.3_f32; 1024];
//! let result = upmixer.upmix_stereo_to_51(&left, &right)?;
//! assert_eq!(result.len(), 6);
//! # Ok(())
//! # }
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::error::{AudioPostError, AudioPostResult};
use oxifft::{fft, ifft, Complex};

/// Upmixing algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpmixAlgorithm {
    /// Frequency-based steering — routes spectral bands to appropriate channels.
    Passive,
    /// Dolby Pro Logic II style matrix decoding using L-R and L+R extraction.
    MatrixDecode,
    /// Decorrelated ambient extraction routed to surround channels.
    AmbientExtract,
}

/// Configuration for the surround upmixer.
#[derive(Debug, Clone)]
pub struct UpmixConfig {
    /// Center channel extraction width in Hz (default: 300 Hz).
    /// Wider values route more of the mono-correlated signal to the center.
    pub center_width_hz: f32,
    /// LFE crossover frequency in Hz (default: 120 Hz).
    /// Content below this frequency is routed to the LFE channel.
    pub lfe_crossover_hz: f32,
    /// LFE gain in linear scale (default: 0.707 / -3 dB).
    pub lfe_gain: f32,
    /// Center channel gain in linear scale (default: 0.707).
    pub center_gain: f32,
    /// Surround channel gain in linear scale (default: 0.5).
    pub surround_gain: f32,
    /// Decorrelation amount for ambient extraction (0.0–1.0, default: 0.7).
    pub decorrelation_amount: f32,
}

impl Default for UpmixConfig {
    fn default() -> Self {
        Self {
            center_width_hz: 300.0,
            lfe_crossover_hz: 120.0,
            lfe_gain: 0.707,
            center_gain: 0.707,
            surround_gain: 0.5,
            decorrelation_amount: 0.7,
        }
    }
}

/// Surround sound upmixer supporting stereo-to-5.1 and 5.1-to-7.1 conversions.
#[derive(Debug)]
pub struct SurroundUpmixer {
    sample_rate: u32,
    algorithm: UpmixAlgorithm,
    config: UpmixConfig,
    /// Allpass state for decorrelation (per-channel).
    allpass_state: Vec<f32>,
}

impl SurroundUpmixer {
    /// Create a new surround upmixer.
    ///
    /// # Errors
    ///
    /// Returns an error if the sample rate is zero or the config is invalid.
    pub fn new(
        sample_rate: u32,
        algorithm: UpmixAlgorithm,
        config: UpmixConfig,
    ) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if config.lfe_crossover_hz <= 0.0 || config.lfe_crossover_hz >= sample_rate as f32 / 2.0 {
            return Err(AudioPostError::InvalidFrequency(config.lfe_crossover_hz));
        }
        if config.center_width_hz <= 0.0 {
            return Err(AudioPostError::InvalidFrequency(config.center_width_hz));
        }
        Ok(Self {
            sample_rate,
            algorithm,
            config,
            allpass_state: vec![0.0; 8],
        })
    }

    /// Get the current algorithm.
    #[must_use]
    pub fn algorithm(&self) -> UpmixAlgorithm {
        self.algorithm
    }

    /// Set the upmix algorithm.
    pub fn set_algorithm(&mut self, algorithm: UpmixAlgorithm) {
        self.algorithm = algorithm;
    }

    /// Set LFE crossover frequency.
    ///
    /// # Errors
    ///
    /// Returns an error if the frequency is out of range.
    pub fn set_lfe_crossover(&mut self, freq_hz: f32) -> AudioPostResult<()> {
        if freq_hz <= 0.0 || freq_hz >= self.sample_rate as f32 / 2.0 {
            return Err(AudioPostError::InvalidFrequency(freq_hz));
        }
        self.config.lfe_crossover_hz = freq_hz;
        Ok(())
    }

    /// Upmix stereo audio to 5.1 surround.
    ///
    /// Returns six channels: `[L, R, C, LFE, Ls, Rs]`.
    ///
    /// # Errors
    ///
    /// Returns an error if left and right channels have mismatched lengths or are empty.
    pub fn upmix_stereo_to_51(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> AudioPostResult<Vec<Vec<f32>>> {
        if left.len() != right.len() {
            return Err(AudioPostError::InvalidBufferSize(left.len()));
        }
        if left.is_empty() {
            return Err(AudioPostError::InvalidBufferSize(0));
        }

        match self.algorithm {
            UpmixAlgorithm::Passive => self.upmix_passive_51(left, right),
            UpmixAlgorithm::MatrixDecode => self.upmix_matrix_decode_51(left, right),
            UpmixAlgorithm::AmbientExtract => self.upmix_ambient_extract_51(left, right),
        }
    }

    /// Upmix 5.1 audio to 7.1 surround.
    ///
    /// Input channels: `[L, R, C, LFE, Ls, Rs]`.
    /// Returns eight channels: `[L, R, C, LFE, Lss, Rss, Lrs, Rrs]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the input does not have exactly 6 channels or channels are mismatched.
    pub fn upmix_51_to_71(
        &mut self,
        channels_51: &[Vec<f32>],
    ) -> AudioPostResult<Vec<Vec<f32>>> {
        if channels_51.len() != 6 {
            return Err(AudioPostError::InvalidChannelCount(channels_51.len()));
        }
        let len = channels_51[0].len();
        if len == 0 {
            return Err(AudioPostError::InvalidBufferSize(0));
        }
        for ch in channels_51 {
            if ch.len() != len {
                return Err(AudioPostError::InvalidBufferSize(ch.len()));
            }
        }

        // L, R, C, LFE pass through; split Ls/Rs into side and rear pairs
        let mut output = Vec::with_capacity(8);

        // L, R, C, LFE — direct copy
        for ch in &channels_51[..4] {
            output.push(ch.clone());
        }

        let ls = &channels_51[4];
        let rs = &channels_51[5];

        // Side channels (Lss, Rss): more direct content
        let mut lss = vec![0.0_f32; len];
        let mut rss = vec![0.0_f32; len];
        // Rear channels (Lrs, Rrs): decorrelated / delayed content
        let mut lrs = vec![0.0_f32; len];
        let mut rrs = vec![0.0_f32; len];

        let side_gain = 0.7071; // -3 dB
        let rear_gain = 0.7071;

        for i in 0..len {
            let sum = (ls[i] + rs[i]) * 0.5;
            let diff = (ls[i] - rs[i]) * 0.5;

            // Side: direct portion
            lss[i] = (ls[i] * 0.6 + sum * 0.4) * side_gain;
            rss[i] = (rs[i] * 0.6 + sum * 0.4) * side_gain;

            // Rear: ambient / decorrelated portion
            lrs[i] = (ls[i] * 0.4 + diff * 0.3) * rear_gain;
            rrs[i] = (rs[i] * 0.4 - diff * 0.3) * rear_gain;
        }

        // Apply simple decorrelation to rear channels via allpass
        self.apply_allpass_decorrelation(&mut lrs);
        self.apply_allpass_decorrelation(&mut rrs);

        output.push(lss);
        output.push(rss);
        output.push(lrs);
        output.push(rrs);

        Ok(output)
    }

    // -- Passive upmix (frequency-based steering) --

    fn upmix_passive_51(&self, left: &[f32], right: &[f32]) -> AudioPostResult<Vec<Vec<f32>>> {
        let len = left.len();
        let fft_size = len.next_power_of_two();

        // Prepare FFT buffers (zero-padded to power of two)
        let mut left_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); fft_size];
        let mut right_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); fft_size];

        for i in 0..len {
            left_buf[i] = Complex::new(left[i], 0.0);
            right_buf[i] = Complex::new(right[i], 0.0);
        }

        let left_fft = fft(&left_buf);
        let right_fft = fft(&right_buf);

        let nyquist = self.sample_rate as f32 / 2.0;
        let lfe_bin = (self.config.lfe_crossover_hz / nyquist * fft_size as f32 / 2.0) as usize;
        let center_bin = (self.config.center_width_hz / nyquist * fft_size as f32 / 2.0) as usize;

        let mut center_fft = vec![Complex::new(0.0_f32, 0.0); fft_size];
        let mut lfe_fft_buf = vec![Complex::new(0.0_f32, 0.0); fft_size];
        let mut ls_fft = vec![Complex::new(0.0_f32, 0.0); fft_size];
        let mut rs_fft = vec![Complex::new(0.0_f32, 0.0); fft_size];
        let mut l_out_fft = left_fft.clone();
        let mut r_out_fft = right_fft.clone();

        let half = fft_size / 2;
        for bin in 0..=half {
            let mirror = if bin > 0 && bin < half { fft_size - bin } else { bin };
            let sum = (left_fft[bin] + right_fft[bin]) * 0.5;
            let diff = (left_fft[bin] - right_fft[bin]) * 0.5;

            // LFE: low-frequency content
            if bin <= lfe_bin {
                lfe_fft_buf[bin] = sum * self.config.lfe_gain;
                if mirror != bin {
                    lfe_fft_buf[mirror] = lfe_fft_buf[bin].conj();
                }
            }

            // Center: mono-correlated mid-range content
            if bin <= center_bin {
                center_fft[bin] = sum * self.config.center_gain;
                if mirror != bin {
                    center_fft[mirror] = center_fft[bin].conj();
                }
                // Reduce center content from L/R
                l_out_fft[bin] = l_out_fft[bin] * 0.5;
                r_out_fft[bin] = r_out_fft[bin] * 0.5;
                if mirror != bin {
                    l_out_fft[mirror] = l_out_fft[mirror] * 0.5;
                    r_out_fft[mirror] = r_out_fft[mirror] * 0.5;
                }
            }

            // Surrounds: difference signal (ambient content)
            ls_fft[bin] = diff * self.config.surround_gain;
            rs_fft[bin] = (diff * -1.0) * self.config.surround_gain;
            if mirror != bin {
                ls_fft[mirror] = ls_fft[bin].conj();
                rs_fft[mirror] = rs_fft[bin].conj();
            }
        }

        // IFFT
        let l_out = ifft(&l_out_fft);
        let r_out = ifft(&r_out_fft);
        let center_out = ifft(&center_fft);
        let lfe_out = ifft(&lfe_fft_buf);
        let ls_out = ifft(&ls_fft);
        let rs_out = ifft(&rs_fft);

        let scale = 1.0 / fft_size as f32;

        let extract = |buf: &[Complex<f32>]| -> Vec<f32> {
            buf[..len].iter().map(|c| c.re * scale).collect()
        };

        Ok(vec![
            extract(&l_out),
            extract(&r_out),
            extract(&center_out),
            extract(&lfe_out),
            extract(&ls_out),
            extract(&rs_out),
        ])
    }

    // -- Matrix Decode (Dolby Pro Logic II style) --

    fn upmix_matrix_decode_51(
        &self,
        left: &[f32],
        right: &[f32],
    ) -> AudioPostResult<Vec<Vec<f32>>> {
        let len = left.len();
        let mut l_out = Vec::with_capacity(len);
        let mut r_out = Vec::with_capacity(len);
        let mut center = Vec::with_capacity(len);
        let mut lfe = Vec::with_capacity(len);
        let mut ls = Vec::with_capacity(len);
        let mut rs = Vec::with_capacity(len);

        let c_gain = self.config.center_gain;
        let surr_gain = self.config.surround_gain;

        // Simple 1-pole lowpass state for LFE
        let lfe_alpha = compute_lp_alpha(self.config.lfe_crossover_hz, self.sample_rate);
        let mut lfe_state = 0.0_f32;

        for i in 0..len {
            let l = left[i];
            let r = right[i];

            // Pro Logic II style extraction
            let sum = (l + r) * 0.5;
            let diff = (l - r) * 0.5;

            // Center = L+R (mono-correlated)
            let c = sum * c_gain;
            center.push(c);

            // Front L/R: original minus center bleed
            l_out.push(l - c * 0.5);
            r_out.push(r - c * 0.5);

            // Surround: L-R (difference / ambient)
            ls.push(diff * surr_gain);
            rs.push(-diff * surr_gain);

            // LFE: lowpassed sum
            lfe_state += lfe_alpha * (sum - lfe_state);
            lfe.push(lfe_state * self.config.lfe_gain);
        }

        Ok(vec![l_out, r_out, center, lfe, ls, rs])
    }

    // -- Ambient Extract --

    fn upmix_ambient_extract_51(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> AudioPostResult<Vec<Vec<f32>>> {
        let len = left.len();
        let mut l_out = Vec::with_capacity(len);
        let mut r_out = Vec::with_capacity(len);
        let mut center = Vec::with_capacity(len);
        let mut lfe = Vec::with_capacity(len);
        let mut ls = Vec::with_capacity(len);
        let mut rs = Vec::with_capacity(len);

        let c_gain = self.config.center_gain;
        let surr_gain = self.config.surround_gain;
        let decor = self.config.decorrelation_amount;

        let lfe_alpha = compute_lp_alpha(self.config.lfe_crossover_hz, self.sample_rate);
        let mut lfe_state = 0.0_f32;

        for i in 0..len {
            let l = left[i];
            let r = right[i];

            let sum = (l + r) * 0.5;
            let diff = (l - r) * 0.5;

            // Center: mono-correlated signal
            let c = sum * c_gain;
            center.push(c);

            // Front L/R: direct signal
            l_out.push(l - c * 0.3);
            r_out.push(r - c * 0.3);

            // Ambient: decorrelated difference signal
            // Apply a simple decorrelation by mixing with a phase-shifted version
            let ambient_l = diff * decor + l * (1.0 - decor) * 0.3;
            let ambient_r = -diff * decor + r * (1.0 - decor) * 0.3;
            ls.push(ambient_l * surr_gain);
            rs.push(ambient_r * surr_gain);

            // LFE
            lfe_state += lfe_alpha * (sum - lfe_state);
            lfe.push(lfe_state * self.config.lfe_gain);
        }

        // Apply allpass decorrelation to surround channels
        self.apply_allpass_decorrelation(&mut ls);
        self.apply_allpass_decorrelation(&mut rs);

        Ok(vec![l_out, r_out, center, lfe, ls, rs])
    }

    /// Apply a simple first-order allpass for decorrelation.
    fn apply_allpass_decorrelation(&mut self, buffer: &mut [f32]) {
        let coeff = 0.6_f32; // allpass coefficient
        let state_idx = 0;
        let mut state = if self.allpass_state.len() > state_idx {
            self.allpass_state[state_idx]
        } else {
            0.0
        };

        for sample in buffer.iter_mut() {
            let input = *sample;
            let output = state + input * coeff;
            state = input - output * coeff;
            *sample = output;
        }

        if self.allpass_state.len() > state_idx {
            self.allpass_state[state_idx] = state;
        }
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        for s in &mut self.allpass_state {
            *s = 0.0;
        }
    }
}

/// Compute 1-pole lowpass filter coefficient.
fn compute_lp_alpha(cutoff_hz: f32, sample_rate: u32) -> f32 {
    let dt = 1.0 / sample_rate as f32;
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    dt / (rc + dt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_passive_upmix_produces_six_channels() {
        let config = UpmixConfig::default();
        let mut upmixer =
            SurroundUpmixer::new(48000, UpmixAlgorithm::Passive, config).expect("create upmixer");
        let left = make_sine(440.0, 48000, 1024);
        let right = make_sine(440.0, 48000, 1024);
        let result = upmixer
            .upmix_stereo_to_51(&left, &right)
            .expect("upmix");
        assert_eq!(result.len(), 6);
        for ch in &result {
            assert_eq!(ch.len(), 1024);
        }
    }

    #[test]
    fn test_matrix_decode_center_extraction() {
        let config = UpmixConfig::default();
        let mut upmixer = SurroundUpmixer::new(48000, UpmixAlgorithm::MatrixDecode, config)
            .expect("create upmixer");

        // Identical L/R should produce strong center
        let mono = vec![0.5_f32; 512];
        let result = upmixer
            .upmix_stereo_to_51(&mono, &mono)
            .expect("upmix");

        let center = &result[2];
        let center_energy: f32 = center.iter().map(|s| s * s).sum();
        assert!(center_energy > 0.0, "Center should have energy for mono signal");

        // Surround should be near-silent for identical L/R
        let ls = &result[4];
        let ls_energy: f32 = ls.iter().map(|s| s * s).sum();
        assert!(
            ls_energy < center_energy * 0.01,
            "Surround should be quiet for mono signal"
        );
    }

    #[test]
    fn test_matrix_decode_surround_from_difference() {
        let config = UpmixConfig::default();
        let mut upmixer = SurroundUpmixer::new(48000, UpmixAlgorithm::MatrixDecode, config)
            .expect("create upmixer");

        // Out-of-phase signals => strong surround
        let left = vec![0.5_f32; 512];
        let right = vec![-0.5_f32; 512];
        let result = upmixer
            .upmix_stereo_to_51(&left, &right)
            .expect("upmix");

        let ls_energy: f32 = result[4].iter().map(|s| s * s).sum();
        let center_energy: f32 = result[2].iter().map(|s| s * s).sum();

        assert!(
            ls_energy > center_energy,
            "Out-of-phase input should produce surround > center"
        );
    }

    #[test]
    fn test_ambient_extract_upmix() {
        let config = UpmixConfig::default();
        let mut upmixer = SurroundUpmixer::new(48000, UpmixAlgorithm::AmbientExtract, config)
            .expect("create upmixer");

        let left = make_sine(200.0, 48000, 2048);
        let right = make_sine(300.0, 48000, 2048);
        let result = upmixer
            .upmix_stereo_to_51(&left, &right)
            .expect("upmix");

        assert_eq!(result.len(), 6);
        // Surround channels should have content due to L/R difference
        let ls_energy: f32 = result[4].iter().map(|s| s * s).sum();
        assert!(ls_energy > 0.0, "Surround should have energy");
    }

    #[test]
    fn test_lfe_crossover_lowpass() {
        let config = UpmixConfig {
            lfe_crossover_hz: 120.0,
            ..UpmixConfig::default()
        };
        let mut upmixer = SurroundUpmixer::new(48000, UpmixAlgorithm::MatrixDecode, config)
            .expect("create upmixer");

        // Low frequency signal
        let low = make_sine(60.0, 48000, 4096);
        let result = upmixer
            .upmix_stereo_to_51(&low, &low)
            .expect("upmix");

        let lfe_energy: f32 = result[3].iter().map(|s| s * s).sum();
        assert!(lfe_energy > 0.0, "LFE should capture low-frequency content");
    }

    #[test]
    fn test_51_to_71_upmix() {
        let config = UpmixConfig::default();
        let mut upmixer = SurroundUpmixer::new(48000, UpmixAlgorithm::Passive, config)
            .expect("create upmixer");

        let channels_51: Vec<Vec<f32>> = (0..6)
            .map(|i| make_sine(100.0 * (i as f32 + 1.0), 48000, 1024))
            .collect();

        let result = upmixer.upmix_51_to_71(&channels_51).expect("upmix 7.1");
        assert_eq!(result.len(), 8);
        for ch in &result {
            assert_eq!(ch.len(), 1024);
        }

        // L, R, C, LFE should pass through unchanged
        assert_eq!(result[0], channels_51[0]);
        assert_eq!(result[1], channels_51[1]);
        assert_eq!(result[2], channels_51[2]);
        assert_eq!(result[3], channels_51[3]);
    }

    #[test]
    fn test_invalid_sample_rate() {
        let result = SurroundUpmixer::new(0, UpmixAlgorithm::Passive, UpmixConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_channel_lengths() {
        let config = UpmixConfig::default();
        let mut upmixer =
            SurroundUpmixer::new(48000, UpmixAlgorithm::Passive, config).expect("create");
        let left = vec![0.0; 100];
        let right = vec![0.0; 200];
        let result = upmixer.upmix_stereo_to_51(&left, &right);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input_rejected() {
        let config = UpmixConfig::default();
        let mut upmixer =
            SurroundUpmixer::new(48000, UpmixAlgorithm::MatrixDecode, config).expect("create");
        let result = upmixer.upmix_stereo_to_51(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_lfe_crossover() {
        let config = UpmixConfig::default();
        let mut upmixer =
            SurroundUpmixer::new(48000, UpmixAlgorithm::Passive, config).expect("create");
        assert!(upmixer.set_lfe_crossover(80.0).is_ok());
        assert!(upmixer.set_lfe_crossover(0.0).is_err());
        assert!(upmixer.set_lfe_crossover(30000.0).is_err());
    }

    #[test]
    fn test_reset_clears_state() {
        let config = UpmixConfig::default();
        let mut upmixer = SurroundUpmixer::new(48000, UpmixAlgorithm::AmbientExtract, config)
            .expect("create");
        let left = make_sine(440.0, 48000, 512);
        let right = make_sine(220.0, 48000, 512);
        let _ = upmixer.upmix_stereo_to_51(&left, &right);
        upmixer.reset();
        assert!(upmixer.allpass_state.iter().all(|&s| s == 0.0));
    }
}
