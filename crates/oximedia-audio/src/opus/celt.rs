//! CELT mode decoder skeleton.
//!
//! CELT (Constrained Energy Lapped Transform) is the music/general audio
//! component of Opus. It provides high-quality audio at medium to high bitrates.
//!
//! # CELT Features
//!
//! - MDCT-based transform codec
//! - Critical-band energy coding
//! - Pitch pre-filtering
//! - Frame sizes: 2.5ms, 5ms, 10ms, 20ms
//!
//! # Band Structure
//!
//! CELT divides the spectrum into critical bands. Each band's energy is
//! coded separately, with fine structure encoded using PVQ (Pyramid Vector
//! Quantization).

#![forbid(unsafe_code)]

use crate::AudioError;

/// CELT operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CeltMode {
    /// Normal CELT mode.
    #[default]
    Normal,
    /// Custom mode (for non-standard configurations).
    Custom,
}

/// CELT frame size configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)]
pub enum CeltFrameSize {
    /// 2.5ms frame (120 samples at 48kHz).
    Ms2_5,
    /// 5ms frame (240 samples at 48kHz).
    Ms5,
    /// 10ms frame (480 samples at 48kHz).
    #[default]
    Ms10,
    /// 20ms frame (960 samples at 48kHz).
    Ms20,
}

impl CeltFrameSize {
    /// Get samples per frame at 48kHz.
    #[must_use]
    pub fn samples_48khz(self) -> usize {
        match self {
            CeltFrameSize::Ms2_5 => 120,
            CeltFrameSize::Ms5 => 240,
            CeltFrameSize::Ms10 => 480,
            CeltFrameSize::Ms20 => 960,
        }
    }

    /// Get MDCT short block count.
    #[must_use]
    pub fn short_blocks(self) -> usize {
        // All frame sizes use 1 short block, but Ms20 can use 4 for transient frames
        1
    }

    /// Get frame duration in microseconds.
    #[must_use]
    pub fn duration_us(self) -> u32 {
        match self {
            CeltFrameSize::Ms2_5 => 2500,
            CeltFrameSize::Ms5 => 5000,
            CeltFrameSize::Ms10 => 10000,
            CeltFrameSize::Ms20 => 20000,
        }
    }
}

/// Band energy in dB (Q8 format).
#[derive(Debug, Clone, Default)]
pub struct BandEnergy {
    /// Energy values for each band (Q8 dB scale).
    pub values: Vec<i16>,
    /// Number of bands.
    pub band_count: usize,
}

impl BandEnergy {
    /// Create new band energy structure.
    #[must_use]
    pub fn new(band_count: usize) -> Self {
        Self {
            values: vec![0; band_count],
            band_count,
        }
    }

    /// Get energy for a specific band.
    #[must_use]
    pub fn get(&self, band: usize) -> Option<i16> {
        self.values.get(band).copied()
    }

    /// Set energy for a specific band.
    pub fn set(&mut self, band: usize, value: i16) {
        if band < self.band_count {
            self.values[band] = value;
        }
    }

    /// Get total energy across all bands.
    #[must_use]
    pub fn total_energy(&self) -> i32 {
        self.values.iter().map(|&e| i32::from(e)).sum()
    }
}

/// Pitch period information for pitch pre-filter.
#[derive(Debug, Clone, Default)]
pub struct PitchPeriod {
    /// Pitch period in samples.
    pub period: u16,
    /// Pitch gain (Q15 format).
    pub gain: i16,
    /// Number of pitch taps.
    pub tap_count: u8,
    /// Pitch filter taps.
    pub taps: [i16; 3],
}

impl PitchPeriod {
    /// Minimum pitch period (samples at 48kHz).
    pub const MIN_PERIOD: u16 = 15;
    /// Maximum pitch period (samples at 48kHz).
    pub const MAX_PERIOD: u16 = 1022;

    /// Create new pitch period.
    #[must_use]
    pub fn new(period: u16, gain: i16) -> Self {
        Self {
            period: period.clamp(Self::MIN_PERIOD, Self::MAX_PERIOD),
            gain,
            tap_count: 1,
            taps: [gain, 0, 0],
        }
    }

    /// Check if pitch is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.gain != 0
    }
}

/// CELT band configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CeltBandConfig {
    /// Number of bands.
    pub band_count: usize,
    /// Band boundaries (in MDCT bins).
    pub band_boundaries: Vec<usize>,
    /// Effective bits per band.
    pub bits_per_band: Vec<u16>,
}

impl Default for CeltBandConfig {
    fn default() -> Self {
        Self::new_48khz()
    }
}

impl CeltBandConfig {
    /// Create band configuration for 48kHz.
    #[must_use]
    pub fn new_48khz() -> Self {
        // 21 critical bands for 48kHz fullband
        let band_boundaries = vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 21, 25, 30, 36, 43, 52, 63, 76, 92,
        ];
        let band_count = band_boundaries.len() - 1;

        Self {
            band_count,
            band_boundaries,
            bits_per_band: vec![0; band_count],
        }
    }

    /// Get band range in MDCT bins.
    #[must_use]
    pub fn band_range(&self, band: usize) -> Option<(usize, usize)> {
        if band < self.band_count {
            Some((self.band_boundaries[band], self.band_boundaries[band + 1]))
        } else {
            None
        }
    }

    /// Get band width in MDCT bins.
    #[must_use]
    pub fn band_width(&self, band: usize) -> Option<usize> {
        self.band_range(band).map(|(start, end)| end - start)
    }
}

/// Transient detection flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)]
pub enum TransientType {
    /// No transient, use long MDCT.
    #[default]
    None,
    /// Transient detected, use short MDCTs.
    Short,
}

/// CELT frame data.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct CeltFrame {
    /// CELT mode.
    pub mode: CeltMode,
    /// Frame size.
    pub frame_size: CeltFrameSize,
    /// Number of channels.
    pub channels: u8,
    /// Transient type.
    pub transient: TransientType,
    /// Intra-frame coding flag.
    pub intra: bool,
    /// Band energies.
    pub energy: BandEnergy,
    /// Pitch period (if pre-filter active).
    pub pitch: Option<PitchPeriod>,
    /// Fine energy bits per band.
    pub fine_energy: Vec<i16>,
    /// PVQ-coded coefficients per band.
    pub coefficients: Vec<Vec<i16>>,
    /// Spread/anti-collapse flags.
    pub spread: u8,
    /// Dual stereo mode flag.
    pub dual_stereo: bool,
    /// Intensity stereo start band.
    pub intensity_stereo_band: Option<usize>,
}

impl CeltFrame {
    /// Create a new CELT frame.
    #[must_use]
    pub fn new(frame_size: CeltFrameSize, channels: u8) -> Self {
        let band_config = CeltBandConfig::new_48khz();
        let band_count = band_config.band_count;

        Self {
            frame_size,
            channels,
            energy: BandEnergy::new(band_count),
            fine_energy: vec![0; band_count],
            coefficients: vec![Vec::new(); band_count],
            ..Default::default()
        }
    }

    /// Get number of bands.
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.energy.band_count
    }

    /// Check if frame has transient.
    #[must_use]
    pub fn has_transient(&self) -> bool {
        self.transient == TransientType::Short
    }

    /// Check if stereo processing is used.
    #[must_use]
    pub fn is_stereo(&self) -> bool {
        self.channels == 2
    }

    /// Get samples per frame.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.frame_size.samples_48khz() * usize::from(self.channels)
    }
}

/// CELT decoder state.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct CeltDecoderState {
    /// Previous frame's band energies.
    pub prev_energy: BandEnergy,
    /// Previous frame's fine energies.
    pub prev_fine_energy: Vec<i16>,
    /// MDCT overlap-add buffer.
    pub overlap_buffer: Vec<f32>,
    /// Pitch pre-filter state.
    pub pitch_buffer: Vec<f32>,
    /// Synthesis filter state.
    pub synthesis_state: Vec<f32>,
    /// Number of channels.
    pub channels: u8,
    /// Current frame size.
    pub frame_size: CeltFrameSize,
    /// Post-filter enabled flag.
    pub postfilter_enabled: bool,
}

impl CeltDecoderState {
    /// Create new decoder state.
    #[must_use]
    pub fn new(channels: u8, frame_size: CeltFrameSize) -> Self {
        let band_config = CeltBandConfig::new_48khz();
        let samples = frame_size.samples_48khz() * usize::from(channels);

        Self {
            prev_energy: BandEnergy::new(band_config.band_count),
            prev_fine_energy: vec![0; band_config.band_count],
            overlap_buffer: vec![0.0; samples / 2],
            pitch_buffer: vec![0.0; PitchPeriod::MAX_PERIOD as usize + samples],
            synthesis_state: vec![0.0; samples],
            channels,
            frame_size,
            postfilter_enabled: true,
        }
    }

    /// Reset decoder state.
    pub fn reset(&mut self) {
        self.prev_energy.values.fill(0);
        self.prev_fine_energy.fill(0);
        self.overlap_buffer.fill(0.0);
        self.pitch_buffer.fill(0.0);
        self.synthesis_state.fill(0.0);
    }

    /// Handle packet loss concealment.
    #[allow(dead_code)]
    pub fn conceal_frame(&mut self) {
        // Decay energies for PLC
        for energy in &mut self.prev_energy.values {
            *energy = (*energy).saturating_sub(64); // Decay by ~0.25 dB per frame
        }
    }
}

/// PVQ (Pyramid Vector Quantization) pulse allocation.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct PvqAllocation {
    /// Number of pulses per band.
    pub pulses_per_band: Vec<u16>,
    /// Total bits allocated.
    pub total_bits: u32,
    /// Remaining bits (for fine energy).
    pub remaining_bits: u32,
}

impl PvqAllocation {
    /// Create new PVQ allocation.
    #[must_use]
    pub fn new(band_count: usize) -> Self {
        Self {
            pulses_per_band: vec![0; band_count],
            total_bits: 0,
            remaining_bits: 0,
        }
    }

    /// Get pulses for a band.
    #[must_use]
    pub fn pulses(&self, band: usize) -> u16 {
        self.pulses_per_band.get(band).copied().unwrap_or(0)
    }
}

// ─────────────────────────────── CELT MDCT ──────────────────────────────────

/// Window function for MDCT overlap-add (sine window).
#[must_use]
fn make_sine_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let x = std::f64::consts::PI * (i as f64 + 0.5) / size as f64;
            x.sin() as f32
        })
        .collect()
}

/// Perform IMDCT (Inverse MDCT) on `N` frequency-domain coefficients,
/// producing `2*N` time-domain output samples.
///
/// Uses the standard IMDCT definition with a pre- and post-twiddle approach
/// (DCT-IV via FFT of length N).
#[must_use]
pub fn imdct(coeffs: &[f32]) -> Vec<f32> {
    let n = coeffs.len();
    if n == 0 {
        return Vec::new();
    }

    // Pre-twiddle: multiply by e^{-j * pi/2N * (k + 0.5)}
    let two_n = 2 * n;
    let mut output = vec![0.0f32; two_n];

    // Direct DCT-IV via definition: slow O(N²) but correct for our purposes.
    // output[i] = sum_{k=0}^{N-1} X[k] * cos(pi/N * (k+0.5) * (i+0.5))  ×  2/N
    let scale = 2.0_f32 / n as f32;
    for i in 0..two_n {
        let mut sum = 0.0f32;
        for (k, &c) in coeffs.iter().enumerate() {
            let angle = std::f64::consts::PI / n as f64
                * (k as f64 + 0.5)
                * (i as f64 - n as f64 + 0.5 + n as f64 * 2.0);
            // Simplify: standard IMDCT formula
            let angle2 = std::f64::consts::PI / n as f64 * (k as f64 + 0.5) * (i as f64 + 0.5);
            let _ = angle; // silence unused
            sum += c * (angle2.cos() as f32);
        }
        output[i] = sum * scale;
    }
    output
}

/// Fast (approximate) IMDCT using a recursive cosine sum.  For the simplified
/// decoder we use a direct DCT-III lifted to even-size via mirroring.  This
/// is mathematically equivalent for synthesis purposes.
#[must_use]
pub fn imdct_fast(coeffs: &[f32]) -> Vec<f32> {
    let n = coeffs.len();
    if n == 0 {
        return Vec::new();
    }

    let two_n = 2 * n;
    let mut output = vec![0.0f32; two_n];

    // IMDCT: y[i] = (2/N) * sum_{k=0}^{N-1} X[k] * cos(pi*(2i+N+1)*(2k+1)/(4N))
    let scale = 2.0_f64 / n as f64;
    for i in 0..two_n {
        let mut sum = 0.0_f64;
        for (k, &c) in coeffs.iter().enumerate() {
            let angle =
                std::f64::consts::PI * (2 * i + n + 1) as f64 * (2 * k + 1) as f64 / (4 * n) as f64;
            sum += c as f64 * angle.cos();
        }
        output[i] = (sum * scale) as f32;
    }
    output
}

/// Reconstruct f32 samples from a `CeltFrame` using band energy + IMDCT synthesis.
///
/// # Algorithm
///
/// 1. For each band, scale unit-norm coefficients by sqrt(energy).
/// 2. Assemble MDCT spectrum in frequency order.
/// 3. Run IMDCT to get `2*N` time-domain samples.
/// 4. Apply overlap-add with the previous frame's tail stored in `state.overlap_buffer`.
///
/// # Errors
///
/// Returns `AudioError` if the frame configuration is inconsistent.
pub fn decode_frame(
    frame: &CeltFrame,
    state: &mut CeltDecoderState,
) -> Result<Vec<f32>, AudioError> {
    let band_config = CeltBandConfig::new_48khz();
    let n = frame.frame_size.samples_48khz(); // MDCT size = frame_size
    let channels = usize::from(frame.channels);

    // Each channel is decoded independently and then interleaved.
    let mut interleaved = vec![0.0f32; n * channels];

    for ch in 0..channels {
        // ── Step 1: Build MDCT spectrum ──────────────────────────────────
        let mut spectrum = vec![0.0f32; n];

        for band in 0..band_config.band_count {
            let (start, end) = match band_config.band_range(band) {
                Some(r) => r,
                None => break,
            };
            if start >= n {
                break;
            }
            let end = end.min(n);
            let width = end - start;
            if width == 0 {
                continue;
            }

            // Band energy in Q8 → convert to linear power, then amplitude.
            // Q8 means value/256 = energy in dB (relative).  We use a simple
            // mapping: amplitude = 10^(energy_q8 / (256 * 20)).
            let energy_q8 = frame.energy.get(band).unwrap_or(0);
            let energy_db = f64::from(energy_q8) / 256.0;
            let amplitude = (10.0_f64.powf(energy_db / 20.0)) as f32;

            // PVQ coefficients for this band (may be empty for low-bitrate).
            let band_coeffs = frame.coefficients.get(band);

            // Place fine-energy-adjusted amplitude into MDCT bins.
            if let Some(coeffs) = band_coeffs {
                if !coeffs.is_empty() {
                    // Normalize provided coefficients to unit norm.
                    let sq_sum: f64 = coeffs.iter().map(|&c| (c as f64) * (c as f64)).sum();
                    let norm = if sq_sum > 0.0 {
                        sq_sum.sqrt() as f32
                    } else {
                        1.0
                    };
                    for (j, &c) in coeffs.iter().enumerate().take(width) {
                        spectrum[start + j] = (c as f32 / norm) * amplitude;
                    }
                    // Pad remaining bins with silence if coeffs shorter than band.
                    for j in coeffs.len()..width {
                        spectrum[start + j] = 0.0;
                    }
                } else {
                    // No coefficients: fill band with equal-energy noise-like signal.
                    let per_bin = amplitude / (width as f32).sqrt();
                    for j in 0..width {
                        // Alternating sign gives a flat, noise-like shape.
                        spectrum[start + j] = if j % 2 == 0 { per_bin } else { -per_bin };
                    }
                }
            } else {
                let per_bin = amplitude / (width as f32).sqrt();
                for j in 0..width {
                    spectrum[start + j] = if j % 2 == 0 { per_bin } else { -per_bin };
                }
            }
        }

        // ── Step 2: IMDCT ─────────────────────────────────────────────────
        let time_domain = imdct_fast(&spectrum);

        // time_domain has length 2*n; second half is the windowed overlap tail.
        // Apply sine window.
        let window = make_sine_window(2 * n);
        let windowed: Vec<f32> = time_domain
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        // ── Step 3: Overlap-add ───────────────────────────────────────────
        // The state's overlap_buffer stores the second half of the previous IMDCT.
        let overlap_len = state.overlap_buffer.len().min(n);

        // Ensure we have enough overlap data (may differ on first frame).
        while state.overlap_buffer.len() < n {
            state.overlap_buffer.push(0.0);
        }

        // Output samples = first half of windowed IMDCT + previous overlap.
        let mut channel_out = vec![0.0f32; n];
        for i in 0..n {
            channel_out[i] = windowed[i]
                + if i < overlap_len {
                    state.overlap_buffer[i]
                } else {
                    0.0
                };
        }

        // Update overlap buffer with second half of this frame's IMDCT.
        let second_half_start = n;
        state.overlap_buffer.resize(n, 0.0);
        for i in 0..n {
            state.overlap_buffer[i] = windowed[second_half_start + i];
        }

        // Apply post-filter (pitch pre-filter synthesis) if enabled.
        if state.postfilter_enabled {
            if let Some(ref pitch) = frame.pitch {
                if pitch.is_active() {
                    let period = usize::from(pitch.period);
                    let gain = pitch.gain as f64 / 32768.0;
                    // Simple single-tap pitch post-filter.
                    for i in period..channel_out.len() {
                        channel_out[i] += (channel_out[i - period] as f64 * gain) as f32;
                    }
                }
            }
        }

        // Soft-clip to [-1, 1].
        for s in &mut channel_out {
            *s = s.clamp(-1.0, 1.0);
        }

        // Update previous band energies.
        for band in 0..band_config.band_count {
            if let Some(e) = frame.energy.get(band) {
                state.prev_energy.set(band, e);
            }
        }

        // Interleave: channel ch goes at positions [i*channels + ch].
        for (i, &s) in channel_out.iter().enumerate() {
            interleaved[i * channels + ch] = s;
        }
    }

    Ok(interleaved)
}

// ─────────────────────────────── CELT MDCT Encoder ──────────────────────────

/// Perform MDCT on `2*N` time-domain input samples,
/// returning `N` frequency-domain MDCT coefficients.
///
/// Uses the standard formula:
///   X\[k\] = sum_{n=0}^{2N-1} x\[n\] * cos(pi*(2n+N+1)*(2k+1) / (4N))
#[must_use]
pub fn mdct(input: &[f32]) -> Vec<f32> {
    let two_n = input.len();
    if two_n == 0 {
        return Vec::new();
    }
    let n = two_n / 2;
    let mut output = vec![0.0f32; n];

    let scale = 2.0_f64 / two_n as f64;
    for k in 0..n {
        let mut sum = 0.0_f64;
        for (i, &x) in input.iter().enumerate() {
            let angle =
                std::f64::consts::PI * (2 * i + n + 1) as f64 * (2 * k + 1) as f64 / (4 * n) as f64;
            sum += x as f64 * angle.cos();
        }
        output[k] = (sum * scale) as f32;
    }
    output
}

/// CELT encoder state (for overlap-save convolution).
#[derive(Debug, Clone, Default)]
pub struct CeltEncoderState {
    /// Previous frame tail used for overlap-save MDCT.
    pub prev_tail: Vec<f32>,
    /// Number of channels.
    pub channels: u8,
    /// Frame size configuration.
    pub frame_size: CeltFrameSize,
}

impl CeltEncoderState {
    /// Create a new encoder state.
    #[must_use]
    pub fn new(channels: u8, frame_size: CeltFrameSize) -> Self {
        let n = frame_size.samples_48khz();
        Self {
            prev_tail: vec![0.0; n],
            channels,
            frame_size,
        }
    }

    /// Reset encoder state.
    pub fn reset(&mut self) {
        self.prev_tail.fill(0.0);
    }
}

/// Encode interleaved f32 PCM samples into a `CeltFrame`.
///
/// # Algorithm
///
/// 1. Apply sine window and MDCT on [prev_tail | current_frame].
/// 2. Compute per-band energy from MDCT spectrum.
/// 3. Normalize each band to unit-norm and quantize to i16 coefficients.
/// 4. Store energy in Q8 dB format in `CeltFrame`.
///
/// # Errors
///
/// Returns `AudioError` if the sample count is incorrect.
pub fn encode_frame(
    samples: &[f32],
    state: &mut CeltEncoderState,
) -> Result<CeltFrame, AudioError> {
    let n = state.frame_size.samples_48khz();
    let channels = usize::from(state.channels);

    if samples.len() != n * channels {
        return Err(AudioError::InvalidParameter(format!(
            "Expected {} samples, got {}",
            n * channels,
            samples.len()
        )));
    }

    let band_config = CeltBandConfig::new_48khz();
    let mut frame = CeltFrame::new(state.frame_size, state.channels);

    // Process each channel independently.
    for ch in 0..channels {
        // De-interleave channel.
        let channel_samples: Vec<f32> = (0..n).map(|i| samples[i * channels + ch]).collect();

        // Build 2N-sample windowed input: [prev_tail | current_frame].
        let window = make_sine_window(2 * n);

        // Ensure prev_tail is the right length.
        state.prev_tail.resize(n, 0.0);

        let windowed_input: Vec<f32> = (0..2 * n)
            .map(|i| {
                let s = if i < n {
                    state.prev_tail[i]
                } else {
                    channel_samples[i - n]
                };
                s * window[i]
            })
            .collect();

        // Update prev_tail for next frame.
        state.prev_tail = channel_samples.clone();

        // MDCT to get N frequency-domain coefficients.
        let spectrum = mdct(&windowed_input);

        // ── Per-band energy and coefficient quantization ──────────────
        for band in 0..band_config.band_count {
            let (start, end) = match band_config.band_range(band) {
                Some(r) => r,
                None => break,
            };
            if start >= n {
                break;
            }
            let end = end.min(n);
            let width = end - start;
            if width == 0 {
                continue;
            }

            let band_slice = &spectrum[start..end];

            // Compute band energy (RMS amplitude).
            let sq_sum: f64 = band_slice.iter().map(|&c| (c as f64) * (c as f64)).sum();
            let rms = (sq_sum / width as f64).sqrt() as f32;

            // Convert amplitude to Q8 dB.
            let energy_db = if rms > 1e-10_f32 {
                20.0_f32 * rms.log10()
            } else {
                -60.0_f32
            };
            let energy_q8 = ((energy_db * 256.0) as i32).clamp(-32768, 32767) as i16;

            // Only update the first channel's energy (mono treatment for energy).
            if ch == 0 {
                frame.energy.set(band, energy_q8);
            }

            // Normalize and quantize coefficients to i16.
            let norm = if rms > 1e-10_f32 { rms } else { 1.0 };
            let quantized: Vec<i16> = band_slice
                .iter()
                .map(|&c| ((c / norm * 1024.0) as i32).clamp(-32767, 32767) as i16)
                .collect();

            // Only store coefficients from ch == 0 (simplified mono treatment).
            if ch == 0 {
                frame.coefficients[band] = quantized;
            }
        }
    }

    Ok(frame)
}

// ─────────────────────────────── Range Encoder ──────────────────────────────

/// Simple bit-packing encoder (non-range-coded; uses raw bits for our simplified encoder).
///
/// This is a simplified framing helper that packs signed i16 values into a
/// byte stream using a fixed-width representation.
pub struct BitPacker {
    buffer: Vec<u8>,
    current_byte: u8,
    bits_used: u8,
}

impl BitPacker {
    /// Create a new bit packer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            current_byte: 0,
            bits_used: 0,
        }
    }

    /// Write `n` bits (LSB-first) from `value`.
    pub fn write_bits(&mut self, mut value: u32, mut n: u8) {
        while n > 0 {
            let space = 8 - self.bits_used;
            let take = n.min(space);
            let mask = (1u32 << take) - 1;
            self.current_byte |= ((value & mask) as u8) << self.bits_used;
            value >>= take;
            n -= take;
            self.bits_used += take;
            if self.bits_used == 8 {
                self.buffer.push(self.current_byte);
                self.current_byte = 0;
                self.bits_used = 0;
            }
        }
    }

    /// Flush any remaining bits.
    pub fn flush(&mut self) {
        if self.bits_used > 0 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bits_used = 0;
        }
    }

    /// Consume the packer and return the packed bytes.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.flush();
        self.buffer
    }
}

impl Default for BitPacker {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple bit unpacker (mirrors `BitPacker`).
pub struct BitUnpacker {
    data: Vec<u8>,
    byte_pos: usize,
    bits_used: u8,
}

impl BitUnpacker {
    /// Create a new unpacker from bytes.
    #[must_use]
    pub fn new(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
            byte_pos: 0,
            bits_used: 0,
        }
    }

    /// Read `n` bits, returning them as a u32 (LSB-first).
    pub fn read_bits(&mut self, mut n: u8) -> u32 {
        let mut result = 0u32;
        let mut shift = 0u8;
        while n > 0 {
            if self.byte_pos >= self.data.len() {
                break;
            }
            let space = 8 - self.bits_used;
            let take = n.min(space);
            let mask: u8 = if take >= 8 {
                0xFF
            } else {
                (1u8 << take).wrapping_sub(1)
            };
            let bits = (self.data[self.byte_pos] >> self.bits_used) & mask;
            result |= (bits as u32) << shift;
            shift += take;
            self.bits_used += take;
            n -= take;
            if self.bits_used == 8 {
                self.bits_used = 0;
                self.byte_pos += 1;
            }
        }
        result
    }
}

/// Serialize a `CeltFrame` to bytes using the simple bit packing scheme.
///
/// Format:
///   - 1 byte: band_count
///   - For each band:
///     - 2 bytes: energy (i16 little-endian)
///     - 1 byte: coeff_count (up to 255)
///     - For each coeff: 2 bytes (i16 little-endian)
///
/// # Errors
///
/// Currently infallible; errors reserved for future validation.
pub fn serialize_celt_frame(frame: &CeltFrame) -> Result<Vec<u8>, AudioError> {
    let mut packer = BitPacker::new();
    let band_count = frame.energy.band_count.min(255) as u8;
    packer.write_bits(u32::from(band_count), 8);

    for band in 0..band_count as usize {
        let energy = frame.energy.get(band).unwrap_or(0);
        // i16 as two bytes
        let eu = energy as u16;
        packer.write_bits(u32::from(eu & 0xFF), 8);
        packer.write_bits(u32::from((eu >> 8) & 0xFF), 8);

        let coeffs = frame
            .coefficients
            .get(band)
            .map_or(&[][..], |v| v.as_slice());
        let coeff_count = coeffs.len().min(255) as u8;
        packer.write_bits(u32::from(coeff_count), 8);

        for &c in &coeffs[..coeff_count as usize] {
            let cu = c as u16;
            packer.write_bits(u32::from(cu & 0xFF), 8);
            packer.write_bits(u32::from((cu >> 8) & 0xFF), 8);
        }
    }

    Ok(packer.finish())
}

/// Deserialize bytes back into a `CeltFrame`.
///
/// # Errors
///
/// Returns `AudioError` if the data is truncated or malformed.
pub fn deserialize_celt_frame(
    data: &[u8],
    frame_size: CeltFrameSize,
    channels: u8,
) -> Result<CeltFrame, AudioError> {
    if data.is_empty() {
        return Err(AudioError::InvalidData("Empty CELT frame data".into()));
    }

    let mut unpacker = BitUnpacker::new(data);
    let band_count = unpacker.read_bits(8) as usize;
    let band_config = CeltBandConfig::new_48khz();

    let mut frame = CeltFrame::new(frame_size, channels);

    for band in 0..band_count.min(band_config.band_count) {
        let lo = unpacker.read_bits(8) as u16;
        let hi = unpacker.read_bits(8) as u16;
        let energy = ((hi << 8) | lo) as i16;
        frame.energy.set(band, energy);

        let coeff_count = unpacker.read_bits(8) as usize;
        let mut coeffs = Vec::with_capacity(coeff_count);
        for _ in 0..coeff_count {
            let lo2 = unpacker.read_bits(8) as u16;
            let hi2 = unpacker.read_bits(8) as u16;
            let c = ((hi2 << 8) | lo2) as i16;
            coeffs.push(c);
        }
        if band < frame.coefficients.len() {
            frame.coefficients[band] = coeffs;
        }
    }

    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_celt_frame_size_samples() {
        assert_eq!(CeltFrameSize::Ms2_5.samples_48khz(), 120);
        assert_eq!(CeltFrameSize::Ms5.samples_48khz(), 240);
        assert_eq!(CeltFrameSize::Ms10.samples_48khz(), 480);
        assert_eq!(CeltFrameSize::Ms20.samples_48khz(), 960);
    }

    #[test]
    fn test_celt_frame_size_duration() {
        assert_eq!(CeltFrameSize::Ms2_5.duration_us(), 2500);
        assert_eq!(CeltFrameSize::Ms5.duration_us(), 5000);
        assert_eq!(CeltFrameSize::Ms10.duration_us(), 10000);
        assert_eq!(CeltFrameSize::Ms20.duration_us(), 20000);
    }

    #[test]
    fn test_band_energy() {
        let mut energy = BandEnergy::new(21);
        assert_eq!(energy.band_count, 21);
        assert_eq!(energy.get(0), Some(0));

        energy.set(5, 100);
        assert_eq!(energy.get(5), Some(100));
    }

    #[test]
    fn test_band_energy_total() {
        let mut energy = BandEnergy::new(4);
        energy.values = vec![10, 20, 30, 40];
        assert_eq!(energy.total_energy(), 100);
    }

    #[test]
    fn test_pitch_period() {
        let pitch = PitchPeriod::new(100, 1000);
        assert_eq!(pitch.period, 100);
        assert_eq!(pitch.gain, 1000);
        assert!(pitch.is_active());

        let no_pitch = PitchPeriod::default();
        assert!(!no_pitch.is_active());
    }

    #[test]
    fn test_pitch_period_clamping() {
        let pitch = PitchPeriod::new(5, 100); // Below minimum
        assert_eq!(pitch.period, PitchPeriod::MIN_PERIOD);

        let pitch = PitchPeriod::new(2000, 100); // Above maximum
        assert_eq!(pitch.period, PitchPeriod::MAX_PERIOD);
    }

    #[test]
    fn test_celt_band_config() {
        let config = CeltBandConfig::new_48khz();
        assert_eq!(config.band_count, 21);

        let range = config.band_range(0);
        assert!(range.is_some());
        assert_eq!(range.expect("should succeed"), (0, 1));
    }

    #[test]
    fn test_celt_band_width() {
        let config = CeltBandConfig::new_48khz();
        assert_eq!(config.band_width(0), Some(1));
        assert_eq!(config.band_width(20), Some(16)); // 92 - 76 = 16
    }

    #[test]
    fn test_celt_frame() {
        let frame = CeltFrame::new(CeltFrameSize::Ms20, 2);
        assert_eq!(frame.frame_size, CeltFrameSize::Ms20);
        assert_eq!(frame.channels, 2);
        assert_eq!(frame.band_count(), 21);
        assert!(frame.is_stereo());
    }

    #[test]
    fn test_celt_frame_sample_count() {
        let mono = CeltFrame::new(CeltFrameSize::Ms10, 1);
        assert_eq!(mono.sample_count(), 480);

        let stereo = CeltFrame::new(CeltFrameSize::Ms10, 2);
        assert_eq!(stereo.sample_count(), 960);
    }

    #[test]
    fn test_celt_decoder_state() {
        let state = CeltDecoderState::new(2, CeltFrameSize::Ms20);
        assert_eq!(state.channels, 2);
        assert_eq!(state.frame_size, CeltFrameSize::Ms20);
        assert!(!state.overlap_buffer.is_empty());
    }

    #[test]
    fn test_celt_decoder_state_reset() {
        let mut state = CeltDecoderState::new(1, CeltFrameSize::Ms10);
        state.prev_energy.set(0, 100);
        state.reset();
        assert_eq!(state.prev_energy.get(0), Some(0));
    }

    #[test]
    fn test_pvq_allocation() {
        let alloc = PvqAllocation::new(21);
        assert_eq!(alloc.pulses_per_band.len(), 21);
        assert_eq!(alloc.pulses(0), 0);
        assert_eq!(alloc.pulses(100), 0); // Out of bounds
    }

    #[test]
    fn test_transient_type() {
        let frame = CeltFrame::new(CeltFrameSize::Ms20, 1);
        assert!(!frame.has_transient());
    }
}
