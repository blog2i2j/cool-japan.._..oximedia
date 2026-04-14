//! Audio resampling utilities.
//!
//! This module provides high-quality audio resampling using the `rubato` library.
//! It supports both synchronous (fixed ratio) and asynchronous (variable ratio) resampling.
//!
//! # Features
//!
//! - Multiple quality presets (Low, Medium, High, Best)
//! - Support for all sample formats (U8, S16, S32, F32, F64, and planar variants)
//! - Multi-channel audio support
//! - Streaming resampling with state management
//! - Zero-copy optimizations where possible
//!
//! # Examples
//!
//! ```rust,ignore
//! use oximedia_audio::{Resampler, ResamplerQuality};
//!
//! let mut resampler = Resampler::new(44100, 48000, 2, ResamplerQuality::High)?;
//! // Process audio frames...
//! ```

use crate::{AudioBuffer, AudioError, AudioFrame, AudioResult, ChannelLayout};
use audioadapter::Adapter;
use audioadapter_buffers::direct::SequentialSliceOfVecs;
use bytes::Bytes;
use oximedia_core::SampleFormat;
use rubato::{
    Async as RubatoAsync, Fft as RubatoFft, FixedAsync, FixedSync, Resampler as RubatoResampler,
    ResamplerConstructionError, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Resampling quality preset.
///
/// Higher quality settings use more CPU and memory but provide better
/// frequency response and lower aliasing.
///
/// # Preset aliases
///
/// Three named aliases follow the common `draft / good / best` convention:
///
/// | Alias | Maps to | Description |
/// |-------|---------|-------------|
/// | [`Draft`](ResamplerQuality::Draft) | `Low` | Fastest; real-time non-critical use |
/// | [`Good`](ResamplerQuality::Good)  | `High` | Sinc with cubic interpolation |
/// | [`Best`](ResamplerQuality::Best)  | `Best` | Sinc with large filter (highest quality) |
///
/// `Low`, `Medium`, `High`, and `Best` remain available for backward
/// compatibility and for when you need the intermediate `Medium` level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ResamplerQuality {
    /// Low quality — fastest, suitable for real-time where quality is not critical.
    /// Uses FFT with small chunk size.
    ///
    /// Equivalent to [`Draft`](ResamplerQuality::Draft).
    Low,

    /// Medium quality — balanced speed and quality.
    /// Uses FFT with medium chunk size.
    #[default]
    Medium,

    /// High quality — good quality for most applications.
    /// Uses Sinc interpolation with cubic interpolation.
    ///
    /// Equivalent to [`Good`](ResamplerQuality::Good).
    High,

    /// Best quality — highest quality, most CPU intensive.
    /// Uses Sinc interpolation with linear interpolation and a large filter.
    Best,

    // ── Named aliases ────────────────────────────────────────────────────────
    /// Draft quality — alias for [`Low`](ResamplerQuality::Low).
    ///
    /// Fastest setting, suitable for scrubbing or non-critical real-time use.
    Draft,

    /// Good quality — alias for [`High`](ResamplerQuality::High).
    ///
    /// Sinc interpolation with cubic interpolation; a solid all-round choice.
    Good,
}

impl ResamplerQuality {
    /// Resolve an alias to its canonical variant.
    ///
    /// `Draft` → `Low`, `Good` → `High`; all other variants resolve to
    /// themselves.
    #[must_use]
    pub const fn canonical(self) -> Self {
        match self {
            Self::Draft => Self::Low,
            Self::Good => Self::High,
            other => other,
        }
    }

    /// Get the chunk size for FFT-based resamplers.
    #[must_use]
    fn fft_chunk_size(&self) -> usize {
        match self.canonical() {
            Self::Low => 256,
            Self::Medium => 1024,
            Self::High | Self::Best => 2048,
            // aliases resolved above; unreachable
            _ => 1024,
        }
    }

    /// Get the sub-chunks value for FFT resamplers.
    #[must_use]
    fn fft_sub_chunks(&self) -> usize {
        match self.canonical() {
            Self::Low => 1,
            Self::Medium => 2,
            Self::High | Self::Best => 4,
            _ => 2,
        }
    }

    /// Get Sinc interpolation parameters.
    #[must_use]
    fn sinc_params(&self) -> SincInterpolationParameters {
        match self.canonical() {
            Self::Low | Self::Medium => SincInterpolationParameters {
                sinc_len: 64,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Cubic,
                oversampling_factor: 128,
                window: WindowFunction::BlackmanHarris2,
            },
            Self::High => SincInterpolationParameters {
                sinc_len: 128,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Cubic,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            },
            Self::Best => SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            },
            _ => SincInterpolationParameters {
                sinc_len: 128,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Cubic,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            },
        }
    }
}

/// Resampling strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResamplerStrategy {
    /// Passthrough (no resampling needed).
    Passthrough,
    /// Fixed input size, fixed output size.
    FixedInOut,
    /// Fixed input size, variable output size.
    FixedIn,
    /// Variable input size, fixed output size.
    FixedOut,
}

/// Audio resampler.
///
/// Provides high-quality audio resampling using polyphase filters (Sinc)
/// or FFT-based methods depending on quality settings.
pub struct Resampler {
    /// Source sample rate.
    source_rate: u32,
    /// Target sample rate.
    target_rate: u32,
    /// Channel count.
    channels: usize,
    /// Quality setting.
    quality: ResamplerQuality,
    /// Resampling strategy.
    strategy: ResamplerStrategy,
    /// Resampling ratio.
    ratio: f64,
    /// Internal resampler engine.
    engine: ResamplerEngine,
    /// Input buffer for partial frames.
    input_buffer: Vec<Vec<f32>>,
    /// Number of samples buffered.
    buffered_samples: usize,
}

/// Internal resampler engine variants.
enum ResamplerEngine {
    /// Passthrough (no processing).
    Passthrough,
    /// FFT-based resampler (synchronous, supports FixedIn/FixedOut/FixedInOut).
    Fft(Box<RubatoFft<f32>>),
    /// Sinc-based async resampler (supports FixedIn/FixedOut).
    SincAsync(Box<RubatoAsync<f32>>),
}

impl Resampler {
    /// Create a new resampler with specified quality.
    ///
    /// # Arguments
    ///
    /// * `source_rate` - Input sample rate in Hz
    /// * `target_rate` - Output sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `quality` - Quality preset
    ///
    /// # Errors
    ///
    /// Returns error if parameters are invalid or resampler construction fails.
    pub fn new(
        source_rate: u32,
        target_rate: u32,
        channels: usize,
        quality: ResamplerQuality,
    ) -> AudioResult<Self> {
        Self::with_max_buffering(source_rate, target_rate, channels, quality, 8192)
    }

    /// Create a new resampler with specified maximum input buffer size.
    ///
    /// # Arguments
    ///
    /// * `source_rate` - Input sample rate in Hz
    /// * `target_rate` - Output sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `quality` - Quality preset
    /// * `max_input_frames` - Maximum input frames to buffer
    ///
    /// # Errors
    ///
    /// Returns error if parameters are invalid or resampler construction fails.
    #[allow(clippy::too_many_arguments)]
    pub fn with_max_buffering(
        source_rate: u32,
        target_rate: u32,
        channels: usize,
        quality: ResamplerQuality,
        max_input_frames: usize,
    ) -> AudioResult<Self> {
        if source_rate == 0 || target_rate == 0 {
            return Err(AudioError::InvalidParameter(
                "Sample rate must be non-zero".into(),
            ));
        }
        if channels == 0 || channels > 32 {
            return Err(AudioError::InvalidParameter(
                "Channel count must be between 1 and 32".into(),
            ));
        }

        let ratio = f64::from(target_rate) / f64::from(source_rate);

        // Determine strategy and create engine
        let (strategy, engine) = if source_rate == target_rate {
            (ResamplerStrategy::Passthrough, ResamplerEngine::Passthrough)
        } else {
            Self::create_engine(
                source_rate,
                target_rate,
                channels,
                quality,
                max_input_frames,
            )?
        };

        let input_buffer = vec![Vec::new(); channels];

        Ok(Self {
            source_rate,
            target_rate,
            channels,
            quality,
            strategy,
            ratio,
            engine,
            input_buffer,
            buffered_samples: 0,
        })
    }

    /// Create resampler engine based on quality and rate relationship.
    fn create_engine(
        source_rate: u32,
        target_rate: u32,
        channels: usize,
        quality: ResamplerQuality,
        _max_input_frames: usize,
    ) -> AudioResult<(ResamplerStrategy, ResamplerEngine)> {
        let chunk_size = quality.fft_chunk_size();

        // Calculate GCD for rational resampling
        let gcd = gcd(source_rate, target_rate);
        let ratio_num = target_rate / gcd;
        let ratio_den = source_rate / gcd;

        // Use high-quality Sinc for High, Good, and Best quality
        if matches!(
            quality.canonical(),
            ResamplerQuality::High | ResamplerQuality::Best
        ) {
            let params = quality.sinc_params();

            // Use Async resampler with sinc interpolation, fixed input
            let engine = RubatoAsync::<f32>::new_sinc(
                f64::from(target_rate) / f64::from(source_rate),
                2.0,
                &params,
                chunk_size,
                channels,
                FixedAsync::Input,
            )
            .map_err(map_rubato_error)?;
            return Ok((
                ResamplerStrategy::FixedIn,
                ResamplerEngine::SincAsync(Box::new(engine)),
            ));
        }

        // Use FFT-based resampling for Low and Medium quality
        let sub_chunks = quality.fft_sub_chunks();

        // Try FixedInOut (Both) if possible with small rational ratios
        if ratio_num < 100 && ratio_den < 100 {
            match RubatoFft::<f32>::new(
                source_rate as usize,
                target_rate as usize,
                chunk_size,
                sub_chunks,
                channels,
                FixedSync::Both,
            ) {
                Ok(engine) => {
                    return Ok((
                        ResamplerStrategy::FixedInOut,
                        ResamplerEngine::Fft(Box::new(engine)),
                    ));
                }
                Err(_) => {
                    // Fall through to FixedIn
                }
            }
        }

        // Fall back to FixedIn
        let engine = RubatoFft::<f32>::new(
            source_rate as usize,
            target_rate as usize,
            chunk_size,
            sub_chunks,
            channels,
            FixedSync::Input,
        )
        .map_err(map_rubato_error)?;
        Ok((
            ResamplerStrategy::FixedIn,
            ResamplerEngine::Fft(Box::new(engine)),
        ))
    }

    /// Resample an audio frame.
    ///
    /// This method handles streaming resampling with internal buffering.
    /// It may return fewer or more samples than the input depending on
    /// the resampling ratio.
    ///
    /// # Errors
    ///
    /// Returns error if resampling fails or format conversion fails.
    pub fn resample(&mut self, input: &AudioFrame) -> AudioResult<AudioFrame> {
        // Fast path for passthrough
        if self.strategy == ResamplerStrategy::Passthrough {
            return Ok(input.clone());
        }

        // Verify channel count matches
        if input.channels.count() != self.channels {
            return Err(AudioError::InvalidParameter(format!(
                "Channel count mismatch: expected {}, got {}",
                self.channels,
                input.channels.count()
            )));
        }

        // Convert input to f32 planar format for rubato
        let input_planar = self.convert_to_f32_planar(input)?;

        // Process through resampler
        let output_planar = self.process_samples(&input_planar)?;

        // Convert back to original format
        self.convert_from_f32_planar(&output_planar, input.format, &input.channels)
    }

    /// Process samples through the resampler engine.
    fn process_samples(&mut self, input: &[Vec<f32>]) -> AudioResult<Vec<Vec<f32>>> {
        match &mut self.engine {
            ResamplerEngine::Passthrough => Ok(input.to_vec()),
            ResamplerEngine::Fft(engine) => Self::process_fixed_in_engine(
                engine.as_mut(),
                input,
                &mut self.input_buffer,
                &mut self.buffered_samples,
                self.channels,
                "FFT",
            ),
            ResamplerEngine::SincAsync(engine) => Self::process_fixed_in_engine(
                engine.as_mut(),
                input,
                &mut self.input_buffer,
                &mut self.buffered_samples,
                self.channels,
                "Sinc",
            ),
        }
    }

    /// Process samples through a fixed-input resampler engine.
    ///
    /// Converts planar `Vec<Vec<f32>>` input to the `audioadapter` format,
    /// processes chunks through the engine, and converts the interleaved
    /// output back to planar `Vec<Vec<f32>>`.
    fn process_fixed_in_engine(
        engine: &mut dyn RubatoResampler<f32>,
        input: &[Vec<f32>],
        input_buffer: &mut [Vec<f32>],
        buffered_samples: &mut usize,
        channels: usize,
        label: &str,
    ) -> AudioResult<Vec<Vec<f32>>> {
        let mut output: Vec<Vec<f32>> = Vec::new();
        let mut remaining = input.to_vec();

        while !remaining.is_empty() && !remaining[0].is_empty() {
            let chunk_size = engine.input_frames_next();
            if remaining[0].len() >= chunk_size {
                let chunk: Vec<Vec<f32>> = remaining
                    .iter()
                    .map(|ch| ch[..chunk_size].to_vec())
                    .collect();

                let input_adapter = SequentialSliceOfVecs::new(&chunk, channels, chunk_size)
                    .map_err(|e| {
                        AudioError::Internal(format!(
                            "{label} resampling: input adapter creation failed: {e}"
                        ))
                    })?;

                let result = engine
                    .process(&input_adapter, 0, None)
                    .map_err(|e| AudioError::Internal(format!("{label} resampling failed: {e}")))?;

                // Convert InterleavedOwned output to Vec<Vec<f32>>
                let result_planar = interleaved_owned_to_planar(&result, channels);

                if output.is_empty() {
                    output = result_planar;
                } else {
                    for (out_ch, res_ch) in output.iter_mut().zip(result_planar.iter()) {
                        out_ch.extend_from_slice(res_ch);
                    }
                }

                remaining = remaining
                    .iter()
                    .map(|ch| ch[chunk_size..].to_vec())
                    .collect::<Vec<_>>();
            } else {
                // Buffer remaining samples for next call
                for (buf_ch, rem_ch) in input_buffer.iter_mut().zip(remaining.iter()) {
                    buf_ch.extend_from_slice(rem_ch);
                }
                *buffered_samples = remaining[0].len();
                break;
            }
        }

        Ok(output)
    }

    /// Convert audio frame to f32 planar format.
    fn convert_to_f32_planar(&self, frame: &AudioFrame) -> AudioResult<Vec<Vec<f32>>> {
        let sample_count = frame.sample_count();
        let mut planar = vec![vec![0.0f32; sample_count]; self.channels];

        match &frame.samples {
            AudioBuffer::Interleaved(data) => {
                self.deinterleave_to_f32(data, frame.format, &mut planar)?;
            }
            AudioBuffer::Planar(planes) => {
                self.planes_to_f32(planes, frame.format, &mut planar)?;
            }
        }

        Ok(planar)
    }

    /// Deinterleave and convert interleaved samples to f32.
    #[allow(clippy::cast_precision_loss)]
    fn deinterleave_to_f32(
        &self,
        data: &[u8],
        format: SampleFormat,
        output: &mut [Vec<f32>],
    ) -> AudioResult<()> {
        let sample_count = output[0].len();
        let bytes_per_sample = format.bytes_per_sample();

        for sample_idx in 0..sample_count {
            for ch in 0..self.channels {
                let offset = (sample_idx * self.channels + ch) * bytes_per_sample;
                let value = self.read_sample(data, offset, format)?;
                output[ch][sample_idx] = value;
            }
        }

        Ok(())
    }

    /// Convert planar samples to f32.
    fn planes_to_f32(
        &self,
        planes: &[Bytes],
        format: SampleFormat,
        output: &mut [Vec<f32>],
    ) -> AudioResult<()> {
        let sample_count = output[0].len();
        let bytes_per_sample = format.bytes_per_sample();

        for ch in 0..self.channels {
            if ch >= planes.len() {
                return Err(AudioError::InvalidData(
                    "Insufficient planes for channel count".into(),
                ));
            }
            for sample_idx in 0..sample_count {
                let offset = sample_idx * bytes_per_sample;
                let value = self.read_sample(&planes[ch], offset, format)?;
                output[ch][sample_idx] = value;
            }
        }

        Ok(())
    }

    /// Read a single sample and convert to f32.
    #[allow(clippy::cast_precision_loss)]
    fn read_sample(&self, data: &[u8], offset: usize, format: SampleFormat) -> AudioResult<f32> {
        let bytes_per_sample = format.bytes_per_sample();
        if offset + bytes_per_sample > data.len() {
            return Err(AudioError::InvalidData(
                "Sample offset out of bounds".into(),
            ));
        }

        let value = match format {
            SampleFormat::U8 => (f32::from(data[offset]) - 128.0) / 128.0,
            SampleFormat::S16 | SampleFormat::S16p => {
                let bytes = [data[offset], data[offset + 1]];
                let sample = i16::from_le_bytes(bytes);
                sample as f32 / 32768.0
            }
            SampleFormat::S32 | SampleFormat::S32p => {
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ];
                let sample = i32::from_le_bytes(bytes);
                sample as f32 / 2_147_483_648.0
            }
            SampleFormat::F32 | SampleFormat::F32p => {
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ];
                f32::from_le_bytes(bytes)
            }
            SampleFormat::F64 | SampleFormat::F64p => {
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ];
                #[allow(clippy::cast_possible_truncation)]
                let result = f64::from_le_bytes(bytes) as f32;
                result
            }
            _ => {
                return Err(AudioError::UnsupportedFormat(format!(
                    "Unsupported sample format: {format}"
                )))
            }
        };

        Ok(value)
    }

    /// Convert f32 planar samples back to audio frame format.
    fn convert_from_f32_planar(
        &self,
        planar: &[Vec<f32>],
        format: SampleFormat,
        channels: &ChannelLayout,
    ) -> AudioResult<AudioFrame> {
        if planar.is_empty() {
            let mut frame = AudioFrame::new(format, self.target_rate, channels.clone());
            frame.samples = AudioBuffer::Interleaved(Bytes::new());
            return Ok(frame);
        }

        let mut frame = AudioFrame::new(format, self.target_rate, channels.clone());

        if format.is_planar() {
            frame.samples = self.f32_to_planar_bytes(planar, format)?;
        } else {
            frame.samples = self.f32_to_interleaved_bytes(planar, format)?;
        }

        Ok(frame)
    }

    /// Convert f32 planar to interleaved bytes.
    fn f32_to_interleaved_bytes(
        &self,
        planar: &[Vec<f32>],
        format: SampleFormat,
    ) -> AudioResult<AudioBuffer> {
        let sample_count = planar[0].len();
        let bytes_per_sample = format.bytes_per_sample();
        let total_bytes = sample_count * self.channels * bytes_per_sample;
        let mut data = vec![0u8; total_bytes];

        for sample_idx in 0..sample_count {
            for ch in 0..self.channels {
                let offset = (sample_idx * self.channels + ch) * bytes_per_sample;
                let value = planar[ch][sample_idx];
                self.write_sample(&mut data, offset, value, format)?;
            }
        }

        Ok(AudioBuffer::Interleaved(Bytes::from(data)))
    }

    /// Convert f32 planar to planar bytes.
    fn f32_to_planar_bytes(
        &self,
        planar: &[Vec<f32>],
        format: SampleFormat,
    ) -> AudioResult<AudioBuffer> {
        let sample_count = planar[0].len();
        let bytes_per_sample = format.bytes_per_sample();
        let plane_size = sample_count * bytes_per_sample;
        let mut planes = Vec::with_capacity(self.channels);

        for ch in 0..self.channels {
            let mut plane_data = vec![0u8; plane_size];
            for sample_idx in 0..sample_count {
                let offset = sample_idx * bytes_per_sample;
                let value = planar[ch][sample_idx];
                self.write_sample(&mut plane_data, offset, value, format)?;
            }
            planes.push(Bytes::from(plane_data));
        }

        Ok(AudioBuffer::Planar(planes))
    }

    /// Write a single f32 sample in the target format.
    #[allow(clippy::cast_possible_truncation)]
    fn write_sample(
        &self,
        data: &mut [u8],
        offset: usize,
        value: f32,
        format: SampleFormat,
    ) -> AudioResult<()> {
        let bytes_per_sample = format.bytes_per_sample();
        if offset + bytes_per_sample > data.len() {
            return Err(AudioError::InvalidData(
                "Sample offset out of bounds".into(),
            ));
        }

        match format {
            SampleFormat::U8 => {
                let sample = ((value.clamp(-1.0, 1.0) * 128.0) + 128.0) as u8;
                data[offset] = sample;
            }
            SampleFormat::S16 | SampleFormat::S16p => {
                let sample = (value.clamp(-1.0, 1.0) * 32767.0) as i16;
                let bytes = sample.to_le_bytes();
                data[offset..offset + 2].copy_from_slice(&bytes);
            }
            SampleFormat::S32 | SampleFormat::S32p => {
                let sample = (value.clamp(-1.0, 1.0) * 2_147_483_647.0) as i32;
                let bytes = sample.to_le_bytes();
                data[offset..offset + 4].copy_from_slice(&bytes);
            }
            SampleFormat::F32 | SampleFormat::F32p => {
                let bytes = value.to_le_bytes();
                data[offset..offset + 4].copy_from_slice(&bytes);
            }
            SampleFormat::F64 | SampleFormat::F64p => {
                let bytes = f64::from(value).to_le_bytes();
                data[offset..offset + 8].copy_from_slice(&bytes);
            }
            _ => {
                return Err(AudioError::UnsupportedFormat(format!(
                    "Unsupported sample format: {format}"
                )))
            }
        }

        Ok(())
    }

    /// Check if resampling is needed.
    #[must_use]
    pub fn is_passthrough(&self) -> bool {
        self.source_rate == self.target_rate
    }

    /// Get the resampling ratio.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// Get output sample count for given input sample count.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn output_sample_count(&self, input_samples: usize) -> usize {
        ((input_samples as f64) * self.ratio).ceil() as usize
    }

    /// Reset the resampler state.
    ///
    /// Clears internal buffers and resets the resampler to its initial state.
    pub fn reset(&mut self) {
        for ch in &mut self.input_buffer {
            ch.clear();
        }
        self.buffered_samples = 0;

        // Reset engine if possible
        match &mut self.engine {
            ResamplerEngine::Fft(engine) => {
                engine.reset();
            }
            ResamplerEngine::SincAsync(engine) => {
                engine.reset();
            }
            ResamplerEngine::Passthrough => {}
        }
    }

    /// Get the source sample rate.
    #[must_use]
    pub const fn source_rate(&self) -> u32 {
        self.source_rate
    }

    /// Get the target sample rate.
    #[must_use]
    pub const fn target_rate(&self) -> u32 {
        self.target_rate
    }

    /// Get the number of channels.
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Get the quality setting.
    #[must_use]
    pub const fn quality(&self) -> ResamplerQuality {
        self.quality
    }
}

/// Calculate greatest common divisor.
#[must_use]
const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Map rubato error to audio error.
fn map_rubato_error(err: ResamplerConstructionError) -> AudioError {
    AudioError::InvalidParameter(format!("Resampler construction failed: {err}"))
}

/// Convert an `InterleavedOwned<f32>` (from rubato's `process()`) to planar `Vec<Vec<f32>>`.
fn interleaved_owned_to_planar(
    interleaved: &dyn Adapter<'_, f32>,
    channels: usize,
) -> Vec<Vec<f32>> {
    let frames = interleaved.frames();
    let mut planar = vec![vec![0.0f32; frames]; channels];
    for ch in 0..channels {
        interleaved.copy_from_channel_to_slice(ch, 0, &mut planar[ch]);
    }
    planar
}

// ── Tests for ResamplerQuality presets (draft/good/best) ─────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioBuffer, ChannelLayout};
    use bytes::Bytes;
    use oximedia_core::SampleFormat;

    fn mono_f32_frame(n: usize, value: f32, sample_rate: u32) -> AudioFrame {
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut frame = AudioFrame::new(SampleFormat::F32, sample_rate, ChannelLayout::Mono);
        frame.samples = AudioBuffer::Interleaved(Bytes::from(bytes));
        frame
    }

    #[test]
    fn test_draft_is_alias_for_low() {
        assert_eq!(ResamplerQuality::Draft.canonical(), ResamplerQuality::Low);
    }

    #[test]
    fn test_good_is_alias_for_high() {
        assert_eq!(ResamplerQuality::Good.canonical(), ResamplerQuality::High);
    }

    #[test]
    fn test_best_canonical_is_best() {
        assert_eq!(ResamplerQuality::Best.canonical(), ResamplerQuality::Best);
    }

    #[test]
    fn test_medium_canonical_is_medium() {
        assert_eq!(
            ResamplerQuality::Medium.canonical(),
            ResamplerQuality::Medium
        );
    }

    #[test]
    fn test_low_fft_chunk_size_smallest() {
        let low = ResamplerQuality::Low.fft_chunk_size();
        let medium = ResamplerQuality::Medium.fft_chunk_size();
        let high = ResamplerQuality::High.fft_chunk_size();
        assert!(low <= medium, "Low ({low}) should be <= Medium ({medium})");
        assert!(
            medium <= high,
            "Medium ({medium}) should be <= High ({high})"
        );
    }

    #[test]
    fn test_best_sinc_len_ge_low() {
        let low_params = ResamplerQuality::Low.sinc_params();
        let best_params = ResamplerQuality::Best.sinc_params();
        assert!(
            best_params.sinc_len >= low_params.sinc_len,
            "Best sinc_len ({}) should be >= Low ({})",
            best_params.sinc_len,
            low_params.sinc_len
        );
    }

    #[test]
    fn test_draft_fft_chunk_same_as_low() {
        assert_eq!(
            ResamplerQuality::Draft.fft_chunk_size(),
            ResamplerQuality::Low.fft_chunk_size(),
        );
    }

    #[test]
    fn test_good_fft_chunk_same_as_high() {
        assert_eq!(
            ResamplerQuality::Good.fft_chunk_size(),
            ResamplerQuality::High.fft_chunk_size(),
        );
    }

    #[test]
    fn test_resampler_passthrough_same_rate() {
        let r = Resampler::new(48_000, 48_000, 1, ResamplerQuality::Medium)
            .expect("passthrough resampler");
        assert!(r.is_passthrough());
        assert_eq!(r.ratio(), 1.0);
    }

    #[test]
    fn test_resampler_upsample_ratio() {
        let r =
            Resampler::new(44_100, 48_000, 1, ResamplerQuality::Low).expect("upsample resampler");
        assert!(!r.is_passthrough());
        let expected = 48_000.0 / 44_100.0;
        assert!((r.ratio() - expected).abs() < 1e-6, "ratio mismatch");
    }

    #[test]
    fn test_resampler_construction_draft_quality() {
        let r = Resampler::new(44_100, 48_000, 2, ResamplerQuality::Draft);
        assert!(
            r.is_ok(),
            "Draft resampler construction failed: {:?}",
            r.err()
        );
    }

    #[test]
    fn test_resampler_construction_good_quality() {
        let r = Resampler::new(44_100, 48_000, 1, ResamplerQuality::Good);
        assert!(
            r.is_ok(),
            "Good resampler construction failed: {:?}",
            r.err()
        );
    }

    #[test]
    fn test_resampler_construction_best_quality() {
        let r = Resampler::new(44_100, 48_000, 1, ResamplerQuality::Best);
        assert!(
            r.is_ok(),
            "Best resampler construction failed: {:?}",
            r.err()
        );
    }

    #[test]
    fn test_resampler_passthrough_returns_correct_sample_rate() {
        let mut r =
            Resampler::new(48_000, 48_000, 1, ResamplerQuality::Medium).expect("passthrough");
        let frame = mono_f32_frame(512, 0.5, 48_000);
        let out = r.resample(&frame).expect("passthrough resample");
        assert_eq!(out.sample_rate, 48_000);
        assert_eq!(out.format, SampleFormat::F32);
    }

    #[test]
    fn test_resampler_source_target_rate_accessors() {
        let r = Resampler::new(44_100, 48_000, 1, ResamplerQuality::Low).expect("resampler");
        assert_eq!(r.source_rate(), 44_100);
        assert_eq!(r.target_rate(), 48_000);
        assert_eq!(r.channels(), 1);
    }

    #[test]
    fn test_resampler_output_sample_count_upsample() {
        let r = Resampler::new(44_100, 48_000, 1, ResamplerQuality::Low).expect("resampler");
        let out_count = r.output_sample_count(441);
        assert!(out_count >= 480, "expected >= 480, got {out_count}");
        assert!(out_count <= 483, "expected <= 483, got {out_count}");
    }

    #[test]
    fn test_resampler_error_on_zero_sample_rate() {
        let r = Resampler::new(0, 48_000, 1, ResamplerQuality::Low);
        assert!(r.is_err(), "zero source rate should fail");
    }

    #[test]
    fn test_resampler_error_on_zero_channels() {
        let r = Resampler::new(44_100, 48_000, 0, ResamplerQuality::Low);
        assert!(r.is_err(), "zero channels should fail");
    }

    #[test]
    fn test_resampler_quality_accessor() {
        let r = Resampler::new(44_100, 48_000, 1, ResamplerQuality::Best).expect("resampler");
        assert_eq!(r.quality(), ResamplerQuality::Best);
    }

    #[test]
    fn test_low_sub_chunks_le_high() {
        let low_sc = ResamplerQuality::Low.fft_sub_chunks();
        let high_sc = ResamplerQuality::High.fft_sub_chunks();
        assert!(
            low_sc <= high_sc,
            "Low sub_chunks ({low_sc}) <= High ({high_sc})"
        );
    }
}

/// Common sample rate constants.
pub mod sample_rates {
    /// 8 kHz - Telephone quality.
    pub const RATE_8000: u32 = 8000;
    /// 11.025 kHz - Low quality audio.
    pub const RATE_11025: u32 = 11025;
    /// 16 kHz - Wideband speech.
    pub const RATE_16000: u32 = 16000;
    /// 22.05 kHz - Quarter of CD quality.
    pub const RATE_22050: u32 = 22050;
    /// 32 kHz - Digital radio.
    pub const RATE_32000: u32 = 32000;
    /// 44.1 kHz - CD quality.
    pub const RATE_44100: u32 = 44100;
    /// 48 kHz - Professional audio, DVD.
    pub const RATE_48000: u32 = 48000;
    /// 88.2 kHz - High-resolution audio (2x CD).
    pub const RATE_88200: u32 = 88200;
    /// 96 kHz - High-resolution audio, Blu-ray.
    pub const RATE_96000: u32 = 96000;
    /// 176.4 kHz - Ultra high-resolution (4x CD).
    pub const RATE_176400: u32 = 176400;
    /// 192 kHz - Ultra high-resolution.
    pub const RATE_192000: u32 = 192000;
}
