//! Python bindings for `oximedia-mixer` audio mixing.
//!
//! Provides `PyMixerConfig`, `PyChannelStrip`, `PyAudioMixer`, and
//! standalone convenience functions for audio mixing from Python.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert dB value to linear gain.
fn db_to_linear(db: f64) -> f64 {
    if db <= -120.0 {
        0.0
    } else {
        10.0_f64.powf(db / 20.0)
    }
}

/// Convert linear gain to dB.
#[cfg(test)]
fn linear_to_db(linear: f64) -> f64 {
    if linear <= 0.0 {
        -120.0
    } else {
        20.0 * linear.log10()
    }
}

// ---------------------------------------------------------------------------
// PyEqBand
// ---------------------------------------------------------------------------

/// A single equalizer band.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyEqBand {
    /// Center frequency in Hz.
    #[pyo3(get)]
    pub frequency: f64,
    /// Gain in dB.
    #[pyo3(get)]
    pub gain: f64,
    /// Q factor (bandwidth).
    #[pyo3(get)]
    pub q: f64,
}

#[pymethods]
impl PyEqBand {
    /// Create a new EQ band.
    #[new]
    #[pyo3(signature = (frequency, gain, q=None))]
    fn new(frequency: f64, gain: f64, q: Option<f64>) -> PyResult<Self> {
        if frequency <= 0.0 || frequency > 22050.0 {
            return Err(PyValueError::new_err(format!(
                "Frequency must be between 0 and 22050 Hz, got {frequency}"
            )));
        }
        if !(-24.0..=24.0).contains(&gain) {
            return Err(PyValueError::new_err(format!(
                "Gain must be between -24 and +24 dB, got {gain}"
            )));
        }
        let q_val = q.unwrap_or(1.0);
        if q_val <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "Q must be positive, got {q_val}"
            )));
        }
        Ok(Self {
            frequency,
            gain,
            q: q_val,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyEqBand(freq={:.0}Hz, gain={:+.1}dB, Q={:.2})",
            self.frequency, self.gain, self.q
        )
    }
}

// ---------------------------------------------------------------------------
// PySendConfig
// ---------------------------------------------------------------------------

/// A send routing configuration.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PySendConfig {
    /// Target bus name.
    #[pyo3(get)]
    pub bus_name: String,
    /// Send level (0.0 to 1.0).
    #[pyo3(get)]
    pub level: f64,
    /// Pre-fader send.
    #[pyo3(get)]
    pub pre_fader: bool,
}

#[pymethods]
impl PySendConfig {
    /// Create a new send config.
    #[new]
    #[pyo3(signature = (bus_name, level=None, pre_fader=None))]
    fn new(bus_name: String, level: Option<f64>, pre_fader: Option<bool>) -> PyResult<Self> {
        let lvl = level.unwrap_or(1.0);
        if !(0.0..=1.0).contains(&lvl) {
            return Err(PyValueError::new_err(format!(
                "Send level must be between 0.0 and 1.0, got {lvl}"
            )));
        }
        Ok(Self {
            bus_name,
            level: lvl,
            pre_fader: pre_fader.unwrap_or(false),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PySendConfig(bus='{}', level={:.2}, pre={})",
            self.bus_name, self.level, self.pre_fader
        )
    }
}

// ---------------------------------------------------------------------------
// PyMixerConfig
// ---------------------------------------------------------------------------

/// Configuration for the audio mixer.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyMixerConfig {
    /// Sample rate in Hz.
    #[pyo3(get)]
    pub sample_rate: u32,
    /// Number of output channels.
    #[pyo3(get)]
    pub channels: u32,
    /// Buffer size in samples.
    #[pyo3(get)]
    pub buffer_size: usize,
    /// Output format name (mono/stereo/5.1/7.1).
    #[pyo3(get)]
    pub output_format: String,
}

#[pymethods]
impl PyMixerConfig {
    /// Create a new mixer configuration.
    #[new]
    #[pyo3(signature = (sample_rate=None, channels=None))]
    fn new(sample_rate: Option<u32>, channels: Option<u32>) -> PyResult<Self> {
        let sr = sample_rate.unwrap_or(48000);
        let ch = channels.unwrap_or(2);
        if sr == 0 || sr > 384000 {
            return Err(PyValueError::new_err(format!(
                "Sample rate must be 1-384000 Hz, got {sr}"
            )));
        }
        let format_name = match ch {
            1 => "mono",
            2 => "stereo",
            6 => "5.1",
            8 => "7.1",
            _ => "custom",
        };
        Ok(Self {
            sample_rate: sr,
            channels: ch,
            buffer_size: 512,
            output_format: format_name.to_string(),
        })
    }

    /// Set buffer size.
    fn with_buffer_size(&mut self, size: usize) -> PyResult<()> {
        if size == 0 {
            return Err(PyValueError::new_err("Buffer size must be > 0"));
        }
        self.buffer_size = size;
        Ok(())
    }

    /// Create a stereo mixer config.
    #[staticmethod]
    fn stereo() -> PyResult<Self> {
        Self::new(Some(48000), Some(2))
    }

    /// Create a 5.1 surround mixer config.
    #[staticmethod]
    fn surround_5_1() -> PyResult<Self> {
        Self::new(Some(48000), Some(6))
    }

    /// Create a 7.1 surround mixer config.
    #[staticmethod]
    fn surround_7_1() -> PyResult<Self> {
        Self::new(Some(48000), Some(8))
    }

    fn __repr__(&self) -> String {
        format!(
            "PyMixerConfig(sr={}, ch={}, buf={}, fmt='{}')",
            self.sample_rate, self.channels, self.buffer_size, self.output_format
        )
    }
}

// ---------------------------------------------------------------------------
// PyChannelStrip
// ---------------------------------------------------------------------------

/// A mixer channel strip with volume, pan, EQ, and sends.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyChannelStrip {
    /// Channel name.
    #[pyo3(get)]
    pub name: String,
    /// Volume in dB.
    #[pyo3(get)]
    pub volume_db: f64,
    /// Pan position (-1.0 to 1.0).
    #[pyo3(get)]
    pub pan: f64,
    /// Muted flag.
    #[pyo3(get)]
    pub mute: bool,
    /// Soloed flag.
    #[pyo3(get)]
    pub solo: bool,
    /// EQ bands.
    eq_bands: Vec<PyEqBand>,
    /// Send configurations.
    sends: Vec<PySendConfig>,
}

#[pymethods]
impl PyChannelStrip {
    /// Create a new channel strip.
    #[new]
    #[pyo3(signature = (name))]
    fn new(name: String) -> Self {
        Self {
            name,
            volume_db: 0.0,
            pan: 0.0,
            mute: false,
            solo: false,
            eq_bands: Vec::new(),
            sends: Vec::new(),
        }
    }

    /// Set volume in dB.
    fn set_volume(&mut self, db: f64) -> PyResult<()> {
        if !(-120.0..=24.0).contains(&db) {
            return Err(PyValueError::new_err(format!(
                "Volume must be between -120 and +24 dB, got {db}"
            )));
        }
        self.volume_db = db;
        Ok(())
    }

    /// Set pan position (-1.0 left, 0.0 center, 1.0 right).
    fn set_pan(&mut self, pos: f64) -> PyResult<()> {
        if !(-1.0..=1.0).contains(&pos) {
            return Err(PyValueError::new_err(format!(
                "Pan must be between -1.0 and 1.0, got {pos}"
            )));
        }
        self.pan = pos;
        Ok(())
    }

    /// Set mute state.
    fn set_mute(&mut self, muted: bool) {
        self.mute = muted;
    }

    /// Set solo state.
    fn set_solo(&mut self, soloed: bool) {
        self.solo = soloed;
    }

    /// Add an EQ band.
    fn add_eq_band(&mut self, freq: f64, gain: f64, q: f64) -> PyResult<()> {
        let band = PyEqBand::new(freq, gain, Some(q))?;
        self.eq_bands.push(band);
        Ok(())
    }

    /// Add a send to a bus.
    fn add_send(&mut self, bus: String, level: f64) -> PyResult<()> {
        let send = PySendConfig::new(bus, Some(level), None)?;
        self.sends.push(send);
        Ok(())
    }

    /// Get number of EQ bands.
    fn eq_band_count(&self) -> usize {
        self.eq_bands.len()
    }

    /// Get number of sends.
    fn send_count(&self) -> usize {
        self.sends.len()
    }

    /// Get linear gain from the current volume_db.
    fn linear_gain(&self) -> f64 {
        db_to_linear(self.volume_db)
    }

    /// Convert to a simple Python-compatible dict representation.
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), self.name.clone());
        m.insert("volume_db".to_string(), format!("{:.2}", self.volume_db));
        m.insert("pan".to_string(), format!("{:.2}", self.pan));
        m.insert("mute".to_string(), self.mute.to_string());
        m.insert("solo".to_string(), self.solo.to_string());
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyChannelStrip(name='{}', vol={:+.1}dB, pan={:+.2}, mute={}, solo={})",
            self.name, self.volume_db, self.pan, self.mute, self.solo
        )
    }
}

// ---------------------------------------------------------------------------
// PyAudioMixer
// ---------------------------------------------------------------------------

/// Professional audio mixer with multiple channels.
#[pyclass]
pub struct PyAudioMixer {
    config: PyMixerConfig,
    channels: Vec<PyChannelStrip>,
    master_volume_db: f64,
}

#[pymethods]
impl PyAudioMixer {
    /// Create a new audio mixer.
    #[new]
    fn new(config: &PyMixerConfig) -> PyResult<Self> {
        Ok(Self {
            config: config.clone(),
            channels: Vec::new(),
            master_volume_db: 0.0,
        })
    }

    /// Add a channel strip.
    fn add_channel(&mut self, strip: &PyChannelStrip) -> usize {
        let idx = self.channels.len();
        self.channels.push(strip.clone());
        idx
    }

    /// Remove a channel by index.
    fn remove_channel(&mut self, index: usize) -> PyResult<()> {
        if index >= self.channels.len() {
            return Err(PyValueError::new_err(format!(
                "Channel index {} out of range (0..{})",
                index,
                self.channels.len()
            )));
        }
        self.channels.remove(index);
        Ok(())
    }

    /// Set master volume in dB.
    fn set_master_volume(&mut self, db: f64) -> PyResult<()> {
        if !(-120.0..=24.0).contains(&db) {
            return Err(PyValueError::new_err(format!(
                "Master volume must be between -120 and +24 dB, got {db}"
            )));
        }
        self.master_volume_db = db;
        Ok(())
    }

    /// Get master volume in dB.
    fn master_volume(&self) -> f64 {
        self.master_volume_db
    }

    /// Get number of channels.
    fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Get a channel by index (returns a clone).
    fn get_channel(&self, index: usize) -> PyResult<PyChannelStrip> {
        self.channels.get(index).cloned().ok_or_else(|| {
            PyValueError::new_err(format!(
                "Channel index {} out of range (0..{})",
                index,
                self.channels.len()
            ))
        })
    }

    /// Mix input audio buffers into a single output.
    ///
    /// Each element of `inputs` is a Vec<f32> of mono samples for that channel.
    /// Returns the mixed output as a Vec<f32>.
    fn mix_frames(&self, inputs: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let max_len = inputs.iter().map(|v| v.len()).max().unwrap_or(0);
        if max_len == 0 {
            return Ok(Vec::new());
        }

        let master_gain = db_to_linear(self.master_volume_db) as f32;
        let mut output = vec![0.0_f32; max_len];

        for (ch_idx, samples) in inputs.iter().enumerate() {
            let strip = if ch_idx < self.channels.len() {
                &self.channels[ch_idx]
            } else {
                // If more inputs than channels, use unity gain
                for (i, &s) in samples.iter().enumerate() {
                    if i < output.len() {
                        output[i] += s * master_gain;
                    }
                }
                continue;
            };

            if strip.mute {
                continue;
            }

            let ch_gain = db_to_linear(strip.volume_db) as f32;
            for (i, &s) in samples.iter().enumerate() {
                if i < output.len() {
                    output[i] += s * ch_gain * master_gain;
                }
            }
        }

        Ok(output)
    }

    /// Process a block of interleaved multi-channel samples.
    ///
    /// `samples` is a list of per-channel sample buffers.
    /// `block_size` is the number of samples per channel to process.
    fn process_block(&self, samples: Vec<Vec<f32>>, block_size: usize) -> PyResult<Vec<f32>> {
        if block_size == 0 {
            return Err(PyValueError::new_err("block_size must be > 0"));
        }

        // Truncate each channel to block_size
        let truncated: Vec<Vec<f32>> = samples
            .into_iter()
            .map(|mut ch| {
                ch.truncate(block_size);
                ch
            })
            .collect();

        self.mix_frames(truncated)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAudioMixer(channels={}, master={:+.1}dB, sr={})",
            self.channels.len(),
            self.master_volume_db,
            self.config.sample_rate
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Mix two mono signals into a stereo signal with balance control.
///
/// Args:
///     left: Left channel samples.
///     right: Right channel samples.
///     balance: Balance (-1.0 = full left, 0.0 = center, 1.0 = full right).
///
/// Returns:
///     Interleaved stereo output (L0, R0, L1, R1, ...).
#[pyfunction]
#[pyo3(signature = (left, right, balance=None))]
pub fn mix_stereo(left: Vec<f32>, right: Vec<f32>, balance: Option<f64>) -> PyResult<Vec<f32>> {
    let bal = balance.unwrap_or(0.0);
    if !(-1.0..=1.0).contains(&bal) {
        return Err(PyValueError::new_err(format!(
            "Balance must be between -1.0 and 1.0, got {bal}"
        )));
    }

    let len = left.len().max(right.len());
    let left_gain = ((1.0 - bal) / 2.0 + 0.5) as f32;
    let right_gain = ((1.0 + bal) / 2.0 + 0.5) as f32;

    // Clamp gains to [0, 1]
    let left_gain = left_gain.min(1.0).max(0.0);
    let right_gain = right_gain.min(1.0).max(0.0);

    let mut output = Vec::with_capacity(len * 2);
    for i in 0..len {
        let l = left.get(i).copied().unwrap_or(0.0) * left_gain;
        let r = right.get(i).copied().unwrap_or(0.0) * right_gain;
        output.push(l);
        output.push(r);
    }

    Ok(output)
}

/// Apply pan law to mono samples, returning (left, right) channels.
///
/// Args:
///     samples: Mono input samples.
///     pan: Pan position (-1.0 to 1.0).
///
/// Returns:
///     Tuple of (left_samples, right_samples).
#[pyfunction]
pub fn apply_pan(samples: Vec<f32>, pan: f64) -> PyResult<(Vec<f32>, Vec<f32>)> {
    if !(-1.0..=1.0).contains(&pan) {
        return Err(PyValueError::new_err(format!(
            "Pan must be between -1.0 and 1.0, got {pan}"
        )));
    }

    // Constant-power pan law
    let angle = (pan + 1.0) * std::f64::consts::FRAC_PI_4;
    let left_gain = angle.cos() as f32;
    let right_gain = angle.sin() as f32;

    let left: Vec<f32> = samples.iter().map(|&s| s * left_gain).collect();
    let right: Vec<f32> = samples.iter().map(|&s| s * right_gain).collect();

    Ok((left, right))
}

/// Apply gain in dB to audio samples.
///
/// Args:
///     samples: Input samples.
///     gain_db: Gain in dB.
///
/// Returns:
///     Amplified samples.
#[pyfunction]
pub fn apply_gain(samples: Vec<f32>, gain_db: f64) -> Vec<f32> {
    let gain = db_to_linear(gain_db) as f32;
    samples.iter().map(|&s| s * gain).collect()
}

// ---------------------------------------------------------------------------
// Registration helper
// ---------------------------------------------------------------------------

/// Register all mixer bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMixerConfig>()?;
    m.add_class::<PyEqBand>()?;
    m.add_class::<PySendConfig>()?;
    m.add_class::<PyChannelStrip>()?;
    m.add_class::<PyAudioMixer>()?;
    m.add_function(wrap_pyfunction!(mix_stereo, m)?)?;
    m.add_function(wrap_pyfunction!(apply_pan, m)?)?;
    m.add_function(wrap_pyfunction!(apply_gain, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_to_linear_unity() {
        let gain = db_to_linear(0.0);
        assert!((gain - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_db_to_linear_silence() {
        let gain = db_to_linear(-120.0);
        assert!((gain - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_to_db_roundtrip() {
        let db = -6.0;
        let linear = db_to_linear(db);
        let back = linear_to_db(linear);
        assert!((back - db).abs() < 0.001);
    }

    #[test]
    fn test_channel_strip_defaults() {
        let strip = PyChannelStrip::new("Test".to_string());
        assert_eq!(strip.name, "Test");
        assert!((strip.volume_db - 0.0).abs() < f64::EPSILON);
        assert!((strip.pan - 0.0).abs() < f64::EPSILON);
        assert!(!strip.mute);
        assert!(!strip.solo);
    }

    #[test]
    fn test_mixer_config_stereo() {
        let cfg = PyMixerConfig::stereo().expect("should create stereo config");
        assert_eq!(cfg.channels, 2);
        assert_eq!(cfg.sample_rate, 48000);
        assert_eq!(cfg.output_format, "stereo");
    }

    #[test]
    fn test_apply_pan_center() {
        let samples = vec![1.0_f32; 4];
        let (left, right) = apply_pan(samples, 0.0).expect("pan should succeed");
        assert_eq!(left.len(), 4);
        assert_eq!(right.len(), 4);
        // At center, both should be approximately equal (constant-power)
        for i in 0..4 {
            assert!((left[i] - right[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_apply_gain_positive() {
        let samples = vec![0.5_f32; 3];
        let gained = apply_gain(samples, 6.0);
        // +6dB approximately doubles
        for &s in &gained {
            assert!((s - 0.998).abs() < 0.1);
        }
    }
}
