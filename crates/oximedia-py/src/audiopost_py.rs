//! Python bindings for `oximedia-audiopost` audio post-production.
//!
//! Provides `PyAdrSession`, `PyMixingConsole`, `PyStemExporter`,
//! `PyDeliveryChecker`, `PyDeliveryReport`, and standalone functions.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn db_to_linear(db: f64) -> f64 {
    if db <= -120.0 {
        0.0
    } else {
        10.0_f64.powf(db / 20.0)
    }
}

/// Parse delivery spec name to target.
fn parse_delivery_target(
    spec: &str,
) -> PyResult<oximedia_audiopost::delivery_spec::DeliveryTarget> {
    use oximedia_audiopost::delivery_spec::DeliveryTarget;
    match spec {
        "broadcast" => Ok(DeliveryTarget::Broadcast),
        "cinema" => Ok(DeliveryTarget::Cinema),
        "streaming" => Ok(DeliveryTarget::Streaming),
        "podcast" => Ok(DeliveryTarget::Podcast),
        other => Err(PyValueError::new_err(format!(
            "Unknown delivery spec '{}'. Expected: broadcast, cinema, streaming, podcast",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// PyAdrCue
// ---------------------------------------------------------------------------

/// A single ADR cue.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyAdrCue {
    /// Timecode string (e.g. "01:00:05:12").
    #[pyo3(get)]
    pub timecode: String,
    /// Duration in seconds.
    #[pyo3(get)]
    pub duration: f64,
    /// Cue text / dialogue line.
    #[pyo3(get)]
    pub text: String,
    /// Character / actor name.
    #[pyo3(get)]
    pub character: String,
}

#[pymethods]
impl PyAdrCue {
    /// Create a new ADR cue.
    #[new]
    #[pyo3(signature = (timecode, duration, text, character=None))]
    fn new(
        timecode: String,
        duration: f64,
        text: String,
        character: Option<String>,
    ) -> PyResult<Self> {
        if duration <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "Duration must be positive, got {duration}"
            )));
        }
        Ok(Self {
            timecode,
            duration,
            text,
            character: character.unwrap_or_default(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAdrCue(tc='{}', dur={:.2}s, text='{}', char='{}')",
            self.timecode, self.duration, self.text, self.character
        )
    }
}

// ---------------------------------------------------------------------------
// PyAdrSession
// ---------------------------------------------------------------------------

/// ADR session management.
#[pyclass]
pub struct PyAdrSession {
    cues: Vec<PyAdrCue>,
    sample_rate: u32,
    pre_roll: f64,
    post_roll: f64,
    session_name: String,
}

#[pymethods]
impl PyAdrSession {
    /// Create a new ADR session.
    #[new]
    #[pyo3(signature = (sample_rate=None, name=None))]
    fn new(sample_rate: Option<u32>, name: Option<String>) -> PyResult<Self> {
        let sr = sample_rate.unwrap_or(48000);
        if sr == 0 {
            return Err(PyValueError::new_err("Sample rate must be > 0"));
        }
        Ok(Self {
            cues: Vec::new(),
            sample_rate: sr,
            pre_roll: 3.0,
            post_roll: 2.0,
            session_name: name.unwrap_or_else(|| "ADR Session".to_string()),
        })
    }

    /// Add a cue to the session.
    fn add_cue(&mut self, timecode: String, duration: f64, text: String) -> PyResult<usize> {
        let cue = PyAdrCue::new(timecode, duration, text, None)?;
        let idx = self.cues.len();
        self.cues.push(cue);
        Ok(idx)
    }

    /// Add a cue with character name.
    fn add_cue_with_character(
        &mut self,
        timecode: String,
        duration: f64,
        text: String,
        character: String,
    ) -> PyResult<usize> {
        let cue = PyAdrCue::new(timecode, duration, text, Some(character))?;
        let idx = self.cues.len();
        self.cues.push(cue);
        Ok(idx)
    }

    /// Remove a cue by index.
    fn remove_cue(&mut self, index: usize) -> PyResult<()> {
        if index >= self.cues.len() {
            return Err(PyValueError::new_err(format!(
                "Cue index {} out of range (0..{})",
                index,
                self.cues.len()
            )));
        }
        self.cues.remove(index);
        Ok(())
    }

    /// Set pre-roll duration in seconds.
    fn set_pre_roll(&mut self, secs: f64) -> PyResult<()> {
        if secs < 0.0 {
            return Err(PyValueError::new_err("Pre-roll must be >= 0"));
        }
        self.pre_roll = secs;
        Ok(())
    }

    /// Set post-roll duration in seconds.
    fn set_post_roll(&mut self, secs: f64) -> PyResult<()> {
        if secs < 0.0 {
            return Err(PyValueError::new_err("Post-roll must be >= 0"));
        }
        self.post_roll = secs;
        Ok(())
    }

    /// Get cue count.
    fn cue_count(&self) -> usize {
        self.cues.len()
    }

    /// Get a cue by index.
    fn get_cue(&self, index: usize) -> PyResult<PyAdrCue> {
        self.cues.get(index).cloned().ok_or_else(|| {
            PyValueError::new_err(format!(
                "Cue index {} out of range (0..{})",
                index,
                self.cues.len()
            ))
        })
    }

    /// Get all cues as a list.
    fn get_all_cues(&self) -> Vec<PyAdrCue> {
        self.cues.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAdrSession(name='{}', sr={}, cues={}, pre={:.1}s, post={:.1}s)",
            self.session_name,
            self.sample_rate,
            self.cues.len(),
            self.pre_roll,
            self.post_roll,
        )
    }
}

// ---------------------------------------------------------------------------
// PyMixingConsole
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct ConsoleChannel {
    name: String,
    samples: Vec<f32>,
    level_db: f64,
    pan: f64,
    mute: bool,
}

/// Professional mixing console for Python.
#[pyclass]
pub struct PyMixingConsole {
    channels: Vec<ConsoleChannel>,
    sample_rate: u32,
    buses: Vec<String>,
    master_level_db: f64,
}

#[pymethods]
impl PyMixingConsole {
    /// Create a new mixing console.
    #[new]
    #[pyo3(signature = (sample_rate=None))]
    fn new(sample_rate: Option<u32>) -> PyResult<Self> {
        let sr = sample_rate.unwrap_or(48000);
        if sr == 0 {
            return Err(PyValueError::new_err("Sample rate must be > 0"));
        }
        Ok(Self {
            channels: Vec::new(),
            sample_rate: sr,
            buses: Vec::new(),
            master_level_db: 0.0,
        })
    }

    /// Add a channel with audio samples.
    fn add_channel(&mut self, name: String, samples: Vec<f32>) -> usize {
        let idx = self.channels.len();
        self.channels.push(ConsoleChannel {
            name,
            samples,
            level_db: 0.0,
            pan: 0.0,
            mute: false,
        });
        idx
    }

    /// Set channel level in dB.
    fn set_level(&mut self, ch: usize, db: f64) -> PyResult<()> {
        let len = self.channels.len();
        let channel = self.channels.get_mut(ch).ok_or_else(|| {
            PyValueError::new_err(format!("Channel {} not found (0..{})", ch, len))
        })?;
        if !(-120.0..=24.0).contains(&db) {
            return Err(PyValueError::new_err(format!(
                "Level must be between -120 and +24 dB, got {db}"
            )));
        }
        channel.level_db = db;
        Ok(())
    }

    /// Set channel pan position.
    fn set_pan(&mut self, ch: usize, pos: f64) -> PyResult<()> {
        let len = self.channels.len();
        let channel = self.channels.get_mut(ch).ok_or_else(|| {
            PyValueError::new_err(format!("Channel {} not found (0..{})", ch, len))
        })?;
        if !(-1.0..=1.0).contains(&pos) {
            return Err(PyValueError::new_err(format!(
                "Pan must be between -1.0 and 1.0, got {pos}"
            )));
        }
        channel.pan = pos;
        Ok(())
    }

    /// Set channel mute state.
    fn set_mute(&mut self, ch: usize, muted: bool) -> PyResult<()> {
        let len = self.channels.len();
        let channel = self.channels.get_mut(ch).ok_or_else(|| {
            PyValueError::new_err(format!("Channel {} not found (0..{})", ch, len))
        })?;
        channel.mute = muted;
        Ok(())
    }

    /// Add a bus.
    fn add_bus(&mut self, name: String) -> usize {
        let idx = self.buses.len();
        self.buses.push(name);
        idx
    }

    /// Get number of channels.
    fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Get number of buses.
    fn bus_count(&self) -> usize {
        self.buses.len()
    }

    /// Mix all channels into a mono output.
    fn mix(&self) -> PyResult<Vec<f32>> {
        if self.channels.is_empty() {
            return Ok(Vec::new());
        }

        let max_len = self
            .channels
            .iter()
            .map(|ch| ch.samples.len())
            .max()
            .unwrap_or(0);

        if max_len == 0 {
            return Ok(Vec::new());
        }

        let master_gain = db_to_linear(self.master_level_db) as f32;
        let mut output = vec![0.0_f32; max_len];

        for ch in &self.channels {
            if ch.mute {
                continue;
            }
            let gain = db_to_linear(ch.level_db) as f32 * master_gain;
            for (i, &s) in ch.samples.iter().enumerate() {
                if i < output.len() {
                    output[i] += s * gain;
                }
            }
        }

        Ok(output)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyMixingConsole(sr={}, channels={}, buses={}, master={:+.1}dB)",
            self.sample_rate,
            self.channels.len(),
            self.buses.len(),
            self.master_level_db,
        )
    }
}

// ---------------------------------------------------------------------------
// PyStemExporter
// ---------------------------------------------------------------------------

struct StemData {
    name: String,
    samples: Vec<f32>,
}

/// Audio stem exporter.
#[pyclass]
pub struct PyStemExporter {
    stems: Vec<StemData>,
    sample_rate: u32,
}

#[pymethods]
impl PyStemExporter {
    /// Create a new stem exporter.
    #[new]
    #[pyo3(signature = (sample_rate=None))]
    fn new(sample_rate: Option<u32>) -> PyResult<Self> {
        let sr = sample_rate.unwrap_or(48000);
        if sr == 0 {
            return Err(PyValueError::new_err("Sample rate must be > 0"));
        }
        Ok(Self {
            stems: Vec::new(),
            sample_rate: sr,
        })
    }

    /// Add a stem with audio data.
    fn add_stem(&mut self, name: String, samples: Vec<f32>) -> usize {
        let idx = self.stems.len();
        self.stems.push(StemData { name, samples });
        idx
    }

    /// Get stem count.
    fn stem_count(&self) -> usize {
        self.stems.len()
    }

    /// Get stem names.
    fn stem_names(&self) -> Vec<String> {
        self.stems.iter().map(|s| s.name.clone()).collect()
    }

    /// Get samples for a specific stem.
    fn get_stem_samples(&self, index: usize) -> PyResult<Vec<f32>> {
        self.stems
            .get(index)
            .map(|s| s.samples.clone())
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Stem index {} out of range (0..{})",
                    index,
                    self.stems.len()
                ))
            })
    }

    /// Export information as a summary dict.
    fn export_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("stem_count".to_string(), self.stems.len().to_string());
        info.insert("sample_rate".to_string(), self.sample_rate.to_string());
        for (i, stem) in self.stems.iter().enumerate() {
            info.insert(format!("stem_{}_name", i), stem.name.clone());
            info.insert(
                format!("stem_{}_samples", i),
                stem.samples.len().to_string(),
            );
        }
        info
    }

    fn __repr__(&self) -> String {
        format!(
            "PyStemExporter(sr={}, stems={})",
            self.sample_rate,
            self.stems.len()
        )
    }
}

// ---------------------------------------------------------------------------
// PyDeliveryReport
// ---------------------------------------------------------------------------

/// Result of a delivery specification check.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyDeliveryReport {
    /// Whether the audio passed all checks.
    #[pyo3(get)]
    pub passed: bool,
    /// Delivery spec name.
    #[pyo3(get)]
    pub spec_name: String,
    /// Individual check results: (check_name, passed, message).
    checks: Vec<(String, bool, String)>,
}

#[pymethods]
impl PyDeliveryReport {
    /// Get all check results.
    fn get_checks(&self) -> Vec<(String, bool, String)> {
        self.checks.clone()
    }

    /// Get only failed checks.
    fn failed_checks(&self) -> Vec<(String, String)> {
        self.checks
            .iter()
            .filter(|(_, passed, _)| !passed)
            .map(|(name, _, msg)| (name.clone(), msg.clone()))
            .collect()
    }

    /// Get a summary string.
    fn summary(&self) -> String {
        let total = self.checks.len();
        let passed_count = self.checks.iter().filter(|(_, p, _)| *p).count();
        format!(
            "{}: {}/{} checks passed ({})",
            self.spec_name,
            passed_count,
            total,
            if self.passed { "PASS" } else { "FAIL" }
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "PyDeliveryReport(spec='{}', passed={}, checks={})",
            self.spec_name,
            self.passed,
            self.checks.len()
        )
    }
}

// ---------------------------------------------------------------------------
// PyDeliveryChecker
// ---------------------------------------------------------------------------

/// Delivery specification checker.
#[pyclass]
pub struct PyDeliveryChecker {
    spec_name: String,
    target: oximedia_audiopost::delivery_spec::DeliveryTarget,
}

#[pymethods]
impl PyDeliveryChecker {
    /// Create a new delivery checker.
    ///
    /// Args:
    ///     spec: Delivery spec name (broadcast, cinema, streaming, podcast).
    #[new]
    fn new(spec: &str) -> PyResult<Self> {
        let target = parse_delivery_target(spec)?;
        Ok(Self {
            spec_name: spec.to_string(),
            target,
        })
    }

    /// Check audio samples against the delivery spec.
    ///
    /// Args:
    ///     samples: Audio samples (mono f32).
    ///     sample_rate: Sample rate in Hz.
    fn check(&self, samples: Vec<f32>, sample_rate: u32) -> PyResult<PyDeliveryReport> {
        if samples.is_empty() {
            return Err(PyValueError::new_err("No samples provided"));
        }
        if sample_rate == 0 {
            return Err(PyValueError::new_err("Sample rate must be > 0"));
        }

        let spec = oximedia_audiopost::delivery_spec::AudioDeliverySpec::from_target(
            self.target,
            1,
            sample_rate,
        );

        let mut checks = Vec::new();

        // Check sample rate
        let sr_ok = sample_rate >= spec.sample_rate_hz;
        checks.push((
            "sample_rate".to_string(),
            sr_ok,
            format!(
                "Required >= {} Hz, got {} Hz",
                spec.sample_rate_hz, sample_rate
            ),
        ));

        // Compute peak level
        let peak = samples.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        let peak_dbfs = if peak > 0.0 {
            20.0 * (peak as f64).log10()
        } else {
            -120.0
        };

        let peak_ok = peak_dbfs <= spec.max_true_peak_dbtp as f64;
        checks.push((
            "true_peak".to_string(),
            peak_ok,
            format!(
                "Max true peak: {} dBTP, measured: {:.1} dBFS",
                spec.max_true_peak_dbtp, peak_dbfs
            ),
        ));

        // Compute approximate integrated loudness (simplified RMS-based)
        let rms = if !samples.is_empty() {
            let sum: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
            (sum / samples.len() as f64).sqrt()
        } else {
            0.0
        };
        let loudness_approx = if rms > 0.0 {
            20.0 * rms.log10() - 0.691 // K-weighting approximation offset
        } else {
            -120.0
        };

        let loudness_ok = loudness_approx <= spec.max_loudness_lkfs as f64 + 1.0; // 1 LU tolerance
        checks.push((
            "loudness".to_string(),
            loudness_ok,
            format!(
                "Max loudness: {} LKFS, measured: {:.1} LKFS (approx)",
                spec.max_loudness_lkfs, loudness_approx
            ),
        ));

        let all_passed = checks.iter().all(|(_, p, _)| *p);

        Ok(PyDeliveryReport {
            passed: all_passed,
            spec_name: self.spec_name.clone(),
            checks,
        })
    }

    /// List available delivery specs.
    #[staticmethod]
    fn list_specs() -> Vec<String> {
        vec![
            "broadcast".to_string(),
            "cinema".to_string(),
            "streaming".to_string(),
            "podcast".to_string(),
        ]
    }

    fn __repr__(&self) -> String {
        format!("PyDeliveryChecker(spec='{}')", self.spec_name)
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Check audio against a delivery specification.
///
/// Args:
///     samples: Audio samples (mono f32).
///     sample_rate: Sample rate in Hz.
///     spec: Delivery spec name.
///
/// Returns:
///     PyDeliveryReport with check results.
#[pyfunction]
pub fn check_delivery_spec(
    samples: Vec<f32>,
    sample_rate: u32,
    spec: &str,
) -> PyResult<PyDeliveryReport> {
    let checker = PyDeliveryChecker::new(spec)?;
    checker.check(samples, sample_rate)
}

/// Apply basic audio restoration to samples.
///
/// Args:
///     samples: Input audio samples.
///     sample_rate: Sample rate in Hz.
///     declip: Enable declipping.
///     dehum: Enable hum removal.
///     decrackle: Enable crackle removal.
///     denoise: Enable noise reduction.
///
/// Returns:
///     Restored audio samples.
#[pyfunction]
#[pyo3(signature = (samples, sample_rate, declip=false, dehum=false, decrackle=false, denoise=false))]
pub fn restore_audio(
    samples: Vec<f32>,
    sample_rate: u32,
    declip: bool,
    dehum: bool,
    decrackle: bool,
    denoise: bool,
) -> PyResult<Vec<f32>> {
    if samples.is_empty() {
        return Err(PyValueError::new_err("No samples provided"));
    }
    if sample_rate == 0 {
        return Err(PyValueError::new_err("Sample rate must be > 0"));
    }

    let mut output = samples;

    // Declipping: soft-clip any samples that appear clipped
    if declip {
        let threshold = 0.99_f32;
        for s in &mut output {
            if s.abs() > threshold {
                // Soft saturation using tanh-like curve
                let sign = s.signum();
                let abs_val = s.abs();
                *s = sign
                    * (threshold
                        + (1.0 - threshold) * ((abs_val - threshold) / (1.0 - threshold)).tanh());
            }
        }
    }

    // Dehum: simple notch at 50/60 Hz using moving average subtraction
    if dehum {
        let hum_period_50 = (sample_rate as f64 / 50.0).round() as usize;
        let hum_period_60 = (sample_rate as f64 / 60.0).round() as usize;

        for period in &[hum_period_50, hum_period_60] {
            if *period > 0 && output.len() > *period {
                let mut filtered = vec![0.0_f32; output.len()];
                for i in 0..output.len() {
                    let prev = if i >= *period {
                        output[i - period]
                    } else {
                        0.0
                    };
                    filtered[i] = output[i] - prev * 0.3;
                }
                output = filtered;
            }
        }
    }

    // Decrackle: median filter for impulse noise
    if decrackle {
        let window = 3_usize;
        let half = window / 2;
        let original = output.clone();
        for i in half..output.len().saturating_sub(half) {
            let mut window_samples: Vec<f32> = (0..window)
                .filter_map(|j| original.get(i + j - half).copied())
                .collect();
            window_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(&median) = window_samples.get(window_samples.len() / 2) {
                // Only replace if the sample deviates significantly from the median
                if (output[i] - median).abs() > 0.1 {
                    output[i] = median;
                }
            }
        }
    }

    // Denoise: simple spectral subtraction placeholder using windowed averaging
    if denoise {
        // Simple exponential moving average for noise reduction
        let alpha = 0.15_f32;
        let mut prev = output.first().copied().unwrap_or(0.0);
        for s in &mut output {
            *s = alpha * *s + (1.0 - alpha) * prev;
            prev = *s;
        }
    }

    Ok(output)
}

/// List available stem types.
#[pyfunction]
pub fn list_stem_types() -> Vec<String> {
    oximedia_audiopost::stems::StemType::standard_types()
        .iter()
        .map(|t| t.as_str().to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Registration helper
// ---------------------------------------------------------------------------

/// Register all audiopost bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAdrCue>()?;
    m.add_class::<PyAdrSession>()?;
    m.add_class::<PyMixingConsole>()?;
    m.add_class::<PyStemExporter>()?;
    m.add_class::<PyDeliveryReport>()?;
    m.add_class::<PyDeliveryChecker>()?;
    m.add_function(wrap_pyfunction!(check_delivery_spec, m)?)?;
    m.add_function(wrap_pyfunction!(restore_audio, m)?)?;
    m.add_function(wrap_pyfunction!(list_stem_types, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adr_session_creation() {
        let session = PyAdrSession::new(Some(48000), None);
        assert!(session.is_ok());
        let session = session.expect("should create session");
        assert_eq!(session.cue_count(), 0);
    }

    #[test]
    fn test_adr_add_remove_cue() {
        let mut session = PyAdrSession::new(None, None).expect("should create");
        let idx = session.add_cue("01:00:00:00".to_string(), 2.5, "Hello world".to_string());
        assert!(idx.is_ok());
        assert_eq!(session.cue_count(), 1);
        let _ = session.remove_cue(0);
        assert_eq!(session.cue_count(), 0);
    }

    #[test]
    fn test_delivery_checker_broadcast() {
        let checker = PyDeliveryChecker::new("broadcast");
        assert!(checker.is_ok());
        let checker = checker.expect("should create checker");

        // Check with quiet audio (should pass)
        let samples = vec![0.01_f32; 48000];
        let report = checker.check(samples, 48000);
        assert!(report.is_ok());
    }

    #[test]
    fn test_delivery_list_specs() {
        let specs = PyDeliveryChecker::list_specs();
        assert_eq!(specs.len(), 4);
        assert!(specs.contains(&"broadcast".to_string()));
    }

    #[test]
    fn test_restore_audio_declip() {
        let samples = vec![1.0_f32; 100]; // Clipped audio
        let result = restore_audio(samples, 48000, true, false, false, false);
        assert!(result.is_ok());
        let restored = result.expect("should restore");
        // Declipped samples should be below 1.0
        for &s in &restored {
            assert!(s <= 1.0);
        }
    }

    #[test]
    fn test_stem_exporter() {
        let mut exporter = PyStemExporter::new(None).expect("should create");
        exporter.add_stem("Dialogue".to_string(), vec![0.5; 100]);
        exporter.add_stem("Music".to_string(), vec![0.3; 100]);
        assert_eq!(exporter.stem_count(), 2);
        let names = exporter.stem_names();
        assert_eq!(names[0], "Dialogue");
        assert_eq!(names[1], "Music");
    }
}
