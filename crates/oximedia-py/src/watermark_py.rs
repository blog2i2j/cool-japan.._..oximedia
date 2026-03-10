//! Python bindings for audio watermarking and steganography.
//!
//! Wraps `oximedia-watermark` for spread-spectrum, echo, phase, LSB,
//! patchwork, and QIM watermarking algorithms with quality metrics.

use pyo3::prelude::*;

use oximedia_watermark::{Algorithm, WatermarkConfig, WatermarkDetector, WatermarkEmbedder};

// ---------------------------------------------------------------------------
// PyWatermarkConfig
// ---------------------------------------------------------------------------

/// Configuration for watermark embedding and detection.
#[pyclass]
#[derive(Clone)]
pub struct PyWatermarkConfig {
    /// Algorithm name.
    #[pyo3(get)]
    pub algorithm: String,
    /// Embedding strength (0.0 - 1.0).
    #[pyo3(get)]
    pub strength: f64,
    /// Whether to use psychoacoustic masking.
    #[pyo3(get)]
    pub psychoacoustic: bool,
    /// Secret key for cryptographic operations.
    key: Option<u64>,
}

#[pymethods]
impl PyWatermarkConfig {
    /// Create a new watermark configuration.
    #[new]
    #[pyo3(signature = (algorithm = None, strength = None, psychoacoustic = None))]
    fn new(
        algorithm: Option<&str>,
        strength: Option<f64>,
        psychoacoustic: Option<bool>,
    ) -> PyResult<Self> {
        let algo_str = algorithm.unwrap_or("spread_spectrum");
        // Validate algorithm name
        parse_algorithm(algo_str)?;

        let s = strength.unwrap_or(0.5);
        if !(0.0..=1.0).contains(&s) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Strength must be between 0.0 and 1.0",
            ));
        }

        Ok(Self {
            algorithm: algo_str.to_string(),
            strength: s,
            psychoacoustic: psychoacoustic.unwrap_or(true),
            key: None,
        })
    }

    /// Set the cryptographic key.
    fn with_key(&mut self, key: u64) {
        self.key = Some(key);
    }

    /// Set the algorithm.
    fn with_algorithm(&mut self, algorithm: &str) -> PyResult<()> {
        parse_algorithm(algorithm)?;
        self.algorithm = algorithm.to_string();
        Ok(())
    }

    /// Set the embedding strength.
    fn with_strength(&mut self, strength: f64) -> PyResult<()> {
        if !(0.0..=1.0).contains(&strength) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Strength must be between 0.0 and 1.0",
            ));
        }
        self.strength = strength;
        Ok(())
    }

    /// List all available watermarking algorithms.
    #[staticmethod]
    fn available_algorithms() -> Vec<String> {
        vec![
            "spread_spectrum".to_string(),
            "echo".to_string(),
            "phase".to_string(),
            "lsb".to_string(),
            "patchwork".to_string(),
            "qim".to_string(),
        ]
    }

    fn __repr__(&self) -> String {
        format!(
            "PyWatermarkConfig(algorithm='{}', strength={:.3}, psychoacoustic={}, key={})",
            self.algorithm,
            self.strength,
            self.psychoacoustic,
            if self.key.is_some() { "set" } else { "none" }
        )
    }
}

// ---------------------------------------------------------------------------
// PyWatermarkEmbedder
// ---------------------------------------------------------------------------

/// Embeds watermarks into audio samples.
#[pyclass]
pub struct PyWatermarkEmbedder {
    embedder: WatermarkEmbedder,
    algorithm_name: String,
}

#[pymethods]
impl PyWatermarkEmbedder {
    /// Create a new embedder from a config and sample rate.
    #[new]
    fn new(config: &PyWatermarkConfig, sample_rate: u32) -> PyResult<Self> {
        if sample_rate == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Sample rate must be > 0",
            ));
        }

        let wm_config = build_watermark_config(config)?;
        let embedder = WatermarkEmbedder::new(wm_config, sample_rate);

        Ok(Self {
            embedder,
            algorithm_name: config.algorithm.clone(),
        })
    }

    /// Embed a watermark payload into audio samples.
    ///
    /// Returns the watermarked audio samples.
    fn embed(&self, _py: Python<'_>, samples: Vec<f32>, payload: Vec<u8>) -> PyResult<Vec<f32>> {
        if samples.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Audio samples cannot be empty",
            ));
        }
        if payload.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Payload cannot be empty",
            ));
        }

        self.embedder.embed(&samples, &payload).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Embedding failed: {}", e))
        })
    }

    /// Calculate the embedding capacity in bits for a given sample count.
    fn capacity(&self, sample_count: usize) -> usize {
        self.embedder.capacity(sample_count)
    }

    fn __repr__(&self) -> String {
        format!("PyWatermarkEmbedder(algorithm='{}')", self.algorithm_name)
    }
}

// ---------------------------------------------------------------------------
// PyWatermarkDetector
// ---------------------------------------------------------------------------

/// Detects and extracts watermarks from audio samples.
#[pyclass]
pub struct PyWatermarkDetector {
    detector: WatermarkDetector,
    algorithm_name: String,
}

#[pymethods]
impl PyWatermarkDetector {
    /// Create a new detector from a config.
    #[new]
    fn new(config: &PyWatermarkConfig) -> PyResult<Self> {
        let wm_config = build_watermark_config(config)?;
        let detector = WatermarkDetector::new(wm_config);

        Ok(Self {
            detector,
            algorithm_name: config.algorithm.clone(),
        })
    }

    /// Detect and extract a watermark from audio samples.
    ///
    /// `expected_bits` is the number of bits to extract.
    fn detect(
        &self,
        _py: Python<'_>,
        samples: Vec<f32>,
        expected_bits: usize,
    ) -> PyResult<Vec<u8>> {
        if samples.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Audio samples cannot be empty",
            ));
        }
        if expected_bits == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected bits must be > 0",
            ));
        }

        self.detector.detect(&samples, expected_bits).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Detection failed: {}", e))
        })
    }

    fn __repr__(&self) -> String {
        format!("PyWatermarkDetector(algorithm='{}')", self.algorithm_name)
    }
}

// ---------------------------------------------------------------------------
// PyWatermarkMetrics
// ---------------------------------------------------------------------------

/// Quality metrics comparing original and watermarked audio.
#[pyclass]
#[derive(Clone)]
pub struct PyWatermarkMetrics {
    /// Signal-to-noise ratio in dB.
    #[pyo3(get)]
    pub snr_db: f64,
    /// Objective Difference Grade (approximation of PEAQ ODG).
    #[pyo3(get)]
    pub odg: f64,
    /// Whether the watermark is considered perceptually transparent.
    #[pyo3(get)]
    pub transparent: bool,
    /// Peak Signal-to-Noise Ratio in dB.
    #[pyo3(get)]
    pub psnr_db: f64,
    /// Correlation between original and watermarked signals.
    #[pyo3(get)]
    pub correlation: f64,
}

#[pymethods]
impl PyWatermarkMetrics {
    fn __repr__(&self) -> String {
        format!(
            "PyWatermarkMetrics(snr={:.1} dB, odg={:.2}, psnr={:.1} dB, transparent={})",
            self.snr_db, self.odg, self.psnr_db, self.transparent
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Embed a watermark into audio samples (convenience function).
#[pyfunction]
#[pyo3(signature = (samples, payload, sample_rate, algorithm = None, strength = None))]
pub fn embed_watermark(
    samples: Vec<f32>,
    payload: Vec<u8>,
    sample_rate: u32,
    algorithm: Option<&str>,
    strength: Option<f64>,
) -> PyResult<Vec<f32>> {
    if samples.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Audio samples cannot be empty",
        ));
    }
    if payload.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Payload cannot be empty",
        ));
    }
    if sample_rate == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sample rate must be > 0",
        ));
    }

    let algo = parse_algorithm(algorithm.unwrap_or("spread_spectrum"))?;
    let s = strength.unwrap_or(0.5) as f32;

    let config = WatermarkConfig::default()
        .with_algorithm(algo)
        .with_strength(s)
        .with_psychoacoustic(true);

    let embedder = WatermarkEmbedder::new(config, sample_rate);
    embedder
        .embed(&samples, &payload)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Embedding failed: {}", e)))
}

/// Detect a watermark from audio samples (convenience function).
#[pyfunction]
#[pyo3(signature = (samples, expected_bits, algorithm = None))]
pub fn detect_watermark(
    samples: Vec<f32>,
    expected_bits: usize,
    algorithm: Option<&str>,
) -> PyResult<Vec<u8>> {
    if samples.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Audio samples cannot be empty",
        ));
    }
    if expected_bits == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Expected bits must be > 0",
        ));
    }

    let algo = parse_algorithm(algorithm.unwrap_or("spread_spectrum"))?;

    let config = WatermarkConfig::default().with_algorithm(algo);
    let detector = WatermarkDetector::new(config);

    detector
        .detect(&samples, expected_bits)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Detection failed: {}", e)))
}

/// Compute quality metrics comparing original and watermarked audio.
#[pyfunction]
pub fn watermark_quality(
    original: Vec<f32>,
    watermarked: Vec<f32>,
) -> PyResult<PyWatermarkMetrics> {
    if original.is_empty() || watermarked.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Audio samples cannot be empty",
        ));
    }
    if original.len() != watermarked.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Sample count mismatch: original={}, watermarked={}",
            original.len(),
            watermarked.len()
        )));
    }

    let metrics = oximedia_watermark::calculate_metrics(&original, &watermarked);

    let snr = f64::from(metrics.snr_db);
    let odg = f64::from(metrics.odg);
    let psnr = f64::from(oximedia_watermark::metrics::calculate_psnr(
        &original,
        &watermarked,
    ));
    let correlation = f64::from(oximedia_watermark::metrics::calculate_correlation(
        &original,
        &watermarked,
    ));

    // ODG > -1.0 is considered transparent (imperceptible difference)
    let transparent = odg > -1.0;

    Ok(PyWatermarkMetrics {
        snr_db: snr,
        odg,
        transparent,
        psnr_db: psnr,
        correlation,
    })
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register watermark types and functions onto a Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWatermarkConfig>()?;
    m.add_class::<PyWatermarkEmbedder>()?;
    m.add_class::<PyWatermarkDetector>()?;
    m.add_class::<PyWatermarkMetrics>()?;
    m.add_function(wrap_pyfunction!(embed_watermark, m)?)?;
    m.add_function(wrap_pyfunction!(detect_watermark, m)?)?;
    m.add_function(wrap_pyfunction!(watermark_quality, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn parse_algorithm(name: &str) -> PyResult<Algorithm> {
    match name.to_ascii_lowercase().replace('-', "_").as_str() {
        "spread_spectrum" | "spreadspectrum" | "ss" | "dsss" => Ok(Algorithm::SpreadSpectrum),
        "echo" | "echo_hiding" => Ok(Algorithm::Echo),
        "phase" | "phase_coding" => Ok(Algorithm::Phase),
        "lsb" | "steganography" => Ok(Algorithm::Lsb),
        "patchwork" => Ok(Algorithm::Patchwork),
        "qim" | "quantization" => Ok(Algorithm::Qim),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown algorithm '{}'. Available: spread_spectrum, echo, phase, lsb, patchwork, qim",
            other
        ))),
    }
}

fn build_watermark_config(py_config: &PyWatermarkConfig) -> PyResult<WatermarkConfig> {
    let algo = parse_algorithm(&py_config.algorithm)?;
    let strength = py_config.strength as f32;
    let key = py_config.key.unwrap_or(0);

    let config = WatermarkConfig::default()
        .with_algorithm(algo)
        .with_strength(strength)
        .with_key(key)
        .with_psychoacoustic(py_config.psychoacoustic);

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_algorithm_valid() {
        assert_eq!(
            parse_algorithm("spread_spectrum").ok(),
            Some(Algorithm::SpreadSpectrum)
        );
        assert_eq!(parse_algorithm("echo").ok(), Some(Algorithm::Echo));
        assert_eq!(parse_algorithm("phase").ok(), Some(Algorithm::Phase));
        assert_eq!(parse_algorithm("lsb").ok(), Some(Algorithm::Lsb));
        assert_eq!(
            parse_algorithm("patchwork").ok(),
            Some(Algorithm::Patchwork)
        );
        assert_eq!(parse_algorithm("qim").ok(), Some(Algorithm::Qim));
    }

    #[test]
    fn test_parse_algorithm_aliases() {
        assert_eq!(parse_algorithm("ss").ok(), Some(Algorithm::SpreadSpectrum));
        assert_eq!(
            parse_algorithm("dsss").ok(),
            Some(Algorithm::SpreadSpectrum)
        );
        assert_eq!(parse_algorithm("echo_hiding").ok(), Some(Algorithm::Echo));
    }

    #[test]
    fn test_parse_algorithm_invalid() {
        assert!(parse_algorithm("invalid_algo").is_err());
    }

    #[test]
    fn test_build_config_defaults() {
        let py_config = PyWatermarkConfig {
            algorithm: "spread_spectrum".to_string(),
            strength: 0.5,
            psychoacoustic: true,
            key: None,
        };
        let config = build_watermark_config(&py_config).expect("should succeed");
        assert_eq!(config.algorithm, Algorithm::SpreadSpectrum);
        assert!((config.strength - 0.5).abs() < f32::EPSILON);
        assert!(config.psychoacoustic);
    }

    #[test]
    fn test_watermark_quality_metrics() {
        let original: Vec<f32> = vec![0.5; 10000];
        let watermarked: Vec<f32> = original.iter().map(|&s| s + 0.001).collect();

        let metrics = watermark_quality(original, watermarked).expect("should succeed");
        assert!(metrics.snr_db > 40.0);
        assert!(metrics.transparent);
    }

    #[test]
    fn test_available_algorithms() {
        let algos = PyWatermarkConfig::available_algorithms();
        assert_eq!(algos.len(), 6);
        assert!(algos.contains(&"spread_spectrum".to_string()));
        assert!(algos.contains(&"qim".to_string()));
    }

    #[test]
    fn test_config_repr() {
        let config = PyWatermarkConfig {
            algorithm: "echo".to_string(),
            strength: 0.3,
            psychoacoustic: false,
            key: Some(42),
        };
        let r = config.__repr__();
        assert!(r.contains("echo"));
        assert!(r.contains("0.300"));
        assert!(r.contains("key=set"));
    }
}
