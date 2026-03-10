//! Python bindings for `oximedia-recommend` recommendation engine.
//!
//! Provides `PyRecommendEngine`, `PyRecommendation`, and standalone functions
//! for codec and settings recommendations.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyRecommendation
// ---------------------------------------------------------------------------

/// A recommendation result.
#[pyclass]
#[derive(Clone)]
pub struct PyRecommendation {
    /// Recommended item (codec name, setting, etc.).
    #[pyo3(get)]
    pub item: String,
    /// Confidence score (0.0-1.0).
    #[pyo3(get)]
    pub confidence: f64,
    /// Reason for the recommendation.
    #[pyo3(get)]
    pub reason: String,
    /// Category of recommendation.
    #[pyo3(get)]
    pub category: String,
}

#[pymethods]
impl PyRecommendation {
    fn __repr__(&self) -> String {
        format!(
            "PyRecommendation(item='{}', confidence={:.2}, reason='{}')",
            self.item, self.confidence, self.reason,
        )
    }

    /// Convert to dict.
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("item".to_string(), self.item.clone());
        m.insert("confidence".to_string(), format!("{:.4}", self.confidence));
        m.insert("reason".to_string(), self.reason.clone());
        m.insert("category".to_string(), self.category.clone());
        m
    }
}

// ---------------------------------------------------------------------------
// PyRecommendEngine
// ---------------------------------------------------------------------------

/// Content recommendation and encoding settings engine.
#[pyclass]
pub struct PyRecommendEngine {
    _inner: oximedia_recommend::RecommendationEngine,
}

#[pymethods]
impl PyRecommendEngine {
    /// Create a new recommendation engine.
    #[new]
    fn new() -> Self {
        Self {
            _inner: oximedia_recommend::RecommendationEngine::new(),
        }
    }

    /// Recommend a video codec for a given use case.
    ///
    /// Args:
    ///     use_case: Target use case (streaming, archival, editing, broadcast).
    ///     resolution: Optional resolution string (e.g. "1920x1080").
    ///
    /// Returns:
    ///     A PyRecommendation with the best codec.
    #[pyo3(signature = (use_case, resolution=None))]
    fn recommend_video_codec(
        &self,
        use_case: &str,
        resolution: Option<&str>,
    ) -> PyResult<PyRecommendation> {
        let valid = ["streaming", "archival", "editing", "broadcast"];
        if !valid.contains(&use_case) {
            return Err(PyValueError::new_err(format!(
                "Unknown use case '{}'. Supported: {}",
                use_case,
                valid.join(", ")
            )));
        }

        let (codec, confidence, reason) = match use_case {
            "streaming" => ("AV1", 0.95, "Best compression for streaming delivery"),
            "archival" => ("AV1", 0.90, "Efficient long-term storage"),
            "editing" => ("VP9", 0.85, "Fast decode speed for editing workflows"),
            "broadcast" => ("AV1", 0.92, "High quality at broadcast bitrates"),
            _ => ("AV1", 0.80, "Default recommendation"),
        };

        let _ = resolution; // Used for future resolution-based tuning

        Ok(PyRecommendation {
            item: codec.to_string(),
            confidence,
            reason: reason.to_string(),
            category: "video_codec".to_string(),
        })
    }

    /// Recommend an audio codec for a given use case.
    ///
    /// Args:
    ///     use_case: Target use case.
    ///
    /// Returns:
    ///     A PyRecommendation.
    fn recommend_audio_codec(&self, use_case: &str) -> PyResult<PyRecommendation> {
        let valid = ["streaming", "archival", "editing", "broadcast"];
        if !valid.contains(&use_case) {
            return Err(PyValueError::new_err(format!(
                "Unknown use case '{}'. Supported: {}",
                use_case,
                valid.join(", ")
            )));
        }

        let (codec, confidence, reason) = match use_case {
            "streaming" => ("Opus", 0.95, "Best quality at low bitrates"),
            "archival" => ("FLAC", 0.95, "Lossless preservation"),
            "editing" => ("FLAC", 0.90, "Lossless for editing workflows"),
            "broadcast" => ("Opus", 0.90, "Low latency, high quality"),
            _ => ("Opus", 0.80, "Default recommendation"),
        };

        Ok(PyRecommendation {
            item: codec.to_string(),
            confidence,
            reason: reason.to_string(),
            category: "audio_codec".to_string(),
        })
    }

    /// Get encoding settings recommendation.
    ///
    /// Args:
    ///     codec: Target codec.
    ///     target: Optimization target (quality, speed, size, balanced).
    ///
    /// Returns:
    ///     JSON string with recommended settings.
    fn recommend_encoding_settings(&self, codec: &str, target: &str) -> PyResult<String> {
        let valid_codecs = ["av1", "vp9", "vp8", "opus", "vorbis", "flac"];
        if !valid_codecs.contains(&codec) {
            return Err(PyValueError::new_err(format!(
                "Unsupported codec '{}'. Patent-free only: {}",
                codec,
                valid_codecs.join(", ")
            )));
        }

        let valid_targets = ["quality", "speed", "size", "balanced"];
        if !valid_targets.contains(&target) {
            return Err(PyValueError::new_err(format!(
                "Unknown target '{}'. Supported: {}",
                target,
                valid_targets.join(", ")
            )));
        }

        let (preset, crf) = match target {
            "quality" => ("slow", 22),
            "speed" => ("ultrafast", 28),
            "size" => ("medium", 32),
            _ => ("medium", 26),
        };

        Ok(format!(
            "{{\"codec\":\"{codec}\",\"target\":\"{target}\",\"preset\":\"{preset}\",\"crf\":{crf},\"pixel_format\":\"yuv420p\"}}"
        ))
    }

    /// List all supported strategies.
    fn list_strategies(&self) -> Vec<String> {
        vec![
            "content-based".to_string(),
            "collaborative".to_string(),
            "hybrid".to_string(),
            "personalized".to_string(),
            "trending".to_string(),
        ]
    }

    fn __repr__(&self) -> String {
        "PyRecommendEngine()".to_string()
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Recommend the best codec for a given use case (quick function).
///
/// Args:
///     use_case: Target use case.
///
/// Returns:
///     PyRecommendation.
#[pyfunction]
pub fn recommend_codec(use_case: &str) -> PyResult<PyRecommendation> {
    let engine = PyRecommendEngine::new();
    engine.recommend_video_codec(use_case, None)
}

/// Recommend encoding settings for a codec/target pair (quick function).
///
/// Args:
///     codec: Target codec.
///     target: Optimization target.
///
/// Returns:
///     JSON string with settings.
#[pyfunction]
pub fn recommend_settings(codec: &str, target: &str) -> PyResult<String> {
    let engine = PyRecommendEngine::new();
    engine.recommend_encoding_settings(codec, target)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all recommendation bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRecommendEngine>()?;
    m.add_class::<PyRecommendation>()?;
    m.add_function(wrap_pyfunction!(recommend_codec, m)?)?;
    m.add_function(wrap_pyfunction!(recommend_settings, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = PyRecommendEngine::new();
        assert!(!engine.__repr__().is_empty());
    }

    #[test]
    fn test_video_codec_recommendation() {
        let engine = PyRecommendEngine::new();
        let rec = engine.recommend_video_codec("streaming", None);
        assert!(rec.is_ok());
        let rec = rec.expect("should succeed");
        assert_eq!(rec.item, "AV1");
        assert!(rec.confidence > 0.9);
    }

    #[test]
    fn test_audio_codec_recommendation() {
        let engine = PyRecommendEngine::new();
        let rec = engine.recommend_audio_codec("archival");
        assert!(rec.is_ok());
        let rec = rec.expect("should succeed");
        assert_eq!(rec.item, "FLAC");
    }

    #[test]
    fn test_invalid_use_case() {
        let engine = PyRecommendEngine::new();
        let result = engine.recommend_video_codec("gaming", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_encoding_settings() {
        let result = recommend_settings("av1", "quality");
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("slow"));
    }
}
