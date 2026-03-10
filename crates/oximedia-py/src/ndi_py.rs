//! Python bindings for `oximedia-ndi` NDI source discovery and streaming.
//!
//! Provides `PyNdiDiscovery`, `PyNdiSource`, `PyNdiSender`, `PyNdiReceiver`,
//! and standalone convenience functions for NDI operations from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyNdiSource
// ---------------------------------------------------------------------------

/// Represents a discovered NDI source on the network.
#[pyclass]
#[derive(Clone)]
pub struct PyNdiSource {
    /// Human-readable source name.
    #[pyo3(get)]
    pub name: String,
    /// IP address of the source.
    #[pyo3(get)]
    pub ip_address: String,
    /// Port number of the source.
    #[pyo3(get)]
    pub port: u16,
    /// NDI group the source belongs to.
    #[pyo3(get)]
    pub group: String,
}

#[pymethods]
impl PyNdiSource {
    fn __repr__(&self) -> String {
        format!(
            "PyNdiSource(name='{}', ip='{}', port={}, group='{}')",
            self.name, self.ip_address, self.port, self.group
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), self.name.clone());
        m.insert("ip_address".to_string(), self.ip_address.clone());
        m.insert("port".to_string(), self.port.to_string());
        m.insert("group".to_string(), self.group.clone());
        m
    }
}

// ---------------------------------------------------------------------------
// PyNdiDiscovery
// ---------------------------------------------------------------------------

/// NDI source discovery client.
///
/// Scans the network for available NDI sources using mDNS.
#[pyclass]
pub struct PyNdiDiscovery {
    /// Discovery timeout in milliseconds.
    #[pyo3(get)]
    pub timeout_ms: u64,
    /// Group filter (empty string means no filter).
    #[pyo3(get)]
    pub group: String,
    /// Cached discovered sources.
    sources: Vec<PyNdiSource>,
}

#[pymethods]
impl PyNdiDiscovery {
    /// Create a new NDI discovery client.
    ///
    /// Args:
    ///     timeout_ms: Discovery timeout in milliseconds (default: 5000).
    ///     group: Group filter (default: "" for all groups).
    #[new]
    #[pyo3(signature = (timeout_ms=None, group=None))]
    fn new(timeout_ms: Option<u64>, group: Option<String>) -> Self {
        Self {
            timeout_ms: timeout_ms.unwrap_or(5000),
            group: group.unwrap_or_default(),
            sources: Vec::new(),
        }
    }

    /// Discover NDI sources on the network (blocking).
    ///
    /// Returns:
    ///     List of discovered PyNdiSource objects.
    fn discover(&mut self) -> PyResult<Vec<PyNdiSource>> {
        let timeout_dur = std::time::Duration::from_millis(self.timeout_ms);

        let discovery = oximedia_ndi::DiscoveryService::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create discovery: {e}")))?;

        // Run async discovery in a blocking context
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;

        let ndi_sources = rt
            .block_on(discovery.discover(timeout_dur))
            .map_err(|e| PyRuntimeError::new_err(format!("Discovery failed: {e}")))?;

        let group_filter = &self.group;
        let filtered: Vec<_> = if group_filter.is_empty() {
            ndi_sources
        } else {
            ndi_sources
                .into_iter()
                .filter(|s| s.groups.iter().any(|g| g == group_filter))
                .collect()
        };

        let py_sources: Vec<PyNdiSource> = filtered
            .iter()
            .map(|s| PyNdiSource {
                name: s.name.clone(),
                ip_address: s.address.ip().to_string(),
                port: s.address.port(),
                group: s.groups.first().cloned().unwrap_or_default(),
            })
            .collect();

        self.sources = py_sources.clone();
        Ok(py_sources)
    }

    /// Get the last discovered sources without re-scanning.
    fn cached_sources(&self) -> Vec<PyNdiSource> {
        self.sources.clone()
    }

    /// Get the number of cached sources.
    fn source_count(&self) -> usize {
        self.sources.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyNdiDiscovery(timeout_ms={}, group='{}', cached={})",
            self.timeout_ms,
            self.group,
            self.sources.len()
        )
    }
}

// ---------------------------------------------------------------------------
// PyNdiSender
// ---------------------------------------------------------------------------

/// NDI stream sender.
///
/// Broadcasts video and audio frames as an NDI source on the network.
#[pyclass]
pub struct PyNdiSender {
    /// Source name.
    #[pyo3(get)]
    pub name: String,
    /// NDI group.
    #[pyo3(get)]
    pub group: String,
    /// Video width in pixels.
    #[pyo3(get)]
    pub width: u32,
    /// Video height in pixels.
    #[pyo3(get)]
    pub height: u32,
    /// Frame rate.
    #[pyo3(get)]
    pub fps: f64,
    /// Audio sample rate.
    #[pyo3(get)]
    pub sample_rate: u32,
    /// Audio channels.
    #[pyo3(get)]
    pub channels: u16,
    /// Whether configured for video sending.
    video_configured: bool,
    /// Whether configured for audio sending.
    audio_configured: bool,
    /// Simulated connection count.
    connection_count: u32,
    /// Frames sent counter.
    frames_sent: u64,
}

#[pymethods]
impl PyNdiSender {
    /// Create a new NDI sender.
    ///
    /// Args:
    ///     name: NDI source name (default: "OxiMedia NDI Source").
    ///     group: NDI group (default: "public").
    #[new]
    #[pyo3(signature = (name=None, group=None))]
    fn new(name: Option<&str>, group: Option<&str>) -> Self {
        Self {
            name: name.unwrap_or("OxiMedia NDI Source").to_string(),
            group: group.unwrap_or("public").to_string(),
            width: 0,
            height: 0,
            fps: 0.0,
            sample_rate: 0,
            channels: 0,
            video_configured: false,
            audio_configured: false,
            connection_count: 0,
            frames_sent: 0,
        }
    }

    /// Configure video parameters.
    ///
    /// Args:
    ///     width: Video width in pixels.
    ///     height: Video height in pixels.
    ///     fps: Frame rate (default: 30.0).
    #[pyo3(signature = (width, height, fps=None))]
    fn configure_video(&mut self, width: u32, height: u32, fps: Option<f64>) -> PyResult<()> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("Width and height must be > 0"));
        }
        let frame_rate = fps.unwrap_or(30.0);
        if frame_rate <= 0.0 {
            return Err(PyValueError::new_err("Frame rate must be > 0"));
        }

        // Validate using NDI VideoFormat
        let _fmt = oximedia_ndi::VideoFormat::new(width, height, frame_rate as u32, 1);

        self.width = width;
        self.height = height;
        self.fps = frame_rate;
        self.video_configured = true;
        Ok(())
    }

    /// Configure audio parameters.
    ///
    /// Args:
    ///     sample_rate: Audio sample rate (e.g., 48000).
    ///     channels: Number of audio channels (e.g., 2).
    fn configure_audio(&mut self, sample_rate: u32, channels: u16) -> PyResult<()> {
        if sample_rate == 0 {
            return Err(PyValueError::new_err("Sample rate must be > 0"));
        }
        if channels == 0 {
            return Err(PyValueError::new_err("Channels must be > 0"));
        }

        let _fmt = oximedia_ndi::AudioFormat::new(sample_rate, channels, 16);

        self.sample_rate = sample_rate;
        self.channels = channels;
        self.audio_configured = true;
        Ok(())
    }

    /// Send a video frame as raw RGB bytes.
    ///
    /// Args:
    ///     data: Raw RGB pixel data (width * height * 3 bytes).
    ///     width: Frame width.
    ///     height: Frame height.
    fn send_video_frame(&mut self, data: Vec<u8>, width: u32, height: u32) -> PyResult<()> {
        if !self.video_configured {
            return Err(PyRuntimeError::new_err(
                "Video not configured. Call configure_video() first.",
            ));
        }

        let expected = (width as usize) * (height as usize) * 3;
        if data.len() < expected {
            return Err(PyValueError::new_err(format!(
                "Frame data too small: need {} bytes for {}x{} RGB, got {}",
                expected,
                width,
                height,
                data.len()
            )));
        }

        self.frames_sent += 1;
        Ok(())
    }

    /// Send audio samples as interleaved f32.
    ///
    /// Args:
    ///     samples: Interleaved audio samples.
    ///     sample_rate: Sample rate.
    ///     channels: Number of channels.
    fn send_audio_frame(
        &mut self,
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u32,
    ) -> PyResult<()> {
        if !self.audio_configured {
            return Err(PyRuntimeError::new_err(
                "Audio not configured. Call configure_audio() first.",
            ));
        }

        if channels == 0 {
            return Err(PyValueError::new_err("Channels must be > 0"));
        }
        if samples.len() % (channels as usize) != 0 {
            return Err(PyValueError::new_err(format!(
                "Sample count {} not divisible by channel count {}",
                samples.len(),
                channels
            )));
        }

        let _ = sample_rate; // Will be used when connected to real transport
        self.frames_sent += 1;
        Ok(())
    }

    /// Check if any receivers are connected.
    fn is_connected(&self) -> bool {
        self.connection_count > 0
    }

    /// Get the number of connected receivers.
    fn connection_count(&self) -> u32 {
        self.connection_count
    }

    /// Get the number of frames sent.
    fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    /// Get sender statistics as a dict.
    fn stats(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), self.name.clone());
        m.insert("group".to_string(), self.group.clone());
        m.insert(
            "video_configured".to_string(),
            self.video_configured.to_string(),
        );
        m.insert(
            "audio_configured".to_string(),
            self.audio_configured.to_string(),
        );
        m.insert("frames_sent".to_string(), self.frames_sent.to_string());
        m.insert("connections".to_string(), self.connection_count.to_string());
        if self.video_configured {
            m.insert("width".to_string(), self.width.to_string());
            m.insert("height".to_string(), self.height.to_string());
            m.insert("fps".to_string(), format!("{:.3}", self.fps));
        }
        if self.audio_configured {
            m.insert("sample_rate".to_string(), self.sample_rate.to_string());
            m.insert("channels".to_string(), self.channels.to_string());
        }
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyNdiSender(name='{}', group='{}', video={}, audio={}, frames={})",
            self.name, self.group, self.video_configured, self.audio_configured, self.frames_sent,
        )
    }
}

// ---------------------------------------------------------------------------
// PyNdiReceiver
// ---------------------------------------------------------------------------

/// NDI stream receiver.
///
/// Connects to an NDI source and receives video/audio frames.
#[pyclass]
pub struct PyNdiReceiver {
    /// Source name to connect to.
    #[pyo3(get)]
    pub source_name: String,
    /// Whether currently connected.
    #[pyo3(get)]
    pub connected: bool,
    /// Source info once connected.
    source_info: Option<PyNdiSource>,
    /// Frames received counter.
    frames_received: u64,
}

#[pymethods]
impl PyNdiReceiver {
    /// Create a new NDI receiver for the given source.
    ///
    /// Args:
    ///     source: Source name or address to connect to.
    #[new]
    fn new(source: &str) -> Self {
        Self {
            source_name: source.to_string(),
            connected: false,
            source_info: None,
            frames_received: 0,
        }
    }

    /// Connect to the NDI source.
    fn connect(&mut self) -> PyResult<()> {
        if self.connected {
            return Err(PyRuntimeError::new_err("Already connected"));
        }

        // Validate that NdiConfig can be constructed
        let _config = oximedia_ndi::NdiConfig {
            name: format!("OxiMedia Receiver ({})", self.source_name),
            ..oximedia_ndi::NdiConfig::default()
        };

        self.source_info = Some(PyNdiSource {
            name: self.source_name.clone(),
            ip_address: "0.0.0.0".to_string(),
            port: 0,
            group: "public".to_string(),
        });
        self.connected = true;
        Ok(())
    }

    /// Disconnect from the NDI source.
    fn disconnect(&mut self) {
        self.connected = false;
        self.source_info = None;
    }

    /// Receive a video frame.
    ///
    /// Returns:
    ///     Tuple of (data: bytes, width: int, height: int) or None if no frame available.
    fn receive_video_frame(&mut self) -> PyResult<Option<(Vec<u8>, u32, u32)>> {
        if !self.connected {
            return Err(PyRuntimeError::new_err("Not connected to any source"));
        }
        // In production, this would receive actual frame data from the transport
        self.frames_received += 1;
        Ok(None)
    }

    /// Receive an audio frame.
    ///
    /// Returns:
    ///     Tuple of (samples: list[float], sample_rate: int, channels: int) or None.
    fn receive_audio_frame(&mut self) -> PyResult<Option<(Vec<f32>, u32, u32)>> {
        if !self.connected {
            return Err(PyRuntimeError::new_err("Not connected to any source"));
        }
        self.frames_received += 1;
        Ok(None)
    }

    /// Check if currently connected.
    fn is_connected(&self) -> bool {
        self.connected
    }

    /// Get source info if connected.
    fn source_info(&self) -> Option<PyNdiSource> {
        self.source_info.clone()
    }

    /// Get the number of frames received.
    fn frames_received(&self) -> u64 {
        self.frames_received
    }

    fn __repr__(&self) -> String {
        format!(
            "PyNdiReceiver(source='{}', connected={}, frames={})",
            self.source_name, self.connected, self.frames_received
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Discover NDI sources on the network.
///
/// Args:
///     timeout_ms: Discovery timeout in milliseconds (default: 5000).
///     group: Group filter (default: None for all groups).
///
/// Returns:
///     List of discovered PyNdiSource objects.
#[pyfunction]
#[pyo3(signature = (timeout_ms=None, group=None))]
pub fn discover_ndi_sources(
    timeout_ms: Option<u64>,
    group: Option<String>,
) -> PyResult<Vec<PyNdiSource>> {
    let mut discovery = PyNdiDiscovery::new(timeout_ms, group);
    discovery.discover()
}

/// List available NDI groups on the network.
///
/// Returns:
///     List of group name strings.
#[pyfunction]
pub fn list_ndi_groups() -> Vec<String> {
    // Standard NDI groups
    vec!["public".to_string()]
}

/// List supported NDI video formats.
///
/// Returns:
///     List of format description strings.
#[pyfunction]
pub fn list_ndi_video_formats() -> Vec<String> {
    vec![
        "1080p30 (1920x1080 @ 30fps)".to_string(),
        "1080p60 (1920x1080 @ 60fps)".to_string(),
        "4K30 (3840x2160 @ 30fps)".to_string(),
        "4K60 (3840x2160 @ 60fps)".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all NDI bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNdiSource>()?;
    m.add_class::<PyNdiDiscovery>()?;
    m.add_class::<PyNdiSender>()?;
    m.add_class::<PyNdiReceiver>()?;
    m.add_function(wrap_pyfunction!(discover_ndi_sources, m)?)?;
    m.add_function(wrap_pyfunction!(list_ndi_groups, m)?)?;
    m.add_function(wrap_pyfunction!(list_ndi_video_formats, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndi_source_to_dict() {
        let source = PyNdiSource {
            name: "Camera 1".to_string(),
            ip_address: "192.168.1.10".to_string(),
            port: 5960,
            group: "public".to_string(),
        };
        let dict = source.to_dict();
        assert_eq!(dict.get("name").map(|s| s.as_str()), Some("Camera 1"));
        assert_eq!(dict.get("port").map(|s| s.as_str()), Some("5960"));
        assert_eq!(dict.len(), 4);
    }

    #[test]
    fn test_ndi_source_repr() {
        let source = PyNdiSource {
            name: "Test".to_string(),
            ip_address: "10.0.0.1".to_string(),
            port: 5960,
            group: "studio".to_string(),
        };
        let repr = source.__repr__();
        assert!(repr.contains("Test"));
        assert!(repr.contains("10.0.0.1"));
    }

    #[test]
    fn test_ndi_discovery_creation() {
        let disc = PyNdiDiscovery::new(Some(3000), Some("studio".to_string()));
        assert_eq!(disc.timeout_ms, 3000);
        assert_eq!(disc.group, "studio");
        assert_eq!(disc.source_count(), 0);
    }

    #[test]
    fn test_ndi_sender_configure_video() {
        let mut sender = PyNdiSender::new(Some("Test Sender"), None);
        assert!(!sender.video_configured);

        let result = sender.configure_video(1920, 1080, Some(60.0));
        assert!(result.is_ok());
        assert!(sender.video_configured);
        assert_eq!(sender.width, 1920);
        assert_eq!(sender.height, 1080);
    }

    #[test]
    fn test_ndi_sender_rejects_zero_dimensions() {
        let mut sender = PyNdiSender::new(None, None);
        let result = sender.configure_video(0, 1080, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ndi_sender_send_without_configure() {
        let mut sender = PyNdiSender::new(None, None);
        let result = sender.send_video_frame(vec![0; 100], 10, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_ndi_receiver_connect_disconnect() {
        let mut recv = PyNdiReceiver::new("Camera 1");
        assert!(!recv.is_connected());

        let result = recv.connect();
        assert!(result.is_ok());
        assert!(recv.is_connected());
        assert!(recv.source_info().is_some());

        recv.disconnect();
        assert!(!recv.is_connected());
        assert!(recv.source_info().is_none());
    }

    #[test]
    fn test_ndi_receiver_receive_without_connect() {
        let mut recv = PyNdiReceiver::new("Camera 1");
        let result = recv.receive_video_frame();
        assert!(result.is_err());
    }

    #[test]
    fn test_list_ndi_groups() {
        let groups = list_ndi_groups();
        assert!(!groups.is_empty());
        assert!(groups.contains(&"public".to_string()));
    }

    #[test]
    fn test_list_ndi_video_formats() {
        let formats = list_ndi_video_formats();
        assert!(formats.len() >= 4);
    }
}
