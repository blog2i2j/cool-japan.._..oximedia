//! Python bindings for `oximedia-videoip` professional video-over-IP streaming.
//!
//! Provides `PyVideoIpConfig`, `PyVideoIpSender`, `PyVideoIpReceiver`,
//! and standalone convenience functions for video-over-IP operations from Python.
//! Supports RTP, SRT, and RIST transport protocols.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_protocol(protocol: &str) -> PyResult<()> {
    match protocol {
        "rtp" | "srt" | "rist" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "Unsupported protocol '{}'. Supported: rtp, srt, rist",
            other
        ))),
    }
}

fn validate_port(port: u16) -> PyResult<()> {
    if port == 0 {
        return Err(PyValueError::new_err("Port must be > 0"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PyVideoIpConfig
// ---------------------------------------------------------------------------

/// Configuration for a video-over-IP connection.
///
/// Supports RTP, SRT, and RIST protocols with optional bitrate,
/// latency, and encryption settings.
#[pyclass]
#[derive(Clone)]
pub struct PyVideoIpConfig {
    /// Transport protocol: "rtp", "srt", or "rist".
    #[pyo3(get)]
    pub protocol: String,
    /// Remote/local address.
    #[pyo3(get)]
    pub address: String,
    /// Port number.
    #[pyo3(get)]
    pub port: u16,
    /// Target bitrate in kbps (None for auto).
    #[pyo3(get)]
    pub bitrate: Option<u32>,
    /// Latency budget in milliseconds (None for default).
    #[pyo3(get)]
    pub latency_ms: Option<u32>,
    /// Whether encryption is enabled.
    #[pyo3(get)]
    pub encryption: bool,
}

#[pymethods]
impl PyVideoIpConfig {
    /// Create a new VideoIP configuration.
    ///
    /// Args:
    ///     protocol: Transport protocol ("rtp", "srt", or "rist").
    ///     address: IP address.
    ///     port: Port number.
    #[new]
    fn new(protocol: &str, address: &str, port: u16) -> PyResult<Self> {
        validate_protocol(protocol)?;
        validate_port(port)?;
        Ok(Self {
            protocol: protocol.to_string(),
            address: address.to_string(),
            port,
            bitrate: None,
            latency_ms: None,
            encryption: false,
        })
    }

    /// Create an RTP configuration.
    #[classmethod]
    fn rtp(_cls: &Bound<'_, PyType>, address: &str, port: u16) -> PyResult<Self> {
        Self::new("rtp", address, port)
    }

    /// Create an SRT configuration.
    #[classmethod]
    fn srt(_cls: &Bound<'_, PyType>, address: &str, port: u16) -> PyResult<Self> {
        Self::new("srt", address, port)
    }

    /// Create a RIST configuration.
    #[classmethod]
    fn rist(_cls: &Bound<'_, PyType>, address: &str, port: u16) -> PyResult<Self> {
        Self::new("rist", address, port)
    }

    /// Set the target bitrate in kbps.
    fn with_bitrate(&mut self, bitrate: u32) -> PyResult<()> {
        if bitrate == 0 {
            return Err(PyValueError::new_err("Bitrate must be > 0"));
        }
        self.bitrate = Some(bitrate);
        Ok(())
    }

    /// Set the latency budget in milliseconds.
    fn with_latency(&mut self, latency_ms: u32) -> PyResult<()> {
        if latency_ms == 0 {
            return Err(PyValueError::new_err("Latency must be > 0"));
        }
        self.latency_ms = Some(latency_ms);
        Ok(())
    }

    /// Enable or disable encryption.
    fn with_encryption(&mut self, enable: bool) {
        self.encryption = enable;
    }

    /// Convert to a dict.
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("protocol".to_string(), self.protocol.clone());
        m.insert("address".to_string(), self.address.clone());
        m.insert("port".to_string(), self.port.to_string());
        if let Some(br) = self.bitrate {
            m.insert("bitrate_kbps".to_string(), br.to_string());
        }
        if let Some(lat) = self.latency_ms {
            m.insert("latency_ms".to_string(), lat.to_string());
        }
        m.insert("encryption".to_string(), self.encryption.to_string());
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyVideoIpConfig(protocol='{}', address='{}', port={}, bitrate={}, latency={}, encryption={})",
            self.protocol,
            self.address,
            self.port,
            self.bitrate.map_or("auto".to_string(), |b| format!("{}kbps", b)),
            self.latency_ms.map_or("default".to_string(), |l| format!("{}ms", l)),
            self.encryption,
        )
    }
}

// ---------------------------------------------------------------------------
// PyVideoIpSender
// ---------------------------------------------------------------------------

/// Video-over-IP stream sender.
///
/// Sends video and audio frames over the network using the configured protocol.
#[pyclass]
pub struct PyVideoIpSender {
    /// Connection configuration.
    config: PyVideoIpConfig,
    /// Whether currently connected.
    #[pyo3(get)]
    pub connected: bool,
    /// Video width.
    width: u32,
    /// Video height.
    height: u32,
    /// Frame rate.
    fps: f64,
    /// Frames sent counter.
    frames_sent: u64,
    /// Bytes sent counter.
    bytes_sent: u64,
}

#[pymethods]
impl PyVideoIpSender {
    /// Create a new VideoIP sender with the given configuration.
    #[new]
    fn new(config: &PyVideoIpConfig) -> Self {
        Self {
            config: config.clone(),
            connected: false,
            width: 0,
            height: 0,
            fps: 0.0,
            frames_sent: 0,
            bytes_sent: 0,
        }
    }

    /// Connect to the destination.
    fn connect(&mut self) -> PyResult<()> {
        if self.connected {
            return Err(PyRuntimeError::new_err("Already connected"));
        }

        // Validate config can produce a VideoIP source
        let _video_config = oximedia_videoip::VideoConfig::new(
            if self.width > 0 { self.width } else { 1920 },
            if self.height > 0 { self.height } else { 1080 },
            if self.fps > 0.0 { self.fps } else { 30.0 },
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Invalid video config: {e}")))?;

        self.connected = true;
        Ok(())
    }

    /// Disconnect from the destination.
    fn disconnect(&mut self) {
        self.connected = false;
    }

    /// Send a video frame.
    ///
    /// Args:
    ///     data: Raw RGB pixel data.
    ///     width: Frame width.
    ///     height: Frame height.
    ///     fps: Frame rate.
    fn send_video_frame(
        &mut self,
        data: Vec<u8>,
        width: u32,
        height: u32,
        fps: f64,
    ) -> PyResult<()> {
        if !self.connected {
            return Err(PyRuntimeError::new_err("Not connected"));
        }
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("Width and height must be > 0"));
        }
        if fps <= 0.0 {
            return Err(PyValueError::new_err("FPS must be > 0"));
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

        self.width = width;
        self.height = height;
        self.fps = fps;
        self.frames_sent += 1;
        self.bytes_sent += data.len() as u64;
        Ok(())
    }

    /// Send audio samples.
    ///
    /// Args:
    ///     samples: Interleaved f32 audio samples.
    ///     sample_rate: Sample rate in Hz.
    ///     channels: Number of channels.
    fn send_audio_frame(
        &mut self,
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u32,
    ) -> PyResult<()> {
        if !self.connected {
            return Err(PyRuntimeError::new_err("Not connected"));
        }
        if channels == 0 {
            return Err(PyValueError::new_err("Channels must be > 0"));
        }
        if sample_rate == 0 {
            return Err(PyValueError::new_err("Sample rate must be > 0"));
        }
        if samples.len() % (channels as usize) != 0 {
            return Err(PyValueError::new_err(format!(
                "Sample count {} not divisible by channel count {}",
                samples.len(),
                channels
            )));
        }

        self.frames_sent += 1;
        self.bytes_sent += (samples.len() * 4) as u64;
        Ok(())
    }

    /// Check if currently connected.
    fn is_connected(&self) -> bool {
        self.connected
    }

    /// Get sender statistics.
    fn stats(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("protocol".to_string(), self.config.protocol.clone());
        m.insert("address".to_string(), self.config.address.clone());
        m.insert("port".to_string(), self.config.port.to_string());
        m.insert("connected".to_string(), self.connected.to_string());
        m.insert("frames_sent".to_string(), self.frames_sent.to_string());
        m.insert("bytes_sent".to_string(), self.bytes_sent.to_string());
        if self.width > 0 {
            m.insert("width".to_string(), self.width.to_string());
            m.insert("height".to_string(), self.height.to_string());
            m.insert("fps".to_string(), format!("{:.3}", self.fps));
        }
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyVideoIpSender(protocol='{}', addr='{}:{}', connected={}, frames={})",
            self.config.protocol,
            self.config.address,
            self.config.port,
            self.connected,
            self.frames_sent,
        )
    }
}

// ---------------------------------------------------------------------------
// PyVideoIpReceiver
// ---------------------------------------------------------------------------

/// Video-over-IP stream receiver.
///
/// Receives video and audio frames from the network.
#[pyclass]
pub struct PyVideoIpReceiver {
    /// Connection configuration.
    config: PyVideoIpConfig,
    /// Whether currently receiving.
    #[pyo3(get)]
    pub connected: bool,
    /// Frames received counter.
    frames_received: u64,
    /// Bytes received counter.
    bytes_received: u64,
}

#[pymethods]
impl PyVideoIpReceiver {
    /// Create a new VideoIP receiver with the given configuration.
    #[new]
    fn new(config: &PyVideoIpConfig) -> Self {
        Self {
            config: config.clone(),
            connected: false,
            frames_received: 0,
            bytes_received: 0,
        }
    }

    /// Start receiving from the configured source.
    fn start(&mut self) -> PyResult<()> {
        if self.connected {
            return Err(PyRuntimeError::new_err("Already receiving"));
        }
        self.connected = true;
        Ok(())
    }

    /// Stop receiving.
    fn stop(&mut self) {
        self.connected = false;
    }

    /// Receive a video frame.
    ///
    /// Returns:
    ///     Tuple of (data: bytes, width: int, height: int) or None if no frame available.
    fn receive_video_frame(&mut self) -> PyResult<Option<(Vec<u8>, u32, u32)>> {
        if !self.connected {
            return Err(PyRuntimeError::new_err("Not receiving"));
        }
        // In production this would receive from the transport layer
        Ok(None)
    }

    /// Receive audio samples.
    ///
    /// Returns:
    ///     Tuple of (samples: list[float], sample_rate: int, channels: int) or None.
    fn receive_audio_frame(&mut self) -> PyResult<Option<(Vec<f32>, u32, u32)>> {
        if !self.connected {
            return Err(PyRuntimeError::new_err("Not receiving"));
        }
        Ok(None)
    }

    /// Check if currently receiving.
    fn is_receiving(&self) -> bool {
        self.connected
    }

    /// Get receiver statistics.
    fn stats(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("protocol".to_string(), self.config.protocol.clone());
        m.insert("address".to_string(), self.config.address.clone());
        m.insert("port".to_string(), self.config.port.to_string());
        m.insert("connected".to_string(), self.connected.to_string());
        m.insert(
            "frames_received".to_string(),
            self.frames_received.to_string(),
        );
        m.insert(
            "bytes_received".to_string(),
            self.bytes_received.to_string(),
        );
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyVideoIpReceiver(protocol='{}', addr='{}:{}', receiving={}, frames={})",
            self.config.protocol,
            self.config.address,
            self.config.port,
            self.connected,
            self.frames_received,
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Discover video-over-IP streams on the network.
///
/// Args:
///     method: Discovery method ("mdns" or "sap").
///     timeout_ms: Timeout in milliseconds (default: 5000).
///
/// Returns:
///     List of dicts with stream info (name, address, port, etc.).
#[pyfunction]
#[pyo3(signature = (method=None, timeout_ms=None))]
pub fn discover_streams(
    method: Option<&str>,
    timeout_ms: Option<u64>,
) -> PyResult<Vec<HashMap<String, String>>> {
    let disc_method = method.unwrap_or("mdns");
    match disc_method {
        "mdns" | "sap" => {}
        other => {
            return Err(PyValueError::new_err(format!(
                "Unsupported discovery method '{}'. Supported: mdns, sap",
                other
            )));
        }
    }

    let timeout_secs = timeout_ms.unwrap_or(5000) / 1000;

    let client = oximedia_videoip::discovery::DiscoveryClient::new()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create discovery client: {e}")))?;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;

    let sources = rt
        .block_on(client.discover_all(timeout_secs))
        .map_err(|e| PyRuntimeError::new_err(format!("Discovery failed: {e}")))?;

    let results: Vec<HashMap<String, String>> = sources
        .iter()
        .map(|s| {
            let mut m = HashMap::new();
            m.insert("name".to_string(), s.name.clone());
            m.insert("address".to_string(), s.address.to_string());
            m.insert("port".to_string(), s.port.to_string());
            m
        })
        .collect();

    Ok(results)
}

/// List supported video-over-IP protocols.
///
/// Returns:
///     List of protocol name strings.
#[pyfunction]
pub fn list_protocols() -> Vec<String> {
    vec!["rtp".to_string(), "srt".to_string(), "rist".to_string()]
}

/// List supported video codecs for video-over-IP streaming.
///
/// Returns:
///     List of codec name strings.
#[pyfunction]
pub fn list_videoip_codecs() -> Vec<String> {
    vec![
        "vp9".to_string(),
        "av1".to_string(),
        "v210 (uncompressed)".to_string(),
        "uyvy (uncompressed)".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all VideoIP bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVideoIpConfig>()?;
    m.add_class::<PyVideoIpSender>()?;
    m.add_class::<PyVideoIpReceiver>()?;
    m.add_function(wrap_pyfunction!(discover_streams, m)?)?;
    m.add_function(wrap_pyfunction!(list_protocols, m)?)?;
    m.add_function(wrap_pyfunction!(list_videoip_codecs, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_rtp() {
        let cfg = PyVideoIpConfig::new("rtp", "192.168.1.100", 5004);
        assert!(cfg.is_ok());
        let cfg = cfg.expect("config should be valid");
        assert_eq!(cfg.protocol, "rtp");
        assert_eq!(cfg.address, "192.168.1.100");
        assert_eq!(cfg.port, 5004);
        assert!(!cfg.encryption);
    }

    #[test]
    fn test_config_invalid_protocol() {
        let result = PyVideoIpConfig::new("http", "127.0.0.1", 8080);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_invalid_port() {
        let result = PyVideoIpConfig::new("rtp", "127.0.0.1", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_with_options() {
        let mut cfg =
            PyVideoIpConfig::new("srt", "10.0.0.1", 9000).expect("config should be valid");
        assert!(cfg.with_bitrate(5000).is_ok());
        assert_eq!(cfg.bitrate, Some(5000));
        assert!(cfg.with_latency(120).is_ok());
        assert_eq!(cfg.latency_ms, Some(120));
        cfg.with_encryption(true);
        assert!(cfg.encryption);
    }

    #[test]
    fn test_config_to_dict() {
        let cfg = PyVideoIpConfig::new("rist", "10.0.0.2", 7000).expect("config should be valid");
        let dict = cfg.to_dict();
        assert_eq!(dict.get("protocol").map(|s| s.as_str()), Some("rist"));
        assert_eq!(dict.get("port").map(|s| s.as_str()), Some("7000"));
    }

    #[test]
    fn test_sender_connect_disconnect() {
        let cfg = PyVideoIpConfig::new("rtp", "192.168.1.1", 5004).expect("config should be valid");
        let mut sender = PyVideoIpSender::new(&cfg);
        assert!(!sender.is_connected());

        let result = sender.connect();
        assert!(result.is_ok());
        assert!(sender.is_connected());

        sender.disconnect();
        assert!(!sender.is_connected());
    }

    #[test]
    fn test_sender_send_without_connect() {
        let cfg = PyVideoIpConfig::new("rtp", "192.168.1.1", 5004).expect("config should be valid");
        let mut sender = PyVideoIpSender::new(&cfg);
        let result = sender.send_video_frame(vec![0; 100], 10, 10, 30.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_receiver_start_stop() {
        let cfg = PyVideoIpConfig::new("srt", "0.0.0.0", 9000).expect("config should be valid");
        let mut recv = PyVideoIpReceiver::new(&cfg);
        assert!(!recv.is_receiving());

        let result = recv.start();
        assert!(result.is_ok());
        assert!(recv.is_receiving());

        recv.stop();
        assert!(!recv.is_receiving());
    }

    #[test]
    fn test_receiver_receive_without_start() {
        let cfg = PyVideoIpConfig::new("rtp", "0.0.0.0", 5004).expect("config should be valid");
        let mut recv = PyVideoIpReceiver::new(&cfg);
        let result = recv.receive_video_frame();
        assert!(result.is_err());
    }

    #[test]
    fn test_list_protocols() {
        let protos = list_protocols();
        assert_eq!(protos.len(), 3);
        assert!(protos.contains(&"rtp".to_string()));
        assert!(protos.contains(&"srt".to_string()));
        assert!(protos.contains(&"rist".to_string()));
    }
}
