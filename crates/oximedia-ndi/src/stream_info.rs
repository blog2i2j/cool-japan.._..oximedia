//! NDI stream information and capability advertisement for `oximedia-ndi`.
//!
//! Captures the advertised properties of a discovered or connected NDI stream
//! and provides helpers for checking capability compatibility.

#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

// ---------------------------------------------------------------------------
// StreamCapability
// ---------------------------------------------------------------------------

/// An optional feature that an NDI stream may support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamCapability {
    /// Supports tally light signalling.
    Tally,
    /// Supports PTZ camera control.
    Ptz,
    /// Can operate in low-bandwidth (compressed) mode.
    LowBandwidth,
    /// Supports embedded closed captions / metadata.
    Metadata,
    /// Stream carries audio in addition to video.
    Audio,
}

// ---------------------------------------------------------------------------
// StreamDirection
// ---------------------------------------------------------------------------

/// Whether a stream is sending or receiving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamDirection {
    /// This endpoint is producing video/audio frames.
    Send,
    /// This endpoint is consuming video/audio frames.
    Receive,
    /// Bidirectional (e.g. for control channels).
    Both,
}

impl StreamDirection {
    /// Returns `true` when the endpoint can receive data.
    pub fn can_receive(self) -> bool {
        matches!(self, Self::Receive | Self::Both)
    }

    /// Returns `true` when the endpoint can send data.
    pub fn can_send(self) -> bool {
        matches!(self, Self::Send | Self::Both)
    }
}

// ---------------------------------------------------------------------------
// NdiStreamInfo
// ---------------------------------------------------------------------------

/// Complete information about a discovered or connected NDI stream.
#[derive(Debug, Clone)]
pub struct NdiStreamInfo {
    /// Human-readable stream name (e.g. `"Studio Cam 1"`).
    pub name: String,
    /// IP address of the host machine.
    pub host_ip: String,
    /// Port on which the stream is accessible.
    pub port: u16,
    /// Advertised capabilities.
    pub capabilities: Vec<StreamCapability>,
    /// Direction (sender / receiver / both).
    pub direction: StreamDirection,
    /// Advertised video frame rate as a rational (numerator / denominator).
    pub fps_num: u32,
    /// Denominator of the video frame rate rational.
    pub fps_den: u32,
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
}

impl NdiStreamInfo {
    /// Create a new stream info record.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &str,
        host_ip: &str,
        port: u16,
        capabilities: Vec<StreamCapability>,
        direction: StreamDirection,
        fps_num: u32,
        fps_den: u32,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            name: name.to_string(),
            host_ip: host_ip.to_string(),
            port,
            capabilities,
            direction,
            fps_num,
            fps_den,
            width,
            height,
        }
    }

    /// Returns `true` when the stream advertises `cap`.
    pub fn has_capability(&self, cap: StreamCapability) -> bool {
        self.capabilities.contains(&cap)
    }

    /// Frame rate as an `f64`.
    pub fn frame_rate(&self) -> f64 {
        if self.fps_den == 0 {
            return 0.0;
        }
        f64::from(self.fps_num) / f64::from(self.fps_den)
    }

    /// Full address string in the form `"host:port"`.
    pub fn address(&self) -> String {
        format!("{}:{}", self.host_ip, self.port)
    }

    /// Returns `true` when the resolution is at least 1920 × 1080.
    pub fn is_hd_or_better(&self) -> bool {
        self.width >= 1920 && self.height >= 1080
    }
}

// ---------------------------------------------------------------------------
// NdiStreamRegistry
// ---------------------------------------------------------------------------

/// A simple in-memory registry of known NDI streams.
#[derive(Debug, Clone, Default)]
pub struct NdiStreamRegistry {
    streams: Vec<NdiStreamInfo>,
}

impl NdiStreamRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a stream.  If a stream with the same name already exists it is
    /// replaced.
    pub fn register(&mut self, info: NdiStreamInfo) {
        if let Some(pos) = self.streams.iter().position(|s| s.name == info.name) {
            self.streams[pos] = info;
        } else {
            self.streams.push(info);
        }
    }

    /// Remove a stream by name.  Returns `true` when it was present.
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.streams.len();
        self.streams.retain(|s| s.name != name);
        self.streams.len() < before
    }

    /// Find a stream by name.
    pub fn find(&self, name: &str) -> Option<&NdiStreamInfo> {
        self.streams.iter().find(|s| s.name == name)
    }

    /// All streams that advertise the given capability.
    pub fn with_capability(&self, cap: StreamCapability) -> Vec<&NdiStreamInfo> {
        self.streams
            .iter()
            .filter(|s| s.has_capability(cap))
            .collect()
    }

    /// All HD-or-better streams.
    pub fn hd_streams(&self) -> Vec<&NdiStreamInfo> {
        self.streams
            .iter()
            .filter(|s| s.is_hd_or_better())
            .collect()
    }

    /// Total number of registered streams.
    pub fn count(&self) -> usize {
        self.streams.len()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_info(name: &str, w: u32, h: u32, caps: Vec<StreamCapability>) -> NdiStreamInfo {
        NdiStreamInfo::new(
            name,
            "192.168.1.10",
            5960,
            caps,
            StreamDirection::Send,
            60,
            1,
            w,
            h,
        )
    }

    #[test]
    fn test_capability_variants_distinct() {
        let caps = [
            StreamCapability::Tally,
            StreamCapability::Ptz,
            StreamCapability::LowBandwidth,
            StreamCapability::Metadata,
            StreamCapability::Audio,
        ];
        assert_eq!(caps.len(), 5);
    }

    #[test]
    fn test_direction_can_receive_receive() {
        assert!(StreamDirection::Receive.can_receive());
    }

    #[test]
    fn test_direction_can_receive_send_false() {
        assert!(!StreamDirection::Send.can_receive());
    }

    #[test]
    fn test_direction_can_send_send() {
        assert!(StreamDirection::Send.can_send());
    }

    #[test]
    fn test_direction_can_send_both() {
        assert!(StreamDirection::Both.can_send());
        assert!(StreamDirection::Both.can_receive());
    }

    #[test]
    fn test_stream_info_has_capability_true() {
        let info = make_info("cam1", 1920, 1080, vec![StreamCapability::Tally]);
        assert!(info.has_capability(StreamCapability::Tally));
    }

    #[test]
    fn test_stream_info_has_capability_false() {
        let info = make_info("cam1", 1920, 1080, vec![]);
        assert!(!info.has_capability(StreamCapability::Ptz));
    }

    #[test]
    fn test_stream_info_frame_rate() {
        let info = make_info("cam1", 1920, 1080, vec![]);
        assert!((info.frame_rate() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_stream_info_frame_rate_zero_den() {
        let mut info = make_info("cam1", 1920, 1080, vec![]);
        info.fps_den = 0;
        assert_eq!(info.frame_rate(), 0.0);
    }

    #[test]
    fn test_stream_info_address() {
        let info = make_info("cam1", 1920, 1080, vec![]);
        assert_eq!(info.address(), "192.168.1.10:5960");
    }

    #[test]
    fn test_stream_info_is_hd_true() {
        let info = make_info("cam1", 1920, 1080, vec![]);
        assert!(info.is_hd_or_better());
    }

    #[test]
    fn test_stream_info_is_hd_false() {
        let info = make_info("cam1", 1280, 720, vec![]);
        assert!(!info.is_hd_or_better());
    }

    #[test]
    fn test_registry_register_and_find() {
        let mut reg = NdiStreamRegistry::new();
        reg.register(make_info("cam1", 1920, 1080, vec![]));
        assert!(reg.find("cam1").is_some());
    }

    #[test]
    fn test_registry_register_replaces_existing() {
        let mut reg = NdiStreamRegistry::new();
        reg.register(make_info("cam1", 1920, 1080, vec![]));
        reg.register(make_info("cam1", 3840, 2160, vec![]));
        assert_eq!(reg.count(), 1);
        assert_eq!(
            reg.find("cam1").expect("expected item to be found").width,
            3840
        );
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = NdiStreamRegistry::new();
        reg.register(make_info("cam1", 1920, 1080, vec![]));
        assert!(reg.remove("cam1"));
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_registry_remove_missing() {
        let mut reg = NdiStreamRegistry::new();
        assert!(!reg.remove("ghost"));
    }

    #[test]
    fn test_registry_with_capability() {
        let mut reg = NdiStreamRegistry::new();
        reg.register(make_info("cam1", 1920, 1080, vec![StreamCapability::Ptz]));
        reg.register(make_info("cam2", 1920, 1080, vec![]));
        let ptz = reg.with_capability(StreamCapability::Ptz);
        assert_eq!(ptz.len(), 1);
        assert_eq!(ptz[0].name, "cam1");
    }

    #[test]
    fn test_registry_hd_streams() {
        let mut reg = NdiStreamRegistry::new();
        reg.register(make_info("hd", 1920, 1080, vec![]));
        reg.register(make_info("sd", 720, 576, vec![]));
        assert_eq!(reg.hd_streams().len(), 1);
    }
}
