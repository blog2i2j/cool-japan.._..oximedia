//! ST 2110-compatible SDP (Session Description Protocol) builder.
//!
//! Provides a fluent builder for generating RFC 4566 SDP documents that
//! describe SMPTE ST 2110 media sessions.  Unlike the lower-level
//! [`crate::sdp`] module, which models individual SDP sections and
//! attributes, `Sdp2110Builder` exposes a higher-level API specifically
//! tuned for ST 2110-20 video and ST 2110-30 audio streams.
//!
//! # Example
//!
//! ```rust
//! use oximedia_videoip::sdp_gen::Sdp2110Builder;
//!
//! let sdp = Sdp2110Builder::new()
//!     .video_stream("239.0.0.1", 5004, "29.97")
//!     .audio_stream("239.0.0.2", 5006, 48000, 2)
//!     .build();
//! assert!(sdp.contains("m=video"));
//! assert!(sdp.contains("m=audio"));
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ─── Stream descriptors ──────────────────────────────────────────────────────

/// Description of a single ST 2110-20 video stream.
#[derive(Debug, Clone)]
struct VideoStreamDesc {
    /// Destination multicast or unicast IP.
    dest_ip: String,
    /// Destination UDP port.
    port: u16,
    /// Frame rate string (e.g. `"29.97"`, `"60"`, `"25"`).
    frame_rate: String,
    /// RTP payload type (default 96).
    payload_type: u8,
}

/// Description of a single ST 2110-30 audio stream.
#[derive(Debug, Clone)]
struct AudioStreamDesc {
    /// Destination multicast or unicast IP.
    dest_ip: String,
    /// Destination UDP port.
    port: u16,
    /// Sample rate in Hz (e.g. 48000).
    sample_rate: u32,
    /// Number of audio channels.
    channels: u8,
    /// RTP payload type (default 97).
    payload_type: u8,
}

// ─── Sdp2110Builder ──────────────────────────────────────────────────────────

/// Builder that produces a complete RFC 4566 SDP document for ST 2110 sessions.
///
/// Call [`Sdp2110Builder::video_stream`] and/or [`Sdp2110Builder::audio_stream`]
/// any number of times to add media sections, then call [`Sdp2110Builder::build`]
/// to generate the SDP string.
#[derive(Debug, Default)]
pub struct Sdp2110Builder {
    session_name: String,
    originator: String,
    video_streams: Vec<VideoStreamDesc>,
    audio_streams: Vec<AudioStreamDesc>,
}

impl Sdp2110Builder {
    /// Creates a new builder with default session metadata.
    #[must_use]
    pub fn new() -> Self {
        Self {
            session_name: "OxiMedia ST 2110 Session".into(),
            originator: "oximedia".into(),
            video_streams: Vec::new(),
            audio_streams: Vec::new(),
        }
    }

    /// Sets the session name (`s=` line).
    #[must_use]
    pub fn session_name(mut self, name: impl Into<String>) -> Self {
        self.session_name = name.into();
        self
    }

    /// Adds a **video** stream section (ST 2110-20).
    ///
    /// # Arguments
    ///
    /// * `ip` – destination address (unicast or multicast).
    /// * `port` – destination UDP port.
    /// * `rate` – frame rate string (e.g. `"29.97"`, `"60"`).
    #[must_use]
    pub fn video_stream(mut self, ip: impl Into<String>, port: u16, rate: impl Into<String>) -> Self {
        self.video_streams.push(VideoStreamDesc {
            dest_ip: ip.into(),
            port,
            frame_rate: rate.into(),
            payload_type: 96,
        });
        self
    }

    /// Adds an **audio** stream section (ST 2110-30).
    ///
    /// # Arguments
    ///
    /// * `ip` – destination address.
    /// * `port` – destination UDP port.
    /// * `sample_rate` – audio sample rate in Hz.
    /// * `channels` – number of audio channels.
    #[must_use]
    pub fn audio_stream(
        mut self,
        ip: impl Into<String>,
        port: u16,
        sample_rate: u32,
        channels: u8,
    ) -> Self {
        self.audio_streams.push(AudioStreamDesc {
            dest_ip: ip.into(),
            port,
            sample_rate,
            channels,
            payload_type: 97,
        });
        self
    }

    /// Generates and returns the complete SDP string.
    ///
    /// The output follows RFC 4566 and includes ST 2110-compatible
    /// `a=rtpmap` and `a=fmtp` attributes.
    #[must_use]
    pub fn build(&self) -> String {
        let mut out = String::with_capacity(512);

        // ── Session-level lines ──────────────────────────────────────────────
        out.push_str("v=0\r\n");
        out.push_str(&format!(
            "o={} 0 0 IN IP4 127.0.0.1\r\n",
            self.originator
        ));
        out.push_str(&format!("s={}\r\n", self.session_name));
        out.push_str("t=0 0\r\n");

        // ── Video media sections ─────────────────────────────────────────────
        for v in &self.video_streams {
            out.push_str(&format!(
                "m=video {} RTP/AVP {}\r\n",
                v.port, v.payload_type
            ));
            out.push_str(&format!("c=IN IP4 {}\r\n", v.dest_ip));
            out.push_str(&format!(
                "a=rtpmap:{} raw/90000\r\n",
                v.payload_type
            ));
            out.push_str(&format!(
                "a=fmtp:{} sampling=YCbCr-4:2:2; width=1920; height=1080; exactframerate={}\r\n",
                v.payload_type, v.frame_rate
            ));
            out.push_str("a=mediaclk:direct=0\r\n");
            out.push_str("a=ts-refclk:ptp=IEEE1588-2008\r\n");
        }

        // ── Audio media sections ─────────────────────────────────────────────
        for a in &self.audio_streams {
            out.push_str(&format!(
                "m=audio {} RTP/AVP {}\r\n",
                a.port, a.payload_type
            ));
            out.push_str(&format!("c=IN IP4 {}\r\n", a.dest_ip));
            out.push_str(&format!(
                "a=rtpmap:{} L24/{}/{}\r\n",
                a.payload_type, a.sample_rate, a.channels
            ));
            out.push_str(&format!(
                "a=fmtp:{} channel-order=SMPTE2110.(ST)\r\n",
                a.payload_type
            ));
            out.push_str("a=mediaclk:direct=0\r\n");
            out.push_str("a=ts-refclk:ptp=IEEE1588-2008\r\n");
        }

        out
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_builder_produces_session_lines() {
        let sdp = Sdp2110Builder::new().build();
        assert!(sdp.starts_with("v=0\r\n"));
        assert!(sdp.contains("s=OxiMedia ST 2110 Session\r\n"));
        assert!(sdp.contains("t=0 0\r\n"));
    }

    #[test]
    fn video_stream_present() {
        let sdp = Sdp2110Builder::new()
            .video_stream("239.0.0.1", 5004, "29.97")
            .build();
        assert!(sdp.contains("m=video 5004"));
        assert!(sdp.contains("c=IN IP4 239.0.0.1"));
        assert!(sdp.contains("a=rtpmap:96 raw/90000"));
        assert!(sdp.contains("exactframerate=29.97"));
    }

    #[test]
    fn audio_stream_present() {
        let sdp = Sdp2110Builder::new()
            .audio_stream("239.0.0.2", 5006, 48000, 2)
            .build();
        assert!(sdp.contains("m=audio 5006"));
        assert!(sdp.contains("c=IN IP4 239.0.0.2"));
        assert!(sdp.contains("a=rtpmap:97 L24/48000/2"));
    }

    #[test]
    fn combined_video_and_audio() {
        let sdp = Sdp2110Builder::new()
            .video_stream("239.0.0.1", 5004, "60")
            .audio_stream("239.0.0.2", 5006, 48000, 8)
            .build();
        assert!(sdp.contains("m=video"));
        assert!(sdp.contains("m=audio"));
        assert!(sdp.contains("L24/48000/8"));
    }

    #[test]
    fn custom_session_name() {
        let sdp = Sdp2110Builder::new()
            .session_name("Studio Camera 1")
            .build();
        assert!(sdp.contains("s=Studio Camera 1\r\n"));
    }

    #[test]
    fn multiple_video_streams() {
        let sdp = Sdp2110Builder::new()
            .video_stream("239.0.0.1", 5004, "30")
            .video_stream("239.0.0.3", 5008, "25")
            .build();
        let count = sdp.matches("m=video").count();
        assert_eq!(count, 2);
    }

    #[test]
    fn ptp_reference_clock_present() {
        let sdp = Sdp2110Builder::new()
            .video_stream("239.0.0.1", 5004, "30")
            .build();
        assert!(sdp.contains("a=ts-refclk:ptp=IEEE1588-2008"));
    }
}
