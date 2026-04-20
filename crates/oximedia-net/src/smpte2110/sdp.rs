//! SDP (Session Description Protocol) for SMPTE ST 2110.
//!
//! This module provides SDP generation and parsing for SMPTE ST 2110 streams.
//! SDP is used to describe the media streams, including format parameters,
//! connection information, and timing details.

use crate::error::{NetError, NetResult};
use crate::smpte2110::ancillary::AncillaryConfig;
use crate::smpte2110::audio::AudioConfig;
use crate::smpte2110::video::VideoConfig;
use std::fmt;
use std::net::IpAddr;

/// Media type for SDP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    /// Video media.
    Video,
    /// Audio media.
    Audio,
    /// Ancillary data.
    Ancillary,
}

impl MediaType {
    /// Gets the media type string for SDP.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Ancillary => "video", // Ancillary uses video media type
        }
    }
}

/// SDP session description.
#[derive(Debug, Clone)]
pub struct SdpSession {
    /// Session name.
    pub session_name: String,
    /// Session information.
    pub session_info: Option<String>,
    /// Connection information (originator).
    pub origin_address: IpAddr,
    /// Session ID.
    pub session_id: u64,
    /// Session version.
    pub session_version: u64,
    /// Media descriptions.
    pub media: Vec<SdpMedia>,
}

impl SdpSession {
    /// Creates a new SDP session.
    #[must_use]
    pub fn new(session_name: impl Into<String>, origin_address: IpAddr) -> Self {
        Self {
            session_name: session_name.into(),
            session_info: None,
            origin_address,
            session_id: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(std::time::Duration::ZERO)
                .as_secs(),
            session_version: 1,
            media: Vec::new(),
        }
    }

    /// Adds a video media description.
    pub fn add_video_media(&mut self, destination: IpAddr, port: u16, config: &VideoConfig) {
        let mut media = SdpMedia {
            media_type: MediaType::Video,
            port,
            protocol: "RTP/AVP".to_string(),
            payload_type: super::video::RTP_PAYLOAD_TYPE_VIDEO,
            connection_address: destination,
            attributes: Vec::new(),
        };

        // Format-specific parameters
        let sampling = config.pixel_format.sampling();
        let width = config.width;
        let height = config.height;
        let exactframerate = format!(
            "{}/{}",
            config.frame_rate.numerator, config.frame_rate.denominator
        );
        let depth = config.pixel_format.bit_depth();

        let fmtp = format!(
            "a=fmtp:{} sampling={}; width={}; height={}; exactframerate={}; depth={}; colorimetry=BT709; PM=2110GPM; SSN=ST2110-20:2017; TP=2110TPN",
            media.payload_type,
            sampling,
            width,
            height,
            exactframerate,
            depth
        );

        media.attributes.push(fmtp);

        // RTP map
        let rtpmap = format!("a=rtpmap:{} raw/90000", media.payload_type);
        media.attributes.push(rtpmap);

        // Media clocking
        media.attributes.push("a=mediaclk:direct=0".to_string());

        // Timestamp reference
        media
            .attributes
            .push("a=ts-refclk:ptp=IEEE1588-2008:00-00-00-00-00-00-00-00:0".to_string());

        self.media.push(media);
    }

    /// Adds an audio media description.
    pub fn add_audio_media(&mut self, destination: IpAddr, port: u16, config: &AudioConfig) {
        let mut media = SdpMedia {
            media_type: MediaType::Audio,
            port,
            protocol: "RTP/AVP".to_string(),
            payload_type: super::audio::RTP_PAYLOAD_TYPE_AUDIO,
            connection_address: destination,
            attributes: Vec::new(),
        };

        // Format-specific parameters
        let encoding = config.format.sdp_format(config.bit_depth);
        let sample_rate = config.sample_rate.as_u32();
        let channels = config.channels;
        let ptime = config.packet_time_us / 1000; // Convert to ms

        let fmtp = format!("a=fmtp:{} channel-order=SMPTE2110.(ST)", media.payload_type);
        media.attributes.push(fmtp);

        // RTP map
        let rtpmap = format!(
            "a=rtpmap:{} {}/{}{}",
            media.payload_type,
            encoding,
            sample_rate,
            if channels > 1 {
                format!("/{}", channels)
            } else {
                String::new()
            }
        );
        media.attributes.push(rtpmap);

        // Packet time
        media.attributes.push(format!("a=ptime:{}", ptime));

        // Media clocking
        media.attributes.push("a=mediaclk:direct=0".to_string());

        // Timestamp reference
        media
            .attributes
            .push("a=ts-refclk:ptp=IEEE1588-2008:00-00-00-00-00-00-00-00:0".to_string());

        self.media.push(media);
    }

    /// Adds an ancillary data media description.
    pub fn add_ancillary_media(
        &mut self,
        destination: IpAddr,
        port: u16,
        _config: &AncillaryConfig,
    ) {
        let mut media = SdpMedia {
            media_type: MediaType::Ancillary,
            port,
            protocol: "RTP/AVP".to_string(),
            payload_type: super::ancillary::RTP_PAYLOAD_TYPE_ANC,
            connection_address: destination,
            attributes: Vec::new(),
        };

        // Format-specific parameters
        let fmtp = format!("a=fmtp:{} VPID_Code=133", media.payload_type);
        media.attributes.push(fmtp);

        // RTP map
        let rtpmap = format!("a=rtpmap:{} smpte291/90000", media.payload_type);
        media.attributes.push(rtpmap);

        // Media clocking
        media.attributes.push("a=mediaclk:direct=0".to_string());

        // Timestamp reference
        media
            .attributes
            .push("a=ts-refclk:ptp=IEEE1588-2008:00-00-00-00-00-00-00-00:0".to_string());

        self.media.push(media);
    }

    /// Parses an SDP session from string.
    pub fn parse(sdp_str: &str) -> NetResult<Self> {
        let mut session_name = String::new();
        let mut origin_address = None;
        let mut session_id = 0;
        let mut session_version = 1;
        let mut media = Vec::new();
        let mut current_media: Option<SdpMedia> = None;

        for line in sdp_str.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if line.len() < 2 || !line.starts_with(|c: char| c.is_ascii_alphabetic()) {
                continue;
            }

            let field_type = &line[0..1];
            let value = if line.len() > 2 && line.chars().nth(1) == Some('=') {
                &line[2..]
            } else {
                continue;
            };

            match field_type {
                "v" => {
                    // Version (should be 0)
                    if value != "0" {
                        return Err(NetError::protocol(format!(
                            "Unsupported SDP version: {}",
                            value
                        )));
                    }
                }
                "o" => {
                    // Origin
                    let parts: Vec<&str> = value.split_whitespace().collect();
                    if parts.len() >= 6 {
                        session_id = parts[1].parse().unwrap_or(0);
                        session_version = parts[2].parse().unwrap_or(1);
                        origin_address = Some(parts[5].parse().map_err(|_| {
                            NetError::protocol(format!("Invalid origin address: {}", parts[5]))
                        })?);
                    }
                }
                "s" => {
                    // Session name
                    session_name = value.to_string();
                }
                "m" => {
                    // Media description
                    if let Some(m) = current_media.take() {
                        media.push(m);
                    }

                    let parts: Vec<&str> = value.split_whitespace().collect();
                    if parts.len() >= 4 {
                        let media_type = match parts[0] {
                            "video" => MediaType::Video,
                            "audio" => MediaType::Audio,
                            _ => MediaType::Video,
                        };

                        let port = parts[1].parse().unwrap_or(0);
                        let protocol = parts[2].to_string();
                        let payload_type = parts[3].parse().unwrap_or(96);

                        current_media = Some(SdpMedia {
                            media_type,
                            port,
                            protocol,
                            payload_type,
                            connection_address: origin_address
                                .unwrap_or(IpAddr::V4(std::net::Ipv4Addr::new(0, 0, 0, 0))),
                            attributes: Vec::new(),
                        });
                    }
                }
                "c" => {
                    // Connection information
                    let parts: Vec<&str> = value.split_whitespace().collect();
                    if parts.len() >= 3 {
                        if let Ok(addr) = parts[2].parse() {
                            if let Some(m) = &mut current_media {
                                m.connection_address = addr;
                            }
                        }
                    }
                }
                "a" => {
                    // Attribute
                    if let Some(m) = &mut current_media {
                        m.attributes.push(format!("a={}", value));
                    }
                }
                _ => {}
            }
        }

        // Add last media
        if let Some(m) = current_media {
            media.push(m);
        }

        if session_name.is_empty() {
            session_name = "SMPTE ST 2110 Session".to_string();
        }

        let origin_addr =
            origin_address.ok_or_else(|| NetError::protocol("Missing origin address"))?;

        Ok(Self {
            session_name,
            session_info: None,
            origin_address: origin_addr,
            session_id,
            session_version,
            media,
        })
    }
}

impl fmt::Display for SdpSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Version
        writeln!(f, "v=0")?;

        // Origin
        writeln!(
            f,
            "o=- {} {} IN IP4 {}",
            self.session_id, self.session_version, self.origin_address
        )?;

        // Session name
        writeln!(f, "s={}", self.session_name)?;

        // Session information
        if let Some(info) = &self.session_info {
            writeln!(f, "i={}", info)?;
        }

        // Time (permanent session)
        writeln!(f, "t=0 0")?;

        // Media descriptions
        for media in &self.media {
            write!(f, "{}", media)?;
        }

        Ok(())
    }
}

/// SDP media description.
#[derive(Debug, Clone)]
pub struct SdpMedia {
    /// Media type.
    pub media_type: MediaType,
    /// Port number.
    pub port: u16,
    /// Protocol (e.g., "RTP/AVP").
    pub protocol: String,
    /// Payload type.
    pub payload_type: u8,
    /// Connection address.
    pub connection_address: IpAddr,
    /// Attributes.
    pub attributes: Vec<String>,
}

impl fmt::Display for SdpMedia {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Media line
        writeln!(
            f,
            "m={} {} {} {}",
            self.media_type.as_str(),
            self.port,
            self.protocol,
            self.payload_type
        )?;

        // Connection line
        writeln!(f, "c=IN IP4 {}", self.connection_address)?;

        // Attributes
        for attr in &self.attributes {
            writeln!(f, "{}", attr)?;
        }

        Ok(())
    }
}

/// SDP builder for fluent API.
pub struct SdpBuilder {
    session: SdpSession,
}

impl SdpBuilder {
    /// Creates a new SDP builder.
    #[must_use]
    pub fn new(session_name: impl Into<String>, origin_address: IpAddr) -> Self {
        Self {
            session: SdpSession::new(session_name, origin_address),
        }
    }

    /// Sets the session information.
    #[must_use]
    pub fn session_info(mut self, info: impl Into<String>) -> Self {
        self.session.session_info = Some(info.into());
        self
    }

    /// Adds a video media stream.
    #[must_use]
    pub fn video(mut self, destination: IpAddr, port: u16, config: &VideoConfig) -> Self {
        self.session.add_video_media(destination, port, config);
        self
    }

    /// Adds an audio media stream.
    #[must_use]
    pub fn audio(mut self, destination: IpAddr, port: u16, config: &AudioConfig) -> Self {
        self.session.add_audio_media(destination, port, config);
        self
    }

    /// Adds an ancillary data stream.
    #[must_use]
    pub fn ancillary(mut self, destination: IpAddr, port: u16, config: &AncillaryConfig) -> Self {
        self.session.add_ancillary_media(destination, port, config);
        self
    }

    /// Builds the SDP session.
    #[must_use]
    pub fn build(self) -> SdpSession {
        self.session
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smpte2110::audio::AudioSampleRate;
    use crate::smpte2110::timing::FrameRate;
    use crate::smpte2110::video::PixelFormat;
    use std::net::Ipv4Addr;

    #[test]
    fn test_sdp_session_creation() {
        let origin = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));
        let session = SdpSession::new("Test Session", origin);

        assert_eq!(session.session_name, "Test Session");
        assert_eq!(session.origin_address, origin);
    }

    #[test]
    fn test_sdp_video_media() {
        let origin = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));
        let dest = IpAddr::V4(Ipv4Addr::new(239, 0, 0, 1));

        let video_config = VideoConfig {
            width: 1920,
            height: 1080,
            frame_rate: FrameRate::FPS_25,
            pixel_format: PixelFormat::YCbCr422_10bit,
            ..Default::default()
        };

        let mut session = SdpSession::new("Video Test", origin);
        session.add_video_media(dest, 5004, &video_config);

        assert_eq!(session.media.len(), 1);
        assert_eq!(session.media[0].media_type, MediaType::Video);
        assert_eq!(session.media[0].port, 5004);
    }

    #[test]
    fn test_sdp_audio_media() {
        let origin = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));
        let dest = IpAddr::V4(Ipv4Addr::new(239, 0, 0, 2));

        let audio_config = AudioConfig {
            sample_rate: AudioSampleRate::Rate48kHz,
            bit_depth: 24,
            channels: 2,
            ..Default::default()
        };

        let mut session = SdpSession::new("Audio Test", origin);
        session.add_audio_media(dest, 5006, &audio_config);

        assert_eq!(session.media.len(), 1);
        assert_eq!(session.media[0].media_type, MediaType::Audio);
        assert_eq!(session.media[0].port, 5006);
    }

    #[test]
    fn test_sdp_to_string() {
        let origin = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));
        let dest = IpAddr::V4(Ipv4Addr::new(239, 0, 0, 1));

        let video_config = VideoConfig {
            width: 1920,
            height: 1080,
            frame_rate: FrameRate::FPS_25,
            pixel_format: PixelFormat::YCbCr422_10bit,
            ..Default::default()
        };

        let mut session = SdpSession::new("Test", origin);
        session.add_video_media(dest, 5004, &video_config);

        let sdp_str = session.to_string();

        assert!(sdp_str.contains("v=0"));
        assert!(sdp_str.contains("s=Test"));
        assert!(sdp_str.contains("m=video"));
        assert!(sdp_str.contains("5004"));
    }

    #[test]
    fn test_sdp_parse() {
        let sdp_str = r#"v=0
o=- 123456 1 IN IP4 192.168.1.1
s=Test Session
t=0 0
m=video 5004 RTP/AVP 96
c=IN IP4 239.0.0.1
a=rtpmap:96 raw/90000
"#;

        let session = SdpSession::parse(sdp_str).expect("should succeed in test");

        assert_eq!(session.session_name, "Test Session");
        assert_eq!(session.media.len(), 1);
        assert_eq!(session.media[0].media_type, MediaType::Video);
        assert_eq!(session.media[0].port, 5004);
    }

    #[test]
    fn test_sdp_builder() {
        let origin = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));
        let dest = IpAddr::V4(Ipv4Addr::new(239, 0, 0, 1));

        let video_config = VideoConfig {
            width: 1920,
            height: 1080,
            frame_rate: FrameRate::FPS_25,
            pixel_format: PixelFormat::YCbCr422_10bit,
            ..Default::default()
        };

        let session = SdpBuilder::new("Builder Test", origin)
            .session_info("Test information")
            .video(dest, 5004, &video_config)
            .build();

        assert_eq!(session.session_name, "Builder Test");
        assert_eq!(session.session_info, Some("Test information".to_string()));
        assert_eq!(session.media.len(), 1);
    }

    #[test]
    fn test_media_type_str() {
        assert_eq!(MediaType::Video.as_str(), "video");
        assert_eq!(MediaType::Audio.as_str(), "audio");
        assert_eq!(MediaType::Ancillary.as_str(), "video");
    }
}
