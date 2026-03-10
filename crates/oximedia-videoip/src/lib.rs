//! Professional video-over-IP protocol for `OxiMedia`.
//!
//! This crate provides a patent-free alternative to NDI (Network Device Interface)
//! for professional video streaming over IP networks. It implements:
//!
//! - **Low-latency video/audio transport** over UDP with FEC (Forward Error Correction)
//! - **mDNS/DNS-SD service discovery** for automatic source detection
//! - **Multiple video codecs**: VP9, AV1 (compressed), v210, UYVY (uncompressed)
//! - **Multiple audio formats**: Opus (compressed), PCM (uncompressed)
//! - **Professional features**: Tally lights, PTZ control, timecode, metadata
//! - **Network resilience**: FEC, jitter buffering, packet loss recovery
//! - **Multi-stream support**: Program, preview, alpha channels
//!
//! # Protocol Design
//!
//! The protocol is designed for professional broadcast environments with:
//! - Target latency < 16ms at 60fps (less than 1 frame)
//! - Support for SD, HD, and UHD resolutions
//! - Frame rates: 23.976, 24, 25, 29.97, 30, 50, 59.94, 60 fps
//! - Up to 16 audio channels at 48kHz or 96kHz
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐                           ┌─────────────┐
//! │  VideoIP    │  ─── UDP + FEC ────>      │  VideoIP    │
//! │  Source     │  <── Control Msgs ──      │  Receiver   │
//! └─────────────┘                           └─────────────┘
//!       │                                          │
//!       └──── mDNS Announcement                   │
//!                                                  │
//!                                  mDNS Discovery ─┘
//! ```
//!
//! # Example
//!
//! ## Broadcasting a Video Stream
//!
//! ```ignore
//! use oximedia_videoip::{VideoIpSource, VideoConfig, AudioConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let video_config = VideoConfig::new(1920, 1080, 60.0)?;
//!     let audio_config = AudioConfig::new(48000, 2)?;
//!
//!     let mut source = VideoIpSource::new("Camera 1", video_config, audio_config)?;
//!     source.start_broadcasting().await?;
//!
//!     // Send frames...
//!     let video_frame = get_video_frame();
//!     let audio_samples = get_audio_samples();
//!     source.send_frame(video_frame, audio_samples).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Receiving a Video Stream
//!
//! ```ignore
//! use oximedia_videoip::VideoIpReceiver;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut receiver = VideoIpReceiver::discover("Camera 1").await?;
//!     receiver.start_receiving().await?;
//!
//!     loop {
//!         let (video_frame, audio_samples) = receiver.receive_frame().await?;
//!         // Process frame...
//!     }
//! }
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]

pub mod bandwidth_est;
pub mod bonding;
pub mod codec;
pub mod color_space_conv;
pub mod congestion;
pub mod discovery;
pub mod encryption;
pub mod error;
pub mod fec;
pub mod flow_monitor;
pub mod flow_stats;
pub mod frame_pacing;
pub mod jitter;
pub mod metadata;
pub mod multicast;
pub mod multicast_group;
pub mod ndi_bridge;
pub mod nmos;
pub mod packet;
pub mod packet_loss;
pub mod ptp_boundary;
pub mod ptz;
pub mod quic_transport;
pub mod receiver;
pub mod redundancy;
pub mod rist;
pub mod sdp;
pub mod smpte2110;
pub mod source;
pub mod srt_config;
pub mod stats;
pub mod stream_descriptor;
pub mod stream_health;
pub mod stream_recorder;
pub mod stream_sync;
pub mod tally;
pub mod transport;
pub mod types;
pub mod utils;

// Re-export main types
pub use error::{VideoIpError, VideoIpResult};
pub use receiver::VideoIpReceiver;
pub use source::VideoIpSource;
pub use types::{AudioConfig, AudioFormat, VideoConfig, VideoFormat};
