//! Video-over-IP receiver for receiving video/audio streams.

use crate::codec::{create_audio_decoder, create_video_decoder, AudioSamples, VideoFrame};
use crate::discovery::DiscoveryClient;
use crate::error::{VideoIpError, VideoIpResult};
use crate::fec::FecDecoder;
use crate::jitter::JitterBuffer;
use crate::metadata::MetadataPacket;
use crate::packet::{Packet, PacketFlags};
use crate::ptz::PtzMessage;
use crate::stats::StatsTracker;
use crate::tally::TallyMessage;
use crate::transport::UdpTransport;
use crate::types::{AudioCodec, VideoCodec};
use bytes::{Bytes, BytesMut};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::timeout;

/// Maximum time to wait for a complete frame (milliseconds).
const FRAME_TIMEOUT_MS: u64 = 100;

/// Video-over-IP receiver for receiving streams.
#[allow(dead_code)]
pub struct VideoIpReceiver {
    /// UDP transport.
    transport: UdpTransport,
    /// Source address.
    source_addr: Option<SocketAddr>,
    /// Video decoder.
    video_decoder: Box<dyn crate::codec::VideoDecoder>,
    /// Audio decoder.
    audio_decoder: Box<dyn crate::codec::AudioDecoder>,
    /// FEC decoder.
    fec_decoder: Option<FecDecoder>,
    /// Jitter buffer.
    jitter_buffer: JitterBuffer,
    /// Statistics tracker.
    stats: StatsTracker,
    /// Last received sequence number.
    last_sequence: Option<u16>,
    /// Incomplete frames being assembled.
    frame_assembly: HashMap<u64, FrameAssembly>,
    /// Control message sender.
    control_tx: mpsc::Sender<ControlEvent>,
    /// Control message receiver (for external access).
    control_rx: Arc<RwLock<mpsc::Receiver<ControlEvent>>>,
}

/// Frame assembly state for multi-packet frames.
#[allow(dead_code)]
struct FrameAssembly {
    /// Frame chunks.
    chunks: Vec<Option<Bytes>>,
    /// Total expected chunks.
    total_chunks: usize,
    /// Timestamp of first packet.
    timestamp: u64,
    /// Time when assembly started.
    start_time: Instant,
    /// Whether this is a keyframe.
    _is_keyframe: bool,
}

/// Control events from the receiver.
#[derive(Debug, Clone)]
pub enum ControlEvent {
    /// PTZ message received.
    Ptz(PtzMessage),
    /// Tally message received.
    Tally(TallyMessage),
    /// Metadata received.
    Metadata(MetadataPacket),
}

impl VideoIpReceiver {
    /// Creates a new video-over-IP receiver.
    ///
    /// # Errors
    ///
    /// Returns an error if the receiver cannot be created.
    pub async fn new(video_codec: VideoCodec, audio_codec: AudioCodec) -> VideoIpResult<Self> {
        let bind_addr = "0.0.0.0:0"
            .parse()
            .map_err(|e: std::net::AddrParseError| VideoIpError::Transport(e.to_string()))?;
        let transport = UdpTransport::bind(bind_addr).await?;

        let video_decoder = create_video_decoder(video_codec)?;
        let audio_decoder = create_audio_decoder(audio_codec)?;

        let jitter_buffer = JitterBuffer::new(100, 20);

        let (control_tx, control_rx) = mpsc::channel(100);

        Ok(Self {
            transport,
            source_addr: None,
            video_decoder,
            audio_decoder,
            fec_decoder: None,
            jitter_buffer,
            stats: StatsTracker::new(),
            last_sequence: None,
            frame_assembly: HashMap::new(),
            control_tx,
            control_rx: Arc::new(RwLock::new(control_rx)),
        })
    }

    /// Discovers and connects to a source by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the source is not found.
    pub async fn discover(name: &str) -> VideoIpResult<Self> {
        let client = DiscoveryClient::new()?;
        let source = client.discover_by_name(name, 5).await?;

        // Use the discovered codec types
        let video_codec = source.video_format.codec;
        let audio_codec = source.audio_format.codec;

        let mut receiver = Self::new(video_codec, audio_codec).await?;
        receiver.source_addr = Some(source.socket_addr());

        Ok(receiver)
    }

    /// Connects to a specific source address.
    ///
    /// # Errors
    ///
    /// Returns an error if connection fails.
    pub async fn connect(
        addr: SocketAddr,
        video_codec: VideoCodec,
        audio_codec: AudioCodec,
    ) -> VideoIpResult<Self> {
        let mut receiver = Self::new(video_codec, audio_codec).await?;
        receiver.source_addr = Some(addr);
        Ok(receiver)
    }

    /// Starts receiving packets.
    pub fn start_receiving(&self) {
        // In a real implementation, this would start background tasks
    }

    /// Stops receiving packets.
    pub fn stop_receiving(&self) {
        // In a real implementation, this would stop background tasks
    }

    /// Enables FEC decoding.
    ///
    /// # Errors
    ///
    /// Returns an error if FEC cannot be enabled.
    pub fn enable_fec(&mut self, data_shards: usize, parity_shards: usize) -> VideoIpResult<()> {
        self.fec_decoder = Some(FecDecoder::new(data_shards, parity_shards)?);
        Ok(())
    }

    /// Receives a complete frame (video and audio).
    ///
    /// # Errors
    ///
    /// Returns an error if receiving fails or times out.
    pub async fn receive_frame(&mut self) -> VideoIpResult<(VideoFrame, Option<AudioSamples>)> {
        let deadline = Duration::from_millis(FRAME_TIMEOUT_MS);

        timeout(deadline, async {
            loop {
                // Receive packets
                let (packet, _addr) = self.transport.recv_packet().await?;

                // Update stats
                self.stats.record_received(packet.size());

                // Check sequence for packet loss
                self.check_sequence(packet.header.sequence);

                // Handle FEC packets
                if packet.header.flags.contains(PacketFlags::FEC) {
                    if let Some(ref mut fec) = self.fec_decoder {
                        let recovered = fec.add_packet(packet)?;
                        for p in recovered {
                            self.jitter_buffer.add_packet(p)?;
                        }
                    }
                    continue;
                }

                // Add to jitter buffer
                self.jitter_buffer.add_packet(packet)?;

                // Try to get packets from jitter buffer and assemble frames
                while let Some(packet) = self.jitter_buffer.get_packet() {
                    if packet.header.flags.contains(PacketFlags::VIDEO) {
                        if let Some(frame) = self.process_video_packet(packet)? {
                            // We have a complete video frame
                            // Try to get corresponding audio
                            let audio = self.get_audio_sample().await.ok();
                            return Ok((frame, audio));
                        }
                    } else if packet.header.flags.contains(PacketFlags::AUDIO) {
                        // Store audio for later retrieval
                        self.process_audio_packet(packet)?;
                    } else if packet.header.flags.contains(PacketFlags::METADATA) {
                        self.process_metadata_packet(packet)?;
                    }
                }

                // Cleanup old incomplete frames
                self.cleanup_old_frames();
            }
        })
        .await
        .map_err(|_| VideoIpError::Timeout)?
    }

    /// Processes a video packet and assembles frames.
    fn process_video_packet(&mut self, packet: Packet) -> VideoIpResult<Option<VideoFrame>> {
        let pts = packet.header.timestamp;
        let is_keyframe = packet.header.flags.contains(PacketFlags::KEYFRAME);
        let is_start = packet.header.flags.contains(PacketFlags::START_OF_FRAME);
        let is_end = packet.header.flags.contains(PacketFlags::END_OF_FRAME);

        // Single-packet frame
        if is_start && is_end {
            return self.decode_video_frame(packet.payload, is_keyframe, pts);
        }

        // Multi-packet frame assembly
        let assembly = self
            .frame_assembly
            .entry(pts)
            .or_insert_with(|| FrameAssembly {
                chunks: Vec::new(),
                total_chunks: 0,
                timestamp: pts,
                start_time: Instant::now(),
                _is_keyframe: is_keyframe,
            });

        if is_start {
            assembly.chunks.clear();
            assembly.chunks.push(Some(packet.payload));
        } else if is_end {
            assembly.chunks.push(Some(packet.payload));
            assembly.total_chunks = assembly.chunks.len();

            // Assemble complete frame
            let complete = assembly.chunks.iter().all(Option::is_some);
            if complete {
                let mut data = BytesMut::new();
                for bytes in assembly.chunks.iter().flatten() {
                    data.extend_from_slice(bytes);
                }

                let frame = self.decode_video_frame(data.freeze(), is_keyframe, pts)?;
                self.frame_assembly.remove(&pts);
                return Ok(frame);
            }
        } else {
            assembly.chunks.push(Some(packet.payload));
        }

        Ok(None)
    }

    /// Decodes a complete video frame.
    fn decode_video_frame(
        &mut self,
        data: Bytes,
        _is_keyframe: bool,
        _pts: u64,
    ) -> VideoIpResult<Option<VideoFrame>> {
        self.video_decoder.decode(&data)
    }

    /// Processes an audio packet.
    fn process_audio_packet(&mut self, _packet: Packet) -> VideoIpResult<()> {
        // Store for later retrieval
        // In a real implementation, we'd maintain an audio buffer
        Ok(())
    }

    /// Gets an audio sample if available.
    async fn get_audio_sample(&mut self) -> VideoIpResult<AudioSamples> {
        // In a real implementation, retrieve from audio buffer
        Err(VideoIpError::Timeout)
    }

    /// Processes a metadata packet.
    fn process_metadata_packet(&mut self, packet: Packet) -> VideoIpResult<()> {
        // Try to parse as different metadata types
        if let Ok(ptz_msg) = PtzMessage::decode(&packet.payload) {
            let _ = self.control_tx.try_send(ControlEvent::Ptz(ptz_msg));
        } else if let Ok(tally_msg) = TallyMessage::decode(&packet.payload) {
            let _ = self.control_tx.try_send(ControlEvent::Tally(tally_msg));
        } else if let Ok(metadata) = MetadataPacket::decode(&packet.payload) {
            let _ = self.control_tx.try_send(ControlEvent::Metadata(metadata));
        }

        Ok(())
    }

    /// Checks for packet loss by comparing sequence numbers.
    fn check_sequence(&mut self, sequence: u16) {
        if let Some(last) = self.last_sequence {
            let expected = last.wrapping_add(1);
            if sequence != expected {
                // Packet loss detected
                let lost = if sequence > expected {
                    u64::from(sequence - expected)
                } else {
                    u64::from((u16::MAX - expected) + sequence + 1)
                };

                for _ in 0..lost {
                    self.stats.record_lost();
                }
            }
        }

        self.last_sequence = Some(sequence);
    }

    /// Cleans up old incomplete frames.
    fn cleanup_old_frames(&mut self) {
        let now = Instant::now();
        let timeout = Duration::from_millis(FRAME_TIMEOUT_MS);

        self.frame_assembly
            .retain(|_, assembly| now.duration_since(assembly.start_time) < timeout);
    }

    /// Returns the current statistics.
    #[must_use]
    pub fn stats(&self) -> crate::stats::NetworkStats {
        self.stats.get_stats()
    }

    /// Returns a receiver for control events.
    #[must_use]
    pub fn control_receiver(&self) -> Arc<RwLock<mpsc::Receiver<ControlEvent>>> {
        Arc::clone(&self.control_rx)
    }

    /// Returns the local socket address.
    #[must_use]
    pub fn local_addr(&self) -> SocketAddr {
        self.transport.local_addr()
    }

    /// Returns the jitter buffer statistics.
    #[must_use]
    pub fn jitter_stats(&self) -> crate::jitter::JitterStats {
        self.jitter_buffer.stats().clone()
    }

    /// Adjusts the jitter buffer delay dynamically.
    pub fn adjust_jitter_buffer(&mut self) {
        self.jitter_buffer.adjust_delay();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_receiver_creation() {
        let receiver = VideoIpReceiver::new(VideoCodec::Vp9, AudioCodec::Opus).await;
        assert!(receiver.is_ok());
    }

    #[tokio::test]
    async fn test_receiver_connect() {
        let addr = "127.0.0.1:5000".parse().expect("should succeed in test");
        let receiver = VideoIpReceiver::connect(addr, VideoCodec::Vp9, AudioCodec::Opus).await;
        assert!(receiver.is_ok());
    }

    #[tokio::test]
    async fn test_receiver_enable_fec() {
        let mut receiver = VideoIpReceiver::new(VideoCodec::Vp9, AudioCodec::Opus)
            .await
            .expect("should succeed in test");

        assert!(receiver.enable_fec(20, 2).is_ok());
        assert!(receiver.fec_decoder.is_some());
    }

    #[test]
    fn test_sequence_check() {
        let rt = tokio::runtime::Runtime::new().expect("should succeed in test");
        let mut receiver = rt
            .block_on(VideoIpReceiver::new(VideoCodec::Vp9, AudioCodec::Opus))
            .expect("should succeed in test");

        receiver.check_sequence(0);
        receiver.check_sequence(1);
        receiver.check_sequence(2);

        let stats = receiver.stats();
        assert_eq!(stats.packets_lost, 0);

        // Skip sequence 3
        receiver.check_sequence(4);
        let stats = receiver.stats();
        assert_eq!(stats.packets_lost, 1);
    }

    #[test]
    fn test_cleanup_old_frames() {
        let rt = tokio::runtime::Runtime::new().expect("should succeed in test");
        let mut receiver = rt
            .block_on(VideoIpReceiver::new(VideoCodec::Vp9, AudioCodec::Opus))
            .expect("should succeed in test");

        // Add an old frame assembly
        receiver.frame_assembly.insert(
            12345,
            FrameAssembly {
                chunks: vec![],
                total_chunks: 0,
                timestamp: 12345,
                start_time: Instant::now() - Duration::from_secs(1),
                _is_keyframe: false,
            },
        );

        receiver.cleanup_old_frames();
        assert_eq!(receiver.frame_assembly.len(), 0);
    }
}
