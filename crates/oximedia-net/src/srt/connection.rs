//! SRT connection management with UDP socket.
//!
//! Provides high-level connection handling with async I/O.

use super::congestion::CongestionControl;
use super::crypto::AesContext;
use super::loss::{LossList, ReceiveBuffer};
use super::packet::{ControlPacket, DataPacket, SrtPacket};
use super::socket::{SrtConfig, SrtSocket};
use crate::error::{NetError, NetResult};
use bytes::Bytes;
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::UdpSocket;
use tokio::sync::Mutex;
use tokio::time;

/// Send queue entry.
#[derive(Debug, Clone)]
struct SendQueueEntry {
    /// Sequence number.
    seq: u32,
    /// Packet data.
    packet: DataPacket,
    /// Time first sent.
    sent_at: Option<Instant>,
    /// Number of retransmissions.
    retransmit_count: u32,
}

/// SRT connection with UDP transport.
pub struct SrtConnection {
    /// UDP socket.
    socket: Arc<UdpSocket>,
    /// Remote peer address.
    peer_addr: SocketAddr,
    /// SRT state machine.
    state: Arc<Mutex<SrtSocket>>,
    /// Congestion control.
    congestion: Arc<Mutex<CongestionControl>>,
    /// Loss list.
    loss_list: Arc<Mutex<LossList>>,
    /// Receive buffer.
    recv_buffer: Arc<Mutex<ReceiveBuffer>>,
    /// Send queue (unacknowledged packets).
    send_queue: Arc<Mutex<VecDeque<SendQueueEntry>>>,
    /// Encryption context.
    crypto: Arc<Mutex<Option<AesContext>>>,
    /// Last keepalive sent time.
    last_keepalive: Arc<Mutex<Instant>>,
    /// Read buffer for received data.
    read_buffer: Arc<Mutex<VecDeque<Bytes>>>,
}

impl SrtConnection {
    /// Creates a new SRT connection.
    ///
    /// # Errors
    ///
    /// Returns an error if socket binding fails.
    pub async fn new(
        local_addr: SocketAddr,
        peer_addr: SocketAddr,
        config: SrtConfig,
    ) -> NetResult<Self> {
        let socket = UdpSocket::bind(local_addr).await?;
        socket.connect(peer_addr).await?;

        let srt_socket = SrtSocket::new(config.clone());
        let initial_seq = srt_socket.send_seq;

        Ok(Self {
            socket: Arc::new(socket),
            peer_addr,
            state: Arc::new(Mutex::new(srt_socket)),
            congestion: Arc::new(Mutex::new(CongestionControl::new(
                config.flow_window,
                config.flow_window,
            ))),
            loss_list: Arc::new(Mutex::new(LossList::new(1000))),
            recv_buffer: Arc::new(Mutex::new(ReceiveBuffer::new(initial_seq, 1000))),
            send_queue: Arc::new(Mutex::new(VecDeque::new())),
            crypto: Arc::new(Mutex::new(None)),
            last_keepalive: Arc::new(Mutex::new(Instant::now())),
            read_buffer: Arc::new(Mutex::new(VecDeque::new())),
        })
    }

    /// Connects to a remote SRT peer (caller mode).
    ///
    /// # Errors
    ///
    /// Returns an error if the handshake fails or times out.
    pub async fn connect(&self, timeout: Duration) -> NetResult<()> {
        // Generate and send initial handshake
        let handshake_packet = {
            let mut state = self.state.lock().await;
            state.generate_caller_handshake()
        };

        self.send_packet(&handshake_packet).await?;

        // Wait for handshake response
        let deadline = Instant::now() + timeout;
        let mut buf = vec![0u8; 2048];

        loop {
            if Instant::now() > deadline {
                return Err(NetError::timeout("Connection timeout"));
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            let recv_result = time::timeout(remaining, self.socket.recv(&mut buf)).await;

            match recv_result {
                Ok(Ok(len)) => {
                    if let Ok(packet) = SrtPacket::decode(&buf[..len]) {
                        let responses = {
                            let mut state = self.state.lock().await;
                            state.process_packet(packet)?
                        };

                        for response in responses {
                            self.send_packet(&response).await?;
                        }

                        let is_connected = {
                            let state = self.state.lock().await;
                            state.is_connected()
                        };

                        if is_connected {
                            // Initialize encryption if configured
                            self.initialize_crypto().await?;
                            return Ok(());
                        }
                    }
                }
                Ok(Err(e)) => return Err(e.into()),
                Err(_) => {}
            }
        }
    }

    /// Creates an `SrtConnection` from an already-bound UDP socket that has
    /// received its first inbound packet.
    ///
    /// The socket is connected to `peer_addr` so that `send()` / `recv()`
    /// work without explicit addressing.  The first raw UDP datagram
    /// (`first_packet`) is decoded and fed through the SRT state machine
    /// (INDUCTION phase) before returning, so that callers can immediately
    /// proceed with `accept()` for the CONCLUSION phase.
    ///
    /// # Errors
    ///
    /// Returns an error if `socket.connect` fails or if the first packet
    /// triggers a protocol error.
    pub async fn from_inbound(
        socket: UdpSocket,
        peer_addr: SocketAddr,
        config: SrtConfig,
        first_packet: Vec<u8>,
    ) -> NetResult<Self> {
        socket.connect(peer_addr).await?;

        let srt_socket = SrtSocket::new(config.clone());
        let initial_seq = srt_socket.send_seq;

        let conn = Self {
            socket: Arc::new(socket),
            peer_addr,
            state: Arc::new(Mutex::new(srt_socket)),
            congestion: Arc::new(Mutex::new(CongestionControl::new(
                config.flow_window,
                config.flow_window,
            ))),
            loss_list: Arc::new(Mutex::new(LossList::new(1000))),
            recv_buffer: Arc::new(Mutex::new(ReceiveBuffer::new(initial_seq, 1000))),
            send_queue: Arc::new(Mutex::new(VecDeque::new())),
            crypto: Arc::new(Mutex::new(None)),
            last_keepalive: Arc::new(Mutex::new(Instant::now())),
            read_buffer: Arc::new(Mutex::new(VecDeque::new())),
        };

        // Process the INDUCTION packet that was already received on the
        // pre-bound socket.  Any responses (e.g. INDUCTION reply) are sent
        // immediately so the peer does not time out waiting for an answer.
        if let Ok(packet) = SrtPacket::decode(&first_packet) {
            let responses = {
                let mut state = conn.state.lock().await;
                state.process_packet(packet)?
            };

            for response in responses {
                conn.send_packet(&response).await?;
            }
        }

        Ok(conn)
    }

    /// Accepts an incoming SRT connection (listener mode).
    ///
    /// # Errors
    ///
    /// Returns an error if the handshake fails.
    pub async fn accept(&self) -> NetResult<()> {
        let mut buf = vec![0u8; 2048];

        loop {
            let (len, _addr) = self.socket.recv_from(&mut buf).await?;

            if let Ok(packet) = SrtPacket::decode(&buf[..len]) {
                let responses = {
                    let mut state = self.state.lock().await;
                    state.process_packet(packet)?
                };

                for response in responses {
                    self.send_packet(&response).await?;
                }

                let is_connected = {
                    let state = self.state.lock().await;
                    state.is_connected()
                };

                if is_connected {
                    self.initialize_crypto().await?;
                    return Ok(());
                }
            }
        }
    }

    /// Sends data over the SRT connection.
    ///
    /// # Errors
    ///
    /// Returns an error if not connected or send fails.
    pub async fn send(&self, data: &[u8]) -> NetResult<usize> {
        let is_connected = {
            let state = self.state.lock().await;
            state.is_connected()
        };

        if !is_connected {
            return Err(NetError::invalid_state("Not connected"));
        }

        // Check congestion window
        let cwnd = {
            let cc = self.congestion.lock().await;
            cc.window_size()
        };

        let send_queue_len = {
            let queue = self.send_queue.lock().await;
            queue.len()
        };

        if send_queue_len >= cwnd as usize {
            return Err(NetError::buffer("Send queue full"));
        }

        // Create data packet
        let mut packet = {
            let mut state = self.state.lock().await;
            state.create_data_packet(Bytes::copy_from_slice(data))
        };

        // Encrypt if needed
        if let Some(crypto) = self.crypto.lock().await.as_ref() {
            let iv = generate_iv(packet.sequence_number);
            packet.payload = crypto.encrypt(&packet.payload, &iv)?;
        }

        // Add to send queue
        {
            let mut queue = self.send_queue.lock().await;
            queue.push_back(SendQueueEntry {
                seq: packet.sequence_number,
                packet: packet.clone(),
                sent_at: Some(Instant::now()),
                retransmit_count: 0,
            });
        }

        // Send packet
        self.send_packet(&SrtPacket::Data(packet)).await?;

        Ok(data.len())
    }

    /// Receives data from the SRT connection.
    ///
    /// # Errors
    ///
    /// Returns an error if not connected or receive fails.
    pub async fn recv(&self, buf: &mut [u8]) -> NetResult<usize> {
        loop {
            // Check read buffer first
            {
                let mut read_buf = self.read_buffer.lock().await;
                if let Some(data) = read_buf.pop_front() {
                    let len = data.len().min(buf.len());
                    buf[..len].copy_from_slice(&data[..len]);
                    return Ok(len);
                }
            }

            // Receive from network
            let mut recv_buf = vec![0u8; 2048];
            let len = self.socket.recv(&mut recv_buf).await?;

            if let Ok(packet) = SrtPacket::decode(&recv_buf[..len]) {
                match packet {
                    SrtPacket::Data(data_packet) => {
                        // Decrypt if needed
                        let payload = if let Some(crypto) = self.crypto.lock().await.as_ref() {
                            let iv = generate_iv(data_packet.sequence_number);
                            crypto.decrypt(&data_packet.payload, &iv)?
                        } else {
                            data_packet.payload.clone()
                        };

                        let copy_len = payload.len().min(buf.len());
                        buf[..copy_len].copy_from_slice(&payload[..copy_len]);

                        // Process packet for sequencing
                        let responses = {
                            let mut state = self.state.lock().await;
                            state.process_packet(SrtPacket::Data(data_packet))?
                        };

                        for response in responses {
                            self.send_packet(&response).await?;
                        }

                        return Ok(copy_len);
                    }
                    SrtPacket::Control(ctrl) => {
                        let responses = {
                            let mut state = self.state.lock().await;
                            state.process_packet(SrtPacket::Control(ctrl))?
                        };

                        for response in responses {
                            self.send_packet(&response).await?;
                        }

                        // No data received, loop again to try receiving data
                    }
                }
            } else {
                return Err(NetError::protocol("Invalid packet"));
            }
        }
    }

    /// Runs background tasks (keepalive, retransmission, etc.).
    ///
    /// # Errors
    ///
    /// Returns an error if a critical failure occurs.
    pub async fn run_background_tasks(&self) -> NetResult<()> {
        let mut interval = time::interval(Duration::from_millis(10));

        loop {
            interval.tick().await;

            // Check if connection is still alive
            {
                let state = self.state.lock().await;
                if state.state().is_finished() {
                    break;
                }

                if state.check_timeout() {
                    return Err(NetError::timeout("Peer timeout"));
                }
            }

            // Send keepalive if needed
            self.send_keepalive_if_needed().await?;

            // Check for retransmissions
            self.check_retransmissions().await?;

            // Detect packet loss
            self.detect_loss().await?;
        }

        Ok(())
    }

    /// Closes the connection gracefully.
    ///
    /// # Errors
    ///
    /// Returns an error if sending shutdown packet fails.
    pub async fn close(&self) -> NetResult<()> {
        let shutdown_packet = {
            let mut state = self.state.lock().await;
            state.close()
        };

        if let Some(packet) = shutdown_packet {
            self.send_packet(&packet).await?;
        }

        Ok(())
    }

    /// Returns the peer address.
    #[must_use]
    pub const fn peer_addr(&self) -> SocketAddr {
        self.peer_addr
    }

    /// Returns true if connected.
    pub async fn is_connected(&self) -> bool {
        let state = self.state.lock().await;
        state.is_connected()
    }

    /// Returns current RTT estimate in microseconds.
    pub async fn rtt(&self) -> u32 {
        let cc = self.congestion.lock().await;
        cc.rtt()
    }

    async fn send_packet(&self, packet: &SrtPacket) -> NetResult<()> {
        let encoded = packet.encode();
        self.socket.send(&encoded).await?;
        Ok(())
    }

    async fn initialize_crypto(&self) -> NetResult<()> {
        let config = {
            let state = self.state.lock().await;
            state.config().clone()
        };

        if let Some(passphrase) = config.passphrase {
            let ctx = AesContext::from_passphrase(&passphrase, config.key_size as usize)?;
            let mut crypto = self.crypto.lock().await;
            *crypto = Some(ctx);
        }

        Ok(())
    }

    async fn send_keepalive_if_needed(&self) -> NetResult<()> {
        let mut last_ka = self.last_keepalive.lock().await;

        if last_ka.elapsed() > Duration::from_secs(1) {
            let peer_socket_id = {
                let state = self.state.lock().await;
                state.peer_socket_id()
            };

            let keepalive = ControlPacket::keepalive(peer_socket_id);
            self.send_packet(&SrtPacket::Control(keepalive)).await?;

            *last_ka = Instant::now();
        }

        Ok(())
    }

    async fn check_retransmissions(&self) -> NetResult<()> {
        let rto = {
            let cc = self.congestion.lock().await;
            cc.rto()
        };

        let mut to_retransmit = Vec::new();

        {
            let mut queue = self.send_queue.lock().await;

            for entry in queue.iter_mut() {
                if let Some(sent_at) = entry.sent_at {
                    let elapsed = sent_at.elapsed().as_micros() as u32;

                    if elapsed > rto && entry.retransmit_count < 5 {
                        to_retransmit.push(entry.packet.clone());
                        entry.sent_at = Some(Instant::now());
                        entry.retransmit_count += 1;
                    }
                }
            }
        }

        for packet in to_retransmit {
            self.send_packet(&SrtPacket::Data(packet)).await?;

            let mut cc = self.congestion.lock().await;
            cc.on_loss();
        }

        Ok(())
    }

    async fn detect_loss(&self) -> NetResult<()> {
        let gaps = {
            let recv_buf = self.recv_buffer.lock().await;
            recv_buf.detect_gaps()
        };

        if !gaps.is_empty() {
            let peer_socket_id = {
                let state = self.state.lock().await;
                state.peer_socket_id()
            };

            // Send NAK for lost packets
            let nak = ControlPacket::nak(&gaps, peer_socket_id);
            self.send_packet(&SrtPacket::Control(nak)).await?;

            let mut loss_list = self.loss_list.lock().await;
            for gap in gaps {
                loss_list.add(gap);
            }
        }

        Ok(())
    }
}

/// Generates an IV from sequence number.
fn generate_iv(seq: u32) -> [u8; 16] {
    let mut iv = [0u8; 16];
    iv[0..4].copy_from_slice(&seq.to_be_bytes());
    iv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_iv() {
        let iv1 = generate_iv(12345);
        let iv2 = generate_iv(12345);
        assert_eq!(iv1, iv2);

        let iv3 = generate_iv(54321);
        assert_ne!(iv1, iv3);
    }
}
