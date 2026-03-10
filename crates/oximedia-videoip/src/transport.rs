//! UDP transport layer with socket tuning for low latency.

use crate::error::{VideoIpError, VideoIpResult};
use crate::packet::{Packet, MAX_PACKET_SIZE};
use bytes::{Bytes, BytesMut};
use socket2::{Domain, Protocol, Socket, Type};
use std::net::{SocketAddr, UdpSocket as StdUdpSocket};
use tokio::net::UdpSocket;

/// UDP transport for sending and receiving packets.
pub struct UdpTransport {
    /// The UDP socket.
    socket: UdpSocket,
    /// Local address.
    local_addr: SocketAddr,
    /// Send buffer for reuse.
    send_buffer: BytesMut,
    /// Receive buffer for reuse.
    recv_buffer: BytesMut,
}

impl UdpTransport {
    /// Creates a new UDP transport bound to the specified address.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket cannot be created or bound.
    pub async fn bind(addr: SocketAddr) -> VideoIpResult<Self> {
        let socket = Self::create_tuned_socket(addr)?;
        let local_addr = socket.local_addr()?;

        Ok(Self {
            socket,
            local_addr,
            send_buffer: BytesMut::with_capacity(MAX_PACKET_SIZE),
            recv_buffer: BytesMut::with_capacity(MAX_PACKET_SIZE),
        })
    }

    /// Creates a tuned UDP socket with optimized buffer sizes and `QoS` settings.
    fn create_tuned_socket(addr: SocketAddr) -> VideoIpResult<UdpSocket> {
        let domain = if addr.is_ipv4() {
            Domain::IPV4
        } else {
            Domain::IPV6
        };

        let socket = Socket::new(domain, Type::DGRAM, Some(Protocol::UDP))
            .map_err(|e| VideoIpError::Transport(format!("failed to create socket: {e}")))?;

        // Set SO_REUSEADDR to allow multiple sockets to bind to the same port
        socket
            .set_reuse_address(true)
            .map_err(|e| VideoIpError::Transport(format!("failed to set reuse address: {e}")))?;

        // Set large send and receive buffers (8 MB each)
        const BUFFER_SIZE: usize = 8 * 1024 * 1024;
        socket
            .set_send_buffer_size(BUFFER_SIZE)
            .map_err(|e| VideoIpError::Transport(format!("failed to set send buffer: {e}")))?;
        socket
            .set_recv_buffer_size(BUFFER_SIZE)
            .map_err(|e| VideoIpError::Transport(format!("failed to set recv buffer: {e}")))?;

        // Set ToS/DSCP for QoS (Expedited Forwarding - EF)
        // This prioritizes video traffic on QoS-enabled networks.
        // Uses socket2's pure-Rust API (works on Linux/macOS/Windows).
        #[cfg(unix)]
        {
            const DSCP_EF: u32 = 46 << 2; // Expedited Forwarding
            if let Err(e) = socket.set_tos_v4(DSCP_EF) {
                tracing::warn!("failed to set IP TOS: {}", e);
            }
        }

        // Bind the socket
        socket
            .bind(&addr.into())
            .map_err(|e| VideoIpError::Transport(format!("failed to bind socket: {e}")))?;

        // Convert to tokio UdpSocket
        let std_socket: StdUdpSocket = socket.into();
        std_socket
            .set_nonblocking(true)
            .map_err(|e| VideoIpError::Transport(format!("failed to set nonblocking: {e}")))?;

        UdpSocket::from_std(std_socket)
            .map_err(|e| VideoIpError::Transport(format!("failed to create tokio socket: {e}")))
    }

    /// Sends a packet to the specified destination.
    ///
    /// # Errors
    ///
    /// Returns an error if the send operation fails.
    pub async fn send_packet(&mut self, packet: &Packet, dest: SocketAddr) -> VideoIpResult<()> {
        self.send_buffer.clear();
        packet.header.encode(&mut self.send_buffer);
        self.send_buffer.extend_from_slice(&packet.payload);

        self.socket
            .send_to(&self.send_buffer, dest)
            .await
            .map_err(|e| VideoIpError::Transport(format!("failed to send packet: {e}")))?;

        Ok(())
    }

    /// Receives a packet from the socket.
    ///
    /// Returns the packet and the sender's address.
    ///
    /// # Errors
    ///
    /// Returns an error if the receive operation fails or the packet is invalid.
    pub async fn recv_packet(&mut self) -> VideoIpResult<(Packet, SocketAddr)> {
        self.recv_buffer.clear();
        self.recv_buffer.resize(MAX_PACKET_SIZE, 0);

        let (len, addr) = self
            .socket
            .recv_from(&mut self.recv_buffer)
            .await
            .map_err(|e| VideoIpError::Transport(format!("failed to receive packet: {e}")))?;

        self.recv_buffer.truncate(len);

        let packet = Packet::decode(&self.recv_buffer[..])?;

        Ok((packet, addr))
    }

    /// Sends raw bytes to the specified destination.
    ///
    /// # Errors
    ///
    /// Returns an error if the send operation fails.
    pub async fn send_bytes(&mut self, data: &[u8], dest: SocketAddr) -> VideoIpResult<()> {
        self.socket
            .send_to(data, dest)
            .await
            .map_err(|e| VideoIpError::Transport(format!("failed to send bytes: {e}")))?;

        Ok(())
    }

    /// Receives raw bytes from the socket.
    ///
    /// # Errors
    ///
    /// Returns an error if the receive operation fails.
    pub async fn recv_bytes(&mut self) -> VideoIpResult<(Bytes, SocketAddr)> {
        self.recv_buffer.clear();
        self.recv_buffer.resize(MAX_PACKET_SIZE, 0);

        let (len, addr) = self
            .socket
            .recv_from(&mut self.recv_buffer)
            .await
            .map_err(|e| VideoIpError::Transport(format!("failed to receive bytes: {e}")))?;

        Ok((Bytes::copy_from_slice(&self.recv_buffer[..len]), addr))
    }

    /// Returns the local socket address.
    #[must_use]
    pub const fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Joins a multicast group.
    ///
    /// # Errors
    ///
    /// Returns an error if joining the multicast group fails.
    #[cfg(target_os = "linux")]
    pub fn join_multicast(&self, multicast_addr: std::net::IpAddr) -> VideoIpResult<()> {
        use std::net::Ipv4Addr;

        match multicast_addr {
            std::net::IpAddr::V4(addr) => {
                self.socket
                    .join_multicast_v4(addr, Ipv4Addr::UNSPECIFIED)
                    .map_err(|e| {
                        VideoIpError::Transport(format!("failed to join multicast group: {e}"))
                    })?;
            }
            std::net::IpAddr::V6(addr) => {
                self.socket.join_multicast_v6(&addr, 0).map_err(|e| {
                    VideoIpError::Transport(format!("failed to join multicast group: {e}"))
                })?;
            }
        }

        Ok(())
    }

    /// Leaves a multicast group.
    ///
    /// # Errors
    ///
    /// Returns an error if leaving the multicast group fails.
    #[cfg(target_os = "linux")]
    pub fn leave_multicast(&self, multicast_addr: std::net::IpAddr) -> VideoIpResult<()> {
        use std::net::Ipv4Addr;

        match multicast_addr {
            std::net::IpAddr::V4(addr) => {
                self.socket
                    .leave_multicast_v4(addr, Ipv4Addr::UNSPECIFIED)
                    .map_err(|e| {
                        VideoIpError::Transport(format!("failed to leave multicast group: {e}"))
                    })?;
            }
            std::net::IpAddr::V6(addr) => {
                self.socket.leave_multicast_v6(&addr, 0).map_err(|e| {
                    VideoIpError::Transport(format!("failed to leave multicast group: {e}"))
                })?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet::PacketBuilder;

    #[tokio::test]
    async fn test_transport_creation() {
        let addr = "127.0.0.1:0".parse().expect("should succeed in test");
        let transport = UdpTransport::bind(addr)
            .await
            .expect("should succeed in test");
        assert!(transport.local_addr().port() > 0);
    }

    #[tokio::test]
    async fn test_send_recv_packet() {
        let addr1 = "127.0.0.1:0".parse().expect("should succeed in test");
        let addr2 = "127.0.0.1:0".parse().expect("should succeed in test");

        let mut transport1 = UdpTransport::bind(addr1)
            .await
            .expect("should succeed in test");
        let mut transport2 = UdpTransport::bind(addr2)
            .await
            .expect("should succeed in test");

        let packet = PacketBuilder::new(42)
            .video()
            .build(Bytes::from_static(b"Hello, World!"))
            .expect("should succeed in test");

        let dest = transport2.local_addr();
        transport1
            .send_packet(&packet, dest)
            .await
            .expect("should succeed in test");

        let (received, _) = transport2
            .recv_packet()
            .await
            .expect("should succeed in test");
        assert_eq!(received.header.sequence, 42);
        assert_eq!(received.payload, Bytes::from_static(b"Hello, World!"));
    }

    #[tokio::test]
    async fn test_send_recv_bytes() {
        let addr1 = "127.0.0.1:0".parse().expect("should succeed in test");
        let addr2 = "127.0.0.1:0".parse().expect("should succeed in test");

        let mut transport1 = UdpTransport::bind(addr1)
            .await
            .expect("should succeed in test");
        let mut transport2 = UdpTransport::bind(addr2)
            .await
            .expect("should succeed in test");

        let data = b"Test data";
        let dest = transport2.local_addr();
        transport1
            .send_bytes(data, dest)
            .await
            .expect("should succeed in test");

        let (received, _) = transport2
            .recv_bytes()
            .await
            .expect("should succeed in test");
        assert_eq!(&received[..], data);
    }
}
