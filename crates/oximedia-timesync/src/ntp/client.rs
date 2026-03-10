//! NTP client implementation.

use super::packet::{NtpPacket, NtpTimestamp};
use super::pool::ServerPool;
use super::stratum::Stratum;
use crate::error::{TimeSyncError, TimeSyncResult};
use std::net::SocketAddr;
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// NTP client configuration.
#[derive(Debug, Clone)]
pub struct NtpClientConfig {
    /// Request timeout
    pub timeout: Duration,
    /// Maximum retries
    pub max_retries: usize,
    /// Server pool
    pub server_pool: ServerPool,
}

impl Default for NtpClientConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            max_retries: 3,
            server_pool: ServerPool::default(),
        }
    }
}

/// NTP synchronization result.
#[derive(Debug, Clone)]
pub struct NtpSyncResult {
    /// Clock offset (seconds)
    pub offset: f64,
    /// Round-trip delay (seconds)
    pub delay: f64,
    /// Server stratum
    pub stratum: Stratum,
    /// Server address
    pub server: SocketAddr,
    /// Leap indicator
    pub leap_indicator: super::LeapIndicator,
}

/// NTP client.
pub struct NtpClient {
    /// Configuration
    config: NtpClientConfig,
    /// UDP socket
    socket: Option<UdpSocket>,
}

impl NtpClient {
    /// Create a new NTP client with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: NtpClientConfig::default(),
            socket: None,
        }
    }

    /// Create a new NTP client with custom configuration.
    #[must_use]
    pub fn with_config(config: NtpClientConfig) -> Self {
        Self {
            config,
            socket: None,
        }
    }

    /// Add a server to the pool.
    pub fn add_server(&mut self, addr: SocketAddr) {
        self.config.server_pool.add_server(addr);
    }

    /// Synchronize with NTP servers.
    pub async fn synchronize(&mut self) -> TimeSyncResult<NtpSyncResult> {
        // Bind socket if not already bound
        if self.socket.is_none() {
            let socket = UdpSocket::bind("0.0.0.0:0").await?;
            self.socket = Some(socket);
        }

        let socket = self
            .socket
            .as_ref()
            .ok_or_else(|| TimeSyncError::InvalidConfig("Socket not initialized".to_string()))?;

        // Try each server in the pool
        let servers: Vec<_> = self
            .config
            .server_pool
            .servers()
            .into_iter()
            .copied()
            .collect();
        for server in servers {
            for attempt in 0..self.config.max_retries {
                debug!("Querying NTP server {} (attempt {})", server, attempt + 1);

                match self.query_server(socket, server).await {
                    Ok(result) => {
                        info!(
                            "NTP sync successful: offset={:.6}s, delay={:.6}s, stratum={}",
                            result.offset, result.delay, result.stratum
                        );
                        self.config.server_pool.mark_success(server);
                        return Ok(result);
                    }
                    Err(e) => {
                        warn!("NTP query to {} failed: {}", server, e);
                        self.config.server_pool.mark_failure(server);
                    }
                }
            }
        }

        Err(TimeSyncError::Ntp("All NTP servers failed".to_string()))
    }

    /// Query a single NTP server.
    async fn query_server(
        &self,
        socket: &UdpSocket,
        server: SocketAddr,
    ) -> TimeSyncResult<NtpSyncResult> {
        // Create and send request
        let request = NtpPacket::new_client_request();
        let t1 = request.transmit_timestamp;

        let request_data = request.serialize()?;
        socket.send_to(&request_data, server).await?;

        // Receive response with timeout
        let mut response_buf = [0u8; 48];
        let receive_future = socket.recv_from(&mut response_buf);

        let (len, recv_addr) = timeout(self.config.timeout, receive_future)
            .await
            .map_err(|_| TimeSyncError::Timeout)??;

        if recv_addr != server {
            return Err(TimeSyncError::Network(
                "Response from unexpected address".to_string(),
            ));
        }

        if len < 48 {
            return Err(TimeSyncError::InvalidPacket(
                "Response too short".to_string(),
            ));
        }

        // Parse response
        let response = NtpPacket::deserialize(&response_buf[..])?;
        let t4 = NtpTimestamp::now();

        // Validate response
        if response.mode != super::Mode::Server {
            return Err(TimeSyncError::InvalidPacket(
                "Invalid mode in response".to_string(),
            ));
        }

        if response.transmit_timestamp.is_zero() {
            return Err(TimeSyncError::InvalidPacket(
                "Zero transmit timestamp".to_string(),
            ));
        }

        // Set origin timestamp from our request
        let mut response = response;
        response.origin_timestamp = t1;

        // Calculate offset and delay
        let offset = response.calculate_offset(&t4);
        let delay = response.calculate_delay(&t4);

        // Validate delay (should be positive and reasonable)
        if !(0.0..=1.0).contains(&delay) {
            return Err(TimeSyncError::InvalidPacket(
                "Invalid delay calculation".to_string(),
            ));
        }

        Ok(NtpSyncResult {
            offset,
            delay,
            stratum: Stratum::from_u8(response.stratum),
            server,
            leap_indicator: response.leap_indicator,
        })
    }

    /// Get the current configuration.
    pub fn config(&self) -> &NtpClientConfig {
        &self.config
    }

    /// Get mutable configuration.
    pub fn config_mut(&mut self) -> &mut NtpClientConfig {
        &mut self.config
    }
}

impl Default for NtpClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntp_client_creation() {
        let client = NtpClient::new();
        assert!(client.socket.is_none());
    }

    #[test]
    fn test_add_server() {
        let mut client = NtpClient::new();
        let addr: SocketAddr = "127.0.0.1:123".parse().expect("should succeed in test");
        client.add_server(addr);
        assert_eq!(client.config.server_pool.servers().len(), 1);
    }

    #[test]
    fn test_config() {
        let mut config = NtpClientConfig::default();
        config.timeout = Duration::from_secs(10);

        let client = NtpClient::with_config(config);
        assert_eq!(client.config().timeout, Duration::from_secs(10));
    }
}
