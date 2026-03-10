//! NDI source discovery using mDNS
//!
//! This module implements NDI source discovery using multicast DNS (mDNS).
//! NDI sources advertise themselves using the `_ndi._tcp.local` service type.
#![allow(dead_code)]

use crate::{NdiError, Result};
use mdns_sd::{ResolvedService, ServiceDaemon, ServiceEvent, ServiceInfo};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// NDI service type for mDNS
const NDI_SERVICE_TYPE: &str = "_ndi._tcp.local.";

/// Default NDI port
const DEFAULT_NDI_PORT: u16 = 5960;

/// Information about an NDI source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NdiSourceInfo {
    /// Unique identifier for this source
    pub id: Uuid,

    /// Human-readable name of the source
    pub name: String,

    /// Network address of the source
    pub address: SocketAddr,

    /// Groups this source belongs to
    pub groups: Vec<String>,

    /// Whether the source supports audio
    pub has_audio: bool,

    /// Whether the source supports video
    pub has_video: bool,

    /// Whether the source supports metadata
    pub has_metadata: bool,

    /// Additional properties
    pub properties: HashMap<String, String>,
}

impl NdiSourceInfo {
    /// Create a new NDI source info
    pub fn new(name: String, address: SocketAddr) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            address,
            groups: vec!["public".to_string()],
            has_audio: true,
            has_video: true,
            has_metadata: true,
            properties: HashMap::new(),
        }
    }

    /// Set the groups for this source
    pub fn with_groups(mut self, groups: Vec<String>) -> Self {
        self.groups = groups;
        self
    }

    /// Set audio support
    pub fn with_audio(mut self, has_audio: bool) -> Self {
        self.has_audio = has_audio;
        self
    }

    /// Set video support
    pub fn with_video(mut self, has_video: bool) -> Self {
        self.has_video = has_video;
        self
    }

    /// Set metadata support
    pub fn with_metadata(mut self, has_metadata: bool) -> Self {
        self.has_metadata = has_metadata;
        self
    }

    /// Add a property
    pub fn with_property(mut self, key: String, value: String) -> Self {
        self.properties.insert(key, value);
        self
    }

    /// Check if this source is in a specific group
    pub fn is_in_group(&self, group: &str) -> bool {
        self.groups.iter().any(|g| g == group)
    }

    /// Check if this source matches a filter
    pub fn matches_filter(&self, filter: &SourceFilter) -> bool {
        // Check name pattern
        if let Some(pattern) = &filter.name_pattern {
            if !self.name.contains(pattern) {
                return false;
            }
        }

        // Check groups
        if !filter.groups.is_empty() {
            let has_matching_group = filter
                .groups
                .iter()
                .any(|g| self.groups.iter().any(|sg| sg == g));
            if !has_matching_group {
                return false;
            }
        }

        // Check capabilities
        if filter.require_audio && !self.has_audio {
            return false;
        }
        if filter.require_video && !self.has_video {
            return false;
        }

        true
    }
}

/// Filter for discovering NDI sources
#[derive(Debug, Clone, Default)]
pub struct SourceFilter {
    /// Only include sources whose name contains this string
    pub name_pattern: Option<String>,

    /// Only include sources in these groups
    pub groups: Vec<String>,

    /// Only include sources with audio support
    pub require_audio: bool,

    /// Only include sources with video support
    pub require_video: bool,
}

impl SourceFilter {
    /// Create a new empty filter (matches everything)
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by name pattern
    pub fn with_name_pattern(mut self, pattern: String) -> Self {
        self.name_pattern = Some(pattern);
        self
    }

    /// Filter by groups
    pub fn with_groups(mut self, groups: Vec<String>) -> Self {
        self.groups = groups;
        self
    }

    /// Require audio support
    pub fn require_audio(mut self) -> Self {
        self.require_audio = true;
        self
    }

    /// Require video support
    pub fn require_video(mut self) -> Self {
        self.require_video = true;
        self
    }
}

/// NDI discovery service
pub struct DiscoveryService {
    /// mDNS service daemon
    daemon: ServiceDaemon,

    /// Known sources
    sources: Arc<RwLock<HashMap<String, NdiSourceInfo>>>,

    /// Event receiver
    receiver: Arc<RwLock<Option<mdns_sd::Receiver<ServiceEvent>>>>,
}

impl DiscoveryService {
    /// Create a new discovery service
    pub fn new() -> Result<Self> {
        let daemon = ServiceDaemon::new()
            .map_err(|e| NdiError::Discovery(format!("Failed to create mDNS daemon: {e}")))?;

        Ok(Self {
            daemon,
            sources: Arc::new(RwLock::new(HashMap::new())),
            receiver: Arc::new(RwLock::new(None)),
        })
    }

    /// Start browsing for NDI sources
    pub fn start_browsing(&self) -> Result<()> {
        let browse_channel = self
            .daemon
            .browse(NDI_SERVICE_TYPE)
            .map_err(|e| NdiError::Discovery(format!("Failed to start browsing: {e}")))?;

        *self.receiver.write() = Some(browse_channel);
        Ok(())
    }

    /// Discover NDI sources on the network
    ///
    /// # Arguments
    ///
    /// * `timeout` - How long to wait for discovery responses
    ///
    /// # Returns
    ///
    /// A list of discovered NDI sources
    pub async fn discover(&self, timeout: Duration) -> Result<Vec<NdiSourceInfo>> {
        self.discover_filtered(timeout, &SourceFilter::default())
            .await
    }

    /// Discover NDI sources with a filter
    ///
    /// # Arguments
    ///
    /// * `timeout` - How long to wait for discovery responses
    /// * `filter` - Filter to apply to discovered sources
    ///
    /// # Returns
    ///
    /// A list of discovered NDI sources that match the filter
    pub async fn discover_filtered(
        &self,
        timeout: Duration,
        filter: &SourceFilter,
    ) -> Result<Vec<NdiSourceInfo>> {
        debug!("Starting NDI source discovery with timeout {:?}", timeout);

        // Start browsing
        self.start_browsing()?;

        // Wait for the timeout period
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            if let Some(receiver) = self.receiver.write().as_mut() {
                // Process events with a short timeout using spawn_blocking
                let receiver_clone = receiver.clone();
                match tokio::task::spawn_blocking(move || receiver_clone.try_recv()).await {
                    Ok(Ok(event)) => {
                        self.handle_service_event(event)?;
                    }
                    Ok(Err(_)) => {
                        // No event available - continue
                    }
                    Err(_) => {
                        // Task panicked or was cancelled
                        break;
                    }
                }
            }

            // Small delay to avoid busy waiting
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Collect and filter sources
        let sources: Vec<NdiSourceInfo> = self
            .sources
            .read()
            .values()
            .filter(|s| s.matches_filter(filter))
            .cloned()
            .collect();

        info!("Discovered {} NDI sources", sources.len());
        Ok(sources)
    }

    /// Handle a service event
    fn handle_service_event(&self, event: ServiceEvent) -> Result<()> {
        match event {
            ServiceEvent::ServiceResolved(info) => {
                debug!("Service resolved: {}", info.get_fullname());
                if let Some(source_info) = self.parse_service_info(&*info) {
                    let name = source_info.name.clone();
                    self.sources.write().insert(name.clone(), source_info);
                    info!("Added NDI source: {}", name);
                }
            }
            ServiceEvent::ServiceRemoved(_ty, fullname) => {
                debug!("Service removed: {}", fullname);
                // Try to remove by fullname
                self.sources.write().retain(|_k, v| {
                    let service_name = format!("{}._ndi._tcp.local.", v.name);
                    service_name != fullname
                });
            }
            ServiceEvent::SearchStarted(service_type) => {
                debug!("Search started for: {}", service_type);
            }
            ServiceEvent::SearchStopped(service_type) => {
                debug!("Search stopped for: {}", service_type);
            }
            ServiceEvent::ServiceFound(fullname, service_type) => {
                debug!("Service found: {} of type {}", fullname, service_type);
            }
            _ => {
                debug!("Unhandled service event");
            }
        }
        Ok(())
    }

    /// Parse service info into NDI source info
    fn parse_service_info(&self, info: &ResolvedService) -> Option<NdiSourceInfo> {
        // Get the service name (remove the service type suffix)
        let fullname = info.get_fullname();
        let name = fullname
            .strip_suffix("._ndi._tcp.local.")
            .unwrap_or(fullname)
            .to_string();

        // Get the address
        let addresses = info.get_addresses();
        if addresses.is_empty() {
            warn!("No addresses found for NDI source: {}", name);
            return None;
        }

        let scoped_ip = addresses.iter().next()?;
        let port = info.get_port();
        let address = SocketAddr::new(scoped_ip.to_ip_addr(), port);

        // Parse properties
        let properties = info.get_properties();
        let mut source_info = NdiSourceInfo::new(name, address);

        // Parse groups
        if let Some(groups_str) = properties.get("groups") {
            let groups_str_value = format!("{}", groups_str);
            let groups: Vec<String> = groups_str_value
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            source_info.groups = groups;
        }

        // Parse capabilities
        if let Some(has_audio) = properties.get("audio") {
            let has_audio_str = format!("{}", has_audio);
            source_info.has_audio = has_audio_str == "true" || has_audio_str == "1";
        }
        if let Some(has_video) = properties.get("video") {
            let has_video_str = format!("{}", has_video);
            source_info.has_video = has_video_str == "true" || has_video_str == "1";
        }
        if let Some(has_metadata) = properties.get("metadata") {
            let has_metadata_str = format!("{}", has_metadata);
            source_info.has_metadata = has_metadata_str == "true" || has_metadata_str == "1";
        }

        // Store all properties
        for prop in properties.iter() {
            source_info
                .properties
                .insert(format!("{}", prop), "true".to_string());
        }

        Some(source_info)
    }

    /// Get all currently known sources
    pub fn get_sources(&self) -> Vec<NdiSourceInfo> {
        self.sources.read().values().cloned().collect()
    }

    /// Get a source by name
    pub fn get_source(&self, name: &str) -> Option<NdiSourceInfo> {
        self.sources.read().get(name).cloned()
    }

    /// Clear all known sources
    pub fn clear_sources(&self) {
        self.sources.write().clear();
    }
}

/// NDI source announcer
///
/// Announces an NDI source on the network using mDNS
pub struct SourceAnnouncer {
    /// mDNS service daemon
    daemon: ServiceDaemon,

    /// Source information
    source_info: NdiSourceInfo,

    /// Service instance name
    instance_name: String,
}

impl SourceAnnouncer {
    /// Create a new source announcer
    pub fn new(source_info: NdiSourceInfo) -> Result<Self> {
        let daemon = ServiceDaemon::new()
            .map_err(|e| NdiError::Discovery(format!("Failed to create mDNS daemon: {e}")))?;

        let instance_name = source_info.name.clone();

        Ok(Self {
            daemon,
            source_info,
            instance_name,
        })
    }

    /// Start announcing this source
    pub fn announce(&self) -> Result<()> {
        let mut properties = HashMap::new();

        // Add groups
        if !self.source_info.groups.is_empty() {
            properties.insert("groups".to_string(), self.source_info.groups.join(","));
        }

        // Add capabilities
        properties.insert(
            "audio".to_string(),
            if self.source_info.has_audio {
                "true"
            } else {
                "false"
            }
            .to_string(),
        );
        properties.insert(
            "video".to_string(),
            if self.source_info.has_video {
                "true"
            } else {
                "false"
            }
            .to_string(),
        );
        properties.insert(
            "metadata".to_string(),
            if self.source_info.has_metadata {
                "true"
            } else {
                "false"
            }
            .to_string(),
        );

        // Add custom properties
        for (key, value) in &self.source_info.properties {
            properties.insert(key.clone(), value.clone());
        }

        // Create service info
        let port = self.source_info.address.port();
        let host_name = match self.source_info.address.ip() {
            IpAddr::V4(ip) => format!("{}.local.", ip).replace('.', "-"),
            IpAddr::V6(ip) => format!("{}.local.", ip).to_string(),
        };

        let service_info = ServiceInfo::new(
            NDI_SERVICE_TYPE,
            &self.instance_name,
            &host_name,
            &self.source_info.address.ip().to_string(),
            port,
            Some(properties),
        )
        .map_err(|e| NdiError::Discovery(format!("Failed to create service info: {e}")))?;

        // Register the service
        self.daemon
            .register(service_info)
            .map_err(|e| NdiError::Discovery(format!("Failed to register service: {e}")))?;

        info!("Announced NDI source: {}", self.instance_name);
        Ok(())
    }

    /// Stop announcing this source
    pub fn unannounce(&self) -> Result<()> {
        self.daemon
            .unregister(&format!("{}._ndi._tcp.local.", self.instance_name))
            .map_err(|e| NdiError::Discovery(format!("Failed to unregister service: {e}")))?;

        info!("Unannounced NDI source: {}", self.instance_name);
        Ok(())
    }

    /// Update the source information
    pub fn update(&mut self, source_info: NdiSourceInfo) -> Result<()> {
        // Unannounce the old service
        self.unannounce()?;

        // Update the source info
        self.source_info = source_info;
        self.instance_name = self.source_info.name.clone();

        // Re-announce with new info
        self.announce()?;

        Ok(())
    }
}

impl Drop for SourceAnnouncer {
    fn drop(&mut self) {
        if let Err(e) = self.unannounce() {
            error!("Failed to unannounce NDI source on drop: {}", e);
        }
    }
}

/// Auto-discovery manager
///
/// Continuously monitors the network for NDI sources and maintains an up-to-date list
pub struct AutoDiscovery {
    /// Discovery service
    discovery: Arc<DiscoveryService>,

    /// Source change callback
    callback: Arc<RwLock<Option<Box<dyn Fn(Vec<NdiSourceInfo>) + Send + Sync>>>>,

    /// Whether discovery is running
    running: Arc<RwLock<bool>>,
}

impl AutoDiscovery {
    /// Create a new auto-discovery manager
    pub fn new() -> Result<Self> {
        Ok(Self {
            discovery: Arc::new(DiscoveryService::new()?),
            callback: Arc::new(RwLock::new(None)),
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Set the callback for source changes
    pub fn set_callback<F>(&self, callback: F)
    where
        F: Fn(Vec<NdiSourceInfo>) + Send + Sync + 'static,
    {
        *self.callback.write() = Some(Box::new(callback));
    }

    /// Start auto-discovery
    pub async fn start(&self) -> Result<()> {
        if *self.running.read() {
            return Ok(());
        }

        *self.running.write() = true;
        self.discovery.start_browsing()?;

        let discovery = self.discovery.clone();
        let callback = self.callback.clone();
        let running = self.running.clone();

        tokio::spawn(async move {
            let mut last_sources: Vec<String> = Vec::new();

            while *running.read() {
                // Wait a bit between checks
                tokio::time::sleep(Duration::from_secs(1)).await;

                let sources = discovery.get_sources();
                let current_sources: Vec<String> = sources.iter().map(|s| s.name.clone()).collect();

                // Check if sources have changed
                if current_sources != last_sources {
                    debug!("NDI sources changed: {} sources", sources.len());
                    last_sources = current_sources;

                    // Call the callback if set
                    if let Some(cb) = callback.read().as_ref() {
                        cb(sources);
                    }
                }
            }
        });

        info!("Auto-discovery started");
        Ok(())
    }

    /// Stop auto-discovery
    pub fn stop(&self) {
        *self.running.write() = false;
        info!("Auto-discovery stopped");
    }

    /// Get all currently known sources
    pub fn get_sources(&self) -> Vec<NdiSourceInfo> {
        self.discovery.get_sources()
    }

    /// Get a source by name
    pub fn get_source(&self, name: &str) -> Option<NdiSourceInfo> {
        self.discovery.get_source(name)
    }
}

impl Default for AutoDiscovery {
    fn default() -> Self {
        Self::new().expect("invariant: mDNS daemon must be available for auto-discovery")
    }
}

impl Drop for AutoDiscovery {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Helper function to get the local IP address
pub fn get_local_ip() -> Result<IpAddr> {
    use std::net::UdpSocket;

    // Connect to a public DNS server to determine our local IP
    let socket = UdpSocket::bind("0.0.0.0:0").map_err(|e| NdiError::Network(e))?;
    socket
        .connect("8.8.8.8:80")
        .map_err(|e| NdiError::Network(e))?;

    let local_addr = socket.local_addr().map_err(|e| NdiError::Network(e))?;

    Ok(local_addr.ip())
}

/// Helper function to find an available port
pub fn find_available_port() -> Result<u16> {
    use std::net::TcpListener;

    let listener = TcpListener::bind("0.0.0.0:0").map_err(|e| NdiError::Network(e))?;
    let port = listener
        .local_addr()
        .map_err(|e| NdiError::Network(e))?
        .port();

    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_info_creation() {
        let addr = "127.0.0.1:5960".parse().expect("expected valid parse");
        let info = NdiSourceInfo::new("Test Source".to_string(), addr);

        assert_eq!(info.name, "Test Source");
        assert_eq!(info.address, addr);
        assert!(info.has_audio);
        assert!(info.has_video);
        assert!(info.is_in_group("public"));
    }

    #[test]
    fn test_source_filter() {
        let addr = "127.0.0.1:5960".parse().expect("expected valid parse");
        let info = NdiSourceInfo::new("Test Source".to_string(), addr)
            .with_groups(vec!["test".to_string()])
            .with_audio(true)
            .with_video(false);

        let filter = SourceFilter::new()
            .with_groups(vec!["test".to_string()])
            .require_audio();

        assert!(info.matches_filter(&filter));

        let filter = filter.require_video();
        assert!(!info.matches_filter(&filter));
    }

    #[test]
    fn test_source_filter_name_pattern() {
        let addr = "127.0.0.1:5960".parse().expect("expected valid parse");
        let info = NdiSourceInfo::new("Test Source".to_string(), addr);

        let filter = SourceFilter::new().with_name_pattern("Test".to_string());
        assert!(info.matches_filter(&filter));

        let filter = SourceFilter::new().with_name_pattern("Other".to_string());
        assert!(!info.matches_filter(&filter));
    }

    #[test]
    fn test_find_available_port() {
        let port = find_available_port().expect("expected available port");
        assert!(port > 0);
    }
}
