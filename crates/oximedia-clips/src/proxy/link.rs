//! Proxy link types.

use crate::clip::ClipId;
use crate::error::{ClipError, ClipResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// Unique identifier for a proxy link.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProxyLinkId(Uuid);

impl ProxyLinkId {
    /// Creates a new random proxy link ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Creates a proxy link ID from a UUID.
    #[must_use]
    pub const fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Returns the inner UUID.
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for ProxyLinkId {
    fn default() -> Self {
        Self::new()
    }
}

/// Proxy quality level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProxyQuality {
    /// Low quality proxy (e.g., 480p).
    Low,
    /// Medium quality proxy (e.g., 720p).
    Medium,
    /// High quality proxy (e.g., 1080p).
    High,
    /// Custom quality.
    Custom,
}

impl ProxyQuality {
    /// Returns all quality levels.
    #[must_use]
    pub const fn all() -> [Self; 4] {
        [Self::Low, Self::Medium, Self::High, Self::Custom]
    }

    /// Parses a proxy quality from a string.
    ///
    /// # Errors
    ///
    /// Returns an error if the string is invalid.
    pub fn parse(s: &str) -> ClipResult<Self> {
        match s.to_lowercase().as_str() {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "custom" => Ok(Self::Custom),
            _ => Err(ClipError::InvalidProxyQuality(s.to_string())),
        }
    }
}

impl std::fmt::Display for ProxyQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// A link between a clip and its proxy media.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyLink {
    /// Unique identifier.
    pub id: ProxyLinkId,

    /// Original clip ID.
    pub clip_id: ClipId,

    /// Proxy file path.
    pub proxy_path: PathBuf,

    /// Proxy quality.
    pub quality: ProxyQuality,

    /// Resolution (e.g., "1920x1080").
    pub resolution: Option<String>,

    /// Bitrate in kbps.
    pub bitrate: Option<u32>,

    /// Codec used.
    pub codec: Option<String>,
}

impl ProxyLink {
    /// Creates a new proxy link.
    #[must_use]
    pub fn new(clip_id: ClipId, proxy_path: PathBuf, quality: ProxyQuality) -> Self {
        Self {
            id: ProxyLinkId::new(),
            clip_id,
            proxy_path,
            quality,
            resolution: None,
            bitrate: None,
            codec: None,
        }
    }

    /// Sets the resolution.
    pub fn set_resolution(&mut self, resolution: impl Into<String>) {
        self.resolution = Some(resolution.into());
    }

    /// Sets the bitrate.
    pub fn set_bitrate(&mut self, bitrate: u32) {
        self.bitrate = Some(bitrate);
    }

    /// Sets the codec.
    pub fn set_codec(&mut self, codec: impl Into<String>) {
        self.codec = Some(codec.into());
    }

    /// Checks if the proxy file exists.
    #[must_use]
    pub fn proxy_exists(&self) -> bool {
        self.proxy_path.exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_quality() {
        assert_eq!(
            ProxyQuality::parse("low").expect("parse should succeed"),
            ProxyQuality::Low
        );
        assert_eq!(
            ProxyQuality::parse("medium").expect("parse should succeed"),
            ProxyQuality::Medium
        );
        assert!(ProxyQuality::parse("invalid").is_err());
    }

    #[test]
    fn test_proxy_link() {
        let clip_id = ClipId::new();
        let proxy_path = PathBuf::from("/proxy/test.mov");
        let mut link = ProxyLink::new(clip_id, proxy_path, ProxyQuality::Medium);

        link.set_resolution("1920x1080");
        link.set_bitrate(10_000);
        link.set_codec("ProRes");

        assert_eq!(link.resolution, Some("1920x1080".to_string()));
        assert_eq!(link.bitrate, Some(10_000));
    }
}
