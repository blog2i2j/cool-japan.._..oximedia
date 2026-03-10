#![allow(dead_code)]
//! AAF export utilities.
//!
//! This module provides a high-level configuration-driven AAF exporter that
//! serialises an AAF object graph into a byte stream suitable for writing to
//! a file or network stream.  [`AafExporter`] wraps an [`AafExportConfig`]
//! and produces a header block followed by the content storage and optional
//! essence data.

use std::collections::HashMap;
use uuid::Uuid;

/// Controls how the exporter serialises essence data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportMode {
    /// Embed all essence inside the AAF container.
    Embedded,
    /// Write essence as external files referenced by the AAF.
    ExternalRef,
    /// Mixed: small essence is embedded, large essence is external.
    Mixed,
}

impl ExportMode {
    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Embedded => "Embedded",
            Self::ExternalRef => "External Reference",
            Self::Mixed => "Mixed",
        }
    }

    /// Whether this mode may embed essence.
    #[must_use]
    pub const fn may_embed(&self) -> bool {
        matches!(self, Self::Embedded | Self::Mixed)
    }
}

impl std::fmt::Display for ExportMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Configuration for the AAF exporter.
#[derive(Debug, Clone)]
pub struct AafExportConfig {
    /// Export mode for essence handling.
    mode: ExportMode,
    /// AAF file format version (major).
    version_major: u16,
    /// AAF file format version (minor).
    version_minor: u16,
    /// Threshold in bytes: essence larger than this is externalised in Mixed mode.
    external_threshold: u64,
    /// Optional application name stamped into the header.
    application_name: Option<String>,
    /// Custom key-value properties to embed in the header.
    custom_properties: HashMap<String, String>,
}

impl AafExportConfig {
    /// Create a default export configuration (embedded, version 1.2).
    #[must_use]
    pub fn new() -> Self {
        Self {
            mode: ExportMode::Embedded,
            version_major: 1,
            version_minor: 2,
            external_threshold: 10 * 1024 * 1024, // 10 MiB
            application_name: None,
            custom_properties: HashMap::new(),
        }
    }

    /// Builder: set the export mode.
    #[must_use]
    pub fn with_mode(mut self, mode: ExportMode) -> Self {
        self.mode = mode;
        self
    }

    /// Builder: set the file format version.
    #[must_use]
    pub fn with_version(mut self, major: u16, minor: u16) -> Self {
        self.version_major = major;
        self.version_minor = minor;
        self
    }

    /// Builder: set the external-reference threshold (bytes).
    #[must_use]
    pub fn with_external_threshold(mut self, bytes: u64) -> Self {
        self.external_threshold = bytes;
        self
    }

    /// Builder: set the application name.
    #[must_use]
    pub fn with_application_name(mut self, name: impl Into<String>) -> Self {
        self.application_name = Some(name.into());
        self
    }

    /// Add a custom header property.
    pub fn add_property(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.custom_properties.insert(key.into(), value.into());
    }

    /// Export mode.
    #[must_use]
    pub fn mode(&self) -> ExportMode {
        self.mode
    }

    /// File version as `(major, minor)`.
    #[must_use]
    pub fn version(&self) -> (u16, u16) {
        (self.version_major, self.version_minor)
    }

    /// External-reference threshold.
    #[must_use]
    pub fn external_threshold(&self) -> u64 {
        self.external_threshold
    }

    /// Application name.
    #[must_use]
    pub fn application_name(&self) -> Option<&str> {
        self.application_name.as_deref()
    }

    /// Custom properties.
    #[must_use]
    pub fn custom_properties(&self) -> &HashMap<String, String> {
        &self.custom_properties
    }
}

impl Default for AafExportConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level AAF exporter.
///
/// Given an [`AafExportConfig`], the exporter produces a header byte block
/// that can be written to an output stream.
#[derive(Debug, Clone)]
pub struct AafExporter {
    config: AafExportConfig,
    /// Unique export session id.
    session_id: Uuid,
}

impl AafExporter {
    /// Create a new exporter with the given configuration.
    #[must_use]
    pub fn new(config: AafExportConfig) -> Self {
        Self {
            config,
            session_id: Uuid::new_v4(),
        }
    }

    /// Create an exporter with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(AafExportConfig::new())
    }

    /// Session id for this export run.
    #[must_use]
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &AafExportConfig {
        &self.config
    }

    /// Write the AAF header block into a byte vector.
    ///
    /// The header contains:
    /// - A 16-byte magic signature
    /// - 4-byte version (major + minor)
    /// - 16-byte session UUID
    /// - Mode byte
    /// - Application name (length-prefixed UTF-8)
    /// - Custom properties count + key/value pairs
    #[must_use]
    pub fn write_header(&self) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(256);

        // Magic: "AAF-OxiMedia    " (16 bytes, padded with spaces)
        let magic = b"AAF-OxiMedia    ";
        buf.extend_from_slice(magic);

        // Version: major (2 bytes BE) + minor (2 bytes BE)
        buf.extend_from_slice(&self.config.version_major.to_be_bytes());
        buf.extend_from_slice(&self.config.version_minor.to_be_bytes());

        // Session UUID (16 bytes)
        buf.extend_from_slice(self.session_id.as_bytes());

        // Mode byte
        let mode_byte: u8 = match self.config.mode {
            ExportMode::Embedded => 0,
            ExportMode::ExternalRef => 1,
            ExportMode::Mixed => 2,
        };
        buf.push(mode_byte);

        // Application name (u16-length-prefixed UTF-8)
        if let Some(name) = &self.config.application_name {
            let name_bytes = name.as_bytes();
            let len = name_bytes.len().min(u16::MAX as usize) as u16;
            buf.extend_from_slice(&len.to_be_bytes());
            buf.extend_from_slice(&name_bytes[..len as usize]);
        } else {
            buf.extend_from_slice(&0u16.to_be_bytes());
        }

        // Custom properties: count (u16 BE), then key-len + key + val-len + val
        let prop_count = self.config.custom_properties.len().min(u16::MAX as usize) as u16;
        buf.extend_from_slice(&prop_count.to_be_bytes());
        for (k, v) in &self.config.custom_properties {
            let kb = k.as_bytes();
            let kl = kb.len().min(u16::MAX as usize) as u16;
            buf.extend_from_slice(&kl.to_be_bytes());
            buf.extend_from_slice(&kb[..kl as usize]);
            let vb = v.as_bytes();
            let vl = vb.len().min(u16::MAX as usize) as u16;
            buf.extend_from_slice(&vl.to_be_bytes());
            buf.extend_from_slice(&vb[..vl as usize]);
        }

        buf
    }

    /// Returns the expected minimum header size in bytes (without properties).
    #[must_use]
    pub fn min_header_size(&self) -> usize {
        // magic(16) + version(4) + uuid(16) + mode(1) + name_len(2) + props_count(2) = 41
        41
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_mode_label() {
        assert_eq!(ExportMode::Embedded.label(), "Embedded");
        assert_eq!(ExportMode::ExternalRef.label(), "External Reference");
        assert_eq!(ExportMode::Mixed.label(), "Mixed");
    }

    #[test]
    fn test_export_mode_may_embed() {
        assert!(ExportMode::Embedded.may_embed());
        assert!(ExportMode::Mixed.may_embed());
        assert!(!ExportMode::ExternalRef.may_embed());
    }

    #[test]
    fn test_export_mode_display() {
        assert_eq!(format!("{}", ExportMode::Mixed), "Mixed");
    }

    #[test]
    fn test_config_defaults() {
        let cfg = AafExportConfig::new();
        assert_eq!(cfg.mode(), ExportMode::Embedded);
        assert_eq!(cfg.version(), (1, 2));
        assert_eq!(cfg.external_threshold(), 10 * 1024 * 1024);
        assert!(cfg.application_name().is_none());
        assert!(cfg.custom_properties().is_empty());
    }

    #[test]
    fn test_config_builders() {
        let cfg = AafExportConfig::new()
            .with_mode(ExportMode::ExternalRef)
            .with_version(2, 0)
            .with_external_threshold(5_000_000)
            .with_application_name("TestApp");
        assert_eq!(cfg.mode(), ExportMode::ExternalRef);
        assert_eq!(cfg.version(), (2, 0));
        assert_eq!(cfg.external_threshold(), 5_000_000);
        assert_eq!(cfg.application_name(), Some("TestApp"));
    }

    #[test]
    fn test_config_custom_properties() {
        let mut cfg = AafExportConfig::new();
        cfg.add_property("vendor", "OxiMedia");
        cfg.add_property("project", "TestProject");
        assert_eq!(cfg.custom_properties().len(), 2);
        assert_eq!(
            cfg.custom_properties()
                .get("vendor")
                .expect("get should succeed"),
            "OxiMedia"
        );
    }

    #[test]
    fn test_exporter_creation() {
        let exporter = AafExporter::with_defaults();
        assert_eq!(exporter.config().mode(), ExportMode::Embedded);
    }

    #[test]
    fn test_exporter_session_id_unique() {
        let a = AafExporter::with_defaults();
        let b = AafExporter::with_defaults();
        assert_ne!(a.session_id(), b.session_id());
    }

    #[test]
    fn test_write_header_magic() {
        let exporter = AafExporter::with_defaults();
        let header = exporter.write_header();
        assert!(header.len() >= exporter.min_header_size());
        assert_eq!(&header[..16], b"AAF-OxiMedia    ");
    }

    #[test]
    fn test_write_header_version() {
        let cfg = AafExportConfig::new().with_version(3, 7);
        let exporter = AafExporter::new(cfg);
        let header = exporter.write_header();
        assert_eq!(u16::from_be_bytes([header[16], header[17]]), 3);
        assert_eq!(u16::from_be_bytes([header[18], header[19]]), 7);
    }

    #[test]
    fn test_write_header_mode_byte() {
        for (mode, expected) in [
            (ExportMode::Embedded, 0u8),
            (ExportMode::ExternalRef, 1u8),
            (ExportMode::Mixed, 2u8),
        ] {
            let cfg = AafExportConfig::new().with_mode(mode);
            let exporter = AafExporter::new(cfg);
            let header = exporter.write_header();
            // offset: magic(16) + version(4) + uuid(16) = 36
            assert_eq!(header[36], expected, "mode byte mismatch for {mode}");
        }
    }

    #[test]
    fn test_write_header_with_app_name() {
        let cfg = AafExportConfig::new().with_application_name("MyApp");
        let exporter = AafExporter::new(cfg);
        let header = exporter.write_header();
        // App name starts at offset 37: u16 len (2) + "MyApp" (5) = 7 bytes
        let name_len = u16::from_be_bytes([header[37], header[38]]);
        assert_eq!(name_len, 5);
        assert_eq!(&header[39..44], b"MyApp");
    }

    #[test]
    fn test_write_header_no_app_name() {
        let exporter = AafExporter::with_defaults();
        let header = exporter.write_header();
        let name_len = u16::from_be_bytes([header[37], header[38]]);
        assert_eq!(name_len, 0);
    }

    #[test]
    fn test_config_default_trait() {
        let cfg = AafExportConfig::default();
        assert_eq!(cfg.mode(), ExportMode::Embedded);
    }

    #[test]
    fn test_min_header_size() {
        let exporter = AafExporter::with_defaults();
        assert_eq!(exporter.min_header_size(), 41);
    }
}
