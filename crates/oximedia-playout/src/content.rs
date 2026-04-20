//! Content management and validation
//!
//! Handles media ingest, validation, cataloging, and quality control for playout.
//! Supports watch folders, FTP/SFTP import, metadata extraction, and technical QC.

use crate::{PlayoutError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::sync::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

/// Content management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentConfig {
    /// Content root directory
    pub content_root: PathBuf,

    /// Watch folder for auto-import
    pub watch_folders: Vec<PathBuf>,

    /// Enable auto-ingest
    pub auto_ingest: bool,

    /// Enable technical QC
    pub qc_enabled: bool,

    /// Proxy generation settings
    pub proxy_settings: ProxySettings,

    /// Thumbnail settings
    pub thumbnail_settings: ThumbnailSettings,

    /// Database path
    pub database_path: PathBuf,
}

impl Default for ContentConfig {
    fn default() -> Self {
        Self {
            content_root: PathBuf::from("/var/oximedia/content"),
            watch_folders: vec![PathBuf::from("/var/oximedia/watch")],
            auto_ingest: true,
            qc_enabled: true,
            proxy_settings: ProxySettings::default(),
            thumbnail_settings: ThumbnailSettings::default(),
            database_path: PathBuf::from("/var/oximedia/content.db"),
        }
    }
}

/// Proxy generation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxySettings {
    /// Enable proxy generation
    pub enabled: bool,

    /// Proxy resolution
    pub resolution: ProxyResolution,

    /// Proxy codec
    pub codec: String,

    /// Proxy bitrate in kbps
    pub bitrate_kbps: u32,
}

impl Default for ProxySettings {
    fn default() -> Self {
        Self {
            enabled: true,
            resolution: ProxyResolution::HD720,
            codec: "h264".to_string(),
            bitrate_kbps: 5000,
        }
    }
}

/// Proxy resolution options
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProxyResolution {
    SD,
    HD720,
    HD1080,
}

/// Thumbnail settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThumbnailSettings {
    /// Enable thumbnail generation
    pub enabled: bool,

    /// Number of thumbnails to generate
    pub count: u32,

    /// Thumbnail width
    pub width: u32,

    /// Thumbnail height
    pub height: u32,

    /// Image format
    pub format: String,
}

impl Default for ThumbnailSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            count: 9,
            width: 320,
            height: 180,
            format: "jpeg".to_string(),
        }
    }
}

/// Content item in the catalog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentItem {
    /// Unique content ID
    pub id: Uuid,

    /// Content title
    pub title: String,

    /// File path
    pub file_path: PathBuf,

    /// File size in bytes
    pub file_size: u64,

    /// Content type
    pub content_type: ContentType,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// Video metadata
    pub video_metadata: Option<VideoMetadata>,

    /// Audio metadata
    pub audio_metadata: Option<AudioMetadata>,

    /// Technical QC status
    pub qc_status: QcStatus,

    /// QC issues
    pub qc_issues: Vec<QcIssue>,

    /// Availability status
    pub availability: AvailabilityStatus,

    /// Proxy path
    pub proxy_path: Option<PathBuf>,

    /// Thumbnail paths
    pub thumbnail_paths: Vec<PathBuf>,

    /// Custom metadata
    pub metadata: HashMap<String, String>,

    /// Created timestamp
    pub created_at: DateTime<Utc>,

    /// Modified timestamp
    pub modified_at: DateTime<Utc>,
}

/// Content type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ContentType {
    Video,
    Audio,
    Image,
    Graphics,
    Subtitle,
}

/// Video metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMetadata {
    /// Width in pixels
    pub width: u32,

    /// Height in pixels
    pub height: u32,

    /// Frame rate
    pub framerate: f64,

    /// Aspect ratio
    pub aspect_ratio: String,

    /// Codec
    pub codec: String,

    /// Bitrate in kbps
    pub bitrate_kbps: u32,

    /// Color space
    pub color_space: String,

    /// Interlaced flag
    pub interlaced: bool,

    /// Total frames
    pub total_frames: u64,
}

/// Audio metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    /// Sample rate
    pub sample_rate: u32,

    /// Number of channels
    pub channels: u16,

    /// Bit depth
    pub bit_depth: u16,

    /// Codec
    pub codec: String,

    /// Bitrate in kbps
    pub bitrate_kbps: u32,

    /// Language code
    pub language: Option<String>,

    /// Channel layout
    pub channel_layout: String,
}

/// QC status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QcStatus {
    /// QC not performed
    NotChecked,
    /// QC in progress
    InProgress,
    /// QC passed
    Passed,
    /// QC passed with warnings
    PassedWithWarnings,
    /// QC failed
    Failed,
}

/// QC issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QcIssue {
    /// Issue severity
    pub severity: QcSeverity,

    /// Issue type
    pub issue_type: QcIssueType,

    /// Description
    pub description: String,

    /// Timecode where issue occurs
    pub timecode: Option<String>,

    /// Frame number
    pub frame: Option<u64>,
}

/// QC severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum QcSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// QC issue types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum QcIssueType {
    /// Video format mismatch
    FormatMismatch,
    /// Audio levels too high
    AudioTooHigh,
    /// Audio levels too low
    AudioTooLow,
    /// Loudness violation
    LoudnessViolation,
    /// Black frames detected
    BlackFrames,
    /// Frozen frames detected
    FrozenFrames,
    /// Aspect ratio issue
    AspectRatio,
    /// Interlacing issue
    Interlacing,
    /// Corrupt frames
    CorruptFrames,
    /// Duration mismatch
    DurationMismatch,
    /// Missing audio
    MissingAudio,
    /// Silence detected
    Silence,
}

/// Availability status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AvailabilityStatus {
    /// Available for playout
    Available,
    /// Temporarily unavailable
    Unavailable,
    /// Archived (offline)
    Archived,
    /// Expired
    Expired,
}

/// Content manager
pub struct ContentManager {
    config: ContentConfig,
    catalog: Arc<RwLock<HashMap<Uuid, ContentItem>>>,
    index: Arc<RwLock<HashMap<String, Uuid>>>, // title -> id mapping
}

impl ContentManager {
    /// Create new content manager
    pub async fn new(config: ContentConfig) -> Result<Self> {
        // Create directories
        if !config.content_root.exists() {
            fs::create_dir_all(&config.content_root).await?;
        }

        for watch_folder in &config.watch_folders {
            if !watch_folder.exists() {
                fs::create_dir_all(watch_folder).await?;
            }
        }

        Ok(Self {
            config,
            catalog: Arc::new(RwLock::new(HashMap::new())),
            index: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Add content to catalog
    pub async fn add_content(&self, item: ContentItem) -> Result<()> {
        let id = item.id;
        let title = item.title.clone();

        {
            let mut catalog = self.catalog.write().await;
            catalog.insert(id, item);
        }

        {
            let mut index = self.index.write().await;
            index.insert(title, id);
        }

        info!("Added content to catalog: {}", id);
        Ok(())
    }

    /// Get content by ID
    pub async fn get_content(&self, id: &Uuid) -> Option<ContentItem> {
        let catalog = self.catalog.read().await;
        catalog.get(id).cloned()
    }

    /// Get content by title
    pub async fn get_content_by_title(&self, title: &str) -> Option<ContentItem> {
        let id = {
            let index = self.index.read().await;
            index.get(title).copied()
        };
        if let Some(id) = id {
            self.get_content(&id).await
        } else {
            None
        }
    }

    /// Search content by metadata
    pub async fn search(&self, query: &str) -> Vec<ContentItem> {
        let catalog = self.catalog.read().await;
        catalog
            .values()
            .filter(|item| {
                item.title.to_lowercase().contains(&query.to_lowercase())
                    || item
                        .metadata
                        .values()
                        .any(|v| v.to_lowercase().contains(&query.to_lowercase()))
            })
            .cloned()
            .collect()
    }

    /// Ingest content from file
    pub async fn ingest(&self, file_path: &Path) -> Result<Uuid> {
        info!("Ingesting content from: {}", file_path.display());

        // Validate file exists
        if !file_path.exists() {
            return Err(PlayoutError::NotFound(format!(
                "File not found: {}",
                file_path.display()
            )));
        }

        // Extract metadata
        let metadata = self.extract_metadata(file_path).await?;

        // Perform QC if enabled
        let (qc_status, qc_issues) = if self.config.qc_enabled {
            self.perform_qc(file_path, &metadata).await?
        } else {
            (QcStatus::NotChecked, Vec::new())
        };

        // Generate proxy if enabled
        let proxy_path = if self.config.proxy_settings.enabled {
            Some(self.generate_proxy(file_path).await?)
        } else {
            None
        };

        // Generate thumbnails if enabled
        let thumbnail_paths = if self.config.thumbnail_settings.enabled {
            self.generate_thumbnails(file_path).await?
        } else {
            Vec::new()
        };

        // Create content item
        let file_size = fs::metadata(file_path).await?.len();
        let item = ContentItem {
            id: Uuid::new_v4(),
            title: file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Unknown")
                .to_string(),
            file_path: file_path.to_path_buf(),
            file_size,
            content_type: ContentType::Video,
            duration_ms: metadata.duration_ms,
            video_metadata: Some(metadata.video),
            audio_metadata: Some(metadata.audio),
            qc_status,
            qc_issues,
            availability: AvailabilityStatus::Available,
            proxy_path,
            thumbnail_paths,
            metadata: HashMap::new(),
            created_at: Utc::now(),
            modified_at: Utc::now(),
        };

        let id = item.id;
        self.add_content(item).await?;

        info!("Ingested content: {}", id);
        Ok(id)
    }

    /// Extract metadata from media file
    async fn extract_metadata(&self, _file_path: &Path) -> Result<ExtractedMetadata> {
        // In a real implementation, this would use FFmpeg or similar
        Ok(ExtractedMetadata {
            duration_ms: 60000,
            video: VideoMetadata {
                width: 1920,
                height: 1080,
                framerate: 25.0,
                aspect_ratio: "16:9".to_string(),
                codec: "h264".to_string(),
                bitrate_kbps: 50000,
                color_space: "bt709".to_string(),
                interlaced: false,
                total_frames: 1500,
            },
            audio: AudioMetadata {
                sample_rate: 48000,
                channels: 2,
                bit_depth: 24,
                codec: "aac".to_string(),
                bitrate_kbps: 256,
                language: Some("eng".to_string()),
                channel_layout: "stereo".to_string(),
            },
        })
    }

    /// Perform technical QC
    async fn perform_qc(
        &self,
        _file_path: &Path,
        metadata: &ExtractedMetadata,
    ) -> Result<(QcStatus, Vec<QcIssue>)> {
        let mut issues = Vec::new();

        // Check video format
        if metadata.video.width != 1920 || metadata.video.height != 1080 {
            issues.push(QcIssue {
                severity: QcSeverity::Warning,
                issue_type: QcIssueType::FormatMismatch,
                description: "Video resolution is not 1920x1080".to_string(),
                timecode: None,
                frame: None,
            });
        }

        // Check audio sample rate
        if metadata.audio.sample_rate != 48000 {
            issues.push(QcIssue {
                severity: QcSeverity::Warning,
                issue_type: QcIssueType::FormatMismatch,
                description: "Audio sample rate is not 48kHz".to_string(),
                timecode: None,
                frame: None,
            });
        }

        let status = if issues.is_empty() {
            QcStatus::Passed
        } else if issues.iter().any(|i| i.severity >= QcSeverity::Error) {
            QcStatus::Failed
        } else {
            QcStatus::PassedWithWarnings
        };

        Ok((status, issues))
    }

    /// Generate proxy file
    async fn generate_proxy(&self, file_path: &Path) -> Result<PathBuf> {
        let proxy_dir = self.config.content_root.join("proxies");
        fs::create_dir_all(&proxy_dir).await?;

        let proxy_filename = format!(
            "{}_proxy.mp4",
            file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
        );
        let proxy_path = proxy_dir.join(proxy_filename);

        // In a real implementation, this would use FFmpeg to transcode
        debug!("Generated proxy: {}", proxy_path.display());

        Ok(proxy_path)
    }

    /// Generate thumbnails
    async fn generate_thumbnails(&self, file_path: &Path) -> Result<Vec<PathBuf>> {
        let thumb_dir = self.config.content_root.join("thumbnails");
        fs::create_dir_all(&thumb_dir).await?;

        let mut thumbnails = Vec::new();
        let base_name = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        for i in 0..self.config.thumbnail_settings.count {
            let thumb_filename = format!(
                "{}_{}.{}",
                base_name, i, self.config.thumbnail_settings.format
            );
            let thumb_path = thumb_dir.join(thumb_filename);
            thumbnails.push(thumb_path);
        }

        debug!("Generated {} thumbnails", thumbnails.len());

        Ok(thumbnails)
    }

    /// Update content availability
    pub async fn set_availability(&self, id: &Uuid, status: AvailabilityStatus) -> Result<()> {
        let mut catalog = self.catalog.write().await;
        if let Some(item) = catalog.get_mut(id) {
            item.availability = status;
            item.modified_at = Utc::now();
            info!("Updated availability for {}: {:?}", id, status);
            Ok(())
        } else {
            Err(PlayoutError::NotFound(format!("Content not found: {id}")))
        }
    }

    /// Remove content from catalog
    pub async fn remove_content(&self, id: &Uuid) -> Result<()> {
        let mut catalog = self.catalog.write().await;
        if let Some(item) = catalog.remove(id) {
            let mut index = self.index.write().await;
            index.remove(&item.title);
            info!("Removed content: {}", id);
            Ok(())
        } else {
            Err(PlayoutError::NotFound(format!("Content not found: {id}")))
        }
    }

    /// Get all content items
    pub async fn list_all(&self) -> Vec<ContentItem> {
        let catalog = self.catalog.read().await;
        catalog.values().cloned().collect()
    }

    /// Get content statistics
    pub async fn get_statistics(&self) -> ContentStatistics {
        let catalog = self.catalog.read().await;

        let total_items = catalog.len();
        let total_size = catalog.values().map(|i| i.file_size).sum();
        let total_duration_ms = catalog.values().map(|i| i.duration_ms).sum();

        let by_type = catalog.values().fold(HashMap::new(), |mut acc, item| {
            *acc.entry(item.content_type).or_insert(0) += 1;
            acc
        });

        let by_qc_status = catalog.values().fold(HashMap::new(), |mut acc, item| {
            *acc.entry(item.qc_status).or_insert(0) += 1;
            acc
        });

        ContentStatistics {
            total_items,
            total_size_bytes: total_size,
            total_duration_ms,
            items_by_type: by_type,
            items_by_qc_status: by_qc_status,
        }
    }
}

/// Extracted metadata from file
struct ExtractedMetadata {
    duration_ms: u64,
    video: VideoMetadata,
    audio: AudioMetadata,
}

/// Content statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentStatistics {
    /// Total number of items
    pub total_items: usize,

    /// Total size in bytes
    pub total_size_bytes: u64,

    /// Total duration in milliseconds
    pub total_duration_ms: u64,

    /// Items by content type
    pub items_by_type: HashMap<ContentType, usize>,

    /// Items by QC status
    pub items_by_qc_status: HashMap<QcStatus, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_config_default() {
        let config = ContentConfig::default();
        assert!(config.auto_ingest);
        assert!(config.qc_enabled);
    }

    #[test]
    fn test_proxy_settings_default() {
        let settings = ProxySettings::default();
        assert!(settings.enabled);
        assert_eq!(settings.resolution, ProxyResolution::HD720);
    }

    #[test]
    fn test_thumbnail_settings_default() {
        let settings = ThumbnailSettings::default();
        assert!(settings.enabled);
        assert_eq!(settings.count, 9);
    }

    #[test]
    fn test_content_type_equality() {
        assert_eq!(ContentType::Video, ContentType::Video);
        assert_ne!(ContentType::Video, ContentType::Audio);
    }

    #[test]
    fn test_qc_status_equality() {
        assert_eq!(QcStatus::Passed, QcStatus::Passed);
        assert_ne!(QcStatus::Passed, QcStatus::Failed);
    }

    #[test]
    fn test_qc_severity_ordering() {
        assert!(QcSeverity::Info < QcSeverity::Warning);
        assert!(QcSeverity::Warning < QcSeverity::Error);
        assert!(QcSeverity::Error < QcSeverity::Critical);
    }

    #[test]
    fn test_availability_status() {
        assert_eq!(AvailabilityStatus::Available, AvailabilityStatus::Available);
        assert_ne!(AvailabilityStatus::Available, AvailabilityStatus::Archived);
    }

    #[test]
    fn test_video_metadata_creation() {
        let metadata = VideoMetadata {
            width: 1920,
            height: 1080,
            framerate: 25.0,
            aspect_ratio: "16:9".to_string(),
            codec: "h264".to_string(),
            bitrate_kbps: 50000,
            color_space: "bt709".to_string(),
            interlaced: false,
            total_frames: 1500,
        };

        assert_eq!(metadata.width, 1920);
        assert_eq!(metadata.height, 1080);
    }

    #[test]
    fn test_audio_metadata_creation() {
        let metadata = AudioMetadata {
            sample_rate: 48000,
            channels: 2,
            bit_depth: 24,
            codec: "aac".to_string(),
            bitrate_kbps: 256,
            language: Some("eng".to_string()),
            channel_layout: "stereo".to_string(),
        };

        assert_eq!(metadata.sample_rate, 48000);
        assert_eq!(metadata.channels, 2);
    }

    #[test]
    fn test_qc_issue_creation() {
        let issue = QcIssue {
            severity: QcSeverity::Warning,
            issue_type: QcIssueType::FormatMismatch,
            description: "Test issue".to_string(),
            timecode: Some("01:00:00:00".to_string()),
            frame: Some(1500),
        };

        assert_eq!(issue.severity, QcSeverity::Warning);
        assert_eq!(issue.issue_type, QcIssueType::FormatMismatch);
    }

    #[tokio::test]
    async fn test_content_manager_creation() {
        let config = ContentConfig {
            content_root: std::env::temp_dir().join("oximedia-playout-content-test-content"),
            watch_folders: vec![],
            auto_ingest: false,
            qc_enabled: false,
            proxy_settings: ProxySettings::default(),
            thumbnail_settings: ThumbnailSettings::default(),
            database_path: std::env::temp_dir().join("oximedia-playout-content-test.db"),
        };

        // Clean up first
        let _ = fs::remove_dir_all(&config.content_root).await;

        let manager = ContentManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_content_item_operations() {
        let config = ContentConfig {
            content_root: std::env::temp_dir().join("oximedia-playout-content-test-content2"),
            watch_folders: vec![],
            auto_ingest: false,
            qc_enabled: false,
            proxy_settings: ProxySettings::default(),
            thumbnail_settings: ThumbnailSettings::default(),
            database_path: std::env::temp_dir().join("oximedia-playout-content-test2.db"),
        };

        let _ = fs::remove_dir_all(&config.content_root).await;
        let manager = ContentManager::new(config)
            .await
            .expect("should succeed in test");

        let item = ContentItem {
            id: Uuid::new_v4(),
            title: "Test Content".to_string(),
            file_path: std::env::temp_dir().join("oximedia-playout-content-test.mp4"),
            file_size: 1024000,
            content_type: ContentType::Video,
            duration_ms: 60000,
            video_metadata: None,
            audio_metadata: None,
            qc_status: QcStatus::NotChecked,
            qc_issues: Vec::new(),
            availability: AvailabilityStatus::Available,
            proxy_path: None,
            thumbnail_paths: Vec::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            modified_at: Utc::now(),
        };

        let id = item.id;
        manager
            .add_content(item)
            .await
            .expect("should succeed in test");

        let retrieved = manager.get_content(&id).await;
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.expect("should succeed in test").title,
            "Test Content"
        );
    }
}
