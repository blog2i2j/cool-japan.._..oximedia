//! Index builder for creating and populating search indices.

use crate::error::SearchResult;
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;

/// Document to be indexed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDocument {
    /// Unique asset ID
    pub asset_id: Uuid,
    /// File path
    pub file_path: String,
    /// Title
    pub title: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Keywords
    pub keywords: Vec<String>,
    /// Categories
    pub categories: Vec<String>,
    /// MIME type
    pub mime_type: Option<String>,
    /// File format
    pub format: Option<String>,
    /// Video codec
    pub codec: Option<String>,
    /// Resolution (e.g., "1920x1080")
    pub resolution: Option<String>,
    /// Duration in milliseconds
    pub duration_ms: Option<i64>,
    /// File size in bytes
    pub file_size: Option<i64>,
    /// Bitrate in bits per second
    pub bitrate: Option<i64>,
    /// Frame rate
    pub framerate: Option<f64>,
    /// Created timestamp (unix)
    pub created_at: i64,
    /// Modified timestamp (unix)
    pub modified_at: i64,
    /// Transcript text
    pub transcript: Option<String>,
    /// OCR text extracted from video
    pub ocr_text: Option<String>,
    /// Visual features (perceptual hash, etc.)
    pub visual_features: Option<VisualFeatures>,
    /// Audio fingerprint
    pub audio_fingerprint: Option<Vec<u8>>,
    /// Detected faces
    pub faces: Option<Vec<FaceDescriptor>>,
    /// Dominant colors
    pub dominant_colors: Option<Vec<Color>>,
    /// Scene tags
    pub scene_tags: Vec<String>,
    /// Detected objects
    pub detected_objects: Vec<String>,
    /// Custom metadata
    pub metadata: serde_json::Value,
}

/// Visual features for similarity search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualFeatures {
    /// Perceptual hash
    pub phash: Vec<u8>,
    /// Color histogram
    pub color_histogram: Vec<f32>,
    /// Edge histogram
    pub edge_histogram: Vec<f32>,
    /// Texture features
    pub texture_features: Vec<f32>,
}

/// Face descriptor for face search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceDescriptor {
    /// Face embedding vector
    pub embedding: Vec<f32>,
    /// Bounding box (x, y, width, height)
    pub bbox: (f32, f32, f32, f32),
    /// Confidence score
    pub confidence: f32,
    /// Person ID (if identified)
    pub person_id: Option<Uuid>,
}

/// Color representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Color {
    /// Red component (0-255)
    pub r: u8,
    /// Green component (0-255)
    pub g: u8,
    /// Blue component (0-255)
    pub b: u8,
    /// Percentage of image
    pub percentage: f32,
}

/// Index builder for constructing search indices
pub struct IndexBuilder {
    index_path: String,
    documents: Vec<IndexDocument>,
}

impl IndexBuilder {
    /// Create a new index builder
    #[must_use]
    pub fn new(index_path: &str) -> Self {
        Self {
            index_path: index_path.to_string(),
            documents: Vec::new(),
        }
    }

    /// Add a document to the builder
    pub fn add_document(&mut self, doc: IndexDocument) {
        self.documents.push(doc);
    }

    /// Build the index from accumulated documents
    ///
    /// # Errors
    ///
    /// Returns an error if index building fails
    pub fn build(&self) -> SearchResult<()> {
        let path = Path::new(&self.index_path);

        // Create index directory
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }

        // Build indices for different search types
        self.build_text_index()?;
        self.build_visual_index()?;
        self.build_audio_index()?;
        self.build_face_index()?;
        self.build_ocr_index()?;
        self.build_color_index()?;

        Ok(())
    }

    /// Build text search index
    fn build_text_index(&self) -> SearchResult<()> {
        // Text index building implementation
        // This would use Tantivy to create full-text index
        Ok(())
    }

    /// Build visual similarity index
    fn build_visual_index(&self) -> SearchResult<()> {
        // Visual index building implementation
        // This would create perceptual hash index and feature vectors
        Ok(())
    }

    /// Build audio fingerprint index
    fn build_audio_index(&self) -> SearchResult<()> {
        // Audio fingerprint index building
        // This would create audio fingerprint database
        Ok(())
    }

    /// Build face recognition index
    fn build_face_index(&self) -> SearchResult<()> {
        // Face index building
        // This would create face embedding index
        Ok(())
    }

    /// Build OCR text index
    fn build_ocr_index(&self) -> SearchResult<()> {
        // OCR text index building
        Ok(())
    }

    /// Build color search index
    fn build_color_index(&self) -> SearchResult<()> {
        // Color index building
        Ok(())
    }

    /// Get number of documents
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    /// Clear all documents
    pub fn clear(&mut self) {
        self.documents.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_index() -> String {
        std::env::temp_dir()
            .join("oximedia-search-index-builder-test_index")
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_index_builder_new() {
        let builder = IndexBuilder::new(&tmp_index());
        assert_eq!(builder.document_count(), 0);
    }

    #[test]
    fn test_add_document() {
        let mut builder = IndexBuilder::new(&tmp_index());
        let doc = IndexDocument {
            asset_id: Uuid::new_v4(),
            file_path: "/test/video.mp4".to_string(),
            title: Some("Test Video".to_string()),
            description: None,
            keywords: vec!["test".to_string()],
            categories: vec![],
            mime_type: Some("video/mp4".to_string()),
            format: Some("mp4".to_string()),
            codec: Some("h264".to_string()),
            resolution: Some("1920x1080".to_string()),
            duration_ms: Some(60000),
            file_size: Some(10_000_000),
            bitrate: Some(5_000_000),
            framerate: Some(30.0),
            created_at: 1234567890,
            modified_at: 1234567890,
            transcript: None,
            ocr_text: None,
            visual_features: None,
            audio_fingerprint: None,
            faces: None,
            dominant_colors: None,
            scene_tags: vec![],
            detected_objects: vec![],
            metadata: serde_json::json!({}),
        };

        builder.add_document(doc);
        assert_eq!(builder.document_count(), 1);
    }

    #[test]
    fn test_clear() {
        let mut builder = IndexBuilder::new(&tmp_index());
        let doc = IndexDocument {
            asset_id: Uuid::new_v4(),
            file_path: "/test/video.mp4".to_string(),
            title: Some("Test Video".to_string()),
            description: None,
            keywords: vec![],
            categories: vec![],
            mime_type: None,
            format: None,
            codec: None,
            resolution: None,
            duration_ms: None,
            file_size: None,
            bitrate: None,
            framerate: None,
            created_at: 0,
            modified_at: 0,
            transcript: None,
            ocr_text: None,
            visual_features: None,
            audio_fingerprint: None,
            faces: None,
            dominant_colors: None,
            scene_tags: vec![],
            detected_objects: vec![],
            metadata: serde_json::json!({}),
        };

        builder.add_document(doc);
        builder.clear();
        assert_eq!(builder.document_count(), 0);
    }
}
