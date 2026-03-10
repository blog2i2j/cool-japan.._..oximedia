//! Media catalog for managing and searching media files.

use crate::database::Database;
use crate::error::ConformResult;
use crate::types::MediaFile;
use parking_lot::RwLock;
use std::path::Path;
use std::sync::Arc;

/// Media catalog for managing scanned media files.
pub struct MediaCatalog {
    /// Database backend.
    db: Database,
    /// In-memory cache of media files.
    cache: Arc<RwLock<Vec<MediaFile>>>,
}

impl MediaCatalog {
    /// Create a new media catalog with a database.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub fn new<P: AsRef<Path>>(db_path: P) -> ConformResult<Self> {
        let db = Database::open(db_path)?;
        Ok(Self {
            db,
            cache: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Create an in-memory catalog (for testing).
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created.
    pub fn in_memory() -> ConformResult<Self> {
        let db = Database::in_memory()?;
        Ok(Self {
            db,
            cache: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Add a media file to the catalog.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be added to the database.
    pub fn add(&self, media: MediaFile) -> ConformResult<()> {
        self.db.add_media_file(&media)?;
        self.cache.write().push(media);
        Ok(())
    }

    /// Add multiple media files to the catalog.
    ///
    /// # Errors
    ///
    /// Returns an error if any file cannot be added.
    pub fn add_many(&self, media_files: Vec<MediaFile>) -> ConformResult<()> {
        for media in &media_files {
            self.db.add_media_file(media)?;
        }
        self.cache.write().extend(media_files);
        Ok(())
    }

    /// Find media files by filename.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn find_by_filename(&self, filename: &str) -> ConformResult<Vec<MediaFile>> {
        self.db.find_by_filename(filename)
    }

    /// Find media files by path pattern.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn find_by_path_pattern(&self, pattern: &str) -> ConformResult<Vec<MediaFile>> {
        self.db.find_by_path_pattern(pattern)
    }

    /// Find media files by MD5 checksum.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn find_by_md5(&self, md5: &str) -> ConformResult<Vec<MediaFile>> {
        self.db.find_by_md5(md5)
    }

    /// Get all media files from the catalog.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_all(&self) -> ConformResult<Vec<MediaFile>> {
        self.db.get_all_media_files()
    }

    /// Get the number of media files in the catalog.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn count(&self) -> ConformResult<usize> {
        self.db.get_media_file_count()
    }

    /// Clear all media files from the catalog.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub fn clear(&self) -> ConformResult<()> {
        self.db.clear_media_files()?;
        self.cache.write().clear();
        Ok(())
    }

    /// Reload the cache from the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn reload_cache(&self) -> ConformResult<()> {
        let media_files = self.db.get_all_media_files()?;
        *self.cache.write() = media_files;
        Ok(())
    }

    /// Search media files by various criteria.
    pub fn search(&self, query: &SearchQuery) -> ConformResult<Vec<MediaFile>> {
        let all_media = self.cache.read();
        let mut results = Vec::new();

        for media in all_media.iter() {
            if query.matches(media) {
                results.push(media.clone());
            }
        }

        Ok(results)
    }
}

/// Search query for finding media files.
#[derive(Debug, Clone, Default)]
pub struct SearchQuery {
    /// Filename pattern (substring match).
    pub filename: Option<String>,
    /// Minimum duration in seconds.
    pub min_duration: Option<f64>,
    /// Maximum duration in seconds.
    pub max_duration: Option<f64>,
    /// Required width.
    pub width: Option<u32>,
    /// Required height.
    pub height: Option<u32>,
}

impl SearchQuery {
    /// Create a new empty search query.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set filename pattern.
    #[must_use]
    pub fn with_filename(mut self, filename: String) -> Self {
        self.filename = Some(filename);
        self
    }

    /// Set duration range.
    #[must_use]
    pub fn with_duration_range(mut self, min: f64, max: f64) -> Self {
        self.min_duration = Some(min);
        self.max_duration = Some(max);
        self
    }

    /// Set resolution.
    #[must_use]
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Check if a media file matches this query.
    #[must_use]
    pub fn matches(&self, media: &MediaFile) -> bool {
        if let Some(ref pattern) = self.filename {
            if !media.filename.contains(pattern) {
                return false;
            }
        }

        if let Some(min_dur) = self.min_duration {
            if media.duration.map_or(true, |d| d < min_dur) {
                return false;
            }
        }

        if let Some(max_dur) = self.max_duration {
            if media.duration.map_or(true, |d| d > max_dur) {
                return false;
            }
        }

        if let Some(w) = self.width {
            if media.width != Some(w) {
                return false;
            }
        }

        if let Some(h) = self.height {
            if media.height != Some(h) {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_catalog_creation() {
        let catalog = MediaCatalog::in_memory().expect("catalog should be valid");
        assert_eq!(catalog.count().expect("count should succeed"), 0);
    }

    #[test]
    fn test_add_media() {
        let catalog = MediaCatalog::in_memory().expect("catalog should be valid");
        let media = MediaFile::new(PathBuf::from("/test/file.mov"));
        catalog.add(media).expect("add should succeed");
        assert_eq!(catalog.count().expect("count should succeed"), 1);
    }

    #[test]
    fn test_find_by_filename() {
        let catalog = MediaCatalog::in_memory().expect("catalog should be valid");
        let media = MediaFile::new(PathBuf::from("/test/file.mov"));
        catalog.add(media).expect("add should succeed");

        let found = catalog
            .find_by_filename("file.mov")
            .expect("found should be valid");
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_clear_catalog() {
        let catalog = MediaCatalog::in_memory().expect("catalog should be valid");
        let media1 = MediaFile::new(PathBuf::from("/test/file1.mov"));
        let media2 = MediaFile::new(PathBuf::from("/test/file2.mov"));
        catalog.add(media1).expect("add should succeed");
        catalog.add(media2).expect("add should succeed");
        assert_eq!(catalog.count().expect("count should succeed"), 2);

        catalog.clear().expect("clear should succeed");
        assert_eq!(catalog.count().expect("count should succeed"), 0);
    }

    #[test]
    fn test_search_query() {
        let catalog = MediaCatalog::in_memory().expect("catalog should be valid");

        let mut media1 = MediaFile::new(PathBuf::from("/test/video.mov"));
        media1.duration = Some(10.0);
        media1.width = Some(1920);
        media1.height = Some(1080);

        let mut media2 = MediaFile::new(PathBuf::from("/test/audio.wav"));
        media2.duration = Some(5.0);

        catalog.add(media1).expect("add should succeed");
        catalog.add(media2).expect("add should succeed");

        let query = SearchQuery::new().with_duration_range(8.0, 12.0);
        let results = catalog.search(&query).expect("results should be valid");
        assert_eq!(results.len(), 1);
        assert!(results[0].filename.contains("video"));
    }
}
