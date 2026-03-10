//! Main clip manager that integrates all functionality.

use crate::clip::{Clip, ClipId};
use crate::database::ClipDatabase;
use crate::error::{ClipError, ClipResult};
use crate::export::{ClipListExporter, EdlExporter};
use crate::group::{Bin, BinId, Collection, CollectionId, Folder, FolderId, SmartCollection};
use crate::import::{BatchImporter, MediaScanner};
use crate::marker::{Marker, MarkerId, MarkerManager};
use crate::proxy::{ProxyLink, ProxyManager};
use crate::search::{ClipFilter, SearchEngine};
use crate::take::{Take, TakeManager};
use oximedia_core::types::Rational;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Main clip management system.
pub struct ClipManager {
    database: ClipDatabase,
    marker_manager: MarkerManager,
    take_manager: TakeManager,
    proxy_manager: ProxyManager,
    #[allow(dead_code)]
    search_engine: SearchEngine,
    bins: HashMap<BinId, Bin>,
    folders: HashMap<FolderId, Folder>,
    collections: HashMap<CollectionId, Collection>,
    smart_collections: Vec<SmartCollection>,
}

impl ClipManager {
    /// Creates a new clip manager.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be initialized.
    pub async fn new(database_url: impl AsRef<str>) -> ClipResult<Self> {
        let database = ClipDatabase::new(database_url).await?;

        Ok(Self {
            database,
            marker_manager: MarkerManager::new(),
            take_manager: TakeManager::new(),
            proxy_manager: ProxyManager::new(),
            search_engine: SearchEngine::new(),
            bins: HashMap::new(),
            folders: HashMap::new(),
            collections: HashMap::new(),
            smart_collections: Vec::new(),
        })
    }

    // Clip operations

    /// Adds a clip to the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip cannot be saved.
    pub async fn add_clip(&self, clip: Clip) -> ClipResult<ClipId> {
        let clip_id = clip.id;
        self.database.save_clip(&clip).await?;
        Ok(clip_id)
    }

    /// Gets a clip by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip is not found.
    pub async fn get_clip(&self, clip_id: &ClipId) -> ClipResult<Clip> {
        self.database.get_clip(clip_id).await
    }

    /// Updates a clip.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip cannot be updated.
    pub async fn update_clip(&self, clip: Clip) -> ClipResult<()> {
        self.database.save_clip(&clip).await
    }

    /// Deletes a clip.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip cannot be deleted.
    pub async fn delete_clip(&self, clip_id: &ClipId) -> ClipResult<()> {
        self.database.delete_clip(clip_id).await
    }

    /// Gets all clips.
    ///
    /// # Errors
    ///
    /// Returns an error if clips cannot be loaded.
    pub async fn get_all_clips(&self) -> ClipResult<Vec<Clip>> {
        self.database.get_all_clips().await
    }

    /// Returns the number of clips.
    ///
    /// # Errors
    ///
    /// Returns an error if the count fails.
    pub async fn clip_count(&self) -> ClipResult<i64> {
        self.database.count_clips().await
    }

    // Search operations

    /// Searches clips by query string.
    ///
    /// # Errors
    ///
    /// Returns an error if the search fails.
    pub async fn search(&self, query: &str) -> ClipResult<Vec<Clip>> {
        self.database.search_clips(query).await
    }

    /// Filters clips using advanced criteria.
    ///
    /// # Errors
    ///
    /// Returns an error if the filter operation fails.
    pub async fn filter(&self, filter: &ClipFilter) -> ClipResult<Vec<Clip>> {
        let clips = self.database.get_all_clips().await?;
        Ok(filter.apply(&clips).into_iter().cloned().collect())
    }

    // Bin operations

    /// Creates a new bin.
    pub fn create_bin(&mut self, name: impl Into<String>) -> BinId {
        let bin = Bin::new(name);
        let bin_id = bin.id;
        self.bins.insert(bin_id, bin);
        bin_id
    }

    /// Gets a bin by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the bin is not found.
    pub fn get_bin(&self, bin_id: &BinId) -> ClipResult<&Bin> {
        self.bins
            .get(bin_id)
            .ok_or_else(|| ClipError::BinNotFound(bin_id.to_string()))
    }

    /// Gets a mutable bin by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the bin is not found.
    pub fn get_bin_mut(&mut self, bin_id: &BinId) -> ClipResult<&mut Bin> {
        self.bins
            .get_mut(bin_id)
            .ok_or_else(|| ClipError::BinNotFound(bin_id.to_string()))
    }

    /// Adds a clip to a bin.
    ///
    /// # Errors
    ///
    /// Returns an error if the bin is not found.
    pub fn add_clip_to_bin(&mut self, bin_id: &BinId, clip_id: ClipId) -> ClipResult<()> {
        let bin = self.get_bin_mut(bin_id)?;
        bin.add_clip(clip_id);
        Ok(())
    }

    /// Lists all bins.
    #[must_use]
    pub fn list_bins(&self) -> Vec<&Bin> {
        self.bins.values().collect()
    }

    // Folder operations

    /// Creates a new folder.
    pub fn create_folder(&mut self, name: impl Into<String>) -> FolderId {
        let folder = Folder::new(name);
        let folder_id = folder.id;
        self.folders.insert(folder_id, folder);
        folder_id
    }

    /// Creates a child folder.
    pub fn create_child_folder(
        &mut self,
        name: impl Into<String>,
        parent_id: FolderId,
    ) -> FolderId {
        let folder = Folder::new_child(name, parent_id);
        let folder_id = folder.id;
        self.folders.insert(folder_id, folder);
        folder_id
    }

    /// Gets a folder by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the folder is not found.
    pub fn get_folder(&self, folder_id: &FolderId) -> ClipResult<&Folder> {
        self.folders
            .get(folder_id)
            .ok_or_else(|| ClipError::FolderNotFound(folder_id.to_string()))
    }

    // Collection operations

    /// Creates a new collection.
    pub fn create_collection(&mut self, name: impl Into<String>) -> CollectionId {
        let collection = Collection::new(name);
        let collection_id = collection.id;
        self.collections.insert(collection_id, collection);
        collection_id
    }

    /// Gets a collection by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection is not found.
    pub fn get_collection(&self, collection_id: &CollectionId) -> ClipResult<&Collection> {
        self.collections
            .get(collection_id)
            .ok_or_else(|| ClipError::CollectionNotFound(collection_id.to_string()))
    }

    /// Adds a clip to a collection.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection is not found.
    pub fn add_clip_to_collection(
        &mut self,
        collection_id: &CollectionId,
        clip_id: ClipId,
    ) -> ClipResult<()> {
        let collection = self
            .collections
            .get_mut(collection_id)
            .ok_or_else(|| ClipError::CollectionNotFound(collection_id.to_string()))?;
        collection.add_clip(clip_id);
        Ok(())
    }

    // Smart collection operations

    /// Creates a new smart collection.
    pub fn create_smart_collection(&mut self, smart_collection: SmartCollection) {
        self.smart_collections.push(smart_collection);
    }

    /// Updates all smart collections.
    ///
    /// # Errors
    ///
    /// Returns an error if clips cannot be loaded.
    pub async fn update_smart_collections(&mut self) -> ClipResult<()> {
        let clips = self.database.get_all_clips().await?;

        for smart_collection in &mut self.smart_collections {
            smart_collection.update(&clips);
        }

        Ok(())
    }

    // Marker operations

    /// Adds a marker to a clip.
    pub fn add_marker(&mut self, clip_id: ClipId, marker: Marker) {
        self.marker_manager.add_marker(clip_id, marker);
    }

    /// Gets markers for a clip.
    #[must_use]
    pub fn get_markers(&self, clip_id: &ClipId) -> Vec<&Marker> {
        self.marker_manager.get_markers(clip_id)
    }

    /// Removes a marker.
    ///
    /// # Errors
    ///
    /// Returns an error if the marker is not found.
    pub fn remove_marker(&mut self, clip_id: &ClipId, marker_id: &MarkerId) -> ClipResult<()> {
        self.marker_manager.remove_marker(clip_id, marker_id)
    }

    // Take operations

    /// Adds a take.
    pub fn add_take(&mut self, take: Take) {
        self.take_manager.add_take(take);
    }

    /// Gets takes for a scene.
    #[must_use]
    pub fn get_scene_takes(&self, scene: &str) -> Vec<&Take> {
        self.take_manager.get_scene_takes(scene)
    }

    /// Gets takes for a clip.
    #[must_use]
    pub fn get_clip_takes(&self, clip_id: &ClipId) -> Vec<&Take> {
        self.take_manager.get_clip_takes(clip_id)
    }

    // Proxy operations

    /// Adds a proxy link.
    pub fn add_proxy(&mut self, link: ProxyLink) {
        self.proxy_manager.add_link(link);
    }

    /// Gets proxy links for a clip.
    #[must_use]
    pub fn get_proxies(&self, clip_id: &ClipId) -> Vec<&ProxyLink> {
        self.proxy_manager.get_links(clip_id)
    }

    /// Gets the best proxy for a clip.
    #[must_use]
    pub fn get_best_proxy(&self, clip_id: &ClipId) -> Option<&ProxyLink> {
        self.proxy_manager.get_best_proxy(clip_id)
    }

    // Import operations

    /// Scans a directory for media files.
    ///
    /// # Errors
    ///
    /// Returns an error if the scan fails.
    pub async fn scan_directory(&self, path: impl AsRef<Path>) -> ClipResult<Vec<Clip>> {
        let scanner = MediaScanner::new();
        scanner.scan(path).await
    }

    /// Imports clips from file paths.
    #[must_use]
    pub fn import_clips(&self, paths: Vec<PathBuf>) -> Vec<Clip> {
        let importer = BatchImporter::default();
        importer.import(paths)
    }

    // Export operations

    /// Exports clips to CSV.
    ///
    /// # Errors
    ///
    /// Returns an error if the export fails.
    pub async fn export_csv(&self, clip_ids: &[ClipId]) -> ClipResult<String> {
        let mut clips = Vec::new();
        for clip_id in clip_ids {
            clips.push(self.database.get_clip(clip_id).await?);
        }

        let exporter = ClipListExporter::new();
        exporter.to_csv(&clips)
    }

    /// Exports clips to EDL.
    ///
    /// # Errors
    ///
    /// Returns an error if the export fails.
    pub async fn export_edl(
        &self,
        clip_ids: &[ClipId],
        frame_rate: Rational,
    ) -> ClipResult<String> {
        let mut clips = Vec::new();
        for clip_id in clip_ids {
            clips.push(self.database.get_clip(clip_id).await?);
        }

        let exporter = EdlExporter::new(frame_rate);
        exporter.to_edl(&clips)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_clip_manager() {
        let manager = ClipManager::new(":memory:")
            .await
            .expect("new should succeed");
        let count = manager
            .clip_count()
            .await
            .expect("clip_count should succeed");
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_add_and_get_clip() {
        let manager = ClipManager::new(":memory:")
            .await
            .expect("new should succeed");

        let clip = Clip::new(PathBuf::from("/test.mov"));
        let clip_id = manager
            .add_clip(clip.clone())
            .await
            .expect("add_clip should succeed");

        let loaded = manager
            .get_clip(&clip_id)
            .await
            .expect("get_clip should succeed");
        assert_eq!(loaded.id, clip_id);
    }

    #[tokio::test]
    async fn test_bins() {
        let mut manager = ClipManager::new(":memory:")
            .await
            .expect("new should succeed");

        let bin_id = manager.create_bin("Test Bin");
        let bin = manager.get_bin(&bin_id).expect("get_bin should succeed");
        assert_eq!(bin.name, "Test Bin");

        let clip = Clip::new(PathBuf::from("/test.mov"));
        let clip_id = manager
            .add_clip(clip)
            .await
            .expect("add_clip should succeed");

        manager
            .add_clip_to_bin(&bin_id, clip_id)
            .expect("add_clip_to_bin should succeed");
        let bin = manager.get_bin(&bin_id).expect("get_bin should succeed");
        assert_eq!(bin.count(), 1);
    }

    #[tokio::test]
    async fn test_search() {
        let manager = ClipManager::new(":memory:")
            .await
            .expect("new should succeed");

        let mut clip = Clip::new(PathBuf::from("/test.mov"));
        clip.set_name("Interview");
        manager
            .add_clip(clip)
            .await
            .expect("operation should succeed");

        let results = manager
            .search("interview")
            .await
            .expect("search should succeed");
        assert_eq!(results.len(), 1);
    }
}
