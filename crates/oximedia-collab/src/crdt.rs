//! CRDT (Conflict-free Replicated Data Type) implementation for collaborative editing
//!
//! This module provides Yjs-based document synchronization with support for
//! timeline operations, operational transformation, and conflict resolution.

use crate::{CollabError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use yrs::updates::decoder::Decode;
use yrs::updates::encoder::Encoder;
use yrs::{Doc, Map, ReadTxn, StateVector, Transact, Update, WriteTxn};

/// Operation type for timeline edits
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OpType {
    InsertClip,
    DeleteClip,
    MoveClip,
    UpdateClip,
    InsertTrack,
    DeleteTrack,
    MoveTrack,
    UpdateTrack,
    UpdateProperty,
}

/// Timeline operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineOp {
    pub id: Uuid,
    pub op_type: OpType,
    pub user_id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub data: serde_json::Value,
    pub version: u64,
}

impl TimelineOp {
    /// Create a new timeline operation
    pub fn new(op_type: OpType, user_id: Uuid, data: serde_json::Value, version: u64) -> Self {
        Self {
            id: Uuid::new_v4(),
            op_type,
            user_id,
            timestamp: chrono::Utc::now(),
            data,
            version,
        }
    }
}

/// Clip data for timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipData {
    pub id: Uuid,
    pub track_id: Uuid,
    pub start_time: f64,
    pub duration: f64,
    pub source_path: String,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Track data for timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackData {
    pub id: Uuid,
    pub name: String,
    pub track_type: String,
    pub index: usize,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Version vector for tracking causality
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VersionVector {
    versions: HashMap<Uuid, u64>,
}

impl VersionVector {
    /// Create a new version vector
    pub fn new() -> Self {
        Self {
            versions: HashMap::new(),
        }
    }

    /// Increment version for a user
    pub fn increment(&mut self, user_id: Uuid) -> u64 {
        let version = self.versions.entry(user_id).or_insert(0);
        *version += 1;
        *version
    }

    /// Get version for a user
    pub fn get(&self, user_id: Uuid) -> u64 {
        self.versions.get(&user_id).copied().unwrap_or(0)
    }

    /// Update with another version vector
    pub fn update(&mut self, other: &VersionVector) {
        for (user_id, version) in &other.versions {
            let current = self.versions.entry(*user_id).or_insert(0);
            *current = (*current).max(*version);
        }
    }

    /// Check if this vector happened before another
    pub fn happens_before(&self, other: &VersionVector) -> bool {
        self.versions
            .iter()
            .all(|(user_id, version)| other.get(*user_id) >= *version)
    }

    /// Check if vectors are concurrent
    pub fn is_concurrent(&self, other: &VersionVector) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }
}

/// Operation metadata for transformation
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct OpMetadata {
    op: TimelineOp,
    dependencies: Vec<Uuid>,
}

/// CRDT document for collaborative editing
pub struct CrdtDocument {
    doc: Doc,
    #[allow(dead_code)]
    session_id: Uuid,
    version_vector: Arc<RwLock<VersionVector>>,
    operations: Arc<RwLock<BTreeMap<u64, TimelineOp>>>,
    clips: Arc<RwLock<HashMap<Uuid, ClipData>>>,
    tracks: Arc<RwLock<HashMap<Uuid, TrackData>>>,
    pending_ops: Arc<RwLock<VecDeque<OpMetadata>>>,
    applied_ops: Arc<RwLock<HashMap<Uuid, bool>>>,
}

impl CrdtDocument {
    /// Create a new CRDT document
    pub fn new(session_id: Uuid) -> Self {
        Self {
            doc: Doc::new(),
            session_id,
            version_vector: Arc::new(RwLock::new(VersionVector::new())),
            operations: Arc::new(RwLock::new(BTreeMap::new())),
            clips: Arc::new(RwLock::new(HashMap::new())),
            tracks: Arc::new(RwLock::new(HashMap::new())),
            pending_ops: Arc::new(RwLock::new(VecDeque::new())),
            applied_ops: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Apply a timeline operation
    #[allow(clippy::too_many_arguments)]
    pub async fn apply_operation(&self, op: TimelineOp) -> Result<()> {
        // Check if already applied
        if self.applied_ops.read().await.contains_key(&op.id) {
            return Ok(());
        }

        match op.op_type {
            OpType::InsertClip => self.insert_clip(&op).await?,
            OpType::DeleteClip => self.delete_clip(&op).await?,
            OpType::MoveClip => self.move_clip(&op).await?,
            OpType::UpdateClip => self.update_clip(&op).await?,
            OpType::InsertTrack => self.insert_track(&op).await?,
            OpType::DeleteTrack => self.delete_track(&op).await?,
            OpType::MoveTrack => self.move_track(&op).await?,
            OpType::UpdateTrack => self.update_track(&op).await?,
            OpType::UpdateProperty => self.update_property(&op).await?,
        }

        // Mark as applied
        self.applied_ops.write().await.insert(op.id, true);
        self.operations.write().await.insert(op.version, op);

        Ok(())
    }

    /// Insert a clip into the timeline
    async fn insert_clip(&self, op: &TimelineOp) -> Result<()> {
        let clip: ClipData = serde_json::from_value(op.data.clone())
            .map_err(|e| CollabError::CrdtError(format!("Invalid clip data: {}", e)))?;

        self.clips.write().await.insert(clip.id, clip.clone());

        // Update Yrs document
        let mut txn = self.doc.transact_mut();
        let clips_map = txn.get_or_insert_map("clips");
        clips_map.insert(
            &mut txn,
            clip.id.to_string(),
            serde_json::to_string(&clip)
                .map_err(|e| CollabError::CrdtError(format!("Failed to serialize clip: {e}")))?,
        );

        Ok(())
    }

    /// Delete a clip from the timeline
    async fn delete_clip(&self, op: &TimelineOp) -> Result<()> {
        let clip_id: Uuid = serde_json::from_value(op.data["clip_id"].clone())
            .map_err(|e| CollabError::CrdtError(format!("Invalid clip ID: {}", e)))?;

        self.clips.write().await.remove(&clip_id);

        // Update Yrs document
        let mut txn = self.doc.transact_mut();
        let clips_map = txn.get_or_insert_map("clips");
        clips_map.remove(&mut txn, &clip_id.to_string());

        Ok(())
    }

    /// Move a clip in the timeline
    async fn move_clip(&self, op: &TimelineOp) -> Result<()> {
        let clip_id: Uuid = serde_json::from_value(op.data["clip_id"].clone())
            .map_err(|e| CollabError::CrdtError(format!("Invalid clip ID: {}", e)))?;
        let new_start: f64 = serde_json::from_value(op.data["start_time"].clone())
            .map_err(|e| CollabError::CrdtError(format!("Invalid start time: {}", e)))?;
        let new_track: Option<Uuid> = serde_json::from_value(op.data["track_id"].clone()).ok();

        let mut clips = self.clips.write().await;
        if let Some(clip) = clips.get_mut(&clip_id) {
            clip.start_time = new_start;
            if let Some(track_id) = new_track {
                clip.track_id = track_id;
            }

            // Update Yrs document
            let mut txn = self.doc.transact_mut();
            let clips_map = txn.get_or_insert_map("clips");
            clips_map.insert(
                &mut txn,
                clip_id.to_string(),
                serde_json::to_string(&*clip).map_err(|e| {
                    CollabError::CrdtError(format!("Failed to serialize clip: {e}"))
                })?,
            );
        }

        Ok(())
    }

    /// Update clip properties
    async fn update_clip(&self, op: &TimelineOp) -> Result<()> {
        let clip_id: Uuid = serde_json::from_value(op.data["clip_id"].clone())
            .map_err(|e| CollabError::CrdtError(format!("Invalid clip ID: {}", e)))?;
        let properties: HashMap<String, serde_json::Value> =
            serde_json::from_value(op.data["properties"].clone())
                .map_err(|e| CollabError::CrdtError(format!("Invalid properties: {}", e)))?;

        let mut clips = self.clips.write().await;
        if let Some(clip) = clips.get_mut(&clip_id) {
            for (key, value) in properties {
                clip.properties.insert(key, value);
            }

            // Update Yrs document
            let mut txn = self.doc.transact_mut();
            let clips_map = txn.get_or_insert_map("clips");
            clips_map.insert(
                &mut txn,
                clip_id.to_string(),
                serde_json::to_string(&*clip).map_err(|e| {
                    CollabError::CrdtError(format!("Failed to serialize clip: {e}"))
                })?,
            );
        }

        Ok(())
    }

    /// Insert a track into the timeline
    async fn insert_track(&self, op: &TimelineOp) -> Result<()> {
        let track: TrackData = serde_json::from_value(op.data.clone())
            .map_err(|e| CollabError::CrdtError(format!("Invalid track data: {}", e)))?;

        self.tracks.write().await.insert(track.id, track.clone());

        // Update Yrs document
        let mut txn = self.doc.transact_mut();
        let tracks_map = txn.get_or_insert_map("tracks");
        tracks_map.insert(
            &mut txn,
            track.id.to_string(),
            serde_json::to_string(&track)
                .map_err(|e| CollabError::CrdtError(format!("Failed to serialize track: {e}")))?,
        );

        Ok(())
    }

    /// Delete a track from the timeline
    async fn delete_track(&self, op: &TimelineOp) -> Result<()> {
        let track_id: Uuid = serde_json::from_value(op.data["track_id"].clone())
            .map_err(|e| CollabError::CrdtError(format!("Invalid track ID: {}", e)))?;

        // Remove all clips in this track
        let clips_to_remove: Vec<Uuid> = self
            .clips
            .read()
            .await
            .iter()
            .filter(|(_, clip)| clip.track_id == track_id)
            .map(|(id, _)| *id)
            .collect();

        let mut clips = self.clips.write().await;
        for clip_id in clips_to_remove {
            clips.remove(&clip_id);
        }

        self.tracks.write().await.remove(&track_id);

        // Update Yrs document
        let mut txn = self.doc.transact_mut();
        let tracks_map = txn.get_or_insert_map("tracks");
        tracks_map.remove(&mut txn, &track_id.to_string());

        Ok(())
    }

    /// Move a track in the timeline
    async fn move_track(&self, op: &TimelineOp) -> Result<()> {
        let track_id: Uuid = serde_json::from_value(op.data["track_id"].clone())
            .map_err(|e| CollabError::CrdtError(format!("Invalid track ID: {}", e)))?;
        let new_index: usize = serde_json::from_value(op.data["index"].clone())
            .map_err(|e| CollabError::CrdtError(format!("Invalid index: {}", e)))?;

        let mut tracks = self.tracks.write().await;
        if let Some(track) = tracks.get_mut(&track_id) {
            track.index = new_index;

            // Update Yrs document
            let mut txn = self.doc.transact_mut();
            let tracks_map = txn.get_or_insert_map("tracks");
            tracks_map.insert(
                &mut txn,
                track_id.to_string(),
                serde_json::to_string(&*track).map_err(|e| {
                    CollabError::CrdtError(format!("Failed to serialize track: {e}"))
                })?,
            );
        }

        Ok(())
    }

    /// Update track properties
    async fn update_track(&self, op: &TimelineOp) -> Result<()> {
        let track_id: Uuid = serde_json::from_value(op.data["track_id"].clone())
            .map_err(|e| CollabError::CrdtError(format!("Invalid track ID: {}", e)))?;
        let properties: HashMap<String, serde_json::Value> =
            serde_json::from_value(op.data["properties"].clone())
                .map_err(|e| CollabError::CrdtError(format!("Invalid properties: {}", e)))?;

        let mut tracks = self.tracks.write().await;
        if let Some(track) = tracks.get_mut(&track_id) {
            for (key, value) in properties {
                track.properties.insert(key, value);
            }

            // Update Yrs document
            let mut txn = self.doc.transact_mut();
            let tracks_map = txn.get_or_insert_map("tracks");
            tracks_map.insert(
                &mut txn,
                track_id.to_string(),
                serde_json::to_string(&*track).map_err(|e| {
                    CollabError::CrdtError(format!("Failed to serialize track: {e}"))
                })?,
            );
        }

        Ok(())
    }

    /// Update arbitrary property
    async fn update_property(&self, op: &TimelineOp) -> Result<()> {
        let key: String = serde_json::from_value(op.data["key"].clone())
            .map_err(|e| CollabError::CrdtError(format!("Invalid key: {}", e)))?;
        let value = op.data["value"].clone();

        // Update Yrs document
        let mut txn = self.doc.transact_mut();
        let props_map = txn.get_or_insert_map("properties");
        props_map.insert(
            &mut txn,
            key,
            serde_json::to_string(&value).map_err(|e| {
                CollabError::CrdtError(format!("Failed to serialize property value: {e}"))
            })?,
        );

        Ok(())
    }

    /// Transform operation against concurrent operations
    pub async fn transform_operation(
        &self,
        op: &TimelineOp,
        against: &[TimelineOp],
    ) -> Result<TimelineOp> {
        let mut transformed = op.clone();

        for concurrent_op in against {
            if self.should_transform(op, concurrent_op) {
                transformed = self
                    .apply_transformation(&transformed, concurrent_op)
                    .await?;
            }
        }

        Ok(transformed)
    }

    /// Check if operations should be transformed against each other
    fn should_transform(&self, op1: &TimelineOp, op2: &TimelineOp) -> bool {
        // Don't transform against self
        if op1.id == op2.id {
            return false;
        }

        // Check for overlapping effects
        match (&op1.op_type, &op2.op_type) {
            (OpType::MoveClip, OpType::MoveClip) => {
                // Check if same clip
                op1.data["clip_id"] == op2.data["clip_id"]
            }
            (OpType::DeleteClip, OpType::UpdateClip) | (OpType::UpdateClip, OpType::DeleteClip) => {
                op1.data["clip_id"] == op2.data["clip_id"]
            }
            (OpType::DeleteTrack, OpType::InsertClip) => {
                op2.data.get("track_id") == op1.data.get("track_id")
            }
            _ => false,
        }
    }

    /// Apply operational transformation
    async fn apply_transformation(
        &self,
        op: &TimelineOp,
        against: &TimelineOp,
    ) -> Result<TimelineOp> {
        let mut transformed = op.clone();

        match (&op.op_type, &against.op_type) {
            (OpType::MoveClip, OpType::MoveClip) => {
                // Resolve conflict by timestamp (later operation wins)
                if against.timestamp > op.timestamp {
                    // Keep our operation as-is
                } else if against.timestamp < op.timestamp {
                    // Our operation is later, we win
                } else {
                    // Same timestamp, use user ID as tiebreaker
                    if against.user_id > op.user_id {
                        // Other user wins
                        transformed.data["start_time"] = against.data["start_time"].clone();
                    }
                }
            }
            (OpType::UpdateClip, OpType::DeleteClip) => {
                // If clip was deleted, drop the update
                return Err(CollabError::InvalidOperation("Clip deleted".to_string()));
            }
            (OpType::InsertClip, OpType::DeleteTrack) => {
                // If track was deleted, create new track or move to different track
                let default_track = Uuid::new_v4();
                transformed.data["track_id"] = serde_json::json!(default_track);
            }
            _ => {}
        }

        Ok(transformed)
    }

    /// Resolve conflicts between operations
    pub async fn resolve_conflict(&self, op1: &TimelineOp, op2: &TimelineOp) -> Result<TimelineOp> {
        // Last-write-wins based on timestamp
        if op1.timestamp > op2.timestamp {
            Ok(op1.clone())
        } else if op2.timestamp > op1.timestamp {
            Ok(op2.clone())
        } else {
            // Timestamps equal, use user ID as tiebreaker (deterministic)
            if op1.user_id > op2.user_id {
                Ok(op1.clone())
            } else {
                Ok(op2.clone())
            }
        }
    }

    /// Get current state vector
    pub fn get_state_vector(&self) -> StateVector {
        let txn = self.doc.transact();
        txn.state_vector()
    }

    /// Get update since a state vector
    pub fn get_update(&self, state_vector: &StateVector) -> Result<Vec<u8>> {
        let txn = self.doc.transact();
        let mut encoder = yrs::updates::encoder::EncoderV1::new();
        txn.encode_state_as_update(state_vector, &mut encoder);
        Ok(encoder.to_vec())
    }

    /// Apply an update from another client
    pub fn apply_update(&self, update: &[u8]) -> Result<()> {
        let update = Update::decode_v1(update)
            .map_err(|e| CollabError::CrdtError(format!("Failed to decode update: {}", e)))?;
        let mut txn = self.doc.transact_mut();
        let _ = txn.apply_update(update);
        Ok(())
    }

    /// Get all clips
    pub async fn get_clips(&self) -> Vec<ClipData> {
        self.clips.read().await.values().cloned().collect()
    }

    /// Get all tracks
    pub async fn get_tracks(&self) -> Vec<TrackData> {
        let mut tracks: Vec<_> = self.tracks.read().await.values().cloned().collect();
        tracks.sort_by_key(|t| t.index);
        tracks
    }

    /// Get clip by ID
    pub async fn get_clip(&self, clip_id: Uuid) -> Option<ClipData> {
        self.clips.read().await.get(&clip_id).cloned()
    }

    /// Get track by ID
    pub async fn get_track(&self, track_id: Uuid) -> Option<TrackData> {
        self.tracks.read().await.get(&track_id).cloned()
    }

    /// Run garbage collection
    pub async fn garbage_collect(&self, keep_versions: usize) -> Result<()> {
        let mut ops = self.operations.write().await;
        let total = ops.len();

        if total > keep_versions {
            let to_remove = total - keep_versions;
            let keys_to_remove: Vec<u64> = ops.keys().take(to_remove).copied().collect();

            for key in keys_to_remove {
                ops.remove(&key);
            }
        }

        // Clean up applied ops tracking (keep last 10000)
        let mut applied = self.applied_ops.write().await;
        if applied.len() > 10000 {
            let to_remove = applied.len() - 10000;
            let keys: Vec<Uuid> = applied.keys().take(to_remove).copied().collect();
            for key in keys {
                applied.remove(&key);
            }
        }

        Ok(())
    }

    /// Merge state from another document
    pub async fn merge(&self, other: &CrdtDocument) -> Result<()> {
        // Get updates from other document
        let state_vector = self.get_state_vector();
        let update = other.get_update(&state_vector)?;
        self.apply_update(&update)?;

        // Merge version vectors
        let other_vv = other.version_vector.read().await;
        self.version_vector.write().await.update(&other_vv);

        // Merge operations
        let other_ops = other.operations.read().await;
        let mut ops = self.operations.write().await;
        for (version, op) in other_ops.iter() {
            ops.entry(*version).or_insert_with(|| op.clone());
        }

        Ok(())
    }

    /// Get version vector
    pub async fn get_version_vector(&self) -> VersionVector {
        self.version_vector.read().await.clone()
    }

    /// Increment version for a user
    pub async fn increment_version(&self, user_id: Uuid) -> u64 {
        self.version_vector.write().await.increment(user_id)
    }

    /// Export document state
    pub async fn export_state(&self) -> Result<Vec<u8>> {
        let txn = self.doc.transact();
        let state_vector = StateVector::default();
        let mut encoder = yrs::updates::encoder::EncoderV1::new();
        txn.encode_state_as_update(&state_vector, &mut encoder);
        Ok(encoder.to_vec())
    }

    /// Import document state
    pub fn import_state(&self, state: &[u8]) -> Result<()> {
        self.apply_update(state)
    }

    /// Get operation count
    pub async fn operation_count(&self) -> usize {
        self.operations.read().await.len()
    }

    /// Clear all data
    pub async fn clear(&self) -> Result<()> {
        self.clips.write().await.clear();
        self.tracks.write().await.clear();
        self.operations.write().await.clear();
        self.applied_ops.write().await.clear();
        self.pending_ops.write().await.clear();
        Ok(())
    }
}

/// CRDT manager for handling multiple documents
pub struct CrdtManager {
    documents: Arc<RwLock<HashMap<Uuid, Arc<CrdtDocument>>>>,
}

impl CrdtManager {
    /// Create a new CRDT manager
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new document
    pub async fn create_document(&self, session_id: Uuid) -> Arc<CrdtDocument> {
        let doc = Arc::new(CrdtDocument::new(session_id));
        self.documents.write().await.insert(session_id, doc.clone());
        doc
    }

    /// Get a document
    pub async fn get_document(&self, session_id: Uuid) -> Option<Arc<CrdtDocument>> {
        self.documents.read().await.get(&session_id).cloned()
    }

    /// Remove a document
    pub async fn remove_document(&self, session_id: Uuid) -> Result<()> {
        self.documents.write().await.remove(&session_id);
        Ok(())
    }

    /// Run garbage collection on all documents
    pub async fn garbage_collect_all(&self, keep_versions: usize) -> Result<()> {
        let docs = self.documents.read().await;
        for doc in docs.values() {
            doc.garbage_collect(keep_versions).await?;
        }
        Ok(())
    }
}

impl Default for CrdtManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_version_vector() {
        let mut vv1 = VersionVector::new();
        let mut vv2 = VersionVector::new();

        let user1 = Uuid::new_v4();
        let user2 = Uuid::new_v4();

        vv1.increment(user1);
        vv1.increment(user1);
        vv2.increment(user2);

        assert!(!vv1.happens_before(&vv2));
        assert!(vv1.is_concurrent(&vv2));

        vv1.update(&vv2);
        assert!(vv2.happens_before(&vv1));
    }

    #[tokio::test]
    async fn test_insert_clip() {
        let doc = CrdtDocument::new(Uuid::new_v4());
        let clip = ClipData {
            id: Uuid::new_v4(),
            track_id: Uuid::new_v4(),
            start_time: 0.0,
            duration: 5.0,
            source_path: "test.mp4".to_string(),
            properties: HashMap::new(),
        };

        let op = TimelineOp::new(
            OpType::InsertClip,
            Uuid::new_v4(),
            serde_json::to_value(&clip).expect("collab test operation should succeed"),
            1,
        );

        doc.apply_operation(op)
            .await
            .expect("collab test operation should succeed");

        let clips = doc.get_clips().await;
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].id, clip.id);
    }

    #[tokio::test]
    async fn test_move_clip() {
        let doc = CrdtDocument::new(Uuid::new_v4());
        let clip = ClipData {
            id: Uuid::new_v4(),
            track_id: Uuid::new_v4(),
            start_time: 0.0,
            duration: 5.0,
            source_path: "test.mp4".to_string(),
            properties: HashMap::new(),
        };

        // Insert clip
        let insert_op = TimelineOp::new(
            OpType::InsertClip,
            Uuid::new_v4(),
            serde_json::to_value(&clip).expect("collab test operation should succeed"),
            1,
        );
        doc.apply_operation(insert_op)
            .await
            .expect("collab test operation should succeed");

        // Move clip
        let move_data = serde_json::json!({
            "clip_id": clip.id,
            "start_time": 10.0,
        });
        let move_op = TimelineOp::new(OpType::MoveClip, Uuid::new_v4(), move_data, 2);
        doc.apply_operation(move_op)
            .await
            .expect("collab test operation should succeed");

        let moved_clip = doc
            .get_clip(clip.id)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(moved_clip.start_time, 10.0);
    }

    #[tokio::test]
    async fn test_conflict_resolution() {
        let doc = CrdtDocument::new(Uuid::new_v4());
        let user1 = Uuid::new_v4();
        let user2 = Uuid::new_v4();

        let op1 = TimelineOp::new(
            OpType::MoveClip,
            user1,
            serde_json::json!({"clip_id": Uuid::new_v4(), "start_time": 5.0}),
            1,
        );

        // Create op2 with later timestamp
        let mut op2 = TimelineOp::new(
            OpType::MoveClip,
            user2,
            serde_json::json!({"clip_id": op1.data["clip_id"], "start_time": 10.0}),
            2,
        );
        op2.timestamp = op1.timestamp + chrono::Duration::seconds(1);

        let resolved = doc
            .resolve_conflict(&op1, &op2)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(resolved.id, op2.id);
    }

    #[tokio::test]
    async fn test_garbage_collection() {
        let doc = CrdtDocument::new(Uuid::new_v4());

        // Create 100 operations
        for i in 0..100 {
            let op = TimelineOp::new(
                OpType::UpdateProperty,
                Uuid::new_v4(),
                serde_json::json!({"key": format!("prop{}", i), "value": i}),
                i,
            );
            doc.apply_operation(op)
                .await
                .expect("collab test operation should succeed");
        }

        assert_eq!(doc.operation_count().await, 100);

        // Run GC, keep only 10 versions
        doc.garbage_collect(10)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(doc.operation_count().await, 10);
    }

    #[tokio::test]
    async fn test_crdt_manager() {
        let manager = CrdtManager::new();
        let session_id = Uuid::new_v4();

        let _doc = manager.create_document(session_id).await;
        assert!(manager.get_document(session_id).await.is_some());

        manager
            .remove_document(session_id)
            .await
            .expect("collab test operation should succeed");
        assert!(manager.get_document(session_id).await.is_none());
    }
}

/// Lamport timestamp for causal ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LamportTimestamp {
    counter: u64,
    node_id: Uuid,
}

impl LamportTimestamp {
    /// Create a new Lamport timestamp
    pub fn new(counter: u64, node_id: Uuid) -> Self {
        Self { counter, node_id }
    }

    /// Increment timestamp
    pub fn increment(&mut self) {
        self.counter += 1;
    }

    /// Update timestamp based on received timestamp
    pub fn update(&mut self, other: &LamportTimestamp) {
        self.counter = self.counter.max(other.counter) + 1;
    }

    /// Compare timestamps for causal ordering
    pub fn happens_before(&self, other: &LamportTimestamp) -> bool {
        self.counter < other.counter
            || (self.counter == other.counter && self.node_id < other.node_id)
    }
}

/// Vector clock for distributed causal ordering
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorClock {
    clocks: HashMap<Uuid, u64>,
}

impl VectorClock {
    /// Create a new vector clock
    pub fn new() -> Self {
        Self {
            clocks: HashMap::new(),
        }
    }

    /// Increment clock for a node
    pub fn increment(&mut self, node_id: Uuid) -> u64 {
        let clock = self.clocks.entry(node_id).or_insert(0);
        *clock += 1;
        *clock
    }

    /// Get clock value for a node
    pub fn get(&self, node_id: Uuid) -> u64 {
        self.clocks.get(&node_id).copied().unwrap_or(0)
    }

    /// Update clock with another clock
    pub fn merge(&mut self, other: &VectorClock) {
        for (node_id, clock) in &other.clocks {
            let current = self.clocks.entry(*node_id).or_insert(0);
            *current = (*current).max(*clock);
        }
    }

    /// Check if this clock happens before another
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut strictly_less = false;

        // Check all keys in self
        for (node_id, clock) in &self.clocks {
            let other_clock = other.get(*node_id);
            if *clock > other_clock {
                return false;
            }
            if *clock < other_clock {
                strictly_less = true;
            }
        }

        // Check keys that only exist in other (self has 0 for those)
        for (node_id, other_clock) in &other.clocks {
            if !self.clocks.contains_key(node_id) && *other_clock > 0 {
                strictly_less = true;
            }
        }

        strictly_less
    }

    /// Check if clocks are concurrent
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new()
    }
}

/// Three-way merge strategy
pub struct ThreeWayMerge {
    base: Option<serde_json::Value>,
}

impl ThreeWayMerge {
    /// Create a new three-way merge
    pub fn new(base: Option<serde_json::Value>) -> Self {
        Self { base }
    }

    /// Perform three-way merge
    pub fn merge(
        &self,
        local: &serde_json::Value,
        remote: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        match &self.base {
            Some(base) => self.merge_with_base(base, local, remote),
            None => self.merge_without_base(local, remote),
        }
    }

    fn merge_with_base(
        &self,
        base: &serde_json::Value,
        local: &serde_json::Value,
        remote: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        if local == remote {
            return Ok(local.clone());
        }

        if local == base {
            return Ok(remote.clone());
        }

        if remote == base {
            return Ok(local.clone());
        }

        // Both changed - need conflict resolution
        match (local, remote) {
            (serde_json::Value::Object(l), serde_json::Value::Object(r)) => {
                let mut result = serde_json::Map::new();
                let base_obj = base.as_object();

                // Get all keys
                let mut all_keys: std::collections::HashSet<String> = l.keys().cloned().collect();
                all_keys.extend(r.keys().cloned());

                for key in all_keys {
                    let base_val = base_obj.and_then(|b| b.get(&key));
                    let local_val = l.get(&key);
                    let remote_val = r.get(&key);

                    let merged = match (base_val, local_val, remote_val) {
                        (Some(b), Some(l), Some(r)) => {
                            let merge = ThreeWayMerge::new(Some(b.clone()));
                            merge.merge(l, r)?
                        }
                        (_, Some(l), None) => l.clone(),
                        (_, None, Some(r)) => r.clone(),
                        (None, Some(l), Some(r)) => {
                            if l == r {
                                l.clone()
                            } else {
                                r.clone() // Remote wins
                            }
                        }
                        _ => continue,
                    };

                    result.insert(key, merged);
                }

                Ok(serde_json::Value::Object(result))
            }
            (serde_json::Value::Array(_l), serde_json::Value::Array(r)) => {
                // For arrays, use operational transformation
                Ok(serde_json::Value::Array(r.clone()))
            }
            _ => Ok(remote.clone()), // Remote wins for scalar conflicts
        }
    }

    fn merge_without_base(
        &self,
        local: &serde_json::Value,
        remote: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        if local == remote {
            Ok(local.clone())
        } else {
            // No base - use simple strategy
            Ok(remote.clone())
        }
    }
}

/// Snapshot manager for efficient synchronization
pub struct SnapshotManager {
    snapshots: Arc<RwLock<BTreeMap<u64, Vec<u8>>>>,
    max_snapshots: usize,
}

impl SnapshotManager {
    /// Create a new snapshot manager
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            snapshots: Arc::new(RwLock::new(BTreeMap::new())),
            max_snapshots,
        }
    }

    /// Create a snapshot at version
    pub async fn create_snapshot(&self, version: u64, data: Vec<u8>) -> Result<()> {
        let mut snapshots = self.snapshots.write().await;

        snapshots.insert(version, data);

        // Keep only recent snapshots
        while snapshots.len() > self.max_snapshots {
            if let Some(first_key) = snapshots.keys().next().copied() {
                snapshots.remove(&first_key);
            }
        }

        Ok(())
    }

    /// Get closest snapshot to version
    pub async fn get_snapshot(&self, version: u64) -> Option<(u64, Vec<u8>)> {
        let snapshots = self.snapshots.read().await;

        snapshots
            .range(..=version)
            .next_back()
            .map(|(v, data)| (*v, data.clone()))
    }

    /// List all snapshot versions
    pub async fn list_versions(&self) -> Vec<u64> {
        self.snapshots.read().await.keys().copied().collect()
    }

    /// Clear all snapshots
    pub async fn clear(&self) {
        self.snapshots.write().await.clear();
    }
}

/// Operation batch for efficient processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationBatch {
    pub id: Uuid,
    pub operations: Vec<TimelineOp>,
    pub start_version: u64,
    pub end_version: u64,
    pub user_id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl OperationBatch {
    /// Create a new operation batch
    pub fn new(operations: Vec<TimelineOp>, user_id: Uuid) -> Self {
        let start_version = operations.first().map(|op| op.version).unwrap_or(0);
        let end_version = operations.last().map(|op| op.version).unwrap_or(0);

        Self {
            id: Uuid::new_v4(),
            operations,
            start_version,
            end_version,
            user_id,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Get operation count
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Merge with another batch
    pub fn merge(&mut self, other: OperationBatch) {
        self.operations.extend(other.operations);
        self.end_version = self.end_version.max(other.end_version);
    }
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictStrategy {
    /// Last write wins based on timestamp
    LastWriteWins,
    /// First write wins
    FirstWriteWins,
    /// Higher user ID wins
    UserIdWins,
    /// Manual resolution required
    Manual,
}

/// Conflict resolver
pub struct ConflictResolver {
    strategy: ConflictStrategy,
}

impl ConflictResolver {
    /// Create a new conflict resolver
    pub fn new(strategy: ConflictStrategy) -> Self {
        Self { strategy }
    }

    /// Resolve conflict between two operations
    pub fn resolve(&self, op1: &TimelineOp, op2: &TimelineOp) -> Result<TimelineOp> {
        match self.strategy {
            ConflictStrategy::LastWriteWins => {
                if op1.timestamp >= op2.timestamp {
                    Ok(op1.clone())
                } else {
                    Ok(op2.clone())
                }
            }
            ConflictStrategy::FirstWriteWins => {
                if op1.timestamp <= op2.timestamp {
                    Ok(op1.clone())
                } else {
                    Ok(op2.clone())
                }
            }
            ConflictStrategy::UserIdWins => {
                if op1.user_id >= op2.user_id {
                    Ok(op1.clone())
                } else {
                    Ok(op2.clone())
                }
            }
            ConflictStrategy::Manual => Err(CollabError::InvalidOperation(
                "Manual conflict resolution required".to_string(),
            )),
        }
    }

    /// Check if operations conflict
    pub fn conflicts(&self, op1: &TimelineOp, op2: &TimelineOp) -> bool {
        if op1.id == op2.id {
            return false;
        }

        match (&op1.op_type, &op2.op_type) {
            (OpType::MoveClip, OpType::MoveClip)
            | (OpType::UpdateClip, OpType::UpdateClip)
            | (OpType::DeleteClip, OpType::UpdateClip)
            | (OpType::UpdateClip, OpType::DeleteClip) => {
                op1.data.get("clip_id") == op2.data.get("clip_id")
            }
            (OpType::MoveTrack, OpType::MoveTrack)
            | (OpType::UpdateTrack, OpType::UpdateTrack)
            | (OpType::DeleteTrack, OpType::UpdateTrack)
            | (OpType::UpdateTrack, OpType::DeleteTrack) => {
                op1.data.get("track_id") == op2.data.get("track_id")
            }
            _ => false,
        }
    }
}

/// Delta compression for efficient sync
pub struct DeltaCompression;

impl DeltaCompression {
    /// Compress delta between two states
    pub fn compress(old_state: &[u8], new_state: &[u8]) -> Result<Vec<u8>> {
        // Simple delta: only send differences
        if old_state == new_state {
            return Ok(Vec::new());
        }

        // For now, just return the new state
        // In production, use a proper delta compression algorithm
        Ok(new_state.to_vec())
    }

    /// Apply compressed delta to state
    pub fn apply_delta(state: &[u8], delta: &[u8]) -> Result<Vec<u8>> {
        if delta.is_empty() {
            return Ok(state.to_vec());
        }

        // For now, just return the delta as new state
        Ok(delta.to_vec())
    }
}

/// Causal order tracker
pub struct CausalOrderTracker {
    vector_clock: Arc<RwLock<VectorClock>>,
    pending_ops: Arc<RwLock<Vec<(TimelineOp, VectorClock)>>>,
}

impl CausalOrderTracker {
    /// Create a new causal order tracker
    pub fn new() -> Self {
        Self {
            vector_clock: Arc::new(RwLock::new(VectorClock::new())),
            pending_ops: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add operation with its vector clock
    pub async fn add_operation(
        &self,
        op: TimelineOp,
        clock: VectorClock,
    ) -> Result<Vec<TimelineOp>> {
        let mut current_clock = self.vector_clock.write().await;

        // Check if operation can be applied
        if clock.happens_before(&current_clock) || clock == *current_clock {
            // Already applied
            return Ok(Vec::new());
        }

        // Check if we can apply this operation
        let mut can_apply = true;
        for (node_id, op_clock) in &clock.clocks {
            let current = current_clock.get(*node_id);
            if *op_clock > current + 1 {
                can_apply = false;
                break;
            }
        }

        if can_apply {
            // Apply this operation
            current_clock.merge(&clock);
            current_clock.increment(op.user_id);

            // Check pending operations
            let mut applied = vec![op.clone()];
            let mut pending = self.pending_ops.write().await;

            let mut i = 0;
            while i < pending.len() {
                let (_pending_op, pending_clock) = &pending[i];

                let mut can_apply_pending = true;
                for (node_id, op_clock) in &pending_clock.clocks {
                    let current = current_clock.get(*node_id);
                    if *op_clock > current + 1 {
                        can_apply_pending = false;
                        break;
                    }
                }

                if can_apply_pending {
                    let (pending_op, pending_clock) = pending.remove(i);
                    current_clock.merge(&pending_clock);
                    current_clock.increment(pending_op.user_id);
                    applied.push(pending_op);
                } else {
                    i += 1;
                }
            }

            Ok(applied)
        } else {
            // Add to pending
            self.pending_ops.write().await.push((op, clock));
            Ok(Vec::new())
        }
    }

    /// Get current vector clock
    pub async fn get_clock(&self) -> VectorClock {
        self.vector_clock.read().await.clone()
    }

    /// Get pending operation count
    pub async fn pending_count(&self) -> usize {
        self.pending_ops.read().await.len()
    }
}

impl Default for CausalOrderTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge coordinator for multi-way merges
pub struct MergeCoordinator {
    documents: Arc<RwLock<HashMap<Uuid, Arc<CrdtDocument>>>>,
}

impl MergeCoordinator {
    /// Create a new merge coordinator
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a document
    pub async fn register_document(&self, id: Uuid, doc: Arc<CrdtDocument>) {
        self.documents.write().await.insert(id, doc);
    }

    /// Perform multi-way merge
    pub async fn multi_way_merge(&self, doc_ids: Vec<Uuid>) -> Result<Arc<CrdtDocument>> {
        let docs = self.documents.read().await;

        if doc_ids.is_empty() {
            return Err(CollabError::InvalidOperation(
                "No documents to merge".to_string(),
            ));
        }

        if doc_ids.len() == 1 {
            return docs
                .get(&doc_ids[0])
                .cloned()
                .ok_or(CollabError::InvalidOperation(
                    "Document not found".to_string(),
                ));
        }

        // Get first document as base
        let base_doc = docs.get(&doc_ids[0]).ok_or(CollabError::InvalidOperation(
            "Base document not found".to_string(),
        ))?;

        // Merge all other documents into base
        for doc_id in doc_ids.iter().skip(1) {
            let doc = docs.get(doc_id).ok_or(CollabError::InvalidOperation(
                "Document not found".to_string(),
            ))?;

            base_doc.merge(doc).await?;
        }

        Ok(base_doc.clone())
    }

    /// Get document count
    pub async fn document_count(&self) -> usize {
        self.documents.read().await.len()
    }
}

impl Default for MergeCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    #[test]
    fn test_lamport_timestamp() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();

        let mut ts1 = LamportTimestamp::new(0, node1);
        let mut ts2 = LamportTimestamp::new(0, node2);

        ts1.increment();
        assert_eq!(ts1.counter, 1);

        ts2.update(&ts1);
        assert_eq!(ts2.counter, 2);

        assert!(ts1.happens_before(&ts2));
    }

    #[test]
    fn test_vector_clock() {
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();

        let mut clock1 = VectorClock::new();
        let mut clock2 = VectorClock::new();

        clock1.increment(node1);
        clock1.increment(node1);

        clock2.increment(node2);

        assert!(clock1.is_concurrent(&clock2));

        clock2.merge(&clock1);
        assert!(clock1.happens_before(&clock2));
    }

    #[tokio::test]
    async fn test_three_way_merge() {
        let base = serde_json::json!({"name": "test", "value": 10});
        let local = serde_json::json!({"name": "test", "value": 20});
        let remote = serde_json::json!({"name": "updated", "value": 10});

        let merger = ThreeWayMerge::new(Some(base));
        let result = merger
            .merge(&local, &remote)
            .expect("collab test operation should succeed");

        assert_eq!(result["name"], "updated");
        assert_eq!(result["value"], 20);
    }

    #[tokio::test]
    async fn test_snapshot_manager() {
        let manager = SnapshotManager::new(3);

        manager
            .create_snapshot(1, vec![1, 2, 3])
            .await
            .expect("collab test operation should succeed");
        manager
            .create_snapshot(5, vec![4, 5, 6])
            .await
            .expect("collab test operation should succeed");
        manager
            .create_snapshot(10, vec![7, 8, 9])
            .await
            .expect("collab test operation should succeed");

        let (version, data) = manager
            .get_snapshot(7)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(version, 5);
        assert_eq!(data, vec![4, 5, 6]);

        let versions = manager.list_versions().await;
        assert_eq!(versions.len(), 3);
    }

    #[test]
    fn test_operation_batch() {
        let user_id = Uuid::new_v4();
        let op1 = TimelineOp::new(OpType::InsertClip, user_id, serde_json::json!({}), 1);
        let op2 = TimelineOp::new(OpType::MoveClip, user_id, serde_json::json!({}), 2);

        let batch = OperationBatch::new(vec![op1, op2], user_id);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.start_version, 1);
        assert_eq!(batch.end_version, 2);
    }

    #[test]
    fn test_conflict_resolver() {
        let user1 = Uuid::new_v4();
        let user2 = Uuid::new_v4();

        let op1 = TimelineOp::new(
            OpType::MoveClip,
            user1,
            serde_json::json!({"clip_id": "123"}),
            1,
        );
        let mut op2 = TimelineOp::new(
            OpType::MoveClip,
            user2,
            serde_json::json!({"clip_id": "123"}),
            2,
        );
        op2.timestamp = op1.timestamp + chrono::Duration::seconds(1);

        let resolver = ConflictResolver::new(ConflictStrategy::LastWriteWins);
        assert!(resolver.conflicts(&op1, &op2));

        let winner = resolver
            .resolve(&op1, &op2)
            .expect("collab test operation should succeed");
        assert_eq!(winner.user_id, user2);
    }

    #[tokio::test]
    async fn test_causal_order_tracker() {
        let tracker = CausalOrderTracker::new();
        let user_id = Uuid::new_v4();

        let mut clock1 = VectorClock::new();
        clock1.increment(user_id);

        let op1 = TimelineOp::new(OpType::InsertClip, user_id, serde_json::json!({}), 1);
        let applied = tracker
            .add_operation(op1, clock1)
            .await
            .expect("collab test operation should succeed");

        assert_eq!(applied.len(), 1);
        assert_eq!(tracker.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_merge_coordinator() {
        let coordinator = MergeCoordinator::new();
        let session1 = Uuid::new_v4();
        let session2 = Uuid::new_v4();

        let doc1 = Arc::new(CrdtDocument::new(session1));
        let doc2 = Arc::new(CrdtDocument::new(session2));

        coordinator.register_document(session1, doc1).await;
        coordinator.register_document(session2, doc2).await;

        assert_eq!(coordinator.document_count().await, 2);

        let merged = coordinator
            .multi_way_merge(vec![session1, session2])
            .await
            .expect("collab test operation should succeed");
        assert!(Arc::strong_count(&merged) > 0);
    }
}
