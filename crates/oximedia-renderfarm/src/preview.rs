// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Preview rendering system.

use crate::job::JobId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Preview quality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreviewQuality {
    /// Low quality (draft)
    Low,
    /// Medium quality
    Medium,
    /// High quality
    High,
}

/// Preview request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewRequest {
    /// Job ID
    pub job_id: JobId,
    /// Frame numbers to preview
    pub frames: Vec<u32>,
    /// Quality level
    pub quality: PreviewQuality,
    /// Resolution scale (0.0 to 1.0)
    pub resolution_scale: f64,
}

/// Preview result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewResult {
    /// Job ID
    pub job_id: JobId,
    /// Frame number
    pub frame: u32,
    /// Output path
    pub output_path: PathBuf,
    /// Render time (seconds)
    pub render_time: f64,
    /// Created at
    pub created_at: DateTime<Utc>,
}

/// Preview cache
pub struct PreviewCache {
    previews: Vec<PreviewResult>,
    max_size: usize,
}

impl PreviewCache {
    /// Create a new preview cache
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            previews: Vec::new(),
            max_size,
        }
    }

    /// Add preview to cache
    pub fn add(&mut self, preview: PreviewResult) {
        if self.previews.len() >= self.max_size {
            self.previews.remove(0);
        }
        self.previews.push(preview);
    }

    /// Get preview
    #[must_use]
    pub fn get(&self, job_id: JobId, frame: u32) -> Option<&PreviewResult> {
        self.previews
            .iter()
            .find(|p| p.job_id == job_id && p.frame == frame)
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.previews.clear();
    }
}

impl Default for PreviewCache {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preview_cache_creation() {
        let cache = PreviewCache::new(10);
        assert_eq!(cache.previews.len(), 0);
    }

    #[test]
    fn test_preview_cache_add() {
        let mut cache = PreviewCache::new(10);
        let job_id = JobId::new();

        let preview = PreviewResult {
            job_id,
            frame: 1,
            output_path: std::env::temp_dir().join("oximedia-renderfarm-preview_1.jpg"),
            render_time: 0.5,
            created_at: Utc::now(),
        };

        cache.add(preview);
        assert_eq!(cache.previews.len(), 1);
    }

    #[test]
    fn test_preview_cache_get() {
        let mut cache = PreviewCache::new(10);
        let job_id = JobId::new();

        let preview = PreviewResult {
            job_id,
            frame: 1,
            output_path: std::env::temp_dir().join("oximedia-renderfarm-preview_1.jpg"),
            render_time: 0.5,
            created_at: Utc::now(),
        };

        cache.add(preview);
        assert!(cache.get(job_id, 1).is_some());
        assert!(cache.get(job_id, 2).is_none());
    }
}
