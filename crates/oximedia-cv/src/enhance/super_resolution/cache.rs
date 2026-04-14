//! Model cache for reusing loaded ONNX sessions.

use crate::error::{CvError, CvResult};
use oxionnx::Session;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Model cache for reusing loaded ONNX sessions.
///
/// Caches ONNX Runtime sessions to avoid reloading models from disk.
/// Thread-safe and can be shared across multiple processing threads.
#[derive(Clone)]
pub struct ModelCache {
    cache: Arc<Mutex<HashMap<PathBuf, Arc<Mutex<Session>>>>>,
}

impl ModelCache {
    /// Create a new empty model cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::<PathBuf, Arc<Mutex<Session>>>::new())),
        }
    }

    /// Get or load a model from cache.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    pub fn get_or_load(&self, path: impl AsRef<Path>) -> CvResult<Arc<Mutex<Session>>> {
        let path = path.as_ref().to_path_buf();
        let mut cache = self
            .cache
            .lock()
            .map_err(|e| CvError::model_load(format!("Cache lock error: {e}")))?;

        if let Some(session) = cache.get(&path) {
            return Ok(Arc::clone(session));
        }

        // Load new session
        let session = Session::builder()
            .with_optimization_level(oxionnx::OptLevel::All)
            .load(&path)
            .map_err(|e| CvError::model_load(format!("Failed to load model: {e}")))?;

        let session = Arc::new(Mutex::new(session));
        cache.insert(path, Arc::clone(&session));
        Ok(session)
    }

    /// Clear all cached models.
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    /// Get the number of cached models.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.lock().map_or(0, |cache| cache.len())
    }

    /// Check if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_cache() {
        let cache = ModelCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        let cache2 = ModelCache::default();
        assert!(cache2.is_empty());
    }
}
