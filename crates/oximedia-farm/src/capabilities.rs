//! Worker capability advertisement and querying.
//!
//! [`WorkerCapabilities`] tracks the set of capabilities a farm worker exposes:
//! supported codecs, GPU hardware models, and arbitrary named capabilities.
//! It is used by [`crate::affinity::JobAffinityRule`] to match jobs to workers
//! that have the required hardware or software support.
//!
//! # Example
//!
//! ```
//! use oximedia_farm::capabilities::WorkerCapabilities;
//!
//! let mut caps = WorkerCapabilities::new(42);
//! caps.add_codec("av1");
//! caps.add_codec("h264");
//! caps.add_gpu("nvidia-a100");
//!
//! assert!(caps.supports("av1"));
//! assert!(caps.supports("h264"));
//! assert!(caps.supports("nvidia-a100"));
//! assert!(!caps.supports("vp9"));
//! ```

use std::collections::HashSet;

/// Capability set for a single farm worker.
///
/// Capabilities are stored as lowercase strings so that `"H264"` and `"h264"`
/// are treated as identical.  Both codec names and GPU model strings are stored
/// in the same flat capability set, enabling a single [`supports`](Self::supports)
/// call for all capability types.
#[derive(Debug, Clone)]
pub struct WorkerCapabilities {
    /// Worker identifier.
    worker_id: u64,
    /// Flat set of capability strings (codecs, GPU models, custom tags).
    capabilities: HashSet<String>,
    /// Codec names (subset of `capabilities`, kept for introspection).
    codecs: Vec<String>,
    /// GPU model names (subset of `capabilities`, kept for introspection).
    gpus: Vec<String>,
}

impl WorkerCapabilities {
    /// Create a new, empty capability set for the given `worker_id`.
    #[must_use]
    pub fn new(worker_id: u64) -> Self {
        Self {
            worker_id,
            capabilities: HashSet::new(),
            codecs: Vec::new(),
            gpus: Vec::new(),
        }
    }

    /// Return the worker ID this capability set belongs to.
    #[must_use]
    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    /// Register a supported codec.
    ///
    /// The codec name is normalised to lowercase before storage.
    pub fn add_codec(&mut self, codec: &str) {
        let normalised = codec.to_lowercase();
        if self.capabilities.insert(normalised.clone()) {
            self.codecs.push(normalised);
        }
    }

    /// Register a supported GPU model.
    ///
    /// The model string is normalised to lowercase before storage.
    pub fn add_gpu(&mut self, model: &str) {
        let normalised = model.to_lowercase();
        if self.capabilities.insert(normalised.clone()) {
            self.gpus.push(normalised);
        }
    }

    /// Register an arbitrary named capability (e.g. `"hdr10"`, `"dolby-vision"`).
    ///
    /// The capability name is normalised to lowercase.
    pub fn add_capability(&mut self, cap: &str) {
        self.capabilities.insert(cap.to_lowercase());
    }

    /// Return `true` when the worker supports the given capability.
    ///
    /// The query is case-insensitive.
    #[must_use]
    pub fn supports(&self, cap: &str) -> bool {
        self.capabilities.contains(&cap.to_lowercase())
    }

    /// Return the list of registered codecs (normalised lowercase).
    #[must_use]
    pub fn codecs(&self) -> &[String] {
        &self.codecs
    }

    /// Return the list of registered GPU model strings (normalised lowercase).
    #[must_use]
    pub fn gpus(&self) -> &[String] {
        &self.gpus
    }

    /// Return all registered capabilities (codecs + GPUs + custom).
    #[must_use]
    pub fn all_capabilities(&self) -> &HashSet<String> {
        &self.capabilities
    }

    /// Return the number of registered capabilities.
    #[must_use]
    pub fn capability_count(&self) -> usize {
        self.capabilities.len()
    }

    /// Return `true` if no capabilities have been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.capabilities.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let caps = WorkerCapabilities::new(1);
        assert!(caps.is_empty());
        assert_eq!(caps.worker_id(), 1);
    }

    #[test]
    fn test_add_codec_supports() {
        let mut caps = WorkerCapabilities::new(1);
        caps.add_codec("av1");
        assert!(caps.supports("av1"));
        assert!(!caps.supports("vp9"));
    }

    #[test]
    fn test_add_gpu_supports() {
        let mut caps = WorkerCapabilities::new(2);
        caps.add_gpu("nvidia-a100");
        assert!(caps.supports("nvidia-a100"));
        assert!(!caps.supports("amd-rdna2"));
    }

    #[test]
    fn test_case_insensitive() {
        let mut caps = WorkerCapabilities::new(1);
        caps.add_codec("H264");
        assert!(caps.supports("h264"));
        assert!(caps.supports("H264"));
    }

    #[test]
    fn test_add_duplicate_codec_once() {
        let mut caps = WorkerCapabilities::new(1);
        caps.add_codec("av1");
        caps.add_codec("av1"); // duplicate
        assert_eq!(caps.codecs().len(), 1);
        assert_eq!(caps.capability_count(), 1);
    }

    #[test]
    fn test_codec_and_gpu_lists() {
        let mut caps = WorkerCapabilities::new(5);
        caps.add_codec("vp9");
        caps.add_codec("h265");
        caps.add_gpu("intel-arc");
        assert_eq!(caps.codecs().len(), 2);
        assert_eq!(caps.gpus().len(), 1);
    }

    #[test]
    fn test_add_arbitrary_capability() {
        let mut caps = WorkerCapabilities::new(3);
        caps.add_capability("hdr10");
        assert!(caps.supports("hdr10"));
        assert!(caps.supports("HDR10"));
    }

    #[test]
    fn test_all_capabilities_union() {
        let mut caps = WorkerCapabilities::new(1);
        caps.add_codec("av1");
        caps.add_gpu("nvidia-t4");
        caps.add_capability("8k");
        assert_eq!(caps.capability_count(), 3);
    }
}
