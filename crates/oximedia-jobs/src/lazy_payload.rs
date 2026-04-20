#![allow(dead_code)]
// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Lazy deserialization of job payloads.
//!
//! When a queue contains thousands of jobs, eagerly deserializing every
//! [`JobPayload`] at load time wastes CPU and memory — especially when
//! only a handful of jobs will actually be dispatched in the near future.
//!
//! This module provides [`LazyPayload`], a wrapper that stores the raw
//! JSON bytes and only deserializes on demand via [`LazyPayload::resolve`].
//! The resolved value is cached so that subsequent accesses are free.
//!
//! # Features
//!
//! * **Deferred parsing** – payload bytes are stored as-is until needed.
//! * **One-shot cache** – once resolved, the deserialized value is memoized.
//! * **Type-safe** – the generic parameter `T` ensures the caller gets the
//!   correct type.
//! * **Introspection** – [`LazyPayload::peek_type`] extracts the top-level
//!   JSON tag without full deserialization, allowing routers to inspect the
//!   job type cheaply.
//! * **Size tracking** – [`LazyPayload::raw_size`] returns the byte length
//!   of the serialized form.

use serde::{de::DeserializeOwned, Serialize};
use std::cell::RefCell;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during lazy deserialization.
#[derive(Debug, Clone)]
pub enum LazyError {
    /// The raw bytes could not be parsed as valid JSON.
    InvalidJson(String),
    /// The JSON structure did not match the expected type `T`.
    TypeMismatch(String),
    /// The raw payload is empty.
    EmptyPayload,
}

impl fmt::Display for LazyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(msg) => write!(f, "invalid JSON: {msg}"),
            Self::TypeMismatch(msg) => write!(f, "type mismatch: {msg}"),
            Self::EmptyPayload => write!(f, "empty payload"),
        }
    }
}

impl std::error::Error for LazyError {}

// ---------------------------------------------------------------------------
// LazyPayload
// ---------------------------------------------------------------------------

/// A lazily-deserialized job payload.
///
/// Stores the raw JSON bytes and only deserializes into `T` on first access.
/// The result is then cached for all subsequent calls.
pub struct LazyPayload<T> {
    /// Raw JSON bytes.
    raw: Vec<u8>,
    /// Cached deserialized value (interior mutability for resolve-on-read).
    cache: RefCell<Option<T>>,
}

impl<T: fmt::Debug> fmt::Debug for LazyPayload<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazyPayload")
            .field("raw_size", &self.raw.len())
            .field("resolved", &self.cache.borrow().is_some())
            .finish()
    }
}

impl<T: Clone + DeserializeOwned> Clone for LazyPayload<T> {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            cache: RefCell::new(self.cache.borrow().clone()),
        }
    }
}

impl<T: DeserializeOwned + Serialize> LazyPayload<T> {
    /// Create a new lazy payload from raw JSON bytes.
    ///
    /// No validation is performed at this point.
    #[must_use]
    pub fn from_bytes(raw: Vec<u8>) -> Self {
        Self {
            raw,
            cache: RefCell::new(None),
        }
    }

    /// Create a lazy payload from a string slice.
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        Self::from_bytes(s.as_bytes().to_vec())
    }

    /// Create a lazy payload by serializing a value upfront.
    ///
    /// This is useful when you *have* the value but want to store it in lazy
    /// form for consistency with payloads loaded from storage.
    ///
    /// # Errors
    ///
    /// Returns [`LazyError::InvalidJson`] if serialization fails.
    pub fn from_value(value: &T) -> Result<Self, LazyError> {
        let raw = serde_json::to_vec(value).map_err(|e| LazyError::InvalidJson(e.to_string()))?;
        let payload = Self {
            raw,
            cache: RefCell::new(None),
        };
        Ok(payload)
    }

    /// Resolve (deserialize) the payload.  The first call parses the JSON;
    /// subsequent calls return the cached value.
    ///
    /// # Errors
    ///
    /// Returns a [`LazyError`] if deserialization fails.
    pub fn resolve(&self) -> Result<std::cell::Ref<'_, T>, LazyError> {
        // Populate the cache if empty.
        {
            let cached = self.cache.borrow();
            if cached.is_some() {
                drop(cached);
                return Ok(std::cell::Ref::map(self.cache.borrow(), |opt| {
                    opt.as_ref().expect("just checked")
                }));
            }
        }

        if self.raw.is_empty() {
            return Err(LazyError::EmptyPayload);
        }

        let value: T =
            serde_json::from_slice(&self.raw).map_err(|e| LazyError::TypeMismatch(e.to_string()))?;

        *self.cache.borrow_mut() = Some(value);

        Ok(std::cell::Ref::map(self.cache.borrow(), |opt| {
            opt.as_ref().expect("just populated")
        }))
    }

    /// Check whether the payload has already been resolved.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        self.cache.borrow().is_some()
    }

    /// Size of the raw JSON payload in bytes.
    #[must_use]
    pub fn raw_size(&self) -> usize {
        self.raw.len()
    }

    /// Return a reference to the raw bytes.
    #[must_use]
    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw
    }

    /// Return the raw payload as a UTF-8 string, if valid.
    pub fn raw_as_str(&self) -> Result<&str, LazyError> {
        std::str::from_utf8(&self.raw).map_err(|e| LazyError::InvalidJson(e.to_string()))
    }

    /// Peek at the top-level JSON tag name without full deserialization.
    ///
    /// For an externally-tagged enum like `{"Transcode": {...}}`, this returns
    /// `Some("Transcode")`.  For non-object or non-enum values, returns `None`.
    pub fn peek_type(&self) -> Option<String> {
        // Quick scan for the first key in a JSON object.
        let s = std::str::from_utf8(&self.raw).ok()?;
        let trimmed = s.trim();
        if !trimmed.starts_with('{') {
            return None;
        }
        // Find the first quoted key.
        let after_brace = &trimmed[1..];
        let quote_start = after_brace.find('"')?;
        let rest = &after_brace[quote_start + 1..];
        let quote_end = rest.find('"')?;
        Some(rest[..quote_end].to_string())
    }

    /// Discard the cached value, forcing the next `resolve()` to re-parse.
    pub fn invalidate_cache(&self) {
        *self.cache.borrow_mut() = None;
    }

    /// Replace the raw bytes and invalidate any cache.
    pub fn replace_raw(&mut self, raw: Vec<u8>) {
        self.raw = raw;
        *self.cache.borrow_mut() = None;
    }
}

// ---------------------------------------------------------------------------
// LazyPayloadBatch
// ---------------------------------------------------------------------------

/// A batch of lazy payloads, useful for bulk-loaded job queues.
#[derive(Debug)]
pub struct LazyPayloadBatch<T: DeserializeOwned + Serialize> {
    payloads: Vec<(String, LazyPayload<T>)>,
}

impl<T: DeserializeOwned + Serialize> LazyPayloadBatch<T> {
    /// Create an empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            payloads: Vec::new(),
        }
    }

    /// Add a payload keyed by job ID.
    pub fn add(&mut self, job_id: impl Into<String>, payload: LazyPayload<T>) {
        self.payloads.push((job_id.into(), payload));
    }

    /// Number of payloads in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.payloads.len()
    }

    /// Whether the batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.payloads.is_empty()
    }

    /// Total raw size across all payloads.
    pub fn total_raw_size(&self) -> usize {
        self.payloads.iter().map(|(_, p)| p.raw_size()).sum()
    }

    /// Count how many payloads have been resolved.
    pub fn resolved_count(&self) -> usize {
        self.payloads.iter().filter(|(_, p)| p.is_resolved()).count()
    }

    /// Get a payload by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&(String, LazyPayload<T>)> {
        self.payloads.get(index)
    }

    /// Iterate over `(job_id, payload)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = &(String, LazyPayload<T>)> {
        self.payloads.iter()
    }

    /// Filter payloads by peeked type name without resolving them.
    pub fn filter_by_type(&self, type_name: &str) -> Vec<&(String, LazyPayload<T>)> {
        self.payloads
            .iter()
            .filter(|(_, p)| p.peek_type().as_deref() == Some(type_name))
            .collect()
    }
}

impl<T: DeserializeOwned + Serialize> Default for LazyPayloadBatch<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::JobPayload;

    fn sample_transcode_json() -> String {
        r#"{"Transcode":{"input":"in.mp4","output":"out.mp4","video_codec":"vp9","audio_codec":"opus","video_bitrate":5000000,"audio_bitrate":128000,"resolution":null,"framerate":null,"preset":"medium","hw_accel":null}}"#.to_string()
    }

    fn sample_thumbnail_json() -> String {
        let thumbs_dir = std::env::temp_dir()
            .join("oximedia-jobs-lazy-thumbs")
            .to_string_lossy()
            .into_owned();
        format!(
            r#"{{"Thumbnail":{{"input":"video.mp4","output_dir":"{thumbs_dir}","count":10,"width":320,"height":180,"quality":85}}}}"#
        )
    }

    #[test]
    fn test_lazy_from_str_and_resolve() {
        let json = sample_transcode_json();
        let lazy: LazyPayload<JobPayload> = LazyPayload::from_str(&json);
        assert!(!lazy.is_resolved());
        let resolved = lazy.resolve();
        assert!(resolved.is_ok());
        assert!(lazy.is_resolved());
    }

    #[test]
    fn test_lazy_resolve_caches() {
        let json = sample_transcode_json();
        let lazy: LazyPayload<JobPayload> = LazyPayload::from_str(&json);
        let _ = lazy.resolve();
        assert!(lazy.is_resolved());
        // Second resolve should also succeed (from cache).
        let r2 = lazy.resolve();
        assert!(r2.is_ok());
    }

    #[test]
    fn test_lazy_raw_size() {
        let json = sample_transcode_json();
        let lazy: LazyPayload<JobPayload> = LazyPayload::from_str(&json);
        assert_eq!(lazy.raw_size(), json.len());
    }

    #[test]
    fn test_lazy_peek_type() {
        let json = sample_transcode_json();
        let lazy: LazyPayload<JobPayload> = LazyPayload::from_str(&json);
        assert_eq!(lazy.peek_type().as_deref(), Some("Transcode"));
        assert!(!lazy.is_resolved()); // peek should NOT resolve
    }

    #[test]
    fn test_lazy_peek_type_thumbnail() {
        let json = sample_thumbnail_json();
        let lazy: LazyPayload<JobPayload> = LazyPayload::from_str(&json);
        assert_eq!(lazy.peek_type().as_deref(), Some("Thumbnail"));
    }

    #[test]
    fn test_lazy_empty_payload() {
        let lazy: LazyPayload<JobPayload> = LazyPayload::from_bytes(Vec::new());
        let result = lazy.resolve();
        assert!(result.is_err());
        match result.err() {
            Some(LazyError::EmptyPayload) => {}
            other => panic!("Expected EmptyPayload, got {other:?}"),
        }
    }

    #[test]
    fn test_lazy_invalid_json() {
        let lazy: LazyPayload<JobPayload> = LazyPayload::from_str("not json at all");
        let result = lazy.resolve();
        assert!(result.is_err());
    }

    #[test]
    fn test_lazy_type_mismatch() {
        // Valid JSON but wrong structure for JobPayload.
        let lazy: LazyPayload<JobPayload> = LazyPayload::from_str(r#"{"unknown_key": 42}"#);
        let result = lazy.resolve();
        assert!(result.is_err());
    }

    #[test]
    fn test_lazy_invalidate_cache() {
        let json = sample_transcode_json();
        let lazy: LazyPayload<JobPayload> = LazyPayload::from_str(&json);
        let _ = lazy.resolve();
        assert!(lazy.is_resolved());
        lazy.invalidate_cache();
        assert!(!lazy.is_resolved());
        // Re-resolve should work.
        assert!(lazy.resolve().is_ok());
    }

    #[test]
    fn test_lazy_replace_raw() {
        let json = sample_transcode_json();
        let mut lazy: LazyPayload<JobPayload> = LazyPayload::from_str(&json);
        let _ = lazy.resolve();
        let new_json = sample_thumbnail_json();
        lazy.replace_raw(new_json.as_bytes().to_vec());
        assert!(!lazy.is_resolved());
        assert_eq!(lazy.peek_type().as_deref(), Some("Thumbnail"));
    }

    #[test]
    fn test_lazy_raw_as_str() {
        let json = sample_transcode_json();
        let lazy: LazyPayload<JobPayload> = LazyPayload::from_str(&json);
        let s = lazy.raw_as_str();
        assert!(s.is_ok());
        assert_eq!(s.unwrap_or(""), &json);
    }

    #[test]
    fn test_lazy_from_value() {
        use crate::job::TranscodeParams;
        let params = TranscodeParams {
            input: "in.mp4".to_string(),
            output: "out.mp4".to_string(),
            video_codec: "vp9".to_string(),
            audio_codec: "opus".to_string(),
            video_bitrate: 5_000_000,
            audio_bitrate: 128_000,
            resolution: None,
            framerate: None,
            preset: "medium".to_string(),
            hw_accel: None,
        };
        let payload = JobPayload::Transcode(params);
        let lazy = LazyPayload::from_value(&payload);
        assert!(lazy.is_ok());
        let lazy = lazy.expect("from_value should succeed");
        assert!(lazy.raw_size() > 0);
        assert!(lazy.resolve().is_ok());
    }

    #[test]
    fn test_batch_basic() {
        let mut batch: LazyPayloadBatch<JobPayload> = LazyPayloadBatch::new();
        assert!(batch.is_empty());
        batch.add("j1", LazyPayload::from_str(&sample_transcode_json()));
        batch.add("j2", LazyPayload::from_str(&sample_thumbnail_json()));
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_total_raw_size() {
        let mut batch: LazyPayloadBatch<JobPayload> = LazyPayloadBatch::new();
        let t = sample_transcode_json();
        let th = sample_thumbnail_json();
        let expected = t.len() + th.len();
        batch.add("j1", LazyPayload::from_str(&t));
        batch.add("j2", LazyPayload::from_str(&th));
        assert_eq!(batch.total_raw_size(), expected);
    }

    #[test]
    fn test_batch_resolved_count() {
        let mut batch: LazyPayloadBatch<JobPayload> = LazyPayloadBatch::new();
        batch.add("j1", LazyPayload::from_str(&sample_transcode_json()));
        batch.add("j2", LazyPayload::from_str(&sample_thumbnail_json()));
        assert_eq!(batch.resolved_count(), 0);
        // Resolve one.
        if let Some((_, p)) = batch.get(0) {
            let _ = p.resolve();
        }
        assert_eq!(batch.resolved_count(), 1);
    }

    #[test]
    fn test_batch_filter_by_type() {
        let mut batch: LazyPayloadBatch<JobPayload> = LazyPayloadBatch::new();
        batch.add("j1", LazyPayload::from_str(&sample_transcode_json()));
        batch.add("j2", LazyPayload::from_str(&sample_thumbnail_json()));
        batch.add("j3", LazyPayload::from_str(&sample_transcode_json()));
        let transcodes = batch.filter_by_type("Transcode");
        assert_eq!(transcodes.len(), 2);
        let thumbs = batch.filter_by_type("Thumbnail");
        assert_eq!(thumbs.len(), 1);
    }

    #[test]
    fn test_lazy_debug_format() {
        let lazy: LazyPayload<JobPayload> = LazyPayload::from_str(&sample_transcode_json());
        let dbg = format!("{lazy:?}");
        assert!(dbg.contains("LazyPayload"));
        assert!(dbg.contains("resolved"));
    }

    #[test]
    fn test_lazy_error_display() {
        let e1 = LazyError::InvalidJson("bad".to_string());
        assert!(format!("{e1}").contains("invalid JSON"));
        let e2 = LazyError::TypeMismatch("wrong type".to_string());
        assert!(format!("{e2}").contains("type mismatch"));
        let e3 = LazyError::EmptyPayload;
        assert!(format!("{e3}").contains("empty"));
    }
}
