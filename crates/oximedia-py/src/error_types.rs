#![allow(dead_code)]
//! Typed error codes and structured error information for Python bindings.
//!
//! Provides a fine-grained error taxonomy that maps OxiMedia internal errors
//! to Python-friendly codes and messages, including severity levels and
//! optional context fields.

use std::collections::HashMap;
use std::fmt;

/// Severity level associated with an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorSeverity {
    /// A warning that does not prevent further processing.
    Warning,
    /// An error that prevents the current operation but not the whole session.
    Error,
    /// A fatal error requiring immediate session teardown.
    Fatal,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Fatal => write!(f, "FATAL"),
        }
    }
}

/// Broad error category used to classify binding errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Codec-related errors (decode/encode failures).
    Codec,
    /// Container format errors (mux/demux).
    Container,
    /// I/O errors (file read/write).
    Io,
    /// Invalid argument passed from Python.
    InvalidArgument,
    /// Resource exhaustion (memory, file descriptors).
    Resource,
    /// Internal logic error.
    Internal,
    /// Operation timed out.
    Timeout,
    /// Feature not implemented or disabled.
    NotImplemented,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec => write!(f, "CODEC"),
            Self::Container => write!(f, "CONTAINER"),
            Self::Io => write!(f, "IO"),
            Self::InvalidArgument => write!(f, "INVALID_ARGUMENT"),
            Self::Resource => write!(f, "RESOURCE"),
            Self::Internal => write!(f, "INTERNAL"),
            Self::Timeout => write!(f, "TIMEOUT"),
            Self::NotImplemented => write!(f, "NOT_IMPLEMENTED"),
        }
    }
}

/// A structured error with code, category, severity, and optional context.
#[derive(Debug, Clone)]
pub struct BindingError {
    /// Numeric error code (application-defined).
    pub code: u32,
    /// Error category.
    pub category: ErrorCategory,
    /// Severity level.
    pub severity: ErrorSeverity,
    /// Human-readable message.
    pub message: String,
    /// Optional key-value context pairs.
    pub context: HashMap<String, String>,
}

impl BindingError {
    /// Create a new binding error with the given code, category, severity, and message.
    pub fn new(
        code: u32,
        category: ErrorCategory,
        severity: ErrorSeverity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            category,
            severity,
            message: message.into(),
            context: HashMap::new(),
        }
    }

    /// Add a context key-value pair and return self for chaining.
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Returns `true` if this error is fatal.
    #[must_use]
    pub fn is_fatal(&self) -> bool {
        self.severity == ErrorSeverity::Fatal
    }

    /// Returns `true` if this error is merely a warning.
    #[must_use]
    pub fn is_warning(&self) -> bool {
        self.severity == ErrorSeverity::Warning
    }

    /// Format a one-line summary: `"[SEVERITY][CATEGORY] code=<code>: <message>"`.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "[{}][{}] code={}: {}",
            self.severity, self.category, self.code, self.message
        )
    }
}

impl fmt::Display for BindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Registry of known error codes with human descriptions.
#[derive(Debug, Clone)]
pub struct ErrorRegistry {
    /// Maps error code to (category, description).
    entries: HashMap<u32, (ErrorCategory, String)>,
}

impl ErrorRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register an error code.
    pub fn register(&mut self, code: u32, category: ErrorCategory, description: impl Into<String>) {
        self.entries.insert(code, (category, description.into()));
    }

    /// Look up an error code.
    pub fn lookup(&self, code: u32) -> Option<&(ErrorCategory, String)> {
        self.entries.get(&code)
    }

    /// Number of registered codes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Create a `BindingError` from a registered code with a runtime message.
    pub fn make_error(
        &self,
        code: u32,
        severity: ErrorSeverity,
        message: impl Into<String>,
    ) -> BindingError {
        let category = self
            .entries
            .get(&code)
            .map(|(c, _)| *c)
            .unwrap_or(ErrorCategory::Internal);
        BindingError::new(code, category, severity, message)
    }

    /// Create a default registry pre-populated with common codes.
    pub fn default_registry() -> Self {
        let mut reg = Self::new();
        reg.register(1000, ErrorCategory::Codec, "Decoder initialization failed");
        reg.register(1001, ErrorCategory::Codec, "Encoder initialization failed");
        reg.register(1002, ErrorCategory::Codec, "Decode error");
        reg.register(1003, ErrorCategory::Codec, "Encode error");
        reg.register(2000, ErrorCategory::Container, "Demuxer open failed");
        reg.register(2001, ErrorCategory::Container, "Muxer write failed");
        reg.register(3000, ErrorCategory::Io, "File not found");
        reg.register(3001, ErrorCategory::Io, "Permission denied");
        reg.register(4000, ErrorCategory::InvalidArgument, "Invalid parameter");
        reg.register(5000, ErrorCategory::Resource, "Out of memory");
        reg.register(6000, ErrorCategory::Timeout, "Operation timed out");
        reg.register(7000, ErrorCategory::NotImplemented, "Feature not available");
        reg
    }
}

impl Default for ErrorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Accumulates multiple errors for batch reporting.
#[derive(Debug, Clone, Default)]
pub struct ErrorCollector {
    /// Collected errors.
    errors: Vec<BindingError>,
}

impl ErrorCollector {
    /// Create an empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push an error.
    pub fn push(&mut self, err: BindingError) {
        self.errors.push(err);
    }

    /// Number of collected errors.
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Whether the collector is empty.
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Whether any fatal error was collected.
    pub fn has_fatal(&self) -> bool {
        self.errors.iter().any(|e| e.is_fatal())
    }

    /// Count errors of a given category.
    pub fn count_by_category(&self, cat: ErrorCategory) -> usize {
        self.errors.iter().filter(|e| e.category == cat).count()
    }

    /// Return all collected errors.
    pub fn errors(&self) -> &[BindingError] {
        &self.errors
    }

    /// Drain all errors out of the collector.
    pub fn drain(&mut self) -> Vec<BindingError> {
        std::mem::take(&mut self.errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ErrorSeverity ──────────────────────────────────────────────────────

    #[test]
    fn test_severity_display() {
        assert_eq!(ErrorSeverity::Warning.to_string(), "WARNING");
        assert_eq!(ErrorSeverity::Error.to_string(), "ERROR");
        assert_eq!(ErrorSeverity::Fatal.to_string(), "FATAL");
    }

    #[test]
    fn test_severity_equality() {
        assert_eq!(ErrorSeverity::Warning, ErrorSeverity::Warning);
        assert_ne!(ErrorSeverity::Warning, ErrorSeverity::Fatal);
    }

    // ── ErrorCategory ──────────────────────────────────────────────────────

    #[test]
    fn test_category_display() {
        assert_eq!(ErrorCategory::Codec.to_string(), "CODEC");
        assert_eq!(ErrorCategory::Io.to_string(), "IO");
        assert_eq!(ErrorCategory::Timeout.to_string(), "TIMEOUT");
    }

    // ── BindingError ───────────────────────────────────────────────────────

    #[test]
    fn test_binding_error_new() {
        let err = BindingError::new(1000, ErrorCategory::Codec, ErrorSeverity::Error, "fail");
        assert_eq!(err.code, 1000);
        assert_eq!(err.category, ErrorCategory::Codec);
        assert_eq!(err.severity, ErrorSeverity::Error);
        assert_eq!(err.message, "fail");
    }

    #[test]
    fn test_binding_error_with_context() {
        let err = BindingError::new(1000, ErrorCategory::Codec, ErrorSeverity::Error, "fail")
            .with_context("file", "test.mp4")
            .with_context("stream", "0");
        assert_eq!(err.context.len(), 2);
        assert_eq!(
            err.context.get("file").expect("get should succeed"),
            "test.mp4"
        );
    }

    #[test]
    fn test_binding_error_is_fatal() {
        let fatal = BindingError::new(5000, ErrorCategory::Resource, ErrorSeverity::Fatal, "oom");
        assert!(fatal.is_fatal());
        assert!(!fatal.is_warning());
    }

    #[test]
    fn test_binding_error_is_warning() {
        let warn = BindingError::new(100, ErrorCategory::Codec, ErrorSeverity::Warning, "drift");
        assert!(warn.is_warning());
        assert!(!warn.is_fatal());
    }

    #[test]
    fn test_binding_error_summary() {
        let err = BindingError::new(3000, ErrorCategory::Io, ErrorSeverity::Error, "not found");
        let s = err.summary();
        assert!(s.contains("[ERROR]"));
        assert!(s.contains("[IO]"));
        assert!(s.contains("code=3000"));
        assert!(s.contains("not found"));
    }

    #[test]
    fn test_binding_error_display() {
        let err = BindingError::new(1000, ErrorCategory::Codec, ErrorSeverity::Error, "bad");
        let d = format!("{err}");
        assert!(d.contains("CODEC"));
    }

    // ── ErrorRegistry ──────────────────────────────────────────────────────

    #[test]
    fn test_registry_empty() {
        let reg = ErrorRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_register_and_lookup() {
        let mut reg = ErrorRegistry::new();
        reg.register(42, ErrorCategory::Io, "test error");
        assert_eq!(reg.len(), 1);
        let (cat, desc) = reg.lookup(42).expect("lookup should succeed");
        assert_eq!(*cat, ErrorCategory::Io);
        assert_eq!(desc, "test error");
    }

    #[test]
    fn test_registry_lookup_missing() {
        let reg = ErrorRegistry::new();
        assert!(reg.lookup(999).is_none());
    }

    #[test]
    fn test_default_registry_populated() {
        let reg = ErrorRegistry::default_registry();
        assert!(reg.len() >= 10);
        assert!(reg.lookup(1000).is_some());
        assert!(reg.lookup(3000).is_some());
    }

    #[test]
    fn test_registry_make_error() {
        let reg = ErrorRegistry::default_registry();
        let err = reg.make_error(1000, ErrorSeverity::Error, "decoder crashed");
        assert_eq!(err.code, 1000);
        assert_eq!(err.category, ErrorCategory::Codec);
    }

    // ── ErrorCollector ─────────────────────────────────────────────────────

    #[test]
    fn test_collector_empty() {
        let c = ErrorCollector::new();
        assert!(c.is_empty());
        assert!(!c.has_fatal());
    }

    #[test]
    fn test_collector_push_and_len() {
        let mut c = ErrorCollector::new();
        c.push(BindingError::new(
            1,
            ErrorCategory::Codec,
            ErrorSeverity::Error,
            "a",
        ));
        c.push(BindingError::new(
            2,
            ErrorCategory::Io,
            ErrorSeverity::Warning,
            "b",
        ));
        assert_eq!(c.len(), 2);
        assert!(!c.is_empty());
    }

    #[test]
    fn test_collector_has_fatal() {
        let mut c = ErrorCollector::new();
        c.push(BindingError::new(
            1,
            ErrorCategory::Resource,
            ErrorSeverity::Fatal,
            "oom",
        ));
        assert!(c.has_fatal());
    }

    #[test]
    fn test_collector_count_by_category() {
        let mut c = ErrorCollector::new();
        c.push(BindingError::new(
            1,
            ErrorCategory::Io,
            ErrorSeverity::Error,
            "a",
        ));
        c.push(BindingError::new(
            2,
            ErrorCategory::Io,
            ErrorSeverity::Error,
            "b",
        ));
        c.push(BindingError::new(
            3,
            ErrorCategory::Codec,
            ErrorSeverity::Error,
            "c",
        ));
        assert_eq!(c.count_by_category(ErrorCategory::Io), 2);
        assert_eq!(c.count_by_category(ErrorCategory::Codec), 1);
        assert_eq!(c.count_by_category(ErrorCategory::Timeout), 0);
    }

    #[test]
    fn test_collector_drain() {
        let mut c = ErrorCollector::new();
        c.push(BindingError::new(
            1,
            ErrorCategory::Io,
            ErrorSeverity::Error,
            "x",
        ));
        let drained = c.drain();
        assert_eq!(drained.len(), 1);
        assert!(c.is_empty());
    }
}
