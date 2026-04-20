//! Structured error context and error chaining utilities.
//!
//! Provides [`ErrorContext`], [`ErrorChain`], [`ErrorContextBuilder`], and
//! [`ErrorFrame`] for attaching structured metadata to errors propagated through
//! the media pipeline.
//!
//! # Examples
//!
//! ```
//! use oximedia_core::error_context::{ErrorContext, ErrorChain, ErrorFrame};
//!
//! let ctx = ErrorContext::new("demuxer", "read_packet", "unexpected EOF");
//! assert_eq!(ctx.component(), "demuxer");
//!
//! let chain = ErrorChain::root(ctx);
//! assert_eq!(chain.depth(), 1);
//!
//! // Structured frame chain on ErrorContext
//! let mut ctx2 = ErrorContext::new("codec", "decode", "buffer underflow");
//! ctx2.push_frame(ErrorFrame {
//!     file: "src/codec.rs",
//!     line: 42,
//!     function: "decode_frame",
//!     message: std::borrow::Cow::Borrowed("buffer underflow"),
//! });
//! let display = ctx2.frames_display();
//! assert!(!display.is_empty());
//! ```
//!
//! ## Building an Error Context Chain
//!
//! Use the `ctx!` macro to attach a location frame, then chain frames as errors
//! propagate up the call stack:
//!
//! ```
//! use oximedia_core::error_context::{ErrorContext, ErrorFrame};
//! use std::borrow::Cow;
//!
//! let mut ctx = ErrorContext::new("demuxer", "read_packet", "initial error");
//! ctx.push_frame(ErrorFrame {
//!     file: file!(),
//!     line: line!(),
//!     function: "my_function",
//!     message: Cow::Borrowed("initial error"),
//! });
//! ctx.push_frame(ErrorFrame {
//!     file: file!(),
//!     line: line!(),
//!     function: "caller",
//!     message: Cow::Borrowed("wrapping context"),
//! });
//! assert_eq!(ctx.frame_count(), 2);
//! // frames_display() returns a multi-line string showing file:line [fn]: message
//! let display = ctx.frames_display();
//! assert!(display.contains("my_function"));
//! assert!(display.contains("caller"));
//! ```

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// ErrorFrame — a single source-location frame in a context chain
// ---------------------------------------------------------------------------

/// A single frame in an error context chain, capturing the source location
/// and a human-readable message at the point where context was attached.
///
/// Instances are typically created via the `ctx!` macro rather than
/// constructed by hand.
///
/// # Examples
///
/// ```
/// use oximedia_core::error_context::ErrorFrame;
///
/// let frame = ErrorFrame {
///     file: "src/lib.rs",
///     line: 10,
///     function: "my_func",
///     message: std::borrow::Cow::Borrowed("something went wrong"),
/// };
/// let s = frame.to_string();
/// assert!(s.contains("src/lib.rs"));
/// assert!(s.contains("10"));
/// assert!(s.contains("my_func"));
/// assert!(s.contains("something went wrong"));
/// ```
#[derive(Debug, Clone)]
pub struct ErrorFrame {
    /// Source file path captured by `file!()`.
    pub file: &'static str,
    /// Line number captured by `line!()`.
    pub line: u32,
    /// Function or module path captured by the `current_fn_name!()` macro.
    pub function: &'static str,
    /// Human-readable context message.
    pub message: Cow<'static, str>,
}

impl fmt::Display for ErrorFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "  at {}:{} [{}]: {}",
            self.file, self.line, self.function, self.message
        )
    }
}

// ---------------------------------------------------------------------------
// current_fn_name! and ctx! macros
// ---------------------------------------------------------------------------

/// Captures the fully-qualified path of the enclosing function at compile time.
///
/// The result is a `&'static str` with the `::f` helper suffix stripped,
/// yielding the module path of the call site (e.g. `"my_crate::module::fn"`).
///
/// # Examples
///
/// ```
/// fn example() {
///     let name = oximedia_core::current_fn_name!();
///     // name ends with "example"
///     assert!(name.contains("example"));
/// }
/// example();
/// ```
#[macro_export]
macro_rules! current_fn_name {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let full = type_name_of(f);
        // Strip the trailing "::f" (3 chars) added by the inner fn
        &full[..full.len() - 3]
    }};
}

/// Creates an [`ErrorFrame`] capturing the current source file, line, and
/// function name via built-in macros.
///
/// Two forms:
/// - `ctx!("literal message")` — zero-allocation borrowed string.
/// - `ctx!("fmt {}", arg)` — allocated owned string via `format!`.
///
/// # Examples
///
/// ```
/// use oximedia_core::{ctx, error_context::ErrorFrame};
///
/// let frame: ErrorFrame = ctx!("something failed");
/// assert!(frame.to_string().contains("something failed"));
///
/// let n = 42u32;
/// let frame2: ErrorFrame = ctx!("value was {}", n);
/// assert!(frame2.to_string().contains("42"));
/// ```
#[macro_export]
macro_rules! ctx {
    ($msg:literal) => {
        $crate::error_context::ErrorFrame {
            file: file!(),
            line: line!(),
            function: $crate::current_fn_name!(),
            message: std::borrow::Cow::Borrowed($msg),
        }
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::error_context::ErrorFrame {
            file: file!(),
            line: line!(),
            function: $crate::current_fn_name!(),
            message: std::borrow::Cow::Owned(format!($fmt, $($arg)*)),
        }
    };
}

// ---------------------------------------------------------------------------
// OxiErrorExt
// ---------------------------------------------------------------------------

/// Extension trait that attaches an [`ErrorFrame`] as additional context to
/// any error type that implements [`std::error::Error`].
///
/// This avoids a dependency on `anyhow` by embedding the frame's `Display`
/// representation into a new [`crate::error::OxiError::InvalidData`] wrapper.
pub trait OxiErrorExt: Sized {
    /// Wraps `self` by prepending the frame's display string as context.
    fn with_oxi_context(self, frame: ErrorFrame) -> crate::error::OxiError;
}

impl<E: std::error::Error> OxiErrorExt for E {
    fn with_oxi_context(self, frame: ErrorFrame) -> crate::error::OxiError {
        crate::error::OxiError::InvalidData(format!("{frame}\ncaused by: {self}"))
    }
}

// ---------------------------------------------------------------------------

/// Structured context attached to a single error occurrence.
///
/// Records where an error happened (`component`, `operation`) and a
/// human-readable `message`.  Optional key/value pairs may carry additional
/// diagnostic information.  An ordered list of [`ErrorFrame`] records captures
/// the source locations where context was attached via the `ctx!` macro.
///
/// # Examples
///
/// ```
/// use oximedia_core::error_context::{ErrorContext, ErrorFrame};
///
/// let mut ctx = ErrorContext::new("muxer", "write_header", "disk full");
/// assert_eq!(ctx.component(), "muxer");
/// assert_eq!(ctx.operation(), "write_header");
/// assert_eq!(ctx.message(), "disk full");
///
/// ctx.push_frame(ErrorFrame {
///     file: "src/muxer.rs",
///     line: 99,
///     function: "write_header",
///     message: std::borrow::Cow::Borrowed("disk full"),
/// });
/// assert_eq!(ctx.frame_count(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct ErrorContext {
    component: String,
    operation: String,
    message: String,
    fields: HashMap<String, String>,
    /// Ordered list of source-location frames attached via [`ctx!`].
    frames: Vec<ErrorFrame>,
}

/// `ErrorContext` values are compared by component, operation, message, and fields only.
/// The `frames` chain is intentionally excluded from equality so that adding trace
/// frames does not alter the identity of an error context.
impl PartialEq for ErrorContext {
    fn eq(&self, other: &Self) -> bool {
        self.component == other.component
            && self.operation == other.operation
            && self.message == other.message
            && self.fields == other.fields
    }
}

impl Eq for ErrorContext {}

impl ErrorContext {
    /// Creates a new context with the given component, operation, and message.
    #[must_use]
    pub fn new(component: &str, operation: &str, message: &str) -> Self {
        Self {
            component: component.to_owned(),
            operation: operation.to_owned(),
            message: message.to_owned(),
            fields: HashMap::new(),
            frames: Vec::new(),
        }
    }

    /// Appends an [`ErrorFrame`] to the context chain.
    ///
    /// Frames accumulate in the order they are pushed (earliest first).
    pub fn push_frame(&mut self, frame: ErrorFrame) {
        self.frames.push(frame);
    }

    /// Consumes `self`, appends `frame`, and returns the modified context.
    ///
    /// Useful for method chaining.
    #[must_use]
    pub fn with_frame(mut self, frame: ErrorFrame) -> Self {
        self.frames.push(frame);
        self
    }

    /// Returns the number of frames in the context chain.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns an iterator over the attached frames (earliest first).
    pub fn frames(&self) -> impl Iterator<Item = &ErrorFrame> {
        self.frames.iter()
    }

    /// Returns a multi-line string rendering of all attached frames,
    /// or `"(no context frames)"` if the chain is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::error_context::{ErrorContext, ErrorFrame};
    ///
    /// let mut ctx = ErrorContext::new("a", "b", "c");
    /// ctx.push_frame(ErrorFrame {
    ///     file: "x.rs", line: 1, function: "f",
    ///     message: std::borrow::Cow::Borrowed("first"),
    /// });
    /// ctx.push_frame(ErrorFrame {
    ///     file: "y.rs", line: 2, function: "g",
    ///     message: std::borrow::Cow::Borrowed("second"),
    /// });
    /// let s = ctx.frames_display();
    /// assert!(s.contains("first"));
    /// assert!(s.contains("second"));
    /// ```
    #[must_use]
    pub fn frames_display(&self) -> String {
        if self.frames.is_empty() {
            return "(no context frames)".to_owned();
        }
        self.frames
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if i == 0 {
                    f.to_string()
                } else {
                    format!("\n{f}")
                }
            })
            .collect()
    }

    /// Returns the name of the component that raised the error.
    #[inline]
    #[must_use]
    pub fn component(&self) -> &str {
        &self.component
    }

    /// Returns the name of the operation that was in progress.
    #[inline]
    #[must_use]
    pub fn operation(&self) -> &str {
        &self.operation
    }

    /// Returns the human-readable error message.
    #[inline]
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Attaches an additional key/value diagnostic field.
    ///
    /// Calling this multiple times with the same key overwrites the previous
    /// value.
    pub fn with_field(&mut self, key: &str, value: &str) -> &mut Self {
        self.fields.insert(key.to_owned(), value.to_owned());
        self
    }

    /// Returns the value of a diagnostic field, or `None` if absent.
    #[must_use]
    pub fn field(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(String::as_str)
    }

    /// Returns an iterator over all attached diagnostic fields.
    pub fn fields(&self) -> impl Iterator<Item = (&str, &str)> {
        self.fields.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

impl std::fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}::{}] {}",
            self.component, self.operation, self.message
        )
    }
}

// ---------------------------------------------------------------------------

/// A chain of [`ErrorContext`] records representing an error's call stack.
///
/// Errors are pushed from innermost (root cause) to outermost (top-level
/// context).  [`depth`](Self::depth) returns the number of frames.
///
/// # Examples
///
/// ```
/// use oximedia_core::error_context::{ErrorContext, ErrorChain};
///
/// let root = ErrorContext::new("io", "read", "timeout");
/// let mut chain = ErrorChain::root(root);
/// chain.push(ErrorContext::new("demuxer", "read_packet", "I/O error"));
/// assert_eq!(chain.depth(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct ErrorChain {
    frames: Vec<ErrorContext>,
}

impl ErrorChain {
    /// Creates a chain containing a single root (innermost) context.
    #[must_use]
    pub fn root(ctx: ErrorContext) -> Self {
        Self { frames: vec![ctx] }
    }

    /// Creates an empty chain.
    #[must_use]
    pub fn empty() -> Self {
        Self { frames: Vec::new() }
    }

    /// Pushes an outer context onto the chain.
    pub fn push(&mut self, ctx: ErrorContext) {
        self.frames.push(ctx);
    }

    /// Returns the number of context frames in the chain.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` if the chain contains no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns the root (innermost / first-cause) context, or `None` if empty.
    #[must_use]
    pub fn root_cause(&self) -> Option<&ErrorContext> {
        self.frames.first()
    }

    /// Returns the outermost context (most recently pushed), or `None` if empty.
    #[must_use]
    pub fn outermost(&self) -> Option<&ErrorContext> {
        self.frames.last()
    }

    /// Returns an iterator over all frames from root to outermost.
    pub fn iter(&self) -> impl Iterator<Item = &ErrorContext> {
        self.frames.iter()
    }

    /// Returns `true` if any frame's component matches `component`.
    #[must_use]
    pub fn involves(&self, component: &str) -> bool {
        self.frames.iter().any(|f| f.component() == component)
    }
}

impl std::fmt::Display for ErrorChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, frame) in self.frames.iter().enumerate() {
            if i > 0 {
                write!(f, " -> ")?;
            }
            write!(f, "{frame}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------

/// A builder that constructs an [`ErrorContext`] with a fluent API.
///
/// # Examples
///
/// ```
/// use oximedia_core::error_context::ErrorContextBuilder;
///
/// let ctx = ErrorContextBuilder::new("codec", "decode_frame")
///     .message("bitstream error")
///     .field("pts", "12345")
///     .build();
///
/// assert_eq!(ctx.component(), "codec");
/// assert_eq!(ctx.field("pts"), Some("12345"));
/// ```
#[derive(Debug, Default)]
pub struct ErrorContextBuilder {
    component: String,
    operation: String,
    message: String,
    fields: HashMap<String, String>,
}

impl ErrorContextBuilder {
    /// Starts a new builder with the given `component` and `operation`.
    #[must_use]
    pub fn new(component: &str, operation: &str) -> Self {
        Self {
            component: component.to_owned(),
            operation: operation.to_owned(),
            message: String::new(),
            fields: HashMap::new(),
        }
    }

    /// Sets the error message.
    #[must_use]
    pub fn message(mut self, msg: &str) -> Self {
        msg.clone_into(&mut self.message);
        self
    }

    /// Attaches a key/value diagnostic field.
    #[must_use]
    pub fn field(mut self, key: &str, value: &str) -> Self {
        self.fields.insert(key.to_owned(), value.to_owned());
        self
    }

    /// Consumes the builder and returns the constructed [`ErrorContext`].
    #[must_use]
    pub fn build(self) -> ErrorContext {
        ErrorContext {
            component: self.component,
            operation: self.operation,
            message: self.message,
            fields: self.fields,
            frames: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_context_accessors() {
        let ctx = ErrorContext::new("demuxer", "read_packet", "EOF");
        assert_eq!(ctx.component(), "demuxer");
        assert_eq!(ctx.operation(), "read_packet");
        assert_eq!(ctx.message(), "EOF");
    }

    #[test]
    fn error_context_with_field() {
        let mut ctx = ErrorContext::new("codec", "decode", "error");
        ctx.with_field("pts", "1000");
        assert_eq!(ctx.field("pts"), Some("1000"));
    }

    #[test]
    fn error_context_missing_field_is_none() {
        let ctx = ErrorContext::new("x", "y", "z");
        assert!(ctx.field("nonexistent").is_none());
    }

    #[test]
    fn error_context_field_overwrite() {
        let mut ctx = ErrorContext::new("a", "b", "c");
        ctx.with_field("k", "v1");
        ctx.with_field("k", "v2");
        assert_eq!(ctx.field("k"), Some("v2"));
    }

    #[test]
    fn error_context_display() {
        let ctx = ErrorContext::new("muxer", "write", "disk full");
        let s = ctx.to_string();
        assert!(s.contains("muxer"));
        assert!(s.contains("write"));
        assert!(s.contains("disk full"));
    }

    #[test]
    fn error_chain_root_depth_one() {
        let ctx = ErrorContext::new("io", "read", "timeout");
        let chain = ErrorChain::root(ctx);
        assert_eq!(chain.depth(), 1);
    }

    #[test]
    fn error_chain_push_increases_depth() {
        let mut chain = ErrorChain::root(ErrorContext::new("a", "op", "msg"));
        chain.push(ErrorContext::new("b", "op2", "msg2"));
        assert_eq!(chain.depth(), 2);
    }

    #[test]
    fn error_chain_root_cause() {
        let ctx = ErrorContext::new("inner", "op", "root cause");
        let chain = ErrorChain::root(ctx.clone());
        assert_eq!(chain.root_cause(), Some(&ctx));
    }

    #[test]
    fn error_chain_outermost() {
        let mut chain = ErrorChain::root(ErrorContext::new("inner", "op", "cause"));
        let outer = ErrorContext::new("outer", "handle", "context");
        chain.push(outer.clone());
        assert_eq!(chain.outermost(), Some(&outer));
    }

    #[test]
    fn error_chain_involves() {
        let mut chain = ErrorChain::root(ErrorContext::new("io", "read", "err"));
        chain.push(ErrorContext::new("demuxer", "parse", "err2"));
        assert!(chain.involves("io"));
        assert!(chain.involves("demuxer"));
        assert!(!chain.involves("encoder"));
    }

    #[test]
    fn error_chain_empty() {
        let chain = ErrorChain::empty();
        assert!(chain.is_empty());
        assert_eq!(chain.depth(), 0);
        assert!(chain.root_cause().is_none());
        assert!(chain.outermost().is_none());
    }

    #[test]
    fn error_chain_display_multi_frame() {
        let mut chain = ErrorChain::root(ErrorContext::new("a", "op", "first"));
        chain.push(ErrorContext::new("b", "op2", "second"));
        let s = chain.to_string();
        assert!(s.contains("first"));
        assert!(s.contains("second"));
        assert!(s.contains("->"));
    }

    #[test]
    fn builder_creates_correct_context() {
        let ctx = ErrorContextBuilder::new("codec", "decode_frame")
            .message("bitstream error")
            .field("pts", "12345")
            .build();
        assert_eq!(ctx.component(), "codec");
        assert_eq!(ctx.operation(), "decode_frame");
        assert_eq!(ctx.message(), "bitstream error");
        assert_eq!(ctx.field("pts"), Some("12345"));
    }

    #[test]
    fn builder_default_message_is_empty() {
        let ctx = ErrorContextBuilder::new("c", "op").build();
        assert_eq!(ctx.message(), "");
    }

    #[test]
    fn error_chain_iter_count_matches_depth() {
        let mut chain = ErrorChain::root(ErrorContext::new("a", "op", "e1"));
        chain.push(ErrorContext::new("b", "op2", "e2"));
        chain.push(ErrorContext::new("c", "op3", "e3"));
        assert_eq!(chain.iter().count(), chain.depth());
    }
}
