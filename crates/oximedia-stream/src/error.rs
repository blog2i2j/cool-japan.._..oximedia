//! Error types for `oximedia-stream`.

use thiserror::Error;

/// All errors produced by the stream crate.
#[derive(Debug, Error)]
pub enum StreamError {
    /// ABR algorithm state error.
    #[error("ABR error: {0}")]
    Abr(String),

    /// Segment lifecycle error.
    #[error("Segment lifecycle error: {0}")]
    Segment(String),

    /// SCTE-35 parse/serialize error.
    #[error("SCTE-35 error: {0}")]
    Scte35(String),

    /// Buffer is full and cannot accept new items.
    #[error("Buffer full: capacity={capacity}")]
    BufferFull {
        /// Maximum capacity of the buffer.
        capacity: usize,
    },

    /// Attempted operation on empty buffer.
    #[error("Buffer empty")]
    BufferEmpty,

    /// Invalid parameter supplied to function.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Numeric overflow during computation.
    #[error("Numeric overflow in {context}")]
    Overflow {
        /// Context where overflow occurred.
        context: &'static str,
    },
}

/// Convenience alias.
pub type StreamResult<T> = Result<T, StreamError>;
