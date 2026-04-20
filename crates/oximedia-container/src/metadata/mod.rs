//! Unified metadata reading and writing for container formats.
//!
//! This module provides a unified API for reading and writing metadata tags
//! across different container formats:
//!
//! - **Matroska/WebM**: Native Matroska tags
//! - **Ogg (Opus/Vorbis)**: Vorbis comments
//! - **FLAC**: Vorbis comments in FLAC metadata blocks
//!
//! # Example: Reading Metadata
//!
//! ```ignore
//! use oximedia_container::metadata::MetadataReader;
//! use oximedia_io::FileSource;
//!
//! let source = FileSource::open("audio.flac").await?;
//! let metadata = MetadataReader::read(source).await?;
//!
//! if let Some(title) = metadata.get("TITLE") {
//!     println!("Title: {}", title);
//! }
//! ```
//!
//! # Example: Writing Metadata
//!
//! ```ignore
//! use oximedia_container::metadata::{MetadataEditor, TagValue};
//!
//! let mut editor = MetadataEditor::open("audio.flac").await?;
//! editor.set("TITLE", "New Title");
//! editor.set("ARTIST", "New Artist");
//! editor.save().await?;
//! ```

pub mod batch;
pub mod editor;
pub mod reader;
pub mod tags;
mod util;
pub mod vorbis;
pub mod writer;

#[cfg(not(target_arch = "wasm32"))]
pub use editor::MetadataEditor;
pub use editor::{BatchMetadataEditor, BatchTagOperation, MetadataFormat, MetadataOp, TagDiff};
pub use reader::MetadataReader;
pub use tags::{StandardTag, TagMap, TagValue};
pub use writer::MetadataWriter;
