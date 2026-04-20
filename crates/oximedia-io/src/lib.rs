//! I/O layer for the OxiMedia multimedia framework.
//!
//! `oximedia-io` provides the full I/O stack used by OxiMedia's demuxers,
//! muxers, and codec pipeline: async media sources, magic-byte format
//! detection, bit-level reading, checksumming, buffered and memory-mapped
//! file I/O, and composable I/O pipeline stages.
//!
//! # Format detection
//!
//! [`format_detector::FormatDetector`] identifies over 45 media formats by
//! inspecting the leading bytes of a buffer (magic numbers).  Detected
//! categories include video containers (MP4, MKV, AVI, MOV, WebM, MXF, Ogg,
//! TS, M2TS, FLV), audio formats (FLAC, WAV, MP3, AAC, Opus, Vorbis, AIFF,
//! AU, CAF), still-image formats (JPEG, PNG, GIF, WebP, BMP, TIFF, HEIC,
//! AVIF, DPX, EXR, DNG, JPEG-XL), subtitle formats (SRT, VTT, ASS), and
//! archive formats (ZIP, TAR, GZ, BZ2, XZ, Zstandard).
//!
//! ```
//! use oximedia_io::format_detector::FormatDetector;
//!
//! let mp4_header = b"\x00\x00\x00\x1cftyp";
//! let result = FormatDetector::detect(mp4_header);
//! println!("format: {:?}", result.format);
//! println!("mime:   {}", result.mime_type);
//! ```
//!
//! # Media sources
//!
//! The [`source`] module provides the [`MediaSource`] async trait along with
//! two built-in implementations:
//!
//! - [`FileSource`] — Tokio async file reader (non-WASM targets only)
//! - [`MemorySource`] — zero-copy in-memory reader backed by `bytes::Bytes`
//!
//! ```no_run
//! use oximedia_io::source::{FileSource, MediaSource};
//!
//! #[tokio::main]
//! async fn main() -> oximedia_core::OxiResult<()> {
//!     let mut source = FileSource::open("video.webm").await?;
//!     let mut buf = [0u8; 4096];
//!     let n = source.read(&mut buf).await?;
//!     println!("read {n} bytes");
//!     Ok(())
//! }
//! ```
//!
//! # Bit-level reading
//!
//! [`BitReader`] (re-exported from the [`bits`] module) reads individual bits
//! and multi-bit values from a byte slice in MSB-first order — the standard
//! ordering used by H.264, HEVC, AV1, VP9, and most other video codecs.
//! Exp-Golomb coded integers (unsigned `ue(v)` and signed `se(v)`) are
//! supported via [`bits::BitReader::read_exp_golomb`] /
//! [`bits::BitReader::read_signed_exp_golomb`].
//!
//! ```
//! use oximedia_io::bits::BitReader;
//!
//! // Parse a tiny AVC-style bitfield: profile_idc (8) | constraint flags (6) | level_idc (8)
//! let sps_bytes = [0x64u8, 0x00, 0x1f];
//! let mut r = BitReader::new(&sps_bytes);
//! let profile_idc = r.read_bits(8).unwrap();   // 100 — High Profile
//! let _constraint = r.read_bits(6).unwrap();
//! let level_idc   = r.read_bits(8).unwrap();   // 31 — Level 3.1
//! assert_eq!(profile_idc, 100);
//! assert_eq!(level_idc,    31);
//! ```
//!
//! # MXF probing
//!
//! [`mxf_probe`] provides a lightweight parser for MXF (Material Exchange
//! Format) containers.  It detects the Header Partition Pack, identifies the
//! SMPTE Operational Pattern (OP1a, OPAtom, etc.), and enumerates essence
//! tracks (video, audio, data) without parsing the full KLV body.
//!
//! # Other utilities
//!
//! | Module | Capability |
//! |--------|------------|
//! | [`aligned_io`] | Memory-aligned I/O for DMA-friendly transfers |
//! | [`async_io`] | Async I/O abstractions |
//! | [`buffer_pool`] | Pooled byte buffers to reduce allocations |
//! | [`buffered_io`] | Read-ahead buffering with configurable window |
//! | [`buffered_reader`] | Buffered synchronous reader |
//! | [`checksum`] | CRC32, CRC64, SHA-256, BLAKE3 checksums |
//! | [`chunked_writer`] | Write large outputs in fixed-size chunks |
//! | [`compression`] | Compress / decompress (zstd, LZ4, gzip, bzip2) |
//! | [`content_detect`] | Text encoding and binary-vs-text detection |
//! | [`copy_engine`] | High-throughput async file copy |
//! | [`file_metadata`] | Extended file metadata (size, timestamps) |
//! | [`file_watch`] | File system event watching |
//! | [`io_pipeline`] | Composable I/O pipeline stages |
//! | [`io_stats`] | Throughput and latency statistics |
//! | [`mmap`] | Memory-mapped zero-copy file access |
//! | [`progress_reader`] | Async reader with progress callback |
//! | [`rate_limiter`] | Bandwidth-limited I/O |
//! | [`retrying_source`] | Automatic retry on transient I/O errors |
//! | [`ring_buffer`] | Lock-free ring buffer for streaming pipelines |
//! | [`scatter_gather`] | Vectorized scatter-gather I/O |
//! | [`seekable`] | Seekable source utilities |
//! | [`splice_pipe`] | Zero-copy splice between descriptors (Linux) |
//! | [`temp_files`] | Secure temporary file management |
//! | [`verify_io`] | Read-back verification for write integrity |
//! | [`write_journal`] | Journaled writes for crash-safe I/O |

pub mod aligned_io;
pub mod async_io;
pub mod bits;
pub mod buffer_pool;
pub mod buffered_io;
pub mod buffered_reader;
pub mod checksum;
pub mod chunked_writer;
pub mod compression;
pub mod content_detect;
pub mod copy_engine;
pub mod file_metadata;
pub mod file_watch;
pub mod format_detector;
pub mod io_pipeline;
pub mod io_stats;
pub mod mmap;
pub mod mxf_probe;
pub mod progress_reader;
pub mod rate_limiter;
pub mod retrying_source;
pub mod ring_buffer;
pub mod scatter_gather;
pub mod seekable;
pub mod source;
pub mod splice_pipe;
pub mod temp_files;
pub mod verify_io;
pub mod write_journal;

// Re-export commonly used types
pub use bits::BitReader;
#[cfg(not(target_arch = "wasm32"))]
pub use source::FileSource;
pub use source::{MediaSource, MemorySource};
