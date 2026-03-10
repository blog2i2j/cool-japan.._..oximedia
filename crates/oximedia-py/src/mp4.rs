//! Python bindings for the MP4/ISOBMFF container demuxer.
//!
//! Only royalty-free codecs (AV1, VP9, Opus, FLAC, Vorbis) are supported.
//! Attempting to demux files containing H.264, H.265, AAC, or other
//! patent-encumbered codecs will raise a `RuntimeError`.

use crate::container::{Packet, StreamInfo};
use crate::error::from_oxi_error;
use oximedia_container::demux::{Demuxer as RustDemuxer, Mp4Demuxer as RustMp4Demuxer};
use oximedia_io::FileSource;
use pyo3::prelude::*;
use tokio::runtime::Runtime;

/// Demuxer for MP4/ISOBMFF containers.
///
/// Supports royalty-free codecs only (AV1, VP9, Opus, FLAC, Vorbis).
///
/// # Example (Python)
/// ```python
/// import oximedia
/// demuxer = oximedia.Mp4Demuxer("video.mp4")
/// demuxer.probe()
/// for stream in demuxer.streams():
///     print(f"Stream {stream.index}: {stream.codec}")
/// while True:
///     pkt = demuxer.read_packet()
///     if pkt is None:
///         break
///     print(f"Packet: stream={pkt.stream_index}, size={pkt.size()}")
/// ```
#[pyclass]
pub struct Mp4Demuxer {
    inner: Option<RustMp4Demuxer<FileSource>>,
    path: String,
    rt: Runtime,
}

#[pymethods]
impl Mp4Demuxer {
    /// Create a new MP4 demuxer for the given file path.
    ///
    /// # Arguments
    /// * `path` - Path to the MP4 file
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let rt = Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create async runtime: {}",
                e
            ))
        })?;

        let source = rt
            .block_on(async { FileSource::open(&path).await })
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to open '{}': {}", path, e))
            })?;

        let demuxer = RustMp4Demuxer::new(source);

        Ok(Self {
            inner: Some(demuxer),
            path,
            rt,
        })
    }

    /// Probe the MP4 file and populate stream information.
    ///
    /// Must be called before `streams()` or `read_packet()`.
    fn probe(&mut self) -> PyResult<()> {
        let demuxer = self
            .inner
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Demuxer has been closed"))?;

        self.rt
            .block_on(async { demuxer.probe().await })
            .map_err(from_oxi_error)?;

        Ok(())
    }

    /// Get a list of streams found in the container.
    ///
    /// Returns a list of `StreamInfo` objects.  Call `probe()` first.
    fn streams(&self) -> PyResult<Vec<StreamInfo>> {
        let demuxer = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Demuxer has been closed"))?;

        let streams = demuxer
            .streams()
            .iter()
            .map(|s| StreamInfo::from_rust(s.clone()))
            .collect();

        Ok(streams)
    }

    /// Read the next packet from the container.
    ///
    /// Returns `None` when the end of the stream is reached.
    fn read_packet(&mut self) -> PyResult<Option<Packet>> {
        let demuxer = self
            .inner
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Demuxer has been closed"))?;

        match self.rt.block_on(async { demuxer.read_packet().await }) {
            Ok(pkt) => Ok(Some(Packet::from_rust(pkt))),
            Err(oximedia_core::OxiError::Eof) => Ok(None),
            Err(e) => Err(from_oxi_error(e)),
        }
    }

    /// Get the file path associated with this demuxer.
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    fn __repr__(&self) -> String {
        format!("Mp4Demuxer('{}')", self.path)
    }
}
