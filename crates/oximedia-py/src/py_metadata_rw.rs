//! Python bindings for reading and writing media metadata via `oximedia-metadata`.
//!
//! Distinct from the existing `py_metadata.rs` (which exposes typed metadata
//! fields from the container layer), this module wraps the full
//! `oximedia-metadata` crate to support ID3v2, Vorbis Comments, EXIF, etc.

use std::collections::HashMap;

use pyo3::prelude::*;

use oximedia_metadata::{Metadata, MetadataFormat, MetadataValue};

// ---------------------------------------------------------------------------
// Python-visible types
// ---------------------------------------------------------------------------

/// Common metadata fields readable from a media file.
///
/// Fields map to the most widely-used tags across formats.  ``None`` means the
/// tag was not present in the source data.
#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct PyMediaMetadata {
    /// Track / clip title.
    #[pyo3(get, set)]
    pub title: Option<String>,
    /// Primary artist or creator.
    #[pyo3(get, set)]
    pub artist: Option<String>,
    /// Album or collection name.
    #[pyo3(get, set)]
    pub album: Option<String>,
    /// Release year (integer).
    #[pyo3(get, set)]
    pub year: Option<u32>,
    /// Genre tag string.
    #[pyo3(get, set)]
    pub genre: Option<String>,
    /// Free-form comment.
    #[pyo3(get, set)]
    pub comment: Option<String>,
    /// Track number within the album.
    #[pyo3(get, set)]
    pub track_number: Option<u32>,
    /// All raw tag key-value pairs as a dictionary.
    ///
    /// Keys are format-specific (e.g. ``"TIT2"`` for ID3v2, ``"TITLE"`` for
    /// Vorbis Comments).  Values are always strings.
    #[pyo3(get, set)]
    pub tags: HashMap<String, String>,
}

#[pymethods]
impl PyMediaMetadata {
    #[new]
    #[pyo3(signature = (title=None, artist=None, album=None, year=None, genre=None, comment=None, track_number=None))]
    pub fn new(
        title: Option<String>,
        artist: Option<String>,
        album: Option<String>,
        year: Option<u32>,
        genre: Option<String>,
        comment: Option<String>,
        track_number: Option<u32>,
    ) -> Self {
        Self {
            title,
            artist,
            album,
            year,
            genre,
            comment,
            track_number,
            tags: HashMap::new(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyMediaMetadata(title={:?}, artist={:?}, album={:?}, year={:?})",
            self.title, self.artist, self.album, self.year
        )
    }
}

/// EXIF metadata extracted from an image file.
#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct PyExifData {
    /// Camera make (manufacturer).
    #[pyo3(get)]
    pub make: Option<String>,
    /// Camera model.
    #[pyo3(get)]
    pub model: Option<String>,
    /// Original date-time string (as stored in EXIF).
    #[pyo3(get)]
    pub datetime_original: Option<String>,
    /// Artist / photographer name.
    #[pyo3(get)]
    pub artist: Option<String>,
    /// Copyright string.
    #[pyo3(get)]
    pub copyright: Option<String>,
    /// All raw EXIF key-value pairs.
    #[pyo3(get)]
    pub tags: HashMap<String, String>,
}

#[pymethods]
impl PyExifData {
    fn __repr__(&self) -> String {
        format!(
            "PyExifData(make={:?}, model={:?}, datetime_original={:?})",
            self.make, self.model, self.datetime_original
        )
    }
}

// ---------------------------------------------------------------------------
// Format auto-detection heuristic
// ---------------------------------------------------------------------------

/// Attempt to identify the metadata format from the first few bytes of `data`.
fn detect_format(data: &[u8]) -> MetadataFormat {
    if data.len() >= 3 && &data[0..3] == b"ID3" {
        return MetadataFormat::Id3v2;
    }
    if data.len() >= 4 && &data[0..4] == b"fLaC" {
        return MetadataFormat::VorbisComments;
    }
    if data.len() >= 4 && &data[0..4] == b"OggS" {
        return MetadataFormat::VorbisComments;
    }
    if data.len() >= 4 && (&data[0..4] == b"APET" || &data[0..4] == b"APEV") {
        return MetadataFormat::Apev2;
    }
    // EXIF / TIFF markers
    if data.len() >= 2 && ((&data[0..2] == b"II") || (&data[0..2] == b"MM")) {
        return MetadataFormat::Exif;
    }
    // Fallback: treat as ID3v2
    MetadataFormat::Id3v2
}

/// Extract a text value from a `MetadataValue` if possible.
fn value_as_str(v: &MetadataValue) -> Option<String> {
    match v {
        MetadataValue::Text(s) => Some(s.clone()),
        MetadataValue::TextList(list) => list.first().cloned(),
        MetadataValue::Integer(i) => Some(i.to_string()),
        MetadataValue::Float(f) => Some(f.to_string()),
        MetadataValue::DateTime(dt) => Some(dt.clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Helpers to map between `Metadata` and `PyMediaMetadata`
// ---------------------------------------------------------------------------

fn metadata_to_py(meta: &Metadata) -> PyMediaMetadata {
    let common = meta.common();
    let mut tags: HashMap<String, String> = HashMap::new();
    for (k, v) in meta.fields() {
        if let Some(s) = value_as_str(v) {
            tags.insert(k.clone(), s);
        }
    }

    PyMediaMetadata {
        title: common.title.clone(),
        artist: common.artist.clone(),
        album: common.album.clone(),
        year: common.year,
        genre: common.genre.clone(),
        comment: common.comment.clone(),
        track_number: common.track_number,
        tags,
    }
}

fn py_to_metadata(py_meta: &PyMediaMetadata, format: MetadataFormat) -> Metadata {
    let mut meta = Metadata::new(format);

    // Apply fields via the CommonFields bridge
    let mut common = oximedia_metadata::common::CommonFields::default();
    common.title = py_meta.title.clone();
    common.artist = py_meta.artist.clone();
    common.album = py_meta.album.clone();
    common.year = py_meta.year;
    common.genre = py_meta.genre.clone();
    common.comment = py_meta.comment.clone();
    common.track_number = py_meta.track_number;
    meta.set_common(&common);

    // Also inject raw tags
    for (k, v) in &py_meta.tags {
        meta.insert(k.clone(), MetadataValue::Text(v.clone()));
    }

    meta
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Read metadata embedded in a media file's raw bytes.
///
/// The format (ID3v2, Vorbis Comments, EXIF, APEv2, …) is auto-detected from
/// the magic bytes at the beginning of ``file_bytes``.
///
/// Parameters
/// ----------
/// file_bytes : bytes
///     Raw bytes of the media file (or just its header / tag block).
///
/// Returns
/// -------
/// PyMediaMetadata
#[pyfunction]
#[pyo3(signature = (file_bytes))]
pub fn read_metadata(file_bytes: &[u8]) -> PyResult<PyMediaMetadata> {
    if file_bytes.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "file_bytes must not be empty",
        ));
    }

    let format = detect_format(file_bytes);
    let meta = Metadata::parse(file_bytes, format)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(metadata_to_py(&meta))
}

/// Write updated metadata back into the file bytes.
///
/// Parameters
/// ----------
/// file_bytes : bytes
///     Original raw bytes of the media file.
/// meta : PyMediaMetadata
///     Updated metadata fields to embed.
///
/// Returns
/// -------
/// bytes
///     New raw bytes with the updated metadata embedded.
///
/// Notes
/// -----
/// The output contains only the serialised metadata tag block (not the full
/// media file).  In a production workflow you would replace the original tag
/// block inside the file with this data.
#[pyfunction]
#[pyo3(signature = (file_bytes, meta))]
pub fn write_metadata(file_bytes: &[u8], meta: &PyMediaMetadata) -> PyResult<Vec<u8>> {
    if file_bytes.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "file_bytes must not be empty",
        ));
    }

    let format = detect_format(file_bytes);
    let metadata_obj = py_to_metadata(meta, format);
    let encoded = metadata_obj
        .write()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(encoded)
}

/// Extract EXIF metadata from raw image bytes.
///
/// Parameters
/// ----------
/// image_bytes : bytes
///     Raw JPEG / TIFF image bytes (or just the EXIF block).
///
/// Returns
/// -------
/// PyExifData
#[pyfunction]
#[pyo3(signature = (image_bytes))]
pub fn extract_exif(image_bytes: &[u8]) -> PyResult<PyExifData> {
    if image_bytes.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "image_bytes must not be empty",
        ));
    }

    let meta = Metadata::parse(image_bytes, MetadataFormat::Exif)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mut tags: HashMap<String, String> = HashMap::new();
    for (k, v) in meta.fields() {
        if let Some(s) = value_as_str(v) {
            tags.insert(k.clone(), s);
        }
    }

    let make = tags
        .get("Make")
        .cloned()
        .or_else(|| tags.get("0x010f").cloned());
    let model = tags
        .get("Model")
        .cloned()
        .or_else(|| tags.get("0x0110").cloned());
    let datetime_original = tags
        .get("DateTimeOriginal")
        .cloned()
        .or_else(|| tags.get("0x9003").cloned());
    let artist = tags
        .get("Artist")
        .cloned()
        .or_else(|| tags.get("0x013b").cloned());
    let copyright = tags
        .get("Copyright")
        .cloned()
        .or_else(|| tags.get("0x8298").cloned());

    Ok(PyExifData {
        make,
        model,
        datetime_original,
        artist,
        copyright,
        tags,
    })
}

// ---------------------------------------------------------------------------
// Module registration helper
// ---------------------------------------------------------------------------

/// Register all metadata-rw classes and free functions into the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMediaMetadata>()?;
    m.add_class::<PyExifData>()?;
    m.add_function(wrap_pyfunction!(read_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(write_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(extract_exif, m)?)?;
    Ok(())
}
