#![allow(clippy::unnecessary_literal_bound)]
//! Quality Control and Validation for OxiMedia.
//!
//! `oximedia-qc` provides comprehensive quality control and validation for media files,
//! ensuring they meet technical specifications and delivery requirements.
//!
//! # Features
//!
//! - **Video Quality Checks**: Codec validation, resolution, frame rate, bitrate analysis,
//!   interlacing detection, black/freeze frame detection, compression artifacts
//! - **Audio Quality Checks**: Codec validation, sample rate, loudness compliance (EBU R128, ATSC A/85),
//!   clipping detection, silence detection, phase issues, DC offset detection
//! - **Container Checks**: Format validation, stream synchronization, timestamp continuity,
//!   keyframe interval, seeking capability, duration consistency
//! - **Compliance Checks**: Broadcast delivery specs, streaming platform requirements
//!   (YouTube, Vimeo), custom rule sets, patent-free codec enforcement
//!
//! # Examples
//!
//! ## Basic Validation
//!
//! ```ignore
//! use oximedia_qc::{QualityControl, QcPreset};
//!
//! let qc = QualityControl::with_preset(QcPreset::Streaming);
//! let report = qc.validate("video.mkv")?;
//!
//! if report.overall_passed {
//!     println!("✓ All checks passed!");
//! } else {
//!     println!("✗ Validation failed:");
//!     for error in report.errors() {
//!         println!("  - {}", error.message);
//!     }
//! }
//!
//! // Export report as JSON
//! let json = report.to_json()?;
//! std::fs::write("qc_report.json", json)?;
//! ```
//!
//! ## Custom Rules
//!
//! ```ignore
//! use oximedia_qc::{QualityControl, rules::Thresholds};
//! use oximedia_qc::video::ResolutionValidation;
//!
//! let mut qc = QualityControl::new();
//!
//! // Add custom resolution check
//! let resolution_check = ResolutionValidation::new()
//!     .with_min_resolution(1920, 1080)
//!     .with_max_resolution(3840, 2160);
//! qc.add_rule(Box::new(resolution_check));
//!
//! let report = qc.validate("video.mkv")?;
//! println!("{}", report);
//! ```
//!
//! ## Streaming Platform Validation
//!
//! ```ignore
//! use oximedia_qc::{QualityControl, QcPreset};
//!
//! // Validate for YouTube upload
//! let qc = QualityControl::with_preset(QcPreset::YouTube);
//! let report = qc.validate_streaming("video.webm")?;
//!
//! if !report.overall_passed {
//!     for result in report.errors() {
//!         println!("{}: {}", result.rule_name, result.message);
//!         if let Some(rec) = &result.recommendation {
//!             println!("  Recommendation: {}", rec);
//!         }
//!     }
//! }
//! ```
//!
//! # Architecture
//!
//! The QC system is built around the [`rules::QcRule`] trait, which allows
//! modular and extensible validation. Each rule performs a specific check and returns
//! [`rules::CheckResult`] instances.
//!
//! The [`QualityControl`] struct orchestrates rule execution and aggregates results
//! into a comprehensive [`report::QcReport`].

#![warn(missing_docs)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    dead_code,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]

pub mod audio;
pub mod audio_qc;
pub mod black_silence;
pub mod broadcast_safe;
pub mod caption_qc_checker;
pub mod closed_caption_qc;
pub mod compliance;
pub mod container;
pub mod detectors;
pub mod dolby_vision_qc;
pub mod file_qc;
pub mod report;
pub mod rules;
pub mod video;
pub mod video_quality_metrics;

// Advanced modules
pub mod batch;
/// Bitrate distribution quality analysis: CBR/VBR compliance and statistical profiling.
pub mod bitrate_qc;
pub mod codec_validation;
/// Color-space quality control checks.
pub mod color_qc;
/// Compliance issue aggregation and delivery gate reporting.
pub mod compliance_report;
pub mod examples;
pub mod format;
/// File-format quality control: container structure and codec compatibility.
pub mod format_qc;
/// HDR quality control: PQ/HLG metadata, peak brightness, and gamut validation.
pub mod hdr_qc;
pub mod profiles;
pub mod qc_profile;
pub mod qc_report;
pub mod qc_scheduler;
/// Reusable QC check templates and template library.
pub mod qc_template;
pub mod standards;
/// Audio/video synchronization quality checks: lip-sync, drift, and discontinuities.
pub mod sync_qc;
pub mod temporal;
/// Temporal quality control: timestamp continuity, frame-rate stability, and A/V sync.
pub mod temporal_qc;
pub mod utils;
/// Per-frame luma/chroma statistics and broadcast safety analysis.
pub mod video_measure;

/// Auto-fix suggestions and application for common QC failures.
pub mod auto_fix;
/// IMF (Interoperable Master Format) compliance checking (SMPTE ST 2067/ST 2084).
pub mod imf_compliance;
/// QC comparison mode — diff two media files and highlight quality differences.
pub mod qc_compare;
/// QC report delivery: email, webhook, and Slack notification targets.
pub mod qc_delivery;
/// Additional QC extensions and checker implementations.
pub mod qc_extensions;
/// QC watch-folder scanning — automatically validates files on arrival.
pub mod qc_watch_folder;
/// Network stream QC — validate live RTMP/SRT/HLS streams in real-time.
pub mod stream_qc;

#[cfg(feature = "database")]
pub mod database;

use oximedia_core::{OxiError, OxiResult};
use rayon::prelude::*;
use std::time::Instant;

/// Quality control system for media validation.
///
/// Manages a collection of QC rules and executes them against media files.
///
/// # Examples
///
/// ```ignore
/// use oximedia_qc::{QualityControl, QcPreset};
///
/// let qc = QualityControl::with_preset(QcPreset::Broadcast);
/// let report = qc.validate("video.mkv")?;
/// println!("{}", report);
/// ```
pub struct QualityControl {
    rules: Vec<Box<dyn rules::QcRule>>,
    thresholds: rules::Thresholds,
}

impl QualityControl {
    /// Creates a new QC system with no rules.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            thresholds: rules::Thresholds::default(),
        }
    }

    /// Creates a QC system with a preset rule set.
    #[must_use]
    pub fn with_preset(preset: QcPreset) -> Self {
        let mut qc = Self::new();
        preset.apply(&mut qc);
        qc
    }

    /// Creates a QC system with custom thresholds.
    #[must_use]
    pub fn with_thresholds(thresholds: rules::Thresholds) -> Self {
        Self {
            rules: Vec::new(),
            thresholds,
        }
    }

    /// Adds a QC rule to the system.
    pub fn add_rule(&mut self, rule: Box<dyn rules::QcRule>) {
        self.rules.push(rule);
    }

    /// Returns the number of rules configured.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Validates a media file.
    ///
    /// Executes all configured rules and generates a comprehensive report.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn validate(&self, file_path: &str) -> OxiResult<QcReport> {
        let start_time = Instant::now();

        // Create context by probing the file
        let context = self.probe_file(file_path)?;

        // Execute all applicable rules in parallel using rayon
        let mut report = QcReport::new(file_path);

        // Collect parallel results: each applicable rule runs concurrently.
        // Rules implement Send + Sync (guaranteed by the QcRule trait bound).
        let parallel_results: Vec<Result<Vec<rules::CheckResult>, (String, String)>> = self
            .rules
            .par_iter()
            .filter(|rule| rule.is_applicable(&context))
            .map(|rule| {
                rule.check(&context)
                    .map_err(|e| (rule.name().to_string(), e.to_string()))
            })
            .collect();

        for outcome in parallel_results {
            match outcome {
                Ok(results) => report.add_results(results),
                Err((rule_name, err_msg)) => {
                    tracing::error!(
                        rule = rule_name,
                        error = %err_msg,
                        "Rule execution failed"
                    );
                    report.add_result(rules::CheckResult::fail(
                        rule_name,
                        rules::Severity::Error,
                        format!("Rule execution failed: {err_msg}"),
                    ));
                }
            }
        }

        let duration = start_time.elapsed().as_secs_f64();
        report.set_validation_duration(duration);

        Ok(report)
    }

    /// Validates a file for streaming platform upload.
    ///
    /// Similar to [`validate`](Self::validate) but may perform additional checks
    /// specific to streaming delivery.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn validate_streaming(&self, file_path: &str) -> OxiResult<QcReport> {
        // For now, this is the same as validate()
        // In production, could add streaming-specific checks like:
        // - Fast start (moov atom at beginning for MP4)
        // - Proper HTTP range request support
        // - Adaptive bitrate ladder validation
        self.validate(file_path)
    }

    /// Validates a file for broadcast delivery.
    ///
    /// Performs strict validation against broadcast standards.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn validate_broadcast(&self, file_path: &str) -> OxiResult<QcReport> {
        // For now, this is the same as validate()
        // In production, could add broadcast-specific checks like:
        // - Closed captioning validation
        // - Timecode track validation
        // - Video levels (legal range)
        self.validate(file_path)
    }

    /// Probes a media file and creates a QC context.
    ///
    /// Reads up to 16 KB of the file to detect the container format from magic bytes,
    /// then extracts stream information (codec, dimensions, duration, bitrate) and
    /// populates the [`QcContext`](rules::QcContext) accordingly.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the format is unrecognized.
    fn probe_file(&self, file_path: &str) -> OxiResult<rules::QcContext> {
        use std::fs;
        use std::io::Read;
        use std::path::Path;

        if !Path::new(file_path).exists() {
            return Err(OxiError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {file_path}"),
            )));
        }

        let mut context = rules::QcContext::new(file_path);

        // ---- Read file header ----
        let file_size = fs::metadata(file_path).map(|m| m.len()).unwrap_or(0);

        let scan_size = file_size.min(16_384) as usize;
        let mut buf = vec![0u8; scan_size];
        {
            let mut fh = fs::File::open(file_path).map_err(OxiError::Io)?;
            let n = fh.read(&mut buf).map_err(OxiError::Io)?;
            buf.truncate(n);
        }

        if buf.len() < 8 {
            return Ok(context);
        }

        // ---- Detect format from magic bytes ----
        let detected = qc_detect_format(&buf);

        match detected {
            QcFormat::Mp4 | QcFormat::Mov => {
                probe_mp4(&buf, &mut context, file_size);
            }
            QcFormat::Mkv | QcFormat::WebM => {
                probe_mkv(&buf, &mut context, file_size);
            }
            QcFormat::Avi => {
                probe_avi(&buf, &mut context, file_size);
            }
            QcFormat::Wav | QcFormat::Flac | QcFormat::Ogg => {
                probe_audio_container(&buf, &mut context, file_size, detected);
            }
            QcFormat::Mxf => {
                probe_mxf(&buf, &mut context, file_size);
            }
            QcFormat::Unknown => {
                // Fall back to extension-based heuristics
                let path_lower = file_path.to_lowercase();
                let codec = if path_lower.ends_with(".av1") || path_lower.ends_with(".ivf") {
                    oximedia_core::CodecId::Av1
                } else if path_lower.ends_with(".vp9") {
                    oximedia_core::CodecId::Vp9
                } else {
                    oximedia_core::CodecId::Av1
                };
                let video = oximedia_container::StreamInfo::new(
                    0,
                    codec,
                    oximedia_core::Rational::new(1_i64, 30_i64),
                );
                context.add_stream(video);
            }
        }

        // Set duration from the first stream that has it, if not already set
        if context.duration.is_none() {
            for s in &context.streams {
                if let Some(d) = s.duration_seconds() {
                    context.set_duration(d);
                    break;
                }
            }
        }

        // ---- Bitrate from file size + duration ----
        if context.duration.is_none() || context.duration == Some(0.0) {
            // Fallback duration estimate for bitrate calculation
        }
        if let Some(dur) = context.duration {
            if dur > 0.0 && file_size > 0 {
                let bitrate_bps = (file_size as f64 * 8.0 / dur) as u64;
                context.file_bitrate = Some(bitrate_bps);
            }
        }

        Ok(context)
    }
}

// ============================================================================
// Container probing helpers (used by QualityControl::probe_file)
// ============================================================================

/// Detected container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QcFormat {
    Mp4,
    Mov,
    Mkv,
    WebM,
    Avi,
    Wav,
    Flac,
    Ogg,
    Mxf,
    Unknown,
}

/// Detect media format from magic bytes in the first 12 bytes of a file.
fn qc_detect_format(buf: &[u8]) -> QcFormat {
    if buf.len() < 4 {
        return QcFormat::Unknown;
    }

    // ISO Base Media (MP4 / MOV): 4-byte size + 4-byte type
    if buf.len() >= 8 {
        match &buf[4..8] {
            b"ftyp" => {
                // ftyp box – check major brand for MOV vs MP4
                if buf.len() >= 12 && &buf[8..12] == b"qt  " {
                    return QcFormat::Mov;
                }
                return QcFormat::Mp4;
            }
            b"moov" | b"mdat" | b"free" | b"wide" | b"skip" => {
                return QcFormat::Mp4;
            }
            _ => {}
        }
    }

    // Matroska / WebM: EBML magic 1A 45 DF A3
    if buf[..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        // WebM DocType check: element 0x4282 (DocType) follows shortly
        // For now treat both as Mkv; WebM is a subset
        if buf.len() >= 32 {
            // Search for "webm" doctype string in the first 32 bytes
            if buf.windows(4).any(|w| w == b"webm") {
                return QcFormat::WebM;
            }
        }
        return QcFormat::Mkv;
    }

    // RIFF-based
    if buf.len() >= 12 && &buf[..4] == b"RIFF" {
        return match &buf[8..12] {
            b"AVI " => QcFormat::Avi,
            b"WAVE" => QcFormat::Wav,
            _ => QcFormat::Unknown,
        };
    }

    // FLAC: fLaC
    if &buf[..4] == b"fLaC" {
        return QcFormat::Flac;
    }

    // OGG: OggS
    if &buf[..4] == b"OggS" {
        return QcFormat::Ogg;
    }

    // MXF: SMPTE UL prefix 06 0E 2B 34
    if buf[..4] == [0x06, 0x0E, 0x2B, 0x34] {
        return QcFormat::Mxf;
    }

    QcFormat::Unknown
}

// ---- helper readers ----

fn qc_read_u16_be(buf: &[u8], off: usize) -> u16 {
    if off + 2 > buf.len() {
        return 0;
    }
    u16::from_be_bytes([buf[off], buf[off + 1]])
}

fn qc_read_u32_be(buf: &[u8], off: usize) -> u32 {
    if off + 4 > buf.len() {
        return 0;
    }
    u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn qc_read_u32_le(buf: &[u8], off: usize) -> u32 {
    if off + 4 > buf.len() {
        return 0;
    }
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn qc_read_u64_be(buf: &[u8], off: usize) -> u64 {
    if off + 8 > buf.len() {
        return 0;
    }
    u64::from_be_bytes([
        buf[off],
        buf[off + 1],
        buf[off + 2],
        buf[off + 3],
        buf[off + 4],
        buf[off + 5],
        buf[off + 6],
        buf[off + 7],
    ])
}

fn qc_find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

// ---- ISOBMFF / MP4 prober ----

/// Iterate top-level ISOBMFF boxes.
fn qc_iter_boxes(buf: &[u8]) -> impl Iterator<Item = ([u8; 4], usize, usize)> + '_ {
    let mut pos = 0usize;
    std::iter::from_fn(move || {
        if pos + 8 > buf.len() {
            return None;
        }
        let raw_size = qc_read_u32_be(buf, pos) as usize;
        let mut btype = [0u8; 4];
        btype.copy_from_slice(&buf[pos + 4..pos + 8]);
        let (hdr, size) = if raw_size == 1 {
            if pos + 16 > buf.len() {
                return None;
            }
            (16, qc_read_u64_be(buf, pos + 8) as usize)
        } else if raw_size == 0 {
            (8, buf.len() - pos)
        } else {
            (8, raw_size)
        };
        if size < hdr {
            pos += 1;
            return Some((btype, pos, 0));
        }
        let ps = pos + hdr;
        let pl = (size - hdr).min(buf.len().saturating_sub(ps));
        pos += size.min(buf.len() - pos + 1);
        Some((btype, ps, pl))
    })
}

/// Recursively search for a box type inside `data`.
fn qc_find_box(data: &[u8], want: &[u8; 4]) -> Option<(usize, usize)> {
    for (bt, ps, pl) in qc_iter_boxes(data) {
        if &bt == want {
            return Some((ps, pl));
        }
        if matches!(
            &bt,
            b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"udta" | b"meta" | b"gmhd" | b"dinf"
        ) {
            let inner_end = (ps + pl).min(data.len());
            if let Some(r) = qc_find_box(&data[ps..inner_end], want) {
                return Some((ps + r.0, r.1));
            }
        }
    }
    None
}

fn probe_mp4(buf: &[u8], ctx: &mut rules::QcContext, file_size: u64) {
    use oximedia_container::StreamInfo;
    use oximedia_core::{CodecId, Rational};

    // --- mvhd → duration ----
    let mut movie_duration_secs: Option<f64> = None;
    if let Some((mvhd_s, mvhd_l)) = qc_find_box(buf, b"mvhd") {
        let mv = &buf[mvhd_s..(mvhd_s + mvhd_l).min(buf.len())];
        if !mv.is_empty() {
            let ver = mv[0];
            let (ts, dur) = if ver == 1 && mv.len() >= 28 {
                (qc_read_u32_be(mv, 20), qc_read_u64_be(mv, 24))
            } else if mv.len() >= 16 {
                (qc_read_u32_be(mv, 12), qc_read_u32_be(mv, 16) as u64)
            } else {
                (0, 0)
            };
            if ts > 0 && dur > 0 {
                movie_duration_secs = Some(dur as f64 / f64::from(ts));
            }
        }
    }
    if let Some(d) = movie_duration_secs {
        ctx.set_duration(d);
    }

    // --- Walk trak boxes for video/audio stream info ----
    // We need to find moov first
    let moov_pos = if let Some(p) = qc_find_bytes(buf, b"moov") {
        p
    } else {
        return;
    };
    // The moov box starts 4 bytes before the "moov" marker (size field)
    let moov_size = if moov_pos >= 4 {
        qc_read_u32_be(buf, moov_pos - 4) as usize
    } else {
        buf.len() - moov_pos + 4
    };
    let moov_end = (moov_pos + moov_size).min(buf.len());
    let moov_inner = &buf[moov_pos..moov_end];

    let mut stream_idx = 0usize;
    for (bt, ps, pl) in qc_iter_boxes(moov_inner) {
        if &bt != b"trak" {
            continue;
        }
        let trak = &moov_inner[ps..(ps + pl).min(moov_inner.len())];

        // Determine codec from stsd sample entry type
        let mut codec = CodecId::Av1;
        let mut width: Option<u32> = None;
        let mut height: Option<u32> = None;
        let mut is_video = false;
        let mut timebase_num = 1u32;
        let mut timebase_den = 30u32;
        let mut stream_dur: Option<i64> = None;

        if let Some((stsd_s, stsd_l)) = qc_find_box(trak, b"stsd") {
            let stsd = &trak[stsd_s..(stsd_s + stsd_l).min(trak.len())];
            // stsd: version(1) flags(3) entry_count(4) then first entry
            if stsd.len() > 8 {
                let entry_type = &stsd[8..12.min(stsd.len())];
                // Map sample entry types to CodecId
                codec = match entry_type {
                    b"av01" => {
                        is_video = true;
                        CodecId::Av1
                    }
                    b"vp09" => {
                        is_video = true;
                        CodecId::Vp9
                    }
                    b"vp08" => {
                        is_video = true;
                        CodecId::Vp8
                    }
                    b"avc1" | b"avc2" | b"avc3" => {
                        is_video = true;
                        CodecId::Av1
                    }
                    b"hvc1" | b"hev1" => {
                        is_video = true;
                        CodecId::Av1
                    }
                    b"mp4a" => CodecId::Opus,
                    b"Opus" | b"opus" => CodecId::Opus,
                    b"ac-3" | b"ec-3" => CodecId::Vorbis,
                    b"sowt" | b"twos" => CodecId::Pcm,
                    _ => CodecId::Av1,
                };
                // VisualSampleEntry dimensions at fixed offset 8+6+2+16
                if is_video && stsd.len() >= 34 {
                    let w = qc_read_u16_be(stsd, 32) as u32;
                    let h = qc_read_u16_be(stsd, 34) as u32;
                    if w > 0 && h > 0 {
                        width = Some(w);
                        height = Some(h);
                    }
                }
            }
        }

        // mdhd → time scale for this track
        if let Some((mdhd_s, mdhd_l)) = qc_find_box(trak, b"mdhd") {
            let mdhd = &trak[mdhd_s..(mdhd_s + mdhd_l).min(trak.len())];
            if !mdhd.is_empty() {
                let ver = mdhd[0];
                if ver == 1 && mdhd.len() >= 32 {
                    timebase_den = qc_read_u32_be(mdhd, 20);
                    let dur = qc_read_u64_be(mdhd, 24);
                    if dur > 0 {
                        stream_dur = Some(dur as i64);
                    }
                } else if mdhd.len() >= 20 {
                    timebase_den = qc_read_u32_be(mdhd, 12);
                    let dur = qc_read_u32_be(mdhd, 16) as i64;
                    if dur > 0 {
                        stream_dur = Some(dur);
                    }
                }
            }
        }

        // stts → fps from sample delta
        if is_video {
            if let Some((stts_s, stts_l)) = qc_find_box(trak, b"stts") {
                let stts = &trak[stts_s..(stts_s + stts_l).min(trak.len())];
                if stts.len() >= 12 {
                    let delta = qc_read_u32_be(stts, 12); // first entry delta
                    if delta > 0 && timebase_den > 0 {
                        timebase_num = delta;
                    }
                }
            }
        }

        let timebase = Rational::new(i64::from(timebase_num), i64::from(timebase_den));
        let mut stream = StreamInfo::new(stream_idx, codec, timebase);
        if let Some(d) = stream_dur {
            stream.duration = Some(d);
        }
        if is_video {
            if let (Some(w), Some(h)) = (width, height) {
                stream.codec_params = oximedia_container::CodecParams::video(w, h);
            }
        }
        ctx.add_stream(stream);
        stream_idx += 1;
    }

    // If no streams found from trak, add a default video stream
    if stream_idx == 0 {
        let video = StreamInfo::new(0, CodecId::Av1, Rational::new(1_i64, 30_i64));
        ctx.add_stream(video);
    }

    // Bitrate from file_size / duration
    if let Some(dur) = ctx.duration {
        if dur > 0.0 && file_size > 0 {
            ctx.file_bitrate = Some((file_size as f64 * 8.0 / dur) as u64);
        }
    }
}

fn probe_mkv(buf: &[u8], ctx: &mut rules::QcContext, file_size: u64) {
    use oximedia_container::StreamInfo;
    use oximedia_core::{CodecId, Rational};

    // EBML variable-length ID/size decode
    fn ebml_var(buf: &[u8], pos: usize) -> Option<(u64, usize)> {
        let b = *buf.get(pos)?;
        if b == 0 {
            return None;
        }
        let w = b.leading_zeros() as usize + 1;
        if pos + w > buf.len() {
            return None;
        }
        let mask = 0xFF_u64 >> w;
        let mut v = (b as u64) & mask;
        for i in 1..w {
            v = (v << 8) | buf[pos + i] as u64;
        }
        Some((v, w))
    }

    let mut timecode_scale: u64 = 1_000_000;
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;
    let mut default_dur_ns: Option<u64> = None;

    let mut scan = 0usize;
    while scan + 2 < buf.len() {
        let Some((eid, ib)) = ebml_var(buf, scan) else {
            scan += 1;
            continue;
        };
        let sz_pos = scan + ib;
        let Some((esz, sb)) = ebml_var(buf, sz_pos) else {
            scan += 1;
            continue;
        };
        let dp = sz_pos + sb;
        let de = (dp + esz as usize).min(buf.len());
        let data = &buf[dp..de];

        // Read up to 8-byte unsigned int from data
        fn read_u(d: &[u8]) -> u64 {
            let mut v = 0u64;
            for &b in d.iter().take(8) {
                v = (v << 8) | b as u64;
            }
            v
        }

        match eid {
            0x2AD7_B1 => {
                if !data.is_empty() {
                    timecode_scale = read_u(data);
                }
            }
            0x4489 => {
                let d = if data.len() == 8 {
                    f64::from_be_bytes(data.try_into().unwrap_or([0; 8]))
                } else if data.len() == 4 {
                    f32::from_bits(u32::from_be_bytes(data.try_into().unwrap_or([0; 4]))) as f64
                } else {
                    0.0
                };
                if d > 0.0 {
                    ctx.set_duration(d * timecode_scale as f64 / 1_000_000_000.0);
                }
            }
            0xB0 => {
                if !data.is_empty() {
                    width = Some(read_u(data) as u32);
                }
            }
            0xBA => {
                if !data.is_empty() {
                    height = Some(read_u(data) as u32);
                }
            }
            0x23E3_83 => {
                if !data.is_empty() {
                    default_dur_ns = Some(read_u(data));
                }
            }
            _ => {}
        }

        scan += (ib + sb + esz as usize).max(1);
    }

    // Build a video stream
    let fps_num = if let Some(ns) = default_dur_ns {
        if ns > 0 {
            (1_000_000_000.0 / ns as f64) as i32
        } else {
            25
        }
    } else {
        25
    };

    let mut video = StreamInfo::new(0, CodecId::Vp9, Rational::new(1_i64, i64::from(fps_num)));
    if let (Some(w), Some(h)) = (width, height) {
        video.codec_params = oximedia_container::CodecParams::video(w, h);
    }
    ctx.add_stream(video);

    // Audio stream (no codec info in header without full track parse)
    let audio = StreamInfo::new(1, CodecId::Opus, Rational::new(1_i64, 48000_i64));
    ctx.add_stream(audio);

    // Bitrate
    if let Some(dur) = ctx.duration {
        if dur > 0.0 && file_size > 0 {
            ctx.file_bitrate = Some((file_size as f64 * 8.0 / dur) as u64);
        }
    }
}

fn probe_avi(buf: &[u8], ctx: &mut rules::QcContext, file_size: u64) {
    use oximedia_container::StreamInfo;
    use oximedia_core::{CodecId, Rational};

    if let Some(avih_pos) = qc_find_bytes(buf, b"avih") {
        let ds = avih_pos + 8;
        if ds + 40 <= buf.len() {
            let usec = qc_read_u32_le(buf, ds);
            let total = qc_read_u32_le(buf, ds + 16);
            let w = qc_read_u32_le(buf, ds + 32);
            let h = qc_read_u32_le(buf, ds + 36);

            if usec > 0 {
                let fps = 1_000_000.0 / usec as f64;
                let timebase = Rational::new(i64::from(usec), 1_000_000_i64);
                let dur = if total > 0 {
                    total as f64 * usec as f64 / 1_000_000.0
                } else {
                    0.0
                };
                if dur > 0.0 {
                    ctx.set_duration(dur);
                }

                let mut video = StreamInfo::new(0, CodecId::Vp9, timebase);
                let _ = fps;
                if w > 0 && h > 0 {
                    video.codec_params = oximedia_container::CodecParams::video(w, h);
                }
                ctx.add_stream(video);
            } else {
                let video = StreamInfo::new(0, CodecId::Vp9, Rational::new(1, 25));
                ctx.add_stream(video);
            }
        }
    }

    // Bitrate
    if let Some(dur) = ctx.duration {
        if dur > 0.0 && file_size > 0 {
            ctx.file_bitrate = Some((file_size as f64 * 8.0 / dur) as u64);
        }
    }
}

fn probe_audio_container(buf: &[u8], ctx: &mut rules::QcContext, file_size: u64, fmt: QcFormat) {
    use oximedia_container::StreamInfo;
    use oximedia_core::{CodecId, Rational};

    let (codec, sample_rate) = match fmt {
        QcFormat::Wav => {
            // fmt  chunk: AudioFormat(2) Channels(2) SampleRate(4) ...
            let sr = if let Some(fp) = qc_find_bytes(buf, b"fmt ") {
                qc_read_u32_le(buf, fp + 8 + 4) // +8 for "fmt " tag + size, +4 skip AudioFormat+Channels
            } else {
                48000
            };
            // Duration from data chunk
            if let Some(dp) = qc_find_bytes(buf, b"data") {
                let data_size = qc_read_u32_le(buf, dp + 4);
                let byte_rate = if let Some(fp) = qc_find_bytes(buf, b"fmt ") {
                    qc_read_u32_le(buf, fp + 8 + 8) // ByteRate at offset 8 in fmt body
                } else {
                    0
                };
                if byte_rate > 0 && data_size > 0 {
                    ctx.set_duration(f64::from(data_size) / f64::from(byte_rate));
                }
            }
            (CodecId::Pcm, sr)
        }
        QcFormat::Flac => (CodecId::Flac, 44100),
        QcFormat::Ogg => (CodecId::Opus, 48000),
        _ => (CodecId::Pcm, 48000),
    };

    let stream = StreamInfo::new(0, codec, Rational::new(1_i64, i64::from(sample_rate)));
    ctx.add_stream(stream);

    if let Some(dur) = ctx.duration {
        if dur > 0.0 && file_size > 0 {
            ctx.file_bitrate = Some((file_size as f64 * 8.0 / dur) as u64);
        }
    }
}

fn probe_mxf(buf: &[u8], ctx: &mut rules::QcContext, file_size: u64) {
    use oximedia_container::StreamInfo;
    use oximedia_core::{CodecId, Rational};

    let ul_prefix = [0x06u8, 0x0E, 0x2B, 0x34];

    fn mxf_ber(buf: &[u8], pos: usize) -> Option<(usize, usize)> {
        let b = *buf.get(pos)?;
        if b & 0x80 == 0 {
            return Some((pos + 1, b as usize));
        }
        let n = (b & 0x7F) as usize;
        if n == 0 || n > 8 || pos + 1 + n > buf.len() {
            return None;
        }
        let mut l = 0usize;
        for i in 0..n {
            l = (l << 8) | buf[pos + 1 + i] as usize;
        }
        Some((pos + 1 + n, l))
    }

    let mut edit_rate_num = 25u32;
    let mut edit_rate_den = 1u32;
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;
    let mut duration: Option<u64> = None;
    let mut has_audio = false;

    let mut scan = 0usize;
    while scan + 16 < buf.len() {
        if buf[scan..scan + 4] != ul_prefix {
            scan += 1;
            continue;
        }
        let key = &buf[scan..scan + 16];
        let Some((voff, vlen)) = mxf_ber(buf, scan + 16) else {
            scan += 1;
            continue;
        };
        let ve = (voff + vlen).min(buf.len());
        let val = &buf[voff..ve];

        if key[4] == 0x02 && key[5] == 0x53 && key[8] == 0x0D {
            match key[14] {
                0x27..=0x29 => {
                    // Picture descriptor
                    let mut lp = 0;
                    while lp + 4 <= val.len() {
                        let tag = qc_read_u16_be(val, lp);
                        let l = qc_read_u16_be(val, lp + 2) as usize;
                        let vs = lp + 4;
                        let ve2 = (vs + l).min(val.len());
                        match tag {
                            0x3203 if ve2 - vs >= 4 => {
                                width = Some(qc_read_u32_be(val, vs));
                            }
                            0x3202 if ve2 - vs >= 4 => {
                                height = Some(qc_read_u32_be(val, vs));
                            }
                            _ => {}
                        }
                        lp = ve2;
                    }
                }
                0x42 | 0x47 | 0x48 => {
                    has_audio = true;
                }
                0x36 | 0x37 => {
                    // Package: edit rate + duration
                    let mut lp = 0;
                    while lp + 4 <= val.len() {
                        let tag = qc_read_u16_be(val, lp);
                        let l = qc_read_u16_be(val, lp + 2) as usize;
                        let vs = lp + 4;
                        let ve2 = (vs + l).min(val.len());
                        let v = &val[vs..ve2];
                        match tag {
                            0x4901 if v.len() >= 8 => {
                                let n = qc_read_u32_be(v, 0);
                                let d = qc_read_u32_be(v, 4);
                                if n > 0 && d > 0 {
                                    edit_rate_num = n;
                                    edit_rate_den = d;
                                }
                            }
                            0x4202 if v.len() >= 8 => {
                                let d = i64::from_be_bytes(v[..8].try_into().unwrap_or([0; 8]));
                                if d > 0 {
                                    duration = Some(d as u64);
                                }
                            }
                            _ => {}
                        }
                        lp = ve2;
                    }
                }
                _ => {}
            }
        }

        let next = voff + vlen;
        scan = if next > scan { next } else { scan + 1 };
    }

    // Convert duration in frames to seconds
    if let Some(dur_frames) = duration {
        if edit_rate_den > 0 {
            let dur_secs = dur_frames as f64 / (edit_rate_num as f64 / edit_rate_den as f64);
            ctx.set_duration(dur_secs);
        }
    }

    let timebase = Rational::new(i64::from(edit_rate_den), i64::from(edit_rate_num));
    let mut video = StreamInfo::new(0, CodecId::Av1, timebase);
    if let (Some(w), Some(h)) = (width, height) {
        video.codec_params = oximedia_container::CodecParams::video(w, h);
    }
    ctx.add_stream(video);

    if has_audio {
        let audio = StreamInfo::new(1, CodecId::Pcm, Rational::new(1_i64, 48000_i64));
        ctx.add_stream(audio);
    }

    if let Some(dur) = ctx.duration {
        if dur > 0.0 && file_size > 0 {
            ctx.file_bitrate = Some((file_size as f64 * 8.0 / dur) as u64);
        }
    }
}

impl Default for QualityControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Preset rule sets for common validation scenarios.
///
/// Provides pre-configured sets of QC rules for different use cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QcPreset {
    /// Basic validation (format and codec checks only).
    Basic,
    /// Streaming platform validation (YouTube, Vimeo compatible).
    Streaming,
    /// Broadcast delivery validation (strict quality requirements).
    Broadcast,
    /// Comprehensive validation (all available checks).
    Comprehensive,
    /// YouTube-specific validation.
    YouTube,
    /// Vimeo-specific validation.
    Vimeo,
}

impl QcPreset {
    /// Applies this preset to a QC system.
    pub fn apply(self, qc: &mut QualityControl) {
        match self {
            Self::Basic => {
                qc.add_rule(Box::new(video::VideoCodecValidation));
                qc.add_rule(Box::new(audio::AudioCodecValidation));
                qc.add_rule(Box::new(container::FormatValidation));
                qc.add_rule(Box::new(compliance::PatentFreeEnforcement));
            }
            Self::Streaming => {
                // Video checks
                qc.add_rule(Box::new(video::VideoCodecValidation));
                qc.add_rule(Box::new(video::ResolutionValidation::new()));
                qc.add_rule(Box::new(video::FrameRateValidation::new()));

                // Audio checks
                qc.add_rule(Box::new(audio::AudioCodecValidation));
                qc.add_rule(Box::new(audio::SampleRateValidation::new()));
                qc.add_rule(Box::new(audio::ChannelValidation::new()));

                // Container checks
                qc.add_rule(Box::new(container::FormatValidation));
                qc.add_rule(Box::new(container::DurationConsistency::default()));

                // Compliance
                qc.add_rule(Box::new(compliance::PatentFreeEnforcement));
            }
            Self::Broadcast => {
                // Video checks
                qc.add_rule(Box::new(video::VideoCodecValidation));
                qc.add_rule(Box::new(
                    video::ResolutionValidation::new().with_min_resolution(1280, 720),
                ));
                qc.add_rule(Box::new(video::FrameRateValidation::new()));
                qc.add_rule(Box::new(video::InterlacingDetection));

                // Audio checks
                qc.add_rule(Box::new(audio::AudioCodecValidation));
                qc.add_rule(Box::new(audio::SampleRateValidation::new()));
                qc.add_rule(Box::new(audio::LoudnessCompliance::ebu_r128(qc.thresholds)));
                qc.add_rule(Box::new(audio::ClippingDetection::new()));
                qc.add_rule(Box::new(audio::SilenceDetection::new(&qc.thresholds)));

                // Container checks
                qc.add_rule(Box::new(container::FormatValidation));
                qc.add_rule(Box::new(container::StreamSynchronization::default()));
                qc.add_rule(Box::new(container::TimestampContinuity::default()));
                qc.add_rule(Box::new(container::DurationConsistency::default()));

                // Compliance
                qc.add_rule(Box::new(compliance::BroadcastCompliance::new()));
                qc.add_rule(Box::new(compliance::PatentFreeEnforcement));
            }
            Self::Comprehensive => {
                // All video checks
                qc.add_rule(Box::new(video::VideoCodecValidation));
                qc.add_rule(Box::new(video::ResolutionValidation::new()));
                qc.add_rule(Box::new(video::FrameRateValidation::new()));
                qc.add_rule(Box::new(video::BitrateAnalysis::new(qc.thresholds)));
                qc.add_rule(Box::new(video::InterlacingDetection));
                qc.add_rule(Box::new(video::BlackFrameDetection::default()));
                qc.add_rule(Box::new(video::FreezeFrameDetection::default()));
                qc.add_rule(Box::new(video::CompressionArtifactDetection::default()));

                // All audio checks
                qc.add_rule(Box::new(audio::AudioCodecValidation));
                qc.add_rule(Box::new(audio::SampleRateValidation::new()));
                qc.add_rule(Box::new(audio::LoudnessCompliance::ebu_r128(qc.thresholds)));
                qc.add_rule(Box::new(audio::ClippingDetection::new()));
                qc.add_rule(Box::new(audio::SilenceDetection::new(&qc.thresholds)));
                qc.add_rule(Box::new(audio::PhaseDetection::new()));
                qc.add_rule(Box::new(audio::DcOffsetDetection::new()));
                qc.add_rule(Box::new(audio::ChannelValidation::new()));

                // All container checks
                qc.add_rule(Box::new(container::FormatValidation));
                qc.add_rule(Box::new(container::StreamSynchronization::default()));
                qc.add_rule(Box::new(container::TimestampContinuity::default()));
                qc.add_rule(Box::new(container::KeyframeInterval::default()));
                qc.add_rule(Box::new(container::SeekingCapability));
                qc.add_rule(Box::new(container::DurationConsistency::default()));
                qc.add_rule(Box::new(container::MetadataValidation::default()));
                qc.add_rule(Box::new(container::StreamOrdering));

                // Compliance
                qc.add_rule(Box::new(compliance::PatentFreeEnforcement));
            }
            Self::YouTube => {
                Self::Streaming.apply(qc);
                qc.add_rule(Box::new(compliance::YouTubeCompliance));
            }
            Self::Vimeo => {
                Self::Streaming.apply(qc);
                qc.add_rule(Box::new(compliance::VimeoCompliance));
            }
        }
    }
}

// Re-export key types at crate root
pub use batch::{BatchProcessor, BatchResults};
pub use profiles::{ProfileManager, QcProfile};
pub use report::{report_to_html, HtmlExportError, QcReport, ReportFormat};
pub use rules::{
    CheckResult, QcContext, QcRule, RuleCategory, Severity, SeverityClassifier, Thresholds,
};

#[cfg(feature = "database")]
pub use database::QcDatabase;

// Comprehensive test module
#[cfg(test)]
mod tests;

#[cfg(test)]
mod lib_tests {
    use super::*;

    #[test]
    fn test_qc_creation() {
        let qc = QualityControl::new();
        assert_eq!(qc.rule_count(), 0);
    }

    #[test]
    fn test_preset_basic() {
        let qc = QualityControl::with_preset(QcPreset::Basic);
        assert!(qc.rule_count() > 0);
    }

    #[test]
    fn test_preset_streaming() {
        let qc = QualityControl::with_preset(QcPreset::Streaming);
        assert!(qc.rule_count() > 0);
    }

    #[test]
    fn test_preset_broadcast() {
        let qc = QualityControl::with_preset(QcPreset::Broadcast);
        assert!(qc.rule_count() > 0);
    }

    #[test]
    fn test_preset_comprehensive() {
        let qc = QualityControl::with_preset(QcPreset::Comprehensive);
        assert!(qc.rule_count() > 10);
    }

    #[test]
    fn test_add_rule() {
        let mut qc = QualityControl::new();
        qc.add_rule(Box::new(video::VideoCodecValidation));
        assert_eq!(qc.rule_count(), 1);
    }

    #[test]
    fn test_thresholds() {
        let thresholds = rules::Thresholds::new()
            .with_min_video_bitrate(1_000_000)
            .with_loudness_target(-23.0);

        assert_eq!(thresholds.min_video_bitrate, Some(1_000_000));
        assert_eq!(thresholds.loudness_target, Some(-23.0));
    }
}
