//! Media file fingerprinting and metadata extraction.

use crate::error::{ConformError, ConformResult};
use crate::types::{FrameRate, Timecode};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use xxhash_rust::xxh3::Xxh3;

/// File fingerprint containing checksums.
#[derive(Debug, Clone)]
pub struct Fingerprint {
    /// MD5 checksum.
    pub md5: String,
    /// `XXHash` checksum.
    pub xxhash: String,
}

/// Media file information.
#[derive(Debug, Clone)]
pub struct MediaInfo {
    /// Duration in seconds.
    pub duration: Option<f64>,
    /// Width in pixels.
    pub width: Option<u32>,
    /// Height in pixels.
    pub height: Option<u32>,
    /// Frame rate.
    pub fps: Option<FrameRate>,
    /// Start timecode.
    pub timecode_start: Option<Timecode>,
}

/// Generate fingerprint for a media file.
///
/// # Errors
///
/// Returns an error if the file cannot be read.
pub fn generate_fingerprint<P: AsRef<Path>>(path: P) -> ConformResult<Fingerprint> {
    let mut file = File::open(path)?;
    let mut md5_context = md5::Context::new();
    let mut xxh_hasher = Xxh3::new();
    let mut buffer = vec![0; 8192];

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        md5_context.consume(&buffer[..n]);
        xxh_hasher.update(&buffer[..n]);
    }

    Ok(Fingerprint {
        md5: format!("{:x}", md5_context.finalize()),
        xxhash: format!("{:x}", xxh_hasher.digest()),
    })
}

/// Extract media information from a file by reading up to 16 KB of its binary content
/// and parsing container-specific header structures.
///
/// Supported formats and what is extracted:
/// - **MP4/MOV**: ftyp/moov box → duration from mvhd, fps from stts/stsd, dimensions, tmcd timecode
/// - **MKV/WebM**: EBML header → duration, pixel dimensions from `TrackEntry`
/// - **AVI**: RIFF/AVI header → duration and fps from avih chunk
/// - **WAV**: RIFF/WAVE fmt chunk → sample rate (fps field unused for audio-only)
/// - **MXF**: KLV scan for `OperationalPattern`, duration and edit rate
///
/// # Errors
///
/// Returns an error if the file cannot be opened or read.
pub fn extract_media_info<P: AsRef<Path>>(path: P) -> ConformResult<MediaInfo> {
    use std::fs::File;
    use std::io::Read;

    let path = path.as_ref();
    let mut file = File::open(path)?;

    // Read first 16 KB for magic / header scanning
    let mut buf = vec![0u8; 16_384];
    let n = file.read(&mut buf)?;
    buf.truncate(n);

    if buf.len() < 8 {
        return Ok(MediaInfo {
            duration: None,
            width: None,
            height: None,
            fps: None,
            timecode_start: None,
        });
    }

    // Dispatch on magic bytes / container signatures
    if is_mp4_or_mov(&buf) {
        return parse_mp4(&buf);
    }
    if is_mkv(&buf) {
        return parse_mkv(&buf);
    }
    if is_avi(&buf) {
        return parse_avi(&buf);
    }
    if is_wav(&buf) {
        return parse_wav(&buf);
    }
    if is_mxf(&buf) {
        return parse_mxf(&buf);
    }

    // Unknown format – return empty info rather than an error
    Ok(MediaInfo {
        duration: None,
        width: None,
        height: None,
        fps: None,
        timecode_start: None,
    })
}

// ---------------------------------------------------------------------------
// Format detection
// ---------------------------------------------------------------------------

/// Returns true if the buffer starts with an ISOBMFF/MP4 ftyp or moov box.
fn is_mp4_or_mov(buf: &[u8]) -> bool {
    // An ISO Base Media box starts with a 4-byte size then a 4-byte type.
    // We look for "ftyp", "moov", "mdat", or "free" / "skip" in the first box.
    if buf.len() < 8 {
        return false;
    }
    let box_type = &buf[4..8];
    matches!(
        box_type,
        b"ftyp" | b"moov" | b"mdat" | b"free" | b"skip" | b"wide"
    )
}

/// Returns true if the buffer begins with the Matroska/WebM EBML magic.
fn is_mkv(buf: &[u8]) -> bool {
    buf.len() >= 4 && buf[..4] == [0x1A, 0x45, 0xDF, 0xA3]
}

/// Returns true if this looks like an AVI file (RIFF....AVI ).
fn is_avi(buf: &[u8]) -> bool {
    buf.len() >= 12 && &buf[..4] == b"RIFF" && &buf[8..12] == b"AVI "
}

/// Returns true if this looks like a WAV file (RIFF....WAVE).
fn is_wav(buf: &[u8]) -> bool {
    buf.len() >= 12 && &buf[..4] == b"RIFF" && &buf[8..12] == b"WAVE"
}

/// Returns true if this looks like an MXF file (starts with SMPTE UL prefix).
fn is_mxf(buf: &[u8]) -> bool {
    buf.len() >= 4 && buf[..4] == [0x06, 0x0E, 0x2B, 0x34]
}

// ---------------------------------------------------------------------------
// Helper: big-endian integer reads
// ---------------------------------------------------------------------------

fn read_u16_be(buf: &[u8], off: usize) -> u16 {
    if off + 2 > buf.len() {
        return 0;
    }
    u16::from_be_bytes([buf[off], buf[off + 1]])
}

fn read_u32_be(buf: &[u8], off: usize) -> u32 {
    if off + 4 > buf.len() {
        return 0;
    }
    u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn read_u64_be(buf: &[u8], off: usize) -> u64 {
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

fn read_u32_le(buf: &[u8], off: usize) -> u32 {
    if off + 4 > buf.len() {
        return 0;
    }
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Find `needle` in `haystack`, returning the start position if found.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

// ---------------------------------------------------------------------------
// MP4 / MOV / ISOBMFF parser
// ---------------------------------------------------------------------------

/// Iterate over top-level ISO boxes in `buf`.
///
/// Yields `(box_type: [u8;4], start_of_payload: usize, payload_length: usize)`.
fn iter_mp4_boxes(buf: &[u8]) -> impl Iterator<Item = ([u8; 4], usize, usize)> + '_ {
    let mut pos = 0;
    std::iter::from_fn(move || {
        if pos + 8 > buf.len() {
            return None;
        }
        let raw_size = read_u32_be(buf, pos) as usize;
        let mut box_type = [0u8; 4];
        box_type.copy_from_slice(&buf[pos + 4..pos + 8]);

        let (hdr, size) = if raw_size == 1 {
            // Extended size: 8-byte size immediately after type
            if pos + 16 > buf.len() {
                return None;
            }
            let ext = read_u64_be(buf, pos + 8) as usize;
            (16, ext)
        } else if raw_size == 0 {
            // Extends to end of file
            (8, buf.len() - pos)
        } else {
            (8, raw_size)
        };

        if size < hdr || pos + size > buf.len() {
            // Truncated or malformed – just consume what's there
            let payload_start = pos + hdr;
            let payload_len = buf.len().saturating_sub(payload_start);
            pos = buf.len(); // stop after this
            return Some((box_type, payload_start, payload_len));
        }

        let payload_start = pos + hdr;
        let payload_len = size - hdr;
        pos += size;
        Some((box_type, payload_start, payload_len))
    })
}

/// Recursively find a box with the given 4-byte type inside `data`.
fn find_mp4_box(data: &[u8], want: &[u8; 4]) -> Option<(usize, usize)> {
    for (bt, pstart, plen) in iter_mp4_boxes(data) {
        if &bt == want {
            return Some((pstart, plen));
        }
        // Recurse into container boxes
        if matches!(
            &bt,
            b"moov"
                | b"trak"
                | b"mdia"
                | b"minf"
                | b"stbl"
                | b"udta"
                | b"meta"
                | b"ilst"
                | b"dinf"
                | b"gmhd"
        ) {
            let inner_end = (pstart + plen).min(data.len());
            if let Some(r) = find_mp4_box(&data[pstart..inner_end], want) {
                return Some((pstart + r.0, r.1));
            }
        }
    }
    None
}

fn parse_mp4(buf: &[u8]) -> ConformResult<MediaInfo> {
    let mut info = MediaInfo {
        duration: None,
        width: None,
        height: None,
        fps: None,
        timecode_start: None,
    };

    // ---- mvhd: movie header ----
    if let Some((mvhd_start, mvhd_len)) = find_mp4_box(buf, b"mvhd") {
        let mv = &buf[mvhd_start..(mvhd_start + mvhd_len).min(buf.len())];
        if mv.len() >= 20 {
            let version = mv[0];
            let (time_scale, duration_units) = if version == 1 {
                // v1: creation(8) modification(8) time_scale(4) duration(8) ...
                if mv.len() >= 28 {
                    (read_u32_be(mv, 20), read_u64_be(mv, 24))
                } else {
                    (0, 0)
                }
            } else {
                // v0: creation(4) modification(4) time_scale(4) duration(4) ...
                if mv.len() >= 16 {
                    (read_u32_be(mv, 12), u64::from(read_u32_be(mv, 16)))
                } else {
                    (0, 0)
                }
            };
            if time_scale > 0 && duration_units > 0 {
                info.duration = Some(duration_units as f64 / f64::from(time_scale));
            }
        }
    }

    // ---- tkhd + mdhd + stsd for first video track ----
    // Walk trak boxes
    let mut trak_pos = 0usize;
    while let Some(moov_start) = find_bytes(buf, b"moov") {
        // Walk boxes inside moov
        let moov_size_raw = read_u32_be(buf, moov_start.saturating_sub(4));
        let moov_end = if moov_start >= 4 {
            (moov_start + moov_size_raw as usize).min(buf.len())
        } else {
            buf.len()
        };
        let moov_inner = &buf[moov_start..moov_end];

        for (bt, pstart, plen) in iter_mp4_boxes(moov_inner) {
            if &bt != b"trak" {
                continue;
            }
            let trak = &moov_inner[pstart..(pstart + plen).min(moov_inner.len())];

            // Look for mdia → minf → stbl → stsd to detect video
            if let Some((stsd_s, stsd_l)) = find_mp4_box(trak, b"stsd") {
                let stsd = &trak[stsd_s..(stsd_s + stsd_l).min(trak.len())];
                // stsd: version(1) flags(3) entry_count(4) then sample entries
                if stsd.len() > 8 {
                    let codec_type = &stsd[8..12];
                    // Known video sample types start with 'a' for avc1, vp09, av01, etc.
                    let is_video = !matches!(
                        codec_type,
                        b"mp4a" | b"ac-3" | b"ec-3" | b"Opus" | b"sowt" | b"twos"
                    );
                    if is_video && stsd.len() >= 28 {
                        // VisualSampleEntry: reserved(6) data_ref(2) reserved(16) width(2) height(2)
                        let width_off = 8 + 6 + 2 + 16;
                        if stsd.len() >= width_off + 4 {
                            let w = u32::from(read_u16_be(stsd, width_off));
                            let h = u32::from(read_u16_be(stsd, width_off + 2));
                            if w > 0 && h > 0 {
                                info.width = Some(w);
                                info.height = Some(h);
                            }
                        }
                    }
                }
            }

            // stts → sample-to-time table: entry_count pairs of (count, delta)
            // fps = time_scale / delta  (for constant frame rate)
            if let Some((stts_s, stts_l)) = find_mp4_box(trak, b"stts") {
                let stts = &trak[stts_s..(stts_s + stts_l).min(trak.len())];
                // version(1) flags(3) entry_count(4) [count(4) delta(4)] ...
                if stts.len() >= 12 {
                    let _entry_count = read_u32_be(stts, 4);
                    let sample_delta = read_u32_be(stts, 12); // first entry delta

                    // Also need time_scale from mdhd
                    if let Some((mdhd_s, mdhd_l)) = find_mp4_box(trak, b"mdhd") {
                        let mdhd = &trak[mdhd_s..(mdhd_s + mdhd_l).min(trak.len())];
                        if mdhd.len() >= 4 {
                            let version = mdhd[0];
                            let ts = if version == 1 {
                                if mdhd.len() >= 28 {
                                    read_u32_be(mdhd, 20)
                                } else {
                                    0
                                }
                            } else {
                                if mdhd.len() >= 16 {
                                    read_u32_be(mdhd, 12)
                                } else {
                                    0
                                }
                            };
                            if ts > 0 && sample_delta > 0 {
                                let fps_val = f64::from(ts) / f64::from(sample_delta);
                                info.fps = Some(fps_from_f64(fps_val));
                            }
                        }
                    }
                }
            }
        }

        trak_pos = moov_end;
        break;
    }
    let _ = trak_pos; // suppress unused variable warning

    // ---- tmcd track: timecode ----
    if let Some(tmcd_pos) = find_bytes(buf, b"tmcd") {
        // The timecode value is stored as a frame count 8 bytes after the type tag.
        if tmcd_pos + 12 < buf.len() {
            let frames = read_u32_be(buf, tmcd_pos + 8);
            let fps_val = info
                .fps
                .map_or(25.0, super::super::types::FrameRate::as_f64);
            if fps_val > 0.0 {
                let total_secs = f64::from(frames) / fps_val;
                let h = (total_secs / 3600.0) as u8;
                let m = ((total_secs % 3600.0) / 60.0) as u8;
                let s = (total_secs % 60.0) as u8;
                let f = (frames % (fps_val as u32).max(1)) as u8;
                info.timecode_start = Some(Timecode::new(h, m, s, f));
            }
        }
    }

    Ok(info)
}

// ---------------------------------------------------------------------------
// MKV / WebM (EBML) parser
// ---------------------------------------------------------------------------

/// Decode a variable-length EBML ID or size from `buf` at `pos`.
///
/// Returns `(value, bytes_consumed)` or `None` if buffer is too short.
fn ebml_decode_var(buf: &[u8], pos: usize) -> Option<(u64, usize)> {
    let b = *buf.get(pos)?;
    if b == 0 {
        return None;
    }
    let width = b.leading_zeros() as usize + 1;
    if pos + width > buf.len() {
        return None;
    }
    let mask = 0xFF_u64 >> width;
    let mut val = u64::from(b) & mask;
    for i in 1..width {
        val = (val << 8) | u64::from(buf[pos + i]);
    }
    Some((val, width))
}

fn parse_mkv(buf: &[u8]) -> ConformResult<MediaInfo> {
    let mut info = MediaInfo {
        duration: None,
        width: None,
        height: None,
        fps: None,
        timecode_start: None,
    };

    // Scan for EBML elements by ID
    // Duration element ID: 0x4489, stored as float64 or float32
    // PixelWidth  ID: 0xB0, PixelHeight ID: 0xBA
    // DefaultDuration (ns/frame) ID: 0x23E383
    // TimecodeScale ID: 0x2AD7B1

    let mut timecode_scale: u64 = 1_000_000; // default 1ms per tick
    let mut default_duration_ns: Option<u64> = None;

    let mut pos = 0usize;
    while pos + 2 < buf.len() {
        let Some((elem_id, id_bytes)) = ebml_decode_var(buf, pos) else {
            pos += 1;
            continue;
        };
        let size_pos = pos + id_bytes;
        let Some((elem_size, sz_bytes)) = ebml_decode_var(buf, size_pos) else {
            pos += 1;
            continue;
        };
        let data_pos = size_pos + sz_bytes;
        let data_end = (data_pos + elem_size as usize).min(buf.len());
        let data = &buf[data_pos..data_end];

        match elem_id {
            // TimecodeScale (nanoseconds per tick)
            0x2AD7B1 => {
                if data.len() >= 4 {
                    timecode_scale = read_uint_be(data);
                }
            }
            // Duration (float, in TimecodeScale ticks)
            0x4489 => {
                let dur = if data.len() == 8 {
                    f64::from_be_bytes(data.try_into().unwrap_or([0; 8]))
                } else if data.len() == 4 {
                    let v = u32::from_be_bytes(data.try_into().unwrap_or([0; 4]));
                    f64::from(f32::from_bits(v))
                } else {
                    0.0
                };
                if dur > 0.0 {
                    // Duration in MKV is in TimecodeScale ticks * (scale / 1e9)
                    info.duration = Some(dur * timecode_scale as f64 / 1_000_000_000.0);
                }
            }
            // PixelWidth
            0xB0 => {
                if !data.is_empty() {
                    info.width = Some(read_uint_be(data) as u32);
                }
            }
            // PixelHeight
            0xBA => {
                if !data.is_empty() {
                    info.height = Some(read_uint_be(data) as u32);
                }
            }
            // DefaultDuration: nanoseconds per frame
            0x23E383 => {
                if !data.is_empty() {
                    default_duration_ns = Some(read_uint_be(data));
                }
            }
            _ => {}
        }

        let advance = id_bytes + sz_bytes + elem_size as usize;
        pos += advance.max(1);
    }

    if let Some(ns) = default_duration_ns {
        if ns > 0 {
            let fps_val = 1_000_000_000.0 / ns as f64;
            info.fps = Some(fps_from_f64(fps_val));
        }
    }

    Ok(info)
}

/// Read a big-endian unsigned integer from up to 8 bytes.
fn read_uint_be(data: &[u8]) -> u64 {
    let mut v = 0u64;
    for &b in data.iter().take(8) {
        v = (v << 8) | u64::from(b);
    }
    v
}

// ---------------------------------------------------------------------------
// AVI parser
// ---------------------------------------------------------------------------

fn parse_avi(buf: &[u8]) -> ConformResult<MediaInfo> {
    let mut info = MediaInfo {
        duration: None,
        width: None,
        height: None,
        fps: None,
        timecode_start: None,
    };

    // RIFF layout: "RIFF" size "AVI " then LIST "hdrl" → avih → streams
    // avih chunk: 56 bytes
    // Offset in avih: MicroSecPerFrame(4) MaxBytesPerSec(4) PaddingGranularity(4)
    //                 Flags(4) TotalFrames(4) InitialFrames(4) Streams(4)
    //                 SuggestedBufferSize(4) Width(4) Height(4) ...

    if let Some(avih_pos) = find_bytes(buf, b"avih") {
        // After the 4-byte type + 4-byte size
        let data_start = avih_pos + 8;
        if data_start + 40 <= buf.len() {
            let usec_per_frame = read_u32_le(buf, data_start);
            let total_frames = read_u32_le(buf, data_start + 16);
            let width = read_u32_le(buf, data_start + 32);
            let height = read_u32_le(buf, data_start + 36);

            if usec_per_frame > 0 {
                let fps_val = 1_000_000.0 / f64::from(usec_per_frame);
                info.fps = Some(fps_from_f64(fps_val));
                if total_frames > 0 {
                    info.duration =
                        Some(f64::from(total_frames) * f64::from(usec_per_frame) / 1_000_000.0);
                }
            }
            if width > 0 {
                info.width = Some(width);
            }
            if height > 0 {
                info.height = Some(height);
            }
        }
    }

    Ok(info)
}

// ---------------------------------------------------------------------------
// WAV parser
// ---------------------------------------------------------------------------

fn parse_wav(buf: &[u8]) -> ConformResult<MediaInfo> {
    let mut info = MediaInfo {
        duration: None,
        width: None,
        height: None,
        fps: None,
        timecode_start: None,
    };

    // RIFF/WAVE structure: "RIFF" size "WAVE" then chunks
    // fmt  chunk: AudioFormat(2) NumChannels(2) SampleRate(4) ByteRate(4)
    //             BlockAlign(2) BitsPerSample(2)
    // data chunk: the audio data

    if let Some(fmt_pos) = find_bytes(buf, b"fmt ") {
        let data_start = fmt_pos + 8;
        if data_start + 16 <= buf.len() {
            let sample_rate = read_u32_le(buf, data_start + 4);
            let bits_per_sample = u32::from(read_u16_be(buf, data_start + 14));

            // For WAV, fps isn't meaningful but set sample_rate as metadata via fps
            if sample_rate > 0 {
                // audio-only: duration from data chunk size / byte_rate
                let byte_rate = read_u32_le(buf, data_start + 8);
                if let Some(data_pos) = find_bytes(buf, b"data") {
                    let data_size = read_u32_le(buf, data_pos + 4);
                    if byte_rate > 0 && data_size > 0 {
                        info.duration = Some(f64::from(data_size) / f64::from(byte_rate));
                    }
                }
                let _ = bits_per_sample;
            }
        }
    }

    Ok(info)
}

// ---------------------------------------------------------------------------
// MXF parser (header scan only)
// ---------------------------------------------------------------------------

fn parse_mxf(buf: &[u8]) -> ConformResult<MediaInfo> {
    let mut info = MediaInfo {
        duration: None,
        width: None,
        height: None,
        fps: None,
        timecode_start: None,
    };

    // Re-use the same KLV scanning logic as oximedia-imf/essence.rs:
    // Find Header Partition Pack to read OperationalPattern, then look for
    // GenericPictureEssenceDescriptor (width/height) and MaterialPackage (edit_rate/duration).
    let ul_prefix = [0x06u8, 0x0E, 0x2B, 0x34];

    // Scan for GenericPictureEssenceDescriptor (key byte[14] == 0x27..0x29)
    let mut scan = 0usize;
    while scan + 16 < buf.len() {
        if buf[scan..scan + 4] != ul_prefix {
            scan += 1;
            continue;
        }
        let key = &buf[scan..scan + 16];
        let Some((val_off, val_len)) = mxf_ber(buf, scan + 16) else {
            scan += 1;
            continue;
        };
        let val_end = (val_off + val_len).min(buf.len());
        let val = &buf[val_off..val_end];

        if key[4] == 0x02 && key[5] == 0x53 && key[8] == 0x0D {
            match key[14] {
                0x27..=0x29 => {
                    // LocalSet TLV scan for width/height
                    let mut lp = 0usize;
                    while lp + 4 <= val.len() {
                        let tag = read_u16_be(val, lp);
                        let len = read_u16_be(val, lp + 2) as usize;
                        let vs = lp + 4;
                        let ve = (vs + len).min(val.len());
                        let v = &val[vs..ve];
                        match tag {
                            0x3203 if v.len() >= 4 => {
                                info.width = Some(read_u32_be(v, 0));
                            }
                            0x3202 if v.len() >= 4 => {
                                info.height = Some(read_u32_be(v, 0));
                            }
                            _ => {}
                        }
                        lp = ve;
                    }
                }
                0x36 | 0x37 => {
                    // MaterialPackage / SourcePackage: edit rate + duration
                    let mut lp = 0usize;
                    while lp + 4 <= val.len() {
                        let tag = read_u16_be(val, lp);
                        let len = read_u16_be(val, lp + 2) as usize;
                        let vs = lp + 4;
                        let ve = (vs + len).min(val.len());
                        let v = &val[vs..ve];
                        match tag {
                            0x4901 if v.len() >= 8 => {
                                let num = read_u32_be(v, 0);
                                let den = read_u32_be(v, 4);
                                if num > 0 && den > 0 {
                                    info.fps = Some(fps_from_f64(f64::from(num) / f64::from(den)));
                                }
                            }
                            0x4202 if v.len() >= 8 => {
                                let d = i64::from_be_bytes(v[..8].try_into().unwrap_or([0; 8]));
                                if d > 0 {
                                    info.duration = Some(d as f64);
                                }
                            }
                            _ => {}
                        }
                        lp = ve;
                    }
                }
                _ => {}
            }
        }

        let next = val_off + val_len;
        scan = if next > scan { next } else { scan + 1 };
    }

    // Convert MXF duration (in frames) to seconds if we also have fps
    if let (Some(dur_frames), Some(ref fps)) = (info.duration, &info.fps) {
        let fps_val = fps.as_f64();
        if fps_val > 0.0 {
            info.duration = Some(dur_frames / fps_val);
        }
    }

    Ok(info)
}

/// Minimal BER-length decoder for the MXF parse helper above.
fn mxf_ber(buf: &[u8], pos: usize) -> Option<(usize, usize)> {
    let b = *buf.get(pos)?;
    if b & 0x80 == 0 {
        Some((pos + 1, b as usize))
    } else {
        let n = (b & 0x7F) as usize;
        if n == 0 || n > 8 || pos + 1 + n > buf.len() {
            return None;
        }
        let mut len = 0usize;
        for i in 0..n {
            len = (len << 8) | buf[pos + 1 + i] as usize;
        }
        Some((pos + 1 + n, len))
    }
}

// ---------------------------------------------------------------------------
// FPS matching helper
// ---------------------------------------------------------------------------

/// Map a floating-point fps value to the nearest known `FrameRate` variant.
fn fps_from_f64(fps: f64) -> FrameRate {
    // Tolerant comparison within ±0.02 fps
    if (fps - 23.976).abs() < 0.02 {
        return FrameRate::Fps23976;
    }
    if (fps - 24.0).abs() < 0.02 {
        return FrameRate::Fps24;
    }
    if (fps - 25.0).abs() < 0.02 {
        return FrameRate::Fps25;
    }
    if (fps - 29.97).abs() < 0.02 {
        return FrameRate::Fps2997NDF;
    }
    if (fps - 30.0).abs() < 0.02 {
        return FrameRate::Fps30;
    }
    if (fps - 50.0).abs() < 0.02 {
        return FrameRate::Fps50;
    }
    if (fps - 59.94).abs() < 0.02 {
        return FrameRate::Fps5994;
    }
    if (fps - 60.0).abs() < 0.02 {
        return FrameRate::Fps60;
    }
    FrameRate::Custom(fps)
}

/// Extract duration from a media file.
///
/// # Errors
///
/// Returns an error if duration cannot be determined.
pub fn extract_duration<P: AsRef<Path>>(path: P) -> ConformResult<f64> {
    extract_media_info(path)?
        .duration
        .ok_or_else(|| ConformError::Other("Duration not available".to_string()))
}

/// Extract frame rate from a media file.
///
/// # Errors
///
/// Returns an error if frame rate cannot be determined.
pub fn extract_frame_rate<P: AsRef<Path>>(path: P) -> ConformResult<FrameRate> {
    extract_media_info(path)?
        .fps
        .ok_or_else(|| ConformError::Other("Frame rate not available".to_string()))
}

/// Extract timecode from a media file.
///
/// # Errors
///
/// Returns an error if timecode cannot be determined.
pub fn extract_timecode<P: AsRef<Path>>(path: P) -> ConformResult<Timecode> {
    extract_media_info(path)?
        .timecode_start
        .ok_or_else(|| ConformError::Other("Timecode not available".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_generate_fingerprint() {
        let mut temp_file = NamedTempFile::new().expect("test expectation failed");
        temp_file
            .write_all(b"test content")
            .expect("write_all should succeed");
        temp_file.flush().expect("flush should succeed");

        let fingerprint =
            generate_fingerprint(temp_file.path()).expect("fingerprint should be valid");
        assert_eq!(fingerprint.md5.len(), 32); // MD5 is 32 hex chars
        assert!(!fingerprint.xxhash.is_empty());
    }

    #[test]
    fn test_fingerprint_consistency() {
        let mut temp_file = NamedTempFile::new().expect("test expectation failed");
        temp_file
            .write_all(b"test content")
            .expect("write_all should succeed");
        temp_file.flush().expect("flush should succeed");

        let fp1 = generate_fingerprint(temp_file.path()).expect("fp1 should be valid");
        let fp2 = generate_fingerprint(temp_file.path()).expect("fp2 should be valid");

        assert_eq!(fp1.md5, fp2.md5);
        assert_eq!(fp1.xxhash, fp2.xxhash);
    }

    #[test]
    fn test_extract_media_info() {
        let mut temp_file = NamedTempFile::new().expect("test expectation failed");
        temp_file
            .write_all(b"test content")
            .expect("write_all should succeed");
        temp_file.flush().expect("flush should succeed");

        let info = extract_media_info(temp_file.path()).expect("info should be valid");
        // Placeholder returns None values
        assert!(info.duration.is_none());
    }
}
