//! Video frame extraction utilities for the OxiMedia CLI.
//!
//! Provides helpers for extracting raw RGB24 frames from video files.
//! Y4M (YUV4MPEG2) is supported natively via [`oximedia_container`].
//! Other formats produce a descriptive error asking the user to convert first.

use anyhow::{bail, Context, Result};
use oximedia_container::demux::y4m::{Y4mChroma, Y4mDemuxer};
use std::io::Cursor;
use std::path::Path;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Extract a single video frame from `input` as packed RGB24 bytes.
///
/// Returns `(rgb_data, width, height)`.
///
/// # Format support
///
/// - **Y4M (YUV4MPEG2)**: supported for all standard chroma modes.
/// - **All other formats**: returns an error directing the user to convert to
///   Y4M first with `oximedia convert`.
///
/// # Errors
///
/// Returns an error if the file does not exist, is not a recognised Y4M file,
/// does not contain enough frames, or is corrupted.
pub fn extract_video_frame_rgb(input: &Path, frame_num: u64) -> Result<(Vec<u8>, u32, u32)> {
    if !input.exists() {
        bail!("Input file not found: {}", input.display());
    }

    // Peek at the first 9 bytes to detect Y4M magic.
    let magic = read_magic(input)?;
    if magic.starts_with(b"YUV4MPEG2") {
        extract_y4m_frame_rgb(input, frame_num)
    } else {
        bail!(
            "Frame extraction for this format is not yet supported. \
             Convert to Y4M first: oximedia convert --input {} --output /tmp/out.y4m",
            input.display()
        )
    }
}

/// Extract multiple frames from `input` in a single pass.
///
/// `frame_indices` must be **sorted ascending**. Frames not present in the
/// file are silently omitted from the result.
///
/// Returns a `Vec` of `(rgb_data, width, height)` tuples, one per requested
/// frame index that was successfully read.
///
/// # Errors
///
/// See [`extract_video_frame_rgb`].
pub fn extract_video_frames_rgb(
    input: &Path,
    frame_indices: &[u64],
) -> Result<Vec<(Vec<u8>, u32, u32)>> {
    if frame_indices.is_empty() {
        return Ok(Vec::new());
    }

    if !input.exists() {
        bail!("Input file not found: {}", input.display());
    }

    let magic = read_magic(input)?;
    if !magic.starts_with(b"YUV4MPEG2") {
        bail!(
            "Frame extraction for this format is not yet supported. \
             Convert to Y4M first: oximedia convert --input {} --output /tmp/out.y4m",
            input.display()
        )
    }

    extract_y4m_frames_rgb(input, frame_indices)
}

// ---------------------------------------------------------------------------
// Internal: Y4M single-frame extraction
// ---------------------------------------------------------------------------

fn extract_y4m_frame_rgb(input: &Path, frame_num: u64) -> Result<(Vec<u8>, u32, u32)> {
    let data = std::fs::read(input).context("Failed to read Y4M file")?;
    let mut demuxer = Y4mDemuxer::new(Cursor::new(data)).context("Failed to parse Y4M header")?;

    let width = demuxer.width();
    let height = demuxer.height();
    let chroma = demuxer.chroma();

    // Walk forward to the requested frame.
    let mut current: u64 = 0;
    loop {
        let raw = demuxer
            .read_frame()
            .with_context(|| format!("I/O error reading frame {current}"))?;

        match raw {
            None => {
                bail!(
                    "Y4M file has fewer than {} frames (only {} found)",
                    frame_num + 1,
                    current
                );
            }
            Some(yuv) => {
                if current == frame_num {
                    let rgb = yuv_to_rgb24(&yuv, width, height, chroma);
                    return Ok((rgb, width, height));
                }
                current += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: Y4M multi-frame extraction (single pass)
// ---------------------------------------------------------------------------

fn extract_y4m_frames_rgb(input: &Path, frame_indices: &[u64]) -> Result<Vec<(Vec<u8>, u32, u32)>> {
    let data = std::fs::read(input).context("Failed to read Y4M file")?;
    let mut demuxer = Y4mDemuxer::new(Cursor::new(data)).context("Failed to parse Y4M header")?;

    let width = demuxer.width();
    let height = demuxer.height();
    let chroma = demuxer.chroma();

    // We'll collect matching frames in order.
    let mut results: Vec<(Vec<u8>, u32, u32)> = Vec::with_capacity(frame_indices.len());
    let mut idx_iter = frame_indices.iter().peekable();
    let mut current: u64 = 0;

    loop {
        // If all requested indices have been satisfied, stop early.
        let Some(&next_wanted) = idx_iter.peek() else {
            break;
        };

        let raw = demuxer
            .read_frame()
            .with_context(|| format!("I/O error reading frame {current}"))?;

        match raw {
            None => break, // EOF — remaining indices are past the end of file.
            Some(yuv) => {
                if current == *next_wanted {
                    let rgb = yuv_to_rgb24(&yuv, width, height, chroma);
                    results.push((rgb, width, height));
                    idx_iter.next(); // consume this index
                }
                current += 1;
            }
        }
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Internal: magic detection
// ---------------------------------------------------------------------------

fn read_magic(input: &Path) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut f = std::fs::File::open(input).context("Failed to open input file")?;
    let mut buf = [0u8; 9];
    let n = f.read(&mut buf).context("Failed to read file header")?;
    Ok(buf[..n].to_vec())
}

// ---------------------------------------------------------------------------
// YUV → RGB conversion
// ---------------------------------------------------------------------------

/// Convert raw planar YUV data to packed RGB24, dispatching on chroma format.
///
/// BT.601 studio-swing coefficients are used (Y: 16-235, UV: 16-240).
/// For mono frames a grey ramp is produced.
pub fn yuv_to_rgb24(yuv: &[u8], width: u32, height: u32, chroma: Y4mChroma) -> Vec<u8> {
    match chroma {
        Y4mChroma::Mono => yuv_mono_to_rgb24(yuv, width, height),
        Y4mChroma::C420jpeg | Y4mChroma::C420mpeg2 | Y4mChroma::C420paldv => {
            yuv420_to_rgb24(yuv, width, height)
        }
        Y4mChroma::C422 => yuv422_to_rgb24(yuv, width, height),
        Y4mChroma::C444 => yuv444_to_rgb24(yuv, width, height),
        Y4mChroma::C444alpha => {
            // Strip the alpha plane (last w*h bytes) and treat remainder as 444.
            let pixel_count = (width as usize) * (height as usize);
            yuv444_to_rgb24(&yuv[..pixel_count * 3], width, height)
        }
    }
}

/// BT.601 full-range YUV → RGB clamped to [0, 255].
///
/// `y`, `u`, `v` are in the 8-bit range [0, 255] with U/V centred at 128.
#[inline(always)]
fn yuv_pixel_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let yf = y as f32;
    let uf = u as f32 - 128.0;
    let vf = v as f32 - 128.0;

    let r = (yf + 1.402 * vf).clamp(0.0, 255.0) as u8;
    let g = (yf - 0.344_136 * uf - 0.714_136 * vf).clamp(0.0, 255.0) as u8;
    let b = (yf + 1.772 * uf).clamp(0.0, 255.0) as u8;
    (r, g, b)
}

fn yuv420_to_rgb24(yuv: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    // Y4mChroma::bytes_per_frame uses (w+1)/2 for odd dimensions — match here.
    let chroma_w = (w + 1) / 2;
    let chroma_h = (h + 1) / 2;

    let y_size = w * h;
    let u_size = chroma_w * chroma_h;

    let y_plane = &yuv[..y_size];
    let u_plane = &yuv[y_size..y_size + u_size];
    let v_plane = &yuv[y_size + u_size..];

    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        for col in 0..w {
            let y = y_plane[row * w + col];
            let u = u_plane[(row / 2) * chroma_w + (col / 2)];
            let v = v_plane[(row / 2) * chroma_w + (col / 2)];
            let (r, g, b) = yuv_pixel_to_rgb(y, u, v);
            let idx = (row * w + col) * 3;
            rgb[idx] = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
    }
    rgb
}

fn yuv422_to_rgb24(yuv: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let chroma_w = (w + 1) / 2;

    let y_size = w * h;
    let u_size = chroma_w * h;

    let y_plane = &yuv[..y_size];
    let u_plane = &yuv[y_size..y_size + u_size];
    let v_plane = &yuv[y_size + u_size..];

    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        for col in 0..w {
            let y = y_plane[row * w + col];
            let u = u_plane[row * chroma_w + (col / 2)];
            let v = v_plane[row * chroma_w + (col / 2)];
            let (r, g, b) = yuv_pixel_to_rgb(y, u, v);
            let idx = (row * w + col) * 3;
            rgb[idx] = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
    }
    rgb
}

fn yuv444_to_rgb24(yuv: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let pixel_count = w * h;

    let y_plane = &yuv[..pixel_count];
    let u_plane = &yuv[pixel_count..pixel_count * 2];
    let v_plane = &yuv[pixel_count * 2..];

    let mut rgb = vec![0u8; pixel_count * 3];
    for i in 0..pixel_count {
        let (r, g, b) = yuv_pixel_to_rgb(y_plane[i], u_plane[i], v_plane[i]);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    rgb
}

fn yuv_mono_to_rgb24(yuv: &[u8], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width as usize) * (height as usize);
    let mut rgb = vec![0u8; pixel_count * 3];
    for (i, &luma) in yuv[..pixel_count].iter().enumerate() {
        rgb[i * 3] = luma;
        rgb[i * 3 + 1] = luma;
        rgb[i * 3 + 2] = luma;
    }
    rgb
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_y4m(width: u32, height: u32, frame_count: usize) -> Vec<u8> {
        let mut data: Vec<u8> = Vec::new();
        let header = format!("YUV4MPEG2 W{width} H{height} F25:1 Ip C420jpeg\n");
        data.extend_from_slice(header.as_bytes());
        let frame_size = Y4mChroma::C420jpeg.bytes_per_frame(width, height);
        for i in 0..frame_count {
            data.extend_from_slice(b"FRAME\n");
            let fill = (i & 0xFF) as u8;
            data.extend(std::iter::repeat(fill).take(frame_size));
        }
        data
    }

    #[test]
    fn test_yuv_pixel_to_rgb_grey() {
        // Y=128, U=128, V=128 should give near-grey (128, 128, 128)
        let (r, g, b) = yuv_pixel_to_rgb(128, 128, 128);
        assert!((r as i32 - 128).abs() <= 2);
        assert!((g as i32 - 128).abs() <= 2);
        assert!((b as i32 - 128).abs() <= 2);
    }

    #[test]
    fn test_extract_nonexistent_file() {
        let p = std::path::PathBuf::from("/tmp/oximedia_no_such_file_9999.y4m");
        let result = extract_video_frame_rgb(&p, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_frame_from_y4m_file() {
        let tmp = std::env::temp_dir().join("oximedia_fe_test_single.y4m");
        let data = make_y4m(4, 4, 3);
        std::fs::write(&tmp, &data).unwrap();

        let result = extract_video_frame_rgb(&tmp, 0);
        assert!(result.is_ok(), "{:?}", result);
        let (rgb, w, h) = result.unwrap();
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(rgb.len(), 4 * 4 * 3);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_extract_frame_out_of_range() {
        let tmp = std::env::temp_dir().join("oximedia_fe_test_oor.y4m");
        let data = make_y4m(4, 4, 2);
        std::fs::write(&tmp, &data).unwrap();

        let result = extract_video_frame_rgb(&tmp, 5);
        assert!(result.is_err());

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_extract_multiple_frames_single_pass() {
        let tmp = std::env::temp_dir().join("oximedia_fe_test_multi.y4m");
        let data = make_y4m(4, 4, 10);
        std::fs::write(&tmp, &data).unwrap();

        let indices = vec![0u64, 2, 5, 9];
        let result = extract_video_frames_rgb(&tmp, &indices);
        assert!(result.is_ok(), "{:?}", result);
        let frames = result.unwrap();
        assert_eq!(frames.len(), 4);
        for (rgb, w, h) in &frames {
            assert_eq!(*w, 4);
            assert_eq!(*h, 4);
            assert_eq!(rgb.len(), 4 * 4 * 3);
        }

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_yuv420_odd_dimensions() {
        // 7x7 — chroma planes use (7+1)/2=4 per axis, total frame = 49 + 16 + 16 = 81
        let frame_size = Y4mChroma::C420jpeg.bytes_per_frame(7, 7);
        assert_eq!(frame_size, 81);
        let yuv = vec![128u8; frame_size];
        let rgb = yuv420_to_rgb24(&yuv, 7, 7);
        assert_eq!(rgb.len(), 7 * 7 * 3);
    }

    #[test]
    fn test_unsupported_format_error() {
        let tmp = std::env::temp_dir().join("oximedia_fe_test_unsupported.mkv");
        // Write a fake MKV magic
        std::fs::write(&tmp, b"\x1a\x45\xdf\xa3not_real_mkv_data_here").unwrap();

        let result = extract_video_frame_rgb(&tmp, 0);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("not yet supported"), "unexpected error: {msg}");

        std::fs::remove_file(&tmp).ok();
    }
}
