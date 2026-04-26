//! Compression and decompression for EXR pixel data.

use crate::error::{ImageError, ImageResult};
use std::io::Read;

pub(crate) fn decompress_rle(compressed: &[u8]) -> ImageResult<Vec<u8>> {
    let mut output = Vec::new();
    let mut i = 0;

    while i < compressed.len() {
        let count = compressed[i] as i8;
        i += 1;

        if count < 0 {
            // Run of different bytes
            let run_length = (-count + 1) as usize;
            if i + run_length > compressed.len() {
                break;
            }
            output.extend_from_slice(&compressed[i..i + run_length]);
            i += run_length;
        } else {
            // Run of same byte
            let run_length = (count + 1) as usize;
            if i >= compressed.len() {
                break;
            }
            let byte = compressed[i];
            i += 1;
            output.extend(std::iter::repeat(byte).take(run_length));
        }
    }

    Ok(output)
}

pub(crate) fn decompress_zip(compressed: &[u8]) -> ImageResult<Vec<u8>> {
    use oxiarc_deflate::ZlibStreamDecoder;

    let mut decoder = ZlibStreamDecoder::new(compressed);
    let mut output = Vec::new();
    decoder
        .read_to_end(&mut output)
        .map_err(|e| ImageError::Compression(format!("ZIP decompression failed: {e}")))?;

    Ok(output)
}

pub(crate) fn compress_rle(data: &[u8]) -> ImageResult<Vec<u8>> {
    let mut output = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let start = i;
        let current = data[i];

        // Find run length
        let mut run_len = 1;
        while i + run_len < data.len() && data[i + run_len] == current && run_len < 127 {
            run_len += 1;
        }

        if run_len >= 3 {
            // Encode as run
            output.push((run_len - 1) as u8);
            output.push(current);
            i += run_len;
        } else {
            // Find literal run
            let mut lit_len = 1;
            while i + lit_len < data.len() && lit_len < 127 {
                let next_run = count_run(&data[i + lit_len..]);
                if next_run >= 3 {
                    break;
                }
                lit_len += 1;
            }

            output.push((-(lit_len as i8) + 1) as u8);
            output.extend_from_slice(&data[start..start + lit_len]);
            i += lit_len;
        }
    }

    Ok(output)
}

fn count_run(data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }

    let current = data[0];
    let mut count = 1;

    while count < data.len() && data[count] == current {
        count += 1;
    }

    count
}

pub(crate) fn compress_zip(data: &[u8]) -> ImageResult<Vec<u8>> {
    use oxiarc_deflate::ZlibStreamEncoder;
    use std::io::Write;

    let mut encoder = ZlibStreamEncoder::new(Vec::new(), 6);
    encoder
        .write_all(data)
        .map_err(|e| ImageError::Compression(format!("ZIP compression failed: {e}")))?;

    encoder
        .finish()
        .map_err(|e| ImageError::Compression(format!("ZIP compression failed: {e}")))
}
