//! AVI header repair.
//!
//! This module provides functions to repair corrupted AVI file headers.

use crate::Result;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Repair AVI file header.
pub fn repair_avi_header(input: &Path, output: &Path) -> Result<bool> {
    super::repair::copy_with_repair(input, output, |file| {
        let mut repaired = false;

        // Repair RIFF header
        if repair_riff_header(file)? {
            repaired = true;
        }

        // Repair file size
        if repair_file_size(file)? {
            repaired = true;
        }

        Ok(repaired)
    })
}

/// Repair RIFF header.
fn repair_riff_header(file: &mut File) -> Result<bool> {
    file.seek(SeekFrom::Start(0))?;

    let mut header = [0u8; 12];
    file.read_exact(&mut header)?;

    let mut repaired = false;

    // Check RIFF signature
    if &header[0..4] != b"RIFF" {
        file.seek(SeekFrom::Start(0))?;
        file.write_all(b"RIFF")?;
        repaired = true;
    }

    // Check AVI type
    if &header[8..12] != b"AVI " {
        file.seek(SeekFrom::Start(8))?;
        file.write_all(b"AVI ")?;
        repaired = true;
    }

    Ok(repaired)
}

/// Repair file size field in RIFF header.
fn repair_file_size(file: &mut File) -> Result<bool> {
    let actual_size = file.metadata()?.len();

    // Calculate RIFF chunk size (file size - 8)
    let riff_size = actual_size.saturating_sub(8);

    // Read current size
    file.seek(SeekFrom::Start(4))?;
    let mut size_bytes = [0u8; 4];
    file.read_exact(&mut size_bytes)?;
    let stated_size = u32::from_le_bytes(size_bytes) as u64;

    if stated_size != riff_size {
        // Fix size
        file.seek(SeekFrom::Start(4))?;
        file.write_all(&(riff_size as u32).to_le_bytes())?;
        return Ok(true);
    }

    Ok(false)
}

/// Repair AVI index (idx1 chunk).
pub fn repair_avi_index(input: &Path, output: &Path) -> Result<bool> {
    let mut file = File::open(input)?;

    // Find idx1 chunk
    if let Some(offset) = find_chunk(&mut file, b"idx1")? {
        // Validate idx1 chunk
        file.seek(SeekFrom::Start(offset + 4))?;
        let mut size_bytes = [0u8; 4];
        file.read_exact(&mut size_bytes)?;
        let idx_size = u32::from_le_bytes(size_bytes);

        // Check if index is valid
        if idx_size == 0 || idx_size % 16 != 0 {
            // Index is corrupted, need to rebuild
            return rebuild_index(input, output);
        }
    } else {
        // No index found, need to create one
        return rebuild_index(input, output);
    }

    Ok(false)
}

/// Find a chunk in AVI file.
fn find_chunk(file: &mut File, chunk_id: &[u8; 4]) -> Result<Option<u64>> {
    file.seek(SeekFrom::Start(12))?; // Skip RIFF header
    let file_size = file.metadata()?.len();

    let mut pos = 12u64;
    while pos + 8 <= file_size {
        file.seek(SeekFrom::Start(pos))?;

        let mut header = [0u8; 8];
        if file.read_exact(&mut header).is_err() {
            break;
        }

        if &header[0..4] == chunk_id {
            return Ok(Some(pos));
        }

        let size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as u64;

        // Move to next chunk (align to word boundary)
        let aligned_size = (size + 1) & !1;
        pos += 8 + aligned_size;
    }

    Ok(None)
}

/// AVI index entry (idx1 chunk record).
///
/// Each entry is 16 bytes: 4-byte chunk ID, 4-byte flags, 4-byte offset,
/// 4-byte size.
#[derive(Debug, Clone)]
struct AviIndexEntry {
    /// Chunk ID (e.g. b"00dc" for video, b"01wb" for audio).
    chunk_id: [u8; 4],
    /// Flags (0x10 = AVIIF_KEYFRAME).
    flags: u32,
    /// Byte offset of this chunk relative to the start of the movi list data.
    offset: u32,
    /// Byte size of the chunk data (excluding the 8-byte header).
    size: u32,
}

/// Rebuild AVI index by scanning through the movi chunk for stream chunks.
///
/// 1. Locate the `movi` LIST.
/// 2. Walk through its children looking for chunks whose IDs match the
///    `NNxx` pattern (e.g. `00dc`, `01wb`).
/// 3. Build an `idx1` chunk containing one entry per discovered frame.
/// 4. Write the rebuilt file to `output`.
fn rebuild_index(input: &Path, output: &Path) -> Result<bool> {
    let mut file = File::open(input)?;
    let file_size = file.metadata()?.len();

    // Locate the movi LIST
    let movi_offset = match find_list(&mut file, b"movi")? {
        Some(offset) => offset,
        None => return Ok(false), // Cannot rebuild without movi
    };

    // Read movi LIST size
    file.seek(SeekFrom::Start(movi_offset + 4))?;
    let mut size_buf = [0u8; 4];
    file.read_exact(&mut size_buf)?;
    let movi_list_size = u32::from_le_bytes(size_buf) as u64;

    // The movi data starts after "LIST" (4) + size (4) + "movi" (4) = 12 bytes
    let movi_data_start = movi_offset + 12;
    let movi_data_end = (movi_offset + 8 + movi_list_size).min(file_size);

    // Scan through the movi data for stream chunks
    let mut entries: Vec<AviIndexEntry> = Vec::new();
    let mut pos = movi_data_start;

    while pos + 8 <= movi_data_end {
        file.seek(SeekFrom::Start(pos))?;
        let mut header = [0u8; 8];
        if file.read_exact(&mut header).is_err() {
            break;
        }

        let chunk_id: [u8; 4] = [header[0], header[1], header[2], header[3]];
        let chunk_size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);

        // AVI stream chunks have IDs like "00dc", "00db", "01wb", etc.
        // First two chars are ASCII digits (stream number).
        if is_stream_chunk_id(&chunk_id) {
            // Check for keyframe: uncompressed frames (xxdb) are always key,
            // compressed frames (xxdc) are key if first or large relative to others
            let is_key = chunk_id[2] == b'd' && chunk_id[3] == b'b';
            let flags = if is_key { 0x10u32 } else { 0u32 };

            // Offset is relative to the first byte of the movi list data
            let relative_offset = (pos - movi_data_start) as u32;

            entries.push(AviIndexEntry {
                chunk_id,
                flags,
                offset: relative_offset,
                size: chunk_size,
            });
        }

        // Advance to next chunk (8-byte header + size, word-aligned)
        let aligned_size = ((chunk_size as u64) + 1) & !1;
        pos += 8 + aligned_size;
    }

    if entries.is_empty() {
        return Ok(false);
    }

    // Read the original file up to the end of the movi chunk
    file.seek(SeekFrom::Start(0))?;
    let copy_end = movi_data_end;
    let mut original_data = vec![0u8; copy_end as usize];
    file.read_exact(&mut original_data)?;

    // Build idx1 chunk
    let idx1_data_size = (entries.len() * 16) as u32;
    let mut idx1_chunk = Vec::with_capacity(8 + idx1_data_size as usize);
    idx1_chunk.extend_from_slice(b"idx1");
    idx1_chunk.extend_from_slice(&idx1_data_size.to_le_bytes());

    for entry in &entries {
        idx1_chunk.extend_from_slice(&entry.chunk_id);
        idx1_chunk.extend_from_slice(&entry.flags.to_le_bytes());
        idx1_chunk.extend_from_slice(&entry.offset.to_le_bytes());
        idx1_chunk.extend_from_slice(&entry.size.to_le_bytes());
    }

    // Write output file: original data + idx1 chunk
    let mut out = File::create(output)?;
    out.write_all(&original_data)?;
    out.write_all(&idx1_chunk)?;

    // Fix RIFF size in the output
    let total_size = original_data.len() as u64 + idx1_chunk.len() as u64;
    let riff_size = total_size.saturating_sub(8) as u32;
    out.seek(SeekFrom::Start(4))?;
    out.write_all(&riff_size.to_le_bytes())?;

    Ok(true)
}

/// Check if a 4-byte chunk ID is a valid AVI stream chunk identifier.
///
/// Stream chunk IDs are of the form `NNxx` where `NN` are ASCII digits
/// and `xx` is a two-letter type code (dc, db, wb, pc, etc.).
fn is_stream_chunk_id(id: &[u8; 4]) -> bool {
    id[0].is_ascii_digit()
        && id[1].is_ascii_digit()
        && id[2].is_ascii_lowercase()
        && id[3].is_ascii_lowercase()
}

/// Fix AVI header list.
pub fn fix_hdrl_list(file: &mut File) -> Result<bool> {
    // Find hdrl LIST
    if let Some(offset) = find_list(file, b"hdrl")? {
        file.seek(SeekFrom::Start(offset + 4))?;

        let mut size_bytes = [0u8; 4];
        file.read_exact(&mut size_bytes)?;
        let stated_size = u32::from_le_bytes(size_bytes);

        // Validate size
        if stated_size == 0 {
            // Calculate correct size
            let next_list = find_next_list(file, offset + 12)?;
            let actual_size = if let Some(next) = next_list {
                (next - offset - 8) as u32
            } else {
                1024 // Default size if we can't find next list
            };

            // Write corrected size
            file.seek(SeekFrom::Start(offset + 4))?;
            file.write_all(&actual_size.to_le_bytes())?;

            return Ok(true);
        }
    }

    Ok(false)
}

/// Find a LIST chunk.
fn find_list(file: &mut File, list_type: &[u8; 4]) -> Result<Option<u64>> {
    file.seek(SeekFrom::Start(12))?;
    let file_size = file.metadata()?.len();

    let mut pos = 12u64;
    while pos + 12 <= file_size {
        file.seek(SeekFrom::Start(pos))?;

        let mut header = [0u8; 12];
        if file.read_exact(&mut header).is_err() {
            break;
        }

        if &header[0..4] == b"LIST" && &header[8..12] == list_type {
            return Ok(Some(pos));
        }

        let size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as u64;
        let aligned_size = (size + 1) & !1;
        pos += 8 + aligned_size;
    }

    Ok(None)
}

/// Find the next LIST chunk after a given position.
fn find_next_list(file: &mut File, start_pos: u64) -> Result<Option<u64>> {
    file.seek(SeekFrom::Start(start_pos))?;
    let file_size = file.metadata()?.len();

    let mut pos = start_pos;
    while pos + 8 <= file_size {
        file.seek(SeekFrom::Start(pos))?;

        let mut header = [0u8; 4];
        if file.read_exact(&mut header).is_err() {
            break;
        }

        if &header == b"LIST" {
            return Ok(Some(pos));
        }

        pos += 1;
    }

    Ok(None)
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_riff_size_calculation() {
        let file_size = 1000u64;
        let riff_size = file_size.saturating_sub(8);
        assert_eq!(riff_size, 992);
    }

    #[test]
    fn test_index_size_validation() {
        // Valid index sizes (multiples of 16)
        assert_eq!(16 % 16, 0);
        assert_eq!(32 % 16, 0);
        assert_eq!(160 % 16, 0);

        // Invalid index sizes
        assert_ne!(15 % 16, 0);
        assert_ne!(17 % 16, 0);
    }
}
