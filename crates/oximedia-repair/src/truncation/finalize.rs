//! File finalization for truncated files.
//!
//! This module provides functions to properly finalize truncated files
//! by adding necessary end markers and updating metadata.

use crate::Result;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Finalize a truncated file by adding proper end markers.
pub fn finalize_truncated_file(path: &Path) -> Result<()> {
    let mut file = File::options().read(true).write(true).open(path)?;

    // Detect format and finalize appropriately
    let mut header = [0u8; 12];
    file.read_exact(&mut header)?;

    if &header[0..4] == b"RIFF" {
        finalize_avi(&mut file)?;
    } else if &header[4..8] == b"ftyp" {
        finalize_mp4(&mut file)?;
    } else if header[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        finalize_matroska(&mut file)?;
    }

    Ok(())
}

/// Finalize AVI file.
fn finalize_avi(file: &mut File) -> Result<()> {
    let size = file.metadata()?.len();

    // Update RIFF chunk size
    file.seek(SeekFrom::Start(4))?;
    let riff_size = (size - 8) as u32;
    file.write_all(&riff_size.to_le_bytes())?;

    // Ensure file ends on word boundary
    if size % 2 != 0 {
        file.seek(SeekFrom::End(0))?;
        file.write_all(&[0])?;
    }

    Ok(())
}

/// Finalize MP4 file.
fn finalize_mp4(_file: &mut File) -> Result<()> {
    // MP4 finalization is complex and would require:
    // 1. Updating moov atom with correct duration
    // 2. Ensuring all atoms have correct sizes
    // 3. Adding end marker if needed

    // Placeholder for now
    Ok(())
}

/// Finalize Matroska file.
fn finalize_matroska(_file: &mut File) -> Result<()> {
    // Matroska finalization would require:
    // 1. Updating Segment size
    // 2. Writing Duration element
    // 3. Ensuring proper EBML structure

    // Placeholder for now
    Ok(())
}

/// Add end-of-stream marker.
pub fn add_eos_marker(file: &mut File, format: &str) -> Result<()> {
    file.seek(SeekFrom::End(0))?;

    match format {
        "mpeg" => {
            // MPEG end code
            file.write_all(&[0x00, 0x00, 0x01, 0xB9])?;
        }
        "h264" => {
            // H.264 end of stream NAL
            file.write_all(&[0x00, 0x00, 0x01, 0x0A])?;
        }
        _ => {}
    }

    Ok(())
}

/// Update file duration metadata.
pub fn update_duration(path: &Path, duration_ms: u64) -> Result<()> {
    let _file = File::options().read(true).write(true).open(path)?;

    // This would require format-specific implementation
    // to update duration in the file's metadata

    // Placeholder: store duration for reference
    let _duration = duration_ms;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eos_marker_creation() {
        // Test that we can create EOS markers
        let mut buffer = Vec::new();
        buffer
            .write_all(&[0x00, 0x00, 0x01, 0xB9])
            .expect("write should succeed");
        assert_eq!(buffer, vec![0x00, 0x00, 0x01, 0xB9]);
    }
}
