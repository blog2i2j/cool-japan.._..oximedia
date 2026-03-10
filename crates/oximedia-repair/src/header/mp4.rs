//! MP4/MOV header repair.
//!
//! This module provides functions to repair corrupted MP4/MOV file headers.

use crate::{RepairError, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Repair MP4 file header.
pub fn repair_mp4_header(input: &Path, output: &Path) -> Result<bool> {
    super::repair::copy_with_repair(input, output, |file| {
        let mut repaired = false;

        // Repair ftyp atom
        if repair_ftyp_atom(file)? {
            repaired = true;
        }

        // Repair moov atom
        if repair_moov_atom(file)? {
            repaired = true;
        }

        Ok(repaired)
    })
}

/// Repair ftyp (file type) atom.
fn repair_ftyp_atom(file: &mut File) -> Result<bool> {
    file.seek(SeekFrom::Start(0))?;

    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;

    // Check if ftyp atom exists
    if &header[4..8] != b"ftyp" {
        // Try to insert ftyp atom
        return insert_ftyp_atom(file);
    }

    // Check atom size
    let atom_size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);

    if atom_size == 0 || atom_size < 8 {
        // Fix atom size
        return fix_ftyp_size(file);
    }

    Ok(false)
}

/// Insert missing ftyp atom.
fn insert_ftyp_atom(file: &mut File) -> Result<bool> {
    // Create a basic ftyp atom
    let ftyp = create_ftyp_atom();

    // Read existing content
    file.seek(SeekFrom::Start(0))?;
    let mut content = Vec::new();
    file.read_to_end(&mut content)?;

    // Write ftyp followed by content
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&ftyp)?;
    file.write_all(&content)?;

    Ok(true)
}

/// Create a basic ftyp atom.
fn create_ftyp_atom() -> Vec<u8> {
    let mut atom = Vec::new();

    // Atom size (32 bytes)
    atom.extend_from_slice(&32u32.to_be_bytes());

    // Atom type
    atom.extend_from_slice(b"ftyp");

    // Major brand
    atom.extend_from_slice(b"mp42");

    // Minor version
    atom.extend_from_slice(&0u32.to_be_bytes());

    // Compatible brands
    atom.extend_from_slice(b"mp42");
    atom.extend_from_slice(b"iso2");
    atom.extend_from_slice(b"avc1");

    atom
}

/// Fix ftyp atom size.
fn fix_ftyp_size(file: &mut File) -> Result<bool> {
    // Calculate correct size
    file.seek(SeekFrom::Start(4))?;

    // Read past ftyp marker
    let mut buffer = [0u8; 4];
    file.read_exact(&mut buffer)?;

    // Find end of ftyp (start of next atom)
    let mut pos = 8u64;
    loop {
        file.seek(SeekFrom::Start(pos))?;
        if file.read_exact(&mut buffer).is_err() {
            break;
        }

        // Check if this looks like a valid atom type
        if is_valid_atom_type(&buffer) {
            break;
        }

        pos += 4;
        if pos > 1024 {
            // ftyp should not be larger than 1KB
            return Err(RepairError::RepairFailed(
                "Cannot determine ftyp size".to_string(),
            ));
        }
    }

    // Write corrected size
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&(pos as u32).to_be_bytes())?;

    Ok(true)
}

/// Check if bytes look like a valid atom type.
fn is_valid_atom_type(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b.is_ascii_alphanumeric())
}

/// Repair moov (movie) atom.
fn repair_moov_atom(file: &mut File) -> Result<bool> {
    // Try to find moov atom
    if let Some(offset) = find_atom(file, b"moov")? {
        // Validate moov atom
        file.seek(SeekFrom::Start(offset))?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header)?;

        let size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);

        if size == 0 {
            // Calculate correct size
            let file_size = file.metadata()?.len();
            let remaining = file_size - offset;

            file.seek(SeekFrom::Start(offset))?;
            file.write_all(&(remaining as u32).to_be_bytes())?;

            return Ok(true);
        }
    }

    Ok(false)
}

/// Find an atom in the file.
fn find_atom(file: &mut File, atom_type: &[u8; 4]) -> Result<Option<u64>> {
    file.seek(SeekFrom::Start(0))?;
    let file_size = file.metadata()?.len();

    let mut pos = 0u64;
    while pos + 8 <= file_size {
        file.seek(SeekFrom::Start(pos))?;

        let mut header = [0u8; 8];
        if file.read_exact(&mut header).is_err() {
            break;
        }

        if &header[4..8] == atom_type {
            return Ok(Some(pos));
        }

        let size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as u64;

        if size == 0 || size < 8 {
            pos += 8;
        } else {
            pos += size;
        }
    }

    Ok(None)
}

/// Relocate moov atom to beginning of file (if at end).
pub fn relocate_moov_atom(input: &Path, output: &Path) -> Result<bool> {
    let mut file = File::open(input)?;

    // Find moov atom
    let moov_offset = match find_atom(&mut file, b"moov")? {
        Some(offset) => offset,
        None => return Ok(false),
    };

    // Check if moov is already at beginning (after ftyp)
    if moov_offset < 1024 {
        return Ok(false);
    }

    // Read moov atom
    file.seek(SeekFrom::Start(moov_offset))?;
    let mut size_bytes = [0u8; 4];
    file.read_exact(&mut size_bytes)?;
    let moov_size = u32::from_be_bytes(size_bytes) as usize;

    file.seek(SeekFrom::Start(moov_offset))?;
    let mut moov_data = vec![0u8; moov_size];
    file.read_exact(&mut moov_data)?;

    // Read everything else
    file.seek(SeekFrom::Start(0))?;
    let mut before_moov = vec![0u8; moov_offset as usize];
    file.read_exact(&mut before_moov)?;

    file.seek(SeekFrom::Start(moov_offset + moov_size as u64))?;
    let mut after_moov = Vec::new();
    file.read_to_end(&mut after_moov)?;

    // Write to output: before + moov + after
    let mut output_file = File::create(output)?;
    output_file.write_all(&before_moov)?;
    output_file.write_all(&moov_data)?;
    output_file.write_all(&after_moov)?;

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_atom_type() {
        assert!(is_valid_atom_type(b"moov"));
        assert!(is_valid_atom_type(b"ftyp"));
        assert!(is_valid_atom_type(b"mdat"));
        assert!(!is_valid_atom_type(&[0xFF, 0xFF, 0xFF, 0xFF]));
    }

    #[test]
    fn test_create_ftyp_atom() {
        let ftyp = create_ftyp_atom();
        assert_eq!(ftyp.len(), 28);
        assert_eq!(&ftyp[4..8], b"ftyp");
        assert_eq!(&ftyp[8..12], b"mp42");
    }
}
