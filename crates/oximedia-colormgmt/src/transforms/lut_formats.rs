//! LUT file format parsers (.cube, .3dl, .csp, etc.).

use crate::error::{ColorError, Result};
use crate::transforms::lut::Lut3D;
use std::io::{BufRead, BufReader, Read};

/// Supported LUT file formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutFormat {
    /// Adobe .cube format
    Cube,
    /// Autodesk .3dl format
    Threedly,
    /// Cinespace .csp format
    Csp,
    /// Resolve .mga format (binary)
    Mga,
}

/// Parses a .cube LUT file.
///
/// # Errors
///
/// Returns an error if the file is invalid or cannot be parsed.
pub fn parse_cube<R: Read>(reader: R) -> Result<Lut3D> {
    let reader = BufReader::new(reader);
    let mut size = 0;
    let mut data = Vec::new();
    let mut domain_min = [0.0_f32; 3];
    let mut domain_max = [1.0_f32; 3];

    for line in reader.lines() {
        let line = line.map_err(|e| ColorError::Parse(format!("Failed to read line: {e}")))?;
        let line = line.trim();

        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse size
        if line.starts_with("LUT_3D_SIZE") || line.starts_with("LUT_1D_SIZE") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                size = parts[1]
                    .parse()
                    .map_err(|_| ColorError::Parse("Invalid LUT size".to_string()))?;
            }
            continue;
        }

        // Parse domain min/max
        if line.starts_with("DOMAIN_MIN") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                domain_min[0] = parts[1].parse().unwrap_or(0.0);
                domain_min[1] = parts[2].parse().unwrap_or(0.0);
                domain_min[2] = parts[3].parse().unwrap_or(0.0);
            }
            continue;
        }

        if line.starts_with("DOMAIN_MAX") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                domain_max[0] = parts[1].parse().unwrap_or(1.0);
                domain_max[1] = parts[2].parse().unwrap_or(1.0);
                domain_max[2] = parts[3].parse().unwrap_or(1.0);
            }
            continue;
        }

        // Skip title
        if line.starts_with("TITLE") {
            continue;
        }

        // Parse data line
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let r: f32 = parts[0]
                .parse()
                .map_err(|_| ColorError::Parse("Invalid R value".to_string()))?;
            let g: f32 = parts[1]
                .parse()
                .map_err(|_| ColorError::Parse("Invalid G value".to_string()))?;
            let b: f32 = parts[2]
                .parse()
                .map_err(|_| ColorError::Parse("Invalid B value".to_string()))?;

            data.push(r);
            data.push(g);
            data.push(b);
        }
    }

    if size == 0 {
        return Err(ColorError::Parse("LUT size not specified".to_string()));
    }

    // Normalize data if domain is not [0, 1]
    if domain_min != [0.0, 0.0, 0.0] || domain_max != [1.0, 1.0, 1.0] {
        for i in (0..data.len()).step_by(3) {
            data[i] = (data[i] - domain_min[0]) / (domain_max[0] - domain_min[0]);
            data[i + 1] = (data[i + 1] - domain_min[1]) / (domain_max[1] - domain_min[1]);
            data[i + 2] = (data[i + 2] - domain_min[2]) / (domain_max[2] - domain_min[2]);
        }
    }

    Lut3D::new(data, size)
}

/// Parses a .3dl LUT file (Autodesk format).
///
/// # Errors
///
/// Returns an error if the file is invalid or cannot be parsed.
pub fn parse_3dl<R: Read>(reader: R) -> Result<Lut3D> {
    let reader = BufReader::new(reader);
    let mut data = Vec::new();
    let size;

    for line in reader.lines() {
        let line = line.map_err(|e| ColorError::Parse(format!("Failed to read line: {e}")))?;
        let line = line.trim();

        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse data line
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            // .3dl format uses 10-bit integer values (0-1023)
            let r: u16 = parts[0]
                .parse()
                .map_err(|_| ColorError::Parse("Invalid R value".to_string()))?;
            let g: u16 = parts[1]
                .parse()
                .map_err(|_| ColorError::Parse("Invalid G value".to_string()))?;
            let b: u16 = parts[2]
                .parse()
                .map_err(|_| ColorError::Parse("Invalid B value".to_string()))?;

            // Convert to 0.0-1.0 range
            data.push(f32::from(r) / 1023.0);
            data.push(f32::from(g) / 1023.0);
            data.push(f32::from(b) / 1023.0);
        }
    }

    // .3dl files are typically 33x33x33
    if data.len() == 33 * 33 * 33 * 3 {
        size = 33;
    } else if data.len() == 17 * 17 * 17 * 3 {
        size = 17;
    } else if data.len() == 65 * 65 * 65 * 3 {
        size = 65;
    } else {
        // Try to infer size
        let total_points = data.len() / 3;
        size = (total_points as f64).cbrt().round() as usize;
    }

    if size == 0 {
        return Err(ColorError::Parse(
            "Could not determine LUT size".to_string(),
        ));
    }

    Lut3D::new(data, size)
}

/// Writes a .cube LUT file.
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn write_cube(lut: &Lut3D, title: &str) -> Result<String> {
    let mut output = String::new();

    output.push_str(&format!("TITLE \"{title}\"\n"));
    output.push_str(&format!("LUT_3D_SIZE {}\n", lut.size));
    output.push_str("DOMAIN_MIN 0.0 0.0 0.0\n");
    output.push_str("DOMAIN_MAX 1.0 1.0 1.0\n\n");

    for i in (0..lut.data.len()).step_by(3) {
        output.push_str(&format!(
            "{:.6} {:.6} {:.6}\n",
            lut.data[i],
            lut.data[i + 1],
            lut.data[i + 2]
        ));
    }

    Ok(output)
}

/// Generates an identity 3D LUT in .cube format.
///
/// # Errors
///
/// Returns an error if generation fails.
pub fn generate_identity_cube(size: usize) -> Result<String> {
    let lut = Lut3D::identity(size);
    write_cube(&lut, "Identity LUT")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_cube_simple() {
        let cube_data = r#"TITLE "Test LUT"
LUT_3D_SIZE 2
0.0 0.0 0.0
0.0 0.0 1.0
0.0 1.0 0.0
0.0 1.0 1.0
1.0 0.0 0.0
1.0 0.0 1.0
1.0 1.0 0.0
1.0 1.0 1.0
"#;

        let cursor = Cursor::new(cube_data);
        let lut = parse_cube(cursor).expect("CUBE parsing should succeed");
        assert_eq!(lut.size, 2);
        assert_eq!(lut.data.len(), 2 * 2 * 2 * 3);
    }

    #[test]
    fn test_write_cube() {
        let lut = Lut3D::identity(2);
        let cube = write_cube(&lut, "Test").expect("CUBE writing should succeed");
        assert!(cube.contains("LUT_3D_SIZE 2"));
        assert!(cube.contains("TITLE \"Test\""));
    }

    #[test]
    fn test_generate_identity() {
        let cube = generate_identity_cube(2).expect("identity LUT generation should succeed");
        assert!(cube.contains("Identity LUT"));
        assert!(cube.contains("LUT_3D_SIZE 2"));
    }
}
