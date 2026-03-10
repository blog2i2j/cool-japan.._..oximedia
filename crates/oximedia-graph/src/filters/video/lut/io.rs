//! LUT file I/O: parsing and exporting .cube, .3dl, and CSV formats.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

use super::lut3d::Lut3d;
use super::types::{LutFormat, RgbColor};

/// Parse a .cube format LUT file.
pub fn parse_cube_file(path: &Path) -> Result<Lut3d, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {e}"))?;
    let reader = BufReader::new(file);

    let mut lut_size = 0;
    let mut title = String::new();
    let mut domain_min = RgbColor::new(0.0, 0.0, 0.0);
    let mut domain_max = RgbColor::new(1.0, 1.0, 1.0);
    let mut data = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read line: {e}"))?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse header
        if let Some(rest) = line.strip_prefix("TITLE") {
            title = rest.trim().trim_matches('"').to_string();
        } else if line.starts_with("LUT_3D_SIZE") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                lut_size = parts[1]
                    .parse()
                    .map_err(|e| format!("Invalid LUT size: {e}"))?;
            }
        } else if line.starts_with("DOMAIN_MIN") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                domain_min = RgbColor::new(
                    parts[1]
                        .parse()
                        .map_err(|e| format!("Invalid domain min: {e}"))?,
                    parts[2]
                        .parse()
                        .map_err(|e| format!("Invalid domain min: {e}"))?,
                    parts[3]
                        .parse()
                        .map_err(|e| format!("Invalid domain min: {e}"))?,
                );
            }
        } else if line.starts_with("DOMAIN_MAX") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                domain_max = RgbColor::new(
                    parts[1]
                        .parse()
                        .map_err(|e| format!("Invalid domain max: {e}"))?,
                    parts[2]
                        .parse()
                        .map_err(|e| format!("Invalid domain max: {e}"))?,
                    parts[3]
                        .parse()
                        .map_err(|e| format!("Invalid domain max: {e}"))?,
                );
            }
        } else {
            // Parse data line
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let r: f64 = parts[0]
                    .parse()
                    .map_err(|e| format!("Invalid R value: {e}"))?;
                let g: f64 = parts[1]
                    .parse()
                    .map_err(|e| format!("Invalid G value: {e}"))?;
                let b: f64 = parts[2]
                    .parse()
                    .map_err(|e| format!("Invalid B value: {e}"))?;
                data.push(RgbColor::new(r, g, b));
            }
        }
    }

    if lut_size == 0 {
        return Err("No LUT_3D_SIZE found in file".to_string());
    }

    let expected_size = lut_size * lut_size * lut_size;
    if data.len() != expected_size {
        return Err(format!(
            "Data size mismatch: expected {expected_size}, got {}",
            data.len()
        ));
    }

    let mut lut = Lut3d::new(lut_size);
    lut.data = data;
    lut.domain_min = domain_min;
    lut.domain_max = domain_max;
    lut.title = title;

    Ok(lut)
}

/// Parse a .3dl format LUT file (Autodesk/Lustre).
pub fn parse_3dl_file(path: &Path) -> Result<Lut3d, String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open file: {e}"))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| format!("Failed to read file: {e}"))?;

    // .3dl files are typically 33x33x33 and use integer values 0-4095
    const SIZE: usize = 33;
    let mut lut = Lut3d::new(SIZE);

    let mut data_count = 0;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let r: u32 = parts[0]
                .parse()
                .map_err(|e| format!("Invalid R value: {e}"))?;
            let g: u32 = parts[1]
                .parse()
                .map_err(|e| format!("Invalid G value: {e}"))?;
            let b: u32 = parts[2]
                .parse()
                .map_err(|e| format!("Invalid B value: {e}"))?;

            // Convert from 0-4095 to 0.0-1.0
            let color = RgbColor::new(r as f64 / 4095.0, g as f64 / 4095.0, b as f64 / 4095.0);

            if data_count < lut.data.len() {
                lut.data[data_count] = color;
                data_count += 1;
            }
        }
    }

    if data_count != SIZE * SIZE * SIZE {
        return Err(format!("Incomplete .3dl data: got {data_count} entries"));
    }

    Ok(lut)
}

/// Parse a CSV format LUT file.
pub fn parse_csv_file(path: &Path) -> Result<Lut3d, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {e}"))?;
    let reader = BufReader::new(file);

    let mut data = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read line: {e}"))?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            let r: f64 = parts[0]
                .trim()
                .parse()
                .map_err(|e| format!("Invalid R: {e}"))?;
            let g: f64 = parts[1]
                .trim()
                .parse()
                .map_err(|e| format!("Invalid G: {e}"))?;
            let b: f64 = parts[2]
                .trim()
                .parse()
                .map_err(|e| format!("Invalid B: {e}"))?;
            data.push(RgbColor::new(r, g, b));
        }
    }

    // Infer size from data length
    let total = data.len();
    let size = (total as f64).cbrt().round() as usize;

    if size * size * size != total {
        return Err(format!(
            "Invalid CSV data size: {total} is not a perfect cube"
        ));
    }

    let mut lut = Lut3d::new(size);
    lut.data = data;

    Ok(lut)
}

/// Load a LUT from a file.
pub fn load_lut_file(path: &Path) -> Result<Lut3d, String> {
    let format = LutFormat::from_extension(path)
        .ok_or_else(|| format!("Unknown LUT file format: {}", path.display()))?;

    match format {
        LutFormat::Cube => parse_cube_file(path),
        LutFormat::Threedl => parse_3dl_file(path),
        LutFormat::Csv => parse_csv_file(path),
        LutFormat::Csp => Err("CSP format not yet implemented".to_string()),
    }
}

/// Export a LUT to .cube format.
pub fn export_cube_file(lut: &Lut3d, path: &Path, title: Option<&str>) -> Result<(), String> {
    let mut file = File::create(path).map_err(|e| format!("Failed to create file: {e}"))?;

    // Write header
    if let Some(title_str) = title {
        writeln!(file, "TITLE \"{title_str}\"").map_err(|e| format!("Write failed: {e}"))?;
    } else if !lut.title.is_empty() {
        writeln!(file, "TITLE \"{}\"", lut.title).map_err(|e| format!("Write failed: {e}"))?;
    }

    writeln!(file, "LUT_3D_SIZE {}", lut.size).map_err(|e| format!("Write failed: {e}"))?;

    // Write domain if not default
    if lut.domain_min != RgbColor::new(0.0, 0.0, 0.0)
        || lut.domain_max != RgbColor::new(1.0, 1.0, 1.0)
    {
        writeln!(
            file,
            "DOMAIN_MIN {:.6} {:.6} {:.6}",
            lut.domain_min.r, lut.domain_min.g, lut.domain_min.b
        )
        .map_err(|e| format!("Write failed: {e}"))?;

        writeln!(
            file,
            "DOMAIN_MAX {:.6} {:.6} {:.6}",
            lut.domain_max.r, lut.domain_max.g, lut.domain_max.b
        )
        .map_err(|e| format!("Write failed: {e}"))?;
    }

    // Write data
    for color in &lut.data {
        writeln!(file, "{:.6} {:.6} {:.6}", color.r, color.g, color.b)
            .map_err(|e| format!("Write failed: {e}"))?;
    }

    Ok(())
}

/// Export a LUT to .3dl format.
pub fn export_3dl_file(lut: &Lut3d, path: &Path) -> Result<(), String> {
    // .3dl format is typically 33x33x33
    let export_lut = if lut.size != 33 {
        lut.resize(33)
    } else {
        lut.clone()
    };

    let mut file = File::create(path).map_err(|e| format!("Failed to create file: {e}"))?;

    // Write data (integer values 0-4095)
    for color in &export_lut.data {
        let r = (color.r.clamp(0.0, 1.0) * 4095.0).round() as u32;
        let g = (color.g.clamp(0.0, 1.0) * 4095.0).round() as u32;
        let b = (color.b.clamp(0.0, 1.0) * 4095.0).round() as u32;
        writeln!(file, "{r} {g} {b}").map_err(|e| format!("Write failed: {e}"))?;
    }

    Ok(())
}

/// Export a LUT to CSV format.
pub fn export_csv_file(lut: &Lut3d, path: &Path) -> Result<(), String> {
    let mut file = File::create(path).map_err(|e| format!("Failed to create file: {e}"))?;

    // Write header
    writeln!(file, "R,G,B").map_err(|e| format!("Write failed: {e}"))?;

    // Write data
    for color in &lut.data {
        writeln!(file, "{:.6},{:.6},{:.6}", color.r, color.g, color.b)
            .map_err(|e| format!("Write failed: {e}"))?;
    }

    Ok(())
}
