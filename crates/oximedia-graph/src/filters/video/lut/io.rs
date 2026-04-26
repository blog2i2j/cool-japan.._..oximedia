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

/// Parse a Cinespace CSP 3D LUT from string content.
///
/// The CSP format (CSPLUTV100) is used by Nuke, Cinespace, and OpenColorIO.
/// It optionally includes a pre-LUT 1D section followed by a 3D cube section.
///
/// File order: B is outermost, G middle, R innermost (B-major).
/// Internal `Lut3d` storage: `index(r,g,b) = r*size² + g*size + b`.
/// Therefore we use `lut.set(r, g, b, color)` in the B-major loop to
/// correctly map file order to internal storage.
pub fn parse_csp_file(content: &str) -> Result<Lut3d, String> {
    // Collect non-empty, non-comment lines for sequential parsing.
    let lines: Vec<&str> = content
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    let mut cursor = 0;

    // 1. Verify header.
    let header = lines
        .get(cursor)
        .ok_or_else(|| "CSP file is empty".to_string())?;
    if *header != "CSPLUTV100" {
        return Err(format!(
            "Invalid CSP header: expected 'CSPLUTV100', got '{header}'"
        ));
    }
    cursor += 1;

    // 2. Verify format type.
    let format_line = lines
        .get(cursor)
        .ok_or_else(|| "CSP file missing format line".to_string())?;
    if *format_line != "3D" {
        return Err(format!(
            "Unsupported CSP format: expected '3D', got '{format_line}'"
        ));
    }
    cursor += 1;

    // 3. Skip optional METADATA block.
    if lines.get(cursor).copied() == Some("BEGIN METADATA") {
        cursor += 1;
        loop {
            let meta_line = lines
                .get(cursor)
                .ok_or_else(|| "CSP file: unterminated METADATA block".to_string())?;
            cursor += 1;
            if *meta_line == "END METADATA" {
                break;
            }
        }
    }

    // 4. Detect whether a pre-LUT 1D section is present.
    //
    // After metadata, we read an integer N.
    // - If the next line (after N) has exactly 1 token → N is the cube size (no pre-LUT).
    //   This branch also handles the case where the cube data starts immediately.
    // - If the next line has a different token count, N is the pre-LUT size.
    //
    // Edge case: N=3 and the next line has exactly 3 tokens is ambiguous but very rare;
    // we resolve it as pre-LUT (the safer interpretation for colour grading pipelines).
    let first_n_str = lines
        .get(cursor)
        .ok_or_else(|| "CSP file: expected size or pre-LUT size".to_string())?;
    let first_n: usize = first_n_str
        .parse()
        .map_err(|e| format!("CSP: expected integer, got '{first_n_str}': {e}"))?;
    cursor += 1;

    // Peek at the following line to distinguish pre-LUT from cube section.
    let peek_token_count = lines
        .get(cursor)
        .map(|l| l.split_whitespace().count())
        .unwrap_or(0);

    // A cube-size integer is followed by triplet data lines (3 tokens) or another integer.
    // A pre-LUT size is followed by float-array lines (first_n tokens, typically > 3 or = 2).
    // We treat it as pre-LUT when the following line has token count != 3 or first_n != peek.
    // Heuristic: if peek_token_count != 3 → pre-LUT present (the integer was pre-lut size).
    let cube_size = if peek_token_count != 3 {
        // Pre-LUT section: skip first_n tokens across 3 lines (R, G, B arrays).
        // Each of the 3 lines contains `first_n` space-separated floats.
        for _ in 0..3 {
            let pre_lut_line = lines
                .get(cursor)
                .ok_or_else(|| "CSP file: incomplete pre-LUT 1D section".to_string())?;
            // Validate that the line has the expected number of floats.
            let token_count = pre_lut_line.split_whitespace().count();
            if token_count != first_n {
                return Err(format!(
                    "CSP pre-LUT: expected {first_n} values per channel, got {token_count}"
                ));
            }
            cursor += 1;
        }
        // Read the actual 3D cube size.
        let cube_size_str = lines
            .get(cursor)
            .ok_or_else(|| "CSP file: missing 3D cube size after pre-LUT".to_string())?;
        let sz: usize = cube_size_str
            .parse()
            .map_err(|e| format!("CSP: expected cube size integer, got '{cube_size_str}': {e}"))?;
        cursor += 1;
        sz
    } else {
        // No pre-LUT — `first_n` was the cube size and the current line is cube data.
        first_n
    };

    if cube_size < 2 {
        return Err(format!(
            "CSP: cube size {cube_size} is too small (minimum 2)"
        ));
    }

    // 5. Parse cube_size³ triplets.
    // File iteration order: B outermost, G middle, R innermost (B-major).
    // Lut3d internal index: index(r,g,b) = r*size² + g*size + b.
    // We call lut.set(r, g, b, color) in B-major order to map correctly.
    let mut lut = Lut3d::new(cube_size);

    for b in 0..cube_size {
        for g in 0..cube_size {
            for r in 0..cube_size {
                let data_line = lines.get(cursor).ok_or_else(|| {
                    format!(
                        "CSP: unexpected end of data at entry ({r},{g},{b}); \
                         expected {} entries total",
                        cube_size * cube_size * cube_size
                    )
                })?;
                cursor += 1;

                let parts: Vec<&str> = data_line.split_whitespace().collect();
                if parts.len() < 3 {
                    return Err(format!("CSP: data line has too few values: '{data_line}'"));
                }

                let rv: f64 = parts[0]
                    .parse()
                    .map_err(|e| format!("CSP: invalid R value '{}': {e}", parts[0]))?;
                let gv: f64 = parts[1]
                    .parse()
                    .map_err(|e| format!("CSP: invalid G value '{}': {e}", parts[1]))?;
                let bv: f64 = parts[2]
                    .parse()
                    .map_err(|e| format!("CSP: invalid B value '{}': {e}", parts[2]))?;

                lut.set(r, g, b, RgbColor::new(rv, gv, bv));
            }
        }
    }

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
        LutFormat::Csp => {
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read CSP file: {e}"))?;
            parse_csp_file(&content)
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid CSP (size=2, 8 entries), no pre-LUT.
    const CSP_BASIC: &str = "CSPLUTV100\n\
        3D\n\
        \n\
        2\n\
        0.0 0.0 0.0\n\
        1.0 0.0 0.0\n\
        0.0 1.0 0.0\n\
        1.0 1.0 0.0\n\
        0.0 0.0 1.0\n\
        1.0 0.0 1.0\n\
        0.0 1.0 1.0\n\
        1.0 1.0 1.0\n";

    /// CSP with BEGIN/END METADATA block and explicit pre-LUT 1D section.
    const CSP_WITH_METADATA: &str = "CSPLUTV100\n\
        3D\n\
        \n\
        BEGIN METADATA\n\
        description: test LUT\n\
        author: COOLJAPAN\n\
        END METADATA\n\
        \n\
        2\n\
        0.0 1.0\n\
        0.0 1.0\n\
        0.0 1.0\n\
        \n\
        2\n\
        0.0 0.0 0.0\n\
        1.0 0.0 0.0\n\
        0.0 1.0 0.0\n\
        1.0 1.0 0.0\n\
        0.0 0.0 1.0\n\
        1.0 0.0 1.0\n\
        0.0 1.0 1.0\n\
        1.0 1.0 1.0\n";

    // ---------------------------------------------------------------------------
    // test_parse_csp_basic
    // ---------------------------------------------------------------------------
    // Verify a minimal CSP (no metadata, no pre-LUT) parses correctly.
    //
    // File entry order (B-major): for b in 0..2 { for g in 0..2 { for r in 0..2 }}
    //   Line 0: (r=0,g=0,b=0) → 0.0 0.0 0.0
    //   Line 1: (r=1,g=0,b=0) → 1.0 0.0 0.0
    //   Line 2: (r=0,g=1,b=0) → 0.0 1.0 0.0
    //   Line 3: (r=1,g=1,b=0) → 1.0 1.0 0.0
    //   Line 4: (r=0,g=0,b=1) → 0.0 0.0 1.0
    //   Line 5: (r=1,g=0,b=1) → 1.0 0.0 1.0
    //   Line 6: (r=0,g=1,b=1) → 0.0 1.0 1.0
    //   Line 7: (r=1,g=1,b=1) → 1.0 1.0 1.0
    //
    // This also validates ordering: naive data.push() would map line 4 to
    // internal index 4 = (r=1,g=0,b=0) — wrong.  lut.set(r,g,b) must be used.
    #[test]
    fn test_parse_csp_basic() {
        let lut = parse_csp_file(CSP_BASIC).expect("parse_csp_file should succeed");

        assert_eq!(lut.size, 2);
        assert_eq!(lut.data.len(), 8);

        // Verify corner entries that distinguish correct B-major ordering.
        assert_eq!(lut.get(0, 0, 0), RgbColor::new(0.0, 0.0, 0.0));
        assert_eq!(lut.get(1, 0, 0), RgbColor::new(1.0, 0.0, 0.0));
        assert_eq!(lut.get(0, 1, 0), RgbColor::new(0.0, 1.0, 0.0));
        assert_eq!(lut.get(1, 1, 0), RgbColor::new(1.0, 1.0, 0.0));
        assert_eq!(lut.get(0, 0, 1), RgbColor::new(0.0, 0.0, 1.0));
        assert_eq!(lut.get(1, 0, 1), RgbColor::new(1.0, 0.0, 1.0));
        assert_eq!(lut.get(0, 1, 1), RgbColor::new(0.0, 1.0, 1.0));
        assert_eq!(lut.get(1, 1, 1), RgbColor::new(1.0, 1.0, 1.0));
    }

    // ---------------------------------------------------------------------------
    // test_parse_csp_with_metadata
    // ---------------------------------------------------------------------------
    #[test]
    fn test_parse_csp_with_metadata() {
        let lut =
            parse_csp_file(CSP_WITH_METADATA).expect("parse_csp_file with metadata should succeed");

        assert_eq!(lut.size, 2);
        assert_eq!(lut.data.len(), 8);

        // Same corner checks — metadata and pre-LUT should not affect cube data.
        assert_eq!(lut.get(0, 0, 0), RgbColor::new(0.0, 0.0, 0.0));
        assert_eq!(lut.get(1, 0, 0), RgbColor::new(1.0, 0.0, 0.0));
        assert_eq!(lut.get(0, 0, 1), RgbColor::new(0.0, 0.0, 1.0));
        assert_eq!(lut.get(1, 1, 1), RgbColor::new(1.0, 1.0, 1.0));
    }

    // ---------------------------------------------------------------------------
    // test_parse_csp_invalid_header
    // ---------------------------------------------------------------------------
    #[test]
    fn test_parse_csp_invalid_header() {
        let bad_content = "BADHEADER\n3D\n\n2\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n1.0 1.0 0.0\n0.0 0.0 1.0\n1.0 0.0 1.0\n0.0 1.0 1.0\n1.0 1.0 1.0\n";
        let result = parse_csp_file(bad_content);
        assert!(result.is_err(), "Expected error for invalid header");
        let msg = result.err().unwrap();
        assert!(
            msg.contains("CSPLUTV100"),
            "Error should mention expected header: {msg}"
        );
    }

    // ---------------------------------------------------------------------------
    // test_parse_csp_size_2_roundtrip
    // ---------------------------------------------------------------------------
    // Re-verify size=2 (manageable alternative to generating 4913-line size-17 data).
    // Builds the CSP content programmatically from expected values and round-trips it.
    #[test]
    fn test_parse_csp_size_2_roundtrip() {
        // Build a 2×2×2 CSP programmatically in B-major order.
        let mut content = String::from("CSPLUTV100\n3D\n\n2\n");
        let size = 2usize;
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let rv = r as f64;
                    let gv = g as f64;
                    let bv = b as f64;
                    content.push_str(&format!("{rv:.1} {gv:.1} {bv:.1}\n"));
                }
            }
        }

        let lut = parse_csp_file(&content).expect("roundtrip parse should succeed");
        assert_eq!(lut.size, size);

        // Verify all 8 entries map to (r as f64, g as f64, b as f64).
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let got = lut.get(r, g, b);
                    assert_eq!(
                        got,
                        RgbColor::new(r as f64, g as f64, b as f64),
                        "mismatch at ({r},{g},{b}): got {got:?}"
                    );
                }
            }
        }
    }

    // ---------------------------------------------------------------------------
    // test_parse_csp_empty_metadata
    // ---------------------------------------------------------------------------
    #[test]
    fn test_parse_csp_empty_metadata() {
        let content = "CSPLUTV100\n3D\n\nBEGIN METADATA\nEND METADATA\n\n2\n\
            0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n1.0 1.0 0.0\n\
            0.0 0.0 1.0\n1.0 0.0 1.0\n0.0 1.0 1.0\n1.0 1.0 1.0\n";
        let lut = parse_csp_file(content).expect("empty metadata block should parse ok");
        assert_eq!(lut.size, 2);
        assert_eq!(lut.get(1, 0, 0), RgbColor::new(1.0, 0.0, 0.0));
    }

    // ---------------------------------------------------------------------------
    // test_parse_csp_missing_3d_tag
    // ---------------------------------------------------------------------------
    #[test]
    fn test_parse_csp_missing_3d_tag() {
        let content = "CSPLUTV100\n1D\n\n2\n0.0\n1.0\n";
        let result = parse_csp_file(content);
        assert!(result.is_err(), "Expected error for non-3D format");
        let msg = result.err().unwrap();
        assert!(msg.contains("3D"), "Error should mention '3D': {msg}");
    }
}
