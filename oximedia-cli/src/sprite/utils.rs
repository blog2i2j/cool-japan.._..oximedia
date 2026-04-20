use super::*;

#[allow(dead_code)]
pub fn parse_timestamp(s: &str) -> Result<f64> {
    // Try parsing as seconds first
    if let Ok(seconds) = s.parse::<f64>() {
        return Ok(seconds);
    }

    // Try parsing as HH:MM:SS or MM:SS
    let parts: Vec<&str> = s.split(':').collect();

    match parts.len() {
        1 => parts[0].parse().context("Invalid time format"),
        2 => {
            let minutes: f64 = parts[0].parse().context("Invalid minutes")?;
            let seconds: f64 = parts[1].parse().context("Invalid seconds")?;
            Ok(minutes * 60.0 + seconds)
        }
        3 => {
            let hours: f64 = parts[0].parse().context("Invalid hours")?;
            let minutes: f64 = parts[1].parse().context("Invalid minutes")?;
            let seconds: f64 = parts[2].parse().context("Invalid seconds")?;
            Ok(hours * 3600.0 + minutes * 60.0 + seconds)
        }
        _ => Err(anyhow!("Invalid time format: {}", s)),
    }
}

/// Parse duration string (supports s, m, h suffixes).
pub fn parse_duration(s: &str) -> Result<f64> {
    let s = s.trim();

    if s.is_empty() {
        return Err(anyhow!("Empty duration string"));
    }

    // Check for suffix; s is non-empty (verified by the is_empty() guard above).
    let last_char = s.chars().last().unwrap_or('\0');

    if last_char.is_ascii_digit() || last_char == '.' {
        // No suffix, assume seconds
        s.parse().context("Invalid duration")
    } else {
        let (num_str, multiplier) = match last_char {
            's' | 'S' => (&s[..s.len() - 1], 1.0),
            'm' | 'M' => (&s[..s.len() - 1], 60.0),
            'h' | 'H' => (&s[..s.len() - 1], 3600.0),
            _ => return Err(anyhow!("Invalid duration suffix: {}", last_char)),
        };

        let num: f64 = num_str.parse().context("Invalid duration number")?;
        Ok(num * multiplier)
    }
}

/// Calculate optimal grid dimensions for a given thumbnail count.
#[allow(dead_code)]
pub fn calculate_optimal_grid(count: usize, aspect_ratio: f64) -> (usize, usize) {
    if count == 0 {
        return (0, 0);
    }

    if count == 1 {
        return (1, 1);
    }

    // Try to maintain approximately square layout
    let sqrt = (count as f64).sqrt();
    let mut cols = sqrt.ceil() as usize;
    let mut rows = (count as f64 / cols as f64).ceil() as usize;

    // Adjust for aspect ratio
    if aspect_ratio > 1.5 {
        // Wide aspect ratio, prefer more columns
        cols = (sqrt * 1.2).ceil() as usize;
        rows = (count as f64 / cols as f64).ceil() as usize;
    } else if aspect_ratio < 0.67 {
        // Tall aspect ratio, prefer more rows
        rows = (sqrt * 1.2).ceil() as usize;
        cols = (count as f64 / rows as f64).ceil() as usize;
    }

    (cols, rows)
}

/// Adjust grid dimensions to fit a specific count.
#[allow(dead_code)]
pub fn adjust_grid_for_count(cols: usize, mut rows: usize, count: usize) -> (usize, usize) {
    let total = cols * rows;

    if total == count {
        return (cols, rows);
    }

    if total > count {
        // Grid is too large, reduce
        let rows_needed = (count as f64 / cols as f64).ceil() as usize;
        rows = rows_needed;
    } else {
        // Grid is too small, increase
        let rows_needed = (count as f64 / cols as f64).ceil() as usize;
        rows = rows_needed;
    }

    (cols, rows)
}

/// Validate sprite sheet configuration and adjust if needed.
pub fn validate_and_adjust_config(mut config: SpriteSheetConfig) -> Result<SpriteSheetConfig> {
    // Adjust layout mode to grid dimensions
    match config.layout {
        LayoutMode::Vertical => {
            config.columns = 1;
            if let Some(count) = config.count {
                config.rows = count;
            }
        }
        LayoutMode::Horizontal => {
            config.rows = 1;
            if let Some(count) = config.count {
                config.columns = count;
            }
        }
        LayoutMode::Auto => {
            let count = config.count.unwrap_or(25);
            let aspect = config.thumbnail_width as f64 / config.thumbnail_height as f64;
            let (cols, rows) = calculate_optimal_grid(count, aspect);
            config.columns = cols;
            config.rows = rows;
            config.count = Some(count);
        }
        LayoutMode::Grid => {
            // Use specified grid dimensions
            if config.count.is_none() {
                config.count = Some(config.columns * config.rows);
            }
        }
    }

    // Validate after adjustments
    config.validate()?;

    Ok(config)
}
