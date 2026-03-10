use super::*;

/// Calculate thumbnail timestamps based on sampling strategy.
pub(super) fn calculate_timestamps(config: &SpriteSheetConfig, count: usize) -> Result<Vec<f64>> {
    // Simulated video duration (would come from a real demuxer in production).
    let video_duration = 120.0f64; // 2-minute synthetic video

    let timestamps = match config.strategy {
        SamplingStrategy::Uniform => {
            calculate_uniform_timestamps(video_duration, count, config.interval)?
        }
        SamplingStrategy::SceneBased => calculate_scene_based_timestamps(video_duration, count)?,
        SamplingStrategy::KeyframeOnly => calculate_keyframe_timestamps(video_duration, count)?,
        SamplingStrategy::Smart => {
            calculate_smart_timestamps(video_duration, count, config.interval)?
        }
    };

    Ok(timestamps)
}

/// Calculate scene-based timestamps by simulating scene detection.
///
/// Generates candidate timestamps at fine granularity, then selects those
/// where a hypothetical scene-change score (based on a deterministic hash of
/// the timestamp) exceeds a threshold — approximating real scene detection.
fn calculate_scene_based_timestamps(duration: f64, count: usize) -> Result<Vec<f64>> {
    if count == 0 {
        return Err(anyhow!("Thumbnail count must be greater than zero"));
    }

    // Probe at 2-second intervals to find "scene changes"
    let probe_interval = 2.0;
    let num_probes = (duration / probe_interval) as usize;
    let scene_change_threshold = 0.35;

    let mut scene_timestamps: Vec<f64> = Vec::new();
    let mut prev_score: f64 = 0.0;

    for i in 0..num_probes {
        let t = i as f64 * probe_interval;
        // Deterministic pseudo-scene-score derived from the timestamp
        let score = {
            let bits = (t * 1000.0) as u64;
            let h = bits
                .wrapping_mul(0x9e3779b97f4a7c15)
                .wrapping_add(0x6c62272e07bb0142);
            (h & 0xFFFF) as f64 / 65535.0
        };
        // Detect change relative to previous probe
        let diff = (score - prev_score).abs();
        if diff > scene_change_threshold || scene_timestamps.is_empty() {
            scene_timestamps.push(t);
        }
        prev_score = score;
    }

    // If we found fewer scenes than requested, pad with uniform timestamps
    if scene_timestamps.len() < count {
        let uniform = calculate_uniform_timestamps(duration, count, None)?;
        for t in &uniform {
            if !scene_timestamps.contains(t) {
                scene_timestamps.push(*t);
            }
            if scene_timestamps.len() >= count {
                break;
            }
        }
    }

    // Sort and trim to requested count
    scene_timestamps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    scene_timestamps.truncate(count);

    Ok(scene_timestamps)
}

/// Calculate keyframe-only timestamps.
///
/// Simulates I-frame positions by placing keyframes at GOP boundaries.
/// A typical GOP is 30–90 frames; here we use a deterministic 2-second GOP.
fn calculate_keyframe_timestamps(duration: f64, count: usize) -> Result<Vec<f64>> {
    if count == 0 {
        return Err(anyhow!("Thumbnail count must be greater than zero"));
    }

    // Simulate a 2-second GOP interval (30 fps × GOP size 60 frames)
    let gop_duration = 2.0;
    let mut keyframes: Vec<f64> = Vec::new();
    let mut t = 0.0;
    while t <= duration {
        keyframes.push(t);
        t += gop_duration;
    }

    // Select evenly-spaced keyframes from available list
    if keyframes.len() <= count {
        keyframes.truncate(count);
        return Ok(keyframes);
    }

    let step = keyframes.len() as f64 / count as f64;
    let selected: Vec<f64> = (0..count)
        .map(|i| keyframes[(i as f64 * step) as usize])
        .collect();

    Ok(selected)
}

/// Calculate smart timestamps combining scene detection and quality analysis.
///
/// Merges uniform sampling with scene-based candidates, then de-duplicates
/// and trims to the requested count, preferring well-spaced coverage.
fn calculate_smart_timestamps(
    duration: f64,
    count: usize,
    interval: Option<f64>,
) -> Result<Vec<f64>> {
    if count == 0 {
        return Err(anyhow!("Thumbnail count must be greater than zero"));
    }

    // Gather candidates from uniform and scene-based strategies
    let uniform = calculate_uniform_timestamps(duration, count, interval)?;
    let scene = calculate_scene_based_timestamps(duration, count)?;

    let mut combined: Vec<f64> = uniform;
    for t in scene {
        combined.push(t);
    }
    combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    combined.dedup_by(|a, b| (*a - *b).abs() < 1.0); // collapse timestamps within 1 second

    // Trim or pad to exactly `count` entries
    if combined.len() > count {
        // Prefer evenly-spaced subset
        let step = combined.len() as f64 / count as f64;
        let selected: Vec<f64> = (0..count)
            .map(|i| combined[(i as f64 * step) as usize])
            .collect();
        Ok(selected)
    } else if combined.len() < count {
        // Pad with uniform timestamps
        let extra = calculate_uniform_timestamps(duration, count - combined.len(), None)?;
        combined.extend(extra);
        combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        combined.truncate(count);
        Ok(combined)
    } else {
        Ok(combined)
    }
}

/// Calculate uniform timestamps at fixed intervals.
fn calculate_uniform_timestamps(
    duration: f64,
    count: usize,
    interval: Option<f64>,
) -> Result<Vec<f64>> {
    if count == 0 {
        return Err(anyhow!("Thumbnail count must be greater than zero"));
    }

    let mut timestamps = Vec::new();

    if let Some(interval_secs) = interval {
        // Use fixed interval
        let mut time = 0.0;
        let mut i = 0;

        while time < duration && i < count {
            timestamps.push(time);
            time += interval_secs;
            i += 1;
        }
    } else {
        // Distribute evenly across duration
        if count == 1 {
            // Single thumbnail at midpoint
            timestamps.push(duration / 2.0);
        } else {
            // Multiple thumbnails distributed evenly
            let step = duration / (count - 1) as f64;

            for i in 0..count {
                let time = i as f64 * step;
                timestamps.push(time.min(duration));
            }
        }
    }

    Ok(timestamps)
}

/// Calculate thumbnail position in sprite sheet.
pub(super) fn calculate_thumbnail_position(
    index: usize,
    columns: usize,
    thumb_width: u32,
    thumb_height: u32,
    spacing: u32,
    margin: u32,
) -> (u32, u32) {
    let col = index % columns;
    let row = index / columns;

    let x = margin + col as u32 * (thumb_width + spacing);
    let y = margin + row as u32 * (thumb_height + spacing);

    (x, y)
}
