//! A/V sync correction.
//!
//! This module provides functions to fix audio/video synchronization issues.

use crate::Result;

/// Fix A/V sync by adjusting timestamps.
pub fn fix_sync(
    audio_timestamps: &mut [i64],
    _video_timestamps: &[i64],
    offset: i64,
) -> Result<()> {
    for ts in audio_timestamps.iter_mut() {
        *ts -= offset;
    }
    Ok(())
}

/// Fix drift by stretching/compressing audio.
pub fn fix_drift(audio_timestamps: &mut [i64], video_timestamps: &[i64]) -> Result<()> {
    let min_len = audio_timestamps.len().min(video_timestamps.len());

    if min_len < 2 {
        return Ok(());
    }

    let audio_duration = audio_timestamps[min_len - 1] - audio_timestamps[0];
    let video_duration = video_timestamps[min_len - 1] - video_timestamps[0];

    if audio_duration == 0 {
        return Ok(());
    }

    let ratio = video_duration as f64 / audio_duration as f64;

    // Stretch/compress audio timestamps
    let start_offset = audio_timestamps[0];
    for ts in audio_timestamps.iter_mut() {
        let relative = *ts - start_offset;
        *ts = start_offset + (relative as f64 * ratio) as i64;
    }

    Ok(())
}

/// Align audio and video to start at the same time.
pub fn align_start_times(audio_timestamps: &mut [i64], video_timestamps: &mut [i64]) -> Result<()> {
    if audio_timestamps.is_empty() || video_timestamps.is_empty() {
        return Ok(());
    }

    let audio_start = audio_timestamps[0];
    let video_start = video_timestamps[0];

    let target = audio_start.min(video_start);

    for ts in audio_timestamps.iter_mut() {
        *ts -= audio_start - target;
    }

    for ts in video_timestamps.iter_mut() {
        *ts -= video_start - target;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_sync() {
        let mut audio = vec![100, 200, 300];
        let video = vec![0, 100, 200];

        fix_sync(&mut audio, &video, 100).expect("sync fix should succeed");

        assert_eq!(audio[0], 0);
        assert_eq!(audio[1], 100);
        assert_eq!(audio[2], 200);
    }

    #[test]
    fn test_align_start_times() {
        let mut audio = vec![100, 200, 300];
        let mut video = vec![0, 100, 200];

        align_start_times(&mut audio, &mut video).expect("alignment should succeed");

        assert_eq!(audio[0], video[0]);
    }
}
