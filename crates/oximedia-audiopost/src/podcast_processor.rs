//! Podcast audio processing: intro/outro detection, chapter marker generation,
//! host diarization hints, and episode-level quality metrics.
//!
//! This module provides a pipeline oriented toward spoken-word podcast production,
//! operating on mono or stereo PCM `f32` samples at any supported sample rate.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::error::{AudioPostError, AudioPostResult};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Basic signal helpers
// ---------------------------------------------------------------------------

/// Compute the RMS energy of a slice of samples.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Compute the zero-crossing rate of a slice of samples (crossings per sample).
fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    crossings as f32 / (samples.len() - 1) as f32
}

// ---------------------------------------------------------------------------
// Intro / Outro detection
// ---------------------------------------------------------------------------

/// Configuration for intro/outro segment detection.
#[derive(Debug, Clone)]
pub struct IntroOutroConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Analysis frame size in samples.
    pub frame_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// RMS silence threshold (linear).  Frames below this are candidates for silence.
    pub silence_rms_threshold: f32,
    /// Minimum duration of a silence region (seconds) to be considered an intro/outro boundary.
    pub min_silence_secs: f64,
    /// Maximum time from the beginning of the file to search for an intro end (seconds).
    pub max_intro_search_secs: f64,
    /// Maximum time from the end of the file to search for an outro start (seconds).
    pub max_outro_search_secs: f64,
}

impl IntroOutroConfig {
    /// Construct a default config for 48 kHz mono podcast audio.
    #[must_use]
    pub fn default_48k() -> Self {
        Self {
            sample_rate: 48000,
            frame_size: 2048,
            hop_size: 1024,
            silence_rms_threshold: 0.005,
            min_silence_secs: 0.5,
            max_intro_search_secs: 180.0,
            max_outro_search_secs: 180.0,
        }
    }

    /// Validate configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is out of range.
    pub fn validate(&self) -> AudioPostResult<()> {
        if self.sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(self.sample_rate));
        }
        if self.frame_size == 0 {
            return Err(AudioPostError::InvalidBufferSize(self.frame_size));
        }
        if self.hop_size == 0 || self.hop_size > self.frame_size {
            return Err(AudioPostError::InvalidBufferSize(self.hop_size));
        }
        if self.silence_rms_threshold < 0.0 || self.silence_rms_threshold > 1.0 {
            return Err(AudioPostError::Generic(format!(
                "silence_rms_threshold must be 0.0–1.0, got {}",
                self.silence_rms_threshold
            )));
        }
        if self.min_silence_secs <= 0.0 {
            return Err(AudioPostError::Generic(
                "min_silence_secs must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Detected intro and outro boundaries (in seconds from file start).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntroOutroBoundaries {
    /// End of the intro segment (seconds).  `None` if no clear intro boundary was found.
    pub intro_end_secs: Option<f64>,
    /// Start of the outro segment (seconds).  `None` if no clear outro boundary was found.
    pub outro_start_secs: Option<f64>,
    /// Total duration of the analyzed audio in seconds.
    pub total_duration_secs: f64,
}

/// Detect intro and outro boundaries using energy-based silence detection.
///
/// # Errors
///
/// Returns an error if the buffer is empty or the configuration is invalid.
pub fn detect_intro_outro(
    samples: &[f32],
    config: &IntroOutroConfig,
) -> AudioPostResult<IntroOutroBoundaries> {
    config.validate()?;
    if samples.is_empty() {
        return Err(AudioPostError::InvalidBufferSize(0));
    }

    let sr = config.sample_rate as f64;
    let total_secs = samples.len() as f64 / sr;
    let hop = config.hop_size;
    let win = config.frame_size;

    // Silence frames are frames whose RMS is below threshold.
    // We scan from the start for intro boundary and from the end for outro boundary.

    let min_silent_frames =
        ((config.min_silence_secs * sr) / hop as f64).ceil() as usize;
    let max_intro_frames =
        ((config.max_intro_search_secs * sr) / hop as f64).ceil() as usize;
    let total_frames = if samples.len() >= win {
        (samples.len() - win) / hop + 1
    } else {
        0
    };
    let max_outro_frames =
        ((config.max_outro_search_secs * sr) / hop as f64)
            .ceil()
            .min(total_frames as f64) as usize;

    // Build silence mask
    let silence_mask: Vec<bool> = (0..total_frames)
        .map(|i| {
            let start = i * hop;
            let end = (start + win).min(samples.len());
            rms(&samples[start..end]) < config.silence_rms_threshold
        })
        .collect();

    // Find intro end: the last frame of the first "min_silent_frames" silent run in
    // the intro search window, followed by speech content.
    let intro_end_secs = find_first_speech_after_silence(
        &silence_mask,
        0,
        max_intro_frames.min(total_frames),
        min_silent_frames,
        hop,
        sr,
    );

    // Find outro start: scan from the end backwards.
    let outro_start_secs = find_last_speech_before_silence(
        &silence_mask,
        total_frames.saturating_sub(max_outro_frames),
        total_frames,
        min_silent_frames,
        hop,
        sr,
    );

    Ok(IntroOutroBoundaries {
        intro_end_secs,
        outro_start_secs,
        total_duration_secs: total_secs,
    })
}

/// Within `silence_mask[start..end]`, find the first transition from silence to speech
/// and return the time of the first speech frame.
fn find_first_speech_after_silence(
    mask: &[bool],
    start: usize,
    end: usize,
    min_silent: usize,
    hop: usize,
    sr: f64,
) -> Option<f64> {
    let window = &mask[start..end.min(mask.len())];
    let mut silent_run = 0usize;
    let mut found_silence = false;

    for (i, &is_silent) in window.iter().enumerate() {
        if is_silent {
            silent_run += 1;
            if silent_run >= min_silent {
                found_silence = true;
            }
        } else {
            if found_silence {
                // First speech frame after a long silence
                return Some((start + i) as f64 * hop as f64 / sr);
            }
            silent_run = 0;
        }
    }
    None
}

/// Within `silence_mask[start..end]`, find the last transition from speech to silence
/// and return the time of that last speech frame (outro start).
fn find_last_speech_before_silence(
    mask: &[bool],
    start: usize,
    end: usize,
    min_silent: usize,
    hop: usize,
    sr: f64,
) -> Option<f64> {
    let window = &mask[start..end.min(mask.len())];
    let mut silent_run = 0usize;
    let mut last_speech_frame: Option<usize> = None;

    for (i, &is_silent) in window.iter().enumerate().rev() {
        if is_silent {
            silent_run += 1;
        } else {
            if silent_run >= min_silent && last_speech_frame.is_none() {
                last_speech_frame = Some(start + i + 1);
            }
            silent_run = 0;
        }
    }
    last_speech_frame.map(|f| f as f64 * hop as f64 / sr)
}

// ---------------------------------------------------------------------------
// Chapter marker generation
// ---------------------------------------------------------------------------

/// A single chapter marker for podcast navigation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterMarker {
    /// Chapter index (0-based).
    pub index: usize,
    /// Start time in seconds from the beginning of the episode.
    pub start_secs: f64,
    /// Optional title / label.
    pub title: Option<String>,
    /// Duration of the chapter in seconds (`None` if it is the last chapter).
    pub duration_secs: Option<f64>,
}

/// Configuration for automatic chapter marker generation.
#[derive(Debug, Clone)]
pub struct ChapterConfig {
    /// Minimum duration between chapter markers (seconds).
    pub min_chapter_duration_secs: f64,
    /// Energy drop threshold (ratio relative to local mean) that signals a section boundary.
    pub energy_drop_ratio: f32,
    /// Window (in frames) over which the local mean energy is computed.
    pub local_mean_window_frames: usize,
}

impl Default for ChapterConfig {
    fn default() -> Self {
        Self {
            min_chapter_duration_secs: 120.0,
            energy_drop_ratio: 0.3,
            local_mean_window_frames: 50,
        }
    }
}

/// Generate chapter markers by detecting significant energy drops in the audio.
///
/// # Errors
///
/// Returns an error if the buffer is empty.
pub fn generate_chapter_markers(
    samples: &[f32],
    sample_rate: u32,
    config: &ChapterConfig,
) -> AudioPostResult<Vec<ChapterMarker>> {
    if samples.is_empty() {
        return Err(AudioPostError::InvalidBufferSize(0));
    }
    if sample_rate == 0 {
        return Err(AudioPostError::InvalidSampleRate(sample_rate));
    }

    let sr = sample_rate as f64;
    let hop = 4096_usize;
    let win = 8192_usize;

    let num_frames = if samples.len() >= win {
        (samples.len() - win) / hop + 1
    } else {
        1
    };

    // Frame-level RMS
    let frame_rms: Vec<f32> = (0..num_frames)
        .map(|i| {
            let start = i * hop;
            let end = (start + win).min(samples.len());
            rms(&samples[start..end])
        })
        .collect();

    let min_gap_frames =
        ((config.min_chapter_duration_secs * sr) / hop as f64).ceil() as usize;
    let lm_half = config.local_mean_window_frames / 2;

    let mut markers: Vec<ChapterMarker> = Vec::new();
    markers.push(ChapterMarker {
        index: 0,
        start_secs: 0.0,
        title: None,
        duration_secs: None,
    });

    let mut frames_since_last = 0usize;

    for i in 1..frame_rms.len() {
        frames_since_last += 1;

        // Local mean energy around frame i
        let lo = i.saturating_sub(lm_half);
        let hi = (i + lm_half).min(frame_rms.len());
        let local_mean: f32 = frame_rms[lo..hi].iter().copied().sum::<f32>()
            / (hi - lo) as f32;

        let is_boundary = frame_rms[i] < local_mean * config.energy_drop_ratio
            && frames_since_last >= min_gap_frames;

        if is_boundary {
            let start_secs = i as f64 * hop as f64 / sr;
            markers.push(ChapterMarker {
                index: markers.len(),
                start_secs,
                title: None,
                duration_secs: None,
            });
            frames_since_last = 0;
        }
    }

    // Fill in duration for all but the last marker
    let total_secs = samples.len() as f64 / sr;
    let n = markers.len();
    for i in 0..n {
        let next_start = if i + 1 < n {
            markers[i + 1].start_secs
        } else {
            total_secs
        };
        markers[i].duration_secs = Some(next_start - markers[i].start_secs);
    }

    Ok(markers)
}

// ---------------------------------------------------------------------------
// Host diarization hints
// ---------------------------------------------------------------------------

/// A coarse diarization hint indicating which speaker (Host A or Host B) is likely
/// active in a given time window.  This is a heuristic, not a full diarization engine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SpeakerLabel {
    /// First speaker (assumed to have lower-energy or lower-ZCR profile).
    HostA,
    /// Second speaker (higher-energy or higher-ZCR profile).
    HostB,
    /// Both speakers talking simultaneously (crosstalk).
    Crosstalk,
    /// No active speech detected.
    Silence,
}

/// A time-labeled speaker hint segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationHint {
    /// Start of the segment in seconds.
    pub start_secs: f64,
    /// End of the segment in seconds.
    pub end_secs: f64,
    /// Estimated speaker label.
    pub speaker: SpeakerLabel,
}

/// Configuration for heuristic diarization.
#[derive(Debug, Clone)]
pub struct DiarizationConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Analysis frame size in samples.
    pub frame_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Silence RMS threshold.
    pub silence_threshold: f32,
    /// ZCR threshold above which a frame is considered "high-pitched" (Host B heuristic).
    pub zcr_split_threshold: f32,
}

impl DiarizationConfig {
    /// Default config for 48 kHz stereo (left = Host A, right = Host B).
    #[must_use]
    pub fn default_48k() -> Self {
        Self {
            sample_rate: 48000,
            frame_size: 4096,
            hop_size: 2048,
            silence_threshold: 0.004,
            zcr_split_threshold: 0.12,
        }
    }
}

/// Generate coarse diarization hints for a mono audio buffer using energy and ZCR features.
///
/// The heuristic: frames with ZCR above `zcr_split_threshold` are labeled HostB,
/// frames below are HostA; silent frames are Silence.  Adjacent frames with the same
/// label are merged.
///
/// # Errors
///
/// Returns an error if the buffer is empty or sample_rate is zero.
pub fn generate_diarization_hints(
    samples: &[f32],
    config: &DiarizationConfig,
) -> AudioPostResult<Vec<DiarizationHint>> {
    if samples.is_empty() {
        return Err(AudioPostError::InvalidBufferSize(0));
    }
    if config.sample_rate == 0 {
        return Err(AudioPostError::InvalidSampleRate(config.sample_rate));
    }

    let sr = config.sample_rate as f64;
    let hop = config.hop_size;
    let win = config.frame_size;

    let num_frames = if samples.len() >= win {
        (samples.len() - win) / hop + 1
    } else {
        1
    };

    let frame_labels: Vec<SpeakerLabel> = (0..num_frames)
        .map(|i| {
            let start = i * hop;
            let end = (start + win).min(samples.len());
            let frame = &samples[start..end];
            let r = rms(frame);
            if r < config.silence_threshold {
                SpeakerLabel::Silence
            } else {
                let zcr = zero_crossing_rate(frame);
                if zcr >= config.zcr_split_threshold {
                    SpeakerLabel::HostB
                } else {
                    SpeakerLabel::HostA
                }
            }
        })
        .collect();

    // Merge consecutive identical labels into segments
    let mut hints: Vec<DiarizationHint> = Vec::new();
    if frame_labels.is_empty() {
        return Ok(hints);
    }

    let mut seg_start_frame = 0usize;
    let mut current_label = &frame_labels[0];

    for (i, label) in frame_labels.iter().enumerate().skip(1) {
        if label != current_label {
            hints.push(DiarizationHint {
                start_secs: seg_start_frame as f64 * hop as f64 / sr,
                end_secs: i as f64 * hop as f64 / sr,
                speaker: current_label.clone(),
            });
            seg_start_frame = i;
            current_label = label;
        }
    }
    // Last segment
    hints.push(DiarizationHint {
        start_secs: seg_start_frame as f64 * hop as f64 / sr,
        end_secs: samples.len() as f64 / sr,
        speaker: current_label.clone(),
    });

    Ok(hints)
}

// ---------------------------------------------------------------------------
// Episode quality metrics
// ---------------------------------------------------------------------------

/// Summary quality metrics for a podcast episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeQualityMetrics {
    /// Mean RMS energy of non-silent frames (linear).
    pub mean_active_rms: f32,
    /// Peak sample value encountered.
    pub peak_linear: f32,
    /// Estimated peak in dBFS.
    pub peak_dbfs: f32,
    /// Fraction of frames classified as silence (0.0–1.0).
    pub silence_ratio: f32,
    /// Mean zero-crossing rate of non-silent frames.
    pub mean_zcr: f32,
    /// Dynamic range estimate: difference between 95th and 5th percentile RMS (linear).
    pub dynamic_range_rms: f32,
}

/// Compute episode-level quality metrics from a mono PCM buffer.
///
/// # Errors
///
/// Returns an error if the buffer is empty or sample rate is zero.
pub fn compute_episode_metrics(
    samples: &[f32],
    sample_rate: u32,
    silence_threshold: f32,
) -> AudioPostResult<EpisodeQualityMetrics> {
    if samples.is_empty() {
        return Err(AudioPostError::InvalidBufferSize(0));
    }
    if sample_rate == 0 {
        return Err(AudioPostError::InvalidSampleRate(sample_rate));
    }

    let frame_size = 4096_usize;
    let hop = 2048_usize;

    let num_frames = if samples.len() >= frame_size {
        (samples.len() - frame_size) / hop + 1
    } else {
        1
    };

    let mut frame_rms_vals: Vec<f32> = Vec::with_capacity(num_frames);
    let mut frame_zcr_vals: Vec<f32> = Vec::new();
    let mut peak_linear: f32 = 0.0;

    for i in 0..num_frames {
        let start = i * hop;
        let end = (start + frame_size).min(samples.len());
        let frame = &samples[start..end];

        let r = rms(frame);
        frame_rms_vals.push(r);

        let local_peak = frame.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        if local_peak > peak_linear {
            peak_linear = local_peak;
        }

        if r >= silence_threshold {
            frame_zcr_vals.push(zero_crossing_rate(frame));
        }
    }

    let silence_frames = frame_rms_vals
        .iter()
        .filter(|&&r| r < silence_threshold)
        .count();
    let silence_ratio = silence_frames as f32 / num_frames as f32;

    let active_rms_vals: Vec<f32> = frame_rms_vals
        .iter()
        .copied()
        .filter(|&r| r >= silence_threshold)
        .collect();

    let mean_active_rms = if active_rms_vals.is_empty() {
        0.0
    } else {
        active_rms_vals.iter().sum::<f32>() / active_rms_vals.len() as f32
    };

    let mean_zcr = if frame_zcr_vals.is_empty() {
        0.0
    } else {
        frame_zcr_vals.iter().sum::<f32>() / frame_zcr_vals.len() as f32
    };

    // Dynamic range: 95th – 5th percentile of frame RMS
    let mut sorted_rms = frame_rms_vals.clone();
    sorted_rms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p5_idx = (sorted_rms.len() as f32 * 0.05) as usize;
    let p95_idx = ((sorted_rms.len() as f32 * 0.95) as usize).min(sorted_rms.len() - 1);
    let dynamic_range_rms = sorted_rms[p95_idx] - sorted_rms[p5_idx];

    let peak_dbfs = if peak_linear > 0.0 {
        20.0 * peak_linear.log10()
    } else {
        f32::NEG_INFINITY
    };

    Ok(EpisodeQualityMetrics {
        mean_active_rms,
        peak_linear,
        peak_dbfs,
        silence_ratio,
        mean_zcr,
        dynamic_range_rms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a simple sine-wave segment, with optional leading silence.
    fn make_audio(sr: usize, silence_secs: f64, speech_secs: f64, amplitude: f32) -> Vec<f32> {
        let silence_samples = (silence_secs * sr as f64) as usize;
        let speech_samples = (speech_secs * sr as f64) as usize;
        let mut buf = vec![0.0f32; silence_samples + speech_samples];
        for i in 0..speech_samples {
            buf[silence_samples + i] =
                amplitude * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sr as f32).sin();
        }
        buf
    }

    #[test]
    fn test_intro_outro_config_validation() {
        let cfg = IntroOutroConfig::default_48k();
        assert!(cfg.validate().is_ok());

        let bad = IntroOutroConfig {
            sample_rate: 0,
            ..IntroOutroConfig::default_48k()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_detect_intro_empty_buffer_error() {
        let cfg = IntroOutroConfig::default_48k();
        assert!(detect_intro_outro(&[], &cfg).is_err());
    }

    #[test]
    fn test_detect_intro_total_duration() {
        let sr = 48000_usize;
        let audio = make_audio(sr, 2.0, 30.0, 0.3);
        let cfg = IntroOutroConfig::default_48k();
        let result = detect_intro_outro(&audio, &cfg).expect("ok");
        let expected = audio.len() as f64 / sr as f64;
        assert!((result.total_duration_secs - expected).abs() < 0.1);
    }

    #[test]
    fn test_detect_intro_finds_boundary() {
        // 2 s silence → 30 s speech → 2 s silence
        let sr = 48000_usize;
        let silence = vec![0.0f32; sr * 2];
        let speech = make_audio(sr, 0.0, 30.0, 0.3);
        let silence2 = vec![0.0f32; sr * 2];
        let audio: Vec<f32> = silence
            .iter()
            .chain(speech.iter())
            .chain(silence2.iter())
            .copied()
            .collect();

        let mut cfg = IntroOutroConfig::default_48k();
        cfg.min_silence_secs = 0.5;
        let result = detect_intro_outro(&audio, &cfg).expect("ok");

        // Intro should end somewhere after the 2 s silence and before 5 s
        if let Some(intro_end) = result.intro_end_secs {
            assert!(
                intro_end > 1.5 && intro_end < 5.0,
                "intro_end={intro_end}"
            );
        }
    }

    #[test]
    fn test_generate_chapter_markers_empty_error() {
        let cfg = ChapterConfig::default();
        assert!(generate_chapter_markers(&[], 48000, &cfg).is_err());
    }

    #[test]
    fn test_generate_chapter_markers_returns_first() {
        let sr = 48000_u32;
        let audio = make_audio(sr as usize, 0.0, 10.0, 0.3);
        let cfg = ChapterConfig {
            min_chapter_duration_secs: 5.0,
            ..ChapterConfig::default()
        };
        let markers = generate_chapter_markers(&audio, sr, &cfg).expect("ok");
        assert!(!markers.is_empty());
        assert_eq!(markers[0].start_secs, 0.0);
        assert_eq!(markers[0].index, 0);
    }

    #[test]
    fn test_generate_chapter_markers_duration_filled() {
        let sr = 48000_u32;
        let audio = make_audio(sr as usize, 0.0, 10.0, 0.3);
        let cfg = ChapterConfig::default();
        let markers = generate_chapter_markers(&audio, sr, &cfg).expect("ok");
        // Every marker should have a duration
        for m in &markers {
            assert!(
                m.duration_secs.is_some(),
                "marker {} missing duration",
                m.index
            );
        }
    }

    #[test]
    fn test_diarization_hints_empty_error() {
        let cfg = DiarizationConfig::default_48k();
        assert!(generate_diarization_hints(&[], &cfg).is_err());
    }

    #[test]
    fn test_diarization_hints_silence_label() {
        let cfg = DiarizationConfig {
            silence_threshold: 0.5, // very high → all frames are silence
            ..DiarizationConfig::default_48k()
        };
        let audio = vec![0.01f32; 96000]; // 2 s of near-silence
        let hints = generate_diarization_hints(&audio, &cfg).expect("ok");
        for h in &hints {
            assert_eq!(h.speaker, SpeakerLabel::Silence, "expected silence");
        }
    }

    #[test]
    fn test_diarization_hints_segments_cover_full_duration() {
        let sr = 48000_u32;
        let audio = make_audio(sr as usize, 0.0, 5.0, 0.3);
        let cfg = DiarizationConfig::default_48k();
        let hints = generate_diarization_hints(&audio, &cfg).expect("ok");

        // Segments should be contiguous and cover the full file
        let expected_end = audio.len() as f64 / sr as f64;
        let last_end = hints.last().map(|h| h.end_secs).unwrap_or(0.0);
        assert!(
            (last_end - expected_end).abs() < 0.1,
            "last end={last_end}, expected={expected_end}"
        );
    }

    #[test]
    fn test_episode_metrics_empty_error() {
        assert!(compute_episode_metrics(&[], 48000, 0.005).is_err());
    }

    #[test]
    fn test_episode_metrics_silent_audio() {
        let audio = vec![0.0f32; 48000 * 5]; // 5 s silence
        let m = compute_episode_metrics(&audio, 48000, 0.005).expect("ok");
        assert_eq!(m.mean_active_rms, 0.0);
        assert_eq!(m.peak_linear, 0.0);
        assert_eq!(m.silence_ratio, 1.0);
    }

    #[test]
    fn test_episode_metrics_sine_wave() {
        let sr = 48000_usize;
        let duration = 5.0_f64;
        let mut audio = vec![0.0f32; (sr as f64 * duration) as usize];
        for (i, s) in audio.iter_mut().enumerate() {
            *s = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin();
        }
        let m = compute_episode_metrics(&audio, sr as u32, 0.005).expect("ok");
        // A 0.5-amplitude sine has RMS ≈ 0.5 / sqrt(2) ≈ 0.354
        assert!(
            m.mean_active_rms > 0.3 && m.mean_active_rms < 0.4,
            "RMS={}", m.mean_active_rms
        );
        assert!(m.peak_linear > 0.48 && m.peak_linear <= 0.5001);
        assert!(m.silence_ratio < 0.1);
    }

    #[test]
    fn test_rms_helper() {
        let samples = vec![1.0f32, -1.0, 1.0, -1.0];
        assert!((rms(&samples) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_crossing_rate_helper() {
        // Alternating ±1 → every pair crosses → rate = 1.0
        let samples = vec![1.0f32, -1.0, 1.0, -1.0, 1.0];
        let zcr = zero_crossing_rate(&samples);
        assert!((zcr - 1.0).abs() < 1e-6, "zcr={zcr}");
    }
}
