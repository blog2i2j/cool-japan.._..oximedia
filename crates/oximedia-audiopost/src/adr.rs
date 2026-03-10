#![allow(dead_code)]
//! Automated Dialogue Replacement (ADR) system.
//!
//! Provides comprehensive ADR session management, recording, and synchronization.

use crate::error::{AudioPostError, AudioPostResult};
use crate::timecode::Timecode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// ADR session containing multiple cues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdrSession {
    /// Session ID
    pub id: Uuid,
    /// Session name
    pub name: String,
    /// Sample rate
    pub sample_rate: u32,
    /// Cues in this session
    cues: HashMap<usize, AdrCue>,
    /// Next cue ID
    next_cue_id: usize,
    /// Session notes
    pub notes: String,
}

impl AdrSession {
    /// Create a new ADR session
    #[must_use]
    pub fn new(name: &str, sample_rate: u32) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            sample_rate,
            cues: HashMap::new(),
            next_cue_id: 1,
            notes: String::new(),
        }
    }

    /// Add a cue to the session
    pub fn add_cue(&mut self, cue: AdrCue) -> usize {
        let id = self.next_cue_id;
        self.cues.insert(id, cue);
        self.next_cue_id += 1;
        id
    }

    /// Get a cue by ID
    ///
    /// # Errors
    ///
    /// Returns an error if the cue is not found
    pub fn get_cue(&self, id: usize) -> AudioPostResult<&AdrCue> {
        self.cues.get(&id).ok_or(AudioPostError::CueNotFound(id))
    }

    /// Get a mutable reference to a cue
    ///
    /// # Errors
    ///
    /// Returns an error if the cue is not found
    pub fn get_cue_mut(&mut self, id: usize) -> AudioPostResult<&mut AdrCue> {
        self.cues
            .get_mut(&id)
            .ok_or(AudioPostError::CueNotFound(id))
    }

    /// Remove a cue from the session
    ///
    /// # Errors
    ///
    /// Returns an error if the cue is not found
    pub fn remove_cue(&mut self, id: usize) -> AudioPostResult<AdrCue> {
        self.cues.remove(&id).ok_or(AudioPostError::CueNotFound(id))
    }

    /// Get all cues sorted by timecode
    #[must_use]
    pub fn get_cues_sorted(&self) -> Vec<(usize, &AdrCue)> {
        let mut cues: Vec<_> = self.cues.iter().map(|(id, cue)| (*id, cue)).collect();
        cues.sort_by(|a, b| {
            a.1.start_timecode
                .partial_cmp(&b.1.start_timecode)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        cues
    }

    /// Get total number of cues
    #[must_use]
    pub fn cue_count(&self) -> usize {
        self.cues.len()
    }
}

/// A single ADR cue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdrCue {
    /// Cue description/dialogue
    pub description: String,
    /// Start timecode
    pub start_timecode: Timecode,
    /// End timecode
    pub end_timecode: Timecode,
    /// Original dialogue reference path
    pub reference_audio: Option<String>,
    /// Takes for this cue
    takes: HashMap<usize, AdrTake>,
    /// Next take ID
    next_take_id: usize,
    /// Selected take ID
    pub selected_take: Option<usize>,
    /// Director notes
    pub director_notes: String,
    /// Actor name
    pub actor: String,
    /// Character name
    pub character: String,
}

impl AdrCue {
    /// Create a new ADR cue
    #[must_use]
    pub fn new(description: &str, start_timecode: Timecode, end_timecode: Timecode) -> Self {
        Self {
            description: description.to_string(),
            start_timecode,
            end_timecode,
            reference_audio: None,
            takes: HashMap::new(),
            next_take_id: 1,
            selected_take: None,
            director_notes: String::new(),
            actor: String::new(),
            character: String::new(),
        }
    }

    /// Get duration in seconds
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end_timecode.to_seconds() - self.start_timecode.to_seconds()
    }

    /// Add a take to this cue
    pub fn add_take(&mut self, take: AdrTake) -> usize {
        let id = self.next_take_id;
        self.takes.insert(id, take);
        self.next_take_id += 1;

        // Auto-select first take
        if self.selected_take.is_none() {
            self.selected_take = Some(id);
        }

        id
    }

    /// Get a take by ID
    ///
    /// # Errors
    ///
    /// Returns an error if the take is not found
    pub fn get_take(&self, id: usize) -> AudioPostResult<&AdrTake> {
        self.takes.get(&id).ok_or(AudioPostError::TakeNotFound(id))
    }

    /// Get the selected take
    ///
    /// # Errors
    ///
    /// Returns an error if no take is selected
    pub fn get_selected_take(&self) -> AudioPostResult<&AdrTake> {
        let take_id = self
            .selected_take
            .ok_or_else(|| AudioPostError::Generic("No take selected".to_string()))?;
        self.get_take(take_id)
    }

    /// Get all takes
    #[must_use]
    pub fn get_takes(&self) -> Vec<(usize, &AdrTake)> {
        self.takes.iter().map(|(id, take)| (*id, take)).collect()
    }

    /// Delete a take
    ///
    /// # Errors
    ///
    /// Returns an error if the take is not found
    pub fn delete_take(&mut self, id: usize) -> AudioPostResult<()> {
        self.takes
            .remove(&id)
            .ok_or(AudioPostError::TakeNotFound(id))?;

        // Clear selection if deleted take was selected
        if self.selected_take == Some(id) {
            self.selected_take = None;
        }

        Ok(())
    }
}

/// A recorded take for an ADR cue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdrTake {
    /// Take number
    pub number: u32,
    /// Recording timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Audio file path
    pub audio_path: String,
    /// Rating (1-5 stars)
    pub rating: Option<u8>,
    /// Performance notes
    pub notes: String,
    /// Sync offset in samples
    pub sync_offset: i64,
    /// Approved flag
    pub approved: bool,
    /// Slate information
    pub slate: SlateInfo,
}

impl AdrTake {
    /// Create a new ADR take
    #[must_use]
    pub fn new(number: u32, audio_path: &str) -> Self {
        Self {
            number,
            timestamp: chrono::Utc::now(),
            audio_path: audio_path.to_string(),
            rating: None,
            notes: String::new(),
            sync_offset: 0,
            approved: false,
            slate: SlateInfo::default(),
        }
    }

    /// Set rating (1-5 stars)
    ///
    /// # Errors
    ///
    /// Returns an error if rating is not between 1 and 5
    pub fn set_rating(&mut self, rating: u8) -> AudioPostResult<()> {
        if !(1..=5).contains(&rating) {
            return Err(AudioPostError::Generic(
                "Rating must be between 1 and 5".to_string(),
            ));
        }
        self.rating = Some(rating);
        Ok(())
    }
}

/// Slate information for a take
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SlateInfo {
    /// Scene number
    pub scene: String,
    /// Take number
    pub take: String,
    /// Production name
    pub production: String,
    /// Director name
    pub director: String,
    /// Date
    pub date: String,
}

/// ADR recording settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdrRecordingSettings {
    /// Pre-roll duration in seconds
    pub pre_roll: f64,
    /// Post-roll duration in seconds
    pub post_roll: f64,
    /// Enable loop recording
    pub loop_recording: bool,
    /// Number of beeps in countdown
    pub beep_count: u8,
    /// Beep frequency in Hz
    pub beep_frequency: f32,
    /// Beep duration in seconds
    pub beep_duration: f64,
    /// Auto-punch in/out
    pub auto_punch: bool,
}

impl Default for AdrRecordingSettings {
    fn default() -> Self {
        Self {
            pre_roll: 2.0,
            post_roll: 1.0,
            loop_recording: false,
            beep_count: 3,
            beep_frequency: 1000.0,
            beep_duration: 0.1,
            auto_punch: true,
        }
    }
}

/// ADR sync analyzer for lip-sync analysis
#[derive(Debug)]
pub struct AdrSyncAnalyzer {
    sample_rate: u32,
}

impl AdrSyncAnalyzer {
    /// Create a new sync analyzer
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Analyze sync drift between reference and recorded audio using normalized cross-correlation.
    ///
    /// Computes the normalized cross-correlation between `reference` and `recorded` audio,
    /// finds the lag at peak correlation, and returns the time offset in samples.
    /// A positive result means the recorded audio is delayed relative to the reference.
    ///
    /// Returns drift in samples.
    pub fn analyze_drift(&self, reference: &[f32], recorded: &[f32]) -> i64 {
        if reference.is_empty() || recorded.is_empty() {
            return 0;
        }

        let ref_len = reference.len();
        let rec_len = recorded.len();

        // Search window: allow up to 500ms of drift in either direction.
        let max_lag = (self.sample_rate as usize / 2).min(ref_len.max(rec_len));

        // Precompute RMS of reference and recorded signals for normalization.
        let ref_rms: f64 = {
            let sum_sq: f64 = reference.iter().map(|&x| (x as f64) * (x as f64)).sum();
            (sum_sq / ref_len as f64).sqrt()
        };
        let rec_rms: f64 = {
            let sum_sq: f64 = recorded.iter().map(|&x| (x as f64) * (x as f64)).sum();
            (sum_sq / rec_len as f64).sqrt()
        };

        // Avoid division by zero for silent signals.
        if ref_rms < 1e-10 || rec_rms < 1e-10 {
            return 0;
        }

        let norm = ref_rms * rec_rms * (ref_len.min(rec_len) as f64);

        // Evaluate cross-correlation at each lag in [-max_lag, +max_lag].
        // lag > 0: recorded is delayed (needs to be shifted earlier).
        // lag < 0: recorded is early (needs to be shifted later).
        let mut best_lag: i64 = 0;
        let mut best_corr: f64 = f64::NEG_INFINITY;

        let imax_lag = max_lag as i64;
        for lag in -imax_lag..=imax_lag {
            let mut corr: f64 = 0.0;
            let overlap_start = 0usize;
            let overlap_end = ref_len.min(rec_len);

            for i in overlap_start..overlap_end {
                let j = i as i64 - lag;
                if j < 0 || j as usize >= rec_len {
                    continue;
                }
                corr += (reference[i] as f64) * (recorded[j as usize] as f64);
            }

            let normalized = corr / norm;
            if normalized > best_corr {
                best_corr = normalized;
                best_lag = lag;
            }
        }

        best_lag
    }

    /// Calculate lip-sync confidence score (0.0 to 1.0).
    ///
    /// Estimates alignment between mouth-movement activity (derived from the raw video frame
    /// byte energy acting as a luminance-difference proxy) and audio energy peaks using
    /// short-window zero-crossing rate (ZCR) and RMS analysis.
    pub fn calculate_sync_confidence(&self, video_frames: &[u8], audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        // Window size: ~10 ms at the given sample rate.
        let window_samples = (self.sample_rate as usize / 100).max(1);

        // Build per-window audio activity vector using RMS and ZCR.
        let audio_windows: Vec<f32> = audio
            .chunks(window_samples)
            .map(|w| {
                if w.is_empty() {
                    return 0.0_f32;
                }
                // RMS energy.
                let rms = (w.iter().map(|&s| s * s).sum::<f32>() / w.len() as f32).sqrt();
                // Zero-crossing rate.
                let zcr = w
                    .windows(2)
                    .filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0))
                    .count() as f32
                    / w.len() as f32;
                // Speech activity: high ZCR with measurable energy indicates voiced speech.
                (rms * (1.0 + zcr)).min(1.0)
            })
            .collect();

        if audio_windows.is_empty() {
            return 0.0;
        }

        // Build per-window video activity from raw byte differences as a proxy
        // for luminance change (mouth movement).
        let video_window_bytes = video_frames.len().max(1) / audio_windows.len();
        let video_activity: Vec<f32> = if video_frames.is_empty() || video_window_bytes == 0 {
            vec![0.0; audio_windows.len()]
        } else {
            video_frames
                .chunks(video_window_bytes)
                .take(audio_windows.len())
                .map(|chunk| {
                    if chunk.len() < 2 {
                        return 0.0_f32;
                    }
                    let diff_sum: f32 = chunk
                        .windows(2)
                        .map(|p| (p[0] as f32 - p[1] as f32).abs())
                        .sum();
                    (diff_sum / chunk.len() as f32 / 255.0).min(1.0)
                })
                .collect()
        };

        let n = audio_windows.len().min(video_activity.len());
        if n == 0 {
            return 0.0;
        }

        // Compute Pearson correlation between audio and video activity vectors.
        let a_mean: f32 = audio_windows[..n].iter().sum::<f32>() / n as f32;
        let v_mean: f32 = video_activity[..n].iter().sum::<f32>() / n as f32;

        let mut cov = 0.0_f32;
        let mut a_var = 0.0_f32;
        let mut v_var = 0.0_f32;
        for i in 0..n {
            let a_d = audio_windows[i] - a_mean;
            let v_d = video_activity[i] - v_mean;
            cov += a_d * v_d;
            a_var += a_d * a_d;
            v_var += v_d * v_d;
        }

        let denom = (a_var * v_var).sqrt();
        if denom < 1e-10 {
            // No variation – assume perfect sync.
            return 1.0;
        }

        // Map Pearson r ∈ [-1, 1] to a confidence score ∈ [0, 1].
        ((cov / denom) + 1.0) / 2.0
    }

    /// Suggest sync offset correction
    pub fn suggest_sync_correction(&self, drift: i64) -> i64 {
        drift
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adr_session_creation() {
        let session = AdrSession::new("Scene 42", 48000);
        assert_eq!(session.name, "Scene 42");
        assert_eq!(session.sample_rate, 48000);
        assert_eq!(session.cue_count(), 0);
    }

    #[test]
    fn test_add_cue() {
        let mut session = AdrSession::new("Scene 42", 48000);
        let cue = AdrCue::new(
            "Test dialogue",
            Timecode::from_frames(1000, 24.0),
            Timecode::from_frames(1100, 24.0),
        );
        let id = session.add_cue(cue);
        assert_eq!(id, 1);
        assert_eq!(session.cue_count(), 1);
    }

    #[test]
    fn test_get_cue() {
        let mut session = AdrSession::new("Scene 42", 48000);
        let cue = AdrCue::new(
            "Test dialogue",
            Timecode::from_frames(1000, 24.0),
            Timecode::from_frames(1100, 24.0),
        );
        let id = session.add_cue(cue);
        let retrieved_cue = session.get_cue(id).expect("get_cue should succeed");
        assert_eq!(retrieved_cue.description, "Test dialogue");
    }

    #[test]
    fn test_remove_cue() {
        let mut session = AdrSession::new("Scene 42", 48000);
        let cue = AdrCue::new(
            "Test dialogue",
            Timecode::from_frames(1000, 24.0),
            Timecode::from_frames(1100, 24.0),
        );
        let id = session.add_cue(cue);
        assert!(session.remove_cue(id).is_ok());
        assert_eq!(session.cue_count(), 0);
    }

    #[test]
    fn test_cue_duration() {
        let cue = AdrCue::new(
            "Test dialogue",
            Timecode::from_frames(1000, 24.0),
            Timecode::from_frames(1100, 24.0),
        );
        let duration = cue.duration();
        assert!((duration - (100.0 / 24.0)).abs() < 1e-6);
    }

    #[test]
    fn test_add_take() {
        let mut cue = AdrCue::new(
            "Test dialogue",
            Timecode::from_frames(1000, 24.0),
            Timecode::from_frames(1100, 24.0),
        );
        let take = AdrTake::new(1, "/path/to/audio.wav");
        let id = cue.add_take(take);
        assert_eq!(id, 1);
        assert_eq!(cue.selected_take, Some(1));
    }

    #[test]
    fn test_take_rating() {
        let mut take = AdrTake::new(1, "/path/to/audio.wav");
        assert!(take.set_rating(5).is_ok());
        assert_eq!(take.rating, Some(5));
        assert!(take.set_rating(0).is_err());
        assert!(take.set_rating(6).is_err());
    }

    #[test]
    fn test_delete_take() {
        let mut cue = AdrCue::new(
            "Test dialogue",
            Timecode::from_frames(1000, 24.0),
            Timecode::from_frames(1100, 24.0),
        );
        let take = AdrTake::new(1, "/path/to/audio.wav");
        let id = cue.add_take(take);
        assert!(cue.delete_take(id).is_ok());
        assert_eq!(cue.selected_take, None);
    }

    #[test]
    fn test_recording_settings_default() {
        let settings = AdrRecordingSettings::default();
        assert_eq!(settings.pre_roll, 2.0);
        assert_eq!(settings.post_roll, 1.0);
        assert_eq!(settings.beep_count, 3);
    }

    #[test]
    fn test_sync_analyzer() {
        let analyzer = AdrSyncAnalyzer::new(48000);
        let reference = vec![0.0_f32; 1000];
        let recorded = vec![0.0_f32; 1000];
        let drift = analyzer.analyze_drift(&reference, &recorded);
        assert_eq!(drift, 0);
    }

    #[test]
    fn test_sync_confidence() {
        let analyzer = AdrSyncAnalyzer::new(48000);
        let video = vec![0_u8; 1000];
        let audio = vec![0.0_f32; 1000];
        let confidence = analyzer.calculate_sync_confidence(&video, &audio);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_get_cues_sorted() {
        let mut session = AdrSession::new("Scene 42", 48000);

        let cue1 = AdrCue::new(
            "Cue 1",
            Timecode::from_frames(2000, 24.0),
            Timecode::from_frames(2100, 24.0),
        );
        let cue2 = AdrCue::new(
            "Cue 2",
            Timecode::from_frames(1000, 24.0),
            Timecode::from_frames(1100, 24.0),
        );

        session.add_cue(cue1);
        session.add_cue(cue2);

        let sorted = session.get_cues_sorted();
        assert_eq!(sorted[0].1.description, "Cue 2");
        assert_eq!(sorted[1].1.description, "Cue 1");
    }
}
