//! Multi-camera synchronisation using various sync methods.
//!
//! Supports LTC timecode, clapper detection, audio correlation, and embedded
//! timecode to align multiple video/audio streams to a common timeline.

#![allow(dead_code)]

/// Method used to synchronise multi-camera streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyncMethod {
    /// Linear Timecode embedded in the audio or metadata track.
    Ltc,
    /// Clapper slate detection (visual and/or audio spike).
    Clapper,
    /// Audio waveform cross-correlation.
    AudioCorrelate,
    /// Embedded timecode (SMPTE, VITC, etc.).
    Timecode,
}

impl SyncMethod {
    /// Returns the theoretical accuracy of this sync method in milliseconds.
    #[must_use]
    pub const fn accuracy_ms(&self) -> f64 {
        match self {
            Self::Ltc => 0.5,
            Self::Clapper => 2.0,
            Self::AudioCorrelate => 1.0,
            Self::Timecode => 0.1,
        }
    }

    /// Returns a human-readable name for this method.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Ltc => "LTC Timecode",
            Self::Clapper => "Clapper/Slate",
            Self::AudioCorrelate => "Audio Correlation",
            Self::Timecode => "Embedded Timecode",
        }
    }
}

/// A single input stream to be synchronised.
#[derive(Debug, Clone)]
pub struct SyncStream {
    /// Unique identifier for this stream.
    pub stream_id: String,
    /// Optional audio samples for correlation-based sync (normalised f32).
    pub audio_samples: Vec<f32>,
    /// Optional timecode string (e.g. `"01:00:00:00"`) for timecode-based sync.
    pub timecode: Option<String>,
    /// Optional sample rate when audio samples are provided.
    pub sample_rate: Option<u32>,
}

impl SyncStream {
    /// Creates a new stream with audio samples only.
    #[must_use]
    pub fn audio(stream_id: impl Into<String>, samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            stream_id: stream_id.into(),
            audio_samples: samples,
            timecode: None,
            sample_rate: Some(sample_rate),
        }
    }

    /// Creates a new stream with a timecode string only.
    #[must_use]
    pub fn with_timecode(stream_id: impl Into<String>, timecode: impl Into<String>) -> Self {
        Self {
            stream_id: stream_id.into(),
            audio_samples: Vec::new(),
            timecode: Some(timecode.into()),
            sample_rate: None,
        }
    }
}

/// Synchronisation result for a single stream relative to the reference stream.
#[derive(Debug, Clone)]
pub struct StreamSyncResult {
    /// Identifier of the stream that was synced.
    pub stream_id: String,
    /// Offset to apply to this stream in milliseconds (positive = delay).
    pub offset_ms: f64,
    /// Confidence score for this sync result (0.0–1.0).
    pub confidence: f64,
    /// The method that produced this result.
    pub method: SyncMethod,
}

/// Overall multi-camera sync result.
#[derive(Debug)]
pub struct MulticamSyncResult {
    /// Per-stream synchronisation results (reference stream is index 0).
    pub streams: Vec<StreamSyncResult>,
    /// Method used to produce the sync.
    pub method: SyncMethod,
}

impl MulticamSyncResult {
    /// Returns the maximum absolute offset across all streams in milliseconds.
    #[must_use]
    pub fn max_offset_ms(&self) -> f64 {
        self.streams
            .iter()
            .map(|s| s.offset_ms.abs())
            .fold(0.0f64, f64::max)
    }

    /// Returns the average confidence across all stream results.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn average_confidence(&self) -> f64 {
        if self.streams.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.streams.iter().map(|s| s.confidence).sum();
        sum / self.streams.len() as f64
    }

    /// Returns the sync result for a stream by ID, if present.
    #[must_use]
    pub fn get_stream(&self, stream_id: &str) -> Option<&StreamSyncResult> {
        self.streams.iter().find(|s| s.stream_id == stream_id)
    }
}

/// Synchronises multiple camera streams to a shared timeline.
#[derive(Debug)]
pub struct MulticamSyncer {
    streams: Vec<SyncStream>,
    method: SyncMethod,
}

impl MulticamSyncer {
    /// Creates a new syncer with the given sync method.
    #[must_use]
    pub fn new(method: SyncMethod) -> Self {
        Self {
            streams: Vec::new(),
            method,
        }
    }

    /// Returns the number of streams registered.
    #[must_use]
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Adds a stream to the syncer. The first stream added is used as the
    /// reference (offset 0).
    pub fn add_stream(&mut self, stream: SyncStream) {
        self.streams.push(stream);
    }

    /// Synchronises all streams to the reference (first) stream.
    ///
    /// Returns `None` when fewer than two streams have been added.
    #[must_use]
    pub fn sync_all(&self) -> Option<MulticamSyncResult> {
        if self.streams.len() < 2 {
            return None;
        }

        let mut results = Vec::new();

        // Reference stream always has offset 0.
        let reference = &self.streams[0];
        results.push(StreamSyncResult {
            stream_id: reference.stream_id.clone(),
            offset_ms: 0.0,
            confidence: 1.0,
            method: self.method,
        });

        for stream in self.streams.iter().skip(1) {
            let (offset_ms, confidence) = match self.method {
                SyncMethod::AudioCorrelate => self.audio_correlate_offset(reference, stream),
                SyncMethod::Ltc | SyncMethod::Timecode => self.timecode_offset(reference, stream),
                SyncMethod::Clapper => self.clapper_offset(reference, stream),
            };
            results.push(StreamSyncResult {
                stream_id: stream.stream_id.clone(),
                offset_ms,
                confidence,
                method: self.method,
            });
        }

        Some(MulticamSyncResult {
            streams: results,
            method: self.method,
        })
    }

    /// Computes an audio-correlation-based offset between two streams.
    #[allow(clippy::cast_precision_loss)]
    fn audio_correlate_offset(&self, reference: &SyncStream, other: &SyncStream) -> (f64, f64) {
        let a = &reference.audio_samples;
        let b = &other.audio_samples;
        if a.is_empty() || b.is_empty() {
            return (0.0, 0.0);
        }
        let sr = f64::from(reference.sample_rate.unwrap_or(48_000));
        let max_shift = (a.len().min(b.len()) / 4).max(1);
        let mut best_shift = 0i64;
        let mut best_corr: f64 = -1.0;
        let len = a.len().min(b.len());

        for lag in 0..=max_shift as i64 {
            for sign in [1i64, -1i64] {
                let shift = lag * sign;
                let corr = Self::xcorr(a, b, shift, len);
                if corr > best_corr {
                    best_corr = corr;
                    best_shift = shift;
                }
            }
        }
        let offset_ms = (best_shift as f64 / sr) * 1000.0;
        let confidence = best_corr.clamp(0.0, 1.0);
        (offset_ms, confidence)
    }

    /// Simple normalised cross-correlation.
    #[allow(clippy::cast_precision_loss)]
    fn xcorr(a: &[f32], b: &[f32], lag: i64, len: usize) -> f64 {
        let mut sum = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..len {
            let j = i as i64 + lag;
            if j < 0 || j as usize >= b.len() {
                continue;
            }
            let av = f64::from(a[i]);
            let bv = f64::from(b[j as usize]);
            sum += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        let denom = (na * nb).sqrt();
        if denom == 0.0 {
            0.0
        } else {
            sum / denom
        }
    }

    /// Parses a timecode string to milliseconds (HH:MM:SS:FF @ 25 fps).
    #[allow(clippy::cast_precision_loss)]
    fn parse_timecode_ms(tc: &str) -> Option<f64> {
        let parts: Vec<&str> = tc.split(':').collect();
        if parts.len() != 4 {
            return None;
        }
        let h: f64 = parts[0].parse().ok()?;
        let m: f64 = parts[1].parse().ok()?;
        let s: f64 = parts[2].parse().ok()?;
        let f: f64 = parts[3].parse().ok()?;
        Some((h * 3600.0 + m * 60.0 + s + f / 25.0) * 1000.0)
    }

    fn timecode_offset(&self, reference: &SyncStream, other: &SyncStream) -> (f64, f64) {
        let ref_ms = reference
            .timecode
            .as_deref()
            .and_then(Self::parse_timecode_ms)
            .unwrap_or(0.0);
        let other_ms = other
            .timecode
            .as_deref()
            .and_then(Self::parse_timecode_ms)
            .unwrap_or(0.0);
        (ref_ms - other_ms, 0.95)
    }

    fn clapper_offset(&self, _reference: &SyncStream, _other: &SyncStream) -> (f64, f64) {
        // Simplified: use audio correlation as a stand-in.
        self.audio_correlate_offset(_reference, _other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_method_accuracy_ordering() {
        // Timecode should be most accurate (smallest value)
        assert!(SyncMethod::Timecode.accuracy_ms() < SyncMethod::Ltc.accuracy_ms());
        assert!(SyncMethod::Ltc.accuracy_ms() < SyncMethod::AudioCorrelate.accuracy_ms());
        assert!(SyncMethod::AudioCorrelate.accuracy_ms() < SyncMethod::Clapper.accuracy_ms());
    }

    #[test]
    fn test_sync_method_names_nonempty() {
        let methods = [
            SyncMethod::Ltc,
            SyncMethod::Clapper,
            SyncMethod::AudioCorrelate,
            SyncMethod::Timecode,
        ];
        for m in methods {
            assert!(!m.name().is_empty());
        }
    }

    #[test]
    fn test_sync_stream_audio_constructor() {
        let s = SyncStream::audio("cam1", vec![0.0, 1.0], 48_000);
        assert_eq!(s.stream_id, "cam1");
        assert_eq!(s.sample_rate, Some(48_000));
        assert!(s.timecode.is_none());
    }

    #[test]
    fn test_sync_stream_timecode_constructor() {
        let s = SyncStream::with_timecode("cam2", "01:00:00:00");
        assert_eq!(s.stream_id, "cam2");
        assert_eq!(s.timecode.as_deref(), Some("01:00:00:00"));
        assert!(s.audio_samples.is_empty());
    }

    #[test]
    fn test_syncer_add_stream_count() {
        let mut syncer = MulticamSyncer::new(SyncMethod::AudioCorrelate);
        syncer.add_stream(SyncStream::audio("a", vec![], 48_000));
        syncer.add_stream(SyncStream::audio("b", vec![], 48_000));
        assert_eq!(syncer.stream_count(), 2);
    }

    #[test]
    fn test_syncer_sync_all_requires_two_streams() {
        let mut syncer = MulticamSyncer::new(SyncMethod::AudioCorrelate);
        syncer.add_stream(SyncStream::audio("a", vec![1.0], 48_000));
        assert!(syncer.sync_all().is_none());
    }

    #[test]
    fn test_syncer_sync_all_reference_offset_zero() {
        let mut syncer = MulticamSyncer::new(SyncMethod::AudioCorrelate);
        let sig: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin()).collect();
        syncer.add_stream(SyncStream::audio("ref", sig.clone(), 48_000));
        syncer.add_stream(SyncStream::audio("b", sig, 48_000));
        let result = syncer.sync_all().expect("result should be valid");
        assert!((result.streams[0].offset_ms).abs() < f64::EPSILON);
        assert_eq!(result.streams[0].stream_id, "ref");
    }

    #[test]
    fn test_syncer_sync_all_identical_signals() {
        let mut syncer = MulticamSyncer::new(SyncMethod::AudioCorrelate);
        let sig: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin()).collect();
        syncer.add_stream(SyncStream::audio("ref", sig.clone(), 48_000));
        syncer.add_stream(SyncStream::audio("b", sig, 48_000));
        let result = syncer.sync_all().expect("result should be valid");
        // Identical signals → offset should be 0
        assert!((result.streams[1].offset_ms).abs() < 1.0);
    }

    #[test]
    fn test_multicam_result_max_offset_ms() {
        let results = MulticamSyncResult {
            streams: vec![
                StreamSyncResult {
                    stream_id: "a".to_string(),
                    offset_ms: 0.0,
                    confidence: 1.0,
                    method: SyncMethod::Timecode,
                },
                StreamSyncResult {
                    stream_id: "b".to_string(),
                    offset_ms: -15.0,
                    confidence: 0.9,
                    method: SyncMethod::Timecode,
                },
                StreamSyncResult {
                    stream_id: "c".to_string(),
                    offset_ms: 30.0,
                    confidence: 0.85,
                    method: SyncMethod::Timecode,
                },
            ],
            method: SyncMethod::Timecode,
        };
        assert!((results.max_offset_ms() - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_multicam_result_average_confidence() {
        let results = MulticamSyncResult {
            streams: vec![
                StreamSyncResult {
                    stream_id: "a".to_string(),
                    offset_ms: 0.0,
                    confidence: 1.0,
                    method: SyncMethod::Clapper,
                },
                StreamSyncResult {
                    stream_id: "b".to_string(),
                    offset_ms: 5.0,
                    confidence: 0.8,
                    method: SyncMethod::Clapper,
                },
            ],
            method: SyncMethod::Clapper,
        };
        assert!((results.average_confidence() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_multicam_result_get_stream_found() {
        let results = MulticamSyncResult {
            streams: vec![StreamSyncResult {
                stream_id: "cam3".to_string(),
                offset_ms: 10.0,
                confidence: 0.95,
                method: SyncMethod::Ltc,
            }],
            method: SyncMethod::Ltc,
        };
        assert!(results.get_stream("cam3").is_some());
    }

    #[test]
    fn test_multicam_result_get_stream_not_found() {
        let results = MulticamSyncResult {
            streams: vec![],
            method: SyncMethod::Ltc,
        };
        assert!(results.get_stream("missing").is_none());
    }

    #[test]
    fn test_timecode_sync() {
        let mut syncer = MulticamSyncer::new(SyncMethod::Timecode);
        syncer.add_stream(SyncStream::with_timecode("ref", "01:00:00:00"));
        syncer.add_stream(SyncStream::with_timecode("b", "01:00:01:00")); // 1 s later
        let result = syncer.sync_all().expect("result should be valid");
        // ref_ms - other_ms = 3_600_000 - 3_601_000 = -1000 ms
        assert!((result.streams[1].offset_ms - (-1000.0)).abs() < 1.0);
    }

    #[test]
    fn test_syncer_empty_audio_gives_zero_offset() {
        let mut syncer = MulticamSyncer::new(SyncMethod::AudioCorrelate);
        syncer.add_stream(SyncStream::audio("a", vec![], 48_000));
        syncer.add_stream(SyncStream::audio("b", vec![], 48_000));
        let result = syncer.sync_all().expect("result should be valid");
        assert!((result.streams[1].offset_ms).abs() < f64::EPSILON);
    }
}
