//! Cached chromagram computation shared between chord recognition and key detection.
//!
//! [`ChromaCache`] computes a chromagram once and stores it so that
//! multiple consumers (key detection, chord recognition, similarity, etc.)
//! can reuse the result without redundant FFT computation.
//!
//! All data is stored as plain `Vec<f32>` with manual stride access.
//! Rows are chroma frames; columns are the 12 pitch-class bins.
//! Layout: `data[frame * 12 + bin]`.

use crate::chromagram::{ChromaVector, ChromagramAnalyzer, ChromagramConfig};

/// A computed chromagram with cached metadata.
#[derive(Debug, Clone)]
pub struct CachedChromagram {
    /// Flattened chroma data: `data[frame * 12 + bin]`.
    data: Vec<f64>,
    /// Number of frames.
    n_frames: usize,
    /// Sample rate used for computation.
    pub sample_rate: f32,
    /// Window size used.
    pub window_size: usize,
    /// Hop size used.
    pub hop_size: usize,
}

impl CachedChromagram {
    /// Create a `CachedChromagram` from a vector of [`ChromaVector`]s.
    #[must_use]
    pub fn from_chroma_vectors(
        vectors: &[ChromaVector],
        sample_rate: f32,
        window_size: usize,
        hop_size: usize,
    ) -> Self {
        let n_frames = vectors.len();
        let mut data = vec![0.0_f64; n_frames * 12];
        for (i, cv) in vectors.iter().enumerate() {
            for j in 0..12 {
                data[i * 12 + j] = cv.bins[j];
            }
        }
        Self {
            data,
            n_frames,
            sample_rate,
            window_size,
            hop_size,
        }
    }

    /// Number of chroma frames.
    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_frames == 0
    }

    /// Get the chroma vector for a specific frame.
    ///
    /// Returns `None` if `frame_idx` is out of range.
    #[must_use]
    pub fn frame(&self, frame_idx: usize) -> Option<[f64; 12]> {
        if frame_idx >= self.n_frames {
            return None;
        }
        let base = frame_idx * 12;
        let mut bins = [0.0_f64; 12];
        bins.copy_from_slice(&self.data[base..base + 12]);
        Some(bins)
    }

    /// Compute the mean chroma vector across all frames.
    #[must_use]
    pub fn mean_chroma(&self) -> [f64; 12] {
        if self.n_frames == 0 {
            return [0.0; 12];
        }
        let mut sum = [0.0_f64; 12];
        for frame in 0..self.n_frames {
            let base = frame * 12;
            for j in 0..12 {
                sum[j] += self.data[base + j];
            }
        }
        let n = self.n_frames as f64;
        sum.iter_mut().for_each(|v| *v /= n);
        sum
    }

    /// Compute per-frame sum (useful for onset-weighted key detection).
    #[must_use]
    pub fn frame_energy(&self, frame_idx: usize) -> f64 {
        if frame_idx >= self.n_frames {
            return 0.0;
        }
        let base = frame_idx * 12;
        self.data[base..base + 12].iter().sum()
    }

    /// Return a slice of `n_frames` consecutive frames starting at `start_frame`,
    /// clamped to available data.  Returns a `Vec<[f64; 12]>`.
    #[must_use]
    pub fn frames_range(&self, start_frame: usize, n: usize) -> Vec<[f64; 12]> {
        let end = (start_frame + n).min(self.n_frames);
        (start_frame..end).filter_map(|i| self.frame(i)).collect()
    }

    /// Aggregate chroma over a time window `[start_secs, end_secs]`.
    ///
    /// Frames whose centre time falls within the window are summed and then
    /// normalised to unit sum.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn aggregate_window(&self, start_secs: f32, end_secs: f32) -> [f64; 12] {
        let hop_secs = self.hop_size as f32 / self.sample_rate;
        let mut sum = [0.0_f64; 12];
        let mut count = 0_usize;

        for frame in 0..self.n_frames {
            let frame_time = frame as f32 * hop_secs;
            if frame_time >= start_secs && frame_time < end_secs {
                let base = frame * 12;
                for j in 0..12 {
                    sum[j] += self.data[base + j];
                }
                count += 1;
            }
        }

        if count > 0 {
            let total: f64 = sum.iter().sum();
            if total > 1e-12 {
                sum.iter_mut().for_each(|v| *v /= total);
            }
        }

        sum
    }
}

// ---------------------------------------------------------------------------
// ChromaCache (lazy computation)
// ---------------------------------------------------------------------------

/// A lazy cache that computes a chromagram on first access and reuses it
/// for subsequent calls.
///
/// Wrap your chromagram computation in this type to avoid redundant FFT
/// work when both chord recognition and key detection are enabled.
pub struct ChromaCache {
    /// Configuration for the underlying analyzer.
    config: ChromagramConfig,
    /// Cached result (computed on first call to `get`).
    cache: Option<CachedChromagram>,
}

impl ChromaCache {
    /// Create a new empty cache with the given configuration.
    #[must_use]
    pub fn new(config: ChromagramConfig) -> Self {
        Self {
            config,
            cache: None,
        }
    }

    /// Create a cache using default config at the given sample rate.
    #[must_use]
    pub fn with_sample_rate(sample_rate: f32) -> Self {
        Self::new(ChromagramConfig {
            sample_rate,
            ..ChromagramConfig::default()
        })
    }

    /// Compute the chromagram for `samples` (or return the cached version if
    /// it was already computed for the same signal length).
    ///
    /// Note: The cache is keyed only by existence (not by signal content).
    /// If you need to analyse different signals you should create a fresh
    /// [`ChromaCache`] for each one.
    pub fn get(&mut self, samples: &[f32]) -> &CachedChromagram {
        if self.cache.is_none() {
            let analyzer = ChromagramAnalyzer::new(self.config.clone());
            let vectors = analyzer.compute(samples);
            let cached = CachedChromagram::from_chroma_vectors(
                &vectors,
                self.config.sample_rate,
                self.config.window_size,
                self.config.hop_size,
            );
            self.cache = Some(cached);
        }
        // Safe: the branch above always sets `self.cache` to `Some` when it was
        // `None`, so by here `self.cache` is guaranteed to be `Some`.
        if let Some(ref c) = self.cache {
            c
        } else {
            unreachable!("cache was just populated in the branch above")
        }
    }

    /// Whether the cache has been populated.
    #[must_use]
    pub fn is_populated(&self) -> bool {
        self.cache.is_some()
    }

    /// Invalidate the cache (e.g. if the audio signal changes).
    pub fn invalidate(&mut self) {
        self.cache = None;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn make_sine(freq: f32, sr: f32, seconds: f32) -> Vec<f32> {
        let n = (sr * seconds) as usize;
        (0..n).map(|i| (TAU * freq * i as f32 / sr).sin()).collect()
    }

    #[test]
    fn test_cached_chromagram_empty() {
        let cache = CachedChromagram::from_chroma_vectors(&[], 44100.0, 4096, 512);
        assert!(cache.is_empty());
        assert_eq!(cache.n_frames(), 0);
    }

    #[test]
    fn test_cached_chromagram_frame_access() {
        let cv = ChromaVector {
            bins: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let cache = CachedChromagram::from_chroma_vectors(&[cv], 44100.0, 4096, 512);
        let frame = cache.frame(0).expect("frame 0 must exist");
        assert!((frame[0] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cached_chromagram_out_of_range() {
        let cv = ChromaVector { bins: [0.0; 12] };
        let cache = CachedChromagram::from_chroma_vectors(&[cv], 44100.0, 4096, 512);
        assert!(cache.frame(1).is_none());
    }

    #[test]
    fn test_mean_chroma_all_equal() {
        let cv = ChromaVector { bins: [2.0; 12] };
        let cache =
            CachedChromagram::from_chroma_vectors(&[cv.clone(), cv.clone()], 44100.0, 4096, 512);
        let mean = cache.mean_chroma();
        for &v in &mean {
            assert!((v - 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_frame_energy() {
        let mut cv = ChromaVector { bins: [0.0; 12] };
        cv.bins[0] = 3.0;
        cv.bins[6] = 2.0;
        let cache = CachedChromagram::from_chroma_vectors(&[cv], 44100.0, 4096, 512);
        let energy = cache.frame_energy(0);
        assert!((energy - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_frames_range() {
        let frames: Vec<ChromaVector> = (0..5)
            .map(|i| {
                let mut cv = ChromaVector { bins: [0.0; 12] };
                cv.bins[0] = i as f64;
                cv
            })
            .collect();
        let cache = CachedChromagram::from_chroma_vectors(&frames, 44100.0, 4096, 512);
        let range = cache.frames_range(1, 3);
        assert_eq!(range.len(), 3);
        assert!((range[0][0] - 1.0).abs() < 1e-9);
        assert!((range[2][0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_chroma_cache_lazy_computation() {
        let mut chroma_cache = ChromaCache::with_sample_rate(44100.0);
        assert!(!chroma_cache.is_populated());

        let signal = make_sine(440.0, 44100.0, 0.5);
        let _ = chroma_cache.get(&signal);
        assert!(chroma_cache.is_populated());

        // Second call should use cache
        let result1_n = chroma_cache.get(&signal).n_frames();
        let result2_n = chroma_cache.get(&signal).n_frames();
        assert_eq!(result1_n, result2_n);
    }

    #[test]
    fn test_chroma_cache_invalidate() {
        let mut chroma_cache = ChromaCache::with_sample_rate(44100.0);
        let signal = make_sine(440.0, 44100.0, 0.5);
        let _ = chroma_cache.get(&signal);
        assert!(chroma_cache.is_populated());
        chroma_cache.invalidate();
        assert!(!chroma_cache.is_populated());
    }

    #[test]
    fn test_aggregate_window_silence() {
        let cv = ChromaVector { bins: [0.0; 12] };
        let frames = vec![cv; 10];
        let cache = CachedChromagram::from_chroma_vectors(&frames, 44100.0, 512, 512);
        let agg = cache.aggregate_window(0.0, 1.0);
        let total: f64 = agg.iter().sum();
        // All zero input → result should be all zero
        assert!(total < 1e-12);
    }
}
