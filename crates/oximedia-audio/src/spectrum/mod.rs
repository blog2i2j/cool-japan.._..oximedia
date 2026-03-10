//! Audio spectrum analysis and visualization.
//!
//! This module provides comprehensive tools for analyzing and visualizing audio signals
//! in the frequency domain.
//!
//! # Features
//!
//! - **FFT Analysis**: Fast Fourier Transform with multiple window functions
//! - **Spectrum Analyzer**: Real-time frequency spectrum analysis with peak detection
//! - **Spectrogram**: Time-frequency visualization
//! - **Waveform Display**: Time-domain waveform rendering
//! - **Feature Extraction**: Spectral features (centroid, spread, rolloff, flux, etc.)
//! - **Visualizations**: VU meters, peak meters, band analyzers
//!
//! # Examples
//!
//! ## Basic Spectrum Analysis
//!
//! ```rust,no_run
//! use oximedia_audio::spectrum::{SpectrumAnalyzer, SpectrumConfig};
//! use oximedia_audio::AudioFrame;
//!
//! // Create analyzer
//! let config = SpectrumConfig::new(2048);
//! let mut analyzer = SpectrumAnalyzer::new(config)?;
//!
//! // Analyze audio frame
//! # let frame = AudioFrame::new(
//! #     oximedia_core::SampleFormat::F32,
//! #     44100,
//! #     oximedia_audio::ChannelLayout::Stereo
//! # );
//! let spectrum = analyzer.analyze(&frame)?;
//!
//! // Access spectrum data
//! println!("Peak frequency: {:?} Hz", spectrum.max_frequency());
//! println!("Peaks detected: {}", spectrum.peaks.len());
//! ```
//!
//! ## Generate Spectrogram
//!
//! ```rust,no_run
//! use oximedia_audio::spectrum::{SpectrogramGenerator, SpectrogramConfig, ColorMap};
//!
//! let config = SpectrogramConfig {
//!     fft_size: 2048,
//!     color_map: ColorMap::Viridis,
//!     ..SpectrogramConfig::default()
//! };
//!
//! let mut generator = SpectrogramGenerator::new(config)?;
//!
//! // Process audio frames
//! # let frames: Vec<oximedia_audio::AudioFrame> = vec![];
//! generator.process_frames(&frames)?;
//!
//! // Generate image
//! let spectrogram = generator.generate();
//! spectrogram.save_ppm("spectrogram.ppm")?;
//! ```
//!
//! ## Extract Spectral Features
//!
//! ```rust,no_run
//! use oximedia_audio::spectrum::{SpectrumAnalyzer, SpectrumConfig, FeatureExtractor};
//!
//! let config = SpectrumConfig::default();
//! let mut analyzer = SpectrumAnalyzer::new(config)?;
//! let mut feature_extractor = FeatureExtractor::new();
//!
//! # let frame = oximedia_audio::AudioFrame::new(
//! #     oximedia_core::SampleFormat::F32,
//! #     44100,
//! #     oximedia_audio::ChannelLayout::Stereo
//! # );
//! let spectrum = analyzer.analyze(&frame)?;
//! let features = feature_extractor.extract(&spectrum);
//!
//! println!("Spectral centroid: {} Hz", features.centroid);
//! println!("Spectral spread: {} Hz", features.spread);
//! println!("Fundamental frequency: {:?} Hz", features.fundamental_frequency);
//! ```
//!
//! ## Render Waveform
//!
//! ```rust,no_run
//! use oximedia_audio::spectrum::{WaveformRenderer, WaveformConfig, WaveformMode};
//!
//! let config = WaveformConfig {
//!     width: 800,
//!     height: 200,
//!     mode: WaveformMode::MinMax,
//!     ..WaveformConfig::default()
//! };
//!
//! let renderer = WaveformRenderer::new(config);
//!
//! # let frame = oximedia_audio::AudioFrame::new(
//! #     oximedia_core::SampleFormat::F32,
//! #     44100,
//! #     oximedia_audio::ChannelLayout::Stereo
//! # );
//! let waveform = renderer.render(&frame)?;
//! waveform.save_ppm("waveform.ppm")?;
//! ```

pub mod analyzer;
pub mod features;
pub mod fft;
pub mod spectrogram;
pub mod visualize;
pub mod waveform;

// Re-export main types
pub use analyzer::{Peak, SpectrumAnalyzer, SpectrumConfig, SpectrumData};
pub use features::{
    BandAnalyzer, BandEnergy, DynamicRangeMeter, FeatureExtractor, FrequencyBand, Harmonic,
    SpectralFeatures, TimeDomainFeatures,
};
pub use fft::{FftProcessor, MelScale, OverlapAdd, WindowFunction};
pub use spectrogram::{
    ColorMap, RealtimeSpectrogram, SpectrogramConfig, SpectrogramGenerator, SpectrogramImage,
};
pub use visualize::{
    BandVisualizer, CircularSpectrumVisualizer, SpectrumVisualizer, SpectrumVisualizerConfig,
    SpectrumVisualizerImage, VuMeter, VuMeterConfig, VuMeterImage,
};
pub use waveform::{PhaseScope, WaveformConfig, WaveformImage, WaveformMode, WaveformRenderer};
