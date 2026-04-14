//! Audio codec implementations for `OxiMedia`.
//!
//! This crate provides audio encoding and decoding for royalty-free codecs:
//!
//! - **Opus** - Modern, high-quality codec for speech and music
//! - **Vorbis** - Open source lossy codec (Ogg Vorbis)
//! - **FLAC** - Free Lossless Audio Codec
//! - **MP3** - MPEG-1/2 Layer III (patents expired 2017)
//! - **PCM** - Uncompressed audio
//!
//! # Architecture
//!
//! All codecs implement unified traits:
//!
//! - [`AudioDecoder`] - Decode compressed packets to samples
//! - [`AudioEncoder`] - Encode samples to compressed packets
//!
//! # DSP
//!
//! The [`dsp`] module provides standalone digital signal processing implementations:
//!
//! - **Biquad Filters** - Second-order IIR filters
//! - **Parametric Equalizer** - Multi-band EQ
//! - **Dynamics Compressor** - Full-featured compressor
//! - **Reverb** - Schroeder reverb algorithm
//!
//! # Effects
//!
//! The [`effects`] module provides advanced modulation effects:
//!
//! - **Chorus** - Multi-voice chorus with LFO modulation
//! - **Flanger** - Short delay with feedback and sweeping
//! - **Phaser** - All-pass filter cascade for phase shifting
//!
//! # Spectrum Analysis
//!
//! The [`spectrum`] module provides frequency-domain analysis and visualization:
//!
//! - **FFT Analysis** - Fast Fourier Transform with window functions
//! - **Spectrum Analyzer** - Real-time frequency analysis
//! - **Spectrogram** - Time-frequency visualization
//! - **Waveform Display** - Time-domain rendering
//! - **Feature Extraction** - Spectral features and characteristics
//!
//! # Audio Fingerprinting
//!
//! The [`fingerprint`] module provides audio identification and matching:
//!
//! - **Fingerprint Generation** - Extract robust audio signatures
//! - **Hash-based Matching** - Fast database lookup
//! - **Duplicate Detection** - Find similar audio content
//! - **Content Identification** - Match against reference database
//!
//! # Loudness Normalization
//!
//! The [`loudness`] module provides broadcast-standard loudness measurement and normalization:
//!
//! - **EBU R128** - European Broadcasting Union standard (-23 LUFS)
//! - **ATSC A/85** - Advanced Television Systems Committee standard (-24 LKFS)
//! - **ITU-R BS.1770-4** - International loudness measurement algorithm
//! - **True Peak Detection** - Prevents inter-sample clipping
//! - **Loudness Range (LRA)** - Dynamic range measurement
//! - **K-Weighting Filter** - Perceptually accurate filtering
//! - **Gating Algorithm** - Two-stage gating for integrated loudness
//!
//! # Audio Metering
//!
//! The [`meters`] module provides professional audio metering tools:
//!
//! - **VU Meter** - IEC 60268-10 standard with 300ms ballistics
//! - **PPM** - Peak Programme Meters (BBC, EBU, Nordic, DIN standards)
//! - **Digital Peak Meter** - Sample-accurate dBFS peak detection
//! - **RMS Level Meter** - Root mean square level measurement
//! - **Correlation Meter** - Stereo phase correlation analysis
//! - **Goniometer** - Stereo field visualization (L/R and M/S)
//! - **LUFS Integration** - Broadcast loudness metering
//!
//! # Spatial Audio
//!
//! The [`spatial`] module provides 3D spatial audio processing:
//!
//! - **Ambisonics** - Scene-based spatial audio (1st, 2nd, 3rd order)
//! - **Binaural Rendering** - HRTF-based 3D audio for headphones
//! - **Panning** - Stereo, VBAP, DBAP panning algorithms
//! - **Spatial Reverb** - Room simulation with early reflections
//! - **ITU-R BS.2051** - Compliance with advanced sound systems
//!
//! # Audio Description
//!
//! The [`description`] module provides accessibility features for audio description:
//!
//! - **Multiple Mixing Strategies** - Replace, Mix, Duck, and Pause modes
//! - **WCAG 2.1 Compliance** - Accessibility standards support
//! - **DVS Compatibility** - Descriptive Video Service format
//! - **Automatic Ducking** - Intelligent main audio reduction with VAD
//! - **Frame-Accurate Timing** - Precise synchronization control
//! - **Text-to-Speech** - Generate AD from text with SSML support
//! - **Broadcast Quality** - Professional audio processing

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::unused_self,
    clippy::doc_markdown,
    clippy::needless_return,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::derive_partial_eq_without_eq,
    clippy::match_same_arms,
    clippy::trivially_copy_pass_by_ref,
    clippy::return_self_not_must_use,
    clippy::unnecessary_wraps,
    clippy::too_many_lines,
    clippy::enum_variant_names,
    clippy::unnecessary_cast,
    clippy::bool_to_int_with_if,
    clippy::string_lit_as_bytes,
    clippy::explicit_iter_loop,
    dead_code,
    clippy::single_match,
    clippy::if_same_then_else,
    clippy::collapsible_else_if,
    clippy::single_match_else,
    clippy::comparison_to_empty,
    clippy::bool_comparison,
    clippy::style,
    clippy::float_cmp,
    clippy::struct_excessive_bools,
    clippy::wildcard_imports,
    clippy::redundant_closure,
    clippy::manual_find,
    clippy::manual_memcpy,
    clippy::float_equality_without_abs,
    clippy::bind_instead_of_map,
    clippy::option_map_unit_fn,
    clippy::mixed_read_write_in_expression,
    clippy::unnecessary_map_or,
    clippy::match_like_matches_macro,
    clippy::inefficient_to_string,
    clippy::default_constructed_unit_structs,
    clippy::never_loop,
    clippy::missing_const_for_fn,
    clippy::explicit_write,
    clippy::range_plus_one,
    clippy::unused_unit,
    clippy::clone_on_copy,
    clippy::type_complexity,
    clippy::pedantic,
    clippy::approx_constant
)]

pub mod audio_clock;
pub mod audio_format;
pub mod audio_graph;
pub mod audio_metadata;
pub mod audio_pipeline;
pub mod biquad;
pub mod channel_layout;
pub mod click_remove;
pub mod codec;
pub mod compressor;
pub mod crossfade;
pub mod delay_line;
pub mod description;
pub mod dither;
pub mod dsp;
pub mod ducking;
pub mod effects;
pub mod envelope;
pub mod error;
pub mod fingerprint;
pub mod fingerprinting;
pub mod format_convert;
pub mod frame;
pub mod frame_convert;
pub mod gapless;
pub mod gate;
pub mod graphic_eq;
pub mod jitter_buffer;
pub mod loudness;
pub mod loudness_gating;
pub mod loudness_meter;
pub mod loudness_norm;
pub mod loudness_simple;
pub mod meters;
pub mod mixing;
pub mod multiband_compressor;
pub mod noise_gate;
pub mod noise_reduce;
pub mod pan_law;
pub mod pitch;
pub mod pitch_detector;
pub mod resample;
pub mod resample_quality;
pub mod resampler;
pub mod sample_buffer;
pub mod sample_rate_converter;
pub mod sidechain;
pub mod silence_detect;
pub mod spatial;
pub mod spatial_panning;
pub mod spectrum;
pub mod stereo_widener;
pub mod stream_buffer;
pub mod sync;
pub mod traits;
pub mod transient_detector;
pub mod vad;
pub mod wav;

#[cfg(feature = "opus")]
pub mod opus;

#[cfg(feature = "vorbis")]
pub mod vorbis;

#[cfg(feature = "flac")]
pub mod flac;

#[cfg(feature = "mp3")]
pub mod mp3;

#[cfg(feature = "pcm")]
pub mod pcm;

// Re-exports
pub use error::{AudioError, AudioResult};
pub use frame::{AudioBuffer, AudioFrame, Channel, ChannelLayout};
pub use loudness::normalize::{AutoGainConfig, AutoGainProcessor};
pub use resample::{Resampler, ResamplerQuality};
pub use traits::{
    AudioDecoder, AudioDecoderConfig, AudioEncoder, AudioEncoderConfig, EncodedAudioPacket,
};

#[cfg(feature = "opus")]
pub use opus::{OpusDecoder, OpusEncoder};

#[cfg(feature = "vorbis")]
pub use vorbis::{VorbisDecoder, VorbisEncoder};

#[cfg(feature = "flac")]
pub use flac::{FlacDecoder, FlacEncoder};

#[cfg(feature = "mp3")]
pub use mp3::Mp3Decoder;

#[cfg(feature = "pcm")]
pub use pcm::{PcmDecoder, PcmEncoder};
