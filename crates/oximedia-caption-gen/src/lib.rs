//! # oximedia-caption-gen
//!
//! Advanced caption and subtitle generation for the OxiMedia Sovereign Media
//! Framework.
//!
//! This crate provides speech-to-caption alignment with frame-accurate timing,
//! greedy and optimal (Knuth-Plass DP) line-breaking algorithms, WCAG 2.1
//! accessibility compliance checking, and speaker diarization metadata with
//! crosstalk detection — all in pure Rust.
//!
//! ## Modules
//!
//! - [`alignment`] — Word timestamps, transcript segments, segment
//!   merging/splitting, frame alignment, and caption block construction.
//! - [`autopunct`] — Deterministic auto-punctuation and sentence capitalisation.
//! - [`burn_in`] — Burned-in subtitle rendering onto raw RGBA video frames
//!   using a built-in 8×12 bitmap font.
//! - [`caption_diff`] — Compare two caption tracks and report differences.
//! - [`caption_format_adapter`] — Serialize caption tracks to SRT/VTT/TTML.
//! - [`caption_style_guide`] — Style guide rule enforcement over caption tracks.
//! - [`caption_timing_adjuster`] — Shift, stretch, snap, and EDL-remap
//!   caption timestamps.
//! - [`diarization`] — Speaker metadata, turn merging, per-speaker statistics,
//!   crosstalk detection, voice activity ratio, and speaker-to-caption
//!   assignment.
//! - [`forced_narrative`] — Forced narrative (FN) and SDH subtitle detection
//!   and classification.
//! - [`language_detect`] — Byte-trigram language detection for locale-aware
//!   line-breaking.
//! - [`line_breaking`] — Greedy and optimal line-breaking, reading-speed
//!   helpers (CPS), and line-balance optimisation.
//! - [`multi_language`] — Bilingual caption layout (primary + secondary
//!   language).
//! - [`multi_language_sync`] — Anchor-point synchronisation of multi-language
//!   caption tracks.
//! - [`multilang`] — Multi-language subtitle support with ISO 639-1 validated
//!   language codes, SRT export, and cross-language timing merge.
//! - [`phoneme_timing`] — Phoneme-level timing estimation from word timestamps.
//! - [`profanity`] — Configurable profanity filter for caption text.
//! - [`punctuation_restoration`] — Rule-based punctuation restoration for raw
//!   ASR output.
//! - [`reading_speed`] — Caption reading-speed validation (WPS-based).
//! - [`style_generator`] — Font size, position, and colour suggestions based on
//!   video frame analysis.
//! - [`style_presets`] — Ready-made caption style configs (Netflix, BBC, WCAG).
//! - [`translate`] — Stub subtitle translation pipeline.
//! - [`wcag`] — WCAG 2.1 compliance checks (1.2.2, 1.2.4, 1.2.6), reading
//!   speed validation, minimum display duration, gap detection, and compliance
//!   scoring.

pub mod alignment;
pub mod autopunct;
pub mod burn_in;
pub mod caption_diff;
pub mod caption_format_adapter;
pub mod caption_style_guide;
pub mod caption_timing_adjuster;
pub mod diarization;
pub mod forced_narrative;
pub mod language_detect;
pub mod line_breaking;
pub mod multi_language;
pub mod multi_language_sync;
pub mod multilang;
pub mod phoneme_timing;
pub mod profanity;
pub mod punctuation_restoration;
pub mod reading_speed;
pub mod style_generator;
pub mod style_presets;
pub mod translate;
pub mod wcag;

// ── Re-exports of key public types ──────────────────────────────────────────

pub use alignment::{
    align_to_frames, build_caption_blocks, merge_short_segments, split_long_segments,
    AlignmentError, CaptionBlock, CaptionPosition, TranscriptSegment, WordTimestamp,
};
pub use diarization::{
    assign_speakers_to_blocks, dominant_speaker, format_speaker_label, merge_consecutive_turns,
    speaker_stats, voice_activity_ratio, CrosstalkDetector, DiarizationResult, Speaker,
    SpeakerGender, SpeakerStats, SpeakerTurn,
};
pub use line_breaking::{
    compute_cps, greedy_break, optimal_break, reading_speed_ok, rebalance_lines, LineBalance,
    LineBreakAlgorithm, LineBreakConfig,
};
pub use wcag::{
    check_caption_coverage, check_cps, check_live_latency, check_min_duration, check_sign_language,
    compliance_score, run_all_checks, WcagChecker, WcagLevel, WcagViolation,
};

// ─── Error type ─────────────────────────────────────────────────────────────

/// Errors produced by caption generation operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CaptionGenError {
    /// A speech-to-caption alignment operation failed.
    #[error("alignment error: {0}")]
    Alignment(#[from] AlignmentError),

    /// A parameter value is invalid.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// A timestamp is invalid (e.g. start >= end).
    #[error("invalid timestamp")]
    InvalidTimestamp,

    /// The transcript is empty and cannot be processed.
    #[error("empty transcript")]
    EmptyTranscript,

    /// Parsing of caption data or configuration failed.
    #[error("parse error: {0}")]
    ParseError(String),
}

pub use burn_in::{BurnInConfig, SubtitleBurnIn, SubtitlePosition};
pub use multilang::{CaptionEntry, LanguageCode, MultiLangCaption, MultiLangCaptionBuilder};
