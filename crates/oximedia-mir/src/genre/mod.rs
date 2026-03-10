//! Genre classification from audio features.

pub mod classify;
pub mod features;

pub use classify::{GenreClassifier, StreamingGenreClassifier};
pub use features::GenreFeatures;
