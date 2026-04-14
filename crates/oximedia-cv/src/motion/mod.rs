//! Motion analysis modules.
//!
//! This module groups motion-related processing algorithms including
//! background subtraction.
//!
//! # Modules
//!
//! - [`bg_sub`]: Frame-differencing background subtraction ([`bg_sub::FrameDiffSubtractor`])

pub mod bg_sub;

pub use bg_sub::FrameDiffSubtractor;
