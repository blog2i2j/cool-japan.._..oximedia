//! Performance regression detection modules.

pub mod alert;
pub mod detect;

pub use alert::{AlertLevel, RegressionAlert};
pub use detect::{ExtendedRegressionInfo, RegressionConfig, RegressionDetector, RegressionInfo};
