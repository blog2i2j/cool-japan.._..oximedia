//! Automation module with timecode support.

pub mod curves;
pub mod timeline;

pub use curves::{CurveType, GainAutomation, GainCurveSegment};
pub use timeline::{AutomationAction, AutomationEvent, AutomationTimeline, FrameRate, Timecode};
