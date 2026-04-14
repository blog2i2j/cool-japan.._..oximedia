//! Broadcast automation and control system for OxiMedia.
//!
//! This crate provides comprehensive 24/7 broadcast automation with:
//!
//! - **Master Control**: Centralized broadcast orchestration
//! - **Channel Automation**: Multi-channel automated playout
//! - **Device Control**: VTR, server, and router control (VDCP, Sony 9-pin, GPI/GPO)
//! - **Playlist Execution**: Frame-accurate playlist playback
//! - **Live Switching**: Automated live production workflows
//! - **Failover**: Automatic redundancy and hot standby
//! - **Emergency Alert System**: EAS compliance and alert insertion
//! - **As-run Logging**: Comprehensive broadcast logging
//! - **System Monitoring**: Proactive health monitoring and metrics
//! - **Remote Control**: Web-based control and monitoring
//! - **Lua Scripting**: Custom automation scripting
//!
//! # Architecture
//!
//! The automation system follows a hierarchical architecture:
//!
//! ```text
//! Master Control
//!     ├── Channel 1 Automation
//!     │   ├── Playlist Executor
//!     │   ├── Device Controllers
//!     │   └── Live Switcher
//!     ├── Channel 2 Automation
//!     └── Shared Services
//!         ├── Failover Manager
//!         ├── EAS System
//!         ├── Logging
//!         └── Monitoring
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use oximedia_automation::{MasterControl, MasterControlConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create master control system
//! let config = MasterControlConfig::default();
//! let mut master = MasterControl::new(config).await?;
//!
//! // Start automation
//! master.start().await?;
//!
//! // Monitor system status
//! let status = master.status().await?;
//! println!("System status: {:?}", status);
//! # Ok(())
//! # }
//! ```
//!
//! # Device Control Protocols
//!
//! - **VDCP**: Video Disk Control Protocol (IEEE 1394)
//! - **Sony 9-pin**: RS-422 VTR control
//! - **GPI/GPO**: General Purpose Interface triggers
//!
//! # Features
//!
//! - Frame-accurate timing for broadcast operations
//! - Hot standby failover with automatic switching
//! - Multi-channel support for simultaneous broadcasts
//! - EAS compliance with automatic alert insertion
//! - Comprehensive as-run logging
//! - Real-time monitoring and alerting
//! - Web-based remote control interface
//! - Lua scripting for custom workflows

#![forbid(unsafe_code)]
#![allow(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

pub mod action_catalog;
pub mod audit_trail;
pub mod automation_log;
pub mod channel;
pub mod compliance_checker;
pub mod condition_eval;
pub mod config;
pub mod cue_trigger;
pub mod device;
pub mod device_control;
pub mod eas;
pub mod event_bus;
pub mod event_log;
pub mod examples;
pub mod execution_log;
pub mod failover;
pub mod health_probe;
pub mod interlock;
pub mod live;
pub mod logging;
pub mod macro_ops;
pub mod master;
pub mod media_router;
pub mod monitor;
pub mod playlist;
pub mod priority_queue;
pub mod protocol;
pub mod rate_limiter;
pub mod remote;
pub mod rundown;
pub mod schedule_template;
pub mod scheduler_rule;
pub mod script;
pub mod signal_router;
pub mod state_snapshot;
pub mod task_executor;
pub mod trigger;
pub mod utils;
pub mod workflow;
pub mod workflow_trigger;

// Re-export main types
pub use channel::automation::{ChannelAutomation, ChannelConfig};
pub use channel::playout::{PlayoutEngine, PlayoutState};
pub use device::control::{DeviceController, DeviceType};
pub use eas::alert::{EasAlert, EasAlertType};
pub use failover::manager::{FailoverConfig, FailoverManager};
pub use logging::asrun::{AsRunEntry, AsRunLog, BatchedAsRunLogger};
pub use master::control::{MasterControl, MasterControlConfig};
pub use master::state::{SystemState, SystemStatus};
pub use monitor::system::{MonitorConfig, SystemMonitor};
pub use remote::server::{RemoteConfig, RemoteServer};
pub use script::engine::{ScriptContext, ScriptEngine};

use thiserror::Error;

/// Result type for automation operations.
pub type Result<T> = std::result::Result<T, AutomationError>;

/// Errors that can occur in automation operations.
#[derive(Debug, Error)]
pub enum AutomationError {
    /// Master control error
    #[error("Master control error: {0}")]
    MasterControl(String),

    /// Channel automation error
    #[error("Channel automation error: {0}")]
    ChannelAutomation(String),

    /// Device control error
    #[error("Device control error: {0}")]
    DeviceControl(String),

    /// Protocol error
    #[error("Protocol error: {0}")]
    Protocol(String),

    /// Playlist execution error
    #[error("Playlist execution error: {0}")]
    PlaylistExecution(String),

    /// Live switching error
    #[error("Live switching error: {0}")]
    LiveSwitching(String),

    /// Failover error
    #[error("Failover error: {0}")]
    Failover(String),

    /// EAS error
    #[error("EAS error: {0}")]
    Eas(String),

    /// Logging error
    #[error("Logging error: {0}")]
    Logging(String),

    /// Monitoring error
    #[error("Monitoring error: {0}")]
    Monitoring(String),

    /// Remote control error
    #[error("Remote control error: {0}")]
    RemoteControl(String),

    /// Scripting error
    #[error("Scripting error: {0}")]
    Scripting(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Invalid state
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Timeout
    #[error("Operation timeout")]
    Timeout,

    /// Not found
    #[error("Not found: {0}")]
    NotFound(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AutomationError::MasterControl("test error".to_string());
        assert_eq!(err.to_string(), "Master control error: test error");
    }

    #[test]
    fn test_error_io_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = AutomationError::from(io_err);
        assert!(matches!(err, AutomationError::Io(_)));
    }
}
