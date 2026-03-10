//! Integration tests for oximedia-monitor.

use oximedia_monitor::{MonitorConfig, OximediaMonitor};
use std::time::Duration;
use tempfile::tempdir;

/// Build a [`MonitorConfig`] that is optimised for fast test execution.
///
/// System-metrics collection is **disabled** so the expensive sysinfo I/O
/// (disk/network enumeration) is never triggered.  Tests that specifically
/// need system metrics must opt-in by enabling `metrics.enable_system_metrics`.
fn fast_config(dir: &tempfile::TempDir) -> MonitorConfig {
    let mut cfg = MonitorConfig::default();
    cfg.storage.db_path = dir.path().join("monitor.db");
    cfg.metrics.enable_system_metrics = false;
    cfg.metrics.enable_disk_metrics = false;
    cfg.metrics.collection_interval = Duration::from_millis(100);
    cfg
}

/// Build a config that enables system metrics (CPU + memory) but disables the
/// slow disk-enumeration path.
fn system_metrics_no_disk_config(dir: &tempfile::TempDir) -> MonitorConfig {
    let mut cfg = MonitorConfig::default();
    cfg.storage.db_path = dir.path().join("monitor.db");
    cfg.metrics.enable_system_metrics = true;
    cfg.metrics.enable_disk_metrics = false;
    cfg.metrics.collection_interval = Duration::from_millis(100);
    cfg
}

#[tokio::test]
async fn test_monitor_lifecycle() {
    let dir = tempdir().expect("dir should be valid");
    let monitor = OximediaMonitor::new(fast_config(&dir))
        .await
        .expect("monitor should be valid");
    monitor.start().await.expect("test expectation failed");
    assert!(monitor.metrics_collector().is_running().await);

    monitor.stop().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    assert!(!monitor.metrics_collector().is_running().await);
}

#[tokio::test]
async fn test_system_metrics_collection() {
    // System metrics collection (CPU + memory) is verified here at the
    // integration level.  Disk enumeration is disabled because on macOS with
    // many app-wrapper mounts the CoreFoundation disk-property queries take
    // 15-20 s per call, which would make the entire test suite impractically
    // slow.  Disk-specific behaviour is covered by the collector unit tests.
    let dir = tempdir().expect("dir should be valid");
    let monitor = OximediaMonitor::new(system_metrics_no_disk_config(&dir))
        .await
        .expect("test expectation failed");
    let metrics = monitor
        .system_metrics()
        .await
        .expect("metrics should be valid");
    assert!(metrics.is_some());

    let m = metrics.expect("m should be valid");
    assert!(m.cpu.cpu_count > 0);
    assert!(m.memory.total > 0);
}

#[tokio::test]
async fn test_application_metrics_initial_state() {
    let dir = tempdir().expect("dir should be valid");
    let monitor = OximediaMonitor::new(fast_config(&dir))
        .await
        .expect("monitor should be valid");
    let metrics = monitor.application_metrics();
    assert_eq!(metrics.encoding.total_frames, 0);
    assert_eq!(metrics.jobs.completed, 0);
}

#[tokio::test]
async fn test_quality_metrics_initial_state() {
    let dir = tempdir().expect("dir should be valid");
    let monitor = OximediaMonitor::new(fast_config(&dir))
        .await
        .expect("monitor should be valid");
    let metrics = monitor.quality_metrics();
    assert_eq!(metrics.bitrate.video_bitrate_bps, 0);
    assert!(metrics.scores.psnr.is_none());
}

#[tokio::test]
async fn test_encoding_metrics_tracking() {
    let dir = tempdir().expect("dir should be valid");
    let monitor = OximediaMonitor::new(fast_config(&dir))
        .await
        .expect("monitor should be valid");
    let tracker = monitor.metrics_collector().application_tracker();
    tracker.record_frame_encoded(16.67);
    tracker.record_frame_encoded(16.67);
    tracker.record_job_completed(30.0);

    let metrics = monitor.application_metrics();
    assert_eq!(metrics.encoding.total_frames, 2);
    assert_eq!(metrics.jobs.completed, 1);
}

#[tokio::test]
async fn test_quality_metrics_tracking() {
    let dir = tempdir().expect("dir should be valid");
    let monitor = OximediaMonitor::new(fast_config(&dir))
        .await
        .expect("monitor should be valid");
    let tracker = monitor.metrics_collector().quality_tracker();
    tracker.update_bitrate(5_000_000, 128_000);
    tracker.update_scores(Some(40.0), Some(0.99), Some(92.0));

    let metrics = monitor.quality_metrics();
    assert_eq!(metrics.bitrate.video_bitrate_bps, 5_000_000);
    assert_eq!(metrics.scores.psnr, Some(40.0));
    assert_eq!(metrics.scores.ssim, Some(0.99));
    assert_eq!(metrics.scores.vmaf, Some(92.0));
}

#[tokio::test]
async fn test_alert_manager_present() {
    let dir = tempdir().expect("dir should be valid");
    let monitor = OximediaMonitor::new(fast_config(&dir))
        .await
        .expect("monitor should be valid");
    // Alert manager is created when alerts.enabled == true (the default).
    assert!(monitor.alert_manager().is_some());
}

#[tokio::test]
async fn test_config_validation() {
    let mut config = MonitorConfig::default();
    assert!(config.validate().is_ok());

    // Invalid collection interval.
    config.metrics.collection_interval = std::time::Duration::from_millis(50);
    assert!(config.validate().is_err());
}
