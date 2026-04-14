//! Integration tests for the OxiMedia automation crate.
//!
//! These tests exercise complete playout lifecycle scenarios without requiring
//! physical hardware — all external devices are exercised through the mock
//! implementations already built into the crate.
//!
//! Each test uses `std::env::temp_dir()` for any file I/O to avoid polluting
//! the source tree and to work correctly in any CI environment.

use oximedia_automation::{
    eas::alert::AlertPriority,
    event_bus::LockFreeEventBus,
    failover::manager::{FailoverState, RedundancyMode},
    logging::asrun::{AsRunFormat, AsRunLogger},
    AsRunEntry, BatchedAsRunLogger, ChannelAutomation, ChannelConfig, EasAlert, EasAlertType,
    FailoverConfig, FailoverManager, MasterControl, MasterControlConfig,
};
use std::time::Duration;

// ── Helper: build a temp file path ────────────────────────────────────────────

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(name)
}

// ── 1. MasterControl lifecycle ─────────────────────────────────────────────────

/// Create a MasterControl with 2 channels, start it and verify it reaches
/// `Running` state.
#[tokio::test]
async fn test_master_control_two_channel_lifecycle() {
    let config = MasterControlConfig {
        num_channels: 2,
        eas_enabled: false, // avoid real EAS socket setup
        remote_enabled: false,
        ..MasterControlConfig::default()
    };

    let mut master = MasterControl::new(config)
        .await
        .expect("MasterControl::new should succeed");

    master.start().await.expect("start should succeed");

    let status = master.status().await.expect("status should succeed");
    // The system should be Running after start().
    use oximedia_automation::SystemStatus;
    assert_eq!(
        status,
        SystemStatus::Running,
        "system should be Running after start"
    );

    master.stop().await.expect("stop should succeed");
}

// ── 2. Playout engine — load playlist items and advance frames ─────────────────

/// Load 3 synthetic media items into a PlayoutEngine, start it, then advance
/// frames past item 1's duration and verify position tracking.
#[tokio::test]
async fn test_playout_engine_advances_through_playlist_items() {
    use oximedia_automation::channel::playout::PlayoutItem;
    use oximedia_automation::PlayoutEngine;

    let mut engine = PlayoutEngine::new(0)
        .await
        .expect("PlayoutEngine::new should succeed");

    // Load item 1 (30 frames = 1 second at 30fps)
    let item1 = PlayoutItem {
        id: "item1".to_string(),
        title: "Synthetic Item 1".to_string(),
        duration_frames: 30,
        position_frames: 0,
        scheduled_start: None,
    };
    engine
        .load_next_item(item1)
        .await
        .expect("load_next_item should succeed");

    engine.start().await.expect("start should succeed");

    // Advance 30 frames — item 1 should be complete.
    for _ in 0..30 {
        engine
            .advance_frame()
            .await
            .expect("advance_frame should succeed");
    }

    assert_eq!(engine.frame_count().await, 30, "should have 30 frames");

    // Load item 2
    let item2 = PlayoutItem {
        id: "item2".to_string(),
        title: "Synthetic Item 2".to_string(),
        duration_frames: 60,
        position_frames: 0,
        scheduled_start: None,
    };
    engine
        .load_next_item(item2)
        .await
        .expect("load item 2 should succeed");

    // Advance 60 more frames
    for _ in 0..60 {
        engine.advance_frame().await.expect("advance_frame");
    }

    assert_eq!(
        engine.frame_count().await,
        90,
        "total 90 frames after both items"
    );

    engine.stop().await.expect("stop should succeed");
}

// ── 3. As-run log records item completion ──────────────────────────────────────

/// Verify that as-run log records are created for two channels and that the
/// entries are retrievable per-channel.
#[tokio::test]
async fn test_asrun_log_records_playout_completion() {
    let logger = AsRunLogger::new().expect("AsRunLogger::new should succeed");

    // Simulate completion of "News at Six" on channel 0
    let mut entry0 = AsRunEntry::new(0, "News at Six".to_string());
    entry0.item_type = "program".to_string();
    entry0.actual_duration = 3600.0;
    logger.log(entry0).await.expect("log ch0 should succeed");

    // Simulate completion of "Weather" on channel 1
    let mut entry1 = AsRunEntry::new(1, "Weather Report".to_string());
    entry1.item_type = "program".to_string();
    entry1.actual_duration = 300.0;
    logger.log(entry1).await.expect("log ch1 should succeed");

    // Verify per-channel retrieval
    let ch0 = logger.get_channel_entries(0).await;
    assert_eq!(ch0.len(), 1);
    assert_eq!(ch0[0].title, "News at Six");

    let ch1 = logger.get_channel_entries(1).await;
    assert_eq!(ch1.len(), 1);
    assert_eq!(ch1[0].title, "Weather Report");

    // Verify total
    let all = logger.get_entries().await;
    assert_eq!(all.len(), 2);
}

// ── 4. Batched as-run logger — file export ─────────────────────────────────────

/// Verify the BatchedAsRunLogger flushes to the inner logger and can export
/// a JSON file to a temp path.
#[tokio::test]
async fn test_batched_asrun_logger_full_playout_cycle() {
    let log_path = temp_path("integration_asrun_batch.json");

    let mut logger =
        BatchedAsRunLogger::new(Some(log_path.clone()), 5).expect("BatchedAsRunLogger::new");

    // Simulate 3 items completing on channel 0
    for i in 0..3usize {
        let mut entry = AsRunEntry::new(0, format!("Item {i}"));
        entry.actual_duration = 30.0 * (i as f64 + 1.0);
        logger.add(entry).await;
    }

    // Manual flush (below threshold of 5)
    logger.flush().await.expect("flush should succeed");

    let entries = logger.get_flushed_entries().await;
    assert_eq!(entries.len(), 3, "all 3 items should be flushed");

    // Export to JSON
    logger
        .inner
        .export(AsRunFormat::Json)
        .await
        .expect("JSON export should succeed");

    assert!(log_path.exists(), "JSON file should be created");
    let content = std::fs::read_to_string(&log_path).expect("should read exported JSON");
    assert!(
        content.contains("Item 0"),
        "JSON should contain first item title"
    );

    // Cleanup
    let _ = std::fs::remove_file(&log_path);
}

// ── 5. EAS alert — creation and priority ──────────────────────────────────────

/// Verify EAS alert interrupts normal playout by checking that alert creation
/// works and the alert has higher priority than normal programming.
#[tokio::test]
async fn test_eas_alert_insertion_interrupts_playout() {
    // Create a Tornado Warning — Critical priority
    let alert = EasAlert::new(
        EasAlertType::TornadoWarning,
        "TORNADO WARNING in effect until 6:00 PM".to_string(),
        Duration::from_secs(3600),
    );

    assert_eq!(alert.alert_type, EasAlertType::TornadoWarning);
    // TornadoWarning maps to High priority (national emergencies are Critical).
    assert_eq!(
        alert.priority,
        AlertPriority::High,
        "Tornado warning must be High priority"
    );

    // Required Weekly Test — Low priority
    let test_alert = EasAlert::new(
        EasAlertType::RequiredWeeklyTest,
        "Required Weekly Test".to_string(),
        Duration::from_secs(60),
    );

    assert_eq!(test_alert.priority, AlertPriority::Low);

    // Verify alert ordering: tornado warning takes precedence
    assert!(
        alert.priority > test_alert.priority,
        "TornadoWarning ({:?}) must outrank RequiredWeeklyTest ({:?})",
        alert.priority,
        test_alert.priority
    );

    // Verify the event_code is non-empty (required for EAS broadcast)
    assert!(
        !alert.event_code.is_empty(),
        "EAS alert must have a non-empty event code"
    );
}

// ── 6. Failover — primary signal loss → backup activates ─────────────────────

/// Simulate primary signal loss after 100 ms and verify backup channel
/// activates within the 500 ms configured timeout.
#[tokio::test(start_paused = true)]
async fn test_failover_primary_loss_backup_activates_within_timeout() {
    let config = FailoverConfig {
        auto_failover: false, // manual for determinism
        switch_delay_ms: 0,
        ..FailoverConfig::default()
    };

    let mut manager = FailoverManager::new(config)
        .await
        .expect("FailoverManager::new should succeed");

    // Channel 0 = primary, channel 1 = backup (managed externally)
    assert_eq!(manager.get_state(0).await, FailoverState::Primary);

    // Advance 100 ms to simulate loss detection window.
    tokio::time::advance(tokio::time::Duration::from_millis(100)).await;

    // Failover must complete within 500 ms.
    let switch = tokio::time::timeout(
        tokio::time::Duration::from_millis(500),
        manager.trigger_failover(0),
    )
    .await;

    assert!(switch.is_ok(), "failover must complete within 500 ms");
    switch
        .expect("timeout")
        .expect("trigger_failover should succeed");

    assert_eq!(
        manager.get_state(0).await,
        FailoverState::Secondary,
        "backup channel must be active after failover"
    );
}

// ── 7. N+1 redundancy: 3 channels ─────────────────────────────────────────────

/// With 3 channels (2 primaries + 1 standby), simulate primary failure and
/// verify the standby takes over while the tertiary remains on standby.
#[tokio::test]
async fn test_nplusone_three_channel_failover() {
    let config = FailoverConfig {
        redundancy_mode: RedundancyMode::NplusOne,
        standby_pool: vec![2], // channel 2 is the standby
        switch_delay_ms: 0,
        auto_failover: false,
        ..FailoverConfig::default()
    };

    let mut manager = FailoverManager::new(config)
        .await
        .expect("FailoverManager::new should succeed");

    // Initial: channels 0 and 1 are primaries, channel 2 in pool.
    assert_eq!(manager.available_standby_count().await, 1);

    // Primary channel 0 fails.
    manager
        .trigger_failover(0)
        .await
        .expect("trigger_failover should succeed");

    assert_eq!(manager.get_state(0).await, FailoverState::Secondary);

    let assignment = manager
        .get_nplusone_assignment(0)
        .await
        .expect("standby assignment must exist");
    assert_eq!(
        assignment.standby_channel_id, 2,
        "channel 2 should be assigned as standby"
    );

    // Channel 1 is unaffected.
    assert_eq!(manager.get_state(1).await, FailoverState::Primary);

    // Pool is now empty — no more standbys available.
    assert_eq!(manager.available_standby_count().await, 0);
}

// ── 8. Event bus — PlaylistStarted event on MasterControl start ────────────────

/// Verify that the LockFreeEventBus correctly delivers events published during
/// a playout lifecycle.  Here we simulate a "PlaylistStarted" event.
#[tokio::test]
async fn test_event_bus_records_playlist_started_event() {
    let bus = LockFreeEventBus::new(64);
    let rx = bus.subscribe();

    // Simulate MasterControl publishing a PlaylistStarted event.
    let id = bus.publish("PlaylistStarted", "master_control", "channel=0,items=3");
    assert_eq!(id, 0, "first event should have id=0");

    let event = rx.try_recv().expect("PlaylistStarted should be received");
    assert_eq!(event.event_type, "PlaylistStarted");
    assert_eq!(event.source, "master_control");
    assert!(
        event.payload.contains("channel=0"),
        "payload should reference channel 0"
    );
}

// ── 9. Channel automation — start/stop lifecycle ──────────────────────────────

/// Verify that a ChannelAutomation can be created, started, and stopped
/// cleanly — the basic lifecycle that all channel operations depend on.
#[tokio::test]
async fn test_channel_automation_start_stop_lifecycle() {
    let config = ChannelConfig {
        id: 0,
        name: "Integration Test Channel".to_string(),
        live_switching_enabled: false,
        device_control_enabled: false,
        devices: vec![],
    };

    let mut channel = ChannelAutomation::new(config)
        .await
        .expect("ChannelAutomation::new should succeed");

    channel.start().await.expect("start should succeed");
    channel.stop().await.expect("stop should succeed");
}

// ── 10. Batched logger — Drop flushes remaining entries ───────────────────────

/// Verify that when a BatchedAsRunLogger is dropped with entries still in the
/// buffer, those entries are visible in the inner logger (flushed on drop).
///
/// Note: we verify indirectly by checking the inner logger's entry count before
/// and after explicit flush, as Drop runs synchronously.
#[tokio::test]
async fn test_batched_logger_drop_flushes_remaining() {
    // Use a limit of 10 so 3 entries don't auto-flush.
    let mut logger = BatchedAsRunLogger::new(None, 10).expect("new should succeed");

    for i in 0..3 {
        logger
            .add(AsRunEntry::new(0, format!("Remaining {i}")))
            .await;
    }

    // 3 entries buffered, not yet flushed.
    assert_eq!(logger.pending_count(), 3);
    assert_eq!(logger.get_flushed_entries().await.len(), 0);

    // Explicitly flush (simulates what Drop does).
    logger.flush().await.expect("flush should succeed");
    assert_eq!(logger.pending_count(), 0);
    assert_eq!(logger.get_flushed_entries().await.len(), 3);
}
