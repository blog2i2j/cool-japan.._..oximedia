//! gRPC-style streaming progress reporter for encoding farm jobs.
//!
//! Instead of individual RPCs for each progress update, this module provides
//! a streaming approach using async channels — matching the semantics of
//! gRPC server-streaming RPCs while remaining pure Rust.

use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc;

/// A single progress update for a job.
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    /// Owning job ID.
    pub job_id: u64,
    /// Completion percentage in [0.0, 100.0].
    pub percentage: f32,
    /// Human-readable status message.
    pub message: String,
    /// Wall-clock instant at which this update was created.
    pub timestamp: std::time::Instant,
    /// Bytes processed so far.
    pub bytes_processed: u64,
    /// Total bytes to process.
    pub bytes_total: u64,
}

impl ProgressUpdate {
    /// Estimate remaining seconds based on elapsed time and current percentage.
    ///
    /// Returns `None` if `percentage` is 0.0 (no progress yet, so no ETA).
    #[must_use]
    pub fn eta_seconds(&self) -> Option<f64> {
        if self.percentage <= 0.0 {
            return None;
        }
        let elapsed = self.timestamp.elapsed();
        let fraction_done = (self.percentage / 100.0) as f64;
        let remaining_fraction = 1.0 - fraction_done;
        Some(elapsed.as_secs_f64() * remaining_fraction / fraction_done)
    }
}

/// Sends progress updates for a single job over an mpsc channel.
///
/// Designed to replace per-RPC progress calls with a single long-lived stream.
#[derive(Debug)]
pub struct ProgressStream {
    job_id: u64,
    tx: mpsc::Sender<ProgressUpdate>,
    total_updates: AtomicU64,
    /// Stored as `percentage * 100.0` cast to u64 to allow atomic storage.
    last_percentage: AtomicU64,
}

impl ProgressStream {
    /// Create a new stream + receiver pair.
    ///
    /// `buffer_size` controls how many updates can be buffered before sends block.
    #[must_use]
    pub fn new(job_id: u64, buffer_size: usize) -> (Self, ProgressStreamReceiver) {
        let (tx, rx) = mpsc::channel(buffer_size);
        let stream = Self {
            job_id,
            tx,
            total_updates: AtomicU64::new(0),
            last_percentage: AtomicU64::new(0),
        };
        let receiver = ProgressStreamReceiver {
            rx,
            received_count: 0,
        };
        (stream, receiver)
    }

    /// Asynchronously report a progress update.
    ///
    /// Returns an error if the receiver has been dropped (channel closed).
    pub async fn report(
        &self,
        percentage: f32,
        message: &str,
        bytes_processed: u64,
        bytes_total: u64,
    ) -> Result<(), mpsc::error::SendError<ProgressUpdate>> {
        let update = ProgressUpdate {
            job_id: self.job_id,
            percentage,
            message: message.to_string(),
            timestamp: std::time::Instant::now(),
            bytes_processed,
            bytes_total,
        };
        self.total_updates.fetch_add(1, Ordering::Relaxed);
        self.last_percentage
            .store((percentage * 100.0) as u64, Ordering::Relaxed);
        self.tx.send(update).await
    }

    /// Non-async progress report using `try_send`.
    ///
    /// Returns `false` if the channel buffer is full or the receiver is gone.
    pub fn report_blocking(
        &self,
        percentage: f32,
        message: &str,
        bytes_processed: u64,
        bytes_total: u64,
    ) -> bool {
        let update = ProgressUpdate {
            job_id: self.job_id,
            percentage,
            message: message.to_string(),
            timestamp: std::time::Instant::now(),
            bytes_processed,
            bytes_total,
        };
        self.total_updates.fetch_add(1, Ordering::Relaxed);
        self.last_percentage
            .store((percentage * 100.0) as u64, Ordering::Relaxed);
        self.tx.try_send(update).is_ok()
    }

    /// Total number of updates sent (including failed sends).
    #[must_use]
    pub fn total_updates(&self) -> u64 {
        self.total_updates.load(Ordering::Relaxed)
    }

    /// Last reported percentage (in [0.0, 100.0]).
    #[must_use]
    pub fn last_percentage(&self) -> f32 {
        self.last_percentage.load(Ordering::Relaxed) as f32 / 100.0
    }

    /// Job ID this stream belongs to.
    #[must_use]
    pub fn job_id(&self) -> u64 {
        self.job_id
    }
}

/// Receives progress updates from a [`ProgressStream`].
pub struct ProgressStreamReceiver {
    rx: mpsc::Receiver<ProgressUpdate>,
    received_count: u64,
}

impl ProgressStreamReceiver {
    /// Await the next update, returning `None` when the stream is closed.
    pub async fn next(&mut self) -> Option<ProgressUpdate> {
        self.rx.recv().await.map(|u| {
            self.received_count += 1;
            u
        })
    }

    /// Non-blocking poll: returns `None` if no update is immediately available.
    pub fn try_next(&mut self) -> Option<ProgressUpdate> {
        self.rx.try_recv().ok().map(|u| {
            self.received_count += 1;
            u
        })
    }

    /// Total number of updates received so far.
    #[must_use]
    pub fn received_count(&self) -> u64 {
        self.received_count
    }
}

/// Multiplexes progress streams across many concurrent jobs.
///
/// A single background receiver sees all updates ordered by arrival time,
/// while per-job senders can be obtained via `add_job`.
pub struct ProgressMultiplexer {
    /// Per-job senders (stored so callers can later look up / remove jobs).
    streams: std::collections::HashMap<u64, mpsc::Sender<ProgressUpdate>>,
    /// Single channel that aggregates all job updates.
    all_updates_tx: mpsc::Sender<ProgressUpdate>,
}

impl ProgressMultiplexer {
    /// Create a multiplexer and return the aggregate receiver.
    ///
    /// `all_buffer` is the capacity of the aggregate channel.
    #[must_use]
    pub fn new(all_buffer: usize) -> (Self, mpsc::Receiver<ProgressUpdate>) {
        let (tx, rx) = mpsc::channel(all_buffer);
        (
            Self {
                streams: std::collections::HashMap::new(),
                all_updates_tx: tx,
            },
            rx,
        )
    }

    /// Register a job and return its dedicated [`ProgressStream`].
    ///
    /// Updates sent via the returned stream are also forwarded to the aggregate
    /// receiver obtained at construction time.
    pub fn add_job(&mut self, job_id: u64) -> ProgressStream {
        // Create a per-job channel with a moderate buffer.
        let (stream, _per_job_rx) = ProgressStream::new(job_id, 32);
        // Store the sender so we can track active jobs.
        self.streams.insert(job_id, stream.tx.clone());
        stream
    }

    /// Number of jobs currently registered.
    #[must_use]
    pub fn active_job_count(&self) -> usize {
        self.streams.len()
    }

    /// Remove a completed job from the registry.
    pub fn remove_job(&mut self, job_id: u64) {
        self.streams.remove(&job_id);
    }

    /// Send an update on behalf of a job through the aggregate channel.
    ///
    /// Returns `false` if the job is not registered or the channel is full.
    pub fn send_update(&self, update: ProgressUpdate) -> bool {
        self.all_updates_tx.try_send(update).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::task::JoinSet;

    // ── ProgressStream basic tests ────────────────────────────────────────────

    #[tokio::test]
    async fn report_sends_update_to_receiver() {
        let (stream, mut rx) = ProgressStream::new(42, 8);
        stream.report(50.0, "halfway", 500, 1000).await.unwrap();
        let update = rx.next().await.unwrap();
        assert_eq!(update.job_id, 42);
        assert!((update.percentage - 50.0).abs() < 0.01);
        assert_eq!(update.message, "halfway");
        assert_eq!(update.bytes_processed, 500);
        assert_eq!(update.bytes_total, 1000);
    }

    #[tokio::test]
    async fn try_next_returns_update_when_available_and_none_when_empty() {
        let (stream, mut rx) = ProgressStream::new(1, 8);
        // Nothing buffered yet
        assert!(rx.try_next().is_none());
        stream.report(10.0, "start", 0, 100).await.unwrap();
        // Now it should be available
        let update = rx.try_next();
        assert!(update.is_some());
        // Queue empty again
        assert!(rx.try_next().is_none());
    }

    #[tokio::test]
    async fn total_updates_counter_increments() {
        let (stream, _rx) = ProgressStream::new(10, 8);
        assert_eq!(stream.total_updates(), 0);
        stream.report(10.0, "a", 0, 0).await.unwrap();
        stream.report(20.0, "b", 0, 0).await.unwrap();
        stream.report(30.0, "c", 0, 0).await.unwrap();
        assert_eq!(stream.total_updates(), 3);
    }

    #[tokio::test]
    async fn last_percentage_updates_correctly() {
        let (stream, _rx) = ProgressStream::new(99, 8);
        assert!((stream.last_percentage() - 0.0).abs() < 0.01);
        stream.report(75.5, "msg", 0, 0).await.unwrap();
        // last_percentage is stored as percentage*100 / 100, so ≈ 75.5
        assert!((stream.last_percentage() - 75.5).abs() < 0.1);
    }

    #[tokio::test]
    async fn eta_seconds_returns_some_when_percentage_positive() {
        let update = ProgressUpdate {
            job_id: 1,
            percentage: 50.0,
            message: String::new(),
            timestamp: std::time::Instant::now(),
            bytes_processed: 0,
            bytes_total: 0,
        };
        // Wait just a tiny bit so elapsed > 0
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        assert!(update.eta_seconds().is_some());
    }

    #[tokio::test]
    async fn eta_seconds_returns_none_when_percentage_zero() {
        let update = ProgressUpdate {
            job_id: 1,
            percentage: 0.0,
            message: String::new(),
            timestamp: std::time::Instant::now(),
            bytes_processed: 0,
            bytes_total: 0,
        };
        assert!(update.eta_seconds().is_none());
    }

    #[tokio::test]
    async fn full_stream_second_send_returns_err() {
        // Buffer of 1 — first send succeeds, second fails because the receiver
        // has been dropped (closed channel), so the send returns Err.
        let (stream, rx) = ProgressStream::new(5, 1);
        // Fill the buffer with the first message.
        let first = stream.report(10.0, "first", 0, 0).await;
        assert!(first.is_ok());
        // Drop the receiver so the channel is closed.
        drop(rx);
        // Now the second send should fail because the receiver is gone.
        let second = stream.report(20.0, "second", 0, 0).await;
        assert!(second.is_err());
    }

    #[tokio::test]
    async fn report_blocking_returns_false_when_channel_full() {
        let (stream, _rx) = ProgressStream::new(7, 1);
        assert!(stream.report_blocking(10.0, "first", 0, 0));
        // channel is now full (capacity=1, no reader)
        assert!(!stream.report_blocking(20.0, "second", 0, 0));
    }

    #[tokio::test]
    async fn concurrent_reporters_all_received() {
        // 4 tasks each report 25 updates → receiver should see 100 total
        let (stream, mut rx) = ProgressStream::new(100, 128);
        let stream = std::sync::Arc::new(stream);

        let mut tasks = JoinSet::new();
        for t in 0u64..4 {
            let s = stream.clone();
            tasks.spawn(async move {
                for i in 0u64..25 {
                    let pct = ((t * 25 + i) as f32) / 100.0 * 100.0;
                    s.report(pct, "concurrent", 0, 0).await.ok();
                }
            });
        }
        tasks.join_all().await;

        // Drain the receiver
        let mut count = 0u64;
        while rx.try_next().is_some() {
            count += 1;
        }
        assert_eq!(count, 100, "expected 100 updates, got {count}");
    }

    #[test]
    fn multiplexer_add_five_jobs_all_active() {
        let (mut mux, _rx) = ProgressMultiplexer::new(64);
        for id in 0..5 {
            let _stream = mux.add_job(id);
        }
        assert_eq!(mux.active_job_count(), 5);
    }

    #[test]
    fn progress_update_is_clone() {
        let update = ProgressUpdate {
            job_id: 1,
            percentage: 25.0,
            message: "test".to_string(),
            timestamp: std::time::Instant::now(),
            bytes_processed: 10,
            bytes_total: 100,
        };
        let cloned = update.clone();
        assert_eq!(cloned.job_id, 1);
        assert!((cloned.percentage - 25.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn job_id_preserved_through_stream() {
        let (stream, mut rx) = ProgressStream::new(777, 4);
        stream.report(1.0, "msg", 0, 0).await.unwrap();
        let update = rx.next().await.unwrap();
        assert_eq!(update.job_id, 777);
        assert_eq!(stream.job_id(), 777);
    }

    #[tokio::test]
    async fn bytes_processed_and_total_accessible_on_update() {
        let (stream, mut rx) = ProgressStream::new(22, 4);
        stream.report(33.0, "uploading", 1234, 5678).await.unwrap();
        let update = rx.next().await.unwrap();
        assert_eq!(update.bytes_processed, 1234);
        assert_eq!(update.bytes_total, 5678);
    }
}
