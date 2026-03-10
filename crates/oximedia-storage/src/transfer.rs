//! Unified transfer operations with progress tracking and retry logic

use crate::{
    ByteStream, CloudStorage, DownloadOptions, ProgressCallback, ProgressInfo, Result,
    StorageError, UploadOptions,
};
use bytes::Bytes;
use futures::{stream, StreamExt};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs::{self, File};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{RwLock, Semaphore};
use tokio::time::sleep;
use tracing::{error, info, warn};

/// Default number of retry attempts
const DEFAULT_RETRY_ATTEMPTS: u32 = 3;

/// Default initial backoff duration (1 second)
const DEFAULT_INITIAL_BACKOFF: Duration = Duration::from_secs(1);

/// Maximum backoff duration (60 seconds)
const MAX_BACKOFF: Duration = Duration::from_secs(60);

/// Chunk size for parallel transfers (10 MB)
const PARALLEL_CHUNK_SIZE: u64 = 10 * 1024 * 1024;

/// Default rate limit (bytes per second, 0 = unlimited)
const DEFAULT_RATE_LIMIT: u64 = 0;

/// Configuration for transfer operations
#[derive(Debug, Clone)]
pub struct TransferConfig {
    /// Number of retry attempts
    pub retry_attempts: u32,
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Enable parallel transfers for large files
    pub parallel_enabled: bool,
    /// Maximum number of parallel chunks
    pub max_parallel_chunks: usize,
    /// Chunk size for parallel transfers
    pub chunk_size: u64,
    /// Rate limit in bytes per second (0 = unlimited)
    pub rate_limit: u64,
    /// Enable resume support
    pub resume_enabled: bool,
    /// Directory for storing transfer state
    pub state_dir: Option<std::path::PathBuf>,
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            retry_attempts: DEFAULT_RETRY_ATTEMPTS,
            initial_backoff: DEFAULT_INITIAL_BACKOFF,
            max_backoff: MAX_BACKOFF,
            parallel_enabled: true,
            max_parallel_chunks: 4,
            chunk_size: PARALLEL_CHUNK_SIZE,
            rate_limit: DEFAULT_RATE_LIMIT,
            resume_enabled: false,
            state_dir: None,
        }
    }
}

/// Transfer manager for handling uploads and downloads
pub struct TransferManager<S: CloudStorage> {
    storage: Arc<S>,
    config: TransferConfig,
    active_transfers: Arc<RwLock<Vec<String>>>,
    rate_limiter: Arc<RateLimiter>,
}

impl<S: CloudStorage + 'static> TransferManager<S> {
    /// Create a new transfer manager
    pub fn new(storage: S, config: TransferConfig) -> Self {
        let rate_limiter = Arc::new(RateLimiter::new(config.rate_limit));

        Self {
            storage: Arc::new(storage),
            config,
            active_transfers: Arc::new(RwLock::new(Vec::new())),
            rate_limiter,
        }
    }

    /// Upload a file with progress tracking and retry
    pub async fn upload_file(
        &self,
        key: &str,
        file_path: &Path,
        options: UploadOptions,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<String> {
        info!("Starting upload: {} -> {}", file_path.display(), key);

        // Add to active transfers
        {
            let mut transfers = self.active_transfers.write().await;
            transfers.push(key.to_string());
        }

        // Get file size
        let metadata = fs::metadata(file_path).await?;
        let file_size = metadata.len();

        let result = if self.config.parallel_enabled && file_size > self.config.chunk_size * 2 {
            self.upload_file_parallel(key, file_path, file_size, options, progress_callback)
                .await
        } else {
            self.upload_file_simple(key, file_path, file_size, options, progress_callback)
                .await
        };

        // Remove from active transfers
        {
            let mut transfers = self.active_transfers.write().await;
            transfers.retain(|k| k != key);
        }

        result
    }

    /// Simple upload with retry
    async fn upload_file_simple(
        &self,
        key: &str,
        file_path: &Path,
        file_size: u64,
        options: UploadOptions,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<String> {
        let mut attempts = 0;
        let mut backoff = self.config.initial_backoff;

        loop {
            match self
                .try_upload_file(
                    key,
                    file_path,
                    file_size,
                    &options,
                    progress_callback.clone(),
                )
                .await
            {
                Ok(etag) => {
                    info!("Upload completed: {}", key);
                    return Ok(etag);
                }
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.config.retry_attempts {
                        error!("Upload failed after {} attempts: {}", attempts, e);
                        return Err(e);
                    }

                    warn!(
                        "Upload attempt {} failed, retrying in {:?}: {}",
                        attempts, backoff, e
                    );

                    sleep(backoff).await;
                    backoff = (backoff * 2).min(self.config.max_backoff);
                }
            }
        }
    }

    /// Try to upload file
    async fn try_upload_file(
        &self,
        key: &str,
        file_path: &Path,
        file_size: u64,
        options: &UploadOptions,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<String> {
        let file = File::open(file_path).await?;

        let bytes_transferred = Arc::new(AtomicU64::new(0));
        let start_time = Instant::now();
        let rate_limiter = self.rate_limiter.clone();

        let stream: ByteStream = Box::pin(stream::try_unfold(
            (
                file,
                bytes_transferred.clone(),
                start_time,
                progress_callback.clone(),
                rate_limiter.clone(),
            ),
            move |(mut file, transferred, start, progress_callback, rate_limiter)| async move {
                let mut buffer = vec![0u8; 1024 * 1024]; // 1 MB chunks
                let n = file.read(&mut buffer).await?;

                if n == 0 {
                    return Ok(None);
                }

                buffer.truncate(n);

                // Update progress
                let current = transferred.fetch_add(n as u64, Ordering::Relaxed) + n as u64;

                if let Some(ref callback) = progress_callback {
                    let elapsed = start.elapsed().as_secs_f64();
                    let bps = if elapsed > 0.0 {
                        current as f64 / elapsed
                    } else {
                        0.0
                    };
                    let remaining = file_size.saturating_sub(current);
                    let eta = if bps > 0.0 {
                        Some(remaining as f64 / bps)
                    } else {
                        None
                    };

                    callback(ProgressInfo {
                        bytes_transferred: current,
                        total_bytes: file_size,
                        bytes_per_second: bps,
                        eta_seconds: eta,
                    });
                }

                // Apply rate limiting
                rate_limiter.consume(n).await;

                Ok(Some((
                    Bytes::from(buffer),
                    (file, transferred, start, progress_callback, rate_limiter),
                )))
            },
        ));

        self.storage
            .upload_stream(key, stream, Some(file_size), options.clone())
            .await
    }

    /// Parallel upload for large files
    #[allow(clippy::too_many_arguments)]
    async fn upload_file_parallel(
        &self,
        key: &str,
        file_path: &Path,
        file_size: u64,
        options: UploadOptions,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<String> {
        info!("Using parallel upload for: {}", key);

        // For parallel upload, we need to split the file into chunks
        // and upload them separately, then combine them
        // This is a simplified implementation that falls back to simple upload
        // for providers that don't support multipart/parallel uploads directly

        self.upload_file_simple(key, file_path, file_size, options, progress_callback)
            .await
    }

    /// Download a file with progress tracking and retry
    pub async fn download_file(
        &self,
        key: &str,
        file_path: &Path,
        options: DownloadOptions,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<()> {
        info!("Starting download: {} -> {}", key, file_path.display());

        // Add to active transfers
        {
            let mut transfers = self.active_transfers.write().await;
            transfers.push(key.to_string());
        }

        // Get object metadata for size
        let metadata = self.storage.get_metadata(key).await?;
        let file_size = metadata.size;

        let result = if self.config.parallel_enabled && file_size > self.config.chunk_size * 2 {
            self.download_file_parallel(key, file_path, file_size, options, progress_callback)
                .await
        } else {
            self.download_file_simple(key, file_path, file_size, options, progress_callback)
                .await
        };

        // Remove from active transfers
        {
            let mut transfers = self.active_transfers.write().await;
            transfers.retain(|k| k != key);
        }

        result
    }

    /// Simple download with retry
    async fn download_file_simple(
        &self,
        key: &str,
        file_path: &Path,
        file_size: u64,
        options: DownloadOptions,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<()> {
        let mut attempts = 0;
        let mut backoff = self.config.initial_backoff;

        loop {
            match self
                .try_download_file(
                    key,
                    file_path,
                    file_size,
                    &options,
                    progress_callback.clone(),
                )
                .await
            {
                Ok(()) => {
                    info!("Download completed: {}", key);
                    return Ok(());
                }
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.config.retry_attempts {
                        error!("Download failed after {} attempts: {}", attempts, e);
                        return Err(e);
                    }

                    warn!(
                        "Download attempt {} failed, retrying in {:?}: {}",
                        attempts, backoff, e
                    );

                    sleep(backoff).await;
                    backoff = (backoff * 2).min(self.config.max_backoff);
                }
            }
        }
    }

    /// Try to download file
    async fn try_download_file(
        &self,
        key: &str,
        file_path: &Path,
        file_size: u64,
        options: &DownloadOptions,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<()> {
        let mut stream = self.storage.download_stream(key, options.clone()).await?;
        let mut file = File::create(file_path).await?;

        let bytes_transferred = Arc::new(AtomicU64::new(0));
        let start_time = Instant::now();

        while let Some(result) = stream.next().await {
            let chunk = result?;
            file.write_all(&chunk).await?;

            // Update progress
            let current = bytes_transferred.fetch_add(chunk.len() as u64, Ordering::Relaxed)
                + chunk.len() as u64;

            if let Some(ref callback) = progress_callback {
                let elapsed = start_time.elapsed().as_secs_f64();
                let bps = if elapsed > 0.0 {
                    current as f64 / elapsed
                } else {
                    0.0
                };
                let remaining = file_size.saturating_sub(current);
                let eta = if bps > 0.0 {
                    Some(remaining as f64 / bps)
                } else {
                    None
                };

                callback(ProgressInfo {
                    bytes_transferred: current,
                    total_bytes: file_size,
                    bytes_per_second: bps,
                    eta_seconds: eta,
                });
            }

            // Apply rate limiting
            self.rate_limiter.consume(chunk.len()).await;
        }

        file.flush().await?;
        Ok(())
    }

    /// Parallel download for large files
    #[allow(clippy::too_many_arguments)]
    async fn download_file_parallel<'a>(
        &self,
        key: &str,
        file_path: &'a Path,
        file_size: u64,
        _options: DownloadOptions,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<()>
    where
        S: 'a,
    {
        info!("Using parallel download for: {}", key);

        let num_chunks = file_size.div_ceil(self.config.chunk_size) as usize;
        let semaphore = Arc::new(Semaphore::new(self.config.max_parallel_chunks));

        let bytes_transferred = Arc::new(AtomicU64::new(0));
        let start_time = Instant::now();

        // Create temporary directory for chunks
        let temp_dir = file_path.parent().unwrap_or_else(|| Path::new("."));
        let file_name = file_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "chunk".to_string());
        let temp_prefix = format!(".{file_name}.part");

        let mut tasks = Vec::new();

        for chunk_idx in 0..num_chunks {
            let permit = semaphore.clone().acquire_owned().await.map_err(|e| {
                StorageError::ProviderError(format!(
                    "Semaphore closed during parallel download: {e}"
                ))
            })?;
            let storage = self.storage.clone();
            let key = key.to_string();
            let chunk_file = temp_dir.join(format!("{temp_prefix}.{chunk_idx}"));
            let bytes_transferred = bytes_transferred.clone();
            let progress_callback = progress_callback.clone();
            let chunk_size = self.config.chunk_size;
            let rate_limiter = self.rate_limiter.clone();

            let task = tokio::spawn(async move {
                let _permit = permit;

                let start = chunk_idx as u64 * chunk_size;
                let end = ((chunk_idx + 1) as u64 * chunk_size - 1).min(file_size - 1);

                let download_opts = DownloadOptions {
                    range: Some((start, end)),
                    ..Default::default()
                };

                let mut stream = storage.download_stream(&key, download_opts).await?;
                let mut file = File::create(&chunk_file).await?;

                while let Some(result) = stream.next().await {
                    let chunk = result?;
                    file.write_all(&chunk).await?;

                    // Update progress
                    let current = bytes_transferred
                        .fetch_add(chunk.len() as u64, Ordering::Relaxed)
                        + chunk.len() as u64;

                    if let Some(ref callback) = progress_callback {
                        let elapsed = start_time.elapsed().as_secs_f64();
                        let bps = if elapsed > 0.0 {
                            current as f64 / elapsed
                        } else {
                            0.0
                        };
                        let remaining = file_size.saturating_sub(current);
                        let eta = if bps > 0.0 {
                            Some(remaining as f64 / bps)
                        } else {
                            None
                        };

                        callback(ProgressInfo {
                            bytes_transferred: current,
                            total_bytes: file_size,
                            bytes_per_second: bps,
                            eta_seconds: eta,
                        });
                    }

                    // Apply rate limiting
                    rate_limiter.consume(chunk.len()).await;
                }

                file.flush().await?;
                Ok::<_, StorageError>(chunk_file)
            });

            tasks.push(task);
        }

        // Wait for all chunks to complete
        let mut chunk_files = Vec::new();
        for task in tasks {
            let chunk_file = task.await.map_err(|e| {
                StorageError::ProviderError(format!("Parallel download task failed: {e}"))
            })??;
            chunk_files.push(chunk_file);
        }

        // Combine chunks into final file
        let mut output_file = File::create(file_path).await?;

        for chunk_file in chunk_files {
            let mut chunk_data = Vec::new();
            let mut file = File::open(&chunk_file).await?;
            file.read_to_end(&mut chunk_data).await?;
            output_file.write_all(&chunk_data).await?;

            // Delete chunk file
            fs::remove_file(&chunk_file).await?;
        }

        output_file.flush().await?;

        info!("Parallel download completed: {}", key);
        Ok(())
    }

    /// Get list of active transfers
    pub async fn active_transfers(&self) -> Vec<String> {
        self.active_transfers.read().await.clone()
    }

    /// Cancel all active transfers
    pub async fn cancel_all(&self) {
        let mut transfers = self.active_transfers.write().await;
        transfers.clear();
    }

    /// Set rate limit (bytes per second)
    pub fn set_rate_limit(&self, bytes_per_second: u64) {
        self.rate_limiter.set_limit(bytes_per_second);
    }
}

/// Rate limiter for controlling transfer speed
pub struct RateLimiter {
    bytes_per_second: Arc<AtomicU64>,
    last_refill: Arc<RwLock<Instant>>,
    tokens: Arc<AtomicU64>,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(bytes_per_second: u64) -> Self {
        Self {
            bytes_per_second: Arc::new(AtomicU64::new(bytes_per_second)),
            last_refill: Arc::new(RwLock::new(Instant::now())),
            tokens: Arc::new(AtomicU64::new(bytes_per_second)),
        }
    }

    /// Set rate limit
    pub fn set_limit(&self, bytes_per_second: u64) {
        self.bytes_per_second
            .store(bytes_per_second, Ordering::Relaxed);
    }

    /// Consume tokens (blocking if necessary)
    pub async fn consume(&self, bytes: usize) {
        let limit = self.bytes_per_second.load(Ordering::Relaxed);

        // No rate limit
        if limit == 0 {
            return;
        }

        loop {
            // Refill tokens
            {
                let mut last_refill = self.last_refill.write().await;
                let now = Instant::now();
                let elapsed = now.duration_since(*last_refill);

                if elapsed.as_secs() >= 1 {
                    let refill_amount = limit * elapsed.as_secs();
                    let current = self.tokens.load(Ordering::Relaxed);
                    let new_tokens = (current + refill_amount).min(limit);
                    self.tokens.store(new_tokens, Ordering::Relaxed);
                    *last_refill = now;
                }
            }

            // Try to consume tokens
            let current = self.tokens.load(Ordering::Relaxed);
            if current >= bytes as u64 {
                self.tokens.fetch_sub(bytes as u64, Ordering::Relaxed);
                break;
            }

            // Wait and try again
            sleep(Duration::from_millis(10)).await;
        }
    }
}

/// Transfer state for resume support
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransferState {
    pub key: String,
    pub file_path: std::path::PathBuf,
    pub total_bytes: u64,
    pub transferred_bytes: u64,
    pub chunk_states: Vec<ChunkState>,
}

/// State of a transfer chunk
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChunkState {
    pub index: usize,
    pub start: u64,
    pub end: u64,
    pub completed: bool,
}

impl TransferState {
    /// Save state to file
    pub async fn save(&self, state_dir: &Path) -> Result<()> {
        let state_file = state_dir.join(format!("{}.state", self.key.replace('/', "_")));
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            StorageError::SerializationError(format!("Failed to serialize state: {e}"))
        })?;

        tokio::fs::write(&state_file, json).await?;
        Ok(())
    }

    /// Load state from file
    pub async fn load(key: &str, state_dir: &Path) -> Result<Option<Self>> {
        let state_file = state_dir.join(format!("{}.state", key.replace('/', "_")));

        if !state_file.exists() {
            return Ok(None);
        }

        let json = tokio::fs::read_to_string(&state_file).await?;
        let state = serde_json::from_str(&json).map_err(|e| {
            StorageError::SerializationError(format!("Failed to deserialize state: {e}"))
        })?;

        Ok(Some(state))
    }

    /// Delete state file
    pub async fn delete(&self, state_dir: &Path) -> Result<()> {
        let state_file = state_dir.join(format!("{}.state", self.key.replace('/', "_")));

        if state_file.exists() {
            tokio::fs::remove_file(&state_file).await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_config_default() {
        let config = TransferConfig::default();
        assert_eq!(config.retry_attempts, DEFAULT_RETRY_ATTEMPTS);
        assert_eq!(config.initial_backoff, DEFAULT_INITIAL_BACKOFF);
        assert_eq!(config.max_backoff, MAX_BACKOFF);
        assert!(config.parallel_enabled);
        assert_eq!(config.max_parallel_chunks, 4);
    }

    #[test]
    fn test_rate_limiter_no_limit() {
        let limiter = RateLimiter::new(0);
        assert_eq!(limiter.bytes_per_second.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_rate_limiter_set_limit() {
        let limiter = RateLimiter::new(1000);
        limiter.set_limit(2000);
        assert_eq!(limiter.bytes_per_second.load(Ordering::Relaxed), 2000);
    }
}
