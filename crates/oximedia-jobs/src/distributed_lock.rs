#![allow(dead_code)]
//! Distributed lock coordination for multi-process job queue workers.
//!
//! This module provides a file-system-based distributed lock that prevents
//! multiple processes from concurrently executing the same critical section
//! (e.g. dequeuing the same job).  On a single machine it relies on atomic
//! file creation; for true cross-host coordination callers should back this
//! with a shared network filesystem or a database advisory lock.
//!
//! # Design
//! - `DistributedLock` uses a lock-file in a configurable directory.
//! - The lock file stores the owner's process ID and a unique token.
//! - `try_acquire` attempts atomic creation; `release` removes the file.
//! - `acquire_with_timeout` retries until timeout expires.
//! - Stale locks (older than `ttl`) are automatically broken.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

/// Error type for distributed lock operations.
#[derive(Debug, thiserror::Error)]
pub enum LockError {
    /// The lock is held by another owner.
    #[error("Lock is held by another owner: {0}")]
    AlreadyHeld(String),
    /// Timed out waiting to acquire the lock.
    #[error("Timed out waiting for lock: {0}")]
    Timeout(String),
    /// I/O error while manipulating the lock file.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// Lock file contents could not be parsed.
    #[error("Corrupt lock file: {0}")]
    CorruptFile(String),
}

/// Metadata stored inside a lock file.
#[derive(Debug, Clone)]
pub struct LockFileData {
    /// Identifier of the process/node that holds the lock.
    pub owner_id: String,
    /// Unique random token to detect refresh races.
    pub token: String,
    /// Unix timestamp (seconds) when the lock was acquired.
    pub acquired_at_secs: u64,
}

impl LockFileData {
    fn encode(&self) -> String {
        format!(
            "{}\n{}\n{}",
            self.owner_id, self.token, self.acquired_at_secs
        )
    }

    fn decode(raw: &str) -> Result<Self, LockError> {
        let mut lines = raw.lines();
        let owner_id = lines
            .next()
            .ok_or_else(|| LockError::CorruptFile("missing owner_id".to_string()))?
            .to_string();
        let token = lines
            .next()
            .ok_or_else(|| LockError::CorruptFile("missing token".to_string()))?
            .to_string();
        let acquired_at_secs: u64 = lines
            .next()
            .ok_or_else(|| LockError::CorruptFile("missing timestamp".to_string()))?
            .parse()
            .map_err(|e| LockError::CorruptFile(format!("bad timestamp: {e}")))?;
        Ok(Self {
            owner_id,
            token,
            acquired_at_secs,
        })
    }

    fn is_stale(&self, ttl: Duration) -> bool {
        let now_secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now_secs.saturating_sub(self.acquired_at_secs) > ttl.as_secs()
    }
}

/// A named distributed lock backed by the file system.
#[derive(Debug)]
pub struct DistributedLock {
    /// Path to the lock file.
    pub lock_file: PathBuf,
    /// Owner identifier (e.g. hostname + PID).
    pub owner_id: String,
    /// Time-to-live for stale lock detection.
    pub ttl: Duration,
    /// Token of the currently held lock (if any).
    current_token: Option<String>,
}

impl DistributedLock {
    /// Create a new lock targeting `lock_dir/lock_name.lock`.
    pub fn new(lock_dir: impl AsRef<Path>, lock_name: &str, owner_id: impl Into<String>) -> Self {
        let lock_file = lock_dir.as_ref().join(format!("{lock_name}.lock"));
        Self {
            lock_file,
            owner_id: owner_id.into(),
            ttl: Duration::from_secs(30),
            current_token: None,
        }
    }

    /// Set the stale-lock TTL.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Generate a short pseudo-random token using timestamp + owner hash.
    fn generate_token(&self) -> String {
        let ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
        for byte in self.owner_id.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
        }
        hash ^= ts as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        format!("{hash:016x}")
    }

    /// Attempt to acquire the lock immediately.
    ///
    /// Returns `Ok(())` on success, `Err(LockError::AlreadyHeld(_))` if another
    /// owner holds a non-stale lock, or an I/O error.
    pub fn try_acquire(&mut self) -> Result<(), LockError> {
        // Break stale lock if present.
        if let Ok(raw) = fs::read_to_string(&self.lock_file) {
            if let Ok(data) = LockFileData::decode(&raw) {
                if data.is_stale(self.ttl) {
                    // Break the stale lock
                    let _ = fs::remove_file(&self.lock_file);
                } else {
                    return Err(LockError::AlreadyHeld(data.owner_id));
                }
            } else {
                // Corrupt file — remove and retry
                let _ = fs::remove_file(&self.lock_file);
            }
        }

        // Attempt atomic exclusive create.
        let token = self.generate_token();
        let now_secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let data = LockFileData {
            owner_id: self.owner_id.clone(),
            token: token.clone(),
            acquired_at_secs: now_secs,
        };

        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true) // Atomic: fails if file exists
            .open(&self.lock_file)?;

        file.write_all(data.encode().as_bytes())?;
        file.flush()?;
        self.current_token = Some(token);
        Ok(())
    }

    /// Attempt to acquire the lock, retrying every `poll_interval` until `timeout` expires.
    pub fn acquire_with_timeout(
        &mut self,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<(), LockError> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.try_acquire() {
                Ok(()) => return Ok(()),
                Err(LockError::AlreadyHeld(_)) => {
                    if Instant::now() >= deadline {
                        return Err(LockError::Timeout(self.lock_file.display().to_string()));
                    }
                    std::thread::sleep(poll_interval);
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Release the lock.  Only releases if this instance holds it (matching token).
    pub fn release(&mut self) -> Result<(), LockError> {
        let Some(token) = self.current_token.take() else {
            return Ok(()); // Not held by us
        };
        // Verify we still own the lock before removing.
        if let Ok(raw) = fs::read_to_string(&self.lock_file) {
            if let Ok(data) = LockFileData::decode(&raw) {
                if data.token != token || data.owner_id != self.owner_id {
                    // Lock was taken over (e.g. stale-lock break by another process)
                    return Ok(());
                }
            }
        }
        fs::remove_file(&self.lock_file).or_else(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                Ok(()) // Already removed — fine
            } else {
                Err(LockError::Io(e))
            }
        })
    }

    /// Returns `true` if this instance currently holds the lock.
    #[must_use]
    pub fn is_held(&self) -> bool {
        self.current_token.is_some()
    }

    /// Read the current lock file data without acquiring the lock.
    pub fn peek(&self) -> Option<LockFileData> {
        fs::read_to_string(&self.lock_file)
            .ok()
            .and_then(|raw| LockFileData::decode(&raw).ok())
    }
}

impl Drop for DistributedLock {
    fn drop(&mut self) {
        if self.is_held() {
            let _ = self.release();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn tmp_dir() -> PathBuf {
        let d = temp_dir().join(format!(
            "oximedia_lock_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        ));
        fs::create_dir_all(&d).expect("temp dir creation");
        d
    }

    #[test]
    fn test_acquire_and_release() {
        let dir = tmp_dir();
        let mut lock = DistributedLock::new(&dir, "test", "owner-1");
        assert!(lock.try_acquire().is_ok());
        assert!(lock.is_held());
        assert!(lock.release().is_ok());
        assert!(!lock.is_held());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_acquire_twice_same_owner_fails() {
        let dir = tmp_dir();
        let mut lock1 = DistributedLock::new(&dir, "test", "owner-1");
        let mut lock2 = DistributedLock::new(&dir, "test", "owner-2");
        assert!(lock1.try_acquire().is_ok());
        let result = lock2.try_acquire();
        assert!(matches!(result, Err(LockError::AlreadyHeld(_))));
        let _ = lock1.release();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_release_without_acquire_is_noop() {
        let dir = tmp_dir();
        let mut lock = DistributedLock::new(&dir, "test_noop", "owner-1");
        assert!(lock.release().is_ok()); // should not panic or error
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_stale_lock_is_broken() {
        let dir = tmp_dir();
        // Write a lock file with a very old timestamp
        let lock_path = dir.join("stale.lock");
        let data = LockFileData {
            owner_id: "dead-process".to_string(),
            token: "abc".to_string(),
            acquired_at_secs: 1, // epoch + 1 second → definitely stale
        };
        fs::write(&lock_path, data.encode()).expect("write stale lock");

        let mut lock = DistributedLock::new(&dir, "stale", "owner-1").with_ttl(Duration::from_secs(30));
        assert!(lock.try_acquire().is_ok(), "should break stale lock");
        let _ = lock.release();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_peek_returns_lock_data() {
        let dir = tmp_dir();
        let mut lock = DistributedLock::new(&dir, "peek_test", "owner-peek");
        lock.try_acquire().expect("acquire");
        let data = lock.peek().expect("peek should return data");
        assert_eq!(data.owner_id, "owner-peek");
        let _ = lock.release();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_peek_returns_none_when_no_lock() {
        let dir = tmp_dir();
        let lock = DistributedLock::new(&dir, "no_lock", "owner-1");
        assert!(lock.peek().is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_lock_file_data_encode_decode_roundtrip() {
        let data = LockFileData {
            owner_id: "myhost-12345".to_string(),
            token: "deadbeef00000000".to_string(),
            acquired_at_secs: 1_700_000_000,
        };
        let encoded = data.encode();
        let decoded = LockFileData::decode(&encoded).expect("decode");
        assert_eq!(decoded.owner_id, data.owner_id);
        assert_eq!(decoded.token, data.token);
        assert_eq!(decoded.acquired_at_secs, data.acquired_at_secs);
    }

    #[test]
    fn test_is_stale_fresh_lock() {
        let data = LockFileData {
            owner_id: "x".to_string(),
            token: "y".to_string(),
            acquired_at_secs: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        assert!(!data.is_stale(Duration::from_secs(60)));
    }

    #[test]
    fn test_is_stale_old_lock() {
        let data = LockFileData {
            owner_id: "x".to_string(),
            token: "y".to_string(),
            acquired_at_secs: 1, // epoch-relative, definitely stale
        };
        assert!(data.is_stale(Duration::from_secs(30)));
    }

    #[test]
    fn test_second_acquire_after_release() {
        let dir = tmp_dir();
        let mut lock1 = DistributedLock::new(&dir, "seq", "owner-1");
        let mut lock2 = DistributedLock::new(&dir, "seq", "owner-2");
        lock1.try_acquire().expect("first acquire");
        lock1.release().expect("release");
        assert!(lock2.try_acquire().is_ok(), "second acquire after release");
        let _ = lock2.release();
        let _ = fs::remove_dir_all(&dir);
    }
}
