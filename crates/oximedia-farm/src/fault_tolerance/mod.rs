//! Fault tolerance and recovery mechanisms

mod circuit_breaker;
mod retry;

pub use circuit_breaker::{CircuitBreaker, CircuitBreakerState};
pub use retry::{RetryPolicy, RetryStrategy};

use crate::{FarmError, Result};
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;

/// Fault tolerance manager
pub struct FaultTolerance {
    retry_policy: RetryPolicy,
    circuit_breaker: CircuitBreaker,
}

impl FaultTolerance {
    /// Create a new fault tolerance manager
    #[must_use]
    pub fn new(retry_policy: RetryPolicy) -> Self {
        Self {
            retry_policy,
            circuit_breaker: CircuitBreaker::new(
                5,                       // failure threshold
                Duration::from_secs(60), // timeout
                Duration::from_secs(10), // recovery time
            ),
        }
    }

    /// Execute an operation with retry and circuit breaker
    pub async fn execute<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        // Check circuit breaker
        if self.circuit_breaker.is_open() {
            Err(FarmError::ResourceExhausted(
                "Circuit breaker is open".to_string(),
            ))
        } else {
            // Circuit is closed or half-open, try the operation
            match self.execute_with_retry(&operation).await {
                Ok(result) => {
                    self.circuit_breaker.record_success();
                    Ok(result)
                }
                Err(e) => {
                    self.circuit_breaker.record_failure();
                    Err(e)
                }
            }
        }
    }

    /// Execute with retry policy
    async fn execute_with_retry<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let mut attempts = 0;
        let max_attempts = self.retry_policy.max_attempts;

        loop {
            attempts += 1;

            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if attempts >= max_attempts || !self.should_retry(&e) {
                        return Err(e);
                    }

                    // Calculate backoff delay
                    let delay = self.retry_policy.calculate_delay(attempts);
                    tracing::warn!(
                        "Operation failed (attempt {}/{}), retrying in {:?}: {}",
                        attempts,
                        max_attempts,
                        delay,
                        e
                    );

                    sleep(delay).await;
                }
            }
        }
    }

    /// Determine if an error should be retried
    fn should_retry(&self, error: &FarmError) -> bool {
        match error {
            FarmError::Network(_) => true,
            FarmError::Timeout(_) => true,
            FarmError::Io(_) => true,
            FarmError::Database(_) => false, // Database errors typically shouldn't be retried
            FarmError::NotFound(_) => false,
            FarmError::AlreadyExists(_) => false,
            FarmError::PermissionDenied(_) => false,
            FarmError::InvalidState(_) => false,
            _ => false,
        }
    }

    /// Get circuit breaker state
    #[must_use]
    pub fn circuit_state(&self) -> CircuitBreakerState {
        self.circuit_breaker.state()
    }

    /// Reset circuit breaker
    pub fn reset_circuit(&self) {
        self.circuit_breaker.reset();
    }
}

impl Default for FaultTolerance {
    fn default() -> Self {
        Self::new(RetryPolicy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_successful_execution() {
        let ft = FaultTolerance::default();

        let result = ft.execute(|| async { Ok::<i32, FarmError>(42) }).await;

        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_retry_on_transient_failure() {
        let ft = FaultTolerance::new(RetryPolicy {
            max_attempts: 3,
            strategy: RetryStrategy::Fixed(Duration::from_millis(10)),
        });

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let result = ft
            .execute(move || {
                let count = counter_clone.fetch_add(1, Ordering::SeqCst);
                async move {
                    if count < 2 {
                        Err(FarmError::Io(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "network error",
                        )))
                    } else {
                        Ok(42)
                    }
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_max_retries_exceeded() {
        let ft = FaultTolerance::new(RetryPolicy {
            max_attempts: 2,
            strategy: RetryStrategy::Fixed(Duration::from_millis(10)),
        });

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let result = ft
            .execute(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                async move {
                    Err::<(), _>(FarmError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "network error",
                    )))
                }
            })
            .await;

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_no_retry_on_permanent_error() {
        let ft = FaultTolerance::new(RetryPolicy {
            max_attempts: 3,
            strategy: RetryStrategy::Fixed(Duration::from_millis(10)),
        });

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let result = ft
            .execute(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                async move { Err::<(), _>(FarmError::NotFound("not found".to_string())) }
            })
            .await;

        assert!(result.is_err());
        // Should not retry on NotFound error
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
