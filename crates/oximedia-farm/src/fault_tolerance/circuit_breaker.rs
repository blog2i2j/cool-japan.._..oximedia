//! Circuit breaker implementation

use parking_lot::RwLock;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Circuit is closed, requests pass through
    Closed,
    /// Circuit is open, requests are rejected
    Open,
    /// Circuit is half-open, testing if service recovered
    HalfOpen,
}

/// Circuit breaker for fault tolerance
pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitBreakerState>>,
    failure_count: Arc<AtomicU32>,
    success_count: Arc<AtomicU32>,
    failure_threshold: u32,
    timeout: Duration,
    recovery_time: Duration,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    #[must_use]
    pub fn new(failure_threshold: u32, timeout: Duration, recovery_time: Duration) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
            failure_count: Arc::new(AtomicU32::new(0)),
            success_count: Arc::new(AtomicU32::new(0)),
            failure_threshold,
            timeout,
            recovery_time,
            last_failure_time: Arc::new(RwLock::new(None)),
        }
    }

    /// Check if circuit breaker is open
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.update_state();
        *self.state.read() == CircuitBreakerState::Open
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> CircuitBreakerState {
        self.update_state();
        *self.state.read()
    }

    /// Record a successful operation
    pub fn record_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);

        let current_state = *self.state.read();

        if current_state == CircuitBreakerState::HalfOpen {
            // Successful operation in half-open state, close the circuit
            *self.state.write() = CircuitBreakerState::Closed;
            self.failure_count.store(0, Ordering::Relaxed);
            *self.last_failure_time.write() = None;
            tracing::info!("Circuit breaker closed after successful recovery");
        }
    }

    /// Record a failed operation
    pub fn record_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        *self.last_failure_time.write() = Some(Instant::now());

        if failures >= self.failure_threshold {
            let current_state = *self.state.read();
            if current_state != CircuitBreakerState::Open {
                *self.state.write() = CircuitBreakerState::Open;
                tracing::warn!("Circuit breaker opened after {} failures", failures);
            }
        }
    }

    /// Update circuit breaker state based on time
    fn update_state(&self) {
        let current_state = *self.state.read();

        if current_state == CircuitBreakerState::Open {
            if let Some(last_failure) = *self.last_failure_time.read() {
                if last_failure.elapsed() >= self.recovery_time {
                    // Transition to half-open state
                    *self.state.write() = CircuitBreakerState::HalfOpen;
                    tracing::info!("Circuit breaker transitioned to half-open state");
                }
            }
        }
    }

    /// Reset the circuit breaker
    pub fn reset(&self) {
        *self.state.write() = CircuitBreakerState::Closed;
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        *self.last_failure_time.write() = None;
        tracing::info!("Circuit breaker reset");
    }

    /// Get failure count
    #[must_use]
    pub fn failure_count(&self) -> u32 {
        self.failure_count.load(Ordering::Relaxed)
    }

    /// Get success count
    #[must_use]
    pub fn success_count(&self) -> u32 {
        self.success_count.load(Ordering::Relaxed)
    }
}

impl Clone for CircuitBreaker {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            failure_count: self.failure_count.clone(),
            success_count: self.success_count.clone(),
            failure_threshold: self.failure_threshold,
            timeout: self.timeout,
            recovery_time: self.recovery_time,
            last_failure_time: self.last_failure_time.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_creation() {
        let cb = CircuitBreaker::new(5, Duration::from_secs(60), Duration::from_secs(10));

        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        assert!(!cb.is_open());
    }

    #[test]
    fn test_record_success() {
        let cb = CircuitBreaker::new(5, Duration::from_secs(60), Duration::from_secs(10));

        cb.record_success();
        assert_eq!(cb.success_count(), 1);
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
    }

    #[test]
    fn test_record_failure() {
        let cb = CircuitBreaker::new(5, Duration::from_secs(60), Duration::from_secs(10));

        cb.record_failure();
        assert_eq!(cb.failure_count(), 1);
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
    }

    #[test]
    fn test_circuit_opens_on_threshold() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60), Duration::from_secs(10));

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);
        assert!(cb.is_open());
    }

    #[test]
    fn test_circuit_transitions_to_half_open() {
        let cb = CircuitBreaker::new(
            3,
            Duration::from_secs(60),
            Duration::from_millis(100), // Short recovery time for testing
        );

        // Open the circuit
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);

        // Wait for recovery time
        std::thread::sleep(Duration::from_millis(150));

        // Should transition to half-open
        assert_eq!(cb.state(), CircuitBreakerState::HalfOpen);
    }

    #[test]
    fn test_circuit_closes_from_half_open() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60), Duration::from_millis(100));

        // Open the circuit
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);

        // Wait for recovery
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(cb.state(), CircuitBreakerState::HalfOpen);

        // Successful operation should close the circuit
        cb.record_success();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        assert_eq!(cb.failure_count(), 0);
    }

    #[test]
    fn test_reset() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60), Duration::from_secs(10));

        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);

        cb.reset();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        assert_eq!(cb.failure_count(), 0);
        assert_eq!(cb.success_count(), 0);
    }

    #[test]
    fn test_clone() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60), Duration::from_secs(10));

        cb.record_failure();

        let cb2 = cb.clone();
        assert_eq!(cb2.failure_count(), 1);
        assert_eq!(cb2.state(), cb.state());
    }

    // ── Additional state-transition tests ────────────────────────────────────

    #[test]
    fn test_initial_state_is_closed() {
        let cb = CircuitBreaker::new(5, Duration::from_secs(60), Duration::from_secs(10));
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        assert!(!cb.is_open());
    }

    #[test]
    fn test_is_open_false_in_half_open() {
        // `is_open()` should return `false` when the circuit is HalfOpen,
        // allowing a probe request through.
        let cb = CircuitBreaker::new(1, Duration::from_secs(60), Duration::from_millis(50));
        cb.record_failure(); // → Open
        assert!(cb.is_open());
        std::thread::sleep(Duration::from_millis(80));
        // After recovery window the state should be HalfOpen.
        assert_eq!(cb.state(), CircuitBreakerState::HalfOpen);
        assert!(
            !cb.is_open(),
            "is_open() must return false in HalfOpen state"
        );
    }

    #[test]
    fn test_failure_in_half_open_reopens() {
        let cb = CircuitBreaker::new(1, Duration::from_secs(60), Duration::from_millis(50));
        // Open
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);
        // Wait → HalfOpen
        std::thread::sleep(Duration::from_millis(80));
        assert_eq!(cb.state(), CircuitBreakerState::HalfOpen);
        // Failure in HalfOpen → back to Open
        cb.record_failure();
        // reset last_failure_time so update_state doesn't immediately re-open
        assert_eq!(
            cb.state(),
            CircuitBreakerState::Open,
            "failure in HalfOpen should re-open the circuit"
        );
    }

    #[test]
    fn test_success_in_closed_does_not_change_state() {
        let cb = CircuitBreaker::new(5, Duration::from_secs(60), Duration::from_secs(10));
        cb.record_success();
        cb.record_success();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        assert_eq!(cb.success_count(), 2);
    }

    #[test]
    fn test_full_cycle_closed_open_halfopen_closed() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(60), Duration::from_millis(60));
        // Closed → Open
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);
        // Open → HalfOpen (after recovery window)
        std::thread::sleep(Duration::from_millis(90));
        assert_eq!(cb.state(), CircuitBreakerState::HalfOpen);
        // HalfOpen → Closed (success)
        cb.record_success();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        assert_eq!(cb.failure_count(), 0, "failure count must reset on Closed");
    }

    #[test]
    fn test_second_cycle_after_reset() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60), Duration::from_millis(60));
        // First cycle
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        cb.reset();
        // Second cycle starts fresh
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);
    }

    #[test]
    fn test_failure_count_not_reset_on_open() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60), Duration::from_secs(10));
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.failure_count(), 3);
        // Additional failures while Open should still increment the counter.
        cb.record_failure();
        assert_eq!(cb.failure_count(), 4);
    }

    #[test]
    fn test_concurrent_record_failure() {
        use std::sync::Arc;
        // Concurrent recording must not deadlock and the final failure count
        // must equal the number of calls made.
        let cb = Arc::new(CircuitBreaker::new(
            1000,
            Duration::from_secs(60),
            Duration::from_secs(10),
        ));
        let mut handles = Vec::new();
        for _ in 0..10 {
            let cb_c = cb.clone();
            let h = std::thread::spawn(move || {
                for _ in 0..10 {
                    cb_c.record_failure();
                }
            });
            handles.push(h);
        }
        for h in handles {
            h.join().expect("thread should not panic");
        }
        assert_eq!(cb.failure_count(), 100);
    }

    #[test]
    fn test_state_getter_consistent_with_is_open() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(60), Duration::from_secs(10));
        cb.record_failure();
        cb.record_failure();
        // Both APIs should agree
        assert_eq!(cb.state(), CircuitBreakerState::Open);
        assert!(cb.is_open());
        cb.reset();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        assert!(!cb.is_open());
    }
}
