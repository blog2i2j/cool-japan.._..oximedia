#![allow(dead_code)]
//! Software license pool management for the encoding farm.
//!
//! ## Model
//!
//! A [`LicensePool`] manages a fixed number of floating software licenses.
//! Workers check out a [`LicenseToken`] before starting a licensed job and
//! return it when the job finishes.  Each token has a configurable TTL so that
//! stale leases are automatically reclaimed.
//!
//! ## Features
//!
//! - **Floating check-out / check-in** — at most `capacity` tokens active at
//!   once.
//! - **Per-token TTL** — tokens expire automatically; the pool reclaims them
//!   when `reclaim_expired` (or any checkout attempt) is called.
//! - **License expiry** — the pool itself can carry a hard expiry date; after
//!   that date no new checkouts are allowed.
//! - **Product-keyed pools** — a [`LicenseManager`] can hold multiple pools,
//!   one per software product (e.g. `"dolby-vision-encoder"`,
//!   `"hevc-encoder"`).
//! - **Usage accounting** — the pool tracks total checkouts, peak concurrent
//!   usage, and checkout denials for dashboarding.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::{FarmError, WorkerId};

// ---------------------------------------------------------------------------
// LicenseToken
// ---------------------------------------------------------------------------

/// An opaque token representing one checked-out floating license.
///
/// The token is returned to the pool via [`LicensePool::check_in`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LicenseToken {
    /// Unique token identifier (sequential within the pool).
    pub id: u64,
    /// The worker that holds this lease.
    pub worker_id: WorkerId,
    /// Optional job-level annotation (e.g. job UUID string).
    pub annotation: Option<String>,
    /// Wall-clock instant at which the lease was granted.
    pub checked_out_at: Instant,
    /// Maximum lifetime of this lease before it is reclaimed.
    pub ttl: Duration,
}

impl LicenseToken {
    /// Returns `true` if this token has outlived its TTL.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.checked_out_at.elapsed() >= self.ttl
    }

    /// How much time remains before this token expires.  Returns
    /// [`Duration::ZERO`] if the token has already expired.
    #[must_use]
    pub fn remaining_ttl(&self) -> Duration {
        let elapsed = self.checked_out_at.elapsed();
        self.ttl.saturating_sub(elapsed)
    }
}

// ---------------------------------------------------------------------------
// LicensePoolConfig
// ---------------------------------------------------------------------------

/// Configuration for a single product's license pool.
#[derive(Debug, Clone)]
pub struct LicensePoolConfig {
    /// Human-readable product name (e.g. `"dolby-vision-encoder"`).
    pub product: String,
    /// Total number of floating seats.
    pub capacity: usize,
    /// Default TTL applied to each checkout.  Workers **must** renew or return
    /// the token before this elapses, otherwise it is reclaimed.
    pub default_ttl: Duration,
    /// Optional hard expiry for the entire license (wall-clock `Instant`).
    /// After this instant no new checkouts are allowed.
    pub license_expires_at: Option<Instant>,
}

impl LicensePoolConfig {
    /// Create a simple configuration with no hard expiry.
    #[must_use]
    pub fn new(product: impl Into<String>, capacity: usize, default_ttl: Duration) -> Self {
        Self {
            product: product.into(),
            capacity,
            default_ttl,
            license_expires_at: None,
        }
    }

    /// Attach a hard expiry instant to this configuration.
    #[must_use]
    pub fn with_expiry(mut self, expires_at: Instant) -> Self {
        self.license_expires_at = Some(expires_at);
        self
    }
}

// ---------------------------------------------------------------------------
// LicensePoolStats
// ---------------------------------------------------------------------------

/// Aggregated statistics for one license pool.
#[derive(Debug, Clone, Default)]
pub struct LicensePoolStats {
    /// Total successful checkouts since pool creation.
    pub total_checkouts: u64,
    /// Total tokens that were returned via `check_in`.
    pub total_checkins: u64,
    /// Total checkout attempts that were denied (capacity or license expired).
    pub total_denials: u64,
    /// Total tokens reclaimed because they expired without being returned.
    pub total_reclaims: u64,
    /// Peak number of concurrently held tokens ever observed.
    pub peak_concurrent: usize,
}

// ---------------------------------------------------------------------------
// LicensePool
// ---------------------------------------------------------------------------

/// A floating-license pool for a single software product.
#[derive(Debug)]
pub struct LicensePool {
    config: LicensePoolConfig,
    /// Tokens currently held by workers.  Key = token id.
    active: HashMap<u64, LicenseToken>,
    /// Monotonically increasing token id counter.
    next_id: u64,
    stats: LicensePoolStats,
}

impl LicensePool {
    /// Create a new pool from the supplied configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::InvalidConfig`] if `capacity` is zero.
    pub fn new(config: LicensePoolConfig) -> crate::Result<Self> {
        if config.capacity == 0 {
            return Err(FarmError::InvalidConfig(format!(
                "LicensePool '{}': capacity must be > 0",
                config.product
            )));
        }
        Ok(Self {
            config,
            active: HashMap::new(),
            next_id: 1,
            stats: LicensePoolStats::default(),
        })
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Product name this pool manages.
    #[must_use]
    pub fn product(&self) -> &str {
        &self.config.product
    }

    /// Total license capacity (seats).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }

    /// Number of currently active (non-expired) tokens.
    ///
    /// This eagerly reclaims expired tokens before counting.
    pub fn active_count(&mut self) -> usize {
        self.reclaim_expired();
        self.active.len()
    }

    /// Number of available seats (capacity minus active count).
    pub fn available(&mut self) -> usize {
        let active = self.active_count();
        self.config.capacity.saturating_sub(active)
    }

    /// Returns `true` if the license itself has expired (hard expiry).
    #[must_use]
    pub fn is_license_expired(&self) -> bool {
        match self.config.license_expires_at {
            Some(exp) => Instant::now() >= exp,
            None => false,
        }
    }

    /// Check out one floating license seat for `worker_id`.
    ///
    /// An optional `annotation` string (e.g. job UUID) can be attached for
    /// debugging purposes.  If `ttl` is `None` the pool's `default_ttl` is
    /// used.
    ///
    /// # Errors
    ///
    /// - [`FarmError::ResourceExhausted`] — all seats are in use.
    /// - [`FarmError::PermissionDenied`] — the hard license expiry has passed.
    pub fn check_out(
        &mut self,
        worker_id: WorkerId,
        annotation: Option<String>,
        ttl: Option<Duration>,
    ) -> crate::Result<LicenseToken> {
        // First, reclaim any expired tokens to free seats.
        self.reclaim_expired();

        if self.is_license_expired() {
            self.stats.total_denials += 1;
            return Err(FarmError::PermissionDenied(format!(
                "License for '{}' has expired",
                self.config.product
            )));
        }

        if self.active.len() >= self.config.capacity {
            self.stats.total_denials += 1;
            return Err(FarmError::ResourceExhausted(format!(
                "No available seats for '{}' ({}/{})",
                self.config.product,
                self.active.len(),
                self.config.capacity
            )));
        }

        let token = LicenseToken {
            id: self.next_id,
            worker_id,
            annotation,
            checked_out_at: Instant::now(),
            ttl: ttl.unwrap_or(self.config.default_ttl),
        };
        self.next_id += 1;

        self.stats.total_checkouts += 1;
        self.active.insert(token.id, token.clone());

        let concurrent = self.active.len();
        if concurrent > self.stats.peak_concurrent {
            self.stats.peak_concurrent = concurrent;
        }

        Ok(token)
    }

    /// Return a license token to the pool.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::NotFound`] if the token id is not currently
    /// registered with this pool (it may have already been reclaimed).
    pub fn check_in(&mut self, token_id: u64) -> crate::Result<()> {
        if self.active.remove(&token_id).is_none() {
            return Err(FarmError::NotFound(format!(
                "License token {token_id} not found in pool '{}'",
                self.config.product
            )));
        }
        self.stats.total_checkins += 1;
        Ok(())
    }

    /// Renew a token's TTL in-place, resetting its `checked_out_at` timestamp.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::NotFound`] if the token is unknown or has already
    /// been reclaimed.
    pub fn renew(&mut self, token_id: u64, new_ttl: Option<Duration>) -> crate::Result<()> {
        let ttl = new_ttl.unwrap_or(self.config.default_ttl);
        match self.active.get_mut(&token_id) {
            Some(token) => {
                token.checked_out_at = Instant::now();
                token.ttl = ttl;
                Ok(())
            }
            None => Err(FarmError::NotFound(format!(
                "License token {token_id} not found in pool '{}'",
                self.config.product
            ))),
        }
    }

    /// Forcibly revoke all tokens held by `worker_id` (e.g. worker went
    /// offline).  Returns the number of tokens revoked.
    pub fn revoke_worker(&mut self, worker_id: &WorkerId) -> usize {
        let before = self.active.len();
        self.active.retain(|_, t| &t.worker_id != worker_id);
        let revoked = before - self.active.len();
        self.stats.total_checkins += revoked as u64;
        revoked
    }

    /// Scan active tokens and remove those whose TTL has elapsed.
    /// Returns the number of tokens reclaimed.
    pub fn reclaim_expired(&mut self) -> usize {
        let before = self.active.len();
        self.active.retain(|_, t| !t.is_expired());
        let reclaimed = before - self.active.len();
        self.stats.total_reclaims += reclaimed as u64;
        reclaimed
    }

    /// Snapshot of current pool statistics.
    #[must_use]
    pub fn stats(&self) -> &LicensePoolStats {
        &self.stats
    }

    /// Iterate over all currently active (non-expired) tokens.
    pub fn active_tokens(&self) -> impl Iterator<Item = &LicenseToken> {
        self.active.values()
    }

    /// List all tokens held by a specific worker.
    pub fn tokens_for_worker(&self, worker_id: &WorkerId) -> Vec<&LicenseToken> {
        self.active
            .values()
            .filter(|t| &t.worker_id == worker_id)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// LicenseManager
// ---------------------------------------------------------------------------

/// Manages multiple named [`LicensePool`]s, one per software product.
#[derive(Debug, Default)]
pub struct LicenseManager {
    pools: HashMap<String, LicensePool>,
}

impl LicenseManager {
    /// Create an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new license pool.
    ///
    /// # Errors
    ///
    /// - [`FarmError::AlreadyExists`] if a pool with the same product name is
    ///   already registered.
    /// - Propagates errors from [`LicensePool::new`].
    pub fn register(&mut self, config: LicensePoolConfig) -> crate::Result<()> {
        let product = config.product.clone();
        if self.pools.contains_key(&product) {
            return Err(FarmError::AlreadyExists(format!(
                "License pool '{product}' already registered"
            )));
        }
        let pool = LicensePool::new(config)?;
        self.pools.insert(product, pool);
        Ok(())
    }

    /// Look up a pool by product name (immutable).
    #[must_use]
    pub fn pool(&self, product: &str) -> Option<&LicensePool> {
        self.pools.get(product)
    }

    /// Look up a pool by product name (mutable).
    pub fn pool_mut(&mut self, product: &str) -> Option<&mut LicensePool> {
        self.pools.get_mut(product)
    }

    /// Check out a seat from the named product pool.
    ///
    /// # Errors
    ///
    /// - [`FarmError::NotFound`] if the product is not registered.
    /// - Propagates errors from [`LicensePool::check_out`].
    pub fn check_out(
        &mut self,
        product: &str,
        worker_id: WorkerId,
        annotation: Option<String>,
        ttl: Option<Duration>,
    ) -> crate::Result<LicenseToken> {
        self.pools
            .get_mut(product)
            .ok_or_else(|| FarmError::NotFound(format!("License pool '{product}' not found")))?
            .check_out(worker_id, annotation, ttl)
    }

    /// Return a token to its pool.
    ///
    /// # Errors
    ///
    /// - [`FarmError::NotFound`] if the product is not registered, or if the
    ///   token id is unknown to the pool.
    pub fn check_in(&mut self, product: &str, token_id: u64) -> crate::Result<()> {
        self.pools
            .get_mut(product)
            .ok_or_else(|| FarmError::NotFound(format!("License pool '{product}' not found")))?
            .check_in(token_id)
    }

    /// Revoke all licenses held by `worker_id` across ALL pools.
    /// Returns the total number of tokens revoked.
    pub fn revoke_worker(&mut self, worker_id: &WorkerId) -> usize {
        self.pools
            .values_mut()
            .map(|p| p.revoke_worker(worker_id))
            .sum()
    }

    /// Reclaim expired tokens across all pools.
    pub fn reclaim_all_expired(&mut self) -> usize {
        self.pools.values_mut().map(|p| p.reclaim_expired()).sum()
    }

    /// Iterate over all registered product names.
    pub fn products(&self) -> impl Iterator<Item = &str> {
        self.pools.keys().map(String::as_str)
    }

    /// Aggregate stats snapshot for all pools.
    #[must_use]
    pub fn aggregate_stats(&self) -> LicensePoolStats {
        self.pools.values().fold(LicensePoolStats::default(), |mut acc, p| {
            let s = p.stats();
            acc.total_checkouts += s.total_checkouts;
            acc.total_checkins += s.total_checkins;
            acc.total_denials += s.total_denials;
            acc.total_reclaims += s.total_reclaims;
            if s.peak_concurrent > acc.peak_concurrent {
                acc.peak_concurrent = s.peak_concurrent;
            }
            acc
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(n: u8) -> WorkerId {
        WorkerId::new(format!("worker-{n}"))
    }

    fn make_pool(cap: usize) -> LicensePool {
        let cfg = LicensePoolConfig::new("test-product", cap, Duration::from_secs(60));
        LicensePool::new(cfg).expect("pool creation should succeed")
    }

    #[test]
    fn test_basic_checkout_checkin() {
        let mut pool = make_pool(2);
        let tok = pool.check_out(worker(1), None, None).expect("checkout should succeed");
        assert_eq!(pool.active_count(), 1);
        pool.check_in(tok.id).expect("check-in should succeed");
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.stats().total_checkouts, 1);
        assert_eq!(pool.stats().total_checkins, 1);
    }

    #[test]
    fn test_capacity_exhaustion() {
        let mut pool = make_pool(2);
        let _t1 = pool.check_out(worker(1), None, None).expect("first checkout should succeed");
        let _t2 = pool.check_out(worker(2), None, None).expect("second checkout should succeed");
        let err = pool.check_out(worker(3), None, None);
        assert!(matches!(err, Err(FarmError::ResourceExhausted(_))));
        assert_eq!(pool.stats().total_denials, 1);
    }

    #[test]
    fn test_ttl_expiry_reclaim() {
        let cfg = LicensePoolConfig::new("expiry-test", 1, Duration::from_millis(1));
        let mut pool = LicensePool::new(cfg).expect("pool creation should succeed");

        let _tok = pool.check_out(worker(1), None, Some(Duration::from_millis(1)))
            .expect("checkout should succeed");

        // Spin until the token is expired (1 ms TTL).
        std::thread::sleep(Duration::from_millis(10));

        let reclaimed = pool.reclaim_expired();
        assert_eq!(reclaimed, 1);
        assert_eq!(pool.active_count(), 0);
        // After reclaim a new checkout should succeed.
        pool.check_out(worker(2), None, None).expect("checkout after reclaim should succeed");
    }

    #[test]
    fn test_token_renewal() {
        let cfg = LicensePoolConfig::new("renew-test", 2, Duration::from_millis(1));
        let mut pool = LicensePool::new(cfg).expect("pool creation should succeed");
        let tok = pool.check_out(worker(1), None, Some(Duration::from_millis(1)))
            .expect("checkout should succeed");
        // Renew with a longer TTL before expiry.
        pool.renew(tok.id, Some(Duration::from_secs(60)))
            .expect("renewal should succeed");
        // Sleep past the original TTL.
        std::thread::sleep(Duration::from_millis(10));
        // Token should NOT be reclaimed because it was renewed.
        assert_eq!(pool.reclaim_expired(), 0);
    }

    #[test]
    fn test_renew_unknown_token() {
        let mut pool = make_pool(2);
        let err = pool.renew(9999, None);
        assert!(matches!(err, Err(FarmError::NotFound(_))));
    }

    #[test]
    fn test_revoke_worker() {
        let mut pool = make_pool(4);
        let w = worker(1);
        pool.check_out(w.clone(), None, None).expect("first checkout should succeed");
        pool.check_out(w.clone(), None, None).expect("second checkout should succeed");
        pool.check_out(worker(2), None, None).expect("third checkout should succeed");

        let revoked = pool.revoke_worker(&w);
        assert_eq!(revoked, 2);
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn test_license_hard_expiry() {
        // Construct a pool whose license has already expired.
        let past = Instant::now(); // effectively "now" — expired immediately
        std::thread::sleep(Duration::from_millis(2));
        let cfg = LicensePoolConfig::new("expired-product", 5, Duration::from_secs(60))
            .with_expiry(past);
        let mut pool = LicensePool::new(cfg).expect("pool creation should succeed");
        let err = pool.check_out(worker(1), None, None);
        assert!(matches!(err, Err(FarmError::PermissionDenied(_))));
    }

    #[test]
    fn test_license_manager_multi_product() {
        let mut mgr = LicenseManager::new();
        mgr.register(LicensePoolConfig::new("prod-a", 2, Duration::from_secs(60)))
            .expect("register prod-a should succeed");
        mgr.register(LicensePoolConfig::new("prod-b", 1, Duration::from_secs(60)))
            .expect("register prod-b should succeed");

        let tok_a = mgr.check_out("prod-a", worker(1), None, None)
            .expect("checkout prod-a should succeed");
        mgr.check_out("prod-b", worker(2), None, None)
            .expect("checkout prod-b should succeed");

        // prod-b exhausted.
        let err = mgr.check_out("prod-b", worker(3), None, None);
        assert!(matches!(err, Err(FarmError::ResourceExhausted(_))));

        mgr.check_in("prod-a", tok_a.id).expect("check-in prod-a should succeed");

        let stats = mgr.aggregate_stats();
        assert_eq!(stats.total_checkouts, 2);
        assert_eq!(stats.total_checkins, 1);
        assert_eq!(stats.total_denials, 1);
    }

    #[test]
    fn test_duplicate_registration_error() {
        let mut mgr = LicenseManager::new();
        mgr.register(LicensePoolConfig::new("dup", 1, Duration::from_secs(60)))
            .expect("first registration should succeed");
        let err = mgr.register(LicensePoolConfig::new("dup", 1, Duration::from_secs(60)));
        assert!(matches!(err, Err(FarmError::AlreadyExists(_))));
    }

    #[test]
    fn test_peak_concurrent_tracking() {
        let mut pool = make_pool(5);
        let t1 = pool.check_out(worker(1), None, None).expect("checkout 1 should succeed");
        let t2 = pool.check_out(worker(2), None, None).expect("checkout 2 should succeed");
        let _t3 = pool.check_out(worker(3), None, None).expect("checkout 3 should succeed");
        pool.check_in(t1.id).expect("check-in t1 should succeed");
        pool.check_in(t2.id).expect("check-in t2 should succeed");
        assert_eq!(pool.stats().peak_concurrent, 3);
        assert_eq!(pool.active_count(), 1);
    }
}
