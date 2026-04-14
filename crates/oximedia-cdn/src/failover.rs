//! CDN failover manager — select the healthiest CDN provider.
//!
//! [`CdnFailoverManager`] tracks a pool of CDN providers with health scores
//! and always routes to the provider with the highest health.

/// A CDN failover manager that routes to the healthiest available provider.
///
/// # Example
/// ```
/// use oximedia_cdn::failover::CdnFailoverManager;
///
/// let mut mgr = CdnFailoverManager::new();
/// mgr.add_cdn("fastly", 0.9);
/// mgr.add_cdn("cloudfront", 0.6);
/// assert_eq!(mgr.best_cdn(), Some("fastly"));
/// mgr.add_cdn("akamai", 1.0);
/// assert_eq!(mgr.best_cdn(), Some("akamai"));
/// ```
#[derive(Debug, Default)]
pub struct CdnFailoverManager {
    /// Registered providers in insertion order: `(id, health_score)`.
    providers: Vec<(String, f32)>,
}

impl CdnFailoverManager {
    /// Create an empty failover manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add (or update) a CDN provider with a health score in `[0.0, 1.0]`.
    ///
    /// `health` is clamped to `[0.0, 1.0]`. If `id` already exists its score
    /// is updated.
    pub fn add_cdn(&mut self, id: &str, health: f32) {
        let health = health.clamp(0.0, 1.0);
        if let Some(entry) = self.providers.iter_mut().find(|(pid, _)| pid == id) {
            entry.1 = health;
        } else {
            self.providers.push((id.to_string(), health));
        }
    }

    /// Return the ID of the provider with the highest health score, or `None`
    /// if no providers are registered.
    ///
    /// When multiple providers share the maximum score the one that was
    /// registered (or last-updated) earliest wins.
    pub fn best_cdn(&self) -> Option<&str> {
        self.providers
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id.as_str())
    }

    /// Number of registered providers.
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    /// Update the health score for an existing provider.
    ///
    /// Returns `true` if the provider was found, `false` otherwise.
    pub fn update_health(&mut self, id: &str, health: f32) -> bool {
        let health = health.clamp(0.0, 1.0);
        if let Some(entry) = self.providers.iter_mut().find(|(pid, _)| pid == id) {
            entry.1 = health;
            true
        } else {
            false
        }
    }

    /// Health score of a specific provider, or `None` if not registered.
    pub fn health_of(&self, id: &str) -> Option<f32> {
        self.providers
            .iter()
            .find(|(pid, _)| pid == id)
            .map(|(_, h)| *h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_providers_returns_none() {
        let mgr = CdnFailoverManager::new();
        assert!(mgr.best_cdn().is_none());
    }

    #[test]
    fn test_single_provider() {
        let mut mgr = CdnFailoverManager::new();
        mgr.add_cdn("cf", 0.8);
        assert_eq!(mgr.best_cdn(), Some("cf"));
    }

    #[test]
    fn test_best_cdn_highest_health() {
        let mut mgr = CdnFailoverManager::new();
        mgr.add_cdn("a", 0.5);
        mgr.add_cdn("b", 0.9);
        mgr.add_cdn("c", 0.3);
        assert_eq!(mgr.best_cdn(), Some("b"));
    }

    #[test]
    fn test_update_health() {
        let mut mgr = CdnFailoverManager::new();
        mgr.add_cdn("a", 0.5);
        mgr.add_cdn("b", 0.9);
        assert!(mgr.update_health("a", 1.0));
        assert_eq!(mgr.best_cdn(), Some("a"));
    }

    #[test]
    fn test_update_nonexistent() {
        let mut mgr = CdnFailoverManager::new();
        assert!(!mgr.update_health("ghost", 0.8));
    }

    #[test]
    fn test_health_clamped() {
        let mut mgr = CdnFailoverManager::new();
        mgr.add_cdn("over", 1.5);
        mgr.add_cdn("under", -0.5);
        assert!((mgr.health_of("over").unwrap() - 1.0).abs() < f32::EPSILON);
        assert!((mgr.health_of("under").unwrap() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_add_cdn_updates_existing() {
        let mut mgr = CdnFailoverManager::new();
        mgr.add_cdn("x", 0.3);
        mgr.add_cdn("x", 0.9);
        assert_eq!(mgr.provider_count(), 1);
        assert!((mgr.health_of("x").unwrap() - 0.9).abs() < f32::EPSILON);
    }
}
