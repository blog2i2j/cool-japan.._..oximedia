//! Frame number mapping between proxy and original media.
//!
//! When a proxy is generated at a different frame rate than the original,
//! frame numbers no longer correspond 1:1.  [`ProxyFrameMap`] provides
//! bidirectional conversion between proxy frame indices and their closest
//! original frame counterpart.

/// Maps proxy frame numbers to original media frame numbers and vice versa.
///
/// The mapping is purely temporal: proxy frame *n* corresponds to the original
/// frame nearest to the timestamp `n / proxy_fps`.
#[derive(Debug, Clone)]
pub struct ProxyFrameMap {
    /// Frame rate of the proxy media (frames per second).
    proxy_fps: f64,
    /// Frame rate of the original media (frames per second).
    original_fps: f64,
    /// Pre-computed ratio `original_fps / proxy_fps`.
    ratio: f64,
}

impl ProxyFrameMap {
    /// Create a new frame map.
    ///
    /// # Arguments
    ///
    /// * `proxy_fps`    — Frame rate of the proxy (must be > 0).
    /// * `original_fps` — Frame rate of the original (must be > 0).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if either frame rate is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_proxy::frame_map::ProxyFrameMap;
    ///
    /// // Proxy at 24 fps, original at 48 fps
    /// let map = ProxyFrameMap::new(24.0, 48.0);
    /// assert_eq!(map.proxy_frame_to_original(1), 2);
    /// ```
    #[must_use]
    pub fn new(proxy_fps: f32, original_fps: f32) -> Self {
        debug_assert!(proxy_fps > 0.0, "proxy_fps must be positive");
        debug_assert!(original_fps > 0.0, "original_fps must be positive");
        let proxy_fps = proxy_fps as f64;
        let original_fps = original_fps as f64;
        let ratio = if proxy_fps > 0.0 { original_fps / proxy_fps } else { 1.0 };
        Self { proxy_fps, original_fps, ratio }
    }

    /// Convert a proxy frame number to the nearest original frame number.
    ///
    /// Uses round-to-nearest semantics (ties round away from zero).
    #[must_use]
    pub fn proxy_frame_to_original(&self, n: u64) -> u64 {
        (n as f64 * self.ratio).round() as u64
    }

    /// Convert an original frame number to the nearest proxy frame number.
    #[must_use]
    pub fn original_frame_to_proxy(&self, n: u64) -> u64 {
        if self.original_fps <= 0.0 {
            return n;
        }
        (n as f64 / self.ratio).round() as u64
    }

    /// Return the timestamp in seconds for a given proxy frame number.
    #[must_use]
    pub fn proxy_frame_to_seconds(&self, n: u64) -> f64 {
        if self.proxy_fps <= 0.0 {
            return 0.0;
        }
        n as f64 / self.proxy_fps
    }

    /// Return the proxy frame number that corresponds to a given timestamp.
    #[must_use]
    pub fn seconds_to_proxy_frame(&self, seconds: f64) -> u64 {
        (seconds * self.proxy_fps).round() as u64
    }

    /// Return the proxy FPS.
    #[must_use]
    pub fn proxy_fps(&self) -> f32 {
        self.proxy_fps as f32
    }

    /// Return the original FPS.
    #[must_use]
    pub fn original_fps(&self) -> f32 {
        self.original_fps as f32
    }

    /// Return whether the proxy and original share the same frame rate.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        (self.ratio - 1.0).abs() < f64::EPSILON
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_mapping() {
        let map = ProxyFrameMap::new(24.0, 24.0);
        assert!(map.is_identity());
        for n in [0u64, 1, 100, 999] {
            assert_eq!(map.proxy_frame_to_original(n), n);
            assert_eq!(map.original_frame_to_proxy(n), n);
        }
    }

    #[test]
    fn half_rate_proxy() {
        // Proxy at 12 fps, original at 24 fps → each proxy frame = 2 original frames
        let map = ProxyFrameMap::new(12.0, 24.0);
        assert_eq!(map.proxy_frame_to_original(0), 0);
        assert_eq!(map.proxy_frame_to_original(1), 2);
        assert_eq!(map.proxy_frame_to_original(5), 10);
        assert_eq!(map.proxy_frame_to_original(100), 200);
    }

    #[test]
    fn double_rate_original() {
        // Proxy at 48 fps, original at 24 fps → proxy frame n maps to original frame n/2
        let map = ProxyFrameMap::new(48.0, 24.0);
        assert_eq!(map.proxy_frame_to_original(0), 0);
        assert_eq!(map.proxy_frame_to_original(2), 1);
        assert_eq!(map.proxy_frame_to_original(10), 5);
    }

    #[test]
    fn inverse_mapping_roundtrip() {
        let map = ProxyFrameMap::new(24.0, 48.0);
        for n in [0u64, 1, 5, 100] {
            let orig = map.proxy_frame_to_original(n);
            let back = map.original_frame_to_proxy(orig);
            assert_eq!(back, n, "roundtrip failed for proxy frame {n}");
        }
    }

    #[test]
    fn timestamp_conversion() {
        let map = ProxyFrameMap::new(24.0, 24.0);
        assert!((map.proxy_frame_to_seconds(24) - 1.0).abs() < 0.001);
        assert!((map.proxy_frame_to_seconds(48) - 2.0).abs() < 0.001);
    }

    #[test]
    fn seconds_to_proxy_frame() {
        let map = ProxyFrameMap::new(25.0, 50.0);
        assert_eq!(map.seconds_to_proxy_frame(1.0), 25);
        assert_eq!(map.seconds_to_proxy_frame(2.0), 50);
        assert_eq!(map.seconds_to_proxy_frame(0.0), 0);
    }

    #[test]
    fn fps_accessors() {
        let map = ProxyFrameMap::new(29.97, 59.94);
        assert!((map.proxy_fps() - 29.97).abs() < 0.01);
        assert!((map.original_fps() - 59.94).abs() < 0.01);
    }

    #[test]
    fn non_integer_ratio() {
        // 23.976 → 47.952 (2× ratio)
        let map = ProxyFrameMap::new(23.976, 47.952);
        assert_eq!(map.proxy_frame_to_original(1), 2);
        assert_eq!(map.proxy_frame_to_original(10), 20);
    }
}
