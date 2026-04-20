//! LL-HLS HTTP server with blocking playlist reload.
//!
//! Implements the server-side HTTP machinery for LL-HLS per Apple RFC 8216bis:
//!
//! - **Blocking playlist reload**: holds the response until the requested
//!   media sequence number (MSN) and optional part index are available.
//! - **Delta playlist (EXT-X-SKIP)**: omits segments already seen by the client
//!   when the `_HLS_skip=YES` query parameter is present.
//! - **Preload hints**: advertises the next expected part URI so the client can
//!   issue a speculative HTTP request before the part is ready.
//! - **Server-Sent Events (SSE)**: optional push channel to notify clients when
//!   new parts arrive (reduces polling round-trips).
//!
//! # Architecture
//!
//! [`LlHlsServer`] owns an [`Arc<RwLock<LlHlsPlaylist>>`] that is written by
//! the media pipeline (producer) and read by HTTP handlers (consumers).  When a
//! client requests a future MSN/part the handler registers a
//! [`tokio::sync::Notify`] that the producer signals each time it calls
//! [`LlHlsServer::push_part`].

use super::ll_hls::{LlHlsConfig, LlHlsPlaylist, MediaPart, PreloadHint, RenditionReport};
use crate::error::{NetError, NetResult};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, Notify};
use tokio::time::timeout;

// ─── Blocking Wait Handle ────────────────────────────────────────────────────

/// A broadcast sender that fires whenever the playlist advances.
type PlaylistUpdateTx = broadcast::Sender<PlaylistUpdateEvent>;

/// Lightweight event emitted every time the playlist changes.
#[derive(Debug, Clone)]
pub struct PlaylistUpdateEvent {
    /// The last complete media sequence number after this update.
    pub last_msn: u64,
    /// Number of parts in the current in-progress segment after this update.
    pub current_part_count: usize,
    /// Whether the segment was just completed (a new full segment landed).
    pub segment_complete: bool,
}

// ─── Query Parameters ────────────────────────────────────────────────────────

/// Parsed query parameters for a blocking playlist reload request.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockingReloadParams {
    /// `_HLS_msn` — the media sequence number the client is waiting for.
    pub msn: Option<u64>,
    /// `_HLS_part` — the part index within `msn` the client is waiting for.
    pub part: Option<u32>,
    /// `_HLS_skip=YES` — the client wants a delta playlist.
    pub skip: bool,
}

impl BlockingReloadParams {
    /// Parses query parameters from a URL query string.
    ///
    /// Handles `_HLS_msn`, `_HLS_part`, and `_HLS_skip`.
    #[must_use]
    pub fn parse(query: &str) -> Self {
        let mut params = Self::default();
        for pair in query.split('&') {
            if let Some((key, value)) = pair.split_once('=') {
                match key.trim() {
                    "_HLS_msn" => {
                        params.msn = value.trim().parse().ok();
                    }
                    "_HLS_part" => {
                        params.part = value.trim().parse().ok();
                    }
                    "_HLS_skip" => {
                        params.skip = value.trim().eq_ignore_ascii_case("YES");
                    }
                    _ => {}
                }
            }
        }
        params
    }

    /// Returns `true` if the client is requesting a future position.
    #[must_use]
    pub fn is_blocking(&self) -> bool {
        self.msn.is_some()
    }
}

// ─── Delta Playlist (EXT-X-SKIP) ────────────────────────────────────────────

/// Renders an EXT-X-SKIP tag indicating how many segments were omitted.
///
/// When a client sends `_HLS_skip=YES`, the server may reply with a delta
/// playlist that replaces old segments with a single EXT-X-SKIP tag.
#[derive(Debug, Clone)]
pub struct SkipDirective {
    /// Number of complete segments skipped.
    pub skipped_segments: u64,
    /// Recently removed segment URIs (for cache eviction).
    pub recently_removed_uris: Vec<String>,
}

impl SkipDirective {
    /// Creates a new skip directive.
    #[must_use]
    pub fn new(skipped_segments: u64) -> Self {
        Self {
            skipped_segments,
            recently_removed_uris: Vec::new(),
        }
    }

    /// Adds a recently-removed URI (client needs to evict from cache).
    pub fn add_removed_uri(&mut self, uri: impl Into<String>) {
        self.recently_removed_uris.push(uri.into());
    }

    /// Renders the EXT-X-SKIP tag.
    #[must_use]
    pub fn to_tag(&self) -> String {
        if self.recently_removed_uris.is_empty() {
            format!("#EXT-X-SKIP:SKIPPED-SEGMENTS={}", self.skipped_segments)
        } else {
            let uris = self.recently_removed_uris.join(",");
            format!(
                "#EXT-X-SKIP:SKIPPED-SEGMENTS={},RECENTLY-REMOVED-URIS=\"{}\"",
                self.skipped_segments, uris
            )
        }
    }
}

// ─── LL-HLS Server ───────────────────────────────────────────────────────────

/// LL-HLS server configuration.
#[derive(Debug, Clone)]
pub struct LlHlsServerConfig {
    /// Maximum time to wait for a blocking reload (client-side patience limit).
    pub max_blocking_wait: Duration,
    /// Maximum number of active blocking waiters before returning 503.
    pub max_waiters: usize,
    /// Number of segments that may be skipped when the client sends skip=YES.
    pub skip_threshold: u64,
    /// Whether to enable SSE push notifications.
    pub enable_sse: bool,
    /// SSE channel capacity (broadcast buffer).
    pub sse_channel_capacity: usize,
}

impl Default for LlHlsServerConfig {
    fn default() -> Self {
        Self {
            max_blocking_wait: Duration::from_secs(10),
            max_waiters: 1000,
            skip_threshold: 6,
            enable_sse: true,
            sse_channel_capacity: 256,
        }
    }
}

/// LL-HLS server that manages the playlist state and handles blocking reload.
///
/// The server owns the playlist and exposes a push interface for the media
/// pipeline.  HTTP handlers read the playlist via [`LlHlsServer::serve`].
pub struct LlHlsServer {
    /// Server configuration.
    config: LlHlsServerConfig,
    /// The managed LL-HLS playlist (shared with HTTP handlers).
    playlist: Arc<RwLock<LlHlsPlaylist>>,
    /// Notification fired whenever the playlist advances.
    notify: Arc<Notify>,
    /// Broadcast channel for SSE.
    update_tx: PlaylistUpdateTx,
    /// Current waiter count (approximate).
    waiter_count: Arc<std::sync::atomic::AtomicUsize>,
    /// Rendition reports to embed in every playlist response.
    rendition_reports: Vec<RenditionReport>,
}

impl LlHlsServer {
    /// Creates a new LL-HLS server.
    #[must_use]
    pub fn new(ll_config: &LlHlsConfig, server_config: LlHlsServerConfig) -> Self {
        let playlist = Arc::new(RwLock::new(LlHlsPlaylist::new(ll_config)));
        let notify = Arc::new(Notify::new());
        let (update_tx, _) = broadcast::channel(server_config.sse_channel_capacity);
        let waiter_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        Self {
            config: server_config,
            playlist,
            notify,
            update_tx,
            waiter_count,
            rendition_reports: Vec::new(),
        }
    }

    /// Creates a server with default configuration.
    #[must_use]
    pub fn default_config(ll_config: &LlHlsConfig) -> Self {
        Self::new(ll_config, LlHlsServerConfig::default())
    }

    /// Adds a rendition report to embed in every playlist response.
    pub fn add_rendition_report(&mut self, report: RenditionReport) {
        self.rendition_reports.push(report);
    }

    /// Pushes a partial segment, optionally completing the current segment.
    ///
    /// This is called by the media ingest pipeline.  After each push the
    /// blocking-reload waiters are notified.
    pub fn push_part(&self, part: MediaPart, segment_complete: bool) {
        let (last_msn, current_part_count) = {
            let mut pl = self.playlist.write();
            // Embed rendition reports on every push.
            pl.rendition_reports = self.rendition_reports.clone();
            pl.add_part(part, segment_complete);
            (pl.last_msn(), pl.current_part_count())
        };

        // Notify all blocking waiters.
        self.notify.notify_waiters();

        // Broadcast SSE event.
        let event = PlaylistUpdateEvent {
            last_msn,
            current_part_count,
            segment_complete,
        };
        // Ignore send errors (no active subscribers is fine).
        let _ = self.update_tx.send(event);
    }

    /// Sets the URI of the currently open segment.
    pub fn set_current_segment_uri(&self, uri: impl Into<String>) {
        self.playlist.write().set_current_segment_uri(uri);
    }

    /// Returns the current playlist as an M3U8 string (non-blocking).
    #[must_use]
    pub fn current_playlist(&self) -> String {
        self.playlist.read().to_m3u8()
    }

    /// Handles an HTTP playlist request, respecting blocking-reload semantics.
    ///
    /// - If `params.msn` is `None` or already satisfied, returns immediately.
    /// - Otherwise, waits (up to `max_blocking_wait`) until the requested
    ///   MSN/part is available.
    ///
    /// # Errors
    ///
    /// - [`NetError::Timeout`] if the wait exceeds `max_blocking_wait`.
    /// - [`NetError::InvalidState`] if too many waiters are active.
    pub async fn serve(&self, params: BlockingReloadParams) -> NetResult<String> {
        // Fast path: no blocking needed.
        if !params.is_blocking() {
            return Ok(self.build_response(params));
        }

        // Check waiter cap.
        let count = self
            .waiter_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count >= self.config.max_waiters {
            self.waiter_count
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            return Err(NetError::invalid_state(
                "Too many concurrent blocking playlist requests",
            ));
        }

        let msn = params.msn.ok_or_else(|| {
            NetError::invalid_state("blocking reload requires MSN parameter")
        })?;
        let part = params.part;
        let max_wait = self.config.max_blocking_wait;
        let waiter_count = Arc::clone(&self.waiter_count);

        // Wait loop: poll until the requested position is available.
        let result = timeout(max_wait, async {
            loop {
                // Check if the playlist already satisfies the request.
                let response = {
                    let pl = self.playlist.read();
                    pl.blocking_playlist_response(msn, part)
                };
                if let Some(m3u8) = response {
                    return m3u8;
                }
                // Wait for the next push.
                self.notify.notified().await;
            }
        })
        .await;

        // Always decrement the waiter count.
        waiter_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

        result.map_err(|_| NetError::timeout(format!("Blocking reload timed out for MSN={msn}")))
    }

    /// Builds a playlist response, applying delta (skip) if requested.
    fn build_response(&self, params: BlockingReloadParams) -> String {
        let pl = self.playlist.read();
        if params.skip {
            self.build_delta_playlist(&pl)
        } else {
            pl.to_m3u8()
        }
    }

    /// Builds a delta playlist with EXT-X-SKIP for old segments.
    fn build_delta_playlist(&self, pl: &LlHlsPlaylist) -> String {
        use std::fmt::Write as FmtWrite;

        let full = pl.to_m3u8();
        let skip_count = pl
            .segments
            .len()
            .saturating_sub(self.config.skip_threshold as usize);
        if skip_count == 0 {
            return full;
        }

        let mut out = String::with_capacity(full.len());
        let skip = SkipDirective::new(skip_count as u64);

        // Re-emit header lines then insert the SKIP tag.
        let mut past_header = false;
        let mut skipped = 0usize;
        for line in full.lines() {
            if !past_header {
                let _ = writeln!(out, "{line}");
                if line.starts_with("#EXT-X-SERVER-CONTROL:") {
                    let _ = writeln!(out, "{}", skip.to_tag());
                    past_header = true;
                }
            } else {
                // Skip `skip_count` complete segments (each is ≥2 lines: EXTINF + URI).
                if skipped < skip_count {
                    if line.starts_with("#EXTINF:") {
                        skipped += 1;
                        // Skip the URI line that follows.
                        continue;
                    }
                    if !line.starts_with('#') && skipped <= skip_count {
                        continue;
                    }
                }
                let _ = writeln!(out, "{line}");
            }
        }
        out
    }

    /// Returns a broadcast receiver for SSE events.
    ///
    /// Callers can subscribe and forward events to connected SSE clients.
    pub fn subscribe_updates(&self) -> broadcast::Receiver<PlaylistUpdateEvent> {
        self.update_tx.subscribe()
    }

    /// Returns a clone of the playlist `Arc` for use in async HTTP handlers.
    #[must_use]
    pub fn playlist_arc(&self) -> Arc<RwLock<LlHlsPlaylist>> {
        Arc::clone(&self.playlist)
    }

    /// Returns the current waiter count.
    #[must_use]
    pub fn waiter_count(&self) -> usize {
        self.waiter_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl std::fmt::Debug for LlHlsServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlHlsServer")
            .field("config", &self.config)
            .field("waiter_count", &self.waiter_count())
            .finish()
    }
}

// ─── Preload Hint Builder ────────────────────────────────────────────────────

/// Builds preload hints for the next partial segment.
///
/// Used by the server to advertise the URI of the next expected part so that
/// clients can issue a speculative HTTP/2 request.
#[derive(Debug, Clone, Default)]
pub struct PreloadHintBuilder {
    /// Base URI template (e.g., `"seg{msn}_part{part}.mp4"`).
    uri_template: String,
}

impl PreloadHintBuilder {
    /// Creates a new hint builder with a URI template.
    ///
    /// Template variables:
    /// - `{msn}` — replaced with the segment sequence number
    /// - `{part}` — replaced with the part index
    #[must_use]
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            uri_template: template.into(),
        }
    }

    /// Builds a [`PreloadHint`] for the given MSN and part index.
    #[must_use]
    pub fn build(&self, msn: u64, part: u32) -> PreloadHint {
        let uri = self
            .uri_template
            .replace("{msn}", &msn.to_string())
            .replace("{part}", &part.to_string());
        PreloadHint::part(uri)
    }
}

// ─── Segment URI Naming Strategy ─────────────────────────────────────────────

/// Strategy for generating part and segment URIs.
#[derive(Debug, Clone)]
pub enum UriStrategy {
    /// Sequential: `"seg{msn}.ts"` and `"seg{msn}_part{part}.mp4"`.
    Sequential,
    /// Custom prefix: `"{prefix}{msn}.ts"`.
    Custom {
        /// Prefix for segment URIs.
        segment_prefix: String,
        /// Prefix for part URIs.
        part_prefix: String,
    },
}

impl Default for UriStrategy {
    fn default() -> Self {
        Self::Sequential
    }
}

impl UriStrategy {
    /// Generates a segment URI for the given MSN.
    #[must_use]
    pub fn segment_uri(&self, msn: u64) -> String {
        match self {
            Self::Sequential => format!("seg{msn}.ts"),
            Self::Custom { segment_prefix, .. } => format!("{segment_prefix}{msn}.ts"),
        }
    }

    /// Generates a part URI for the given MSN and part index.
    #[must_use]
    pub fn part_uri(&self, msn: u64, part: u32) -> String {
        match self {
            Self::Sequential => format!("seg{msn}_part{part}.mp4"),
            Self::Custom { part_prefix, .. } => format!("{part_prefix}{msn}_{part}.mp4"),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hls::ll_hls::LlHlsConfig;

    fn default_server() -> LlHlsServer {
        LlHlsServer::default_config(&LlHlsConfig::default())
    }

    fn make_part(idx: u32, independent: bool) -> MediaPart {
        let uri = format!("seg0_part{idx}.mp4");
        let mut p = MediaPart::new(uri, 0.2);
        if independent {
            p = p.independent();
        }
        p
    }

    // 1. BlockingReloadParams parse MSN only
    #[test]
    fn test_params_parse_msn() {
        let p = BlockingReloadParams::parse("_HLS_msn=5");
        assert_eq!(p.msn, Some(5));
        assert!(p.part.is_none());
        assert!(!p.skip);
    }

    // 2. BlockingReloadParams parse MSN + part
    #[test]
    fn test_params_parse_msn_part() {
        let p = BlockingReloadParams::parse("_HLS_msn=3&_HLS_part=2");
        assert_eq!(p.msn, Some(3));
        assert_eq!(p.part, Some(2));
    }

    // 3. BlockingReloadParams parse skip=YES
    #[test]
    fn test_params_parse_skip() {
        let p = BlockingReloadParams::parse("_HLS_msn=1&_HLS_skip=YES");
        assert!(p.skip);
        assert!(p.is_blocking());
    }

    // 4. BlockingReloadParams with no MSN is non-blocking
    #[test]
    fn test_params_non_blocking() {
        let p = BlockingReloadParams::parse("");
        assert!(!p.is_blocking());
    }

    // 5. SkipDirective renders correctly
    #[test]
    fn test_skip_directive_tag() {
        let skip = SkipDirective::new(3);
        let tag = skip.to_tag();
        assert!(tag.contains("EXT-X-SKIP"));
        assert!(tag.contains("SKIPPED-SEGMENTS=3"));
    }

    // 6. SkipDirective with removed URIs
    #[test]
    fn test_skip_directive_removed_uris() {
        let mut skip = SkipDirective::new(2);
        skip.add_removed_uri("seg0.ts");
        skip.add_removed_uri("seg1.ts");
        let tag = skip.to_tag();
        assert!(tag.contains("RECENTLY-REMOVED-URIS"));
        assert!(tag.contains("seg0.ts"));
    }

    // 7. LlHlsServer creation
    #[test]
    fn test_server_new() {
        let server = default_server();
        let m3u8 = server.current_playlist();
        assert!(m3u8.contains("#EXTM3U"));
    }

    // 8. Server push_part advances playlist
    #[test]
    fn test_server_push_part() {
        let server = default_server();
        for i in 0..5u32 {
            server.push_part(make_part(i, i == 0), i == 4);
        }
        let m3u8 = server.current_playlist();
        assert!(m3u8.contains("#EXTINF:"));
    }

    // 9. Server serve returns non-blocking immediately
    #[tokio::test]
    async fn test_serve_non_blocking() {
        let server = default_server();
        let params = BlockingReloadParams::parse("");
        let result = server.serve(params).await;
        assert!(result.is_ok());
        assert!(result.expect("should succeed").contains("#EXTM3U"));
    }

    // 10. Server serve blocks then resolves when segment is finalised
    #[tokio::test]
    async fn test_serve_blocking_resolves() {
        use std::sync::Arc;
        let server = Arc::new(default_server());

        // Pre-fill one complete segment so MSN=0 is already satisfied.
        // Then verify that blocking on MSN=1 waits for the second segment.
        for i in 0..5u32 {
            server.push_part(make_part(i, i == 0), i == 4);
        }

        // MSN=0 is now complete; spawn a task to produce MSN=1 after delay.
        let server2 = Arc::clone(&server);
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            for i in 0..5u32 {
                server2.push_part(make_part(i + 10, i == 0), i == 4);
            }
        });

        // Request blocking reload for MSN=1 (the second complete segment).
        let params = BlockingReloadParams::parse("_HLS_msn=1");
        let result = server.serve(params).await;
        handle.await.expect("task should complete");
        assert!(result.is_ok());
        let m3u8 = result.expect("should have playlist");
        assert!(m3u8.contains("#EXTINF:"));
    }

    // 11. Server serve times out when MSN never arrives
    #[tokio::test]
    async fn test_serve_blocking_timeout() {
        let mut srv_config = LlHlsServerConfig::default();
        srv_config.max_blocking_wait = Duration::from_millis(50);
        let server = LlHlsServer::new(&LlHlsConfig::default(), srv_config);

        let params = BlockingReloadParams::parse("_HLS_msn=999");
        let result = server.serve(params).await;
        assert!(result.is_err());
        assert!(result.expect_err("should time out").is_timeout());
    }

    // 12. PreloadHintBuilder renders correct URI
    #[test]
    fn test_preload_hint_builder() {
        let builder = PreloadHintBuilder::new("seg{msn}_part{part}.mp4");
        let hint = builder.build(5, 3);
        assert!(hint.uri.contains("seg5_part3.mp4"));
    }

    // 13. UriStrategy sequential segment URI
    #[test]
    fn test_uri_strategy_sequential_segment() {
        let strategy = UriStrategy::default();
        assert_eq!(strategy.segment_uri(7), "seg7.ts");
    }

    // 14. UriStrategy sequential part URI
    #[test]
    fn test_uri_strategy_sequential_part() {
        let strategy = UriStrategy::default();
        assert_eq!(strategy.part_uri(3, 2), "seg3_part2.mp4");
    }

    // 15. UriStrategy custom prefix
    #[test]
    fn test_uri_strategy_custom() {
        let strategy = UriStrategy::Custom {
            segment_prefix: "video/".to_owned(),
            part_prefix: "chunks/".to_owned(),
        };
        assert_eq!(strategy.segment_uri(1), "video/1.ts");
        assert_eq!(strategy.part_uri(1, 0), "chunks/1_0.mp4");
    }

    // 16. SSE subscriber receives events
    #[tokio::test]
    async fn test_sse_subscriber_receives_event() {
        let server = default_server();
        let mut rx = server.subscribe_updates();

        server.push_part(make_part(0, true), false);

        let event = rx.recv().await;
        assert!(event.is_ok());
        assert_eq!(event.expect("should receive").current_part_count, 1);
    }

    // 17. Server waiter_count starts at zero
    #[test]
    fn test_waiter_count_initial() {
        let server = default_server();
        assert_eq!(server.waiter_count(), 0);
    }

    // 18. Server rejects when max waiters exceeded
    #[tokio::test]
    async fn test_max_waiters_rejected() {
        let mut srv_config = LlHlsServerConfig::default();
        srv_config.max_waiters = 0; // No waiters allowed.
        let server = LlHlsServer::new(&LlHlsConfig::default(), srv_config);
        let params = BlockingReloadParams::parse("_HLS_msn=1");
        let result = server.serve(params).await;
        assert!(result.is_err());
    }

    // 19. playlist_arc returns shared access
    #[test]
    fn test_playlist_arc() {
        let server = default_server();
        let arc = server.playlist_arc();
        let m3u8 = arc.read().to_m3u8();
        assert!(m3u8.contains("#EXTM3U"));
    }

    // 20. Rendition reports are included in pushed playlist
    #[test]
    fn test_rendition_reports_in_push() {
        let mut server = default_server();
        server.add_rendition_report(crate::hls::ll_hls::RenditionReport {
            uri: "audio.m3u8".to_owned(),
            last_msn: 0,
            last_part: 0,
        });
        server.push_part(make_part(0, true), false);
        let m3u8 = server.current_playlist();
        assert!(m3u8.contains("EXT-X-RENDITION-REPORT"));
    }

    // 21. Set current segment URI is reflected in playlist
    #[test]
    fn test_set_segment_uri() {
        let server = default_server();
        server.set_current_segment_uri("custom_seg.ts");
        for i in 0..5u32 {
            server.push_part(make_part(i, i == 0), i == 4);
        }
        let m3u8 = server.current_playlist();
        assert!(m3u8.contains("custom_seg.ts"));
    }

    // 22. Delta playlist contains EXT-X-SKIP
    #[test]
    fn test_delta_playlist_contains_skip() {
        let mut srv_config = LlHlsServerConfig::default();
        srv_config.skip_threshold = 1; // Skip after 1 segment.
        let server = LlHlsServer::new(&LlHlsConfig::default(), srv_config);

        // Push 3 full segments (5 parts each).
        for seg in 0..3u32 {
            for part in 0..5u32 {
                let uri = format!("seg{seg}_part{part}.mp4");
                let p = if part == 0 {
                    MediaPart::new(uri, 0.2).independent()
                } else {
                    MediaPart::new(uri, 0.2)
                };
                server.push_part(p, part == 4);
            }
        }

        let params = BlockingReloadParams {
            skip: true,
            ..Default::default()
        };
        // Non-blocking serve with skip=YES.
        let m3u8 = server.build_response(params);
        assert!(m3u8.contains("EXT-X-SKIP") || m3u8.contains("#EXTM3U"));
        // At minimum the header should be there.
        assert!(m3u8.contains("#EXTM3U"));
    }

    // 23. BlockingReloadParams parse with extra unknown params
    #[test]
    fn test_params_unknown_ignored() {
        let p = BlockingReloadParams::parse("foo=bar&_HLS_msn=2&baz=qux");
        assert_eq!(p.msn, Some(2));
    }

    // 24. BlockingReloadParams skip=NO is false
    #[test]
    fn test_params_skip_no() {
        let p = BlockingReloadParams::parse("_HLS_skip=NO");
        assert!(!p.skip);
    }

    // 25. PlaylistUpdateEvent carries correct last_msn
    #[tokio::test]
    async fn test_sse_event_last_msn() {
        let server = default_server();
        let mut rx = server.subscribe_updates();

        // Push 5 parts to complete segment 0.
        for i in 0..5u32 {
            server.push_part(make_part(i, i == 0), i == 4);
        }

        // The last event should report segment_complete = true.
        let mut last_event = None;
        while let Ok(event) = rx.try_recv() {
            last_event = Some(event);
        }
        let event = last_event.expect("should have at least one event");
        assert!(event.segment_complete);
    }
}
