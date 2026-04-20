//! HLS segment fetching and caching.
//!
//! This module provides types for downloading and caching HLS segments,
//! including:
//! - HTTP/2 push hint support
//! - Next-segment prefetch pipeline
//! - Partial segment support for Low-Latency HLS (LL-HLS)
//! - LRU-evicting segment cache with TTL

#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::if_not_else)]
#![allow(clippy::format_push_string)]
#![allow(clippy::single_match_else)]
#![allow(clippy::redundant_slicing)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::format_collect)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unused_async)]
#![allow(clippy::identity_op)]

use crate::error::{NetError, NetResult};
use bytes::Bytes;
use reqwest::Client;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

// ─────────────────────────────────────────────────────────────────────────────
// ByteRange
// ─────────────────────────────────────────────────────────────────────────────

/// Byte range for partial content requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    /// Start offset in bytes.
    pub start: u64,
    /// End offset in bytes (inclusive).
    pub end: u64,
}

impl ByteRange {
    /// Creates a new byte range.
    #[must_use]
    pub const fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    /// Creates a byte range from offset and length.
    #[must_use]
    pub const fn from_offset_length(offset: u64, length: u64) -> Self {
        Self {
            start: offset,
            end: offset + length - 1,
        }
    }

    /// Returns the length of this byte range.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.end - self.start + 1
    }

    /// Returns true if this is an empty range.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.start > self.end
    }

    /// Formats as HTTP Range header value.
    #[must_use]
    pub fn to_range_header(&self) -> String {
        format!("bytes={}-{}", self.start, self.end)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FetchConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for segment fetching.
#[derive(Debug, Clone)]
pub struct FetchConfig {
    /// Maximum number of retries.
    pub max_retries: u32,
    /// Initial retry delay.
    pub retry_delay: Duration,
    /// Request timeout.
    pub timeout: Duration,
    /// Maximum segment size.
    pub max_segment_size: usize,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay: Duration::from_millis(500),
            timeout: Duration::from_secs(30),
            max_segment_size: 50 * 1024 * 1024, // 50 MB
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FetchResult
// ─────────────────────────────────────────────────────────────────────────────

/// Fetch result with metadata.
#[derive(Debug, Clone)]
pub struct FetchResult {
    /// Downloaded data.
    pub data: Bytes,
    /// Content type.
    pub content_type: Option<String>,
    /// Actual content length.
    pub content_length: usize,
    /// Time taken to fetch.
    pub fetch_time: Duration,
    /// Bytes per second throughput.
    pub throughput: f64,
    /// Whether this result came from the local cache.
    pub from_cache: bool,
    /// Whether this is a partial segment (LL-HLS).
    pub is_partial: bool,
    /// Byte range actually received (for partial responses).
    pub byte_range: Option<ByteRange>,
}

impl FetchResult {
    /// Creates a new fetch result.
    #[must_use]
    pub fn new(data: Bytes, fetch_time: Duration) -> Self {
        let content_length = data.len();
        let throughput = if fetch_time.as_secs_f64() > 0.0 {
            content_length as f64 / fetch_time.as_secs_f64()
        } else {
            0.0
        };

        Self {
            data,
            content_type: None,
            content_length,
            fetch_time,
            throughput,
            from_cache: false,
            is_partial: false,
            byte_range: None,
        }
    }

    /// Returns throughput in bits per second.
    #[must_use]
    pub fn throughput_bps(&self) -> f64 {
        self.throughput * 8.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PartialSegment — LL-HLS chunk descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// A partial segment descriptor for Low-Latency HLS.
///
/// LL-HLS splits each full segment into smaller "Parts" that are published
/// as they are encoded.  Each part is identified by the containing segment URI
/// plus a byte range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialSegment {
    /// URI of the containing full segment.
    pub segment_uri: String,
    /// Byte range within the segment file.
    pub byte_range: ByteRange,
    /// Nominal duration of this part in seconds (from EXT-X-PART).
    pub duration_secs: u32,
    /// Whether this part is the last in its segment.
    pub is_last: bool,
    /// Sequence number of the parent segment.
    pub media_sequence: u64,
    /// Part index within the parent segment (0-based).
    pub part_index: u32,
}

impl PartialSegment {
    /// Creates a new partial segment descriptor.
    #[must_use]
    pub fn new(
        segment_uri: impl Into<String>,
        byte_range: ByteRange,
        duration_secs: u32,
        media_sequence: u64,
        part_index: u32,
    ) -> Self {
        Self {
            segment_uri: segment_uri.into(),
            byte_range,
            duration_secs,
            is_last: false,
            media_sequence,
            part_index,
        }
    }

    /// Marks this as the last part of its segment.
    #[must_use]
    pub const fn as_last(mut self) -> Self {
        self.is_last = true;
        self
    }

    /// Returns a cache key unique to this partial segment.
    #[must_use]
    pub fn cache_key(&self) -> String {
        format!(
            "{}#{}-{}",
            self.segment_uri, self.byte_range.start, self.byte_range.end
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PrefetchQueue — next-segment pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Status of a prefetch slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefetchStatus {
    /// Queued but not yet started.
    Pending,
    /// Currently being fetched.
    InFlight,
    /// Successfully downloaded.
    Ready,
    /// Fetch failed.
    Failed(String),
}

/// A single entry in the prefetch queue.
#[derive(Debug, Clone)]
pub struct PrefetchEntry {
    /// Segment URI.
    pub uri: String,
    /// Optional byte range (for LL-HLS parts).
    pub byte_range: Option<ByteRange>,
    /// Current status.
    pub status: PrefetchStatus,
    /// Downloaded data (available when `status == Ready`).
    pub data: Option<Bytes>,
    /// When the fetch was queued.
    pub queued_at: Instant,
}

impl PrefetchEntry {
    fn new(uri: impl Into<String>, byte_range: Option<ByteRange>) -> Self {
        Self {
            uri: uri.into(),
            byte_range,
            status: PrefetchStatus::Pending,
            data: None,
            queued_at: Instant::now(),
        }
    }
}

/// Prefetch queue that keeps the next `capacity` segments pre-downloaded.
///
/// When the player moves past a segment the queue slides forward, evicting
/// consumed entries and scheduling new prefetches.
pub struct PrefetchQueue {
    /// Maximum number of segments to keep pre-fetched.
    capacity: usize,
    /// Entries in prefetch order.
    entries: VecDeque<PrefetchEntry>,
}

impl PrefetchQueue {
    /// Creates a new prefetch queue.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::with_capacity(capacity),
        }
    }

    /// Enqueues a segment for prefetch.  If the queue is full, the oldest
    /// entry is dropped.
    pub fn enqueue(&mut self, uri: impl Into<String>, byte_range: Option<ByteRange>) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(PrefetchEntry::new(uri, byte_range));
    }

    /// Marks the first `Pending` entry as `InFlight`.
    /// Returns a clone of the entry to fetch, or `None` if nothing is pending.
    pub fn next_to_fetch(&mut self) -> Option<PrefetchEntry> {
        for entry in self.entries.iter_mut() {
            if entry.status == PrefetchStatus::Pending {
                entry.status = PrefetchStatus::InFlight;
                return Some(entry.clone());
            }
        }
        None
    }

    /// Records the result of a fetch attempt.
    pub fn complete(&mut self, uri: &str, result: Result<Bytes, String>) {
        for entry in self.entries.iter_mut() {
            if entry.uri == uri && entry.status == PrefetchStatus::InFlight {
                match result {
                    Ok(data) => {
                        entry.status = PrefetchStatus::Ready;
                        entry.data = Some(data);
                    }
                    Err(e) => {
                        entry.status = PrefetchStatus::Failed(e);
                    }
                }
                return;
            }
        }
    }

    /// Pops the next ready segment from the front of the queue.
    pub fn pop_ready(&mut self) -> Option<(String, Bytes)> {
        if let Some(front) = self.entries.front() {
            if front.status == PrefetchStatus::Ready {
                // Front was confirmed by `if let Some(front)`.
                let entry = self.entries.pop_front()?;
                let data = entry.data?;
                return Some((entry.uri, data));
            }
        }
        None
    }

    /// Returns the number of entries in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true when the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns how many entries are currently `Ready`.
    #[must_use]
    pub fn ready_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.status == PrefetchStatus::Ready)
            .count()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP/2 push hint
// ─────────────────────────────────────────────────────────────────────────────

/// An HTTP/2 server-push hint embedded in a `Link` header.
///
/// When a server sends a `Link: <seg.ts>; rel=preload` header alongside the
/// playlist response, the client can begin fetching the referenced resource
/// immediately — before parsing the playlist body — reducing latency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Http2PushHint {
    /// URI of the resource being pushed.
    pub uri: String,
    /// `as` attribute (e.g. `"video"`, `"fetch"`).
    pub resource_type: Option<String>,
    /// Whether the push should be treated as a CORS request.
    pub crossorigin: bool,
}

impl Http2PushHint {
    /// Creates a new push hint.
    #[must_use]
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            resource_type: None,
            crossorigin: false,
        }
    }

    /// Parses `Link` header values into push hints.
    ///
    /// Accepts a comma-separated list such as:
    /// `<seg0.ts>; rel=preload; as=fetch, <seg1.ts>; rel=preload`
    #[must_use]
    pub fn parse_link_header(header: &str) -> Vec<Self> {
        let mut hints = Vec::new();
        for part in header.split(',') {
            let part = part.trim();
            let mut segments = part.split(';');
            let uri_part = match segments.next() {
                Some(u) => u.trim(),
                None => continue,
            };

            // URI must be enclosed in angle brackets.
            if !uri_part.starts_with('<') || !uri_part.ends_with('>') {
                continue;
            }
            let uri = uri_part[1..uri_part.len() - 1].to_string();

            let mut is_preload = false;
            let mut resource_type = None;
            let mut crossorigin = false;

            for attr in segments {
                let attr = attr.trim();
                if attr.eq_ignore_ascii_case("rel=preload") {
                    is_preload = true;
                } else if let Some(rest) = attr.strip_prefix("as=") {
                    resource_type = Some(rest.trim_matches('"').to_string());
                } else if attr.eq_ignore_ascii_case("crossorigin") {
                    crossorigin = true;
                }
            }

            if is_preload {
                hints.push(Self {
                    uri,
                    resource_type,
                    crossorigin,
                });
            }
        }
        hints
    }

    /// Formats this hint as a `Link` header fragment.
    #[must_use]
    pub fn to_link_header(&self) -> String {
        let mut s = format!("<{}>; rel=preload", self.uri);
        if let Some(t) = &self.resource_type {
            s.push_str(&format!("; as={t}"));
        }
        if self.crossorigin {
            s.push_str("; crossorigin");
        }
        s
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SegmentFetcher — HTTP fetching with retry, push hints, and LL-HLS support
// ─────────────────────────────────────────────────────────────────────────────

/// Segment fetcher for downloading HLS segments.
#[derive(Debug)]
pub struct SegmentFetcher {
    /// Base URL for resolving relative URIs.
    base_url: Option<String>,
    /// Fetch configuration.
    config: FetchConfig,
    /// HTTP client (lazily initialised on first fetch to avoid TLS-provider
    /// panics in environments that have not called `rustls::crypto::ring::default_provider().install_global()`).
    client: Option<Client>,
    /// Total bytes downloaded.
    bytes_downloaded: u64,
    /// Total fetch time.
    total_fetch_time: Duration,
    /// HTTP/2 push hints received from the server during the last fetch.
    push_hints: Vec<Http2PushHint>,
    /// Number of partial (LL-HLS) fetches performed.
    partial_fetches: u64,
}

impl SegmentFetcher {
    /// Creates a new segment fetcher.
    ///
    /// The underlying HTTP client is **not** constructed until the first call
    /// to `fetch` / `fetch_partial` / `fetch_with_retry`, so creating a
    /// `SegmentFetcher` is safe in any context (no Tokio runtime or TLS
    /// provider needed at construction time).
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_url: None,
            config: FetchConfig::default(),
            client: None,
            bytes_downloaded: 0,
            total_fetch_time: Duration::ZERO,
            push_hints: Vec::new(),
            partial_fetches: 0,
        }
    }

    /// Creates a new segment fetcher with a custom HTTP client.
    #[must_use]
    pub fn with_client(client: Client) -> Self {
        Self {
            base_url: None,
            config: FetchConfig::default(),
            client: Some(client),
            bytes_downloaded: 0,
            total_fetch_time: Duration::ZERO,
            push_hints: Vec::new(),
            partial_fetches: 0,
        }
    }

    /// Returns a reference to the HTTP client, building it on first call.
    fn client(&mut self) -> NetResult<&Client> {
        if self.client.is_none() {
            let c = Client::builder()
                .timeout(self.config.timeout)
                .build()
                .map_err(|e| NetError::connection(format!("Failed to build HTTP client: {e}")))?;
            self.client = Some(c);
        }
        self.client
            .as_ref()
            .ok_or_else(|| NetError::connection("HTTP client failed to initialise"))
    }

    /// Creates a fetcher with a base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Sets the fetch configuration.
    #[must_use]
    pub fn with_config(mut self, config: FetchConfig) -> Self {
        self.config = config;
        self
    }

    /// Returns the current configuration.
    #[must_use]
    pub const fn config(&self) -> &FetchConfig {
        &self.config
    }

    /// Resolves a URI against the base URL.
    #[must_use]
    pub fn resolve_url(&self, uri: &str) -> String {
        if uri.starts_with("http://") || uri.starts_with("https://") {
            return uri.to_string();
        }

        match &self.base_url {
            Some(base) => {
                if uri.starts_with('/') {
                    if let Some(pos) = base.find("://") {
                        if let Some(slash_pos) = base[pos + 3..].find('/') {
                            return format!("{}{uri}", &base[..pos + 3 + slash_pos]);
                        }
                    }
                    format!("{base}{uri}")
                } else {
                    if let Some(last_slash) = base.rfind('/') {
                        format!("{}/{uri}", &base[..last_slash])
                    } else {
                        format!("{base}/{uri}")
                    }
                }
            }
            None => uri.to_string(),
        }
    }

    /// Fetches a segment via HTTP.
    ///
    /// # Errors
    ///
    /// Returns an error if the fetch fails after all retries.
    pub async fn fetch(
        &mut self,
        uri: &str,
        byte_range: Option<ByteRange>,
    ) -> NetResult<FetchResult> {
        let url = self.resolve_url(uri);
        let start = Instant::now();
        let is_partial = byte_range.is_some();

        // Ensure the HTTP client is initialised (lazy init on first call).
        let client = self.client()?;
        let mut request = client.get(&url).timeout(self.config.timeout);

        if let Some(range) = byte_range {
            request = request.header("Range", range.to_range_header());
        }

        let response = request.send().await.map_err(|e| {
            if e.is_timeout() {
                NetError::timeout(format!("Request timed out: {url}"))
            } else if e.is_connect() {
                NetError::connection(format!("Connection failed: {e}"))
            } else {
                NetError::connection(format!("Request failed: {e}"))
            }
        })?;

        let status = response.status();
        if !status.is_success() {
            return Err(NetError::http(
                status.as_u16(),
                format!("Failed to fetch segment: {url}"),
            ));
        }

        // Extract content type.
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(String::from);

        // Parse HTTP/2 push hints from Link headers.
        let link_headers: Vec<String> = response
            .headers()
            .get_all("link")
            .iter()
            .filter_map(|v| v.to_str().ok().map(String::from))
            .collect();
        self.push_hints.clear();
        for lh in &link_headers {
            self.push_hints.extend(Http2PushHint::parse_link_header(lh));
        }

        // Check content length.
        let content_length = response.content_length().unwrap_or(0);
        if content_length > self.config.max_segment_size as u64 {
            return Err(NetError::segment(format!(
                "Segment too large: {content_length} bytes exceeds max of {} bytes",
                self.config.max_segment_size
            )));
        }

        let data = response
            .bytes()
            .await
            .map_err(|e| NetError::connection(format!("Failed to read response body: {e}")))?;

        let fetch_time = start.elapsed();

        self.bytes_downloaded += data.len() as u64;
        self.total_fetch_time += fetch_time;
        if is_partial {
            self.partial_fetches += 1;
        }

        let mut result = FetchResult::new(data, fetch_time);
        result.content_type = content_type;
        result.is_partial = is_partial;
        result.byte_range = byte_range;

        Ok(result)
    }

    /// Fetches a partial segment (LL-HLS `EXT-X-PART`).
    ///
    /// This is a convenience wrapper that sets `is_partial = true` on the
    /// result and supplies the byte range from a `PartialSegment` descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if the fetch fails.
    pub async fn fetch_partial(&mut self, part: &PartialSegment) -> NetResult<FetchResult> {
        self.fetch(&part.segment_uri, Some(part.byte_range)).await
    }

    /// Fetches a segment with retries.
    ///
    /// # Errors
    ///
    /// Returns an error if all retry attempts fail.
    pub async fn fetch_with_retry(
        &mut self,
        uri: &str,
        byte_range: Option<ByteRange>,
    ) -> NetResult<FetchResult> {
        let mut last_error = None;
        let mut delay = self.config.retry_delay;

        for attempt in 0..=self.config.max_retries {
            match self.fetch(uri, byte_range).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.config.max_retries {
                        tokio::time::sleep(delay).await;
                        delay *= 2;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| NetError::segment("Unknown fetch error")))
    }

    /// Returns total bytes downloaded.
    #[must_use]
    pub const fn bytes_downloaded(&self) -> u64 {
        self.bytes_downloaded
    }

    /// Returns average throughput in bytes per second.
    #[must_use]
    pub fn average_throughput(&self) -> f64 {
        if self.total_fetch_time.as_secs_f64() > 0.0 {
            self.bytes_downloaded as f64 / self.total_fetch_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Resets statistics.
    pub fn reset_stats(&mut self) {
        self.bytes_downloaded = 0;
        self.total_fetch_time = Duration::ZERO;
        self.partial_fetches = 0;
        self.push_hints.clear();
    }

    /// Returns the HTTP/2 push hints parsed from the most recent response.
    #[must_use]
    pub fn push_hints(&self) -> &[Http2PushHint] {
        &self.push_hints
    }

    /// Returns the number of partial (LL-HLS) segment fetches performed.
    #[must_use]
    pub const fn partial_fetches(&self) -> u64 {
        self.partial_fetches
    }
}

impl Default for SegmentFetcher {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SegmentCache — LRU cache with TTL
// ─────────────────────────────────────────────────────────────────────────────

/// Cached segment entry.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Segment data.
    data: Bytes,
    /// When the entry was added.
    added_at: Instant,
    /// Last access time.
    last_accessed: Instant,
    /// Access count.
    access_count: u32,
    /// Whether this entry came from an HTTP/2 push.
    from_push: bool,
}

/// Segment cache for buffering downloaded segments with LRU eviction.
#[derive(Debug)]
pub struct SegmentCache {
    /// Cached segments by URI.
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Maximum cache size in bytes.
    max_size: usize,
    /// Maximum number of entries.
    max_entries: usize,
    /// Time-to-live for entries.
    ttl: Duration,
}

impl SegmentCache {
    /// Creates a new segment cache.
    #[must_use]
    pub fn new(max_size: usize, max_entries: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            max_entries,
            ttl: Duration::from_secs(300), // 5 minutes default
        }
    }

    /// Sets the time-to-live for cache entries.
    #[must_use]
    pub const fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Gets a segment from the cache.
    pub async fn get(&self, uri: &str) -> Option<Bytes> {
        let mut entries = self.entries.write().await;

        if let Some(entry) = entries.get_mut(uri) {
            if entry.added_at.elapsed() > self.ttl {
                entries.remove(uri);
                return None;
            }

            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            Some(entry.data.clone())
        } else {
            None
        }
    }

    /// Puts a segment into the cache.
    pub async fn put(&self, uri: impl Into<String>, data: Bytes) {
        self.put_with_options(uri, data, false).await;
    }

    /// Puts a segment received via HTTP/2 server push.
    pub async fn put_pushed(&self, uri: impl Into<String>, data: Bytes) {
        self.put_with_options(uri, data, true).await;
    }

    async fn put_with_options(&self, uri: impl Into<String>, data: Bytes, from_push: bool) {
        let uri = uri.into();
        let mut entries = self.entries.write().await;

        while entries.len() >= self.max_entries {
            self.evict_one(&mut entries);
        }

        while self.current_size(&entries) + data.len() > self.max_size {
            if !self.evict_one(&mut entries) {
                break;
            }
        }

        let now = Instant::now();
        entries.insert(
            uri,
            CacheEntry {
                data,
                added_at: now,
                last_accessed: now,
                access_count: 1,
                from_push,
            },
        );
    }

    /// Removes a segment from the cache.
    pub async fn remove(&self, uri: &str) -> Option<Bytes> {
        let mut entries = self.entries.write().await;
        entries.remove(uri).map(|e| e.data)
    }

    /// Clears all entries from the cache.
    pub async fn clear(&self) {
        let mut entries = self.entries.write().await;
        entries.clear();
    }

    /// Returns the number of cached segments.
    pub async fn len(&self) -> usize {
        let entries = self.entries.read().await;
        entries.len()
    }

    /// Returns true if the cache is empty.
    pub async fn is_empty(&self) -> bool {
        let entries = self.entries.read().await;
        entries.is_empty()
    }

    /// Returns total cached size in bytes.
    pub async fn size(&self) -> usize {
        let entries = self.entries.read().await;
        self.current_size(&entries)
    }

    /// Evicts expired entries.
    pub async fn evict_expired(&self) {
        let mut entries = self.entries.write().await;
        entries.retain(|_, entry| entry.added_at.elapsed() <= self.ttl);
    }

    /// Returns the number of entries that arrived via HTTP/2 server push.
    pub async fn pushed_count(&self) -> usize {
        let entries = self.entries.read().await;
        entries.values().filter(|e| e.from_push).count()
    }

    /// Returns hit statistics: `(hits, misses)` since the cache was created.
    /// (Not tracked internally; placeholder for instrumentation.)
    #[must_use]
    pub async fn entry_stats(&self) -> Vec<(String, u32)> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .map(|(k, v)| (k.clone(), v.access_count))
            .collect()
    }

    fn current_size(&self, entries: &HashMap<String, CacheEntry>) -> usize {
        entries.values().map(|e| e.data.len()).sum()
    }

    fn evict_one(&self, entries: &mut HashMap<String, CacheEntry>) -> bool {
        // LRU: remove the entry with the oldest `last_accessed` timestamp.
        let oldest = entries
            .iter()
            .min_by_key(|(_, e)| e.last_accessed)
            .map(|(k, _)| k.clone());

        if let Some(key) = oldest {
            entries.remove(&key);
            true
        } else {
            false
        }
    }
}

impl Default for SegmentCache {
    fn default() -> Self {
        Self::new(100 * 1024 * 1024, 100)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ByteRange ─────────────────────────────────────────────────────────────

    #[test]
    fn test_byte_range() {
        let range = ByteRange::new(100, 199);
        assert_eq!(range.len(), 100);
        assert!(!range.is_empty());
        assert_eq!(range.to_range_header(), "bytes=100-199");

        let range2 = ByteRange::from_offset_length(100, 100);
        assert_eq!(range2.start, 100);
        assert_eq!(range2.end, 199);
    }

    #[test]
    fn test_byte_range_empty() {
        let range = ByteRange::new(200, 100); // start > end
        assert!(range.is_empty());
    }

    // ── FetchResult ───────────────────────────────────────────────────────────

    #[test]
    fn test_fetch_result() {
        let data = Bytes::from(vec![0u8; 1000]);
        let result = FetchResult::new(data, Duration::from_secs(1));

        assert_eq!(result.content_length, 1000);
        assert!((result.throughput - 1000.0).abs() < 0.1);
        assert!((result.throughput_bps() - 8000.0).abs() < 0.1);
        assert!(!result.from_cache);
        assert!(!result.is_partial);
    }

    // ── SegmentFetcher URL resolution ─────────────────────────────────────────
    // These tests use #[tokio::test] because SegmentFetcher::new() builds an
    // HTTP client that requires a Tokio reactor to be active.

    #[tokio::test]
    async fn test_resolve_url_absolute() {
        let fetcher = SegmentFetcher::new().with_base_url("https://example.com/stream/");
        let url = fetcher.resolve_url("https://cdn.example.com/seg.ts");
        assert_eq!(url, "https://cdn.example.com/seg.ts");
    }

    #[tokio::test]
    async fn test_resolve_url_relative() {
        let fetcher =
            SegmentFetcher::new().with_base_url("https://example.com/stream/playlist.m3u8");
        let url = fetcher.resolve_url("segment0.ts");
        assert_eq!(url, "https://example.com/stream/segment0.ts");
    }

    #[tokio::test]
    async fn test_resolve_url_absolute_path() {
        let fetcher =
            SegmentFetcher::new().with_base_url("https://example.com/stream/playlist.m3u8");
        let url = fetcher.resolve_url("/media/segment0.ts");
        assert_eq!(url, "https://example.com/media/segment0.ts");
    }

    // ── SegmentCache ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_segment_cache_basic() {
        let cache = SegmentCache::new(1024 * 1024, 10);

        assert!(cache.is_empty().await);
        assert_eq!(cache.len().await, 0);

        cache.put("seg1", Bytes::from(vec![1, 2, 3])).await;
        assert_eq!(cache.len().await, 1);
        assert_eq!(cache.size().await, 3);

        let data = cache.get("seg1").await;
        assert!(data.is_some());
        assert_eq!(data.expect("should succeed in test").as_ref(), &[1, 2, 3]);

        let removed = cache.remove("seg1").await;
        assert!(removed.is_some());
        assert!(cache.is_empty().await);
    }

    #[tokio::test]
    async fn test_segment_cache_eviction() {
        let cache = SegmentCache::new(10, 3);

        cache.put("a", Bytes::from(vec![1, 2, 3])).await;
        cache.put("b", Bytes::from(vec![4, 5, 6])).await;
        cache.put("c", Bytes::from(vec![7, 8, 9])).await;

        assert_eq!(cache.len().await, 3);

        cache.put("d", Bytes::from(vec![10, 11, 12])).await;
        assert_eq!(cache.len().await, 3);
    }

    #[tokio::test]
    async fn test_segment_cache_clear() {
        let cache = SegmentCache::new(1024 * 1024, 10);

        cache.put("a", Bytes::from(vec![1])).await;
        cache.put("b", Bytes::from(vec![2])).await;

        cache.clear().await;
        assert!(cache.is_empty().await);
    }

    #[tokio::test]
    async fn test_segment_cache_push() {
        let cache = SegmentCache::new(1024 * 1024, 10);
        cache
            .put_pushed("pushed_seg", Bytes::from(vec![0u8; 64]))
            .await;
        assert_eq!(cache.pushed_count().await, 1);
        let data = cache.get("pushed_seg").await;
        assert!(data.is_some());
    }

    #[tokio::test]
    async fn test_segment_cache_ttl_expired() {
        // Zero-TTL cache: everything expires immediately.
        let cache = SegmentCache::new(1024 * 1024, 10).with_ttl(Duration::ZERO);
        cache.put("seg", Bytes::from(vec![1, 2, 3])).await;
        // Give the TTL a moment to elapse.
        tokio::time::sleep(Duration::from_millis(1)).await;
        let result = cache.get("seg").await;
        assert!(result.is_none());
    }

    // ── FetchConfig ───────────────────────────────────────────────────────────

    #[test]
    fn test_fetch_config_default() {
        let config = FetchConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.timeout, Duration::from_secs(30));
    }

    // ── PartialSegment ────────────────────────────────────────────────────────

    #[test]
    fn test_partial_segment_cache_key() {
        let range = ByteRange::new(0, 511);
        let part = PartialSegment::new("seg0.ts", range, 2, 100, 0);
        assert_eq!(part.cache_key(), "seg0.ts#0-511");
    }

    #[test]
    fn test_partial_segment_as_last() {
        let range = ByteRange::new(512, 1023);
        let part = PartialSegment::new("seg0.ts", range, 2, 100, 1).as_last();
        assert!(part.is_last);
    }

    // ── PrefetchQueue ─────────────────────────────────────────────────────────

    #[test]
    fn test_prefetch_queue_enqueue_and_pop() {
        let mut q = PrefetchQueue::new(4);

        q.enqueue("seg1.ts", None);
        q.enqueue("seg2.ts", None);
        assert_eq!(q.len(), 2);

        // Simulate fetch completion for seg1.
        if let Some(entry) = q.next_to_fetch() {
            q.complete(&entry.uri, Ok(Bytes::from("data1")));
        }
        assert_eq!(q.ready_count(), 1);

        let popped = q.pop_ready();
        assert!(popped.is_some());
        let (uri, data) = popped.expect("should succeed in test");
        assert_eq!(uri, "seg1.ts");
        assert_eq!(data, Bytes::from("data1"));
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_prefetch_queue_capacity_eviction() {
        let mut q = PrefetchQueue::new(2);
        q.enqueue("a.ts", None);
        q.enqueue("b.ts", None);
        q.enqueue("c.ts", None); // should evict "a.ts"
        assert_eq!(q.len(), 2);
        // The first entry should now be "b.ts"
        let first = q.entries.front().expect("should succeed in test");
        assert_eq!(first.uri, "b.ts");
    }

    #[test]
    fn test_prefetch_queue_failed_entry() {
        let mut q = PrefetchQueue::new(4);
        q.enqueue("bad.ts", None);
        if let Some(entry) = q.next_to_fetch() {
            q.complete(&entry.uri, Err("404 Not Found".to_string()));
        }
        // Failed entry should not be returned by pop_ready.
        assert!(q.pop_ready().is_none());
        assert_eq!(q.ready_count(), 0);
    }

    #[test]
    fn test_prefetch_queue_clear() {
        let mut q = PrefetchQueue::new(8);
        q.enqueue("x.ts", None);
        q.enqueue("y.ts", None);
        q.clear();
        assert!(q.is_empty());
    }

    // ── Http2PushHint ─────────────────────────────────────────────────────────

    #[test]
    fn test_push_hint_parse_simple() {
        let hints = Http2PushHint::parse_link_header("<seg0.ts>; rel=preload");
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].uri, "seg0.ts");
        assert!(hints[0].resource_type.is_none());
        assert!(!hints[0].crossorigin);
    }

    #[test]
    fn test_push_hint_parse_with_type() {
        let hints = Http2PushHint::parse_link_header("<seg0.ts>; rel=preload; as=fetch");
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].resource_type.as_deref(), Some("fetch"));
    }

    #[test]
    fn test_push_hint_parse_multiple() {
        let hints = Http2PushHint::parse_link_header(
            "<seg0.ts>; rel=preload, <seg1.ts>; rel=preload; crossorigin",
        );
        assert_eq!(hints.len(), 2);
        assert!(!hints[0].crossorigin);
        assert!(hints[1].crossorigin);
    }

    #[test]
    fn test_push_hint_parse_ignores_non_preload() {
        let hints = Http2PushHint::parse_link_header("<style.css>; rel=stylesheet");
        assert!(hints.is_empty());
    }

    #[test]
    fn test_push_hint_to_link_header() {
        let hint = Http2PushHint {
            uri: "seg.ts".to_string(),
            resource_type: Some("fetch".to_string()),
            crossorigin: false,
        };
        let header = hint.to_link_header();
        assert!(header.contains("rel=preload"));
        assert!(header.contains("as=fetch"));
        assert!(header.contains("<seg.ts>"));
    }

    #[test]
    fn test_push_hint_round_trip() {
        let original = Http2PushHint {
            uri: "video/seg0.ts".to_string(),
            resource_type: Some("fetch".to_string()),
            crossorigin: true,
        };
        let header = original.to_link_header();
        let parsed = Http2PushHint::parse_link_header(&header);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].uri, original.uri);
        assert_eq!(parsed[0].resource_type, original.resource_type);
        assert_eq!(parsed[0].crossorigin, original.crossorigin);
    }
}
