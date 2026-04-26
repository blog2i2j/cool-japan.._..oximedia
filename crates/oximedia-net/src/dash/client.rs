//! DASH client for segment fetching and streaming.
//!
//! This module provides a complete DASH client implementation with support for:
//! - Segment template expansion and URL generation
//! - HTTP segment downloading with retry logic
//! - Period and adaptation set management
//! - Initialization segment handling
//! - Segment timeline navigation
//! - Adaptive bitrate integration

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

#[cfg(test)]
use super::mpd::SegmentTemplate;
use super::mpd::{AdaptationSet, Mpd, Period, Representation};
use super::segment::{DashSegment, SegmentGenerator, SegmentInfo};
use crate::abr::{AbrDecision, AdaptiveBitrateController, QualityLevel};
use crate::error::{NetError, NetResult};
use bytes::Bytes;
use parking_lot::Mutex;
use reqwest::Client;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Configuration for DASH client operations.
#[derive(Debug, Clone)]
pub struct DashClientConfig {
    /// Maximum number of retries per segment.
    pub max_retries: u32,
    /// Initial retry delay.
    pub retry_delay: Duration,
    /// Request timeout.
    pub timeout: Duration,
    /// Maximum segment size in bytes.
    pub max_segment_size: usize,
    /// Base URL for resolving relative URLs.
    pub base_url: Option<String>,
    /// Enable initialization segment caching.
    pub cache_init_segments: bool,
    /// Enable adaptive bitrate switching.
    pub enable_abr: bool,
    /// Minimum buffer duration before playback.
    pub min_buffer_duration: Duration,
    /// Target buffer duration.
    pub target_buffer_duration: Duration,
    /// Maximum buffer duration.
    pub max_buffer_duration: Duration,
}

impl Default for DashClientConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay: Duration::from_millis(500),
            timeout: Duration::from_secs(30),
            max_segment_size: 50 * 1024 * 1024, // 50MB
            base_url: None,
            cache_init_segments: true,
            enable_abr: true,
            min_buffer_duration: Duration::from_secs(2),
            target_buffer_duration: Duration::from_secs(10),
            max_buffer_duration: Duration::from_secs(30),
        }
    }
}

impl DashClientConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the base URL for resolving relative paths.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Sets the maximum number of retry attempts.
    #[must_use]
    pub const fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Sets the request timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enables or disables initialization segment caching.
    #[must_use]
    pub const fn with_init_cache(mut self, enable: bool) -> Self {
        self.cache_init_segments = enable;
        self
    }

    /// Enables or disables adaptive bitrate switching.
    #[must_use]
    pub const fn with_abr(mut self, enable: bool) -> Self {
        self.enable_abr = enable;
        self
    }

    /// Sets the target buffer duration.
    #[must_use]
    pub const fn with_target_buffer(mut self, duration: Duration) -> Self {
        self.target_buffer_duration = duration;
        self
    }
}

/// Result of a segment fetch operation.
#[derive(Debug, Clone)]
pub struct FetchResult {
    /// Downloaded segment data.
    pub data: Bytes,
    /// Segment metadata.
    pub segment_info: SegmentInfo,
    /// Time taken to download.
    pub download_time: Duration,
    /// Bytes downloaded.
    pub bytes_downloaded: usize,
    /// Download throughput in bytes per second.
    pub throughput: f64,
    /// Whether this is an initialization segment.
    pub is_init: bool,
}

impl FetchResult {
    /// Creates a new fetch result.
    #[must_use]
    pub fn new(
        data: Bytes,
        segment_info: SegmentInfo,
        download_time: Duration,
        is_init: bool,
    ) -> Self {
        let bytes_downloaded = data.len();
        let throughput = if download_time.as_secs_f64() > 0.0 {
            bytes_downloaded as f64 / download_time.as_secs_f64()
        } else {
            0.0
        };

        Self {
            data,
            segment_info,
            download_time,
            bytes_downloaded,
            throughput,
            is_init,
        }
    }

    /// Returns the download throughput in bits per second.
    #[must_use]
    pub fn throughput_bps(&self) -> f64 {
        self.throughput * 8.0
    }
}

/// Representation selection for a specific adaptation set.
#[derive(Debug)]
pub struct RepresentationSelection {
    /// Index of the selected representation.
    pub representation_index: usize,
    /// Reference to the representation.
    pub representation: Representation,
    /// Segment generator for this representation.
    pub generator: SegmentGenerator,
}

impl RepresentationSelection {
    /// Creates a new representation selection.
    pub fn new(
        representation_index: usize,
        representation: Representation,
        base_url: Option<&str>,
    ) -> NetResult<Self> {
        let mut generator = SegmentGenerator::from_representation(&representation)
            .ok_or_else(|| NetError::invalid_state("No segment template in representation"))?;

        if let Some(url) = base_url {
            generator = generator.with_base_url(url);
        }

        Ok(Self {
            representation_index,
            representation,
            generator,
        })
    }

    /// Returns the bandwidth of this representation.
    #[must_use]
    pub const fn bandwidth(&self) -> u64 {
        self.representation.bandwidth
    }

    /// Returns the representation ID.
    #[must_use]
    pub fn representation_id(&self) -> &str {
        &self.representation.id
    }
}

/// Active streaming session for a period and adaptation set.
#[derive(Debug)]
pub struct StreamSession {
    /// Period index.
    pub period_index: usize,
    /// Adaptation set index within the period.
    pub adaptation_set_index: usize,
    /// Current representation selection.
    pub representation: RepresentationSelection,
    /// Current segment number.
    pub current_segment_number: u64,
    /// Current playback time in seconds.
    pub current_time: f64,
    /// Total segments downloaded.
    pub segments_downloaded: u64,
    /// Total bytes downloaded.
    pub bytes_downloaded: u64,
    /// Buffered duration.
    pub buffer_duration: Duration,
}

impl StreamSession {
    /// Creates a new stream session.
    pub fn new(
        period_index: usize,
        adaptation_set_index: usize,
        representation: RepresentationSelection,
    ) -> Self {
        let start_number = if let Some(seg) = representation.generator.segment_by_number(1) {
            seg.info.number
        } else {
            1
        };

        Self {
            period_index,
            adaptation_set_index,
            representation,
            current_segment_number: start_number,
            current_time: 0.0,
            segments_downloaded: 0,
            bytes_downloaded: 0,
            buffer_duration: Duration::ZERO,
        }
    }

    /// Returns the next segment to fetch.
    #[must_use]
    pub fn next_segment(&self) -> Option<DashSegment> {
        self.representation
            .generator
            .segment_by_number(self.current_segment_number)
    }

    /// Returns a segment at a specific time position.
    #[must_use]
    pub fn segment_at_time(&self, time_secs: f64) -> Option<DashSegment> {
        self.representation.generator.segment_at_time(time_secs)
    }

    /// Returns segments in a time range.
    pub fn segments_in_range(&self, start_secs: f64, end_secs: f64) -> Vec<DashSegment> {
        self.representation
            .generator
            .segments_for_range(start_secs, end_secs)
    }

    /// Advances to the next segment.
    pub fn advance(&mut self) {
        self.current_segment_number += 1;
    }

    /// Reports a completed download.
    pub fn report_download(&mut self, bytes: usize, segment_duration: Duration) {
        self.segments_downloaded += 1;
        self.bytes_downloaded += bytes as u64;
        self.buffer_duration += segment_duration;
    }

    /// Updates buffer duration (e.g., after playback consumption).
    pub fn update_buffer(&mut self, consumed: Duration) {
        if self.buffer_duration >= consumed {
            self.buffer_duration -= consumed;
        } else {
            self.buffer_duration = Duration::ZERO;
        }
    }

    /// Switches to a different representation.
    pub fn switch_representation(
        &mut self,
        new_representation: RepresentationSelection,
    ) -> NetResult<()> {
        // Preserve current position when switching
        let current_time = self.current_time;
        self.representation = new_representation;

        // Try to find the segment at the current time in the new representation
        if let Some(segment) = self.representation.generator.segment_at_time(current_time) {
            self.current_segment_number = segment.info.number;
        }

        Ok(())
    }
}

/// Initialization segment cache entry.
#[derive(Debug, Clone)]
struct InitSegmentCache {
    /// Cached data.
    data: Bytes,
    /// Representation ID.
    representation_id: String,
    /// When it was cached.
    cached_at: Instant,
}

/// Main DASH streaming client.
pub struct DashClient {
    /// Client configuration.
    config: DashClientConfig,
    /// Media Presentation Description.
    mpd: Mpd,
    /// Initialization segment cache.
    init_cache: Arc<RwLock<HashMap<String, InitSegmentCache>>>,
    /// Adaptive bitrate controller.
    abr_controller: Option<Box<dyn AdaptiveBitrateController>>,
    /// Total bytes downloaded across all sessions.
    total_bytes_downloaded: u64,
    /// Total download time.
    total_download_time: Duration,
    /// Lazily-initialised HTTP client. Using `Mutex<Option<Client>>` so that
    /// `fetch_http_segment` can initialise the client on first use without
    /// requiring `&mut self`.
    http_client: Mutex<Option<Client>>,
}

impl DashClient {
    /// Creates a new DASH client from an MPD.
    #[must_use]
    pub fn new(mpd: Mpd, config: DashClientConfig) -> Self {
        Self {
            config,
            mpd,
            init_cache: Arc::new(RwLock::new(HashMap::new())),
            abr_controller: None,
            total_bytes_downloaded: 0,
            total_download_time: Duration::ZERO,
            http_client: Mutex::new(None),
        }
    }

    /// Creates a client with default configuration.
    #[must_use]
    pub fn with_defaults(mpd: Mpd) -> Self {
        Self::new(mpd, DashClientConfig::default())
    }

    /// Sets the ABR controller.
    pub fn set_abr_controller(&mut self, controller: Box<dyn AdaptiveBitrateController>) {
        self.abr_controller = Some(controller);
    }

    /// Returns a reference to the MPD.
    #[must_use]
    pub const fn mpd(&self) -> &Mpd {
        &self.mpd
    }

    /// Returns the client configuration.
    #[must_use]
    pub const fn config(&self) -> &DashClientConfig {
        &self.config
    }

    /// Returns the number of periods in the presentation.
    #[must_use]
    pub fn period_count(&self) -> usize {
        self.mpd.periods.len()
    }

    /// Returns a reference to a specific period.
    #[must_use]
    pub fn period(&self, index: usize) -> Option<&Period> {
        self.mpd.periods.get(index)
    }

    /// Returns all periods.
    #[must_use]
    pub fn periods(&self) -> &[Period] {
        &self.mpd.periods
    }

    /// Finds video adaptation sets in a period.
    pub fn video_adaptation_sets(
        &self,
        period_index: usize,
    ) -> NetResult<Vec<(usize, &AdaptationSet)>> {
        let period = self
            .period(period_index)
            .ok_or_else(|| NetError::not_found(format!("Period {period_index} not found")))?;

        Ok(period
            .adaptation_sets
            .iter()
            .enumerate()
            .filter(|(_, as_)| as_.is_video())
            .collect())
    }

    /// Finds audio adaptation sets in a period.
    pub fn audio_adaptation_sets(
        &self,
        period_index: usize,
    ) -> NetResult<Vec<(usize, &AdaptationSet)>> {
        let period = self
            .period(period_index)
            .ok_or_else(|| NetError::not_found(format!("Period {period_index} not found")))?;

        Ok(period
            .adaptation_sets
            .iter()
            .enumerate()
            .filter(|(_, as_)| as_.is_audio())
            .collect())
    }

    /// Creates quality levels from an adaptation set for ABR.
    pub fn quality_levels_from_adaptation_set(
        &self,
        adaptation_set: &AdaptationSet,
    ) -> Vec<QualityLevel> {
        crate::abr::dash::representations_to_quality_levels(&adaptation_set.representations)
    }

    /// Selects the best representation from an adaptation set.
    pub fn select_representation(
        &self,
        period_index: usize,
        adaptation_set_index: usize,
        preferred_bandwidth: Option<u64>,
    ) -> NetResult<RepresentationSelection> {
        let period = self
            .period(period_index)
            .ok_or_else(|| NetError::not_found(format!("Period {period_index} not found")))?;

        let adaptation_set = period
            .adaptation_sets
            .get(adaptation_set_index)
            .ok_or_else(|| {
                NetError::not_found(format!("Adaptation set {adaptation_set_index} not found"))
            })?;

        // Get base URL (from period, adaptation set, or MPD)
        let base_url = self.resolve_base_url(period, adaptation_set);

        // Select representation based on bandwidth preference
        let rep_index = if let Some(target_bw) = preferred_bandwidth {
            self.find_best_representation_for_bandwidth(adaptation_set, target_bw)
        } else {
            // Default to lowest bitrate
            0
        };

        let representation = adaptation_set
            .representations
            .get(rep_index)
            .ok_or_else(|| NetError::not_found(format!("Representation {rep_index} not found")))?
            .clone();

        RepresentationSelection::new(rep_index, representation, base_url.as_deref())
    }

    /// Finds the best representation for a given bandwidth.
    fn find_best_representation_for_bandwidth(
        &self,
        adaptation_set: &AdaptationSet,
        bandwidth: u64,
    ) -> usize {
        let mut best_idx = 0;
        let mut best_bandwidth = 0u64;

        for (idx, rep) in adaptation_set.representations.iter().enumerate() {
            if rep.bandwidth <= bandwidth && rep.bandwidth > best_bandwidth {
                best_idx = idx;
                best_bandwidth = rep.bandwidth;
            }
        }

        best_idx
    }

    /// Resolves the base URL for a period and adaptation set.
    fn resolve_base_url(&self, period: &Period, adaptation_set: &AdaptationSet) -> Option<String> {
        // Priority: config > MPD > period > adaptation set
        if let Some(ref base) = self.config.base_url {
            return Some(base.clone());
        }

        if let Some(base) = self.mpd.base_urls.first() {
            return Some(base.clone());
        }

        if let Some(base) = period.base_urls.first() {
            return Some(base.clone());
        }

        adaptation_set
            .representations
            .first()
            .and_then(|rep| rep.base_urls.first())
            .cloned()
    }

    /// Creates a new streaming session for a period and adaptation set.
    pub fn create_session(
        &self,
        period_index: usize,
        adaptation_set_index: usize,
        initial_bandwidth: Option<u64>,
    ) -> NetResult<StreamSession> {
        let representation =
            self.select_representation(period_index, adaptation_set_index, initial_bandwidth)?;
        Ok(StreamSession::new(
            period_index,
            adaptation_set_index,
            representation,
        ))
    }

    /// Fetches the initialization segment for a representation.
    pub async fn fetch_initialization_segment(
        &mut self,
        session: &StreamSession,
    ) -> NetResult<FetchResult> {
        let init_segment = session
            .representation
            .generator
            .initialization_segment()
            .ok_or_else(|| NetError::segment("No initialization segment available"))?;

        let cache_key = format!(
            "{}_{}",
            session.period_index,
            session.representation.representation_id()
        );

        // Check cache first
        if self.config.cache_init_segments {
            let cache = self.init_cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(FetchResult::new(
                    cached.data.clone(),
                    init_segment.info.clone(),
                    Duration::ZERO,
                    true,
                ));
            }
        }

        // Fetch the initialization segment
        let start = Instant::now();
        let data = self
            .fetch_segment_data(&init_segment.url, init_segment.byte_range)
            .await?;
        let download_time = start.elapsed();

        // Cache if enabled
        if self.config.cache_init_segments {
            let mut cache = self.init_cache.write().await;
            cache.insert(
                cache_key,
                InitSegmentCache {
                    data: data.clone(),
                    representation_id: session.representation.representation_id().to_string(),
                    cached_at: Instant::now(),
                },
            );
        }

        Ok(FetchResult::new(
            data,
            init_segment.info,
            download_time,
            true,
        ))
    }

    /// Fetches the next media segment for a session.
    pub async fn fetch_next_segment(
        &mut self,
        session: &mut StreamSession,
    ) -> NetResult<FetchResult> {
        let segment = session
            .next_segment()
            .ok_or_else(|| NetError::segment("No more segments available"))?;

        let start = Instant::now();
        let data = self
            .fetch_segment_data(&segment.url, segment.byte_range)
            .await?;
        let download_time = start.elapsed();

        // Update statistics
        self.total_bytes_downloaded += data.len() as u64;
        self.total_download_time += download_time;

        // Report to ABR controller
        if let Some(ref mut abr) = self.abr_controller {
            abr.report_segment_download(data.len(), download_time);
            abr.report_buffer_level(session.buffer_duration);
        }

        // Update session
        session.report_download(data.len(), segment.info.segment_duration());
        session.advance();

        Ok(FetchResult::new(data, segment.info, download_time, false))
    }

    /// Fetches a specific segment by number.
    pub async fn fetch_segment_by_number(
        &mut self,
        session: &StreamSession,
        segment_number: u64,
    ) -> NetResult<FetchResult> {
        let segment = session
            .representation
            .generator
            .segment_by_number(segment_number)
            .ok_or_else(|| NetError::segment(format!("Segment {segment_number} not available")))?;

        let start = Instant::now();
        let data = self
            .fetch_segment_data(&segment.url, segment.byte_range)
            .await?;
        let download_time = start.elapsed();

        Ok(FetchResult::new(data, segment.info, download_time, false))
    }

    /// Fetches a segment at a specific time position.
    pub async fn fetch_segment_at_time(
        &mut self,
        session: &StreamSession,
        time_secs: f64,
    ) -> NetResult<FetchResult> {
        let segment = session
            .segment_at_time(time_secs)
            .ok_or_else(|| NetError::segment(format!("No segment at time {time_secs}s")))?;

        let start = Instant::now();
        let data = self
            .fetch_segment_data(&segment.url, segment.byte_range)
            .await?;
        let download_time = start.elapsed();

        Ok(FetchResult::new(data, segment.info, download_time, false))
    }

    /// Performs adaptive bitrate decision and switches if needed.
    pub async fn perform_abr_decision(
        &mut self,
        session: &mut StreamSession,
    ) -> NetResult<Option<AbrDecision>> {
        if !self.config.enable_abr {
            return Ok(None);
        }

        let Some(ref abr) = self.abr_controller else {
            return Ok(None);
        };

        // Get the adaptation set
        let period = self
            .period(session.period_index)
            .ok_or_else(|| NetError::invalid_state("Period not found"))?;

        let adaptation_set = period
            .adaptation_sets
            .get(session.adaptation_set_index)
            .ok_or_else(|| NetError::invalid_state("Adaptation set not found"))?;

        // Create quality levels
        let levels = self.quality_levels_from_adaptation_set(adaptation_set);

        // Get decision
        let decision = abr.select_quality(&levels, session.representation.representation_index);

        // Apply decision
        if let Some(target_idx) = decision.switch_target() {
            if target_idx != session.representation.representation_index {
                let new_rep = self.select_representation(
                    session.period_index,
                    session.adaptation_set_index,
                    Some(levels[target_idx].bandwidth),
                )?;

                session.switch_representation(new_rep)?;
            }
        }

        Ok(Some(decision))
    }

    /// Fetches segment data with retry logic.
    async fn fetch_segment_data(
        &self,
        url: &str,
        byte_range: Option<(u64, u64)>,
    ) -> NetResult<Bytes> {
        let mut last_error = None;
        let mut delay = self.config.retry_delay;

        for attempt in 0..=self.config.max_retries {
            match self.fetch_http_segment(url, byte_range).await {
                Ok(data) => return Ok(data),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.config.max_retries {
                        tokio::time::sleep(delay).await;
                        delay *= 2; // Exponential backoff
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| NetError::segment("Unknown fetch error")))
    }

    /// Returns a clone of the lazily-initialised HTTP client.
    ///
    /// The client is built on the first call and reused for all subsequent
    /// requests via `Clone` (reqwest's `Client` is cheaply cloneable — it
    /// shares the underlying connection pool through an `Arc`).
    fn get_or_init_http_client(&self) -> NetResult<Client> {
        let mut guard = self.http_client.lock();
        if guard.is_none() {
            let client = Client::builder()
                .timeout(self.config.timeout)
                .build()
                .map_err(|e| NetError::connection(format!("Failed to build HTTP client: {e}")))?;
            *guard = Some(client);
        }
        guard
            .clone()
            .ok_or_else(|| NetError::connection("HTTP client failed to initialise"))
    }

    /// Fetches a segment via HTTP.
    ///
    /// Sends a GET request for `url`, optionally adding an HTTP `Range` header
    /// when `byte_range` is supplied.  Non-2xx responses are mapped to
    /// [`NetError::Http`]; transport-level failures to the appropriate
    /// [`NetError`] variant.  The overall request is subject to
    /// `self.config.timeout`.
    async fn fetch_http_segment(
        &self,
        url: &str,
        byte_range: Option<(u64, u64)>,
    ) -> NetResult<Bytes> {
        let client = self.get_or_init_http_client()?;
        let timeout = self.config.timeout;
        let max_segment_size = self.config.max_segment_size;
        let range_header = byte_range.map(|(start, end)| format!("bytes={start}-{end}"));
        let url_owned = url.to_owned();

        tokio::time::timeout(timeout, async move {
            let mut request = client.get(&url_owned);
            if let Some(range) = range_header {
                request = request.header("Range", range);
            }

            let response = request.send().await.map_err(|e| {
                if e.is_timeout() {
                    NetError::timeout(format!("Request timed out: {url_owned}"))
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
                    format!("HTTP {status} fetching segment: {url_owned}"),
                ));
            }

            // Guard against pathologically large responses.
            if let Some(content_length) = response.content_length() {
                if content_length > max_segment_size as u64 {
                    return Err(NetError::segment(format!(
                        "Segment too large: {content_length} bytes exceeds limit of {max_segment_size} bytes",
                    )));
                }
            }

            response
                .bytes()
                .await
                .map_err(|e| NetError::connection(format!("Failed to read response body: {e}")))
        })
        .await
        .map_err(|_| NetError::timeout("Segment fetch timed out"))?
    }

    /// Returns average download throughput in bytes per second.
    #[must_use]
    pub fn average_throughput(&self) -> f64 {
        if self.total_download_time.as_secs_f64() > 0.0 {
            self.total_bytes_downloaded as f64 / self.total_download_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Returns average download throughput in bits per second.
    #[must_use]
    pub fn average_throughput_bps(&self) -> f64 {
        self.average_throughput() * 8.0
    }

    /// Returns total bytes downloaded.
    #[must_use]
    pub const fn total_bytes_downloaded(&self) -> u64 {
        self.total_bytes_downloaded
    }

    /// Clears the initialization segment cache.
    pub async fn clear_init_cache(&self) {
        let mut cache = self.init_cache.write().await;
        cache.clear();
    }

    /// Returns the number of cached initialization segments.
    pub async fn init_cache_size(&self) -> usize {
        let cache = self.init_cache.read().await;
        cache.len()
    }

    /// Estimates the presentation duration if available.
    #[must_use]
    pub fn presentation_duration(&self) -> Option<Duration> {
        self.mpd.duration()
    }

    /// Returns whether this is a live stream.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        self.mpd.is_live()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dash::mpd::MpdType;

    fn create_test_mpd() -> Mpd {
        let mut mpd = Mpd::new();
        mpd.mpd_type = MpdType::Static;
        mpd.media_presentation_duration = Some(Duration::from_secs(600));

        let mut period = Period::new();

        // Video adaptation set
        let mut video_as = AdaptationSet::new();
        video_as.content_type = Some("video".to_string());
        video_as.mime_type = Some("video/mp4".to_string());

        // Add representations with different bitrates
        let mut low_rep = Representation::new("low", 500_000);
        low_rep.width = Some(640);
        low_rep.height = Some(360);
        low_rep.segment_template = Some(
            SegmentTemplate::new(90000)
                .with_media("video_$RepresentationID$_$Number$.m4s")
                .with_initialization("video_$RepresentationID$_init.mp4"),
        );
        low_rep
            .segment_template
            .as_mut()
            .expect("should succeed in test")
            .duration = Some(180000); // 2 seconds

        let mut high_rep = Representation::new("high", 2_000_000);
        high_rep.width = Some(1920);
        high_rep.height = Some(1080);
        high_rep.segment_template = low_rep.segment_template.clone();

        video_as.representations.push(low_rep);
        video_as.representations.push(high_rep);

        period.adaptation_sets.push(video_as);
        mpd.periods.push(period);

        mpd
    }

    #[test]
    fn test_client_creation() {
        let mpd = create_test_mpd();
        let client = DashClient::with_defaults(mpd);

        assert_eq!(client.period_count(), 1);
        assert!(!client.is_live());
        assert_eq!(
            client.presentation_duration(),
            Some(Duration::from_secs(600))
        );
    }

    #[test]
    fn test_find_adaptation_sets() {
        let mpd = create_test_mpd();
        let client = DashClient::with_defaults(mpd);

        let video_sets = client
            .video_adaptation_sets(0)
            .expect("should succeed in test");
        assert_eq!(video_sets.len(), 1);

        let audio_sets = client
            .audio_adaptation_sets(0)
            .expect("should succeed in test");
        assert_eq!(audio_sets.len(), 0);
    }

    #[test]
    fn test_representation_selection() {
        let mpd = create_test_mpd();
        let client = DashClient::with_defaults(mpd);

        // Select low bandwidth
        let rep = client
            .select_representation(0, 0, Some(500_000))
            .expect("should succeed in test");
        assert_eq!(rep.representation.id, "low");

        // Select high bandwidth
        let rep = client
            .select_representation(0, 0, Some(2_000_000))
            .expect("should succeed in test");
        assert_eq!(rep.representation.id, "high");

        // Bandwidth too low - should get lowest
        let rep = client
            .select_representation(0, 0, Some(100_000))
            .expect("should succeed in test");
        assert_eq!(rep.representation.id, "low");
    }

    #[test]
    fn test_session_creation() {
        let mpd = create_test_mpd();
        let client = DashClient::with_defaults(mpd);

        let session = client
            .create_session(0, 0, Some(500_000))
            .expect("should succeed in test");
        assert_eq!(session.period_index, 0);
        assert_eq!(session.adaptation_set_index, 0);
        assert_eq!(session.current_segment_number, 1);
        assert_eq!(session.segments_downloaded, 0);
    }

    #[test]
    fn test_session_segment_navigation() {
        let mpd = create_test_mpd();
        let client = DashClient::with_defaults(mpd);
        let mut session = client
            .create_session(0, 0, None)
            .expect("should succeed in test");

        // Get next segment
        let seg1 = session.next_segment().expect("should succeed in test");
        assert_eq!(seg1.info.number, 1);

        session.advance();
        let seg2 = session.next_segment().expect("should succeed in test");
        assert_eq!(seg2.info.number, 2);
    }

    #[test]
    fn test_quality_levels_from_adaptation_set() {
        let mpd = create_test_mpd();
        let client = DashClient::with_defaults(mpd);

        let period = client.period(0).expect("should succeed in test");
        let adaptation_set = &period.adaptation_sets[0];

        let levels = client.quality_levels_from_adaptation_set(adaptation_set);
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].bandwidth, 500_000);
        assert_eq!(levels[1].bandwidth, 2_000_000);
    }

    #[test]
    fn test_config_builder() {
        let config = DashClientConfig::new()
            .with_base_url("https://example.com/")
            .with_max_retries(5)
            .with_timeout(Duration::from_secs(60))
            .with_init_cache(false)
            .with_abr(true);

        assert_eq!(config.base_url, Some("https://example.com/".to_string()));
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(!config.cache_init_segments);
        assert!(config.enable_abr);
    }

    #[test]
    fn test_session_buffer_management() {
        let mpd = create_test_mpd();
        let client = DashClient::with_defaults(mpd);
        let mut session = client
            .create_session(0, 0, None)
            .expect("should succeed in test");

        session.report_download(1000, Duration::from_secs(2));
        assert_eq!(session.buffer_duration, Duration::from_secs(2));
        assert_eq!(session.bytes_downloaded, 1000);

        session.update_buffer(Duration::from_secs(1));
        assert_eq!(session.buffer_duration, Duration::from_secs(1));
    }

    #[test]
    fn test_fetch_result() {
        let data = Bytes::from(vec![0u8; 1000]);
        let info = SegmentInfo::new(1, 0, 90000, 90000);
        let result = FetchResult::new(data, info, Duration::from_secs(1), false);

        assert_eq!(result.bytes_downloaded, 1000);
        assert!((result.throughput - 1000.0).abs() < 0.1);
        assert!((result.throughput_bps() - 8000.0).abs() < 0.1);
        assert!(!result.is_init);
    }
}
