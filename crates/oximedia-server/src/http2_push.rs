//! HTTP/2 server push hints for related media resources.
//!
//! Server push allows a server to proactively send resources to a client
//! before they are explicitly requested. This module models the *hint layer*:
//! it decides **which** resources to push alongside a primary response (e.g.
//! push thumbnail + subtitle when serving media metadata) and serialises the
//! decision as `Link: <url>; rel=preload` headers so it works with both HTTP/2
//! true push and HTTP/1.1 preload-resource-hint intermediaries.
//!
//! # Design
//!
//! The module intentionally avoids tying itself to a specific HTTP framework.
//! It produces a list of [`PushDirective`](crate::http2_push::PushDirective) values that callers convert into
//! response headers via [`PushDirective::as_link_header`](crate::http2_push::PushDirective::as_link_header).
//!
//! # Example
//!
//! ```rust
//! use oximedia_server::http2_push::{PushPlanner, PushContext, ResourceType};
//!
//! let planner = PushPlanner::default();
//! let ctx = PushContext::media_metadata("abc123");
//! let directives = planner.plan(&ctx);
//! assert!(!directives.is_empty());
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ── Resource types ────────────────────────────────────────────────────────────

/// The type of resource to be proactively pushed to the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// Precomputed thumbnail image (JPEG/WebP).
    Thumbnail,
    /// Sprite filmstrip for video scrubbing.
    ThumbnailSprite,
    /// VTT subtitle track.
    Subtitle,
    /// HLS master playlist.
    HlsMaster,
    /// DASH manifest.
    DashManifest,
    /// Waveform image for audio assets.
    AudioWaveform,
    /// Collection metadata when pushing an item within a collection.
    CollectionMeta,
}

impl ResourceType {
    /// Returns the `as` attribute value used in `Link` headers (preload hint).
    pub fn as_attr(self) -> &'static str {
        match self {
            Self::Thumbnail | Self::ThumbnailSprite | Self::AudioWaveform => "image",
            Self::Subtitle => "track",
            Self::HlsMaster | Self::DashManifest => "fetch",
            Self::CollectionMeta => "fetch",
        }
    }

    /// Returns the MIME type hint for the `type` attribute in `Link` headers.
    pub fn mime_hint(self) -> &'static str {
        match self {
            Self::Thumbnail => "image/jpeg",
            Self::ThumbnailSprite => "image/jpeg",
            Self::AudioWaveform => "image/png",
            Self::Subtitle => "text/vtt",
            Self::HlsMaster => "application/vnd.apple.mpegurl",
            Self::DashManifest => "application/dash+xml",
            Self::CollectionMeta => "application/json",
        }
    }
}

// ── Push directive ────────────────────────────────────────────────────────────

/// A single server push or preload hint for a related resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushDirective {
    /// Absolute or root-relative URL of the resource to push.
    pub url: String,
    /// Type of the resource.
    pub resource_type: ResourceType,
    /// Priority hint: `true` = high-priority push, `false` = low-priority hint.
    pub high_priority: bool,
    /// Whether the resource is a cross-origin fetch (affects CORS annotation).
    pub cross_origin: bool,
}

impl PushDirective {
    /// Creates a new high-priority push directive.
    pub fn new(url: impl Into<String>, resource_type: ResourceType) -> Self {
        Self {
            url: url.into(),
            resource_type,
            high_priority: true,
            cross_origin: false,
        }
    }

    /// Marks this directive as low-priority (deferred hint).
    pub fn low_priority(mut self) -> Self {
        self.high_priority = false;
        self
    }

    /// Marks this directive as a cross-origin resource.
    pub fn cross_origin(mut self) -> Self {
        self.cross_origin = true;
        self
    }

    /// Formats this directive as an RFC 8288 `Link` header value.
    ///
    /// Example output:
    /// `</api/v1/media/abc123/thumbnail>; rel=preload; as=image; type="image/jpeg"`
    pub fn as_link_header(&self) -> String {
        let mut parts = vec![
            format!("<{}>", self.url),
            "rel=preload".to_string(),
            format!("as={}", self.resource_type.as_attr()),
            format!("type=\"{}\"", self.resource_type.mime_hint()),
        ];
        if self.cross_origin {
            parts.push("crossorigin".to_string());
        }
        parts.join("; ")
    }
}

// ── Push context ──────────────────────────────────────────────────────────────

/// Describes the primary request being served and is used by [`PushPlanner`]
/// to decide which related resources should be pushed.
#[derive(Debug, Clone)]
pub struct PushContext {
    /// Kind of primary resource being served.
    pub primary: PrimaryResource,
    /// Media ID extracted from the request path, if applicable.
    pub media_id: Option<String>,
    /// Collection ID if the request is collection-scoped.
    pub collection_id: Option<String>,
    /// Client-advertised features (from `Accept` or custom headers).
    pub client_features: ClientFeatures,
}

/// The kind of resource the primary HTTP response is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimaryResource {
    /// GET /media/:id — single media item metadata.
    MediaMetadata,
    /// GET /media — list of media items.
    MediaList,
    /// GET /collections/:id — collection with items.
    CollectionDetail,
    /// GET /stream/:id/master.m3u8 — HLS master playlist.
    HlsStream,
    /// Other request types the planner does not have rules for.
    Other,
}

/// Features the client supports, inferred from request headers.
#[derive(Debug, Clone, Default)]
pub struct ClientFeatures {
    /// Client supports HLS streaming.
    pub hls: bool,
    /// Client supports DASH streaming.
    pub dash: bool,
    /// Client supports VTT subtitles.
    pub subtitles: bool,
}

impl PushContext {
    /// Builds a context for a media metadata response.
    pub fn media_metadata(media_id: impl Into<String>) -> Self {
        Self {
            primary: PrimaryResource::MediaMetadata,
            media_id: Some(media_id.into()),
            collection_id: None,
            client_features: ClientFeatures {
                hls: true,
                dash: false,
                subtitles: true,
            },
        }
    }

    /// Builds a context for a collection detail response.
    pub fn collection_detail(collection_id: impl Into<String>) -> Self {
        Self {
            primary: PrimaryResource::CollectionDetail,
            media_id: None,
            collection_id: Some(collection_id.into()),
            client_features: ClientFeatures::default(),
        }
    }
}

// ── Push planner ──────────────────────────────────────────────────────────────

/// Configuration governing which push directives are generated.
#[derive(Debug, Clone)]
pub struct PushPlannerConfig {
    /// Whether thumbnail push is enabled.
    pub push_thumbnails: bool,
    /// Whether streaming manifest push is enabled.
    pub push_manifests: bool,
    /// Whether subtitle push is enabled.
    pub push_subtitles: bool,
    /// Whether waveform push is enabled (audio assets).
    pub push_waveforms: bool,
    /// Base URL prefix for internal API paths (e.g. `/api/v1`).
    pub api_base: String,
}

impl Default for PushPlannerConfig {
    fn default() -> Self {
        Self {
            push_thumbnails: true,
            push_manifests: true,
            push_subtitles: true,
            push_waveforms: false,
            api_base: "/api/v1".to_string(),
        }
    }
}

/// Plans which resources to push alongside a primary HTTP response.
#[derive(Debug, Clone, Default)]
pub struct PushPlanner {
    config: PushPlannerConfig,
}

impl PushPlanner {
    /// Creates a planner with the given config.
    pub fn new(config: PushPlannerConfig) -> Self {
        Self { config }
    }

    /// Returns the list of [`PushDirective`]s for the given context.
    pub fn plan(&self, ctx: &PushContext) -> Vec<PushDirective> {
        match ctx.primary {
            PrimaryResource::MediaMetadata => self.plan_media_metadata(ctx),
            PrimaryResource::CollectionDetail => self.plan_collection_detail(ctx),
            PrimaryResource::HlsStream => self.plan_hls_stream(ctx),
            PrimaryResource::MediaList | PrimaryResource::Other => vec![],
        }
    }

    /// Converts a list of directives into `Link` header values.
    pub fn as_link_headers(directives: &[PushDirective]) -> Vec<String> {
        directives
            .iter()
            .map(PushDirective::as_link_header)
            .collect()
    }

    /// Builds a single combined `Link` header with multiple values separated by `, `.
    pub fn as_combined_link_header(directives: &[PushDirective]) -> Option<String> {
        if directives.is_empty() {
            return None;
        }
        Some(
            directives
                .iter()
                .map(PushDirective::as_link_header)
                .collect::<Vec<_>>()
                .join(", "),
        )
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn plan_media_metadata(&self, ctx: &PushContext) -> Vec<PushDirective> {
        let media_id = match &ctx.media_id {
            Some(id) => id.clone(),
            None => return vec![],
        };
        let base = &self.config.api_base;
        let mut out = Vec::new();

        if self.config.push_thumbnails {
            out.push(PushDirective::new(
                format!("{base}/media/{media_id}/thumbnail"),
                ResourceType::Thumbnail,
            ));
        }
        if self.config.push_manifests && ctx.client_features.hls {
            out.push(
                PushDirective::new(
                    format!("/stream/{media_id}/master.m3u8"),
                    ResourceType::HlsMaster,
                )
                .low_priority(),
            );
        }
        if self.config.push_manifests && ctx.client_features.dash {
            out.push(
                PushDirective::new(
                    format!("/stream/{media_id}/manifest.mpd"),
                    ResourceType::DashManifest,
                )
                .low_priority(),
            );
        }
        if self.config.push_subtitles && ctx.client_features.subtitles {
            out.push(
                PushDirective::new(
                    format!("{base}/media/{media_id}/subtitles.vtt"),
                    ResourceType::Subtitle,
                )
                .low_priority(),
            );
        }
        out
    }

    fn plan_collection_detail(&self, ctx: &PushContext) -> Vec<PushDirective> {
        let collection_id = match &ctx.collection_id {
            Some(id) => id.clone(),
            None => return vec![],
        };
        let base = &self.config.api_base;
        vec![PushDirective::new(
            format!("{base}/collections/{collection_id}/thumbnail"),
            ResourceType::CollectionMeta,
        )
        .low_priority()]
    }

    fn plan_hls_stream(&self, ctx: &PushContext) -> Vec<PushDirective> {
        let media_id = match &ctx.media_id {
            Some(id) => id.clone(),
            None => return vec![],
        };
        if self.config.push_thumbnails {
            vec![PushDirective::new(
                format!("/api/v1/media/{media_id}/thumbnail"),
                ResourceType::ThumbnailSprite,
            )
            .low_priority()]
        } else {
            vec![]
        }
    }
}

// ── Push budget tracker ───────────────────────────────────────────────────────

/// Tracks per-connection push budgets to avoid overwhelming slow clients.
///
/// HTTP/2 clients signal flow-control capacity via `WINDOW_UPDATE` frames.
/// This lightweight tracker enforces a bytes-based budget ceiling so the
/// planner does not queue more pushes than the client can handle.
#[derive(Debug)]
pub struct PushBudget {
    /// Maximum bytes the planner may push per response cycle.
    max_bytes_per_response: u64,
    /// Estimated per-resource sizes used when no Content-Length is available.
    size_estimates: HashMap<ResourceType, u64>,
    /// Running total of bytes consumed this cycle.
    consumed: u64,
}

impl PushBudget {
    /// Creates a budget with the given per-response byte cap.
    pub fn new(max_bytes_per_response: u64) -> Self {
        let mut size_estimates = HashMap::new();
        size_estimates.insert(ResourceType::Thumbnail, 50_000);
        size_estimates.insert(ResourceType::ThumbnailSprite, 200_000);
        size_estimates.insert(ResourceType::AudioWaveform, 15_000);
        size_estimates.insert(ResourceType::Subtitle, 5_000);
        size_estimates.insert(ResourceType::HlsMaster, 2_000);
        size_estimates.insert(ResourceType::DashManifest, 4_000);
        size_estimates.insert(ResourceType::CollectionMeta, 1_000);

        Self {
            max_bytes_per_response,
            size_estimates,
            consumed: 0,
        }
    }

    /// Returns `true` if the budget allows adding the given resource type.
    pub fn can_push(&self, resource_type: ResourceType) -> bool {
        let estimate = self
            .size_estimates
            .get(&resource_type)
            .copied()
            .unwrap_or(10_000);
        self.consumed + estimate <= self.max_bytes_per_response
    }

    /// Consumes budget for the given resource type.
    ///
    /// Returns `true` if successful, `false` if the budget would be exceeded.
    pub fn consume(&mut self, resource_type: ResourceType) -> bool {
        let estimate = self
            .size_estimates
            .get(&resource_type)
            .copied()
            .unwrap_or(10_000);
        if self.consumed + estimate > self.max_bytes_per_response {
            return false;
        }
        self.consumed += estimate;
        true
    }

    /// Resets the consumed counter for a new response cycle.
    pub fn reset(&mut self) {
        self.consumed = 0;
    }

    /// Returns the remaining budget in bytes.
    pub fn remaining(&self) -> u64 {
        self.max_bytes_per_response.saturating_sub(self.consumed)
    }

    /// Filters a list of directives down to those that fit within the budget.
    pub fn filter_directives(&mut self, directives: Vec<PushDirective>) -> Vec<PushDirective> {
        directives
            .into_iter()
            .filter(|d| self.consume(d.resource_type))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_type_as_attr() {
        assert_eq!(ResourceType::Thumbnail.as_attr(), "image");
        assert_eq!(ResourceType::Subtitle.as_attr(), "track");
        assert_eq!(ResourceType::HlsMaster.as_attr(), "fetch");
    }

    #[test]
    fn test_resource_type_mime_hint() {
        assert_eq!(
            ResourceType::HlsMaster.mime_hint(),
            "application/vnd.apple.mpegurl"
        );
        assert_eq!(
            ResourceType::DashManifest.mime_hint(),
            "application/dash+xml"
        );
        assert_eq!(ResourceType::Subtitle.mime_hint(), "text/vtt");
    }

    #[test]
    fn test_push_directive_as_link_header_basic() {
        let d = PushDirective::new("/api/v1/media/abc/thumbnail", ResourceType::Thumbnail);
        let header = d.as_link_header();
        assert!(header.contains("rel=preload"));
        assert!(header.contains("as=image"));
        assert!(header.contains("/api/v1/media/abc/thumbnail"));
    }

    #[test]
    fn test_push_directive_cross_origin_flag() {
        let d = PushDirective::new("/cdn/thumb.jpg", ResourceType::Thumbnail).cross_origin();
        let header = d.as_link_header();
        assert!(header.contains("crossorigin"));
    }

    #[test]
    fn test_push_directive_no_cross_origin_by_default() {
        let d = PushDirective::new("/local/thumb.jpg", ResourceType::Thumbnail);
        assert!(!d.as_link_header().contains("crossorigin"));
    }

    #[test]
    fn test_planner_media_metadata_pushes_thumbnail() {
        let planner = PushPlanner::default();
        let ctx = PushContext::media_metadata("abc123");
        let directives = planner.plan(&ctx);
        let has_thumb = directives
            .iter()
            .any(|d| d.resource_type == ResourceType::Thumbnail);
        assert!(has_thumb);
    }

    #[test]
    fn test_planner_media_metadata_pushes_hls_when_supported() {
        let planner = PushPlanner::default();
        let mut ctx = PushContext::media_metadata("abc123");
        ctx.client_features.hls = true;
        let directives = planner.plan(&ctx);
        let has_hls = directives
            .iter()
            .any(|d| d.resource_type == ResourceType::HlsMaster);
        assert!(has_hls);
    }

    #[test]
    fn test_planner_no_hls_when_not_supported() {
        let planner = PushPlanner::default();
        let mut ctx = PushContext::media_metadata("abc123");
        ctx.client_features.hls = false;
        ctx.client_features.dash = false;
        let directives = planner.plan(&ctx);
        let has_hls = directives
            .iter()
            .any(|d| d.resource_type == ResourceType::HlsMaster);
        assert!(!has_hls);
    }

    #[test]
    fn test_planner_other_resource_returns_empty() {
        let planner = PushPlanner::default();
        let ctx = PushContext {
            primary: PrimaryResource::Other,
            media_id: None,
            collection_id: None,
            client_features: ClientFeatures::default(),
        };
        assert!(planner.plan(&ctx).is_empty());
    }

    #[test]
    fn test_combined_link_header_some() {
        let directives = vec![
            PushDirective::new("/thumb.jpg", ResourceType::Thumbnail),
            PushDirective::new("/sub.vtt", ResourceType::Subtitle),
        ];
        let combined = PushPlanner::as_combined_link_header(&directives);
        assert!(combined.is_some());
        let s = combined.unwrap();
        assert!(s.contains("rel=preload"));
    }

    #[test]
    fn test_combined_link_header_empty_returns_none() {
        let combined = PushPlanner::as_combined_link_header(&[]);
        assert!(combined.is_none());
    }

    #[test]
    fn test_push_budget_can_push() {
        let budget = PushBudget::new(1_000_000);
        assert!(budget.can_push(ResourceType::Thumbnail));
    }

    #[test]
    fn test_push_budget_consume_and_remaining() {
        let mut budget = PushBudget::new(100_000);
        let before = budget.remaining();
        let ok = budget.consume(ResourceType::HlsMaster);
        assert!(ok);
        assert!(budget.remaining() < before);
    }

    #[test]
    fn test_push_budget_exceeds_cap() {
        let mut budget = PushBudget::new(1_000); // tiny budget
                                                 // ThumbnailSprite estimate = 200_000 bytes → should fail
        assert!(!budget.consume(ResourceType::ThumbnailSprite));
    }

    #[test]
    fn test_push_budget_reset() {
        let mut budget = PushBudget::new(100_000);
        budget.consume(ResourceType::Thumbnail);
        let after_consume = budget.remaining();
        budget.reset();
        assert!(budget.remaining() > after_consume);
    }

    #[test]
    fn test_push_budget_filter_directives() {
        let mut budget = PushBudget::new(60_000);
        let directives = vec![
            PushDirective::new("/thumb.jpg", ResourceType::Thumbnail), // 50_000 bytes
            PushDirective::new("/sprite.jpg", ResourceType::ThumbnailSprite), // 200_000 bytes → too large
        ];
        let allowed = budget.filter_directives(directives);
        assert_eq!(allowed.len(), 1);
        assert_eq!(allowed[0].resource_type, ResourceType::Thumbnail);
    }

    #[test]
    fn test_planner_collection_detail() {
        let planner = PushPlanner::default();
        let ctx = PushContext::collection_detail("col-1");
        let directives = planner.plan(&ctx);
        assert!(!directives.is_empty());
    }
}
