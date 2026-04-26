//! Gzip HTTP response compression middleware using `oxiarc-deflate`.
//!
//! Implements a Tower `Layer` / `Service` pair that compresses HTTP responses
//! with gzip when the client sends `Accept-Encoding: gzip` and the response
//! carries a compressible `Content-Type`.
//!
//! # Selectivity rules
//!
//! The middleware skips compression when **any** of the following are true:
//! - The client did not include `gzip` in `Accept-Encoding`.
//! - The response already carries a `Content-Encoding` header.
//! - The HTTP status code is 101 (Upgrade/WebSocket), 204 (No Content), or 304 (Not Modified).
//! - The `Content-Type` is not in the compressible allowlist (see `is_compressible_content_type`).
//! - The buffered body exceeds `MAX_COMPRESSIBLE_BODY` bytes (safety valve for unexpectedly
//!   large API responses; media/streaming bodies are filtered out by content-type first).
//!
//! When the body-collection step fails the middleware returns the original response unmodified
//! rather than surfacing an internal error — the client still receives valid data.

use axum::{
    body::Body,
    http::{
        header::{CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, VARY},
        Request, Response, StatusCode,
    },
};
use bytes::Bytes;
use oxiarc_deflate::gzip_compress;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};
use tower::{Layer, Service};

/// Maximum body size that will be buffered and compressed.
///
/// Responses larger than this are passed through uncompressed.
/// Set to 32 MiB — large enough for any API / manifest response, small enough
/// to avoid OOM on media downloads that slip through the content-type gate.
pub const MAX_COMPRESSIBLE_BODY: usize = 32 * 1024 * 1024;

/// Gzip compression level used for HTTP responses (balanced speed vs ratio).
const GZIP_LEVEL: u8 = 6;

/// Minimum response body size worth compressing (bytes).
///
/// Compressing tiny bodies wastes CPU and adds gzip header overhead (18+ bytes).
const MIN_COMPRESSIBLE_SIZE: usize = 512;

// ── Layer ─────────────────────────────────────────────────────────────────────

/// Tower [`Layer`] that wraps responses with gzip compression.
///
/// Add this to an axum `ServiceBuilder` stack with `.layer(GzipLayer::new())`.
#[derive(Clone, Copy, Debug, Default)]
pub struct GzipLayer;

impl GzipLayer {
    /// Create a new [`GzipLayer`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for GzipLayer {
    type Service = GzipService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        GzipService { inner }
    }
}

// ── Service ───────────────────────────────────────────────────────────────────

/// Tower [`Service`] that compresses HTTP responses with gzip.
///
/// Produced by [`GzipLayer`].
#[derive(Clone)]
pub struct GzipService<S> {
    inner: S,
}

impl<S, ReqBody> Service<Request<ReqBody>> for GzipService<S>
where
    S: Service<Request<ReqBody>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    S::Error: Send + 'static,
    ReqBody: Send + 'static,
{
    type Response = Response<Body>;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        // Determine whether the client accepts gzip *before* forwarding the request.
        let accepts_gzip = client_accepts_gzip(&req);

        let fut = self.inner.call(req);

        Box::pin(async move {
            let response = fut.await?;

            if !accepts_gzip {
                // Even when we don't compress, inject Vary so caching proxies know
                // the response differs by Accept-Encoding (RFC 7231 §7.1.4).
                return Ok(inject_vary_header(response));
            }

            Ok(maybe_compress_response(response).await)
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns `true` if the `Accept-Encoding` request header contains `gzip`.
fn client_accepts_gzip<B>(req: &Request<B>) -> bool {
    req.headers()
        .get(axum::http::header::ACCEPT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            s.split(',')
                .any(|part| part.trim().eq_ignore_ascii_case("gzip"))
        })
        .unwrap_or(false)
}

/// Returns `true` if a response `Content-Type` value indicates compressible content.
///
/// Only text-like and structured-data types benefit from gzip; binary media
/// (video, audio, images) is already compressed and must not be buffered.
pub fn is_compressible_content_type(content_type: &str) -> bool {
    // Strip parameters (e.g. `application/json; charset=utf-8`).
    let base = content_type.split(';').next().unwrap_or("").trim();

    matches!(
        base,
        "application/json"
            | "application/xml"
            | "application/javascript"
            | "application/x-javascript"
            | "application/xhtml+xml"
            | "application/rss+xml"
            | "application/atom+xml"
            | "application/ld+json"
            | "application/manifest+json"
            | "application/vnd.api+json"
            | "application/dash+xml"
            | "application/x-mpegurl"
            | "application/vnd.apple.mpegurl"
            | "image/svg+xml"
            | "text/plain"
            | "text/html"
            | "text/css"
            | "text/javascript"
            | "text/xml"
            | "text/csv"
    )
    // text/* catch-all (but NOT text/event-stream — SSE must stream)
    || (base.starts_with("text/") && base != "text/event-stream")
}

/// Appends `Accept-Encoding` to the `Vary` response header so that caching
/// proxies know the representation differs by encoding (RFC 7231 §7.1.4).
///
/// If `Vary: Accept-Encoding` (case-insensitive) is already present the
/// response is returned unchanged.
fn inject_vary_header(mut response: Response<Body>) -> Response<Body> {
    let headers = response.headers_mut();
    let new_value = match headers.get(VARY).and_then(|v| v.to_str().ok()) {
        Some(existing)
            if existing
                .split(',')
                .any(|p| p.trim().eq_ignore_ascii_case("accept-encoding")) =>
        {
            // `Accept-Encoding` is already listed — leave unchanged.
            return response;
        }
        Some(existing) => format!("{existing}, Accept-Encoding"),
        None => "Accept-Encoding".to_string(),
    };
    if let Ok(hv) = axum::http::HeaderValue::from_str(&new_value) {
        headers.insert(VARY, hv);
    }
    response
}

/// Returns `true` if the response status code means there is no body to compress.
fn status_has_no_body(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::SWITCHING_PROTOCOLS | StatusCode::NO_CONTENT | StatusCode::NOT_MODIFIED
    )
}

/// Attempt to gzip-compress a response body.
///
/// On any failure (body collection error, compression error, guard failed)
/// the *original* response is returned unmodified so the client still receives
/// valid data.
async fn maybe_compress_response(response: Response<Body>) -> Response<Body> {
    // Gate 1: status codes that carry no body or must not be modified.
    if status_has_no_body(response.status()) {
        return response;
    }

    // Gate 2: already encoded.
    if response.headers().contains_key(CONTENT_ENCODING) {
        return response;
    }

    // Gate 3: content-type allowlist.
    let compressible = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(is_compressible_content_type)
        .unwrap_or(false);

    if !compressible {
        return response;
    }

    // Destructure so we can replace the body without cloning the rest.
    let (mut parts, body) = response.into_parts();

    // Collect body bytes (bounded).
    let body_bytes: Bytes = match axum::body::to_bytes(body, MAX_COMPRESSIBLE_BODY).await {
        Ok(b) => b,
        Err(_) => {
            // Body collection failed — return an empty response (body was consumed).
            return Response::from_parts(parts, Body::empty());
        }
    };

    // Gate 4: minimum size — too small to benefit.
    if body_bytes.len() < MIN_COMPRESSIBLE_SIZE {
        return Response::from_parts(parts, Body::from(body_bytes));
    }

    // Compress.
    let compressed: Vec<u8> = match gzip_compress(&body_bytes, GZIP_LEVEL) {
        Ok(c) => c,
        Err(_) => {
            // Compression failed — return original unmodified.
            return Response::from_parts(parts, Body::from(body_bytes));
        }
    };

    // Update headers.
    parts.headers.insert(
        CONTENT_ENCODING,
        axum::http::HeaderValue::from_static("gzip"),
    );
    // Remove stale Content-Length — the compressed size differs.
    parts.headers.remove(CONTENT_LENGTH);
    // Advertise Vary: Accept-Encoding so caching proxies store separate
    // representations for compressed vs. uncompressed clients.
    if !parts
        .headers
        .get(VARY)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            s.split(',')
                .any(|p| p.trim().eq_ignore_ascii_case("accept-encoding"))
        })
        .unwrap_or(false)
    {
        let new_vary = match parts.headers.get(VARY).and_then(|v| v.to_str().ok()) {
            Some(existing) => format!("{existing}, Accept-Encoding"),
            None => "Accept-Encoding".to_string(),
        };
        if let Ok(hv) = axum::http::HeaderValue::from_str(&new_vary) {
            parts.headers.insert(VARY, hv);
        }
    }

    Response::from_parts(parts, Body::from(compressed))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{header, Request, Response, StatusCode},
    };
    use tower::ServiceExt;

    // ── body fixture ─────────────────────────────────────────────────────────

    /// A body large enough to exceed the min-size threshold so compression is triggered.
    fn large_json_body() -> Vec<u8> {
        let mut s = String::from("[");
        for i in 0..200u32 {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                r#"{{"id":{i},"name":"media_item_{i}","status":"active","size":1048576}}"#
            ));
        }
        s.push(']');
        s.into_bytes()
    }

    // ── test: gzip path ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_gzip_compresses_json_response() {
        let original = large_json_body();
        let original_len = original.len();

        let svc = GzipLayer.layer(tower::service_fn(move |_req: Request<Body>| {
            let body = original.clone();
            let len = body.len().to_string();
            async move {
                let resp = Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(header::CONTENT_LENGTH, len)
                    .body(Body::from(body))
                    .expect("valid response");
                Ok::<_, std::convert::Infallible>(resp)
            }
        }));

        let req = Request::builder()
            .header(header::ACCEPT_ENCODING, "gzip, deflate")
            .body(Body::empty())
            .expect("valid request");

        let resp = svc.oneshot(req).await.expect("call ok");

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(CONTENT_ENCODING)
                .and_then(|v| v.to_str().ok()),
            Some("gzip"),
            "Content-Encoding must be gzip"
        );
        assert!(
            resp.headers().get(CONTENT_LENGTH).is_none(),
            "Content-Length must be absent after compression"
        );

        let compressed = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let decompressed = oxiarc_deflate::gzip_decompress(&compressed).expect("gzip decompress");
        assert_eq!(
            decompressed.len(),
            original_len,
            "decompressed length must match original"
        );
    }

    // ── test: no Accept-Encoding → pass-through ───────────────────────────────

    #[tokio::test]
    async fn test_no_accept_encoding_skips_compression() {
        let original = large_json_body();
        let expected_len = original.len();

        let svc = GzipLayer.layer(tower::service_fn(move |_req: Request<Body>| {
            let body = original.clone();
            async move {
                let resp = Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .expect("valid response");
                Ok::<_, std::convert::Infallible>(resp)
            }
        }));

        let req = Request::builder()
            .body(Body::empty())
            .expect("valid request");

        let resp = svc.oneshot(req).await.expect("call ok");

        assert!(
            resp.headers().get(CONTENT_ENCODING).is_none(),
            "Content-Encoding must be absent when client did not request gzip"
        );
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        assert_eq!(body_bytes.len(), expected_len, "body must be unmodified");
    }

    // ── test: non-gzip Accept-Encoding → pass-through ─────────────────────────

    #[tokio::test]
    async fn test_accept_encoding_without_gzip_skips_compression() {
        let original = large_json_body();

        let svc = GzipLayer.layer(tower::service_fn(move |_req: Request<Body>| {
            let body = original.clone();
            async move {
                let resp = Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .expect("valid response");
                Ok::<_, std::convert::Infallible>(resp)
            }
        }));

        let req = Request::builder()
            .header(header::ACCEPT_ENCODING, "br, deflate")
            .body(Body::empty())
            .expect("valid request");

        let resp = svc.oneshot(req).await.expect("call ok");

        assert!(
            resp.headers().get(CONTENT_ENCODING).is_none(),
            "Content-Encoding must be absent for non-gzip Accept-Encoding"
        );
    }

    // ── test: video/mp4 (non-compressible) → pass-through ────────────────────

    #[tokio::test]
    async fn test_binary_media_content_type_skips_compression() {
        let fake_media: Vec<u8> = vec![0u8; 64 * 1024];
        let expected_len = fake_media.len();

        let svc = GzipLayer.layer(tower::service_fn(move |_req: Request<Body>| {
            let body = fake_media.clone();
            async move {
                let resp = Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "video/mp4")
                    .body(Body::from(body))
                    .expect("valid response");
                Ok::<_, std::convert::Infallible>(resp)
            }
        }));

        let req = Request::builder()
            .header(header::ACCEPT_ENCODING, "gzip")
            .body(Body::empty())
            .expect("valid request");

        let resp = svc.oneshot(req).await.expect("call ok");

        assert!(
            resp.headers().get(CONTENT_ENCODING).is_none(),
            "video/mp4 must not be compressed"
        );
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        assert_eq!(
            body_bytes.len(),
            expected_len,
            "video body must pass through unchanged"
        );
    }

    // ── test: audio/ogg → pass-through ───────────────────────────────────────

    #[tokio::test]
    async fn test_audio_content_type_skips_compression() {
        let fake_audio: Vec<u8> = vec![0xABu8; 1024];

        let svc = GzipLayer.layer(tower::service_fn(move |_req: Request<Body>| {
            let body = fake_audio.clone();
            async move {
                let resp = Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "audio/ogg")
                    .body(Body::from(body))
                    .expect("valid response");
                Ok::<_, std::convert::Infallible>(resp)
            }
        }));

        let req = Request::builder()
            .header(header::ACCEPT_ENCODING, "gzip")
            .body(Body::empty())
            .expect("valid request");

        let resp = svc.oneshot(req).await.expect("call ok");

        assert!(
            resp.headers().get(CONTENT_ENCODING).is_none(),
            "audio/ogg must not be compressed"
        );
    }

    // ── test: 204 No Content → pass-through ──────────────────────────────────

    #[tokio::test]
    async fn test_204_status_skips_compression() {
        let svc = GzipLayer.layer(tower::service_fn(move |_req: Request<Body>| async move {
            let resp = Response::builder()
                .status(StatusCode::NO_CONTENT)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::empty())
                .expect("valid response");
            Ok::<_, std::convert::Infallible>(resp)
        }));

        let req = Request::builder()
            .header(header::ACCEPT_ENCODING, "gzip")
            .body(Body::empty())
            .expect("valid request");

        let resp = svc.oneshot(req).await.expect("call ok");

        assert!(
            resp.headers().get(CONTENT_ENCODING).is_none(),
            "204 responses must not be compressed"
        );
    }

    // ── test: already encoded → pass-through ─────────────────────────────────

    #[tokio::test]
    async fn test_already_encoded_response_skips_compression() {
        let body_bytes = large_json_body();

        let svc = GzipLayer.layer(tower::service_fn(move |_req: Request<Body>| {
            let body = body_bytes.clone();
            async move {
                let resp = Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(CONTENT_ENCODING, "br")
                    .body(Body::from(body))
                    .expect("valid response");
                Ok::<_, std::convert::Infallible>(resp)
            }
        }));

        let req = Request::builder()
            .header(header::ACCEPT_ENCODING, "gzip")
            .body(Body::empty())
            .expect("valid request");

        let resp = svc.oneshot(req).await.expect("call ok");

        assert_eq!(
            resp.headers()
                .get(CONTENT_ENCODING)
                .and_then(|v| v.to_str().ok()),
            Some("br"),
            "pre-existing Content-Encoding must not be overwritten"
        );
    }

    // ── test: small body → pass-through (below MIN_COMPRESSIBLE_SIZE) ─────────

    #[tokio::test]
    async fn test_small_body_skips_compression() {
        let small_body: Vec<u8> = b"{\"ok\":true}".to_vec();
        let expected = small_body.clone();

        let svc = GzipLayer.layer(tower::service_fn(move |_req: Request<Body>| {
            let body = small_body.clone();
            async move {
                let resp = Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .expect("valid response");
                Ok::<_, std::convert::Infallible>(resp)
            }
        }));

        let req = Request::builder()
            .header(header::ACCEPT_ENCODING, "gzip")
            .body(Body::empty())
            .expect("valid request");

        let resp = svc.oneshot(req).await.expect("call ok");

        assert!(
            resp.headers().get(CONTENT_ENCODING).is_none(),
            "small body must not be compressed"
        );
        let resp_body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        assert_eq!(
            resp_body.as_ref(),
            expected.as_slice(),
            "small body must pass through"
        );
    }

    // ── test: text/plain → compressed ────────────────────────────────────────

    #[tokio::test]
    async fn test_text_plain_is_compressed() {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(30);
        let original = text.into_bytes();
        let original_len = original.len();

        let svc = GzipLayer.layer(tower::service_fn(move |_req: Request<Body>| {
            let body = original.clone();
            async move {
                let resp = Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/plain")
                    .body(Body::from(body))
                    .expect("valid response");
                Ok::<_, std::convert::Infallible>(resp)
            }
        }));

        let req = Request::builder()
            .header(header::ACCEPT_ENCODING, "gzip")
            .body(Body::empty())
            .expect("valid request");

        let resp = svc.oneshot(req).await.expect("call ok");

        assert_eq!(
            resp.headers()
                .get(CONTENT_ENCODING)
                .and_then(|v| v.to_str().ok()),
            Some("gzip"),
            "text/plain must be compressed"
        );
        let compressed = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let decompressed = oxiarc_deflate::gzip_decompress(&compressed).expect("gzip decompress");
        assert_eq!(
            decompressed.len(),
            original_len,
            "text/plain round-trip length must match"
        );
    }

    // ── test: text/event-stream → pass-through (SSE must not be compressed) ──

    #[tokio::test]
    async fn test_text_event_stream_skips_compression() {
        // Pad to well above the MIN_COMPRESSIBLE_SIZE threshold.
        let padded: Vec<u8> = b"data: hello\n\n"
            .iter()
            .copied()
            .cycle()
            .take(2048)
            .collect();

        let svc = GzipLayer.layer(tower::service_fn(move |_req: Request<Body>| {
            let body = padded.clone();
            async move {
                let resp = Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .body(Body::from(body))
                    .expect("valid response");
                Ok::<_, std::convert::Infallible>(resp)
            }
        }));

        let req = Request::builder()
            .header(header::ACCEPT_ENCODING, "gzip")
            .body(Body::empty())
            .expect("valid request");

        let resp = svc.oneshot(req).await.expect("call ok");

        assert!(
            resp.headers().get(CONTENT_ENCODING).is_none(),
            "text/event-stream (SSE) must never be compressed"
        );
    }

    // ── test: is_compressible_content_type allowlist ──────────────────────────

    #[test]
    fn test_compressible_content_types() {
        let yes = [
            "application/json",
            "application/json; charset=utf-8",
            "application/xml",
            "text/html",
            "text/plain",
            "text/css",
            "application/javascript",
            "application/dash+xml",
            "application/vnd.apple.mpegurl",
            "image/svg+xml",
            "text/csv",
        ];
        for ct in &yes {
            assert!(
                is_compressible_content_type(ct),
                "{ct} should be compressible"
            );
        }
    }

    #[test]
    fn test_non_compressible_content_types() {
        let no = [
            "video/mp4",
            "video/webm",
            "audio/ogg",
            "audio/opus",
            "audio/flac",
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/avif",
            "application/octet-stream",
            "text/event-stream",
        ];
        for ct in &no {
            assert!(
                !is_compressible_content_type(ct),
                "{ct} should NOT be compressible"
            );
        }
    }
}
