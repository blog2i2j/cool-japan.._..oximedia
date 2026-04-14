//! S3-compatible object storage source.
//!
//! Provides a [`S3Source`] for streaming media from S3-compatible object storage
//! with byte-range request support, multipart transfers, and configurable retry logic.

use std::collections::HashMap;

/// S3 access credentials.
#[derive(Debug, Clone)]
pub struct S3Credentials {
    /// AWS access key ID.
    pub access_key_id: String,
    /// AWS secret access key.
    pub secret_access_key: String,
    /// Optional session token for temporary credentials.
    pub session_token: Option<String>,
}

impl S3Credentials {
    /// Create new credentials.
    #[must_use]
    pub fn new(access_key_id: impl Into<String>, secret_access_key: impl Into<String>) -> Self {
        Self {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token: None,
        }
    }

    /// Set session token.
    #[must_use]
    pub fn with_session_token(mut self, token: impl Into<String>) -> Self {
        self.session_token = Some(token.into());
        self
    }
}

/// Configuration for S3 source connections.
#[derive(Debug, Clone)]
pub struct S3Config {
    /// Endpoint URL (e.g. `https://s3.amazonaws.com` or compatible).
    pub endpoint: String,
    /// AWS region (e.g. `us-east-1`).
    pub region: String,
    /// Bucket name.
    pub bucket: String,
    /// S3 credentials.
    pub credentials: S3Credentials,
    /// Maximum number of retry attempts on transient failures.
    pub max_retries: usize,
    /// Connection timeout in milliseconds.
    pub connect_timeout_ms: u64,
    /// Read timeout in milliseconds.
    pub read_timeout_ms: u64,
    /// Whether to use path-style access (vs. virtual hosted-style).
    pub path_style: bool,
    /// Optional extra headers to include with every request.
    pub extra_headers: HashMap<String, String>,
}

impl Default for S3Config {
    fn default() -> Self {
        Self {
            endpoint: "https://s3.amazonaws.com".to_string(),
            region: "us-east-1".to_string(),
            bucket: String::new(),
            credentials: S3Credentials::new("", ""),
            max_retries: 3,
            connect_timeout_ms: 5_000,
            read_timeout_ms: 30_000,
            path_style: false,
            extra_headers: HashMap::new(),
        }
    }
}

/// Metadata about an S3 object.
#[derive(Debug, Clone)]
pub struct S3ObjectMeta {
    /// Object key.
    pub key: String,
    /// Object size in bytes.
    pub size: u64,
    /// ETag (hex MD5 for non-multipart, or `hash-N` for multipart).
    pub etag: String,
    /// Last-modified timestamp (RFC 2822 / HTTP date string).
    pub last_modified: String,
    /// Content type.
    pub content_type: Option<String>,
    /// User-defined metadata.
    pub user_metadata: HashMap<String, String>,
}

/// Error type for S3 operations.
#[derive(Debug)]
pub enum S3Error {
    /// Network or connection error.
    Network(String),
    /// S3 service returned an error response.
    Service { code: String, message: String },
    /// Authentication or authorisation error.
    Auth(String),
    /// Object not found.
    NotFound { bucket: String, key: String },
    /// Invalid request parameters.
    InvalidRequest(String),
    /// I/O error.
    Io(String),
}

impl std::fmt::Display for S3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Network(msg) => write!(f, "S3 network error: {msg}"),
            Self::Service { code, message } => write!(f, "S3 service error [{code}]: {message}"),
            Self::Auth(msg) => write!(f, "S3 auth error: {msg}"),
            Self::NotFound { bucket, key } => {
                write!(f, "S3 object not found: s3://{bucket}/{key}")
            }
            Self::InvalidRequest(msg) => write!(f, "S3 invalid request: {msg}"),
            Self::Io(msg) => write!(f, "S3 I/O error: {msg}"),
        }
    }
}

impl std::error::Error for S3Error {}

/// Result type for S3 operations.
pub type S3Result<T> = Result<T, S3Error>;

/// A range of bytes `[start, end)` (exclusive end).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    /// Inclusive start offset.
    pub start: u64,
    /// Exclusive end offset (use `u64::MAX` to read to EOF).
    pub end: u64,
}

impl ByteRange {
    /// Create a range from `start` to `end` (exclusive).
    #[must_use]
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    /// Create a range starting at `offset` with the given `length` in bytes.
    #[must_use]
    pub fn from_offset_length(offset: u64, length: u64) -> Self {
        Self {
            start: offset,
            end: offset.saturating_add(length),
        }
    }

    /// Returns an HTTP `Range` header value (e.g. `bytes=0-1023`).
    #[must_use]
    pub fn http_header(&self) -> String {
        if self.end == u64::MAX {
            format!("bytes={}-", self.start)
        } else {
            format!("bytes={}-{}", self.start, self.end.saturating_sub(1))
        }
    }

    /// Number of bytes in this range (returns `u64::MAX` if open-ended).
    #[must_use]
    pub fn length(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
}

/// High-level S3 source for streaming media.
///
/// `S3Source` builds canonical S3 request URLs and produces HTTP byte-range
/// headers without pulling in any HTTP runtime crate.  Callers that need
/// actual network I/O can construct the URL / headers from the helper methods
/// and drive the transport layer themselves (e.g. via `http_source.rs`).
#[derive(Debug, Clone)]
pub struct S3Source {
    config: S3Config,
    key: String,
    /// Cached object size (fetched lazily via HEAD).
    cached_size: Option<u64>,
}

impl S3Source {
    /// Create a new S3 source for the given object key.
    #[must_use]
    pub fn new(config: S3Config, key: impl Into<String>) -> Self {
        Self {
            config,
            key: key.into(),
            cached_size: None,
        }
    }

    /// Returns the object key.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Returns the bucket name.
    #[must_use]
    pub fn bucket(&self) -> &str {
        &self.config.bucket
    }

    /// Build the HTTP URL for this object.
    #[must_use]
    pub fn object_url(&self) -> String {
        let endpoint = self.config.endpoint.trim_end_matches('/');
        if self.config.path_style {
            format!("{}/{}/{}", endpoint, self.config.bucket, self.key)
        } else {
            let host = endpoint
                .strip_prefix("https://")
                .or_else(|| endpoint.strip_prefix("http://"))
                .unwrap_or(endpoint);
            let scheme = if endpoint.starts_with("https://") {
                "https"
            } else {
                "http"
            };
            format!("{scheme}://{}.{host}/{}", self.config.bucket, self.key)
        }
    }

    /// Build request headers for a byte-range GET.
    ///
    /// Returns a `HashMap` of header name → value pairs that callers can
    /// forward to their HTTP client.
    #[must_use]
    pub fn range_request_headers(&self, range: ByteRange) -> HashMap<String, String> {
        let mut headers = self.config.extra_headers.clone();
        headers.insert("Range".to_string(), range.http_header());
        headers.insert(
            "x-amz-content-sha256".to_string(),
            "UNSIGNED-PAYLOAD".to_string(),
        );
        if let Some(token) = &self.config.credentials.session_token {
            headers.insert("x-amz-security-token".to_string(), token.clone());
        }
        headers
    }

    /// Build request headers for a HEAD request (to fetch object metadata).
    #[must_use]
    pub fn head_request_headers(&self) -> HashMap<String, String> {
        let mut headers = self.config.extra_headers.clone();
        headers.insert(
            "x-amz-content-sha256".to_string(),
            "UNSIGNED-PAYLOAD".to_string(),
        );
        if let Some(token) = &self.config.credentials.session_token {
            headers.insert("x-amz-security-token".to_string(), token.clone());
        }
        headers
    }

    /// Set the cached object size (populated after a successful HEAD request).
    pub fn set_size(&mut self, size: u64) {
        self.cached_size = Some(size);
    }

    /// Returns the cached object size, if known.
    #[must_use]
    pub fn cached_size(&self) -> Option<u64> {
        self.cached_size
    }

    /// Compute the S3 HMAC-SHA256 signing key bytes (AWS Signature Version 4).
    ///
    /// This does not depend on a network call and is provided as a utility for
    /// callers who build their own signed requests.
    ///
    /// Algorithm: HMAC-SHA256(HMAC-SHA256(HMAC-SHA256(HMAC-SHA256(
    ///   "AWS4" + SecretKey, DateStamp), Region), ServiceName), "aws4_request")
    #[must_use]
    pub fn signing_key(&self, date_stamp: &str) -> Vec<u8> {
        let k_date = hmac_sha256(
            format!("AWS4{}", self.config.credentials.secret_access_key).as_bytes(),
            date_stamp.as_bytes(),
        );
        let k_region = hmac_sha256(&k_date, self.config.region.as_bytes());
        let k_service = hmac_sha256(&k_region, b"s3");
        hmac_sha256(&k_service, b"aws4_request")
    }

    /// Parse object metadata from HTTP response headers.
    ///
    /// Recognises `Content-Length`, `ETag`, `Last-Modified`, `Content-Type`
    /// and any `x-amz-meta-*` headers.
    #[must_use]
    pub fn parse_object_meta(
        key: &str,
        headers: &HashMap<String, String>,
    ) -> Option<S3ObjectMeta> {
        let size_str = headers
            .get("Content-Length")
            .or_else(|| headers.get("content-length"))?;
        let size: u64 = size_str.trim().parse().ok()?;

        let etag = headers
            .get("ETag")
            .or_else(|| headers.get("etag"))
            .cloned()
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();

        let last_modified = headers
            .get("Last-Modified")
            .or_else(|| headers.get("last-modified"))
            .cloned()
            .unwrap_or_default();

        let content_type = headers
            .get("Content-Type")
            .or_else(|| headers.get("content-type"))
            .cloned();

        let user_metadata: HashMap<String, String> = headers
            .iter()
            .filter_map(|(k, v)| {
                let lower = k.to_lowercase();
                lower
                    .strip_prefix("x-amz-meta-")
                    .map(|suffix| (suffix.to_string(), v.clone()))
            })
            .collect();

        Some(S3ObjectMeta {
            key: key.to_string(),
            size,
            etag,
            last_modified,
            content_type,
            user_metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust HMAC-SHA256 (no external crate)
// ---------------------------------------------------------------------------

/// Compute HMAC-SHA256(key, data) — pure Rust implementation.
fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    const BLOCK: usize = 64;

    // Normalise key length
    let mut k = if key.len() > BLOCK {
        sha256(key)
    } else {
        key.to_vec()
    };
    k.resize(BLOCK, 0u8);

    let i_pad: Vec<u8> = k.iter().map(|b| b ^ 0x36).collect();
    let o_pad: Vec<u8> = k.iter().map(|b| b ^ 0x5c).collect();

    let mut inner = i_pad;
    inner.extend_from_slice(data);
    let inner_hash = sha256(&inner);

    let mut outer = o_pad;
    outer.extend_from_slice(&inner_hash);
    sha256(&outer)
}

/// Pure-Rust SHA-256 digest.
fn sha256(data: &[u8]) -> Vec<u8> {
    // FIPS 180-4 SHA-256 initial hash values
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Round constants
    let k: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    // Pre-processing: padding
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit block
    for block in msg.chunks(64) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut result = Vec::with_capacity(32);
    for word in h {
        result.extend_from_slice(&word.to_be_bytes());
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_range_http_header() {
        let r = ByteRange::new(0, 1024);
        assert_eq!(r.http_header(), "bytes=0-1023");
    }

    #[test]
    fn test_byte_range_open_ended() {
        let r = ByteRange::new(512, u64::MAX);
        assert_eq!(r.http_header(), "bytes=512-");
    }

    #[test]
    fn test_byte_range_from_offset_length() {
        let r = ByteRange::from_offset_length(100, 50);
        assert_eq!(r.start, 100);
        assert_eq!(r.end, 150);
        assert_eq!(r.length(), 50);
    }

    #[test]
    fn test_s3_source_path_style_url() {
        let config = S3Config {
            endpoint: "https://minio.example.com".to_string(),
            bucket: "media".to_string(),
            path_style: true,
            ..Default::default()
        };
        let src = S3Source::new(config, "videos/test.mp4");
        assert_eq!(
            src.object_url(),
            "https://minio.example.com/media/videos/test.mp4"
        );
    }

    #[test]
    fn test_s3_source_virtual_hosted_url() {
        let config = S3Config {
            endpoint: "https://s3.amazonaws.com".to_string(),
            bucket: "my-bucket".to_string(),
            path_style: false,
            ..Default::default()
        };
        let src = S3Source::new(config, "clip.mov");
        assert!(src.object_url().contains("my-bucket"));
        assert!(src.object_url().contains("clip.mov"));
    }

    #[test]
    fn test_parse_object_meta() {
        let mut headers = HashMap::new();
        headers.insert("Content-Length".to_string(), "1048576".to_string());
        headers.insert("ETag".to_string(), "\"abc123\"".to_string());
        headers.insert("Last-Modified".to_string(), "Thu, 01 Jan 2026 00:00:00 GMT".to_string());
        headers.insert("Content-Type".to_string(), "video/mp4".to_string());
        headers.insert("x-amz-meta-title".to_string(), "My Video".to_string());

        let meta = S3Source::parse_object_meta("test.mp4", &headers);
        assert!(meta.is_some());
        let meta = meta.unwrap();
        assert_eq!(meta.size, 1048576);
        assert_eq!(meta.etag, "abc123");
        assert_eq!(meta.user_metadata.get("title"), Some(&"My Video".to_string()));
    }

    #[test]
    fn test_sha256_known_value() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let digest = sha256(b"");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_hello() {
        // SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        let digest = sha256(b"hello");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn test_signing_key_does_not_panic() {
        let config = S3Config {
            credentials: S3Credentials::new("AKID", "secret"),
            region: "us-west-2".to_string(),
            ..Default::default()
        };
        let src = S3Source::new(config, "file.mp4");
        let key = src.signing_key("20260101");
        assert_eq!(key.len(), 32);
    }
}
