//! Security features including credentials management and encryption

use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;

use crate::error::{CloudError, Result};

/// Cloud credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credentials {
    /// Access key ID
    pub access_key: String,
    /// Secret access key
    pub secret_key: String,
    /// Session token (optional, for temporary credentials)
    pub session_token: Option<String>,
    /// Additional provider-specific credentials
    pub extra: HashMap<String, String>,
}

impl Credentials {
    /// Create new credentials
    #[must_use]
    pub fn new(access_key: String, secret_key: String) -> Self {
        Self {
            access_key,
            secret_key,
            session_token: None,
            extra: HashMap::new(),
        }
    }

    /// Create credentials with session token
    #[must_use]
    pub fn with_session_token(
        access_key: String,
        secret_key: String,
        session_token: String,
    ) -> Self {
        Self {
            access_key,
            secret_key,
            session_token: Some(session_token),
            extra: HashMap::new(),
        }
    }

    /// Validate credentials
    pub fn validate(&self) -> Result<()> {
        if self.access_key.is_empty() {
            return Err(CloudError::InvalidConfig("Access key is empty".to_string()));
        }
        if self.secret_key.is_empty() {
            return Err(CloudError::InvalidConfig("Secret key is empty".to_string()));
        }
        Ok(())
    }

    /// Check if credentials are temporary (have session token)
    #[must_use]
    pub fn is_temporary(&self) -> bool {
        self.session_token.is_some()
    }
}

/// Encryption configuration
#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,
    /// KMS configuration (if using KMS)
    pub kms_config: Option<KmsConfig>,
    /// Customer-provided key (if using client-side encryption)
    pub customer_key: Option<Vec<u8>>,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            algorithm: EncryptionAlgorithm::AES256,
            kms_config: None,
            customer_key: None,
        }
    }
}

/// Encryption algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionAlgorithm {
    /// AES-256
    AES256,
    /// AWS KMS
    AwsKms,
    /// Azure Key Vault
    AzureKeyVault,
    /// GCP KMS
    GcpKms,
}

/// KMS (Key Management Service) configuration
#[derive(Debug, Clone)]
pub struct KmsConfig {
    /// KMS key ID or ARN
    pub key_id: String,
    /// KMS endpoint (optional)
    pub endpoint: Option<String>,
    /// Additional context
    pub context: HashMap<String, String>,
}

impl KmsConfig {
    /// Create new KMS configuration
    #[must_use]
    pub fn new(key_id: String) -> Self {
        Self {
            key_id,
            endpoint: None,
            context: HashMap::new(),
        }
    }

    /// Add encryption context
    pub fn add_context(&mut self, key: String, value: String) {
        self.context.insert(key, value);
    }
}

/// IAM role configuration for AWS
#[derive(Debug, Clone)]
pub struct IamRoleConfig {
    /// Role ARN
    pub role_arn: String,
    /// Session name
    pub session_name: String,
    /// External ID (for cross-account access)
    pub external_id: Option<String>,
    /// Session duration in seconds
    pub duration_secs: u32,
}

impl IamRoleConfig {
    /// Create new IAM role configuration
    #[must_use]
    pub fn new(role_arn: String, session_name: String) -> Self {
        Self {
            role_arn,
            session_name,
            external_id: None,
            duration_secs: 3600, // 1 hour default
        }
    }

    /// Set external ID
    #[must_use]
    pub fn with_external_id(mut self, external_id: String) -> Self {
        self.external_id = Some(external_id);
        self
    }

    /// Set session duration
    #[must_use]
    pub fn with_duration(mut self, duration_secs: u32) -> Self {
        self.duration_secs = duration_secs;
        self
    }
}

/// Service principal configuration for Azure
#[derive(Debug, Clone)]
pub struct ServicePrincipalConfig {
    /// Tenant ID
    pub tenant_id: String,
    /// Client ID
    pub client_id: String,
    /// Client secret
    pub client_secret: String,
}

impl ServicePrincipalConfig {
    /// Create new service principal configuration
    #[must_use]
    pub fn new(tenant_id: String, client_id: String, client_secret: String) -> Self {
        Self {
            tenant_id,
            client_id,
            client_secret,
        }
    }
}

/// Service account configuration for GCP
#[derive(Debug, Clone)]
pub struct ServiceAccountConfig {
    /// Project ID
    pub project_id: String,
    /// Service account email
    pub email: String,
    /// Private key (PEM format)
    pub private_key: String,
}

impl ServiceAccountConfig {
    /// Create new service account configuration
    #[must_use]
    pub fn new(project_id: String, email: String, private_key: String) -> Self {
        Self {
            project_id,
            email,
            private_key,
        }
    }
}

/// Credential rotation manager
pub struct CredentialRotation {
    /// Current credentials
    current: Credentials,
    /// Rotation interval in seconds
    rotation_interval_secs: u64,
    /// Last rotation timestamp
    last_rotation: std::time::Instant,
}

impl CredentialRotation {
    /// Create new credential rotation manager
    #[must_use]
    pub fn new(credentials: Credentials, rotation_interval_secs: u64) -> Self {
        Self {
            current: credentials,
            rotation_interval_secs,
            last_rotation: std::time::Instant::now(),
        }
    }

    /// Check if rotation is needed
    #[must_use]
    pub fn needs_rotation(&self) -> bool {
        self.last_rotation.elapsed().as_secs() >= self.rotation_interval_secs
    }

    /// Get current credentials
    #[must_use]
    pub fn current(&self) -> &Credentials {
        &self.current
    }

    /// Update credentials
    pub fn rotate(&mut self, new_credentials: Credentials) {
        self.current = new_credentials;
        self.last_rotation = std::time::Instant::now();
    }
}

/// Access Control List (ACL) options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acl {
    /// Private (owner only)
    Private,
    /// Public read
    PublicRead,
    /// Public read-write
    PublicReadWrite,
    /// Authenticated read
    AuthenticatedRead,
    /// Bucket owner read
    BucketOwnerRead,
    /// Bucket owner full control
    BucketOwnerFullControl,
}

impl Acl {
    /// Convert to AWS S3 ACL string
    #[must_use]
    pub fn to_s3_string(&self) -> &str {
        match self {
            Acl::Private => "private",
            Acl::PublicRead => "public-read",
            Acl::PublicReadWrite => "public-read-write",
            Acl::AuthenticatedRead => "authenticated-read",
            Acl::BucketOwnerRead => "bucket-owner-read",
            Acl::BucketOwnerFullControl => "bucket-owner-full-control",
        }
    }
}

// ── Signed URL generation ────────────────────────────────────────────────────

type HmacSha256 = Hmac<Sha256>;

/// Generates HMAC-SHA256 signed URLs for pre-authorised object access.
///
/// The signed URL encodes the bucket, key, expiry, a timestamp, and an
/// HMAC-SHA256 signature computed over those components using the provided
/// secret.  The format mimics the AWS S3 pre-signed URL query-parameter style.
pub struct SignedUrl;

impl SignedUrl {
    /// Generate a signed URL for the given `bucket`/`key` pair.
    ///
    /// Parameters:
    /// - `bucket`      – cloud storage bucket or container name.
    /// - `key`         – object key / path within the bucket.
    /// - `expiry_secs` – how many seconds the URL remains valid.
    /// - `secret`      – the HMAC secret used to sign the URL.
    ///
    /// The returned URL contains:
    /// - `X-Amz-Expires`         – expiry window in seconds.
    /// - `X-Amz-Date`            – approximate creation date (Unix epoch).
    /// - `X-Amz-SignedHeaders`   – fixed value `host`.
    /// - `X-Amz-Signature`       – hex-encoded HMAC-SHA256 signature.
    ///
    /// The string that is signed is:
    /// `"{bucket}\n{key}\n{expiry_secs}\n{epoch}"`
    #[must_use]
    pub fn generate(bucket: &str, key: &str, expiry_secs: u64, secret: &[u8]) -> String {
        let epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let string_to_sign = format!("{bucket}\n{key}\n{expiry_secs}\n{epoch}");

        let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC accepts any key length");
        mac.update(string_to_sign.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());

        // URL-encode the key (replace '/' with '%2F' for path segments)
        let encoded_key = urlencoding::encode(key);
        let encoded_bucket = urlencoding::encode(bucket);

        format!(
            "https://s3.amazonaws.com/{encoded_bucket}/{encoded_key}\
            ?X-Amz-Expires={expiry_secs}\
            &X-Amz-Date={epoch}\
            &X-Amz-SignedHeaders=host\
            &X-Amz-Signature={signature}"
        )
    }

    /// Verify whether a signed URL is still valid and has the correct signature.
    ///
    /// Returns `true` if the signature matches and the URL has not expired.
    #[must_use]
    pub fn verify(url: &str, secret: &[u8]) -> bool {
        // Parse query parameters from the URL
        let (base, query) = match url.split_once('?') {
            Some(pair) => pair,
            None => return false,
        };

        // Extract path components: /bucket/key
        let path = base.trim_start_matches("https://s3.amazonaws.com/");
        let (encoded_bucket, encoded_key) = match path.split_once('/') {
            Some(pair) => pair,
            None => return false,
        };
        let bucket = urlencoding::decode(encoded_bucket).unwrap_or_default();
        let key = urlencoding::decode(encoded_key).unwrap_or_default();

        // Parse query params
        let mut expiry_secs: Option<u64> = None;
        let mut date_epoch: Option<u64> = None;
        let mut provided_sig: Option<String> = None;

        for param in query.split('&') {
            if let Some(val) = param.strip_prefix("X-Amz-Expires=") {
                expiry_secs = val.parse().ok();
            } else if let Some(val) = param.strip_prefix("X-Amz-Date=") {
                date_epoch = val.parse().ok();
            } else if let Some(val) = param.strip_prefix("X-Amz-Signature=") {
                provided_sig = Some(val.to_string());
            }
        }

        let (Some(expiry), Some(epoch), Some(sig)) = (expiry_secs, date_epoch, provided_sig) else {
            return false;
        };

        // Check expiry
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if now > epoch.saturating_add(expiry) {
            return false;
        }

        // Re-compute expected signature
        let string_to_sign = format!("{bucket}\n{key}\n{expiry}\n{epoch}");
        let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC accepts any key length");
        mac.update(string_to_sign.as_bytes());
        let expected_sig = hex::encode(mac.finalize().into_bytes());

        expected_sig == sig
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credentials_validation() {
        let valid = Credentials::new("access".to_string(), "secret".to_string());
        assert!(valid.validate().is_ok());

        let invalid = Credentials::new("".to_string(), "secret".to_string());
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_credentials_temporary() {
        let permanent = Credentials::new("access".to_string(), "secret".to_string());
        assert!(!permanent.is_temporary());

        let temporary = Credentials::with_session_token(
            "access".to_string(),
            "secret".to_string(),
            "token".to_string(),
        );
        assert!(temporary.is_temporary());
    }

    #[test]
    fn test_kms_config() {
        let mut kms = KmsConfig::new("key-id".to_string());
        kms.add_context("env".to_string(), "prod".to_string());
        assert_eq!(kms.context.len(), 1);
    }

    #[test]
    fn test_iam_role_config() {
        let role = IamRoleConfig::new(
            "arn:aws:iam::123456789012:role/test".to_string(),
            "session".to_string(),
        )
        .with_external_id("external".to_string())
        .with_duration(7200);

        assert_eq!(role.external_id, Some("external".to_string()));
        assert_eq!(role.duration_secs, 7200);
    }

    #[test]
    fn test_credential_rotation() {
        let creds = Credentials::new("access".to_string(), "secret".to_string());
        let mut rotation = CredentialRotation::new(creds.clone(), 60);

        assert!(!rotation.needs_rotation());

        let new_creds = Credentials::new("new_access".to_string(), "new_secret".to_string());
        rotation.rotate(new_creds);
        assert_eq!(rotation.current().access_key, "new_access");
    }

    #[test]
    fn test_acl_to_string() {
        assert_eq!(Acl::Private.to_s3_string(), "private");
        assert_eq!(Acl::PublicRead.to_s3_string(), "public-read");
    }

    // ── SignedUrl tests ───────────────────────────────────────────────────────

    #[test]
    fn test_signed_url_contains_required_params() {
        let url = SignedUrl::generate("my-bucket", "videos/file.mp4", 3600, b"supersecret");
        assert!(
            url.contains("X-Amz-Expires=3600"),
            "URL must include expiry"
        );
        assert!(url.contains("X-Amz-Date="), "URL must include date");
        assert!(
            url.contains("X-Amz-Signature="),
            "URL must include signature"
        );
        assert!(
            url.contains("X-Amz-SignedHeaders=host"),
            "URL must include signed headers"
        );
    }

    #[test]
    fn test_signed_url_contains_bucket_and_key() {
        let url = SignedUrl::generate("my-bucket", "path/to/object.mp4", 300, b"key");
        assert!(url.contains("my-bucket"), "URL must include bucket");
        // Key is URL-encoded
        assert!(url.contains("path"), "URL must include key path");
    }

    #[test]
    fn test_signed_url_different_secrets_produce_different_signatures() {
        let url1 = SignedUrl::generate("bucket", "key", 300, b"secret1");
        let url2 = SignedUrl::generate("bucket", "key", 300, b"secret2");
        // Extract signatures
        let sig1 = url1.split("X-Amz-Signature=").nth(1).unwrap_or("");
        let sig2 = url2.split("X-Amz-Signature=").nth(1).unwrap_or("");
        assert_ne!(
            sig1, sig2,
            "Different secrets must produce different signatures"
        );
    }

    #[test]
    fn test_signed_url_verify_valid() {
        let secret = b"test-signing-secret";
        let url = SignedUrl::generate("bucket", "key/file.mp4", 3600, secret);
        assert!(
            SignedUrl::verify(&url, secret),
            "Freshly generated URL must verify"
        );
    }

    #[test]
    fn test_signed_url_verify_wrong_secret_fails() {
        let url = SignedUrl::generate("bucket", "key.mp4", 3600, b"correct-secret");
        assert!(
            !SignedUrl::verify(&url, b"wrong-secret"),
            "Wrong secret must not verify"
        );
    }

    #[test]
    fn test_signed_url_verify_malformed_url_fails() {
        assert!(!SignedUrl::verify("not-a-valid-url", b"secret"));
        assert!(!SignedUrl::verify(
            "https://s3.amazonaws.com/bucket/key",
            b"secret"
        ));
    }
}
