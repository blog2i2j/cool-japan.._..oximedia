//! Authentication and authorization for live streaming.
//!
//! This module provides authentication mechanisms including:
//! - Token-based authentication
//! - Stream key validation
//! - Access control lists
//! - JWT token support

use crate::error::{NetError, NetResult};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use hmac::{Hmac, KeyInit, Mac};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;
use std::sync::Arc;

/// Authentication result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthResult {
    /// Authentication successful.
    Success,

    /// Authentication failed with reason.
    Failed(String),
}

/// Authentication handler trait.
#[async_trait::async_trait]
pub trait AuthHandler: Send + Sync {
    /// Authenticates a publish request.
    async fn authenticate_publish(
        &self,
        stream_key: &str,
        app_name: &str,
        token: Option<&str>,
    ) -> AuthResult;

    /// Authenticates a playback request.
    async fn authenticate_playback(
        &self,
        stream_key: &str,
        app_name: &str,
        token: Option<&str>,
    ) -> AuthResult;

    /// Authenticates an API request.
    async fn authenticate_api(&self, token: &str) -> AuthResult;
}

/// Simple token-based authentication.
pub struct TokenAuth {
    /// Publish tokens (stream_key -> token).
    publish_tokens: RwLock<HashMap<String, String>>,

    /// Playback tokens (stream_key -> token).
    playback_tokens: RwLock<HashMap<String, String>>,

    /// API tokens.
    api_tokens: RwLock<HashMap<String, TokenInfo>>,

    /// Secret key for signing.
    secret_key: Vec<u8>,
}

impl TokenAuth {
    /// Creates a new token-based auth handler.
    #[must_use]
    pub fn new(secret_key: impl Into<Vec<u8>>) -> Self {
        Self {
            publish_tokens: RwLock::new(HashMap::new()),
            playback_tokens: RwLock::new(HashMap::new()),
            api_tokens: RwLock::new(HashMap::new()),
            secret_key: secret_key.into(),
        }
    }

    /// Adds a publish token.
    pub fn add_publish_token(&self, stream_key: impl Into<String>, token: impl Into<String>) {
        let mut tokens = self.publish_tokens.write();
        tokens.insert(stream_key.into(), token.into());
    }

    /// Adds a playback token.
    pub fn add_playback_token(&self, stream_key: impl Into<String>, token: impl Into<String>) {
        let mut tokens = self.playback_tokens.write();
        tokens.insert(stream_key.into(), token.into());
    }

    /// Generates a signed token.
    pub fn generate_token(
        &self,
        stream_key: &str,
        app_name: &str,
        expires_in: ChronoDuration,
    ) -> String {
        let expires_at = Utc::now() + expires_in;
        let payload = format!("{stream_key}:{app_name}:{}", expires_at.timestamp());

        let mut mac = Hmac::<Sha256>::new_from_slice(&self.secret_key)
            .expect("HMAC can take key of any size");
        mac.update(payload.as_bytes());
        let signature = mac.finalize().into_bytes();

        let signature_hex = hex::encode(signature);
        format!("{payload}:{signature_hex}")
    }

    /// Validates a signed token.
    pub fn validate_token(&self, token: &str, stream_key: &str, app_name: &str) -> bool {
        let parts: Vec<&str> = token.split(':').collect();
        if parts.len() != 4 {
            return false;
        }

        let (token_stream_key, token_app_name, expires_str, signature) =
            (parts[0], parts[1], parts[2], parts[3]);

        // Check stream key and app name
        if token_stream_key != stream_key || token_app_name != app_name {
            return false;
        }

        // Check expiration
        if let Ok(expires_timestamp) = expires_str.parse::<i64>() {
            if let Some(expires_at) = DateTime::from_timestamp(expires_timestamp, 0) {
                if Utc::now() > expires_at {
                    return false; // Expired
                }
            } else {
                return false;
            }
        } else {
            return false;
        }

        // Verify signature
        let payload = format!("{token_stream_key}:{token_app_name}:{expires_str}");
        let mut mac = Hmac::<Sha256>::new_from_slice(&self.secret_key)
            .expect("HMAC can take key of any size");
        mac.update(payload.as_bytes());
        let expected_signature = mac.finalize().into_bytes();
        let expected_hex = hex::encode(expected_signature);

        expected_hex == signature
    }

    /// Adds an API token.
    pub fn add_api_token(&self, token: impl Into<String>, info: TokenInfo) {
        let mut tokens = self.api_tokens.write();
        tokens.insert(token.into(), info);
    }

    /// Removes an API token.
    pub fn remove_api_token(&self, token: &str) {
        let mut tokens = self.api_tokens.write();
        tokens.remove(token);
    }
}

#[async_trait::async_trait]
impl AuthHandler for TokenAuth {
    async fn authenticate_publish(
        &self,
        stream_key: &str,
        app_name: &str,
        token: Option<&str>,
    ) -> AuthResult {
        let token = match token {
            Some(t) => t,
            None => return AuthResult::Failed("Missing token".to_string()),
        };

        // Check static tokens
        {
            let tokens = self.publish_tokens.read();
            if let Some(expected_token) = tokens.get(stream_key) {
                if expected_token == token {
                    return AuthResult::Success;
                }
            }
        }

        // Validate signed token
        if self.validate_token(token, stream_key, app_name) {
            return AuthResult::Success;
        }

        AuthResult::Failed("Invalid token".to_string())
    }

    async fn authenticate_playback(
        &self,
        stream_key: &str,
        app_name: &str,
        token: Option<&str>,
    ) -> AuthResult {
        let token = match token {
            Some(t) => t,
            None => return AuthResult::Success, // Allow playback without token by default
        };

        // Check static tokens
        {
            let tokens = self.playback_tokens.read();
            if let Some(expected_token) = tokens.get(stream_key) {
                if expected_token == token {
                    return AuthResult::Success;
                }
            }
        }

        // Validate signed token
        if self.validate_token(token, stream_key, app_name) {
            return AuthResult::Success;
        }

        AuthResult::Failed("Invalid token".to_string())
    }

    async fn authenticate_api(&self, token: &str) -> AuthResult {
        let tokens = self.api_tokens.read();

        if let Some(info) = tokens.get(token) {
            if let Some(expires_at) = info.expires_at {
                if Utc::now() > expires_at {
                    return AuthResult::Failed("Token expired".to_string());
                }
            }

            if info.enabled {
                return AuthResult::Success;
            }
        }

        AuthResult::Failed("Invalid API token".to_string())
    }
}

/// Token information for API tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    /// Token name/description.
    pub name: String,

    /// Token permissions.
    pub permissions: Vec<String>,

    /// Expiration time.
    pub expires_at: Option<DateTime<Utc>>,

    /// Is token enabled.
    pub enabled: bool,

    /// Creation time.
    pub created_at: DateTime<Utc>,
}

impl TokenInfo {
    /// Creates a new token info.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            permissions: Vec::new(),
            expires_at: None,
            enabled: true,
            created_at: Utc::now(),
        }
    }

    /// Adds a permission.
    #[must_use]
    pub fn with_permission(mut self, permission: impl Into<String>) -> Self {
        self.permissions.push(permission.into());
        self
    }

    /// Sets expiration.
    #[must_use]
    pub fn with_expiration(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Checks if token has permission.
    #[must_use]
    pub fn has_permission(&self, permission: &str) -> bool {
        self.permissions.contains(&permission.to_string())
    }
}

/// Token validator for validating tokens.
pub struct TokenValidator {
    auth_handler: Arc<dyn AuthHandler>,
}

impl TokenValidator {
    /// Creates a new token validator.
    #[must_use]
    pub fn new(auth_handler: Arc<dyn AuthHandler>) -> Self {
        Self { auth_handler }
    }

    /// Validates a publish token.
    pub async fn validate_publish(
        &self,
        stream_key: &str,
        app_name: &str,
        token: Option<&str>,
    ) -> NetResult<()> {
        match self
            .auth_handler
            .authenticate_publish(stream_key, app_name, token)
            .await
        {
            AuthResult::Success => Ok(()),
            AuthResult::Failed(reason) => Err(NetError::authentication(reason)),
        }
    }

    /// Validates a playback token.
    pub async fn validate_playback(
        &self,
        stream_key: &str,
        app_name: &str,
        token: Option<&str>,
    ) -> NetResult<()> {
        match self
            .auth_handler
            .authenticate_playback(stream_key, app_name, token)
            .await
        {
            AuthResult::Success => Ok(()),
            AuthResult::Failed(reason) => Err(NetError::authentication(reason)),
        }
    }

    /// Validates an API token.
    pub async fn validate_api(&self, token: &str) -> NetResult<()> {
        match self.auth_handler.authenticate_api(token).await {
            AuthResult::Success => Ok(()),
            AuthResult::Failed(reason) => Err(NetError::authentication(reason)),
        }
    }
}

/// Simple hex encoding/decoding.
mod hex {
    #[must_use]
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
    }
}
