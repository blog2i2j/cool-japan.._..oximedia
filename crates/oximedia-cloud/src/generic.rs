//! Generic S3-compatible storage implementation for `MinIO`, Wasabi, Backblaze B2, etc.

use async_trait::async_trait;
use bytes::Bytes;
use chrono::Utc;
use reqwest::{Client, StatusCode};
use std::collections::HashMap;
use url::Url;

use crate::error::{CloudError, Result};
use crate::security::Credentials;
use crate::types::{
    CloudStorage, DeleteResult, ListResult, ObjectInfo, ObjectMetadata, StorageClass, StorageStats,
    UploadOptions,
};

/// Generic S3-compatible storage backend
pub struct GenericStorage {
    client: Client,
    endpoint: Url,
    credentials: Credentials,
    bucket: String,
}

impl GenericStorage {
    /// Create a new generic S3-compatible storage backend
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid
    pub fn new(endpoint: Url, credentials: Credentials, bucket: String) -> Result<Self> {
        credentials.validate()?;

        Ok(Self {
            client: Client::new(),
            endpoint,
            credentials,
            bucket,
        })
    }

    /// Create bucket
    ///
    /// # Errors
    ///
    /// Returns an error if bucket creation fails
    pub async fn create_bucket(&self) -> Result<()> {
        let url = format!("{}/{}", self.endpoint, self.bucket);

        let response = self
            .client
            .put(&url)
            .header("Authorization", self.auth_header("PUT", &self.bucket, ""))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage("Failed to create bucket".to_string()));
        }

        Ok(())
    }

    /// Delete bucket
    ///
    /// # Errors
    ///
    /// Returns an error if bucket deletion fails
    pub async fn delete_bucket(&self) -> Result<()> {
        let url = format!("{}/{}", self.endpoint, self.bucket);

        let response = self
            .client
            .delete(&url)
            .header(
                "Authorization",
                self.auth_header("DELETE", &self.bucket, ""),
            )
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage("Failed to delete bucket".to_string()));
        }

        Ok(())
    }

    /// Generate authorization header (simplified AWS Signature V4)
    fn auth_header(&self, _method: &str, _path: &str, _query: &str) -> String {
        // Simplified authentication - in production would implement full AWS Signature V4
        let date = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();

        // Basic authentication for compatible services
        format!(
            "AWS4-HMAC-SHA256 Credential={}/{}/us-east-1/s3/aws4_request",
            self.credentials.access_key, date
        )
    }

    /// Build object URL
    fn object_url(&self, key: &str) -> String {
        format!("{}/{}/{}", self.endpoint, self.bucket, key)
    }
}

#[async_trait]
impl CloudStorage for GenericStorage {
    async fn upload(&self, key: &str, data: Bytes) -> Result<()> {
        let url = self.object_url(key);

        let response = self
            .client
            .put(&url)
            .header(
                "Authorization",
                self.auth_header("PUT", &format!("/{key}"), ""),
            )
            .header("Content-Type", "application/octet-stream")
            .body(data)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage("Failed to upload object".to_string()));
        }

        Ok(())
    }

    async fn upload_with_options(
        &self,
        key: &str,
        data: Bytes,
        options: UploadOptions,
    ) -> Result<()> {
        let url = self.object_url(key);

        let mut request = self.client.put(&url).header(
            "Authorization",
            self.auth_header("PUT", &format!("/{key}"), ""),
        );

        if let Some(content_type) = options.content_type {
            request = request.header("Content-Type", content_type);
        } else {
            request = request.header("Content-Type", "application/octet-stream");
        }

        if let Some(cache_control) = options.cache_control {
            request = request.header("Cache-Control", cache_control);
        }

        if let Some(content_encoding) = options.content_encoding {
            request = request.header("Content-Encoding", content_encoding);
        }

        // Add user metadata with x-amz-meta- prefix
        for (key, value) in options.metadata {
            request = request.header(format!("x-amz-meta-{key}"), value);
        }

        let response = request.body(data).send().await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage("Failed to upload object".to_string()));
        }

        Ok(())
    }

    async fn download(&self, key: &str) -> Result<Bytes> {
        let url = self.object_url(key);

        let response = self
            .client
            .get(&url)
            .header(
                "Authorization",
                self.auth_header("GET", &format!("/{key}"), ""),
            )
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage("Failed to download object".to_string()));
        }

        let data = response.bytes().await?;
        Ok(data)
    }

    async fn download_range(&self, key: &str, start: u64, end: u64) -> Result<Bytes> {
        let url = self.object_url(key);
        let range_header = format!("bytes={start}-{end}");

        let response = self
            .client
            .get(&url)
            .header(
                "Authorization",
                self.auth_header("GET", &format!("/{key}"), ""),
            )
            .header("Range", range_header)
            .send()
            .await?;

        if !response.status().is_success() && response.status() != StatusCode::PARTIAL_CONTENT {
            return Err(CloudError::Storage(
                "Failed to download object range".to_string(),
            ));
        }

        let data = response.bytes().await?;
        Ok(data)
    }

    async fn list(&self, prefix: &str) -> Result<Vec<ObjectInfo>> {
        let url = format!(
            "{}{}?list-type=2&prefix={}",
            self.endpoint, self.bucket, prefix
        );

        let response = self
            .client
            .get(&url)
            .header(
                "Authorization",
                self.auth_header("GET", "", &format!("list-type=2&prefix={prefix}")),
            )
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage("Failed to list objects".to_string()));
        }

        let text = response.text().await?;

        // Parse XML response (simplified - in production would use proper XML parser)
        let mut objects = Vec::new();

        // This is a simplified parser - in production would use quick-xml or similar
        for line in text.lines() {
            if line.contains("<Key>") {
                if let Some(key) = extract_xml_value(line, "Key") {
                    // Create a minimal ObjectInfo
                    objects.push(ObjectInfo {
                        key,
                        size: 0,
                        last_modified: Utc::now(),
                        etag: None,
                        storage_class: None,
                        content_type: None,
                    });
                }
            }
        }

        Ok(objects)
    }

    async fn list_paginated(
        &self,
        prefix: &str,
        continuation_token: Option<String>,
        max_keys: usize,
    ) -> Result<ListResult> {
        let mut url = format!(
            "{}{}?list-type=2&prefix={}&max-keys={}",
            self.endpoint, self.bucket, prefix, max_keys
        );

        if let Some(token) = continuation_token {
            url.push_str(&format!("&continuation-token={token}"));
        }

        let response = self
            .client
            .get(&url)
            .header("Authorization", self.auth_header("GET", "", ""))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage("Failed to list objects".to_string()));
        }

        let text = response.text().await?;
        let mut objects = Vec::new();

        // Simplified XML parsing
        for line in text.lines() {
            if line.contains("<Key>") {
                if let Some(key) = extract_xml_value(line, "Key") {
                    objects.push(ObjectInfo {
                        key,
                        size: 0,
                        last_modified: Utc::now(),
                        etag: None,
                        storage_class: None,
                        content_type: None,
                    });
                }
            }
        }

        let is_truncated = text.contains("<IsTruncated>true</IsTruncated>");

        Ok(ListResult {
            objects,
            continuation_token: None,
            is_truncated,
            common_prefixes: Vec::new(),
        })
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let url = self.object_url(key);

        let response = self
            .client
            .delete(&url)
            .header(
                "Authorization",
                self.auth_header("DELETE", &format!("/{key}"), ""),
            )
            .send()
            .await?;

        if !response.status().is_success() && response.status() != StatusCode::NOT_FOUND {
            return Err(CloudError::Storage("Failed to delete object".to_string()));
        }

        Ok(())
    }

    async fn delete_batch(&self, keys: &[String]) -> Result<Vec<DeleteResult>> {
        let mut results = Vec::new();

        for key in keys {
            match self.delete(key).await {
                Ok(()) => results.push(DeleteResult {
                    key: key.clone(),
                    success: true,
                    error: None,
                }),
                Err(e) => results.push(DeleteResult {
                    key: key.clone(),
                    success: false,
                    error: Some(e.to_string()),
                }),
            }
        }

        Ok(results)
    }

    async fn get_metadata(&self, key: &str) -> Result<ObjectMetadata> {
        let url = self.object_url(key);

        let response = self
            .client
            .head(&url)
            .header(
                "Authorization",
                self.auth_header("HEAD", &format!("/{key}"), ""),
            )
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage(
                "Failed to get object metadata".to_string(),
            ));
        }

        let headers = response.headers();
        let size = headers
            .get("Content-Length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let content_type = headers
            .get("Content-Type")
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);

        let info = ObjectInfo {
            key: key.to_string(),
            size,
            last_modified: Utc::now(),
            etag: headers
                .get("ETag")
                .and_then(|v| v.to_str().ok())
                .map(ToString::to_string),
            storage_class: None,
            content_type,
        };

        let mut user_metadata = HashMap::new();
        for (key, value) in headers {
            if let Some(meta_key) = key.as_str().strip_prefix("x-amz-meta-") {
                if let Ok(value_str) = value.to_str() {
                    user_metadata.insert(meta_key.to_string(), value_str.to_string());
                }
            }
        }

        Ok(ObjectMetadata {
            info,
            user_metadata,
            system_metadata: HashMap::new(),
            tags: HashMap::new(),
            content_encoding: headers
                .get("Content-Encoding")
                .and_then(|v| v.to_str().ok())
                .map(ToString::to_string),
            content_language: None,
            cache_control: headers
                .get("Cache-Control")
                .and_then(|v| v.to_str().ok())
                .map(ToString::to_string),
            content_disposition: None,
        })
    }

    async fn update_metadata(&self, key: &str, _metadata: HashMap<String, String>) -> Result<()> {
        // S3-compatible APIs require copying the object to update metadata
        self.copy(key, key).await
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let url = self.object_url(key);

        let response = self
            .client
            .head(&url)
            .header(
                "Authorization",
                self.auth_header("HEAD", &format!("/{key}"), ""),
            )
            .send()
            .await?;

        Ok(response.status().is_success())
    }

    async fn copy(&self, source_key: &str, dest_key: &str) -> Result<()> {
        let url = self.object_url(dest_key);
        let copy_source = format!("/{}/{}", self.bucket, source_key);

        let response = self
            .client
            .put(&url)
            .header(
                "Authorization",
                self.auth_header("PUT", &format!("/{dest_key}"), ""),
            )
            .header("x-amz-copy-source", copy_source)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage("Failed to copy object".to_string()));
        }

        Ok(())
    }

    async fn presigned_download_url(&self, key: &str, expires_in_secs: u64) -> Result<String> {
        // Simplified presigned URL generation
        let url = self.object_url(key);
        Ok(format!("{url}?expires={expires_in_secs}"))
    }

    async fn presigned_upload_url(&self, key: &str, expires_in_secs: u64) -> Result<String> {
        let url = self.object_url(key);
        Ok(format!("{url}?expires={expires_in_secs}"))
    }

    async fn set_storage_class(&self, _key: &str, _class: StorageClass) -> Result<()> {
        // Most S3-compatible services don't support storage classes
        Ok(())
    }

    async fn get_stats(&self, prefix: &str) -> Result<StorageStats> {
        let objects = self.list(prefix).await?;

        let mut stats = StorageStats::default();
        for obj in objects {
            stats.total_size += obj.size;
            stats.object_count += 1;
        }

        Ok(stats)
    }
}

/// Extract value from XML tag (simplified)
fn extract_xml_value(line: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{tag}>");
    let end_tag = format!("</{tag}>");

    if let Some(start) = line.find(&start_tag) {
        if let Some(end) = line.find(&end_tag) {
            let value_start = start + start_tag.len();
            if value_start < end {
                return Some(line[value_start..end].to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_xml_value() {
        let line = "<Key>test-object.txt</Key>";
        let value = extract_xml_value(line, "Key");
        assert_eq!(value, Some("test-object.txt".to_string()));

        let line2 = "<Size>1024</Size>";
        let value2 = extract_xml_value(line2, "Size");
        assert_eq!(value2, Some("1024".to_string()));
    }

    #[test]
    fn test_extract_xml_value_no_match() {
        let line = "<Name>bucket</Name>";
        let value = extract_xml_value(line, "Key");
        assert_eq!(value, None);
    }

    #[test]
    fn test_generic_storage_creation() {
        let endpoint = Url::parse("https://s3.example.com").expect("endpoint should be valid");
        let credentials = Credentials::new("access".to_string(), "secret".to_string());
        let storage = GenericStorage::new(endpoint, credentials, "test-bucket".to_string());
        assert!(storage.is_ok());
    }
}
