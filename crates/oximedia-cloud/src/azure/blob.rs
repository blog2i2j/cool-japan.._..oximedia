//! Azure Blob Storage implementation using REST API with Azure Shared Key authentication.
//!
//! Implements the Azure Blob Service REST API with Shared Key authentication as documented at:
//! <https://docs.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key>

use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use bytes::Bytes;
use chrono::Utc;
use hmac::{Hmac, KeyInit, Mac};
use reqwest::Client;
use serde::Deserialize;
use sha2::Sha256;
use std::collections::HashMap;

use crate::error::{CloudError, Result};
use crate::types::{
    CloudStorage, DeleteResult, ListResult, ObjectInfo, ObjectMetadata, StorageClass, StorageStats,
    UploadOptions,
};

/// Azure Blob Storage backend using REST API with Shared Key authentication.
pub struct AzureBlobStorage {
    client: Client,
    account_name: String,
    container_name: String,
    /// Raw bytes of the base64-decoded storage account access key.
    access_key: Vec<u8>,
}

impl AzureBlobStorage {
    /// Create a new Azure Blob Storage backend.
    ///
    /// Reads the storage account key from the `AZURE_STORAGE_KEY` environment variable
    /// (base64-encoded, as shown in the Azure Portal).
    ///
    /// # Errors
    ///
    /// Returns `CloudError::Authentication` if the environment variable is absent or
    /// contains invalid base64.
    #[allow(clippy::unused_async)]
    pub async fn new(account_name: String, container_name: String) -> Result<Self> {
        let access_key_b64 = std::env::var("AZURE_STORAGE_KEY")
            .map_err(|_| CloudError::Authentication("AZURE_STORAGE_KEY not set".to_string()))?;

        let access_key = BASE64
            .decode(&access_key_b64)
            .map_err(|e| CloudError::Authentication(format!("Invalid AZURE_STORAGE_KEY: {e}")))?;

        Ok(Self {
            client: Client::new(),
            account_name,
            container_name,
            access_key,
        })
    }

    /// Get base URL for the container.
    fn container_url(&self) -> String {
        format!(
            "https://{}.blob.core.windows.net/{}",
            self.account_name, self.container_name
        )
    }

    /// Get the fully-qualified URL for a specific blob.
    fn blob_url(&self, key: &str) -> String {
        format!("{}/{}", self.container_url(), key)
    }

    /// Return the current time formatted as required by the `x-ms-date` header.
    fn ms_date() -> String {
        Utc::now().format("%a, %d %b %Y %H:%M:%S GMT").to_string()
    }

    /// Compute an HMAC-SHA256 signature over `string_to_sign` using the account key.
    fn sign_string(&self, string_to_sign: &str) -> Result<String> {
        type HmacSha256 = Hmac<Sha256>;
        let mut mac = HmacSha256::new_from_slice(&self.access_key)
            .map_err(|e| CloudError::Authentication(format!("HMAC init failed: {e}")))?;
        mac.update(string_to_sign.as_bytes());
        Ok(BASE64.encode(mac.finalize().into_bytes()))
    }

    /// Build an `Authorization: SharedKey` header value for a blob-level request.
    ///
    /// `canonicalized_headers` must already be formatted as `"header-name:value\n"` lines
    /// sorted alphabetically. `canonicalized_resource` must start with
    /// `"/accountname/containername/blobname"` and optionally contain a query component.
    fn build_auth_header(
        &self,
        method: &str,
        content_type: Option<&str>,
        _ms_date: &str,
        canonicalized_headers: &str,
        canonicalized_resource: &str,
    ) -> Result<String> {
        let content_type_str = content_type.unwrap_or("");

        let string_to_sign = format!(
            "{method}\n\n{content_type_str}\n\n{canonicalized_headers}{canonicalized_resource}",
        );

        let signature = self.sign_string(&string_to_sign)?;
        Ok(format!("SharedKey {}:{}", self.account_name, signature))
    }

    /// Build the canonicalized resource string for a blob (with an optional query component).
    ///
    /// For example, for `Set Blob Metadata` pass `query = Some("comp:metadata")`.
    fn blob_canonicalized_resource(&self, blob_name: &str, query: Option<&str>) -> String {
        let base = format!(
            "/{}/{}/{}",
            self.account_name, self.container_name, blob_name
        );
        match query {
            Some(q) if !q.is_empty() => format!("{base}\n{q}"),
            _ => base,
        }
    }

    /// Convert Azure access tier string to `StorageClass`.
    fn from_azure_tier(tier: &str) -> StorageClass {
        match tier {
            "Hot" => StorageClass::Standard,
            "Cool" => StorageClass::InfrequentAccess,
            "Archive" => StorageClass::Glacier,
            _ => StorageClass::Standard,
        }
    }

    /// Convert `StorageClass` to an Azure `BlobTier`.
    fn to_azure_tier(class: StorageClass) -> BlobTier {
        match class {
            StorageClass::Standard => BlobTier::Hot,
            StorageClass::InfrequentAccess => BlobTier::Cool,
            StorageClass::Glacier | StorageClass::DeepArchive => BlobTier::Archive,
            _ => BlobTier::Hot,
        }
    }

    /// Parse the Azure Blob Storage XML `List Blobs` response.
    ///
    /// Returns a tuple of `(Vec<ObjectInfo>, Option<next_marker>)`.  The next marker is
    /// present and non-empty only when the response is paginated (i.e. there are more results).
    fn parse_list_blobs_xml(xml: &str) -> (Vec<ObjectInfo>, Option<String>) {
        let mut objects = Vec::new();
        let mut next_marker: Option<String> = None;

        // Extract NextMarker for pagination
        if let Some(marker_val) = extract_xml_inner(xml, "NextMarker") {
            if !marker_val.is_empty() {
                next_marker = Some(marker_val);
            }
        }

        // Iterate over every <Blob>...</Blob> element
        let mut remaining = xml;
        while let Some(blob_start) = remaining.find("<Blob>") {
            let after_open = &remaining[blob_start + "<Blob>".len()..];
            let blob_end = after_open.find("</Blob>").unwrap_or(after_open.len());
            let blob_xml = &after_open[..blob_end];

            let name = extract_xml_inner(blob_xml, "Name").unwrap_or_default();

            // Content-Length may appear as <Content-Length> inside <Properties>
            let size = extract_xml_inner(blob_xml, "Content-Length")
                .or_else(|| extract_xml_inner(blob_xml, "Size"))
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);

            let last_modified_str =
                extract_xml_inner(blob_xml, "Last-Modified").unwrap_or_default();
            let last_modified = chrono::DateTime::parse_from_rfc2822(&last_modified_str)
                .ok()
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(Utc::now);

            let etag =
                extract_xml_inner(blob_xml, "Etag").or_else(|| extract_xml_inner(blob_xml, "ETag"));
            let content_type = extract_xml_inner(blob_xml, "Content-Type");
            let access_tier = extract_xml_inner(blob_xml, "AccessTier");
            let storage_class = access_tier.as_deref().map(Self::from_azure_tier);

            if !name.is_empty() {
                objects.push(ObjectInfo {
                    key: name,
                    size,
                    last_modified,
                    etag,
                    storage_class,
                    content_type,
                });
            }

            remaining = &after_open[blob_end..];
        }

        (objects, next_marker)
    }

    /// Poll the copy status of a destination blob until the server-side copy completes.
    ///
    /// After a `Copy Blob` request returns HTTP 202 (Accepted) Azure performs the copy
    /// asynchronously.  This method polls the `x-ms-copy-status` response header until
    /// it transitions to `"success"`, `"failed"`, or `"aborted"`.
    async fn wait_for_copy_completion(&self, dest_key: &str) -> Result<()> {
        const MAX_POLLS: u32 = 60;
        const POLL_INTERVAL_MS: u64 = 2_000;

        for attempt in 0..MAX_POLLS {
            tokio::time::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS)).await;

            let url = self.blob_url(dest_key);
            let ms_date = Self::ms_date();
            let canon_resource = self.blob_canonicalized_resource(dest_key, None);
            let canonicalized_headers = format!("x-ms-date:{ms_date}\nx-ms-version:2021-06-08\n");
            let auth = self.build_auth_header(
                "HEAD",
                None,
                &ms_date,
                &canonicalized_headers,
                &canon_resource,
            )?;

            let response = self
                .client
                .head(&url)
                .header("Authorization", auth)
                .header("x-ms-version", "2021-06-08")
                .header("x-ms-date", ms_date)
                .send()
                .await?;

            if !response.status().is_success() {
                return Err(CloudError::Storage(format!(
                    "Failed to poll copy status for '{dest_key}'"
                )));
            }

            let copy_status = response
                .headers()
                .get("x-ms-copy-status")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("unknown");

            match copy_status {
                "success" => return Ok(()),
                "failed" => {
                    let description = response
                        .headers()
                        .get("x-ms-copy-status-description")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("unknown error");
                    return Err(CloudError::Storage(format!(
                        "Azure server-side copy failed: {description}"
                    )));
                }
                "aborted" => {
                    return Err(CloudError::Storage(
                        "Azure server-side copy was aborted".to_string(),
                    ));
                }
                _ => {
                    tracing::debug!(
                        "Copy pending (attempt {}/{}): status={copy_status}",
                        attempt + 1,
                        MAX_POLLS
                    );
                }
            }
        }

        Err(CloudError::Timeout(format!(
            "Azure server-side copy of '{dest_key}' did not complete within the timeout period"
        )))
    }
}

#[async_trait]
impl CloudStorage for AzureBlobStorage {
    async fn upload(&self, key: &str, data: Bytes) -> Result<()> {
        let url = self.blob_url(key);
        let ms_date = Self::ms_date();
        let content_len = data.len() as u64;
        let canon_resource = self.blob_canonicalized_resource(key, None);
        let canonicalized_headers =
            format!("x-ms-blob-type:BlockBlob\nx-ms-date:{ms_date}\nx-ms-version:2021-06-08\n");
        let auth = self.build_auth_header(
            "PUT",
            Some("application/octet-stream"),
            &ms_date,
            &canonicalized_headers,
            &canon_resource,
        )?;

        let response = self
            .client
            .put(&url)
            .header("x-ms-blob-type", "BlockBlob")
            .header("x-ms-version", "2021-06-08")
            .header("x-ms-date", &ms_date)
            .header("Authorization", auth)
            .header("Content-Type", "application/octet-stream")
            .header("Content-Length", content_len)
            .body(data)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "Failed to upload blob (HTTP {status}): {body}"
            )));
        }

        Ok(())
    }

    async fn upload_with_options(
        &self,
        key: &str,
        data: Bytes,
        options: UploadOptions,
    ) -> Result<()> {
        let url = self.blob_url(key);
        let ms_date = Self::ms_date();
        let content_len = data.len() as u64;
        let content_type = options
            .content_type
            .as_deref()
            .unwrap_or("application/octet-stream");

        let canon_resource = self.blob_canonicalized_resource(key, None);

        // Build canonicalized headers (sorted: x-ms-blob-type, x-ms-date, x-ms-version)
        let mut extra_headers = vec![("x-ms-blob-type", "BlockBlob".to_string())];
        for (k, v) in &options.metadata {
            extra_headers.push((
                Box::leak(format!("x-ms-meta-{k}").into_boxed_str()),
                v.clone(),
            ));
        }
        extra_headers.sort_by_key(|(k, _)| *k);

        let mut canonicalized_headers = String::new();
        for (k, v) in &extra_headers {
            canonicalized_headers.push_str(&format!("{k}:{v}\n"));
        }
        canonicalized_headers.push_str(&format!("x-ms-date:{ms_date}\n"));
        canonicalized_headers.push_str("x-ms-version:2021-06-08\n");

        let auth = self.build_auth_header(
            "PUT",
            Some(content_type),
            &ms_date,
            &canonicalized_headers,
            &canon_resource,
        )?;

        let mut request = self
            .client
            .put(&url)
            .header("x-ms-blob-type", "BlockBlob")
            .header("x-ms-version", "2021-06-08")
            .header("x-ms-date", &ms_date)
            .header("Authorization", auth)
            .header("Content-Type", content_type)
            .header("Content-Length", content_len);

        if let Some(cache_control) = options.cache_control {
            request = request.header("x-ms-blob-cache-control", cache_control);
        }

        if let Some(content_encoding) = options.content_encoding {
            request = request.header("x-ms-blob-content-encoding", content_encoding);
        }

        if let Some(content_disposition) = options.content_disposition {
            request = request.header("x-ms-blob-content-disposition", content_disposition);
        }

        // Add user metadata as x-ms-meta-* request headers
        for (k, v) in &options.metadata {
            request = request.header(format!("x-ms-meta-{k}"), v);
        }

        let response = request.body(data).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "Failed to upload blob with options (HTTP {status}): {body}"
            )));
        }

        Ok(())
    }

    async fn download(&self, key: &str) -> Result<Bytes> {
        let url = self.blob_url(key);
        let ms_date = Self::ms_date();
        let canon_resource = self.blob_canonicalized_resource(key, None);
        let canonicalized_headers = format!("x-ms-date:{ms_date}\nx-ms-version:2021-06-08\n");
        let auth = self.build_auth_header(
            "GET",
            None,
            &ms_date,
            &canonicalized_headers,
            &canon_resource,
        )?;

        let response = self
            .client
            .get(&url)
            .header("Authorization", auth)
            .header("x-ms-version", "2021-06-08")
            .header("x-ms-date", ms_date)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(CloudError::Storage(format!(
                "Failed to download blob (HTTP {status})"
            )));
        }

        let data = response.bytes().await?;
        Ok(data)
    }

    async fn download_range(&self, key: &str, start: u64, end: u64) -> Result<Bytes> {
        let url = self.blob_url(key);
        let ms_date = Self::ms_date();
        let range = format!("bytes={start}-{end}");
        let canon_resource = self.blob_canonicalized_resource(key, None);
        let canonicalized_headers =
            format!("x-ms-date:{ms_date}\nx-ms-range:{range}\nx-ms-version:2021-06-08\n");
        let auth = self.build_auth_header(
            "GET",
            None,
            &ms_date,
            &canonicalized_headers,
            &canon_resource,
        )?;

        let response = self
            .client
            .get(&url)
            .header("Authorization", auth)
            .header("x-ms-version", "2021-06-08")
            .header("x-ms-date", ms_date)
            .header("x-ms-range", range)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(CloudError::Storage(format!(
                "Failed to download blob range (HTTP {status})"
            )));
        }

        let data = response.bytes().await?;
        Ok(data)
    }

    /// List all blobs in the container with the given prefix.
    ///
    /// Issues repeated `GET ?restype=container&comp=list&prefix=...` calls to the
    /// Azure Blob Service REST API, following `NextMarker` tokens until all pages are
    /// exhausted, and returns the aggregated result.
    async fn list(&self, prefix: &str) -> Result<Vec<ObjectInfo>> {
        let mut all_objects = Vec::new();
        let mut marker: Option<String> = None;

        loop {
            let result = self.list_paginated(prefix, marker.clone(), 5000).await?;

            all_objects.extend(result.objects);

            if result.is_truncated {
                marker = result.continuation_token;
            } else {
                break;
            }
        }

        Ok(all_objects)
    }

    /// List blobs with pagination using an Azure continuation marker.
    ///
    /// Issues a single `GET ?restype=container&comp=list&prefix=...&maxresults=...&marker=...`
    /// request, parses the XML response, and returns the blobs along with the next marker
    /// (if the result set was truncated).
    async fn list_paginated(
        &self,
        prefix: &str,
        continuation_token: Option<String>,
        max_keys: usize,
    ) -> Result<ListResult> {
        let ms_date = Self::ms_date();

        // Collect query parameters (will be sorted for the canonicalized resource)
        let max_results_str = max_keys.to_string();
        let mut query_params: Vec<(&str, &str)> = vec![
            ("comp", "list"),
            ("maxresults", &max_results_str),
            ("restype", "container"),
        ];

        let prefix_str;
        if !prefix.is_empty() {
            prefix_str = prefix.to_string();
            query_params.push(("prefix", &prefix_str));
        } else {
            prefix_str = String::new();
        }

        let marker_str;
        if let Some(ref token) = continuation_token {
            marker_str = token.clone();
            query_params.push(("marker", &marker_str));
        } else {
            marker_str = String::new();
        }

        // Prevent unused variable warnings when the branch is not taken
        let _ = &prefix_str;
        let _ = &marker_str;

        // Build the query string for the URL (not sorted for readability)
        let query_string = query_params
            .iter()
            .map(|(k, v)| format!("{k}={}", urlencoding::encode(v)))
            .collect::<Vec<_>>()
            .join("&");

        let url = format!("{}?{}", self.container_url(), query_string);

        // Build canonicalized resource: /accountname/containername\nparam:value\n...
        // Parameters must be sorted by name and use ":" as separator
        let mut sorted_params = query_params.clone();
        sorted_params.sort_by_key(|(k, _)| *k);

        let mut canon_resource = format!("/{}/{}", self.account_name, self.container_name);
        for (k, v) in &sorted_params {
            canon_resource.push_str(&format!("\n{k}:{v}"));
        }

        let canonicalized_headers = format!("x-ms-date:{ms_date}\nx-ms-version:2021-06-08\n");

        let string_to_sign = format!("GET\n\n\n\n{canonicalized_headers}{canon_resource}");
        let signature = self.sign_string(&string_to_sign)?;
        let auth = format!("SharedKey {}:{}", self.account_name, signature);

        let response = self
            .client
            .get(&url)
            .header("Authorization", auth)
            .header("x-ms-version", "2021-06-08")
            .header("x-ms-date", ms_date)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "Failed to list blobs (HTTP {status}): {body}"
            )));
        }

        let xml = response.text().await?;
        let (objects, next_marker) = AzureBlobStorage::parse_list_blobs_xml(&xml);
        let is_truncated = next_marker.is_some();

        Ok(ListResult {
            objects,
            continuation_token: next_marker,
            is_truncated,
            common_prefixes: Vec::new(),
        })
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let url = self.blob_url(key);
        let ms_date = Self::ms_date();
        let canon_resource = self.blob_canonicalized_resource(key, None);
        let canonicalized_headers = format!("x-ms-date:{ms_date}\nx-ms-version:2021-06-08\n");
        let auth = self.build_auth_header(
            "DELETE",
            None,
            &ms_date,
            &canonicalized_headers,
            &canon_resource,
        )?;

        let response = self
            .client
            .delete(&url)
            .header("Authorization", auth)
            .header("x-ms-version", "2021-06-08")
            .header("x-ms-date", ms_date)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() && status.as_u16() != 404 {
            return Err(CloudError::Storage(format!(
                "Failed to delete blob (HTTP {status})"
            )));
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
        let url = self.blob_url(key);
        let ms_date = Self::ms_date();
        let canon_resource = self.blob_canonicalized_resource(key, None);
        let canonicalized_headers = format!("x-ms-date:{ms_date}\nx-ms-version:2021-06-08\n");
        let auth = self.build_auth_header(
            "HEAD",
            None,
            &ms_date,
            &canonicalized_headers,
            &canon_resource,
        )?;

        let response = self
            .client
            .head(&url)
            .header("Authorization", auth)
            .header("x-ms-version", "2021-06-08")
            .header("x-ms-date", ms_date)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(CloudError::Storage(format!(
                "Failed to get blob metadata (HTTP {status})"
            )));
        }

        let headers = response.headers();
        let size = headers
            .get("Content-Length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let last_modified = headers
            .get("Last-Modified")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| chrono::DateTime::parse_from_rfc2822(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        let etag = headers
            .get("ETag")
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);

        let content_type = headers
            .get("Content-Type")
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);

        let access_tier = headers
            .get("x-ms-access-tier")
            .and_then(|v| v.to_str().ok());
        let storage_class = access_tier.map(Self::from_azure_tier);

        // Collect user-defined metadata from x-ms-meta-* response headers
        let mut user_metadata = HashMap::new();
        for (name, value) in headers.iter() {
            if let Some(meta_key) = name.as_str().strip_prefix("x-ms-meta-") {
                if let Ok(v) = value.to_str() {
                    user_metadata.insert(meta_key.to_string(), v.to_string());
                }
            }
        }

        let mut system_metadata = HashMap::new();
        if let Some(blob_type) = headers.get("x-ms-blob-type").and_then(|v| v.to_str().ok()) {
            system_metadata.insert("x-ms-blob-type".to_string(), blob_type.to_string());
        }
        if let Some(server_encrypted) = headers
            .get("x-ms-server-encrypted")
            .and_then(|v| v.to_str().ok())
        {
            system_metadata.insert(
                "x-ms-server-encrypted".to_string(),
                server_encrypted.to_string(),
            );
        }

        let info = ObjectInfo {
            key: key.to_string(),
            size,
            last_modified,
            etag,
            storage_class,
            content_type,
        };

        Ok(ObjectMetadata {
            info,
            user_metadata,
            system_metadata,
            tags: HashMap::new(),
            content_encoding: headers
                .get("Content-Encoding")
                .and_then(|v| v.to_str().ok())
                .map(ToString::to_string),
            content_language: headers
                .get("Content-Language")
                .and_then(|v| v.to_str().ok())
                .map(ToString::to_string),
            cache_control: headers
                .get("Cache-Control")
                .and_then(|v| v.to_str().ok())
                .map(ToString::to_string),
            content_disposition: headers
                .get("Content-Disposition")
                .and_then(|v| v.to_str().ok())
                .map(ToString::to_string),
        })
    }

    /// Update blob metadata using the Azure REST API `Set Blob Metadata` operation.
    ///
    /// Issues `PUT <blob>?comp=metadata` with `x-ms-meta-<key>: <value>` request headers.
    /// This operation **replaces** all existing user-defined metadata on the blob.
    ///
    /// Reference: <https://docs.microsoft.com/en-us/rest/api/storageservices/set-blob-metadata>
    async fn update_metadata(&self, key: &str, metadata: HashMap<String, String>) -> Result<()> {
        let url = format!("{}?comp=metadata", self.blob_url(key));
        let ms_date = Self::ms_date();

        // Canonicalized resource for Set Blob Metadata includes the comp parameter
        let canon_resource = self.blob_canonicalized_resource(key, Some("comp:metadata"));

        // Build x-ms-meta-* headers (sorted alphabetically for canonicalization)
        let mut meta_headers: Vec<(String, String)> = metadata
            .iter()
            .map(|(k, v)| (format!("x-ms-meta-{}", k.to_lowercase()), v.clone()))
            .collect();
        meta_headers.sort_by_key(|(k, _)| k.clone());

        let mut canonicalized_headers = format!("x-ms-date:{ms_date}\n");
        for (k, v) in &meta_headers {
            canonicalized_headers.push_str(&format!("{k}:{v}\n"));
        }
        canonicalized_headers.push_str("x-ms-version:2021-06-08\n");

        let auth = self.build_auth_header(
            "PUT",
            None,
            &ms_date,
            &canonicalized_headers,
            &canon_resource,
        )?;

        let mut request = self
            .client
            .put(&url)
            .header("Authorization", auth)
            .header("x-ms-version", "2021-06-08")
            .header("x-ms-date", &ms_date)
            .header("Content-Length", "0");

        for (k, v) in &meta_headers {
            request = request.header(k, v);
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "Failed to update blob metadata (HTTP {status}): {body}"
            )));
        }

        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let url = self.blob_url(key);
        let ms_date = Self::ms_date();
        let canon_resource = self.blob_canonicalized_resource(key, None);
        let canonicalized_headers = format!("x-ms-date:{ms_date}\nx-ms-version:2021-06-08\n");
        let auth = self.build_auth_header(
            "HEAD",
            None,
            &ms_date,
            &canonicalized_headers,
            &canon_resource,
        )?;

        let response = self
            .client
            .head(&url)
            .header("Authorization", auth)
            .header("x-ms-version", "2021-06-08")
            .header("x-ms-date", ms_date)
            .send()
            .await?;

        Ok(response.status().is_success())
    }

    /// Server-side copy a blob within the same storage account using the Azure
    /// REST API `Copy Blob` operation.
    ///
    /// Issues `PUT <dest>` with `x-ms-copy-source` pointing to the source blob URL.
    /// Azure executes the copy server-side; for small blobs this completes synchronously
    /// (HTTP 201), while large blobs return HTTP 202 and are polled until complete.
    ///
    /// Reference: <https://docs.microsoft.com/en-us/rest/api/storageservices/copy-blob>
    async fn copy(&self, source_key: &str, dest_key: &str) -> Result<()> {
        let dest_url = self.blob_url(dest_key);
        let source_url = self.blob_url(source_key);
        let ms_date = Self::ms_date();

        let canon_resource = self.blob_canonicalized_resource(dest_key, None);

        // x-ms-copy-source must appear in the canonicalized headers (sorted)
        let canonicalized_headers = format!(
            "x-ms-copy-source:{source_url}\nx-ms-date:{ms_date}\nx-ms-version:2021-06-08\n"
        );

        let auth = self.build_auth_header(
            "PUT",
            None,
            &ms_date,
            &canonicalized_headers,
            &canon_resource,
        )?;

        let response = self
            .client
            .put(&dest_url)
            .header("Authorization", auth)
            .header("x-ms-version", "2021-06-08")
            .header("x-ms-date", &ms_date)
            .header("x-ms-copy-source", &source_url)
            .header("Content-Length", "0")
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "Failed to copy blob (HTTP {status}): {body}"
            )));
        }

        // HTTP 202 means the copy is asynchronous; poll until it finishes
        if status.as_u16() == 202 {
            self.wait_for_copy_completion(dest_key).await?;
        }

        Ok(())
    }

    async fn presigned_download_url(&self, key: &str, expires_in_secs: u64) -> Result<String> {
        // Full SAS signing would use Azure Shared Access Signature V2 with HMAC-SHA256.
        // The placeholder URL format below is structurally correct but uses a stub signature.
        let expiry = Utc::now() + chrono::Duration::seconds(expires_in_secs as i64);
        let expiry_str = expiry.format("%Y-%m-%dT%H:%M:%SZ").to_string();
        let url = self.blob_url(key);
        Ok(format!(
            "{url}?sv=2021-06-08&se={expiry_str}&sp=r&spr=https&sig=PLACEHOLDER"
        ))
    }

    async fn presigned_upload_url(&self, key: &str, expires_in_secs: u64) -> Result<String> {
        let expiry = Utc::now() + chrono::Duration::seconds(expires_in_secs as i64);
        let expiry_str = expiry.format("%Y-%m-%dT%H:%M:%SZ").to_string();
        let url = self.blob_url(key);
        Ok(format!(
            "{url}?sv=2021-06-08&se={expiry_str}&sp=w&spr=https&sig=PLACEHOLDER"
        ))
    }

    /// Set the access tier (storage class) of a blob using the `Set Blob Tier` operation.
    ///
    /// Issues `PUT <blob>?comp=tier` with `x-ms-access-tier` set to `Hot`, `Cool`, or `Archive`.
    ///
    /// Reference: <https://docs.microsoft.com/en-us/rest/api/storageservices/set-blob-tier>
    async fn set_storage_class(&self, key: &str, class: StorageClass) -> Result<()> {
        let tier = Self::to_azure_tier(class);
        let tier_name = match tier {
            BlobTier::Hot => "Hot",
            BlobTier::Cool => "Cool",
            BlobTier::Archive => "Archive",
        };

        let url = format!("{}?comp=tier", self.blob_url(key));
        let ms_date = Self::ms_date();

        let canon_resource = self.blob_canonicalized_resource(key, Some("comp:tier"));
        let canonicalized_headers =
            format!("x-ms-access-tier:{tier_name}\nx-ms-date:{ms_date}\nx-ms-version:2021-06-08\n");

        let auth = self.build_auth_header(
            "PUT",
            None,
            &ms_date,
            &canonicalized_headers,
            &canon_resource,
        )?;

        let response = self
            .client
            .put(&url)
            .header("Authorization", auth)
            .header("x-ms-version", "2021-06-08")
            .header("x-ms-date", ms_date)
            .header("x-ms-access-tier", tier_name)
            .header("Content-Length", "0")
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(CloudError::Storage(format!(
                "Failed to set blob tier (HTTP {status})"
            )));
        }

        Ok(())
    }

    async fn get_stats(&self, prefix: &str) -> Result<StorageStats> {
        let objects = self.list(prefix).await?;

        let mut stats = StorageStats::default();

        for obj in objects {
            stats.total_size += obj.size;
            stats.object_count += 1;

            if let Some(class) = obj.storage_class {
                let class_name = format!("{class}");
                *stats.size_by_class.entry(class_name.clone()).or_insert(0) += obj.size;
                *stats.count_by_class.entry(class_name).or_insert(0) += 1;
            }
        }

        Ok(stats)
    }
}

/// Extract the inner text of the first `<tag>...</tag>` occurrence in `xml`.
fn extract_xml_inner(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)?;
    let content_start = start + open.len();
    let end = xml[content_start..].find(&close)?;
    Some(xml[content_start..content_start + end].to_string())
}

/// Azure Blob access tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobTier {
    /// Hot tier – frequently accessed data.
    Hot,
    /// Cool tier – infrequently accessed data, stored for at least 30 days.
    Cool,
    /// Archive tier – rarely accessed data, stored for at least 180 days.
    Archive,
}

/// Azure Shared Access Signature (SAS) permissions.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SasPermissions {
    /// Read permission (`r`)
    Read,
    /// Write permission (`w`)
    Write,
    /// Delete permission (`d`)
    Delete,
    /// List permission (`l`)
    List,
}

#[allow(dead_code)]
impl SasPermissions {
    /// Return the single-character SAS permission string.
    #[must_use]
    pub fn to_string(self) -> &'static str {
        match self {
            SasPermissions::Read => "r",
            SasPermissions::Write => "w",
            SasPermissions::Delete => "d",
            SasPermissions::List => "l",
        }
    }
}

/// Strongly-typed view of the Azure List Blobs XML response (used internally for testing).
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ListBlobsResponse {
    #[serde(rename = "Blobs")]
    blobs: Option<BlobsWrapper>,
    #[serde(rename = "NextMarker")]
    next_marker: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct BlobsWrapper {
    #[serde(rename = "Blob", default)]
    blob: Vec<BlobItem>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct BlobItem {
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Properties")]
    properties: BlobProperties,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct BlobProperties {
    #[serde(rename = "Content-Length", default)]
    content_length: u64,
    #[serde(rename = "Last-Modified")]
    last_modified: Option<String>,
    #[serde(rename = "Etag")]
    etag: Option<String>,
    #[serde(rename = "Content-Type")]
    content_type: Option<String>,
    #[serde(rename = "AccessTier")]
    access_tier: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_tier_conversion_standard() {
        let tier = StorageClass::Standard;
        let azure_tier = AzureBlobStorage::to_azure_tier(tier);
        assert_eq!(azure_tier, BlobTier::Hot);

        let class = AzureBlobStorage::from_azure_tier("Hot");
        assert_eq!(class, StorageClass::Standard);
    }

    #[test]
    fn test_blob_tier_conversion_cool() {
        let tier = AzureBlobStorage::to_azure_tier(StorageClass::InfrequentAccess);
        assert_eq!(tier, BlobTier::Cool);

        let class = AzureBlobStorage::from_azure_tier("Cool");
        assert_eq!(class, StorageClass::InfrequentAccess);
    }

    #[test]
    fn test_blob_tier_conversion_archive() {
        let tier = AzureBlobStorage::to_azure_tier(StorageClass::Glacier);
        assert_eq!(tier, BlobTier::Archive);

        let tier_deep = AzureBlobStorage::to_azure_tier(StorageClass::DeepArchive);
        assert_eq!(tier_deep, BlobTier::Archive);

        let class = AzureBlobStorage::from_azure_tier("Archive");
        assert_eq!(class, StorageClass::Glacier);
    }

    #[test]
    fn test_sas_permissions() {
        assert_eq!(SasPermissions::Read.to_string(), "r");
        assert_eq!(SasPermissions::Write.to_string(), "w");
        assert_eq!(SasPermissions::Delete.to_string(), "d");
        assert_eq!(SasPermissions::List.to_string(), "l");
    }

    #[test]
    fn test_parse_list_blobs_xml_empty() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<EnumerationResults ServiceEndpoint="https://myaccount.blob.core.windows.net/" ContainerName="mycontainer">
  <Blobs/>
  <NextMarker/>
</EnumerationResults>"#;

        let (objects, next_marker) = AzureBlobStorage::parse_list_blobs_xml(xml);
        assert!(objects.is_empty());
        assert!(next_marker.is_none());
    }

    #[test]
    fn test_parse_list_blobs_xml_with_blobs() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<EnumerationResults ServiceEndpoint="https://myaccount.blob.core.windows.net/" ContainerName="mycontainer">
  <Blobs>
    <Blob>
      <Name>video/2024/sample.mp4</Name>
      <Properties>
        <Last-Modified>Mon, 04 Nov 2024 10:00:00 GMT</Last-Modified>
        <Etag>0x8DC3EF1234ABCDEF</Etag>
        <Content-Length>104857600</Content-Length>
        <Content-Type>video/mp4</Content-Type>
        <AccessTier>Hot</AccessTier>
      </Properties>
    </Blob>
    <Blob>
      <Name>video/2024/sample2.mp4</Name>
      <Properties>
        <Last-Modified>Tue, 05 Nov 2024 12:30:00 GMT</Last-Modified>
        <Etag>0x8DC3EF5678FEDCBA</Etag>
        <Content-Length>209715200</Content-Length>
        <Content-Type>video/mp4</Content-Type>
        <AccessTier>Cool</AccessTier>
      </Properties>
    </Blob>
  </Blobs>
  <NextMarker/>
</EnumerationResults>"#;

        let (objects, next_marker) = AzureBlobStorage::parse_list_blobs_xml(xml);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].key, "video/2024/sample.mp4");
        assert_eq!(objects[0].size, 104_857_600);
        assert_eq!(objects[0].storage_class, Some(StorageClass::Standard));
        assert_eq!(objects[1].key, "video/2024/sample2.mp4");
        assert_eq!(objects[1].size, 209_715_200);
        assert_eq!(
            objects[1].storage_class,
            Some(StorageClass::InfrequentAccess)
        );
        assert!(next_marker.is_none());
    }

    #[test]
    fn test_parse_list_blobs_xml_with_next_marker() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<EnumerationResults>
  <Blobs>
    <Blob>
      <Name>file1.mp4</Name>
      <Properties>
        <Content-Length>1024</Content-Length>
        <Last-Modified>Mon, 04 Nov 2024 10:00:00 GMT</Last-Modified>
        <Etag>abc123</Etag>
        <Content-Type>video/mp4</Content-Type>
      </Properties>
    </Blob>
  </Blobs>
  <NextMarker>continuation-token-xyz</NextMarker>
</EnumerationResults>"#;

        let (objects, next_marker) = AzureBlobStorage::parse_list_blobs_xml(xml);
        assert_eq!(objects.len(), 1);
        assert_eq!(next_marker, Some("continuation-token-xyz".to_string()));
    }

    #[test]
    fn test_extract_xml_inner_basic() {
        assert_eq!(
            extract_xml_inner("<Name>hello</Name>", "Name"),
            Some("hello".to_string())
        );
        assert_eq!(
            extract_xml_inner("<Content-Length>1024</Content-Length>", "Content-Length"),
            Some("1024".to_string())
        );
    }

    #[test]
    fn test_extract_xml_inner_no_match() {
        assert_eq!(extract_xml_inner("<Other>x</Other>", "Name"), None);
    }

    #[test]
    fn test_extract_xml_inner_empty_tag() {
        // Self-closing tag returns None (no inner text)
        assert_eq!(extract_xml_inner("<NextMarker/>", "NextMarker"), None);
    }
}
