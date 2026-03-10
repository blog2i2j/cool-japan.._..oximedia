//! Amazon S3 storage implementation

use crate::{
    ByteStream, CloudStorage, DownloadOptions, ListOptions, ListResult, ObjectMetadata, Result,
    StorageError, UnifiedConfig, UploadOptions,
};
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_s3::{
    config::{Credentials, Region},
    operation::create_multipart_upload::CreateMultipartUploadOutput,
    primitives::{ByteStream as AwsByteStream, DateTime as AwsDateTime},
    types::{
        BucketLifecycleConfiguration, BucketVersioningStatus, CompletedMultipartUpload,
        CompletedPart, Delete, LifecycleRule, LifecycleRuleFilter, ObjectIdentifier,
        ServerSideEncryption, StorageClass, Transition, TransitionStorageClass,
        VersioningConfiguration,
    },
    Client,
};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::{stream, StreamExt};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, info};

/// Minimum size for multipart upload chunks (5 MB)
const MULTIPART_CHUNK_SIZE: u64 = 5 * 1024 * 1024;

/// Threshold for using multipart upload (10 MB)
const MULTIPART_THRESHOLD: u64 = 10 * 1024 * 1024;

/// Maximum number of parts in multipart upload
const MAX_MULTIPART_PARTS: usize = 10000;

/// S3 storage client
pub struct S3Storage {
    client: Client,
    bucket: String,
    config: UnifiedConfig,
}

impl S3Storage {
    /// Create a new S3 storage client from configuration
    pub async fn new(config: UnifiedConfig) -> Result<Self> {
        let aws_config = if let (Some(access_key), Some(secret_key)) =
            (&config.access_key, &config.secret_key)
        {
            let creds = Credentials::new(access_key, secret_key, None, None, "oximedia-storage");

            let region = config
                .region
                .clone()
                .ok_or_else(|| StorageError::InvalidConfig("Region required for S3".into()))?;

            aws_config::defaults(BehaviorVersion::latest())
                .credentials_provider(creds)
                .region(Region::new(region))
                .load()
                .await
        } else {
            let region = config
                .region
                .clone()
                .ok_or_else(|| StorageError::InvalidConfig("Region required for S3".into()))?;

            aws_config::defaults(BehaviorVersion::latest())
                .region(Region::new(region))
                .load()
                .await
        };

        let mut s3_config = aws_sdk_s3::config::Builder::from(&aws_config);

        if let Some(endpoint) = &config.endpoint {
            s3_config = s3_config.endpoint_url(endpoint);
        }

        if config.path_style {
            s3_config = s3_config.force_path_style(true);
        }

        let client = Client::from_conf(s3_config.build());

        Ok(Self {
            client,
            bucket: config.bucket.clone(),
            config,
        })
    }

    /// Perform multipart upload
    #[allow(clippy::too_many_arguments)]
    async fn multipart_upload(
        &self,
        key: &str,
        stream: ByteStream,
        total_size: u64,
        options: &UploadOptions,
    ) -> Result<String> {
        info!("Starting multipart upload for key: {}", key);

        // Initiate multipart upload
        let mut create_req = self
            .client
            .create_multipart_upload()
            .bucket(&self.bucket)
            .key(key);

        if let Some(content_type) = &options.content_type {
            create_req = create_req.content_type(content_type);
        }

        if let Some(storage_class) = &options.storage_class {
            if let Some(sc) = parse_storage_class(storage_class) {
                create_req = create_req.storage_class(sc);
            }
        }

        if let Some(cache_control) = &options.cache_control {
            create_req = create_req.cache_control(cache_control);
        }

        if let Some(acl) = &options.acl {
            create_req = create_req.acl(
                acl.parse()
                    .map_err(|_| StorageError::InvalidConfig(format!("Invalid ACL: {acl}")))?,
            );
        }

        // Set server-side encryption
        if let Some(encryption) = &options.encryption {
            match encryption {
                crate::EncryptionConfig::ServerSide => {
                    create_req = create_req.server_side_encryption(ServerSideEncryption::Aes256);
                }
                crate::EncryptionConfig::CustomerKey(key_id) => {
                    create_req = create_req.ssekms_key_id(key_id);
                    create_req = create_req.server_side_encryption(ServerSideEncryption::AwsKms);
                }
                _ => {}
            }
        }

        for (k, v) in &options.metadata {
            create_req = create_req.metadata(k, v);
        }

        let create_output: CreateMultipartUploadOutput = create_req.send().await.map_err(|e| {
            StorageError::MultipartError(format!("Failed to initiate multipart upload: {e}"))
        })?;

        let upload_id = create_output
            .upload_id()
            .ok_or_else(|| StorageError::MultipartError("No upload ID returned".into()))?;

        debug!("Multipart upload initiated with ID: {}", upload_id);

        // Calculate chunk size
        let chunk_size = calculate_chunk_size(total_size);
        let mut part_number: i32 = 1;
        let mut completed_parts = Vec::new();

        // Stream upload parts
        let mut stream = stream;
        let mut buffer = Vec::new();

        while let Some(result) = stream.next().await {
            let chunk = result?;
            buffer.extend_from_slice(&chunk);

            while buffer.len() >= chunk_size as usize || part_number == MAX_MULTIPART_PARTS as i32 {
                let part_data = if buffer.len() > chunk_size as usize {
                    buffer.drain(..chunk_size as usize).collect::<Vec<_>>()
                } else {
                    std::mem::take(&mut buffer)
                };

                let part = self
                    .upload_part(key, upload_id, part_number, part_data)
                    .await?;

                completed_parts.push(part);
                part_number += 1;

                if part_number > MAX_MULTIPART_PARTS as i32 {
                    return Err(StorageError::MultipartError(
                        "Exceeded maximum number of parts".into(),
                    ));
                }
            }
        }

        // Upload remaining data
        if !buffer.is_empty() {
            let part = self
                .upload_part(key, upload_id, part_number, buffer)
                .await?;
            completed_parts.push(part);
        }

        // Complete multipart upload
        let completed_upload = CompletedMultipartUpload::builder()
            .set_parts(Some(completed_parts))
            .build();

        let complete_output = self
            .client
            .complete_multipart_upload()
            .bucket(&self.bucket)
            .key(key)
            .upload_id(upload_id)
            .multipart_upload(completed_upload)
            .send()
            .await
            .map_err(|e| {
                StorageError::MultipartError(format!("Failed to complete multipart upload: {e}"))
            })?;

        let etag = complete_output.e_tag().unwrap_or("unknown").to_string(); // S3 SDK returns None only for non-S3 backends

        info!("Multipart upload completed for key: {}", key);
        Ok(etag)
    }

    /// Upload a single part
    async fn upload_part(
        &self,
        key: &str,
        upload_id: &str,
        part_number: i32,
        data: Vec<u8>,
    ) -> Result<CompletedPart> {
        debug!("Uploading part {} for key: {}", part_number, key);

        let body = AwsByteStream::from(data);

        let upload_output = self
            .client
            .upload_part()
            .bucket(&self.bucket)
            .key(key)
            .upload_id(upload_id)
            .part_number(part_number)
            .body(body)
            .send()
            .await
            .map_err(|e| {
                StorageError::MultipartError(format!("Failed to upload part {part_number}: {e}"))
            })?;

        let etag = upload_output
            .e_tag()
            .ok_or_else(|| StorageError::MultipartError("No ETag in upload part response".into()))?
            .to_string();

        Ok(CompletedPart::builder()
            .part_number(part_number)
            .e_tag(etag)
            .build())
    }

    /// Enable versioning on the bucket
    pub async fn enable_versioning(&self) -> Result<()> {
        info!("Enabling versioning for bucket: {}", self.bucket);

        let versioning_config = VersioningConfiguration::builder()
            .status(BucketVersioningStatus::Enabled)
            .build();

        self.client
            .put_bucket_versioning()
            .bucket(&self.bucket)
            .versioning_configuration(versioning_config)
            .send()
            .await
            .map_err(|e| {
                StorageError::ProviderError(format!("Failed to enable versioning: {e}"))
            })?;

        info!("Versioning enabled for bucket: {}", self.bucket);
        Ok(())
    }

    /// Disable versioning on the bucket
    pub async fn disable_versioning(&self) -> Result<()> {
        info!("Disabling versioning for bucket: {}", self.bucket);

        let versioning_config = VersioningConfiguration::builder()
            .status(BucketVersioningStatus::Suspended)
            .build();

        self.client
            .put_bucket_versioning()
            .bucket(&self.bucket)
            .versioning_configuration(versioning_config)
            .send()
            .await
            .map_err(|e| {
                StorageError::ProviderError(format!("Failed to disable versioning: {e}"))
            })?;

        info!("Versioning disabled for bucket: {}", self.bucket);
        Ok(())
    }

    /// List object versions
    pub async fn list_object_versions(&self, key: &str) -> Result<Vec<ObjectVersion>> {
        debug!("Listing versions for key: {}", key);

        let output = self
            .client
            .list_object_versions()
            .bucket(&self.bucket)
            .prefix(key)
            .send()
            .await
            .map_err(|e| {
                StorageError::ProviderError(format!("Failed to list object versions: {e}"))
            })?;

        let mut versions = Vec::new();

        for version in output.versions() {
            if let Some(k) = version.key() {
                if k == key {
                    versions.push(ObjectVersion {
                        version_id: version.version_id().unwrap_or("null").to_string(),
                        is_latest: version.is_latest().unwrap_or(false),
                        last_modified: version.last_modified().and_then(parse_aws_datetime),
                        size: version.size().unwrap_or(0) as u64,
                        etag: version.e_tag().map(|s: &str| s.to_string()),
                    });
                }
            }
        }

        Ok(versions)
    }

    /// Delete a specific version of an object
    pub async fn delete_object_version(&self, key: &str, version_id: &str) -> Result<()> {
        debug!("Deleting version {} of key: {}", version_id, key);

        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(key)
            .version_id(version_id)
            .send()
            .await
            .map_err(|e| {
                StorageError::ProviderError(format!("Failed to delete object version: {e}"))
            })?;

        info!("Deleted version {} of key: {}", version_id, key);
        Ok(())
    }

    /// Set lifecycle policy for the bucket
    #[allow(clippy::too_many_arguments)]
    pub async fn set_lifecycle_policy(
        &self,
        rule_id: &str,
        prefix: Option<&str>,
        transition_days: Option<i32>,
        transition_storage_class: Option<&str>,
        expiration_days: Option<i32>,
    ) -> Result<()> {
        info!("Setting lifecycle policy for bucket: {}", self.bucket);

        let mut rule = LifecycleRule::builder()
            .id(rule_id)
            .status(aws_sdk_s3::types::ExpirationStatus::Enabled);

        if let Some(p) = prefix {
            rule = rule.filter(LifecycleRuleFilter::builder().prefix(p.to_string()).build());
        }

        if let Some(days) = transition_days {
            if let Some(sc) = transition_storage_class {
                if let Some(storage_class) = parse_transition_storage_class(sc) {
                    rule = rule.transitions(
                        Transition::builder()
                            .days(days)
                            .storage_class(storage_class)
                            .build(),
                    );
                }
            }
        }

        if let Some(days) = expiration_days {
            rule = rule.expiration(
                aws_sdk_s3::types::LifecycleExpiration::builder()
                    .days(days)
                    .build(),
            );
        }

        let lifecycle_config =
            BucketLifecycleConfiguration::builder()
                .rules(rule.build().map_err(|e| {
                    StorageError::InvalidConfig(format!("Invalid lifecycle rule: {e}"))
                })?)
                .build()
                .map_err(|e| {
                    StorageError::InvalidConfig(format!("Invalid lifecycle configuration: {e}"))
                })?;

        self.client
            .put_bucket_lifecycle_configuration()
            .bucket(&self.bucket)
            .lifecycle_configuration(lifecycle_config)
            .send()
            .await
            .map_err(|e| {
                StorageError::ProviderError(format!("Failed to set lifecycle policy: {e}"))
            })?;

        info!("Lifecycle policy set for bucket: {}", self.bucket);
        Ok(())
    }

    /// Enable transfer acceleration
    pub async fn enable_transfer_acceleration(&self) -> Result<()> {
        info!("Enabling transfer acceleration for bucket: {}", self.bucket);

        self.client
            .put_bucket_accelerate_configuration()
            .bucket(&self.bucket)
            .accelerate_configuration(
                aws_sdk_s3::types::AccelerateConfiguration::builder()
                    .status(aws_sdk_s3::types::BucketAccelerateStatus::Enabled)
                    .build(),
            )
            .send()
            .await
            .map_err(|e| {
                StorageError::ProviderError(format!("Failed to enable transfer acceleration: {e}"))
            })?;

        info!("Transfer acceleration enabled for bucket: {}", self.bucket);
        Ok(())
    }

    /// Create bucket
    pub async fn create_bucket(&self) -> Result<()> {
        info!("Creating bucket: {}", self.bucket);

        let mut req = self.client.create_bucket().bucket(&self.bucket);

        if let Some(region) = &self.config.region {
            if region != "us-east-1" {
                req = req.create_bucket_configuration(
                    aws_sdk_s3::types::CreateBucketConfiguration::builder()
                        .location_constraint(aws_sdk_s3::types::BucketLocationConstraint::from(
                            region.as_str(),
                        ))
                        .build(),
                );
            }
        }

        req.send()
            .await
            .map_err(|e| StorageError::ProviderError(format!("Failed to create bucket: {e}")))?;

        info!("Bucket created: {}", self.bucket);
        Ok(())
    }

    /// Delete bucket
    pub async fn delete_bucket(&self) -> Result<()> {
        info!("Deleting bucket: {}", self.bucket);

        self.client
            .delete_bucket()
            .bucket(&self.bucket)
            .send()
            .await
            .map_err(|e| StorageError::ProviderError(format!("Failed to delete bucket: {e}")))?;

        info!("Bucket deleted: {}", self.bucket);
        Ok(())
    }

    /// Check if bucket exists
    pub async fn bucket_exists(&self) -> Result<bool> {
        match self.client.head_bucket().bucket(&self.bucket).send().await {
            Ok(_) => Ok(true),
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("404") || err_str.contains("NoSuchBucket") {
                    Ok(false)
                } else {
                    Err(StorageError::ProviderError(format!(
                        "Failed to check bucket existence: {e}"
                    )))
                }
            }
        }
    }

    /// Restore object from Glacier
    pub async fn restore_object(&self, key: &str, days: i32) -> Result<()> {
        info!("Restoring object from Glacier: {}", key);

        self.client
            .restore_object()
            .bucket(&self.bucket)
            .key(key)
            .restore_request(
                aws_sdk_s3::types::RestoreRequest::builder()
                    .days(days)
                    .build(),
            )
            .send()
            .await
            .map_err(|e| StorageError::ProviderError(format!("Failed to restore object: {e}")))?;

        info!("Object restore initiated: {}", key);
        Ok(())
    }
}

#[async_trait]
impl CloudStorage for S3Storage {
    async fn upload_stream(
        &self,
        key: &str,
        stream: ByteStream,
        size: Option<u64>,
        options: UploadOptions,
    ) -> Result<String> {
        debug!("Uploading stream to key: {}", key);

        // Use multipart upload for large files or unknown sizes
        if size.map_or(true, |s| s > MULTIPART_THRESHOLD) {
            return self
                .multipart_upload(key, stream, size.unwrap_or(0), &options)
                .await;
        }

        // Simple upload for small files
        let mut chunks = Vec::new();
        let mut stream = stream;

        while let Some(result) = stream.next().await {
            let chunk = result?;
            chunks.extend_from_slice(&chunk);
        }

        let body = AwsByteStream::from(chunks);

        let mut req = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(body);

        if let Some(content_type) = &options.content_type {
            req = req.content_type(content_type);
        }

        if let Some(storage_class) = &options.storage_class {
            if let Some(sc) = parse_storage_class(storage_class) {
                req = req.storage_class(sc);
            }
        }

        if let Some(cache_control) = &options.cache_control {
            req = req.cache_control(cache_control);
        }

        for (k, v) in &options.metadata {
            req = req.metadata(k, v);
        }

        let output = req
            .send()
            .await
            .map_err(|e| StorageError::ProviderError(format!("Failed to upload object: {e}")))?;

        let etag = output.e_tag().unwrap_or("unknown").to_string();

        info!("Uploaded object to key: {}", key);
        Ok(etag)
    }

    async fn upload_file(
        &self,
        key: &str,
        file_path: &Path,
        options: UploadOptions,
    ) -> Result<String> {
        debug!("Uploading file {:?} to key: {}", file_path, key);

        let mut file = File::open(file_path).await?;
        let metadata = file.metadata().await?;
        let file_size = metadata.len();

        if file_size > MULTIPART_THRESHOLD {
            // Use multipart upload
            let stream: ByteStream = Box::pin(stream::try_unfold(file, |mut file| async move {
                let mut buffer = vec![0u8; MULTIPART_CHUNK_SIZE as usize];
                let n = file.read(&mut buffer).await?;
                if n == 0 {
                    Ok(None)
                } else {
                    buffer.truncate(n);
                    Ok(Some((Bytes::from(buffer), file)))
                }
            }));

            self.multipart_upload(key, stream, file_size, &options)
                .await
        } else {
            // Simple upload
            let mut contents = Vec::new();
            file.read_to_end(&mut contents).await?;

            let body = AwsByteStream::from(contents);

            let mut req = self
                .client
                .put_object()
                .bucket(&self.bucket)
                .key(key)
                .body(body);

            if let Some(content_type) = &options.content_type {
                req = req.content_type(content_type);
            }

            let output = req
                .send()
                .await
                .map_err(|e| StorageError::ProviderError(format!("Failed to upload file: {e}")))?;

            let etag = output.e_tag().unwrap_or("unknown").to_string();

            info!("Uploaded file to key: {}", key);
            Ok(etag)
        }
    }

    async fn download_stream(&self, key: &str, options: DownloadOptions) -> Result<ByteStream> {
        debug!("Downloading stream from key: {}", key);

        let mut req = self.client.get_object().bucket(&self.bucket).key(key);

        if let Some((start, end)) = options.range {
            req = req.range(format!("bytes={start}-{end}"));
        }

        if let Some(etag) = &options.if_match {
            req = req.if_match(etag);
        }

        if let Some(etag) = &options.if_none_match {
            req = req.if_none_match(etag);
        }

        let output = req.send().await.map_err(|e| {
            let err_str = e.to_string();
            if err_str.contains("404") || err_str.contains("NoSuchKey") {
                StorageError::NotFound(key.to_string())
            } else {
                StorageError::ProviderError(format!("Failed to download object: {e}"))
            }
        })?;

        // Collect all bytes from the ByteStream and return as a single-chunk stream
        let data = output
            .body
            .collect()
            .await
            .map_err(|e| StorageError::NetworkError(format!("Stream error: {e}")))?
            .into_bytes();

        let stream = futures::stream::once(async move { Ok::<Bytes, StorageError>(data) });
        Ok(Box::pin(stream))
    }

    async fn download_file(
        &self,
        key: &str,
        file_path: &Path,
        options: DownloadOptions,
    ) -> Result<()> {
        debug!("Downloading file from key: {} to {:?}", key, file_path);

        let mut stream = self.download_stream(key, options).await?;
        let mut file = File::create(file_path).await?;

        while let Some(result) = stream.next().await {
            let chunk = result?;
            file.write_all(&chunk).await?;
        }

        file.flush().await?;
        info!("Downloaded file from key: {} to {:?}", key, file_path);
        Ok(())
    }

    async fn get_metadata(&self, key: &str) -> Result<ObjectMetadata> {
        debug!("Getting metadata for key: {}", key);

        let output = self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| {
                let err_str = e.to_string();
                if err_str.contains("404") || err_str.contains("NoSuchKey") {
                    StorageError::NotFound(key.to_string())
                } else {
                    StorageError::ProviderError(format!("Failed to get metadata: {e}"))
                }
            })?;

        let last_modified = output
            .last_modified()
            .and_then(parse_aws_datetime)
            .unwrap_or_else(Utc::now);

        let metadata = output
            .metadata()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        Ok(ObjectMetadata {
            key: key.to_string(),
            size: output.content_length().unwrap_or(0) as u64,
            content_type: output.content_type().map(std::string::ToString::to_string),
            last_modified,
            etag: output.e_tag().map(std::string::ToString::to_string),
            metadata,
            storage_class: output.storage_class().map(|sc| sc.as_str().to_string()),
        })
    }

    async fn delete_object(&self, key: &str) -> Result<()> {
        debug!("Deleting object with key: {}", key);

        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| StorageError::ProviderError(format!("Failed to delete object: {e}")))?;

        info!("Deleted object with key: {}", key);
        Ok(())
    }

    async fn delete_objects(&self, keys: &[String]) -> Result<Vec<Result<()>>> {
        debug!("Deleting {} objects", keys.len());

        let objects: Vec<ObjectIdentifier> =
            keys.iter()
                .map(|k| {
                    ObjectIdentifier::builder().key(k).build().map_err(|e| {
                        StorageError::InvalidConfig(format!("Invalid object key: {e}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;

        let delete = Delete::builder()
            .set_objects(Some(objects))
            .build()
            .map_err(|e| StorageError::InvalidConfig(format!("Invalid delete request: {e}")))?;

        let output = self
            .client
            .delete_objects()
            .bucket(&self.bucket)
            .delete(delete)
            .send()
            .await
            .map_err(|e| StorageError::ProviderError(format!("Failed to delete objects: {e}")))?;

        let mut results: Vec<Result<()>> = (0..keys.len()).map(|_| Ok(())).collect();

        for error in output.errors() {
            if let Some(key) = error.key() {
                if let Some(pos) = keys.iter().position(|k| k == key) {
                    results[pos] = Err(StorageError::ProviderError(
                        error.message().unwrap_or("Unknown error").to_string(),
                    ));
                }
            }
        }

        info!("Deleted {} objects", keys.len());
        Ok(results)
    }

    async fn list_objects(&self, options: ListOptions) -> Result<ListResult> {
        debug!("Listing objects with prefix: {:?}", options.prefix);

        let mut req = self.client.list_objects_v2().bucket(&self.bucket);

        if let Some(prefix) = &options.prefix {
            req = req.prefix(prefix);
        }

        if let Some(delimiter) = &options.delimiter {
            req = req.delimiter(delimiter);
        }

        if let Some(max_results) = options.max_results {
            req = req.max_keys(max_results as i32);
        }

        if let Some(token) = &options.continuation_token {
            req = req.continuation_token(token);
        }

        let output = req
            .send()
            .await
            .map_err(|e| StorageError::ProviderError(format!("Failed to list objects: {e}")))?;

        let objects = output
            .contents()
            .iter()
            .filter_map(|obj| {
                let key = obj.key()?.to_string();
                let size = obj.size().unwrap_or(0) as u64;
                let last_modified = obj
                    .last_modified()
                    .and_then(parse_aws_datetime)
                    .unwrap_or_else(Utc::now);

                Some(ObjectMetadata {
                    key,
                    size,
                    content_type: None,
                    last_modified,
                    etag: obj.e_tag().map(std::string::ToString::to_string),
                    metadata: HashMap::new(),
                    storage_class: obj.storage_class().map(|sc| sc.as_str().to_string()),
                })
            })
            .collect();

        let prefixes = output
            .common_prefixes()
            .iter()
            .filter_map(|cp| cp.prefix().map(std::string::ToString::to_string))
            .collect();

        Ok(ListResult {
            objects,
            prefixes,
            next_token: output
                .next_continuation_token()
                .map(std::string::ToString::to_string),
            has_more: output.is_truncated().unwrap_or(false),
        })
    }

    async fn object_exists(&self, key: &str) -> Result<bool> {
        match self.get_metadata(key).await {
            Ok(_) => Ok(true),
            Err(StorageError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    async fn copy_object(&self, source_key: &str, dest_key: &str) -> Result<()> {
        debug!("Copying object from {} to {}", source_key, dest_key);

        let copy_source = format!("{}/{}", self.bucket, source_key);

        self.client
            .copy_object()
            .bucket(&self.bucket)
            .copy_source(&copy_source)
            .key(dest_key)
            .send()
            .await
            .map_err(|e| StorageError::ProviderError(format!("Failed to copy object: {e}")))?;

        info!("Copied object from {} to {}", source_key, dest_key);
        Ok(())
    }

    async fn generate_presigned_url(&self, key: &str, expiration_secs: u64) -> Result<String> {
        debug!("Generating presigned URL for key: {}", key);

        let expires_in = std::time::Duration::from_secs(expiration_secs);

        let presigned_request = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .presigned(
                aws_sdk_s3::presigning::PresigningConfig::expires_in(expires_in)
                    .map_err(|e| StorageError::ProviderError(format!("Invalid expiration: {e}")))?,
            )
            .await
            .map_err(|e| {
                StorageError::ProviderError(format!("Failed to generate presigned URL: {e}"))
            })?;

        Ok(presigned_request.uri().to_string())
    }

    async fn generate_presigned_upload_url(
        &self,
        key: &str,
        expiration_secs: u64,
    ) -> Result<String> {
        debug!("Generating presigned upload URL for key: {}", key);

        let expires_in = std::time::Duration::from_secs(expiration_secs);

        let presigned_request = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .presigned(
                aws_sdk_s3::presigning::PresigningConfig::expires_in(expires_in)
                    .map_err(|e| StorageError::ProviderError(format!("Invalid expiration: {e}")))?,
            )
            .await
            .map_err(|e| {
                StorageError::ProviderError(format!("Failed to generate presigned upload URL: {e}"))
            })?;

        Ok(presigned_request.uri().to_string())
    }
}

/// Object version information
#[derive(Debug, Clone)]
pub struct ObjectVersion {
    pub version_id: String,
    pub is_latest: bool,
    pub last_modified: Option<DateTime<Utc>>,
    pub size: u64,
    pub etag: Option<String>,
}

/// Parse AWS DateTime to chrono DateTime
#[allow(dead_code)]
fn parse_aws_datetime(dt: &AwsDateTime) -> Option<DateTime<Utc>> {
    let secs = dt.secs();
    let nanos = dt.subsec_nanos();
    DateTime::from_timestamp(secs, nanos)
}

/// Parse storage class string (for put_object/create_multipart_upload)
fn parse_storage_class(s: &str) -> Option<StorageClass> {
    match s.to_uppercase().as_str() {
        "STANDARD" => Some(StorageClass::Standard),
        "REDUCED_REDUNDANCY" => Some(StorageClass::ReducedRedundancy),
        "STANDARD_IA" => Some(StorageClass::StandardIa),
        "ONEZONE_IA" => Some(StorageClass::OnezoneIa),
        "INTELLIGENT_TIERING" => Some(StorageClass::IntelligentTiering),
        "GLACIER" => Some(StorageClass::Glacier),
        "DEEP_ARCHIVE" => Some(StorageClass::DeepArchive),
        "GLACIER_IR" => Some(StorageClass::GlacierIr),
        _ => None,
    }
}

/// Parse transition storage class string (for lifecycle transitions)
fn parse_transition_storage_class(s: &str) -> Option<TransitionStorageClass> {
    match s.to_uppercase().as_str() {
        "STANDARD_IA" => Some(TransitionStorageClass::StandardIa),
        "ONEZONE_IA" => Some(TransitionStorageClass::OnezoneIa),
        "INTELLIGENT_TIERING" => Some(TransitionStorageClass::IntelligentTiering),
        "GLACIER" => Some(TransitionStorageClass::Glacier),
        "DEEP_ARCHIVE" => Some(TransitionStorageClass::DeepArchive),
        "GLACIER_IR" => Some(TransitionStorageClass::GlacierIr),
        _ => None,
    }
}

/// Calculate optimal chunk size for multipart upload
fn calculate_chunk_size(total_size: u64) -> u64 {
    if total_size == 0 {
        return MULTIPART_CHUNK_SIZE;
    }

    let mut chunk_size = MULTIPART_CHUNK_SIZE;
    while total_size / chunk_size > MAX_MULTIPART_PARTS as u64 {
        chunk_size *= 2;
    }

    chunk_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_chunk_size() {
        assert_eq!(calculate_chunk_size(0), MULTIPART_CHUNK_SIZE);
        assert_eq!(
            calculate_chunk_size(100 * 1024 * 1024),
            MULTIPART_CHUNK_SIZE
        );

        let large_size = 100 * 1024 * 1024 * 1024; // 100 GB
        let chunk_size = calculate_chunk_size(large_size);
        assert!(large_size / chunk_size <= MAX_MULTIPART_PARTS as u64);
    }

    #[test]
    fn test_parse_storage_class() {
        assert!(matches!(
            parse_storage_class("STANDARD"),
            Some(StorageClass::Standard)
        ));
        assert!(matches!(
            parse_storage_class("GLACIER"),
            Some(StorageClass::Glacier)
        ));
        assert!(parse_storage_class("INVALID").is_none());
    }
}
