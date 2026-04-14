//! Storage inventory report — object count, total size, and class distribution.
//!
//! Generates a comprehensive inventory of all objects in a storage namespace,
//! aggregated by storage class, prefix, and size bucket.  Useful for capacity
//! planning, cost analysis, and compliance audits.
//!
//! # Architecture
//!
//! ```text
//! InventoryBuilder
//!   ├── add_object(key, size, class, modified)
//!   ├── build() → InventoryReport
//!   └── InventoryReport
//!        ├── summary       — total objects, total bytes, earliest/latest modified
//!        ├── by_class      — HashMap<StorageClass, ClassStats>
//!        ├── by_prefix     — HashMap<String, PrefixStats>
//!        └── size_histogram — Vec<SizeBucket>
//! ```

#![allow(dead_code)]

use chrono::{DateTime, Utc};
use std::collections::HashMap;

// ─── StorageClass ───────────────────────────────────────────────────────────

/// Logical storage class / tier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StorageClass {
    /// Standard high-availability storage.
    Standard,
    /// Infrequent access.
    InfrequentAccess,
    /// Archive / cold storage.
    Archive,
    /// Deep archive / glacier-like.
    DeepArchive,
    /// Reduced redundancy.
    ReducedRedundancy,
    /// Custom / provider-specific class.
    Custom(String),
}

impl StorageClass {
    /// Parse a storage class from a string, case-insensitive.
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "standard" => Self::Standard,
            "ia" | "infrequent_access" | "standard_ia" | "onezone_ia" => Self::InfrequentAccess,
            "archive" | "glacier" => Self::Archive,
            "deep_archive" | "glacier_deep_archive" => Self::DeepArchive,
            "reduced_redundancy" | "rrs" => Self::ReducedRedundancy,
            other => Self::Custom(other.to_string()),
        }
    }
}

// ─── SizeBucket ─────────────────────────────────────────────────────────────

/// Predefined size ranges for histogram aggregation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SizeBucket {
    /// Human-readable label (e.g. "1 KiB – 64 KiB").
    pub label: String,
    /// Inclusive lower bound in bytes.
    pub lower_bytes: u64,
    /// Exclusive upper bound in bytes (u64::MAX for unbounded).
    pub upper_bytes: u64,
    /// Number of objects in this bucket.
    pub count: u64,
    /// Sum of object sizes in this bucket.
    pub total_bytes: u64,
}

/// Default size bucket boundaries.
fn default_size_buckets() -> Vec<SizeBucket> {
    let boundaries: &[(u64, u64, &str)] = &[
        (0, 1024, "0 B – 1 KiB"),
        (1024, 64 * 1024, "1 KiB – 64 KiB"),
        (64 * 1024, 1024 * 1024, "64 KiB – 1 MiB"),
        (1024 * 1024, 64 * 1024 * 1024, "1 MiB – 64 MiB"),
        (64 * 1024 * 1024, 1024 * 1024 * 1024, "64 MiB – 1 GiB"),
        (1024 * 1024 * 1024, u64::MAX, "1 GiB+"),
    ];
    boundaries
        .iter()
        .map(|&(lo, hi, label)| SizeBucket {
            label: label.to_string(),
            lower_bytes: lo,
            upper_bytes: hi,
            count: 0,
            total_bytes: 0,
        })
        .collect()
}

// ─── ClassStats ─────────────────────────────────────────────────────────────

/// Aggregated statistics for a single storage class.
#[derive(Debug, Clone)]
pub struct ClassStats {
    pub class: StorageClass,
    pub object_count: u64,
    pub total_bytes: u64,
    pub smallest_object: u64,
    pub largest_object: u64,
}

impl ClassStats {
    fn new(class: StorageClass) -> Self {
        Self {
            class,
            object_count: 0,
            total_bytes: 0,
            smallest_object: u64::MAX,
            largest_object: 0,
        }
    }

    fn record(&mut self, size: u64) {
        self.object_count += 1;
        self.total_bytes += size;
        self.smallest_object = self.smallest_object.min(size);
        self.largest_object = self.largest_object.max(size);
    }

    /// Average object size in bytes (returns 0 if empty).
    pub fn average_size(&self) -> u64 {
        self.total_bytes.checked_div(self.object_count).unwrap_or(0)
    }
}

// ─── PrefixStats ────────────────────────────────────────────────────────────

/// Aggregated statistics for a key prefix (virtual directory).
#[derive(Debug, Clone)]
pub struct PrefixStats {
    pub prefix: String,
    pub object_count: u64,
    pub total_bytes: u64,
}

impl PrefixStats {
    fn new(prefix: String) -> Self {
        Self {
            prefix,
            object_count: 0,
            total_bytes: 0,
        }
    }

    fn record(&mut self, size: u64) {
        self.object_count += 1;
        self.total_bytes += size;
    }
}

// ─── InventoryReport ────────────────────────────────────────────────────────

/// Complete inventory report for a storage namespace.
#[derive(Debug, Clone)]
pub struct InventoryReport {
    /// Total number of objects.
    pub total_objects: u64,
    /// Total size in bytes across all objects.
    pub total_bytes: u64,
    /// Earliest last-modified timestamp (None if empty).
    pub earliest_modified: Option<DateTime<Utc>>,
    /// Latest last-modified timestamp (None if empty).
    pub latest_modified: Option<DateTime<Utc>>,
    /// Per-class statistics.
    pub by_class: HashMap<StorageClass, ClassStats>,
    /// Per-prefix statistics (first path component as prefix).
    pub by_prefix: HashMap<String, PrefixStats>,
    /// Size distribution histogram.
    pub size_histogram: Vec<SizeBucket>,
    /// Report generation timestamp.
    pub generated_at: DateTime<Utc>,
}

impl InventoryReport {
    /// Return fraction of total bytes in a given storage class (0.0 – 1.0).
    pub fn class_byte_fraction(&self, class: &StorageClass) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        let class_bytes = self.by_class.get(class).map_or(0, |s| s.total_bytes);
        class_bytes as f64 / self.total_bytes as f64
    }

    /// Top-N prefixes by total bytes, descending.
    pub fn top_prefixes_by_size(&self, n: usize) -> Vec<&PrefixStats> {
        let mut sorted: Vec<_> = self.by_prefix.values().collect();
        sorted.sort_by(|a, b| b.total_bytes.cmp(&a.total_bytes));
        sorted.truncate(n);
        sorted
    }

    /// Top-N prefixes by object count, descending.
    pub fn top_prefixes_by_count(&self, n: usize) -> Vec<&PrefixStats> {
        let mut sorted: Vec<_> = self.by_prefix.values().collect();
        sorted.sort_by(|a, b| b.object_count.cmp(&a.object_count));
        sorted.truncate(n);
        sorted
    }

    /// Total number of distinct storage classes observed.
    pub fn distinct_classes(&self) -> usize {
        self.by_class.len()
    }

    /// Average object size in bytes (0 if no objects).
    pub fn average_object_size(&self) -> u64 {
        self.total_bytes
            .checked_div(self.total_objects)
            .unwrap_or(0)
    }
}

// ─── ObjectEntry ────────────────────────────────────────────────────────────

/// A single object descriptor fed into the inventory builder.
#[derive(Debug, Clone)]
pub struct ObjectEntry {
    pub key: String,
    pub size: u64,
    pub storage_class: StorageClass,
    pub last_modified: DateTime<Utc>,
}

// ─── InventoryBuilder ───────────────────────────────────────────────────────

/// Builder that accumulates object entries and produces an [`InventoryReport`].
#[derive(Debug)]
pub struct InventoryBuilder {
    /// Delimiter used to extract the prefix from an object key.
    prefix_delimiter: char,
    /// Depth of prefix extraction (1 = first component only).
    prefix_depth: usize,
    entries: Vec<ObjectEntry>,
}

impl Default for InventoryBuilder {
    fn default() -> Self {
        Self {
            prefix_delimiter: '/',
            prefix_depth: 1,
            entries: Vec::new(),
        }
    }
}

impl InventoryBuilder {
    /// Create a new builder with default `/` delimiter and depth 1.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the delimiter used to split keys into prefix components.
    pub fn with_delimiter(mut self, delimiter: char) -> Self {
        self.prefix_delimiter = delimiter;
        self
    }

    /// Set the depth of prefix extraction.
    ///
    /// Depth 1 means only the first path component is used as the prefix;
    /// depth 2 uses the first two components joined by the delimiter, etc.
    pub fn with_prefix_depth(mut self, depth: usize) -> Self {
        self.prefix_depth = depth.max(1);
        self
    }

    /// Add a single object entry.
    pub fn add_object(&mut self, entry: ObjectEntry) {
        self.entries.push(entry);
    }

    /// Add multiple object entries at once.
    pub fn add_objects(&mut self, entries: impl IntoIterator<Item = ObjectEntry>) {
        self.entries.extend(entries);
    }

    /// Extract the prefix from a key according to delimiter and depth.
    fn extract_prefix(&self, key: &str) -> String {
        let parts: Vec<&str> = key.split(self.prefix_delimiter).collect();
        if parts.len() <= self.prefix_depth {
            // Object is at root or within the requested depth — use full dirname
            if parts.len() <= 1 {
                return "(root)".to_string();
            }
            parts[..parts.len() - 1].join(&self.prefix_delimiter.to_string())
        } else {
            parts[..self.prefix_depth].join(&self.prefix_delimiter.to_string())
        }
    }

    /// Build the final inventory report.
    pub fn build(self) -> InventoryReport {
        let mut total_objects: u64 = 0;
        let mut total_bytes: u64 = 0;
        let mut earliest: Option<DateTime<Utc>> = None;
        let mut latest: Option<DateTime<Utc>> = None;
        let mut by_class: HashMap<StorageClass, ClassStats> = HashMap::new();
        let mut by_prefix: HashMap<String, PrefixStats> = HashMap::new();
        let mut size_histogram = default_size_buckets();

        for entry in &self.entries {
            total_objects += 1;
            total_bytes += entry.size;

            // Timestamp tracking
            match earliest {
                Some(e) if entry.last_modified < e => earliest = Some(entry.last_modified),
                None => earliest = Some(entry.last_modified),
                _ => {}
            }
            match latest {
                Some(l) if entry.last_modified > l => latest = Some(entry.last_modified),
                None => latest = Some(entry.last_modified),
                _ => {}
            }

            // Class stats
            by_class
                .entry(entry.storage_class.clone())
                .or_insert_with(|| ClassStats::new(entry.storage_class.clone()))
                .record(entry.size);

            // Prefix stats
            let prefix = self.extract_prefix(&entry.key);
            by_prefix
                .entry(prefix.clone())
                .or_insert_with(|| PrefixStats::new(prefix))
                .record(entry.size);

            // Size histogram
            for bucket in &mut size_histogram {
                if entry.size >= bucket.lower_bytes && entry.size < bucket.upper_bytes {
                    bucket.count += 1;
                    bucket.total_bytes += entry.size;
                    break;
                }
            }
        }

        InventoryReport {
            total_objects,
            total_bytes,
            earliest_modified: earliest,
            latest_modified: latest,
            by_class,
            by_prefix,
            size_histogram,
            generated_at: Utc::now(),
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn make_entry(key: &str, size: u64, class: StorageClass, days_ago: i64) -> ObjectEntry {
        let ts = Utc::now() - chrono::Duration::days(days_ago);
        ObjectEntry {
            key: key.to_string(),
            size,
            storage_class: class,
            last_modified: ts,
        }
    }

    #[test]
    fn test_empty_report() {
        let report = InventoryBuilder::new().build();
        assert_eq!(report.total_objects, 0);
        assert_eq!(report.total_bytes, 0);
        assert!(report.earliest_modified.is_none());
        assert!(report.latest_modified.is_none());
        assert_eq!(report.average_object_size(), 0);
    }

    #[test]
    fn test_single_object_report() {
        let mut builder = InventoryBuilder::new();
        builder.add_object(make_entry("data/file.txt", 1024, StorageClass::Standard, 5));
        let report = builder.build();

        assert_eq!(report.total_objects, 1);
        assert_eq!(report.total_bytes, 1024);
        assert!(report.earliest_modified.is_some());
        assert!(report.latest_modified.is_some());
        assert_eq!(report.distinct_classes(), 1);
        assert_eq!(report.average_object_size(), 1024);
    }

    #[test]
    fn test_multiple_classes() {
        let mut builder = InventoryBuilder::new();
        builder.add_object(make_entry("a/1.bin", 500, StorageClass::Standard, 1));
        builder.add_object(make_entry("a/2.bin", 1500, StorageClass::Archive, 2));
        builder.add_object(make_entry("b/3.bin", 2000, StorageClass::Standard, 3));
        let report = builder.build();

        assert_eq!(report.total_objects, 3);
        assert_eq!(report.total_bytes, 4000);
        assert_eq!(report.distinct_classes(), 2);

        let standard = report
            .by_class
            .get(&StorageClass::Standard)
            .expect("standard class");
        assert_eq!(standard.object_count, 2);
        assert_eq!(standard.total_bytes, 2500);
        assert_eq!(standard.smallest_object, 500);
        assert_eq!(standard.largest_object, 2000);
    }

    #[test]
    fn test_prefix_extraction_depth_1() {
        let mut builder = InventoryBuilder::new().with_prefix_depth(1);
        builder.add_object(make_entry(
            "media/video/clip.mp4",
            1_000_000,
            StorageClass::Standard,
            1,
        ));
        builder.add_object(make_entry(
            "media/audio/track.flac",
            500_000,
            StorageClass::Standard,
            1,
        ));
        builder.add_object(make_entry(
            "logs/access.log",
            200,
            StorageClass::InfrequentAccess,
            10,
        ));
        let report = builder.build();

        assert!(report.by_prefix.contains_key("media"));
        assert!(report.by_prefix.contains_key("logs"));
        let media = report.by_prefix.get("media").expect("media prefix");
        assert_eq!(media.object_count, 2);
    }

    #[test]
    fn test_prefix_extraction_depth_2() {
        let mut builder = InventoryBuilder::new().with_prefix_depth(2);
        builder.add_object(make_entry(
            "media/video/clip.mp4",
            1_000_000,
            StorageClass::Standard,
            1,
        ));
        builder.add_object(make_entry(
            "media/audio/track.flac",
            500_000,
            StorageClass::Standard,
            1,
        ));
        let report = builder.build();

        assert!(report.by_prefix.contains_key("media/video"));
        assert!(report.by_prefix.contains_key("media/audio"));
    }

    #[test]
    fn test_root_objects_prefix() {
        let mut builder = InventoryBuilder::new();
        builder.add_object(make_entry("readme.txt", 100, StorageClass::Standard, 1));
        let report = builder.build();

        assert!(report.by_prefix.contains_key("(root)"));
    }

    #[test]
    fn test_size_histogram() {
        let mut builder = InventoryBuilder::new();
        // Tiny file (0 – 1 KiB bucket)
        builder.add_object(make_entry("a.txt", 500, StorageClass::Standard, 1));
        // Medium file (1 MiB – 64 MiB bucket)
        builder.add_object(make_entry(
            "b.bin",
            5 * 1024 * 1024,
            StorageClass::Standard,
            1,
        ));
        // Large file (1 GiB+ bucket)
        builder.add_object(make_entry(
            "c.dat",
            2 * 1024 * 1024 * 1024,
            StorageClass::Archive,
            1,
        ));
        let report = builder.build();

        let tiny_bucket = report
            .size_histogram
            .iter()
            .find(|b| b.label == "0 B – 1 KiB")
            .expect("tiny bucket");
        assert_eq!(tiny_bucket.count, 1);
        assert_eq!(tiny_bucket.total_bytes, 500);

        let large_bucket = report
            .size_histogram
            .iter()
            .find(|b| b.label == "1 GiB+")
            .expect("large bucket");
        assert_eq!(large_bucket.count, 1);
    }

    #[test]
    fn test_class_byte_fraction() {
        let mut builder = InventoryBuilder::new();
        builder.add_object(make_entry("a.bin", 750, StorageClass::Standard, 1));
        builder.add_object(make_entry("b.bin", 250, StorageClass::Archive, 1));
        let report = builder.build();

        let frac = report.class_byte_fraction(&StorageClass::Standard);
        assert!((frac - 0.75).abs() < 0.001);
        let frac_archive = report.class_byte_fraction(&StorageClass::Archive);
        assert!((frac_archive - 0.25).abs() < 0.001);
        // Non-existent class
        assert!(
            (report.class_byte_fraction(&StorageClass::DeepArchive) - 0.0).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_top_prefixes() {
        let mut builder = InventoryBuilder::new();
        builder.add_object(make_entry("big/a.bin", 10_000, StorageClass::Standard, 1));
        builder.add_object(make_entry("big/b.bin", 20_000, StorageClass::Standard, 1));
        builder.add_object(make_entry("small/c.bin", 100, StorageClass::Standard, 1));
        builder.add_object(make_entry("medium/d.bin", 5_000, StorageClass::Standard, 1));
        let report = builder.build();

        let top = report.top_prefixes_by_size(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].prefix, "big");
        assert_eq!(top[0].total_bytes, 30_000);
    }

    #[test]
    fn test_timestamp_tracking() {
        let t1 = Utc
            .with_ymd_and_hms(2024, 1, 1, 0, 0, 0)
            .single()
            .expect("valid date");
        let t2 = Utc
            .with_ymd_and_hms(2025, 6, 15, 12, 0, 0)
            .single()
            .expect("valid date");

        let mut builder = InventoryBuilder::new();
        builder.add_object(ObjectEntry {
            key: "old.bin".to_string(),
            size: 100,
            storage_class: StorageClass::Standard,
            last_modified: t1,
        });
        builder.add_object(ObjectEntry {
            key: "new.bin".to_string(),
            size: 200,
            storage_class: StorageClass::Standard,
            last_modified: t2,
        });
        let report = builder.build();

        assert_eq!(report.earliest_modified, Some(t1));
        assert_eq!(report.latest_modified, Some(t2));
    }

    #[test]
    fn test_storage_class_from_str_loose() {
        assert_eq!(
            StorageClass::from_str_loose("standard"),
            StorageClass::Standard
        );
        assert_eq!(
            StorageClass::from_str_loose("STANDARD"),
            StorageClass::Standard
        );
        assert_eq!(
            StorageClass::from_str_loose("IA"),
            StorageClass::InfrequentAccess
        );
        assert_eq!(
            StorageClass::from_str_loose("glacier"),
            StorageClass::Archive
        );
        assert_eq!(
            StorageClass::from_str_loose("deep_archive"),
            StorageClass::DeepArchive
        );
        assert_eq!(
            StorageClass::from_str_loose("rrs"),
            StorageClass::ReducedRedundancy
        );
        assert_eq!(
            StorageClass::from_str_loose("intelligent_tiering"),
            StorageClass::Custom("intelligent_tiering".to_string())
        );
    }

    #[test]
    fn test_class_stats_average_size() {
        let mut stats = ClassStats::new(StorageClass::Standard);
        stats.record(100);
        stats.record(200);
        stats.record(300);
        assert_eq!(stats.average_size(), 200);

        let empty = ClassStats::new(StorageClass::Archive);
        assert_eq!(empty.average_size(), 0);
    }

    #[test]
    fn test_add_objects_batch() {
        let entries = vec![
            make_entry("a/1.bin", 100, StorageClass::Standard, 1),
            make_entry("a/2.bin", 200, StorageClass::Standard, 2),
            make_entry("b/3.bin", 300, StorageClass::Archive, 3),
        ];
        let mut builder = InventoryBuilder::new();
        builder.add_objects(entries);
        let report = builder.build();

        assert_eq!(report.total_objects, 3);
        assert_eq!(report.total_bytes, 600);
    }

    #[test]
    fn test_custom_delimiter() {
        let mut builder = InventoryBuilder::new().with_delimiter(':');
        builder.add_object(ObjectEntry {
            key: "bucket:folder:file.txt".to_string(),
            size: 42,
            storage_class: StorageClass::Standard,
            last_modified: Utc::now(),
        });
        let report = builder.build();
        assert!(report.by_prefix.contains_key("bucket"));
    }
}
