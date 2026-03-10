//! Multi-stream alignment for synchronizing groups of audio/video streams.
//!
//! Provides group alignment, reference stream selection, and bulk offset
//! computation for production workflows with many simultaneous camera/audio feeds.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Identifier for a stream in a multi-stream group
pub type StreamId = u32;

/// Offset of one stream relative to a reference stream
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StreamOffset {
    /// ID of this stream
    pub stream_id: StreamId,
    /// ID of the reference stream
    pub reference_id: StreamId,
    /// Offset in samples (positive = this stream is delayed relative to reference)
    pub offset_samples: i64,
    /// Confidence in this measurement (0.0 to 1.0)
    pub confidence: f64,
}

impl StreamOffset {
    /// Create a new stream offset
    #[must_use]
    pub fn new(
        stream_id: StreamId,
        reference_id: StreamId,
        offset_samples: i64,
        confidence: f64,
    ) -> Self {
        Self {
            stream_id,
            reference_id,
            offset_samples,
            confidence,
        }
    }

    /// Convert offset to milliseconds
    #[must_use]
    pub fn to_ms(&self, sample_rate: u32) -> f64 {
        self.offset_samples as f64 / f64::from(sample_rate) * 1000.0
    }

    /// Invert the offset (swap stream and reference perspective)
    #[must_use]
    pub fn invert(&self) -> Self {
        Self {
            stream_id: self.reference_id,
            reference_id: self.stream_id,
            offset_samples: -self.offset_samples,
            confidence: self.confidence,
        }
    }
}

/// Strategy for selecting the reference stream
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceStrategy {
    /// Use stream with highest overall confidence as reference
    HighestConfidence,
    /// Use a manually specified stream ID
    Manual(StreamId),
    /// Use the stream with the most connections to other streams
    MostConnected,
    /// Use the stream that minimizes total adjustment magnitude
    MinTotalAdjustment,
}

/// Group of streams to be aligned together
#[derive(Debug, Clone)]
pub struct StreamGroup {
    /// Name of this group
    pub name: String,
    /// Stream IDs in this group
    pub stream_ids: Vec<StreamId>,
    /// Pairwise offsets measured between streams
    offsets: Vec<StreamOffset>,
    /// Selected reference stream
    reference_id: Option<StreamId>,
}

impl StreamGroup {
    /// Create a new empty stream group
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stream_ids: Vec::new(),
            offsets: Vec::new(),
            reference_id: None,
        }
    }

    /// Add a stream to the group
    pub fn add_stream(&mut self, id: StreamId) {
        if !self.stream_ids.contains(&id) {
            self.stream_ids.push(id);
        }
    }

    /// Add a measured offset between two streams
    pub fn add_offset(&mut self, offset: StreamOffset) {
        self.offsets.push(offset);
    }

    /// Number of streams in the group
    #[must_use]
    pub fn len(&self) -> usize {
        self.stream_ids.len()
    }

    /// Returns true if the group is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stream_ids.is_empty()
    }

    /// Select reference stream using the given strategy
    pub fn select_reference(&mut self, strategy: ReferenceStrategy) -> Option<StreamId> {
        if self.stream_ids.is_empty() {
            return None;
        }

        let ref_id = match strategy {
            ReferenceStrategy::Manual(id) => {
                if self.stream_ids.contains(&id) {
                    id
                } else {
                    return None;
                }
            }
            ReferenceStrategy::HighestConfidence => self.stream_with_highest_confidence(),
            ReferenceStrategy::MostConnected => self.most_connected_stream(),
            ReferenceStrategy::MinTotalAdjustment => self.min_adjustment_stream(),
        };

        self.reference_id = Some(ref_id);
        Some(ref_id)
    }

    /// Get the current reference stream
    #[must_use]
    pub fn reference_id(&self) -> Option<StreamId> {
        self.reference_id
    }

    /// Stream with highest average confidence in its offset measurements
    fn stream_with_highest_confidence(&self) -> StreamId {
        let mut confidence_sum: HashMap<StreamId, (f64, usize)> = HashMap::new();

        for offset in &self.offsets {
            let entry = confidence_sum.entry(offset.stream_id).or_insert((0.0, 0));
            entry.0 += offset.confidence;
            entry.1 += 1;

            let entry = confidence_sum
                .entry(offset.reference_id)
                .or_insert((0.0, 0));
            entry.0 += offset.confidence;
            entry.1 += 1;
        }

        self.stream_ids
            .iter()
            .copied()
            .max_by(|&a, &b| {
                let avg_a = confidence_sum.get(&a).map_or(0.0, |(s, c)| s / *c as f64);
                let avg_b = confidence_sum.get(&b).map_or(0.0, |(s, c)| s / *c as f64);
                avg_a
                    .partial_cmp(&avg_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(self.stream_ids[0])
    }

    /// Stream with the most offset connections to other streams
    fn most_connected_stream(&self) -> StreamId {
        let mut connections: HashMap<StreamId, usize> = HashMap::new();
        for offset in &self.offsets {
            *connections.entry(offset.stream_id).or_insert(0) += 1;
            *connections.entry(offset.reference_id).or_insert(0) += 1;
        }

        self.stream_ids
            .iter()
            .copied()
            .max_by_key(|id| connections.get(id).copied().unwrap_or(0))
            .unwrap_or(self.stream_ids[0])
    }

    /// Stream that minimizes total adjustment magnitude when used as reference
    fn min_adjustment_stream(&self) -> StreamId {
        self.stream_ids
            .iter()
            .copied()
            .min_by(|&a, &b| {
                let total_a = self.total_adjustment_for_reference(a);
                let total_b = self.total_adjustment_for_reference(b);
                total_a
                    .partial_cmp(&total_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(self.stream_ids[0])
    }

    /// Compute total absolute adjustment if a given stream were used as reference
    fn total_adjustment_for_reference(&self, ref_id: StreamId) -> f64 {
        let bulk = self.compute_bulk_offsets_for_reference(ref_id);
        bulk.values()
            .map(|o| o.offset_samples.unsigned_abs() as f64)
            .sum()
    }

    /// Compute bulk offsets for all streams relative to a given reference
    #[must_use]
    pub fn compute_bulk_offsets(&self) -> HashMap<StreamId, StreamOffset> {
        let ref_id = self
            .reference_id
            .unwrap_or_else(|| self.stream_ids.first().copied().unwrap_or(0));
        self.compute_bulk_offsets_for_reference(ref_id)
    }

    /// Compute bulk offsets for all streams relative to a specific reference
    #[must_use]
    pub fn compute_bulk_offsets_for_reference(
        &self,
        ref_id: StreamId,
    ) -> HashMap<StreamId, StreamOffset> {
        let mut result = HashMap::new();

        for &sid in &self.stream_ids {
            if sid == ref_id {
                result.insert(sid, StreamOffset::new(sid, ref_id, 0, 1.0));
                continue;
            }

            // Look for direct measurement
            let direct = self.offsets.iter().find(|o| {
                (o.stream_id == sid && o.reference_id == ref_id)
                    || (o.stream_id == ref_id && o.reference_id == sid)
            });

            if let Some(off) = direct {
                let offset = if off.stream_id == sid {
                    *off
                } else {
                    off.invert()
                };
                result.insert(
                    sid,
                    StreamOffset::new(sid, ref_id, offset.offset_samples, offset.confidence),
                );
            }
        }

        result
    }

    /// Get all offsets in the group
    #[must_use]
    pub fn offsets(&self) -> &[StreamOffset] {
        &self.offsets
    }
}

/// Multi-stream aligner that manages multiple groups
#[derive(Debug, Clone)]
pub struct MultiStreamAligner {
    /// Named groups of streams
    groups: HashMap<String, StreamGroup>,
    /// Default reference strategy
    strategy: ReferenceStrategy,
    /// Sample rate used for time calculations
    sample_rate: u32,
}

impl MultiStreamAligner {
    /// Create a new multi-stream aligner
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            groups: HashMap::new(),
            strategy: ReferenceStrategy::HighestConfidence,
            sample_rate,
        }
    }

    /// Add a stream group
    pub fn add_group(&mut self, group: StreamGroup) {
        self.groups.insert(group.name.clone(), group);
    }

    /// Get a reference to a group by name
    #[must_use]
    pub fn group(&self, name: &str) -> Option<&StreamGroup> {
        self.groups.get(name)
    }

    /// Get a mutable reference to a group by name
    pub fn group_mut(&mut self, name: &str) -> Option<&mut StreamGroup> {
        self.groups.get_mut(name)
    }

    /// Set the reference strategy
    pub fn set_strategy(&mut self, strategy: ReferenceStrategy) {
        self.strategy = strategy;
    }

    /// Align all groups using the current strategy
    pub fn align_all(&mut self) -> HashMap<String, HashMap<StreamId, StreamOffset>> {
        let strategy = self.strategy;
        let mut result = HashMap::new();

        let names: Vec<String> = self.groups.keys().cloned().collect();
        for name in names {
            if let Some(group) = self.groups.get_mut(&name) {
                group.select_reference(strategy);
                let offsets = group.compute_bulk_offsets();
                result.insert(name, offsets);
            }
        }

        result
    }

    /// Total number of streams across all groups
    #[must_use]
    pub fn total_streams(&self) -> usize {
        self.groups.values().map(StreamGroup::len).sum()
    }

    /// Total number of groups
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Get sample rate
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Summary statistics for a completed alignment
#[derive(Debug, Clone)]
pub struct AlignmentSummary {
    /// Number of streams aligned
    pub stream_count: usize,
    /// Maximum offset magnitude in samples
    pub max_offset_samples: i64,
    /// Average confidence
    pub average_confidence: f64,
    /// Number of streams with low confidence
    pub low_confidence_count: usize,
}

impl AlignmentSummary {
    /// Build a summary from a bulk offset map
    #[must_use]
    pub fn from_offsets(offsets: &HashMap<StreamId, StreamOffset>) -> Self {
        let count = offsets.len();
        if count == 0 {
            return Self {
                stream_count: 0,
                max_offset_samples: 0,
                average_confidence: 0.0,
                low_confidence_count: 0,
            };
        }

        let max_off = offsets
            .values()
            .map(|o| o.offset_samples.abs())
            .max()
            .unwrap_or(0);

        let avg_conf = offsets.values().map(|o| o.confidence).sum::<f64>() / count as f64;

        let low_conf = offsets.values().filter(|o| o.confidence < 0.5).count();

        Self {
            stream_count: count,
            max_offset_samples: max_off,
            average_confidence: avg_conf,
            low_confidence_count: low_conf,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_group_with_offsets() -> StreamGroup {
        let mut group = StreamGroup::new("cameras");
        group.add_stream(1);
        group.add_stream(2);
        group.add_stream(3);
        // Stream 2 is 100 samples ahead of stream 1
        group.add_offset(StreamOffset::new(2, 1, -100, 0.9));
        // Stream 3 is 200 samples behind stream 1
        group.add_offset(StreamOffset::new(3, 1, 200, 0.8));
        group
    }

    #[test]
    fn test_stream_offset_creation() {
        let off = StreamOffset::new(2, 1, 500, 0.9);
        assert_eq!(off.stream_id, 2);
        assert_eq!(off.reference_id, 1);
        assert_eq!(off.offset_samples, 500);
    }

    #[test]
    fn test_stream_offset_to_ms() {
        let off = StreamOffset::new(2, 1, 4800, 0.9);
        let ms = off.to_ms(48000);
        assert!((ms - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_stream_offset_invert() {
        let off = StreamOffset::new(2, 1, 500, 0.9);
        let inv = off.invert();
        assert_eq!(inv.stream_id, 1);
        assert_eq!(inv.reference_id, 2);
        assert_eq!(inv.offset_samples, -500);
    }

    #[test]
    fn test_stream_group_creation() {
        let group = StreamGroup::new("test");
        assert_eq!(group.name, "test");
        assert!(group.is_empty());
    }

    #[test]
    fn test_stream_group_add_stream() {
        let mut group = StreamGroup::new("test");
        group.add_stream(1);
        group.add_stream(2);
        group.add_stream(1); // duplicate should not be added
        assert_eq!(group.len(), 2);
    }

    #[test]
    fn test_stream_group_manual_reference() {
        let mut group = make_group_with_offsets();
        let ref_id = group.select_reference(ReferenceStrategy::Manual(1));
        assert_eq!(ref_id, Some(1));
        assert_eq!(group.reference_id(), Some(1));
    }

    #[test]
    fn test_stream_group_manual_reference_invalid() {
        let mut group = make_group_with_offsets();
        let ref_id = group.select_reference(ReferenceStrategy::Manual(99));
        assert_eq!(ref_id, None);
    }

    #[test]
    fn test_stream_group_highest_confidence_reference() {
        let mut group = make_group_with_offsets();
        let ref_id = group.select_reference(ReferenceStrategy::HighestConfidence);
        assert!(ref_id.is_some());
        assert!(group
            .stream_ids
            .contains(&ref_id.expect("test expectation failed")));
    }

    #[test]
    fn test_stream_group_most_connected_reference() {
        let mut group = make_group_with_offsets();
        let ref_id = group.select_reference(ReferenceStrategy::MostConnected);
        // Stream 1 appears in both offsets, so should be most connected
        assert_eq!(ref_id, Some(1));
    }

    #[test]
    fn test_stream_group_bulk_offsets() {
        let mut group = make_group_with_offsets();
        group.select_reference(ReferenceStrategy::Manual(1));
        let offsets = group.compute_bulk_offsets();
        // Stream 1 should have 0 offset (reference)
        assert_eq!(offsets[&1].offset_samples, 0);
        // Stream 2 is -100 relative to 1
        assert_eq!(offsets[&2].offset_samples, -100);
        // Stream 3 is +200 relative to 1
        assert_eq!(offsets[&3].offset_samples, 200);
    }

    #[test]
    fn test_multi_stream_aligner_creation() {
        let aligner = MultiStreamAligner::new(48000);
        assert_eq!(aligner.sample_rate(), 48000);
        assert_eq!(aligner.group_count(), 0);
    }

    #[test]
    fn test_multi_stream_aligner_add_group() {
        let mut aligner = MultiStreamAligner::new(48000);
        let mut group = StreamGroup::new("cameras");
        group.add_stream(1);
        group.add_stream(2);
        aligner.add_group(group);
        assert_eq!(aligner.group_count(), 1);
        assert_eq!(aligner.total_streams(), 2);
    }

    #[test]
    fn test_multi_stream_aligner_get_group() {
        let mut aligner = MultiStreamAligner::new(48000);
        aligner.add_group(StreamGroup::new("cameras"));
        assert!(aligner.group("cameras").is_some());
        assert!(aligner.group("missing").is_none());
    }

    #[test]
    fn test_multi_stream_aligner_align_all() {
        let mut aligner = MultiStreamAligner::new(48000);
        let group = make_group_with_offsets();
        aligner.add_group(group);
        aligner.set_strategy(ReferenceStrategy::Manual(1));
        let results = aligner.align_all();
        assert!(results.contains_key("cameras"));
    }

    #[test]
    fn test_alignment_summary_empty() {
        let summary = AlignmentSummary::from_offsets(&HashMap::new());
        assert_eq!(summary.stream_count, 0);
        assert_eq!(summary.max_offset_samples, 0);
    }

    #[test]
    fn test_alignment_summary_from_offsets() {
        let mut offsets = HashMap::new();
        offsets.insert(1u32, StreamOffset::new(1, 1, 0, 1.0));
        offsets.insert(2u32, StreamOffset::new(2, 1, 200, 0.9));
        offsets.insert(3u32, StreamOffset::new(3, 1, -300, 0.3));

        let summary = AlignmentSummary::from_offsets(&offsets);
        assert_eq!(summary.stream_count, 3);
        assert_eq!(summary.max_offset_samples, 300);
        assert_eq!(summary.low_confidence_count, 1); // stream 3 has 0.3
    }
}
