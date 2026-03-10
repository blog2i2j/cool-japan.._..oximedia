//! Track mapping and routing.
//!
//! Provides flexible track remapping for complex workflows.

#![forbid(unsafe_code)]

use oximedia_core::{OxiError, OxiResult};
use std::collections::HashMap;

use crate::{Packet, StreamInfo};

/// Mapping from input track to output track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackMapping {
    /// Input stream index.
    pub input_index: usize,
    /// Output stream index.
    pub output_index: usize,
}

impl TrackMapping {
    /// Creates a new track mapping.
    #[must_use]
    pub const fn new(input_index: usize, output_index: usize) -> Self {
        Self {
            input_index,
            output_index,
        }
    }

    /// Creates an identity mapping (input == output).
    #[must_use]
    pub const fn identity(index: usize) -> Self {
        Self {
            input_index: index,
            output_index: index,
        }
    }
}

/// Router for remapping tracks.
#[derive(Debug, Clone)]
pub struct TrackRouter {
    mappings: Vec<TrackMapping>,
    reverse_map: HashMap<usize, usize>,
}

impl TrackRouter {
    /// Creates a new track router.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mappings: Vec::new(),
            reverse_map: HashMap::new(),
        }
    }

    /// Adds a mapping.
    pub fn add_mapping(&mut self, mapping: TrackMapping) -> &mut Self {
        self.mappings.push(mapping);
        self.reverse_map
            .insert(mapping.input_index, mapping.output_index);
        self
    }

    /// Adds an identity mapping for a stream.
    pub fn add_identity(&mut self, index: usize) -> &mut Self {
        self.add_mapping(TrackMapping::identity(index))
    }

    /// Returns all mappings.
    #[must_use]
    pub fn mappings(&self) -> &[TrackMapping] {
        &self.mappings
    }

    /// Maps an input index to an output index.
    #[must_use]
    pub fn map(&self, input_index: usize) -> Option<usize> {
        self.reverse_map.get(&input_index).copied()
    }

    /// Remaps a packet to its output index.
    ///
    /// # Errors
    ///
    /// Returns `Err` if there is no mapping for the packet's stream index.
    pub fn remap_packet(&self, mut packet: Packet) -> OxiResult<Packet> {
        let output_index = self.map(packet.stream_index).ok_or_else(|| {
            OxiError::InvalidData(format!("No mapping for stream {}", packet.stream_index))
        })?;

        packet.stream_index = output_index;
        Ok(packet)
    }

    /// Creates a router from a simple index array.
    ///
    /// The array maps input indices to output indices.
    /// Use `None` to skip a stream.
    #[must_use]
    pub fn from_array(mappings: &[Option<usize>]) -> Self {
        let mut router = Self::new();

        for (input_index, output_index) in mappings.iter().enumerate() {
            if let Some(output_index) = output_index {
                router.add_mapping(TrackMapping::new(input_index, *output_index));
            }
        }

        router
    }

    /// Creates an identity router for the given number of streams.
    #[must_use]
    pub fn identity(stream_count: usize) -> Self {
        let mut router = Self::new();
        for i in 0..stream_count {
            router.add_identity(i);
        }
        router
    }

    /// Clears all mappings.
    pub fn clear(&mut self) {
        self.mappings.clear();
        self.reverse_map.clear();
    }

    /// Returns the number of mappings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mappings.len()
    }

    /// Returns true if there are no mappings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }
}

impl Default for TrackRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for complex track routing.
pub struct TrackRoutingBuilder {
    input_streams: Vec<StreamInfo>,
    output_mappings: Vec<Option<usize>>,
}

impl TrackRoutingBuilder {
    /// Creates a new routing builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            input_streams: Vec::new(),
            output_mappings: Vec::new(),
        }
    }

    /// Sets the input streams.
    pub fn with_input_streams(&mut self, streams: Vec<StreamInfo>) -> &mut Self {
        self.input_streams = streams;
        self.output_mappings = vec![None; self.input_streams.len()];
        self
    }

    /// Maps an input stream to an output index.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `input_index` is out of range.
    pub fn map_stream(&mut self, input_index: usize, output_index: usize) -> OxiResult<&mut Self> {
        if input_index >= self.output_mappings.len() {
            return Err(OxiError::InvalidData(format!(
                "Input index {input_index} out of range"
            )));
        }

        self.output_mappings[input_index] = Some(output_index);
        Ok(self)
    }

    /// Skips an input stream (excludes it from output).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `input_index` is out of range.
    pub fn skip_stream(&mut self, input_index: usize) -> OxiResult<&mut Self> {
        if input_index >= self.output_mappings.len() {
            return Err(OxiError::InvalidData(format!(
                "Input index {input_index} out of range"
            )));
        }

        self.output_mappings[input_index] = None;
        Ok(self)
    }

    /// Builds the track router.
    #[must_use]
    pub fn build(&self) -> TrackRouter {
        TrackRouter::from_array(&self.output_mappings)
    }

    /// Returns the output streams in the mapped order.
    #[must_use]
    pub fn output_streams(&self) -> Vec<StreamInfo> {
        let mut output_streams = vec![None; self.count_outputs()];

        for (input_index, output_index) in self.output_mappings.iter().enumerate() {
            if let Some(output_index) = output_index {
                if *output_index < output_streams.len() {
                    output_streams[*output_index] = Some(self.input_streams[input_index].clone());
                }
            }
        }

        output_streams.into_iter().flatten().collect()
    }

    /// Returns the number of output streams.
    fn count_outputs(&self) -> usize {
        self.output_mappings
            .iter()
            .filter_map(|&x| x)
            .max()
            .map_or(0, |max| max + 1)
    }
}

impl Default for TrackRoutingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Common routing patterns.
pub struct RoutingPresets;

impl RoutingPresets {
    /// Creates a router that swaps two tracks.
    #[must_use]
    pub fn swap(index1: usize, index2: usize, total_streams: usize) -> TrackRouter {
        let mut router = TrackRouter::new();

        for i in 0..total_streams {
            let output = if i == index1 {
                index2
            } else if i == index2 {
                index1
            } else {
                i
            };
            router.add_mapping(TrackMapping::new(i, output));
        }

        router
    }

    /// Creates a router that reorders tracks to a specific order.
    #[must_use]
    pub fn reorder(new_order: &[usize]) -> TrackRouter {
        let mut router = TrackRouter::new();

        for (output_index, &input_index) in new_order.iter().enumerate() {
            router.add_mapping(TrackMapping::new(input_index, output_index));
        }

        router
    }

    /// Creates a router that filters tracks by indices.
    #[must_use]
    pub fn filter(indices: &[usize]) -> TrackRouter {
        let mut router = TrackRouter::new();

        for (output_index, &input_index) in indices.iter().enumerate() {
            router.add_mapping(TrackMapping::new(input_index, output_index));
        }

        router
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use oximedia_core::{CodecId, Rational, Timestamp};

    fn create_test_stream(index: usize) -> StreamInfo {
        let mut stream = StreamInfo::new(index, CodecId::Opus, Rational::new(1, 48000));
        stream.codec_params = crate::stream::CodecParams::audio(48000, 2);
        stream
    }

    fn create_test_packet(stream_index: usize) -> Packet {
        Packet::new(
            stream_index,
            Bytes::new(),
            Timestamp::new(0, Rational::new(1, 1000)),
            crate::PacketFlags::empty(),
        )
    }

    #[test]
    fn test_track_mapping() {
        let mapping = TrackMapping::new(0, 1);
        assert_eq!(mapping.input_index, 0);
        assert_eq!(mapping.output_index, 1);

        let identity = TrackMapping::identity(5);
        assert_eq!(identity.input_index, 5);
        assert_eq!(identity.output_index, 5);
    }

    #[test]
    fn test_track_router() {
        let mut router = TrackRouter::new();
        router.add_mapping(TrackMapping::new(0, 1));
        router.add_mapping(TrackMapping::new(1, 0));

        assert_eq!(router.map(0), Some(1));
        assert_eq!(router.map(1), Some(0));
        assert_eq!(router.map(2), None);

        assert_eq!(router.len(), 2);
        assert!(!router.is_empty());
    }

    #[test]
    fn test_remap_packet() {
        let mut router = TrackRouter::new();
        router.add_mapping(TrackMapping::new(0, 5));

        let packet = create_test_packet(0);
        let remapped = router
            .remap_packet(packet)
            .expect("operation should succeed");

        assert_eq!(remapped.stream_index, 5);
    }

    #[test]
    fn test_router_from_array() {
        let mappings = vec![Some(2), None, Some(0), Some(1)];
        let router = TrackRouter::from_array(&mappings);

        assert_eq!(router.map(0), Some(2));
        assert_eq!(router.map(1), None);
        assert_eq!(router.map(2), Some(0));
        assert_eq!(router.map(3), Some(1));
    }

    #[test]
    fn test_identity_router() {
        let router = TrackRouter::identity(3);

        assert_eq!(router.map(0), Some(0));
        assert_eq!(router.map(1), Some(1));
        assert_eq!(router.map(2), Some(2));
        assert_eq!(router.len(), 3);
    }

    #[test]
    fn test_routing_builder() {
        let streams = vec![
            create_test_stream(0),
            create_test_stream(1),
            create_test_stream(2),
        ];

        let mut builder = TrackRoutingBuilder::new();
        builder.with_input_streams(streams);
        builder.map_stream(0, 1).expect("operation should succeed");
        builder.map_stream(1, 0).expect("operation should succeed");
        builder.skip_stream(2).expect("operation should succeed");

        let router = builder.build();
        assert_eq!(router.map(0), Some(1));
        assert_eq!(router.map(1), Some(0));
        assert_eq!(router.map(2), None);
    }

    #[test]
    fn test_routing_presets_swap() {
        let router = RoutingPresets::swap(0, 2, 3);

        assert_eq!(router.map(0), Some(2));
        assert_eq!(router.map(1), Some(1));
        assert_eq!(router.map(2), Some(0));
    }

    #[test]
    fn test_routing_presets_reorder() {
        let new_order = vec![2, 0, 1];
        let router = RoutingPresets::reorder(&new_order);

        assert_eq!(router.map(2), Some(0));
        assert_eq!(router.map(0), Some(1));
        assert_eq!(router.map(1), Some(2));
    }

    #[test]
    fn test_routing_presets_filter() {
        let indices = vec![1, 3];
        let router = RoutingPresets::filter(&indices);

        assert_eq!(router.map(1), Some(0));
        assert_eq!(router.map(3), Some(1));
        assert_eq!(router.map(0), None);
    }
}
