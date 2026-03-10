#![allow(dead_code)]
//! GPU buffer copy and blit operations.
//!
//! This module provides abstractions for copying data between GPU buffers,
//! performing region-based copies, and managing staging buffers for
//! host-device data transfer.

use std::collections::VecDeque;

/// Describes a region within a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferRegion {
    /// Byte offset from buffer start.
    pub offset: u64,
    /// Size of the region in bytes.
    pub size: u64,
}

impl BufferRegion {
    /// Create a new buffer region.
    #[must_use]
    pub fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }

    /// Create a region starting at offset 0.
    #[must_use]
    pub fn from_start(size: u64) -> Self {
        Self { offset: 0, size }
    }

    /// End offset (exclusive) of the region.
    #[must_use]
    pub fn end(&self) -> u64 {
        self.offset + self.size
    }

    /// Check if this region overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &BufferRegion) -> bool {
        self.offset < other.end() && other.offset < self.end()
    }

    /// Check if this region is entirely contained within another.
    #[must_use]
    pub fn contained_in(&self, other: &BufferRegion) -> bool {
        self.offset >= other.offset && self.end() <= other.end()
    }

    /// Compute the intersection of two regions, if any.
    #[must_use]
    pub fn intersection(&self, other: &BufferRegion) -> Option<BufferRegion> {
        let start = self.offset.max(other.offset);
        let end = self.end().min(other.end());
        if start < end {
            Some(BufferRegion::new(start, end - start))
        } else {
            None
        }
    }
}

/// Describes a 2D region for image-like buffer copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageRegion {
    /// X offset in pixels.
    pub x: u32,
    /// Y offset in pixels.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl ImageRegion {
    /// Create a new image region.
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Create a region starting at the origin.
    #[must_use]
    pub fn from_size(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }

    /// Total pixel count.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Check if a point is inside the region.
    #[must_use]
    pub fn contains_point(&self, px: u32, py: u32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }
}

/// Direction of a buffer copy operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyDirection {
    /// Host (CPU) to Device (GPU).
    HostToDevice,
    /// Device (GPU) to Host (CPU).
    DeviceToHost,
    /// Device to Device (same GPU).
    DeviceToDevice,
    /// Peer to Peer (between GPUs).
    PeerToPeer,
}

impl std::fmt::Display for CopyDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "Host -> Device"),
            Self::DeviceToHost => write!(f, "Device -> Host"),
            Self::DeviceToDevice => write!(f, "Device -> Device"),
            Self::PeerToPeer => write!(f, "Peer -> Peer"),
        }
    }
}

/// A single buffer copy command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyCommand {
    /// Source buffer identifier.
    pub src_id: u64,
    /// Destination buffer identifier.
    pub dst_id: u64,
    /// Source region.
    pub src_region: BufferRegion,
    /// Destination offset.
    pub dst_offset: u64,
    /// Direction of the copy.
    pub direction: CopyDirection,
}

impl CopyCommand {
    /// Create a new copy command.
    #[must_use]
    pub fn new(
        src_id: u64,
        dst_id: u64,
        src_region: BufferRegion,
        dst_offset: u64,
        direction: CopyDirection,
    ) -> Self {
        Self {
            src_id,
            dst_id,
            src_region,
            dst_offset,
            direction,
        }
    }

    /// The destination region implied by this copy.
    #[must_use]
    pub fn dst_region(&self) -> BufferRegion {
        BufferRegion::new(self.dst_offset, self.src_region.size)
    }

    /// Check if this copy would alias with another (read-write hazard).
    #[must_use]
    pub fn aliases_with(&self, other: &CopyCommand) -> bool {
        // Writing to same buffer with overlapping regions
        if self.dst_id == other.dst_id {
            let my_dst = self.dst_region();
            let other_dst = other.dst_region();
            if my_dst.overlaps(&other_dst) {
                return true;
            }
        }
        // Source of one is destination of another with overlap
        if self.src_id == other.dst_id {
            let other_dst = other.dst_region();
            if self.src_region.overlaps(&other_dst) {
                return true;
            }
        }
        if self.dst_id == other.src_id {
            let my_dst = self.dst_region();
            if my_dst.overlaps(&other.src_region) {
                return true;
            }
        }
        false
    }
}

/// Batches multiple copy commands for efficient submission.
#[derive(Debug, Default)]
pub struct CopyBatch {
    /// Queued commands.
    commands: VecDeque<CopyCommand>,
    /// Total bytes scheduled.
    total_bytes: u64,
}

impl CopyBatch {
    /// Create a new empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a copy command to the batch.
    pub fn push(&mut self, cmd: CopyCommand) {
        self.total_bytes += cmd.src_region.size;
        self.commands.push_back(cmd);
    }

    /// Number of commands in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Whether the batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Total bytes to be copied.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Drain all commands from the batch.
    pub fn drain(&mut self) -> Vec<CopyCommand> {
        self.total_bytes = 0;
        self.commands.drain(..).collect()
    }

    /// Check if any commands in the batch alias with each other.
    #[must_use]
    pub fn has_hazards(&self) -> bool {
        let cmds: Vec<_> = self.commands.iter().collect();
        for i in 0..cmds.len() {
            for j in (i + 1)..cmds.len() {
                if cmds[i].aliases_with(cmds[j]) {
                    return true;
                }
            }
        }
        false
    }

    /// Split the batch into independent sub-batches that can run in parallel.
    #[must_use]
    pub fn split_independent(mut self) -> Vec<Vec<CopyCommand>> {
        let all = self.drain();
        if all.is_empty() {
            return Vec::new();
        }

        let mut batches: Vec<Vec<CopyCommand>> = Vec::new();

        for cmd in all {
            let mut placed = false;
            for batch in &mut batches {
                let conflicts = batch.iter().any(|existing| existing.aliases_with(&cmd));
                if !conflicts {
                    batch.push(cmd.clone());
                    placed = true;
                    break;
                }
            }
            if !placed {
                batches.push(vec![cmd]);
            }
        }

        batches
    }
}

/// Statistics about copy operations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CopyStats {
    /// Total copies performed.
    pub copy_count: u64,
    /// Total bytes transferred.
    pub total_bytes: u64,
    /// Host-to-device transfers.
    pub h2d_count: u64,
    /// Device-to-host transfers.
    pub d2h_count: u64,
    /// Device-to-device transfers.
    pub d2d_count: u64,
}

impl CopyStats {
    /// Create empty stats.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a copy command.
    pub fn record(&mut self, cmd: &CopyCommand) {
        self.copy_count += 1;
        self.total_bytes += cmd.src_region.size;
        match cmd.direction {
            CopyDirection::HostToDevice => self.h2d_count += 1,
            CopyDirection::DeviceToHost => self.d2h_count += 1,
            CopyDirection::DeviceToDevice | CopyDirection::PeerToPeer => self.d2d_count += 1,
        }
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_region_basic() {
        let r = BufferRegion::new(10, 20);
        assert_eq!(r.offset, 10);
        assert_eq!(r.size, 20);
        assert_eq!(r.end(), 30);
    }

    #[test]
    fn test_buffer_region_from_start() {
        let r = BufferRegion::from_start(100);
        assert_eq!(r.offset, 0);
        assert_eq!(r.size, 100);
    }

    #[test]
    fn test_buffer_region_overlaps() {
        let a = BufferRegion::new(0, 10);
        let b = BufferRegion::new(5, 10);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_buffer_region_no_overlap() {
        let a = BufferRegion::new(0, 10);
        let b = BufferRegion::new(10, 10);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_buffer_region_contained() {
        let inner = BufferRegion::new(5, 5);
        let outer = BufferRegion::new(0, 20);
        assert!(inner.contained_in(&outer));
        assert!(!outer.contained_in(&inner));
    }

    #[test]
    fn test_buffer_region_intersection() {
        let a = BufferRegion::new(0, 10);
        let b = BufferRegion::new(5, 10);
        let i = a.intersection(&b).expect("intersection should succeed");
        assert_eq!(i.offset, 5);
        assert_eq!(i.size, 5);
    }

    #[test]
    fn test_buffer_region_no_intersection() {
        let a = BufferRegion::new(0, 5);
        let b = BufferRegion::new(10, 5);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_image_region_basic() {
        let r = ImageRegion::new(10, 20, 100, 50);
        assert_eq!(r.pixel_count(), 5000);
    }

    #[test]
    fn test_image_region_contains_point() {
        let r = ImageRegion::from_size(100, 100);
        assert!(r.contains_point(50, 50));
        assert!(!r.contains_point(100, 100));
        assert!(r.contains_point(0, 0));
    }

    #[test]
    fn test_copy_direction_display() {
        assert_eq!(format!("{}", CopyDirection::HostToDevice), "Host -> Device");
        assert_eq!(format!("{}", CopyDirection::DeviceToHost), "Device -> Host");
    }

    #[test]
    fn test_copy_command_dst_region() {
        let cmd = CopyCommand::new(
            1,
            2,
            BufferRegion::new(0, 1024),
            512,
            CopyDirection::DeviceToDevice,
        );
        let dst = cmd.dst_region();
        assert_eq!(dst.offset, 512);
        assert_eq!(dst.size, 1024);
    }

    #[test]
    fn test_copy_command_aliases() {
        let a = CopyCommand::new(
            1,
            2,
            BufferRegion::new(0, 100),
            0,
            CopyDirection::DeviceToDevice,
        );
        let b = CopyCommand::new(
            3,
            2,
            BufferRegion::new(0, 100),
            50,
            CopyDirection::DeviceToDevice,
        );
        assert!(a.aliases_with(&b));
    }

    #[test]
    fn test_copy_command_no_alias() {
        let a = CopyCommand::new(
            1,
            2,
            BufferRegion::new(0, 100),
            0,
            CopyDirection::DeviceToDevice,
        );
        let b = CopyCommand::new(
            3,
            4,
            BufferRegion::new(0, 100),
            0,
            CopyDirection::DeviceToDevice,
        );
        assert!(!a.aliases_with(&b));
    }

    #[test]
    fn test_copy_batch_push_and_drain() {
        let mut batch = CopyBatch::new();
        assert!(batch.is_empty());

        batch.push(CopyCommand::new(
            1,
            2,
            BufferRegion::from_start(256),
            0,
            CopyDirection::HostToDevice,
        ));
        batch.push(CopyCommand::new(
            3,
            4,
            BufferRegion::from_start(512),
            0,
            CopyDirection::DeviceToHost,
        ));

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.total_bytes(), 768);

        let cmds = batch.drain();
        assert_eq!(cmds.len(), 2);
        assert!(batch.is_empty());
        assert_eq!(batch.total_bytes(), 0);
    }

    #[test]
    fn test_copy_batch_no_hazards() {
        let mut batch = CopyBatch::new();
        batch.push(CopyCommand::new(
            1,
            2,
            BufferRegion::from_start(100),
            0,
            CopyDirection::DeviceToDevice,
        ));
        batch.push(CopyCommand::new(
            3,
            4,
            BufferRegion::from_start(100),
            0,
            CopyDirection::DeviceToDevice,
        ));
        assert!(!batch.has_hazards());
    }

    #[test]
    fn test_copy_batch_with_hazards() {
        let mut batch = CopyBatch::new();
        batch.push(CopyCommand::new(
            1,
            2,
            BufferRegion::from_start(100),
            0,
            CopyDirection::DeviceToDevice,
        ));
        batch.push(CopyCommand::new(
            3,
            2,
            BufferRegion::from_start(100),
            50,
            CopyDirection::DeviceToDevice,
        ));
        assert!(batch.has_hazards());
    }

    #[test]
    fn test_copy_batch_split_independent() {
        let mut batch = CopyBatch::new();
        batch.push(CopyCommand::new(
            1,
            2,
            BufferRegion::from_start(100),
            0,
            CopyDirection::DeviceToDevice,
        ));
        batch.push(CopyCommand::new(
            3,
            2,
            BufferRegion::from_start(100),
            50,
            CopyDirection::DeviceToDevice,
        ));
        batch.push(CopyCommand::new(
            5,
            6,
            BufferRegion::from_start(100),
            0,
            CopyDirection::DeviceToDevice,
        ));

        let batches = batch.split_independent();
        // First and third can go together, second conflicts with first
        assert!(batches.len() >= 2);
    }

    #[test]
    fn test_copy_stats() {
        let mut stats = CopyStats::new();
        let cmd = CopyCommand::new(
            1,
            2,
            BufferRegion::from_start(1024),
            0,
            CopyDirection::HostToDevice,
        );
        stats.record(&cmd);
        assert_eq!(stats.copy_count, 1);
        assert_eq!(stats.total_bytes, 1024);
        assert_eq!(stats.h2d_count, 1);
        assert_eq!(stats.d2h_count, 0);

        stats.reset();
        assert_eq!(stats.copy_count, 0);
    }

    #[test]
    fn test_copy_stats_multiple_directions() {
        let mut stats = CopyStats::new();
        stats.record(&CopyCommand::new(
            1,
            2,
            BufferRegion::from_start(100),
            0,
            CopyDirection::HostToDevice,
        ));
        stats.record(&CopyCommand::new(
            2,
            1,
            BufferRegion::from_start(200),
            0,
            CopyDirection::DeviceToHost,
        ));
        stats.record(&CopyCommand::new(
            2,
            3,
            BufferRegion::from_start(300),
            0,
            CopyDirection::DeviceToDevice,
        ));
        assert_eq!(stats.copy_count, 3);
        assert_eq!(stats.total_bytes, 600);
        assert_eq!(stats.h2d_count, 1);
        assert_eq!(stats.d2h_count, 1);
        assert_eq!(stats.d2d_count, 1);
    }
}
