#![allow(dead_code)]
//! GPU barrier and synchronization management.
//!
//! This module provides a barrier manager that tracks and optimizes
//! memory and execution barriers between GPU operations, ensuring
//! correct ordering of read/write operations across command buffers.

use std::collections::HashMap;
use std::fmt;

/// Describes the type of resource access for barrier tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessType {
    /// No access (initial state).
    None,
    /// Read-only access from a shader.
    ShaderRead,
    /// Write access from a shader (compute or fragment).
    ShaderWrite,
    /// Transfer source (copy from).
    TransferSrc,
    /// Transfer destination (copy to).
    TransferDst,
    /// Host read (CPU readback).
    HostRead,
    /// Host write (CPU upload).
    HostWrite,
}

impl fmt::Display for AccessType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::ShaderRead => write!(f, "ShaderRead"),
            Self::ShaderWrite => write!(f, "ShaderWrite"),
            Self::TransferSrc => write!(f, "TransferSrc"),
            Self::TransferDst => write!(f, "TransferDst"),
            Self::HostRead => write!(f, "HostRead"),
            Self::HostWrite => write!(f, "HostWrite"),
        }
    }
}

/// Pipeline stage where a barrier is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    /// Top of pipe (no operations have started).
    TopOfPipe,
    /// Compute shader stage.
    Compute,
    /// Transfer/copy stage.
    Transfer,
    /// Host stage (CPU access).
    Host,
    /// Bottom of pipe (all operations completed).
    BottomOfPipe,
}

impl fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TopOfPipe => write!(f, "TopOfPipe"),
            Self::Compute => write!(f, "Compute"),
            Self::Transfer => write!(f, "Transfer"),
            Self::Host => write!(f, "Host"),
            Self::BottomOfPipe => write!(f, "BottomOfPipe"),
        }
    }
}

/// A unique identifier for a tracked resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(pub u64);

/// Describes a single barrier between two access types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BarrierDesc {
    /// The resource this barrier applies to.
    pub resource_id: ResourceId,
    /// The previous access type.
    pub src_access: AccessType,
    /// The next access type.
    pub dst_access: AccessType,
    /// The pipeline stage of the previous access.
    pub src_stage: PipelineStage,
    /// The pipeline stage of the next access.
    pub dst_stage: PipelineStage,
}

impl BarrierDesc {
    /// Create a new barrier description.
    pub fn new(
        resource_id: ResourceId,
        src_access: AccessType,
        dst_access: AccessType,
        src_stage: PipelineStage,
        dst_stage: PipelineStage,
    ) -> Self {
        Self {
            resource_id,
            src_access,
            dst_access,
            src_stage,
            dst_stage,
        }
    }

    /// Check whether this barrier is a read-after-write hazard.
    pub fn is_raw_hazard(&self) -> bool {
        matches!(
            self.src_access,
            AccessType::ShaderWrite | AccessType::TransferDst | AccessType::HostWrite
        ) && matches!(
            self.dst_access,
            AccessType::ShaderRead | AccessType::TransferSrc | AccessType::HostRead
        )
    }

    /// Check whether this barrier is a write-after-write hazard.
    pub fn is_waw_hazard(&self) -> bool {
        matches!(
            self.src_access,
            AccessType::ShaderWrite | AccessType::TransferDst | AccessType::HostWrite
        ) && matches!(
            self.dst_access,
            AccessType::ShaderWrite | AccessType::TransferDst | AccessType::HostWrite
        )
    }

    /// Check whether this barrier is a write-after-read hazard.
    pub fn is_war_hazard(&self) -> bool {
        matches!(
            self.src_access,
            AccessType::ShaderRead | AccessType::TransferSrc | AccessType::HostRead
        ) && matches!(
            self.dst_access,
            AccessType::ShaderWrite | AccessType::TransferDst | AccessType::HostWrite
        )
    }
}

/// Tracks the current access state of each resource.
#[derive(Debug, Clone)]
struct ResourceState {
    /// Current access type.
    access: AccessType,
    /// Current pipeline stage.
    stage: PipelineStage,
}

/// Manages barriers for a set of GPU resources.
///
/// The barrier manager tracks current access states and automatically
/// determines which barriers are needed when a resource transitions
/// to a new access pattern.
pub struct BarrierManager {
    /// Current state of each tracked resource.
    states: HashMap<ResourceId, ResourceState>,
    /// Accumulated pending barriers to be submitted.
    pending: Vec<BarrierDesc>,
    /// Total number of barriers emitted.
    total_barriers: u64,
    /// Number of barriers that were optimized away (redundant).
    optimized_away: u64,
}

impl BarrierManager {
    /// Create a new empty barrier manager.
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            pending: Vec::new(),
            total_barriers: 0,
            optimized_away: 0,
        }
    }

    /// Register a new resource with an initial access type.
    pub fn register_resource(
        &mut self,
        id: ResourceId,
        initial_access: AccessType,
        stage: PipelineStage,
    ) {
        self.states.insert(
            id,
            ResourceState {
                access: initial_access,
                stage,
            },
        );
    }

    /// Transition a resource to a new access type, emitting a barrier if needed.
    ///
    /// Returns `true` if a barrier was emitted, `false` if the transition
    /// was redundant (same access/stage).
    pub fn transition(
        &mut self,
        id: ResourceId,
        new_access: AccessType,
        new_stage: PipelineStage,
    ) -> bool {
        let current = self.states.get(&id).cloned().unwrap_or(ResourceState {
            access: AccessType::None,
            stage: PipelineStage::TopOfPipe,
        });

        // No barrier needed if access type and stage are the same
        if current.access == new_access && current.stage == new_stage {
            self.optimized_away += 1;
            return false;
        }

        // No barrier needed for read-to-read transitions at the same stage
        if is_read_only(current.access) && is_read_only(new_access) && current.stage == new_stage {
            self.optimized_away += 1;
            // Still update state
            self.states.insert(
                id,
                ResourceState {
                    access: new_access,
                    stage: new_stage,
                },
            );
            return false;
        }

        let barrier = BarrierDesc::new(id, current.access, new_access, current.stage, new_stage);
        self.pending.push(barrier);
        self.total_barriers += 1;

        self.states.insert(
            id,
            ResourceState {
                access: new_access,
                stage: new_stage,
            },
        );

        true
    }

    /// Drain all pending barriers, returning them.
    pub fn flush(&mut self) -> Vec<BarrierDesc> {
        std::mem::take(&mut self.pending)
    }

    /// Get the number of pending barriers.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get the total number of barriers emitted since creation.
    pub fn total_barriers(&self) -> u64 {
        self.total_barriers
    }

    /// Get the number of barriers that were optimized away.
    pub fn optimized_away(&self) -> u64 {
        self.optimized_away
    }

    /// Get the current access state of a resource.
    pub fn current_access(&self, id: ResourceId) -> Option<AccessType> {
        self.states.get(&id).map(|s| s.access)
    }

    /// Get the current pipeline stage of a resource.
    pub fn current_stage(&self, id: ResourceId) -> Option<PipelineStage> {
        self.states.get(&id).map(|s| s.stage)
    }

    /// Remove a resource from tracking.
    pub fn unregister_resource(&mut self, id: ResourceId) -> bool {
        self.states.remove(&id).is_some()
    }

    /// Get the number of tracked resources.
    pub fn resource_count(&self) -> usize {
        self.states.len()
    }

    /// Clear all tracked state and pending barriers.
    pub fn reset(&mut self) {
        self.states.clear();
        self.pending.clear();
    }

    /// Batch-transition multiple resources at once.
    pub fn batch_transition(
        &mut self,
        transitions: &[(ResourceId, AccessType, PipelineStage)],
    ) -> usize {
        let mut count = 0;
        for &(id, access, stage) in transitions {
            if self.transition(id, access, stage) {
                count += 1;
            }
        }
        count
    }
}

impl Default for BarrierManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if an access type is read-only.
fn is_read_only(access: AccessType) -> bool {
    matches!(
        access,
        AccessType::ShaderRead | AccessType::TransferSrc | AccessType::HostRead | AccessType::None
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_barrier_manager() {
        let mgr = BarrierManager::new();
        assert_eq!(mgr.resource_count(), 0);
        assert_eq!(mgr.pending_count(), 0);
        assert_eq!(mgr.total_barriers(), 0);
    }

    #[test]
    fn test_register_resource() {
        let mut mgr = BarrierManager::new();
        mgr.register_resource(ResourceId(1), AccessType::None, PipelineStage::TopOfPipe);
        assert_eq!(mgr.resource_count(), 1);
        assert_eq!(mgr.current_access(ResourceId(1)), Some(AccessType::None));
    }

    #[test]
    fn test_transition_emits_barrier() {
        let mut mgr = BarrierManager::new();
        mgr.register_resource(
            ResourceId(1),
            AccessType::ShaderWrite,
            PipelineStage::Compute,
        );
        let emitted = mgr.transition(
            ResourceId(1),
            AccessType::ShaderRead,
            PipelineStage::Compute,
        );
        assert!(emitted);
        assert_eq!(mgr.pending_count(), 1);
    }

    #[test]
    fn test_same_state_no_barrier() {
        let mut mgr = BarrierManager::new();
        mgr.register_resource(
            ResourceId(1),
            AccessType::ShaderRead,
            PipelineStage::Compute,
        );
        let emitted = mgr.transition(
            ResourceId(1),
            AccessType::ShaderRead,
            PipelineStage::Compute,
        );
        assert!(!emitted);
        assert_eq!(mgr.pending_count(), 0);
        assert_eq!(mgr.optimized_away(), 1);
    }

    #[test]
    fn test_read_to_read_same_stage_no_barrier() {
        let mut mgr = BarrierManager::new();
        mgr.register_resource(
            ResourceId(1),
            AccessType::ShaderRead,
            PipelineStage::Compute,
        );
        let emitted = mgr.transition(
            ResourceId(1),
            AccessType::TransferSrc,
            PipelineStage::Compute,
        );
        assert!(!emitted);
    }

    #[test]
    fn test_flush_clears_pending() {
        let mut mgr = BarrierManager::new();
        mgr.register_resource(ResourceId(1), AccessType::None, PipelineStage::TopOfPipe);
        mgr.transition(
            ResourceId(1),
            AccessType::ShaderWrite,
            PipelineStage::Compute,
        );
        let barriers = mgr.flush();
        assert_eq!(barriers.len(), 1);
        assert_eq!(mgr.pending_count(), 0);
    }

    #[test]
    fn test_barrier_desc_raw_hazard() {
        let desc = BarrierDesc::new(
            ResourceId(1),
            AccessType::ShaderWrite,
            AccessType::ShaderRead,
            PipelineStage::Compute,
            PipelineStage::Compute,
        );
        assert!(desc.is_raw_hazard());
        assert!(!desc.is_waw_hazard());
        assert!(!desc.is_war_hazard());
    }

    #[test]
    fn test_barrier_desc_waw_hazard() {
        let desc = BarrierDesc::new(
            ResourceId(1),
            AccessType::ShaderWrite,
            AccessType::TransferDst,
            PipelineStage::Compute,
            PipelineStage::Transfer,
        );
        assert!(desc.is_waw_hazard());
    }

    #[test]
    fn test_barrier_desc_war_hazard() {
        let desc = BarrierDesc::new(
            ResourceId(1),
            AccessType::ShaderRead,
            AccessType::ShaderWrite,
            PipelineStage::Compute,
            PipelineStage::Compute,
        );
        assert!(desc.is_war_hazard());
    }

    #[test]
    fn test_unregister_resource() {
        let mut mgr = BarrierManager::new();
        mgr.register_resource(ResourceId(1), AccessType::None, PipelineStage::TopOfPipe);
        assert!(mgr.unregister_resource(ResourceId(1)));
        assert!(!mgr.unregister_resource(ResourceId(1)));
        assert_eq!(mgr.resource_count(), 0);
    }

    #[test]
    fn test_batch_transition() {
        let mut mgr = BarrierManager::new();
        mgr.register_resource(ResourceId(1), AccessType::None, PipelineStage::TopOfPipe);
        mgr.register_resource(ResourceId(2), AccessType::None, PipelineStage::TopOfPipe);
        let count = mgr.batch_transition(&[
            (
                ResourceId(1),
                AccessType::ShaderWrite,
                PipelineStage::Compute,
            ),
            (
                ResourceId(2),
                AccessType::TransferDst,
                PipelineStage::Transfer,
            ),
        ]);
        assert_eq!(count, 2);
        assert_eq!(mgr.pending_count(), 2);
    }

    #[test]
    fn test_reset() {
        let mut mgr = BarrierManager::new();
        mgr.register_resource(ResourceId(1), AccessType::None, PipelineStage::TopOfPipe);
        mgr.transition(
            ResourceId(1),
            AccessType::ShaderWrite,
            PipelineStage::Compute,
        );
        mgr.reset();
        assert_eq!(mgr.resource_count(), 0);
        assert_eq!(mgr.pending_count(), 0);
    }

    #[test]
    fn test_transition_unregistered_resource() {
        let mut mgr = BarrierManager::new();
        let emitted = mgr.transition(
            ResourceId(99),
            AccessType::ShaderRead,
            PipelineStage::Compute,
        );
        assert!(emitted);
        assert_eq!(mgr.resource_count(), 1);
    }

    #[test]
    fn test_display_access_type() {
        assert_eq!(format!("{}", AccessType::ShaderWrite), "ShaderWrite");
        assert_eq!(format!("{}", AccessType::HostRead), "HostRead");
    }

    #[test]
    fn test_display_pipeline_stage() {
        assert_eq!(format!("{}", PipelineStage::Compute), "Compute");
        assert_eq!(format!("{}", PipelineStage::BottomOfPipe), "BottomOfPipe");
    }
}
