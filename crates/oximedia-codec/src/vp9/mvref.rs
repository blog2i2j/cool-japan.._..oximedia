//! VP9 Motion vector reference building.
//!
//! This module provides motion vector reference candidate finding and
//! motion vector stack building for VP9 decoding.
//!
//! VP9 uses spatial and temporal motion vector prediction to derive
//! reference motion vectors for inter prediction modes.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::manual_div_ceil)]

use super::inter::{InterMode, MAX_MV_REF_CANDIDATES};
use super::mv::{MotionVector, MvRefType, MV_MAX, MV_MIN};
use super::partition::BlockSize;

/// Maximum number of MV reference candidates.
pub const MAX_REF_MV_STACK_SIZE: usize = 8;

/// Number of spatial neighbor positions to check.
pub const NUM_SPATIAL_NEIGHBORS: usize = 8;

/// Weight for spatial MV candidates.
pub const SPATIAL_WEIGHT: u8 = 2;

/// Weight for temporal MV candidates.
pub const TEMPORAL_WEIGHT: u8 = 1;

/// Motion vector reference candidate.
///
/// Represents a candidate motion vector for the reference list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MvRefCandidate {
    /// The motion vector.
    pub mv: MotionVector,
    /// Reference frame type.
    pub ref_frame: MvRefType,
    /// Weight for this candidate (higher = more preferred).
    pub weight: u8,
}

impl MvRefCandidate {
    /// Creates a new motion vector reference candidate.
    #[must_use]
    pub const fn new(mv: MotionVector, ref_frame: MvRefType, weight: u8) -> Self {
        Self {
            mv,
            ref_frame,
            weight,
        }
    }

    /// Creates a zero motion vector candidate.
    #[must_use]
    pub const fn zero(ref_frame: MvRefType) -> Self {
        Self {
            mv: MotionVector::zero(),
            ref_frame,
            weight: 0,
        }
    }

    /// Returns true if this candidate has the same motion vector.
    #[must_use]
    pub const fn same_mv(&self, other: &Self) -> bool {
        self.mv.row == other.mv.row && self.mv.col == other.mv.col
    }

    /// Returns true if this candidate is for the same reference frame.
    #[must_use]
    pub const fn same_ref(&self, other: &Self) -> bool {
        self.ref_frame as u8 == other.ref_frame as u8
    }
}

/// Motion vector reference stack.
///
/// Holds motion vector candidates ordered by weight/preference.
#[derive(Clone, Debug, Default)]
pub struct MvRefStack {
    /// Candidate motion vectors.
    candidates: [MvRefCandidate; MAX_REF_MV_STACK_SIZE],
    /// Number of valid candidates.
    count: usize,
    /// Target reference frame.
    ref_frame: MvRefType,
}

impl MvRefStack {
    /// Creates a new empty MV reference stack.
    #[must_use]
    pub const fn new(ref_frame: MvRefType) -> Self {
        Self {
            candidates: [MvRefCandidate::zero(MvRefType::Intra); MAX_REF_MV_STACK_SIZE],
            count: 0,
            ref_frame,
        }
    }

    /// Clears the stack.
    pub fn clear(&mut self) {
        self.count = 0;
    }

    /// Returns the number of candidates.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the stack is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns true if the stack is full.
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.count >= MAX_REF_MV_STACK_SIZE
    }

    /// Adds a candidate to the stack.
    ///
    /// If a candidate with the same motion vector exists, increases its weight.
    /// Otherwise, adds a new candidate if there's room.
    pub fn add(&mut self, candidate: MvRefCandidate) {
        // Check if this MV already exists
        for i in 0..self.count {
            if self.candidates[i].same_mv(&candidate) && self.candidates[i].same_ref(&candidate) {
                self.candidates[i].weight =
                    self.candidates[i].weight.saturating_add(candidate.weight);
                return;
            }
        }

        // Add new candidate if there's room
        if self.count < MAX_REF_MV_STACK_SIZE {
            self.candidates[self.count] = candidate;
            self.count += 1;
        }
    }

    /// Sorts candidates by weight (descending).
    pub fn sort_by_weight(&mut self) {
        // Simple insertion sort for small array
        for i in 1..self.count {
            let key = self.candidates[i];
            let mut j = i;
            while j > 0 && self.candidates[j - 1].weight < key.weight {
                self.candidates[j] = self.candidates[j - 1];
                j -= 1;
            }
            self.candidates[j] = key;
        }
    }

    /// Gets a candidate by index.
    #[must_use]
    pub const fn get(&self, index: usize) -> Option<&MvRefCandidate> {
        if index < self.count {
            Some(&self.candidates[index])
        } else {
            None
        }
    }

    /// Returns the nearest motion vector.
    #[must_use]
    pub fn nearest_mv(&self) -> MotionVector {
        if self.count > 0 {
            self.candidates[0].mv
        } else {
            MotionVector::zero()
        }
    }

    /// Returns the near motion vector.
    #[must_use]
    pub fn near_mv(&self) -> MotionVector {
        if self.count > 1 {
            self.candidates[1].mv
        } else if self.count > 0 {
            self.candidates[0].mv
        } else {
            MotionVector::zero()
        }
    }

    /// Returns the best reference motion vectors.
    #[must_use]
    pub fn best_ref_mvs(&self) -> [MotionVector; MAX_MV_REF_CANDIDATES] {
        [self.nearest_mv(), self.near_mv()]
    }
}

/// Spatial neighbor position relative to current block.
#[derive(Clone, Copy, Debug)]
pub struct NeighborPosition {
    /// Row offset in 4x4 units.
    pub row_offset: i32,
    /// Column offset in 4x4 units.
    pub col_offset: i32,
}

impl NeighborPosition {
    /// Creates a new neighbor position.
    #[must_use]
    pub const fn new(row_offset: i32, col_offset: i32) -> Self {
        Self {
            row_offset,
            col_offset,
        }
    }
}

/// Spatial neighbor positions for MV reference search.
///
/// These are checked in order to find motion vector candidates.
pub static SPATIAL_NEIGHBORS: [NeighborPosition; NUM_SPATIAL_NEIGHBORS] = [
    NeighborPosition {
        row_offset: 0,
        col_offset: -1,
    }, // Left
    NeighborPosition {
        row_offset: -1,
        col_offset: 0,
    }, // Above
    NeighborPosition {
        row_offset: -1,
        col_offset: -1,
    }, // Above-left
    NeighborPosition {
        row_offset: 0,
        col_offset: -2,
    }, // Left-left
    NeighborPosition {
        row_offset: -2,
        col_offset: 0,
    }, // Above-above
    NeighborPosition {
        row_offset: -1,
        col_offset: 1,
    }, // Above-right
    NeighborPosition {
        row_offset: -2,
        col_offset: -1,
    }, // Above-above-left
    NeighborPosition {
        row_offset: -1,
        col_offset: -2,
    }, // Above-left-left
];

/// Block mode information for a decoded block.
///
/// This is used to store mode and motion vector information for
/// previously decoded blocks, enabling spatial MV prediction.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockModeInfo {
    /// Reference frame types (up to 2 for compound).
    pub ref_frames: [MvRefType; 2],
    /// Motion vectors (up to 2 for compound).
    pub mvs: [MotionVector; 2],
    /// Inter prediction mode.
    pub mode: InterMode,
    /// Block size.
    pub block_size: BlockSize,
    /// Whether this block uses inter prediction.
    pub is_inter: bool,
    /// Whether this block uses compound prediction.
    pub is_compound: bool,
}

impl BlockModeInfo {
    /// Creates a new block mode info for intra prediction.
    #[must_use]
    pub const fn intra() -> Self {
        Self {
            ref_frames: [MvRefType::Intra, MvRefType::Intra],
            mvs: [MotionVector::zero(), MotionVector::zero()],
            mode: InterMode::NearestMv,
            block_size: BlockSize::Block4x4,
            is_inter: false,
            is_compound: false,
        }
    }

    /// Creates a new block mode info for single reference inter prediction.
    #[must_use]
    pub const fn inter_single(ref_frame: MvRefType, mv: MotionVector, mode: InterMode) -> Self {
        Self {
            ref_frames: [ref_frame, MvRefType::Intra],
            mvs: [mv, MotionVector::zero()],
            mode,
            block_size: BlockSize::Block4x4,
            is_inter: true,
            is_compound: false,
        }
    }

    /// Creates a new block mode info for compound inter prediction.
    #[must_use]
    pub const fn inter_compound(
        ref0: MvRefType,
        ref1: MvRefType,
        mv0: MotionVector,
        mv1: MotionVector,
    ) -> Self {
        Self {
            ref_frames: [ref0, ref1],
            mvs: [mv0, mv1],
            mode: InterMode::NearestMv,
            block_size: BlockSize::Block4x4,
            is_inter: true,
            is_compound: true,
        }
    }

    /// Returns the motion vector for a given reference frame, if any.
    #[must_use]
    pub const fn mv_for_ref(&self, ref_frame: MvRefType) -> Option<MotionVector> {
        if self.ref_frames[0] as u8 == ref_frame as u8 {
            Some(self.mvs[0])
        } else if self.ref_frames[1] as u8 == ref_frame as u8 {
            Some(self.mvs[1])
        } else {
            None
        }
    }
}

/// Mode info grid for the frame.
///
/// Stores block mode information for all 4x4 blocks in the frame.
#[derive(Clone, Debug)]
pub struct ModeInfoGrid {
    /// Mode info for each 4x4 block.
    data: Vec<BlockModeInfo>,
    /// Grid width in 4x4 units.
    mi_cols: usize,
    /// Grid height in 4x4 units.
    mi_rows: usize,
}

impl Default for ModeInfoGrid {
    fn default() -> Self {
        Self::new()
    }
}

impl ModeInfoGrid {
    /// Creates a new empty mode info grid.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            mi_cols: 0,
            mi_rows: 0,
        }
    }

    /// Allocates the grid for the given frame size.
    pub fn allocate(&mut self, width: u32, height: u32) {
        self.mi_cols = ((width as usize) + 3) / 4;
        self.mi_rows = ((height as usize) + 3) / 4;
        self.data
            .resize(self.mi_cols * self.mi_rows, BlockModeInfo::intra());
    }

    /// Clears the grid.
    pub fn clear(&mut self) {
        self.data.fill(BlockModeInfo::intra());
    }

    /// Gets the mode info for a block.
    #[must_use]
    pub fn get(&self, mi_row: usize, mi_col: usize) -> Option<&BlockModeInfo> {
        if mi_row < self.mi_rows && mi_col < self.mi_cols {
            self.data.get(mi_row * self.mi_cols + mi_col)
        } else {
            None
        }
    }

    /// Sets the mode info for a block.
    pub fn set(&mut self, mi_row: usize, mi_col: usize, info: BlockModeInfo) {
        if mi_row < self.mi_rows && mi_col < self.mi_cols {
            self.data[mi_row * self.mi_cols + mi_col] = info;
        }
    }

    /// Fills a block region with mode info.
    pub fn fill_block(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        block_size: BlockSize,
        info: BlockModeInfo,
    ) {
        let mi_width = block_size.width_mi();
        let mi_height = block_size.height_mi();

        for row in mi_row..mi_row + mi_height {
            for col in mi_col..mi_col + mi_width {
                self.set(row, col, info);
            }
        }
    }

    /// Returns the grid width in 4x4 units.
    #[must_use]
    pub const fn mi_cols(&self) -> usize {
        self.mi_cols
    }

    /// Returns the grid height in 4x4 units.
    #[must_use]
    pub const fn mi_rows(&self) -> usize {
        self.mi_rows
    }
}

/// Context for finding motion vector references.
#[derive(Clone, Debug)]
pub struct MvRefContext {
    /// Mode info grid.
    pub mode_info: ModeInfoGrid,
    /// Current block row in 4x4 units.
    pub mi_row: usize,
    /// Current block column in 4x4 units.
    pub mi_col: usize,
    /// Current block size.
    pub block_size: BlockSize,
    /// Target reference frame.
    pub ref_frame: MvRefType,
}

impl Default for MvRefContext {
    fn default() -> Self {
        Self::new()
    }
}

impl MvRefContext {
    /// Creates a new MV reference context.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mode_info: ModeInfoGrid::new(),
            mi_row: 0,
            mi_col: 0,
            block_size: BlockSize::Block4x4,
            ref_frame: MvRefType::Last,
        }
    }

    /// Sets the current block position.
    pub fn set_position(&mut self, mi_row: usize, mi_col: usize, block_size: BlockSize) {
        self.mi_row = mi_row;
        self.mi_col = mi_col;
        self.block_size = block_size;
    }

    /// Returns true if a neighbor position is valid.
    #[must_use]
    pub fn is_valid_neighbor(&self, neighbor: &NeighborPosition) -> bool {
        let row = self.mi_row as i32 + neighbor.row_offset;
        let col = self.mi_col as i32 + neighbor.col_offset;

        row >= 0
            && col >= 0
            && (row as usize) < self.mode_info.mi_rows()
            && (col as usize) < self.mode_info.mi_cols()
    }

    /// Gets the mode info for a neighbor position.
    #[must_use]
    pub fn get_neighbor(&self, neighbor: &NeighborPosition) -> Option<&BlockModeInfo> {
        let row = self.mi_row as i32 + neighbor.row_offset;
        let col = self.mi_col as i32 + neighbor.col_offset;

        if row >= 0 && col >= 0 {
            self.mode_info.get(row as usize, col as usize)
        } else {
            None
        }
    }
}

/// Finds motion vector reference candidates for a block.
///
/// This function searches spatial neighbors to build a list of
/// motion vector candidates for the specified reference frame.
///
/// # Arguments
///
/// * `ctx` - MV reference context with mode info grid
/// * `ref_frame` - Target reference frame
/// * `stack` - Output MV reference stack
pub fn find_mv_refs(ctx: &MvRefContext, ref_frame: MvRefType, stack: &mut MvRefStack) {
    stack.clear();

    // Search spatial neighbors
    for neighbor in &SPATIAL_NEIGHBORS {
        if !ctx.is_valid_neighbor(neighbor) {
            continue;
        }

        if let Some(info) = ctx.get_neighbor(neighbor) {
            if !info.is_inter {
                continue;
            }

            // Check if neighbor references the same frame
            if let Some(mv) = info.mv_for_ref(ref_frame) {
                let candidate = MvRefCandidate::new(mv, ref_frame, SPATIAL_WEIGHT);
                stack.add(candidate);

                if stack.len() >= MAX_MV_REF_CANDIDATES {
                    break;
                }
            }
        }
    }

    // Sort by weight
    stack.sort_by_weight();
}

/// Finds the best reference motion vectors for inter prediction.
///
/// # Arguments
///
/// * `ctx` - MV reference context
/// * `ref_frame` - Target reference frame
///
/// # Returns
///
/// A pair of motion vectors [nearest, near].
#[must_use]
pub fn find_best_ref_mvs(ctx: &MvRefContext, ref_frame: MvRefType) -> [MotionVector; 2] {
    let mut stack = MvRefStack::new(ref_frame);
    find_mv_refs(ctx, ref_frame, &mut stack);
    stack.best_ref_mvs()
}

/// Clamps a motion vector to valid range for a block position.
///
/// # Arguments
///
/// * `mv` - Motion vector to clamp
/// * `mi_row` - Block row in 4x4 units
/// * `mi_col` - Block column in 4x4 units
/// * `block_size` - Block size
/// * `frame_width` - Frame width in pixels
/// * `frame_height` - Frame height in pixels
///
/// # Returns
///
/// Clamped motion vector.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn clamp_mv(
    mv: MotionVector,
    mi_row: usize,
    mi_col: usize,
    block_size: BlockSize,
    frame_width: usize,
    frame_height: usize,
) -> MotionVector {
    // Calculate block position in pixels (1/8 pel)
    let block_x = (mi_col * 4) << 3;
    let block_y = (mi_row * 4) << 3;

    // Calculate maximum extent based on frame size
    let max_x = ((frame_width - block_size.width()) << 3) as i16;
    let max_y = ((frame_height - block_size.height()) << 3) as i16;

    // Calculate minimum (can go into negative for border extension)
    let min_x = -(block_x as i16) - 128;
    let min_y = -(block_y as i16) - 128;

    // Clamp the motion vector
    MotionVector::new(
        mv.row.clamp(min_y.max(MV_MIN), max_y.min(MV_MAX)),
        mv.col.clamp(min_x.max(MV_MIN), max_x.min(MV_MAX)),
    )
}

/// Applies motion vector rounding for prediction.
///
/// VP9 may round motion vectors to reduce computational complexity.
///
/// # Arguments
///
/// * `mv` - Motion vector to round
/// * `allow_hp` - Whether high-precision (1/8 pel) is allowed
///
/// # Returns
///
/// Rounded motion vector (to 1/4 or 1/8 pel precision).
#[must_use]
pub fn round_mv(mv: MotionVector, allow_hp: bool) -> MotionVector {
    if allow_hp {
        // High precision: no rounding needed
        mv
    } else {
        // Quarter-pel precision: round to nearest 1/4 pixel
        let round_row = if mv.row < 0 {
            (mv.row - 1) & !1
        } else {
            (mv.row + 1) & !1
        };
        let round_col = if mv.col < 0 {
            (mv.col - 1) & !1
        } else {
            (mv.col + 1) & !1
        };
        MotionVector::new(round_row, round_col)
    }
}

/// Motion vector context for probability selection.
#[derive(Clone, Copy, Debug, Default)]
pub struct MvPredContext {
    /// Number of neighbors with NEAREST mode.
    pub nearest_count: u8,
    /// Number of neighbors with NEAR mode.
    pub near_count: u8,
    /// Number of neighbors with ZERO mode.
    pub zero_count: u8,
    /// Number of neighbors with NEW mode.
    pub new_count: u8,
    /// Total number of inter neighbors.
    pub inter_count: u8,
}

impl MvPredContext {
    /// Creates a new MV prediction context.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nearest_count: 0,
            near_count: 0,
            zero_count: 0,
            new_count: 0,
            inter_count: 0,
        }
    }

    /// Updates context from a neighbor block.
    pub fn add_neighbor(&mut self, info: &BlockModeInfo) {
        if info.is_inter {
            self.inter_count = self.inter_count.saturating_add(1);
            match info.mode {
                InterMode::NearestMv => self.nearest_count = self.nearest_count.saturating_add(1),
                InterMode::NearMv => self.near_count = self.near_count.saturating_add(1),
                InterMode::ZeroMv => self.zero_count = self.zero_count.saturating_add(1),
                InterMode::NewMv => self.new_count = self.new_count.saturating_add(1),
            }
        }
    }

    /// Returns the inter mode context index.
    #[must_use]
    pub const fn mode_context(&self) -> usize {
        // Context based on number of inter neighbors and mode distribution
        match self.inter_count {
            0 => 0,
            1 => {
                if self.new_count > 0 {
                    3
                } else {
                    1
                }
            }
            _ => {
                if self.new_count > 1 {
                    5
                } else if self.new_count > 0 {
                    4
                } else {
                    2
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mv_ref_candidate() {
        let mv = MotionVector::new(10, 20);
        let candidate = MvRefCandidate::new(mv, MvRefType::Last, 5);

        assert_eq!(candidate.mv.row, 10);
        assert_eq!(candidate.mv.col, 20);
        assert_eq!(candidate.weight, 5);
    }

    #[test]
    fn test_mv_ref_candidate_same_mv() {
        let c1 = MvRefCandidate::new(MotionVector::new(10, 20), MvRefType::Last, 1);
        let c2 = MvRefCandidate::new(MotionVector::new(10, 20), MvRefType::Last, 2);
        let c3 = MvRefCandidate::new(MotionVector::new(10, 30), MvRefType::Last, 1);

        assert!(c1.same_mv(&c2));
        assert!(!c1.same_mv(&c3));
    }

    #[test]
    fn test_mv_ref_stack_new() {
        let stack = MvRefStack::new(MvRefType::Last);
        assert!(stack.is_empty());
        assert!(!stack.is_full());
        assert_eq!(stack.len(), 0);
    }

    #[test]
    fn test_mv_ref_stack_add() {
        let mut stack = MvRefStack::new(MvRefType::Last);

        let c1 = MvRefCandidate::new(MotionVector::new(10, 20), MvRefType::Last, 1);
        stack.add(c1);

        assert_eq!(stack.len(), 1);
        assert_eq!(stack.nearest_mv(), MotionVector::new(10, 20));
    }

    #[test]
    fn test_mv_ref_stack_add_duplicate() {
        let mut stack = MvRefStack::new(MvRefType::Last);

        let c1 = MvRefCandidate::new(MotionVector::new(10, 20), MvRefType::Last, 1);
        let c2 = MvRefCandidate::new(MotionVector::new(10, 20), MvRefType::Last, 2);

        stack.add(c1);
        stack.add(c2);

        // Should merge and increase weight
        assert_eq!(stack.len(), 1);
        assert_eq!(stack.get(0).expect("get should return value").weight, 3);
    }

    #[test]
    fn test_mv_ref_stack_sort() {
        let mut stack = MvRefStack::new(MvRefType::Last);

        stack.add(MvRefCandidate::new(
            MotionVector::new(10, 20),
            MvRefType::Last,
            1,
        ));
        stack.add(MvRefCandidate::new(
            MotionVector::new(30, 40),
            MvRefType::Last,
            3,
        ));
        stack.add(MvRefCandidate::new(
            MotionVector::new(50, 60),
            MvRefType::Last,
            2,
        ));

        stack.sort_by_weight();

        assert_eq!(stack.get(0).expect("get should return value").weight, 3);
        assert_eq!(stack.get(1).expect("get should return value").weight, 2);
        assert_eq!(stack.get(2).expect("get should return value").weight, 1);
    }

    #[test]
    fn test_mv_ref_stack_best_ref_mvs() {
        let mut stack = MvRefStack::new(MvRefType::Last);

        stack.add(MvRefCandidate::new(
            MotionVector::new(10, 20),
            MvRefType::Last,
            2,
        ));
        stack.add(MvRefCandidate::new(
            MotionVector::new(30, 40),
            MvRefType::Last,
            1,
        ));

        let [nearest, near] = stack.best_ref_mvs();
        assert_eq!(nearest, MotionVector::new(10, 20));
        assert_eq!(near, MotionVector::new(30, 40));
    }

    #[test]
    fn test_block_mode_info_intra() {
        let info = BlockModeInfo::intra();
        assert!(!info.is_inter);
        assert!(!info.is_compound);
    }

    #[test]
    fn test_block_mode_info_inter_single() {
        let info = BlockModeInfo::inter_single(
            MvRefType::Last,
            MotionVector::new(10, 20),
            InterMode::NearestMv,
        );

        assert!(info.is_inter);
        assert!(!info.is_compound);
        assert_eq!(
            info.mv_for_ref(MvRefType::Last),
            Some(MotionVector::new(10, 20))
        );
        assert_eq!(info.mv_for_ref(MvRefType::Golden), None);
    }

    #[test]
    fn test_block_mode_info_inter_compound() {
        let info = BlockModeInfo::inter_compound(
            MvRefType::Last,
            MvRefType::Golden,
            MotionVector::new(10, 20),
            MotionVector::new(30, 40),
        );

        assert!(info.is_inter);
        assert!(info.is_compound);
        assert_eq!(
            info.mv_for_ref(MvRefType::Last),
            Some(MotionVector::new(10, 20))
        );
        assert_eq!(
            info.mv_for_ref(MvRefType::Golden),
            Some(MotionVector::new(30, 40))
        );
    }

    #[test]
    fn test_mode_info_grid() {
        let mut grid = ModeInfoGrid::new();
        grid.allocate(64, 64);

        assert_eq!(grid.mi_cols(), 16);
        assert_eq!(grid.mi_rows(), 16);

        let info = BlockModeInfo::inter_single(
            MvRefType::Last,
            MotionVector::new(10, 20),
            InterMode::NearestMv,
        );

        grid.set(0, 0, info);
        let retrieved = grid.get(0, 0).expect("get should return value");
        assert!(retrieved.is_inter);
    }

    #[test]
    fn test_mode_info_grid_fill_block() {
        let mut grid = ModeInfoGrid::new();
        grid.allocate(64, 64);

        let info = BlockModeInfo::inter_single(
            MvRefType::Last,
            MotionVector::new(10, 20),
            InterMode::NearestMv,
        );

        grid.fill_block(0, 0, BlockSize::Block8x8, info);

        // Should fill 2x2 area of 4x4 blocks
        assert!(grid.get(0, 0).expect("get should return value").is_inter);
        assert!(grid.get(0, 1).expect("get should return value").is_inter);
        assert!(grid.get(1, 0).expect("get should return value").is_inter);
        assert!(grid.get(1, 1).expect("get should return value").is_inter);
    }

    #[test]
    fn test_clamp_mv() {
        let mv = MotionVector::new(1000, 2000);
        let clamped = clamp_mv(mv, 0, 0, BlockSize::Block8x8, 64, 64);

        // Should be clamped to frame boundaries
        assert!(clamped.row <= (56 << 3) as i16); // 64 - 8 = 56 pixels max
        assert!(clamped.col <= (56 << 3) as i16);
    }

    #[test]
    fn test_round_mv() {
        let mv = MotionVector::new(5, 7);

        // With high precision: no change
        let hp = round_mv(mv, true);
        assert_eq!(hp.row, 5);
        assert_eq!(hp.col, 7);

        // Without high precision: round to even
        let qp = round_mv(mv, false);
        assert_eq!(qp.row & 1, 0);
        assert_eq!(qp.col & 1, 0);
    }

    #[test]
    fn test_mv_pred_context() {
        let mut ctx = MvPredContext::new();

        let nearest_info = BlockModeInfo::inter_single(
            MvRefType::Last,
            MotionVector::zero(),
            InterMode::NearestMv,
        );
        ctx.add_neighbor(&nearest_info);

        assert_eq!(ctx.inter_count, 1);
        assert_eq!(ctx.nearest_count, 1);
        assert_eq!(ctx.mode_context(), 1);
    }

    #[test]
    fn test_find_mv_refs() {
        let mut ctx = MvRefContext::new();
        ctx.mode_info.allocate(64, 64);

        // Add a neighbor with motion vector
        let neighbor_info = BlockModeInfo::inter_single(
            MvRefType::Last,
            MotionVector::new(10, 20),
            InterMode::NearestMv,
        );
        ctx.mode_info.set(0, 0, neighbor_info);

        // Set current position to (0, 1) so left neighbor is valid
        ctx.set_position(0, 1, BlockSize::Block8x8);

        let mut stack = MvRefStack::new(MvRefType::Last);
        find_mv_refs(&ctx, MvRefType::Last, &mut stack);

        assert!(!stack.is_empty());
        assert_eq!(stack.nearest_mv(), MotionVector::new(10, 20));
    }

    #[test]
    fn test_find_best_ref_mvs() {
        let mut ctx = MvRefContext::new();
        ctx.mode_info.allocate(64, 64);

        // Empty grid should return zero MVs
        ctx.set_position(8, 8, BlockSize::Block8x8);
        let [nearest, near] = find_best_ref_mvs(&ctx, MvRefType::Last);

        assert_eq!(nearest, MotionVector::zero());
        assert_eq!(near, MotionVector::zero());
    }
}
