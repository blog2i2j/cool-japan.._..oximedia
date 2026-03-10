//! Reference frame management for H.264/H.265 decoded picture buffers (DPB).
//!
//! Provides [`RefFrameType`], [`RefFrame`], and [`RefFrameList`] to track
//! short-term and long-term reference pictures used during decoding.

#![allow(dead_code)]

/// Indicates whether a reference frame is short-term or long-term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefFrameType {
    /// Short-term reference: identified by frame_num / PicNum.
    ShortTerm,
    /// Long-term reference: identified by a long-term frame index.
    LongTerm,
}

impl RefFrameType {
    /// Returns `true` if this is a long-term reference.
    pub fn is_long_term(self) -> bool {
        self == Self::LongTerm
    }

    /// Returns `true` if this is a short-term reference.
    pub fn is_short_term(self) -> bool {
        self == Self::ShortTerm
    }
}

/// A reference frame entry in the decoded picture buffer.
#[derive(Debug, Clone)]
pub struct RefFrame {
    /// Picture order count (POC).
    pub poc: i32,
    /// `frame_num` for short-term or long-term frame index for long-term.
    pub frame_num: u32,
    /// Reference type.
    pub ref_type: RefFrameType,
    /// `true` when this entry is actually in use (not an empty slot).
    pub in_use: bool,
    /// Opaque frame data buffer index (into a larger buffer pool).
    pub buffer_index: usize,
}

impl RefFrame {
    /// Creates a new short-term reference frame.
    pub fn short_term(poc: i32, frame_num: u32, buffer_index: usize) -> Self {
        Self {
            poc,
            frame_num,
            ref_type: RefFrameType::ShortTerm,
            in_use: true,
            buffer_index,
        }
    }

    /// Creates a new long-term reference frame.
    pub fn long_term(poc: i32, long_term_frame_idx: u32, buffer_index: usize) -> Self {
        Self {
            poc,
            frame_num: long_term_frame_idx,
            ref_type: RefFrameType::LongTerm,
            in_use: true,
            buffer_index,
        }
    }

    /// Returns `true` when this slot is occupied by a valid reference frame.
    pub fn is_valid(&self) -> bool {
        self.in_use
    }

    /// Marks this frame as unused (removes it from the DPB logically).
    pub fn mark_unused(&mut self) {
        self.in_use = false;
    }
}

/// A list of reference frames maintained in the decoded picture buffer.
///
/// Limits itself to `max_size` entries; oldest short-term frames are evicted
/// when capacity is exceeded.
#[derive(Debug)]
pub struct RefFrameList {
    frames: Vec<RefFrame>,
    max_size: usize,
}

impl RefFrameList {
    /// Creates a new empty reference frame list with the given capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            frames: Vec::with_capacity(max_size),
            max_size,
        }
    }

    /// Adds a reference frame to the list.
    ///
    /// If the list is full, the oldest short-term reference is removed first.
    /// Returns `false` if no space could be made (all frames are long-term).
    pub fn add(&mut self, frame: RefFrame) -> bool {
        if self.frames.len() < self.max_size {
            self.frames.push(frame);
            return true;
        }
        // Try to evict oldest short-term
        if self.remove_oldest() {
            self.frames.push(frame);
            true
        } else {
            false
        }
    }

    /// Removes the oldest short-term reference frame (lowest frame_num).
    ///
    /// Returns `true` if a frame was removed.
    pub fn remove_oldest(&mut self) -> bool {
        let pos = self
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.ref_type == RefFrameType::ShortTerm && f.in_use)
            .min_by_key(|(_, f)| f.frame_num)
            .map(|(i, _)| i);

        if let Some(idx) = pos {
            self.frames.remove(idx);
            true
        } else {
            false
        }
    }

    /// Finds the reference frame whose POC is closest to `target_poc`.
    ///
    /// Returns `None` if the list is empty.
    pub fn find_closest_poc(&self, target_poc: i32) -> Option<&RefFrame> {
        self.frames
            .iter()
            .filter(|f| f.in_use)
            .min_by_key(|f| (f.poc - target_poc).unsigned_abs())
    }

    /// Returns the number of active reference frames in the list.
    pub fn len(&self) -> usize {
        self.frames.iter().filter(|f| f.in_use).count()
    }

    /// Returns `true` if the list contains no active reference frames.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over all active reference frames.
    pub fn iter(&self) -> impl Iterator<Item = &RefFrame> {
        self.frames.iter().filter(|f| f.in_use)
    }

    /// Returns the number of long-term reference frames.
    pub fn long_term_count(&self) -> usize {
        self.frames
            .iter()
            .filter(|f| f.in_use && f.ref_type == RefFrameType::LongTerm)
            .count()
    }

    /// Returns the number of short-term reference frames.
    pub fn short_term_count(&self) -> usize {
        self.frames
            .iter()
            .filter(|f| f.in_use && f.ref_type == RefFrameType::ShortTerm)
            .count()
    }

    /// Clears all reference frames from the list (used on IDR).
    pub fn clear(&mut self) {
        self.frames.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ref_frame_type_is_long_term() {
        assert!(RefFrameType::LongTerm.is_long_term());
        assert!(!RefFrameType::ShortTerm.is_long_term());
    }

    #[test]
    fn test_ref_frame_type_is_short_term() {
        assert!(RefFrameType::ShortTerm.is_short_term());
        assert!(!RefFrameType::LongTerm.is_short_term());
    }

    #[test]
    fn test_ref_frame_short_term_constructor() {
        let f = RefFrame::short_term(10, 3, 0);
        assert_eq!(f.poc, 10);
        assert_eq!(f.frame_num, 3);
        assert!(f.is_valid());
        assert_eq!(f.ref_type, RefFrameType::ShortTerm);
    }

    #[test]
    fn test_ref_frame_long_term_constructor() {
        let f = RefFrame::long_term(20, 1, 5);
        assert_eq!(f.ref_type, RefFrameType::LongTerm);
        assert_eq!(f.frame_num, 1);
        assert!(f.is_valid());
    }

    #[test]
    fn test_ref_frame_mark_unused() {
        let mut f = RefFrame::short_term(0, 0, 0);
        f.mark_unused();
        assert!(!f.is_valid());
    }

    #[test]
    fn test_ref_frame_list_add_within_capacity() {
        let mut list = RefFrameList::new(4);
        assert!(list.add(RefFrame::short_term(0, 0, 0)));
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_ref_frame_list_is_empty_initially() {
        let list = RefFrameList::new(4);
        assert!(list.is_empty());
    }

    #[test]
    fn test_ref_frame_list_remove_oldest_evicts_min_frame_num() {
        let mut list = RefFrameList::new(10);
        list.add(RefFrame::short_term(4, 4, 0));
        list.add(RefFrame::short_term(2, 2, 1));
        list.add(RefFrame::short_term(6, 6, 2));
        assert!(list.remove_oldest());
        // frame_num=2 should be gone; remaining: 4, 6
        assert_eq!(list.len(), 2);
        assert!(list.iter().all(|f| f.frame_num != 2));
    }

    #[test]
    fn test_ref_frame_list_add_evicts_oldest_when_full() {
        let mut list = RefFrameList::new(2);
        list.add(RefFrame::short_term(0, 0, 0));
        list.add(RefFrame::short_term(2, 2, 1));
        // Full → should evict frame_num=0 and add new
        let ok = list.add(RefFrame::short_term(4, 4, 2));
        assert!(ok);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_ref_frame_list_find_closest_poc() {
        let mut list = RefFrameList::new(4);
        list.add(RefFrame::short_term(0, 0, 0));
        list.add(RefFrame::short_term(10, 1, 1));
        list.add(RefFrame::short_term(20, 2, 2));
        let closest = list.find_closest_poc(12).expect("should succeed");
        assert_eq!(closest.poc, 10);
    }

    #[test]
    fn test_ref_frame_list_find_closest_poc_empty_returns_none() {
        let list = RefFrameList::new(4);
        assert!(list.find_closest_poc(0).is_none());
    }

    #[test]
    fn test_ref_frame_list_long_term_count() {
        let mut list = RefFrameList::new(4);
        list.add(RefFrame::short_term(0, 0, 0));
        list.add(RefFrame::long_term(10, 0, 1));
        assert_eq!(list.long_term_count(), 1);
        assert_eq!(list.short_term_count(), 1);
    }

    #[test]
    fn test_ref_frame_list_clear() {
        let mut list = RefFrameList::new(4);
        list.add(RefFrame::short_term(0, 0, 0));
        list.add(RefFrame::short_term(2, 1, 1));
        list.clear();
        assert!(list.is_empty());
    }

    #[test]
    fn test_ref_frame_list_iter_only_active() {
        let mut list = RefFrameList::new(4);
        let mut f = RefFrame::short_term(0, 0, 0);
        f.mark_unused();
        list.frames.push(f);
        list.add(RefFrame::short_term(2, 1, 1));
        assert_eq!(list.iter().count(), 1);
    }
}
