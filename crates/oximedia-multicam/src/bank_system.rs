//! Multicam bank switching system.
//!
//! Manages camera banks with configurable layouts and slot assignments.

#![allow(dead_code)]

/// Layout describing how many camera slots a bank contains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BankLayout {
    /// 2-camera layout.
    TwoUp,
    /// 3-camera layout.
    ThreeUp,
    /// 4-camera layout (quad split).
    FourUp,
    /// 9-camera layout (3×3).
    NineUp,
    /// 16-camera layout (4×4).
    SixteenUp,
}

impl BankLayout {
    /// Returns the number of camera slots this layout supports.
    #[must_use]
    pub fn camera_count(&self) -> u8 {
        match self {
            BankLayout::TwoUp => 2,
            BankLayout::ThreeUp => 3,
            BankLayout::FourUp => 4,
            BankLayout::NineUp => 9,
            BankLayout::SixteenUp => 16,
        }
    }
}

/// A single slot inside a camera bank.
#[derive(Debug, Clone)]
pub struct BankSlot {
    /// Zero-based slot identifier within the bank.
    pub slot_id: u8,
    /// Angle assigned to this slot, or `None` when empty.
    pub assigned_angle: Option<u8>,
}

impl BankSlot {
    /// Returns `true` when an angle is assigned to this slot.
    #[must_use]
    pub fn is_assigned(&self) -> bool {
        self.assigned_angle.is_some()
    }
}

/// A camera bank grouping several slots under one layout.
#[derive(Debug, Clone)]
pub struct CameraBank {
    /// Bank identifier.
    pub id: u8,
    /// Layout of this bank.
    pub layout: BankLayout,
    /// Slots belonging to this bank.
    pub slots: Vec<BankSlot>,
    /// The angle currently on-air (active cut).
    pub active_cut: u8,
}

impl CameraBank {
    /// Creates a new bank with the given `id` and `layout`.
    ///
    /// All slots are created unassigned. `active_cut` defaults to `0`.
    #[must_use]
    pub fn new(id: u8, layout: BankLayout) -> Self {
        let count = layout.camera_count();
        let slots = (0..count)
            .map(|i| BankSlot {
                slot_id: i,
                assigned_angle: None,
            })
            .collect();
        Self {
            id,
            layout,
            slots,
            active_cut: 0,
        }
    }

    /// Assigns `angle_id` to the slot with `slot_id`.
    ///
    /// Does nothing if the slot does not exist.
    pub fn assign_angle(&mut self, slot_id: u8, angle_id: u8) {
        if let Some(slot) = self.slots.iter_mut().find(|s| s.slot_id == slot_id) {
            slot.assigned_angle = Some(angle_id);
        }
    }

    /// Sets `angle_id` as the currently active cut.
    pub fn cut_to(&mut self, angle_id: u8) {
        self.active_cut = angle_id;
    }

    /// Returns `true` when `angle_id` is assigned to any slot in this bank.
    #[must_use]
    pub fn is_angle_in_bank(&self, angle_id: u8) -> bool {
        self.slots
            .iter()
            .any(|s| s.assigned_angle == Some(angle_id))
    }

    /// Returns the total number of slots in this bank.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- BankLayout ---

    #[test]
    fn test_two_up_count() {
        assert_eq!(BankLayout::TwoUp.camera_count(), 2);
    }

    #[test]
    fn test_three_up_count() {
        assert_eq!(BankLayout::ThreeUp.camera_count(), 3);
    }

    #[test]
    fn test_four_up_count() {
        assert_eq!(BankLayout::FourUp.camera_count(), 4);
    }

    #[test]
    fn test_nine_up_count() {
        assert_eq!(BankLayout::NineUp.camera_count(), 9);
    }

    #[test]
    fn test_sixteen_up_count() {
        assert_eq!(BankLayout::SixteenUp.camera_count(), 16);
    }

    // --- BankSlot ---

    #[test]
    fn test_slot_is_assigned_false() {
        let slot = BankSlot {
            slot_id: 0,
            assigned_angle: None,
        };
        assert!(!slot.is_assigned());
    }

    #[test]
    fn test_slot_is_assigned_true() {
        let slot = BankSlot {
            slot_id: 0,
            assigned_angle: Some(3),
        };
        assert!(slot.is_assigned());
    }

    // --- CameraBank ---

    #[test]
    fn test_new_bank_slot_count() {
        let bank = CameraBank::new(1, BankLayout::FourUp);
        assert_eq!(bank.slot_count(), 4);
    }

    #[test]
    fn test_new_bank_slots_unassigned() {
        let bank = CameraBank::new(0, BankLayout::TwoUp);
        assert!(bank.slots.iter().all(|s| !s.is_assigned()));
    }

    #[test]
    fn test_assign_angle_success() {
        let mut bank = CameraBank::new(0, BankLayout::FourUp);
        bank.assign_angle(2, 7);
        let slot = bank
            .slots
            .iter()
            .find(|s| s.slot_id == 2)
            .expect("multicam test operation should succeed");
        assert_eq!(slot.assigned_angle, Some(7));
    }

    #[test]
    fn test_assign_angle_nonexistent_slot_ignored() {
        let mut bank = CameraBank::new(0, BankLayout::TwoUp);
        // slot 5 does not exist for TwoUp
        bank.assign_angle(5, 99);
        assert!(bank.slots.iter().all(|s| s.assigned_angle.is_none()));
    }

    #[test]
    fn test_cut_to_updates_active_cut() {
        let mut bank = CameraBank::new(0, BankLayout::ThreeUp);
        bank.cut_to(2);
        assert_eq!(bank.active_cut, 2);
    }

    #[test]
    fn test_is_angle_in_bank_true() {
        let mut bank = CameraBank::new(0, BankLayout::FourUp);
        bank.assign_angle(1, 5);
        assert!(bank.is_angle_in_bank(5));
    }

    #[test]
    fn test_is_angle_in_bank_false() {
        let bank = CameraBank::new(0, BankLayout::FourUp);
        assert!(!bank.is_angle_in_bank(5));
    }
}
