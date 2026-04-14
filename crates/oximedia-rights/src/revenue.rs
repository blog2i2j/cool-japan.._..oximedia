//! Revenue sharing calculation.
//!
//! [`RevenueShare`] holds a set of parties with fractional shares and
//! distributes a total revenue amount proportionally.

/// A revenue-sharing ledger.
///
/// Each party is identified by a `u64` ID and has a non-negative `f32` share
/// weight.  The weights are *not* required to sum to 1.0; `distribute` always
/// normalises them automatically.
#[derive(Debug, Default)]
pub struct RevenueShare {
    parties: Vec<(u64, f32)>,
}

impl RevenueShare {
    /// Create an empty revenue share ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a party with the given share weight.
    ///
    /// If `id` already exists the weight is *replaced* with the new value.
    /// Negative weights are clamped to `0.0`.
    pub fn add_party(&mut self, id: u64, share: f32) {
        let share = share.max(0.0);
        if let Some(entry) = self.parties.iter_mut().find(|(pid, _)| *pid == id) {
            entry.1 = share;
        } else {
            self.parties.push((id, share));
        }
    }

    /// Distribute `total` revenue across all parties proportionally to their
    /// shares.
    ///
    /// Returns a `Vec<(id, amount)>` in the same order the parties were added.
    /// If the total weight is zero (no parties, or all weights are zero),
    /// every party receives `0.0`.
    pub fn distribute(&self, total: f64) -> Vec<(u64, f64)> {
        let total_weight: f64 = self.parties.iter().map(|(_, w)| f64::from(*w)).sum();

        self.parties
            .iter()
            .map(|(id, weight)| {
                let amount = if total_weight > 0.0 {
                    total * f64::from(*weight) / total_weight
                } else {
                    0.0
                };
                (*id, amount)
            })
            .collect()
    }

    /// Number of parties in the ledger.
    pub fn party_count(&self) -> usize {
        self.parties.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_distribute_equal_shares() {
        let mut rs = RevenueShare::new();
        rs.add_party(1, 1.0);
        rs.add_party(2, 1.0);
        let result = rs.distribute(100.0);
        assert_eq!(result.len(), 2);
        assert!((result[0].1 - 50.0).abs() < EPSILON);
        assert!((result[1].1 - 50.0).abs() < EPSILON);
    }

    #[test]
    fn test_distribute_unequal_shares() {
        let mut rs = RevenueShare::new();
        rs.add_party(1, 3.0);
        rs.add_party(2, 1.0);
        let result = rs.distribute(200.0);
        assert!((result[0].1 - 150.0).abs() < EPSILON);
        assert!((result[1].1 - 50.0).abs() < EPSILON);
    }

    #[test]
    fn test_distribute_zero_total() {
        let mut rs = RevenueShare::new();
        rs.add_party(1, 0.5);
        let result = rs.distribute(0.0);
        assert!((result[0].1).abs() < EPSILON);
    }

    #[test]
    fn test_distribute_no_parties() {
        let rs = RevenueShare::new();
        assert!(rs.distribute(1000.0).is_empty());
    }

    #[test]
    fn test_distribute_all_zero_weights() {
        let mut rs = RevenueShare::new();
        rs.add_party(1, 0.0);
        rs.add_party(2, 0.0);
        let result = rs.distribute(999.0);
        for (_, amt) in &result {
            assert!(amt.abs() < EPSILON);
        }
    }

    #[test]
    fn test_add_party_replaces_existing() {
        let mut rs = RevenueShare::new();
        rs.add_party(1, 1.0);
        rs.add_party(1, 3.0); // replace
        rs.add_party(2, 1.0);
        assert_eq!(rs.party_count(), 2);
        let result = rs.distribute(100.0);
        // Party 1 has weight 3, party 2 has weight 1 → 75/25
        assert!((result[0].1 - 75.0).abs() < EPSILON);
        assert!((result[1].1 - 25.0).abs() < EPSILON);
    }

    #[test]
    fn test_negative_share_clamped_to_zero() {
        let mut rs = RevenueShare::new();
        rs.add_party(1, -5.0);
        rs.add_party(2, 2.0);
        let result = rs.distribute(100.0);
        // Party 1 effectively has 0 weight
        assert!((result[0].1).abs() < EPSILON);
        assert!((result[1].1 - 100.0).abs() < EPSILON);
    }

    #[test]
    fn test_party_count_after_add_and_replace() {
        let mut rs = RevenueShare::new();
        assert_eq!(rs.party_count(), 0);
        rs.add_party(1, 1.0);
        rs.add_party(2, 1.0);
        assert_eq!(rs.party_count(), 2);
        rs.add_party(2, 3.0); // replace, not add
        assert_eq!(rs.party_count(), 2);
    }
}
