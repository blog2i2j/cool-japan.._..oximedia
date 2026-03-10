//! Clip rating and flagging system.

pub mod batch;

use std::collections::HashMap;

/// Star rating for a clip.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClipRating {
    /// Rejected clip.
    Reject,
    /// Unrated (default state).
    Unrated,
    /// Marginal / pickup.
    Pickup,
    /// Acceptable.
    Ok,
    /// Good quality clip.
    Good,
    /// Best available - use this.
    Excellent,
}

impl ClipRating {
    /// Number of stars (0–5) corresponding to the rating.
    #[allow(dead_code)]
    #[must_use]
    pub const fn stars(self) -> u8 {
        match self {
            Self::Reject => 0,
            Self::Unrated => 0,
            Self::Pickup => 1,
            Self::Ok => 2,
            Self::Good => 3,
            Self::Excellent => 5,
        }
    }

    /// Construct a rating from a star count (0–5).
    #[allow(dead_code)]
    #[must_use]
    pub const fn from_stars(n: u8) -> Self {
        match n {
            5 => Self::Excellent,
            4 | 3 => Self::Good,
            2 => Self::Ok,
            1 => Self::Pickup,
            _ => Self::Unrated,
        }
    }

    /// Human-readable label.
    #[allow(dead_code)]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Reject => "Reject",
            Self::Unrated => "Unrated",
            Self::Pickup => "Pickup",
            Self::Ok => "OK",
            Self::Good => "Good",
            Self::Excellent => "Excellent",
        }
    }
}

/// Color flag for quick visual identification.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClipFlag {
    /// No flag set.
    None,
    /// Red flag (commonly: reject / problem).
    Red,
    /// Yellow flag (commonly: review needed).
    Yellow,
    /// Green flag (commonly: approved).
    Green,
    /// Blue flag (commonly: informational).
    Blue,
    /// Purple flag (commonly: special / favorite).
    Purple,
}

impl ClipFlag {
    /// Human-readable name.
    #[allow(dead_code)]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Red => "Red",
            Self::Yellow => "Yellow",
            Self::Green => "Green",
            Self::Blue => "Blue",
            Self::Purple => "Purple",
        }
    }
}

/// A rating entry for a single clip.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct RatingEntry {
    /// Clip identifier.
    pub clip_id: u64,
    /// Star rating.
    pub rating: ClipRating,
    /// Color flag.
    pub flag: ClipFlag,
    /// Free-text notes.
    pub notes: String,
    /// Timestamp of when this rating was set (milliseconds since epoch).
    pub rated_at_ms: u64,
}

impl RatingEntry {
    /// Create a new unrated, unflagged entry.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(clip_id: u64) -> Self {
        Self {
            clip_id,
            rating: ClipRating::Unrated,
            flag: ClipFlag::None,
            notes: String::new(),
            rated_at_ms: 0,
        }
    }
}

/// In-memory database of clip ratings.
#[allow(dead_code)]
pub struct RatingDatabase {
    entries: HashMap<u64, RatingEntry>,
}

impl RatingDatabase {
    /// Create a new empty database.
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Set or update the rating for a clip.
    #[allow(dead_code)]
    pub fn set_rating(&mut self, clip_id: u64, rating: ClipRating) {
        self.entries
            .entry(clip_id)
            .or_insert_with(|| RatingEntry::new(clip_id))
            .rating = rating;
    }

    /// Set or update the flag for a clip.
    #[allow(dead_code)]
    pub fn set_flag(&mut self, clip_id: u64, flag: ClipFlag) {
        self.entries
            .entry(clip_id)
            .or_insert_with(|| RatingEntry::new(clip_id))
            .flag = flag;
    }

    /// Set free-text notes for a clip.
    #[allow(dead_code)]
    pub fn set_notes(&mut self, clip_id: u64, notes: impl Into<String>) {
        self.entries
            .entry(clip_id)
            .or_insert_with(|| RatingEntry::new(clip_id))
            .notes = notes.into();
    }

    /// Get the rating entry for a clip, if it exists.
    #[allow(dead_code)]
    #[must_use]
    pub fn get(&self, clip_id: u64) -> Option<&RatingEntry> {
        self.entries.get(&clip_id)
    }

    /// Get all rating entries.
    #[allow(dead_code)]
    #[must_use]
    pub fn all(&self) -> Vec<&RatingEntry> {
        self.entries.values().collect()
    }

    /// Number of rated clips (not Unrated).
    #[allow(dead_code)]
    #[must_use]
    pub fn rated_count(&self) -> usize {
        self.entries
            .values()
            .filter(|e| e.rating != ClipRating::Unrated)
            .count()
    }

    /// Compute aggregate statistics.
    #[allow(dead_code)]
    #[must_use]
    pub fn stats(&self) -> RatingStats {
        let mut stats = RatingStats {
            total: self.entries.len() as u64,
            excellent: 0,
            good: 0,
            ok: 0,
            pickup: 0,
            reject: 0,
        };
        for entry in self.entries.values() {
            match entry.rating {
                ClipRating::Excellent => stats.excellent += 1,
                ClipRating::Good => stats.good += 1,
                ClipRating::Ok => stats.ok += 1,
                ClipRating::Pickup => stats.pickup += 1,
                ClipRating::Reject => stats.reject += 1,
                ClipRating::Unrated => {}
            }
        }
        stats
    }
}

impl Default for RatingDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Filters clips by rating criteria.
#[allow(dead_code)]
pub struct RatingFilter;

impl RatingFilter {
    /// Return clip IDs with a rating >= `min`.
    #[allow(dead_code)]
    #[must_use]
    pub fn filter_by_min_rating(entries: &[RatingEntry], min: ClipRating) -> Vec<u64> {
        entries
            .iter()
            .filter(|e| e.rating >= min)
            .map(|e| e.clip_id)
            .collect()
    }

    /// Return clip IDs with the specified flag.
    #[allow(dead_code)]
    #[must_use]
    pub fn filter_by_flag(entries: &[RatingEntry], flag: ClipFlag) -> Vec<u64> {
        entries
            .iter()
            .filter(|e| e.flag == flag)
            .map(|e| e.clip_id)
            .collect()
    }

    /// Return clip IDs with a rating >= min AND matching flag.
    #[allow(dead_code)]
    #[must_use]
    pub fn filter_by_rating_and_flag(
        entries: &[RatingEntry],
        min: ClipRating,
        flag: ClipFlag,
    ) -> Vec<u64> {
        entries
            .iter()
            .filter(|e| e.rating >= min && e.flag == flag)
            .map(|e| e.clip_id)
            .collect()
    }
}

/// Aggregate rating statistics for a clip collection.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct RatingStats {
    /// Total number of clips.
    pub total: u64,
    /// Number of Excellent clips.
    pub excellent: u64,
    /// Number of Good clips.
    pub good: u64,
    /// Number of OK clips.
    pub ok: u64,
    /// Number of Pickup clips.
    pub pickup: u64,
    /// Number of rejected clips.
    pub reject: u64,
}

impl RatingStats {
    /// Acceptance rate = (excellent + good + ok) / total, or 0.0 if total == 0.
    #[allow(dead_code)]
    #[must_use]
    pub fn acceptance_rate(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        (self.excellent + self.good + self.ok) as f32 / self.total as f32
    }

    /// Rejection rate = reject / total, or 0.0 if total == 0.
    #[allow(dead_code)]
    #[must_use]
    pub fn rejection_rate(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        self.reject as f32 / self.total as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_rating_stars() {
        assert_eq!(ClipRating::Excellent.stars(), 5);
        assert_eq!(ClipRating::Good.stars(), 3);
        assert_eq!(ClipRating::Ok.stars(), 2);
        assert_eq!(ClipRating::Pickup.stars(), 1);
        assert_eq!(ClipRating::Reject.stars(), 0);
        assert_eq!(ClipRating::Unrated.stars(), 0);
    }

    #[test]
    fn test_from_stars() {
        assert_eq!(ClipRating::from_stars(5), ClipRating::Excellent);
        assert_eq!(ClipRating::from_stars(4), ClipRating::Good);
        assert_eq!(ClipRating::from_stars(3), ClipRating::Good);
        assert_eq!(ClipRating::from_stars(2), ClipRating::Ok);
        assert_eq!(ClipRating::from_stars(1), ClipRating::Pickup);
        assert_eq!(ClipRating::from_stars(0), ClipRating::Unrated);
    }

    #[test]
    fn test_clip_rating_ordering() {
        assert!(ClipRating::Excellent > ClipRating::Good);
        assert!(ClipRating::Good > ClipRating::Ok);
        assert!(ClipRating::Ok > ClipRating::Pickup);
        assert!(ClipRating::Pickup > ClipRating::Unrated);
        assert!(ClipRating::Unrated > ClipRating::Reject);
    }

    #[test]
    fn test_clip_flag_name() {
        assert_eq!(ClipFlag::Red.name(), "Red");
        assert_eq!(ClipFlag::None.name(), "None");
        assert_eq!(ClipFlag::Green.name(), "Green");
    }

    #[test]
    fn test_rating_database_set_get() {
        let mut db = RatingDatabase::new();
        db.set_rating(1, ClipRating::Excellent);
        db.set_flag(1, ClipFlag::Green);

        let entry = db.get(1).expect("get should succeed");
        assert_eq!(entry.rating, ClipRating::Excellent);
        assert_eq!(entry.flag, ClipFlag::Green);
    }

    #[test]
    fn test_rating_database_overwrite() {
        let mut db = RatingDatabase::new();
        db.set_rating(1, ClipRating::Ok);
        db.set_rating(1, ClipRating::Excellent);
        assert_eq!(
            db.get(1).expect("get should succeed").rating,
            ClipRating::Excellent
        );
    }

    #[test]
    fn test_rating_database_nonexistent() {
        let db = RatingDatabase::new();
        assert!(db.get(999).is_none());
    }

    #[test]
    fn test_rating_stats_acceptance_rate() {
        let mut db = RatingDatabase::new();
        db.set_rating(1, ClipRating::Excellent);
        db.set_rating(2, ClipRating::Good);
        db.set_rating(3, ClipRating::Ok);
        db.set_rating(4, ClipRating::Reject);

        let stats = db.stats();
        assert_eq!(stats.total, 4);
        assert_eq!(stats.excellent, 1);
        assert_eq!(stats.good, 1);
        assert_eq!(stats.ok, 1);
        assert_eq!(stats.reject, 1);
        assert!((stats.acceptance_rate() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_rating_stats_empty() {
        let db = RatingDatabase::new();
        let stats = db.stats();
        assert_eq!(stats.acceptance_rate(), 0.0);
        assert_eq!(stats.rejection_rate(), 0.0);
    }

    #[test]
    fn test_filter_by_min_rating() {
        let entries = vec![
            RatingEntry {
                clip_id: 1,
                rating: ClipRating::Excellent,
                flag: ClipFlag::None,
                notes: String::new(),
                rated_at_ms: 0,
            },
            RatingEntry {
                clip_id: 2,
                rating: ClipRating::Ok,
                flag: ClipFlag::None,
                notes: String::new(),
                rated_at_ms: 0,
            },
            RatingEntry {
                clip_id: 3,
                rating: ClipRating::Reject,
                flag: ClipFlag::None,
                notes: String::new(),
                rated_at_ms: 0,
            },
        ];

        let ids = RatingFilter::filter_by_min_rating(&entries, ClipRating::Ok);
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_filter_by_flag() {
        let entries = vec![
            RatingEntry {
                clip_id: 1,
                rating: ClipRating::Good,
                flag: ClipFlag::Green,
                notes: String::new(),
                rated_at_ms: 0,
            },
            RatingEntry {
                clip_id: 2,
                rating: ClipRating::Ok,
                flag: ClipFlag::Red,
                notes: String::new(),
                rated_at_ms: 0,
            },
            RatingEntry {
                clip_id: 3,
                rating: ClipRating::Good,
                flag: ClipFlag::Green,
                notes: String::new(),
                rated_at_ms: 0,
            },
        ];

        let ids = RatingFilter::filter_by_flag(&entries, ClipFlag::Green);
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
    }

    #[test]
    fn test_filter_by_rating_and_flag() {
        let entries = vec![
            RatingEntry {
                clip_id: 1,
                rating: ClipRating::Excellent,
                flag: ClipFlag::Green,
                notes: String::new(),
                rated_at_ms: 0,
            },
            RatingEntry {
                clip_id: 2,
                rating: ClipRating::Ok,
                flag: ClipFlag::Green,
                notes: String::new(),
                rated_at_ms: 0,
            },
            RatingEntry {
                clip_id: 3,
                rating: ClipRating::Excellent,
                flag: ClipFlag::Red,
                notes: String::new(),
                rated_at_ms: 0,
            },
        ];

        let ids =
            RatingFilter::filter_by_rating_and_flag(&entries, ClipRating::Good, ClipFlag::Green);
        // Good <= Excellent (clip 1 passes), Good > Ok (clip 2 fails since Ok < Good)
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&1));
    }
}
