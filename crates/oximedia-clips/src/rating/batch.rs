//! Batch rating operations: apply ratings/flags to multiple clips at once.

#![allow(dead_code)]

use super::{ClipFlag, ClipRating, RatingDatabase, RatingEntry};

/// A batch rating command targeting multiple clip IDs.
#[derive(Debug, Clone)]
pub struct BatchRatingCommand {
    /// Clip IDs to update.
    pub clip_ids: Vec<u64>,
    /// New rating to apply (`None` = leave unchanged).
    pub rating: Option<ClipRating>,
    /// New flag to apply (`None` = leave unchanged).
    pub flag: Option<ClipFlag>,
    /// Notes to append (empty string = leave unchanged).
    pub append_note: String,
}

impl BatchRatingCommand {
    /// Create a command that only sets the rating.
    #[must_use]
    pub fn set_rating(clip_ids: Vec<u64>, rating: ClipRating) -> Self {
        Self {
            clip_ids,
            rating: Some(rating),
            flag: None,
            append_note: String::new(),
        }
    }

    /// Create a command that only sets the flag.
    #[must_use]
    pub fn set_flag(clip_ids: Vec<u64>, flag: ClipFlag) -> Self {
        Self {
            clip_ids,
            rating: None,
            flag: Some(flag),
            append_note: String::new(),
        }
    }

    /// Create a command that sets both rating and flag.
    #[must_use]
    pub fn set_rating_and_flag(clip_ids: Vec<u64>, rating: ClipRating, flag: ClipFlag) -> Self {
        Self {
            clip_ids,
            rating: Some(rating),
            flag: Some(flag),
            append_note: String::new(),
        }
    }

    /// Add a note to be appended to each clip's notes.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.append_note = note.into();
        self
    }

    /// Apply this command to a [`RatingDatabase`].
    ///
    /// Returns the number of clips updated.
    pub fn apply(&self, db: &mut RatingDatabase) -> usize {
        let mut count = 0;
        for &id in &self.clip_ids {
            if let Some(rating) = self.rating {
                db.set_rating(id, rating);
            }
            if let Some(flag) = self.flag {
                db.set_flag(id, flag);
            }
            if !self.append_note.is_empty() {
                let existing = db.get(id).map(|e| e.notes.clone()).unwrap_or_default();
                let new_notes = if existing.is_empty() {
                    self.append_note.clone()
                } else {
                    format!("{} {}", existing, self.append_note)
                };
                db.set_notes(id, new_notes);
            }
            count += 1;
        }
        count
    }
}

/// Summary of a batch operation result.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchResult {
    /// Number of clips processed.
    pub processed: usize,
    /// Number of clips that were already at the target rating (not changed).
    pub skipped: usize,
}

impl BatchResult {
    /// Create a new result record.
    #[must_use]
    pub fn new(processed: usize, skipped: usize) -> Self {
        Self { processed, skipped }
    }
}

/// Apply a batch command but skip clips that already have the target rating.
pub fn apply_idempotent(cmd: &BatchRatingCommand, db: &mut RatingDatabase) -> BatchResult {
    let mut processed = 0;
    let mut skipped = 0;

    for &id in &cmd.clip_ids {
        // Check if already at target rating (if rating is specified)
        let already_rated = cmd.rating.map_or(false, |target_rating| {
            db.get(id)
                .map(|e| e.rating == target_rating)
                .unwrap_or(false)
        });

        if already_rated && cmd.flag.is_none() && cmd.append_note.is_empty() {
            skipped += 1;
            continue;
        }

        if let Some(rating) = cmd.rating {
            db.set_rating(id, rating);
        }
        if let Some(flag) = cmd.flag {
            db.set_flag(id, flag);
        }
        if !cmd.append_note.is_empty() {
            let existing = db.get(id).map(|e| e.notes.clone()).unwrap_or_default();
            let new_notes = if existing.is_empty() {
                cmd.append_note.clone()
            } else {
                format!("{} {}", existing, cmd.append_note)
            };
            db.set_notes(id, new_notes);
        }
        processed += 1;
    }

    BatchResult::new(processed, skipped)
}

/// Filter clip IDs from the database that match the given criteria.
#[must_use]
pub fn select_by_rating(db: &RatingDatabase, min: ClipRating) -> Vec<u64> {
    db.all()
        .into_iter()
        .filter(|e| e.rating >= min)
        .map(|e| e.clip_id)
        .collect()
}

/// Filter clip IDs by flag.
#[must_use]
pub fn select_by_flag(db: &RatingDatabase, flag: ClipFlag) -> Vec<u64> {
    db.all()
        .into_iter()
        .filter(|e| e.flag == flag)
        .map(|e| e.clip_id)
        .collect()
}

/// Reject all clips currently rated below `threshold`.
///
/// Returns the list of clip IDs that were rejected.
pub fn reject_below(db: &mut RatingDatabase, threshold: ClipRating) -> Vec<u64> {
    let to_reject: Vec<u64> = db
        .all()
        .into_iter()
        .filter(|e| e.rating < threshold)
        .map(|e| e.clip_id)
        .collect();

    for &id in &to_reject {
        db.set_rating(id, ClipRating::Reject);
    }
    to_reject
}

/// Export a snapshot of all ratings as a simple `clip_id,rating,flag` CSV.
#[must_use]
pub fn export_csv(db: &RatingDatabase) -> String {
    let mut lines = vec!["clip_id,rating,flag,notes".to_string()];
    let mut entries: Vec<&RatingEntry> = db.all();
    entries.sort_by_key(|e| e.clip_id);
    for e in entries {
        lines.push(format!(
            "{},{},{},{}",
            e.clip_id,
            e.rating.label(),
            e.flag.name(),
            e.notes
        ));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn populated_db() -> RatingDatabase {
        let mut db = RatingDatabase::new();
        db.set_rating(1, ClipRating::Excellent);
        db.set_flag(1, ClipFlag::Green);
        db.set_rating(2, ClipRating::Ok);
        db.set_flag(2, ClipFlag::Yellow);
        db.set_rating(3, ClipRating::Pickup);
        db.set_flag(3, ClipFlag::Red);
        db.set_rating(4, ClipRating::Unrated);
        db
    }

    #[test]
    fn test_batch_set_rating_apply() {
        let mut db = populated_db();
        let cmd = BatchRatingCommand::set_rating(vec![2, 3], ClipRating::Good);
        let count = cmd.apply(&mut db);
        assert_eq!(count, 2);
        assert_eq!(
            db.get(2).expect("get should succeed").rating,
            ClipRating::Good
        );
        assert_eq!(
            db.get(3).expect("get should succeed").rating,
            ClipRating::Good
        );
        // Clip 1 unchanged
        assert_eq!(
            db.get(1).expect("get should succeed").rating,
            ClipRating::Excellent
        );
    }

    #[test]
    fn test_batch_set_flag_apply() {
        let mut db = populated_db();
        let cmd = BatchRatingCommand::set_flag(vec![1, 2], ClipFlag::Blue);
        cmd.apply(&mut db);
        assert_eq!(db.get(1).expect("get should succeed").flag, ClipFlag::Blue);
        assert_eq!(db.get(2).expect("get should succeed").flag, ClipFlag::Blue);
    }

    #[test]
    fn test_batch_set_rating_and_flag() {
        let mut db = populated_db();
        let cmd = BatchRatingCommand::set_rating_and_flag(
            vec![3, 4],
            ClipRating::Excellent,
            ClipFlag::Green,
        );
        let count = cmd.apply(&mut db);
        assert_eq!(count, 2);
        assert_eq!(
            db.get(3).expect("get should succeed").rating,
            ClipRating::Excellent
        );
        assert_eq!(db.get(4).expect("get should succeed").flag, ClipFlag::Green);
    }

    #[test]
    fn test_batch_with_note() {
        let mut db = RatingDatabase::new();
        db.set_rating(10, ClipRating::Good);
        let cmd = BatchRatingCommand::set_rating(vec![10], ClipRating::Good)
            .with_note("approved by director");
        cmd.apply(&mut db);
        assert!(db
            .get(10)
            .expect("get should succeed")
            .notes
            .contains("approved by director"));
    }

    #[test]
    fn test_batch_note_appended() {
        let mut db = RatingDatabase::new();
        db.set_rating(5, ClipRating::Ok);
        db.set_notes(5, "first note");
        let cmd = BatchRatingCommand::set_rating(vec![5], ClipRating::Ok).with_note("second note");
        cmd.apply(&mut db);
        let notes = &db.get(5).expect("get should succeed").notes;
        assert!(notes.contains("first note"));
        assert!(notes.contains("second note"));
    }

    #[test]
    fn test_apply_idempotent_already_rated() {
        let mut db = populated_db();
        // Clip 1 is already Excellent
        let cmd = BatchRatingCommand::set_rating(vec![1], ClipRating::Excellent);
        let result = apply_idempotent(&cmd, &mut db);
        assert_eq!(result.skipped, 1);
        assert_eq!(result.processed, 0);
    }

    #[test]
    fn test_apply_idempotent_changes_applied() {
        let mut db = populated_db();
        let cmd = BatchRatingCommand::set_rating(vec![2, 3], ClipRating::Excellent);
        let result = apply_idempotent(&cmd, &mut db);
        assert_eq!(result.processed, 2);
        assert_eq!(result.skipped, 0);
    }

    #[test]
    fn test_select_by_rating() {
        let db = populated_db();
        let ids = select_by_rating(&db, ClipRating::Ok);
        // Excellent and Ok qualify (Ok >= Ok)
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&3));
    }

    #[test]
    fn test_select_by_flag() {
        let db = populated_db();
        let ids = select_by_flag(&db, ClipFlag::Green);
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn test_reject_below() {
        let mut db = populated_db();
        let rejected = reject_below(&mut db, ClipRating::Ok);
        // Pickup (3) and Unrated (4) should be rejected
        assert!(rejected.contains(&3));
        assert!(rejected.contains(&4));
        assert_eq!(
            db.get(3).expect("get should succeed").rating,
            ClipRating::Reject
        );
        assert_eq!(
            db.get(4).expect("get should succeed").rating,
            ClipRating::Reject
        );
        // Excellent and Ok remain
        assert_eq!(
            db.get(1).expect("get should succeed").rating,
            ClipRating::Excellent
        );
        assert_eq!(
            db.get(2).expect("get should succeed").rating,
            ClipRating::Ok
        );
    }

    #[test]
    fn test_export_csv() {
        let mut db = RatingDatabase::new();
        db.set_rating(1, ClipRating::Good);
        db.set_flag(1, ClipFlag::Green);
        let csv = export_csv(&db);
        assert!(csv.starts_with("clip_id,rating,flag,notes"));
        assert!(csv.contains("Good"));
        assert!(csv.contains("Green"));
    }

    #[test]
    fn test_batch_result() {
        let r = BatchResult::new(5, 2);
        assert_eq!(r.processed, 5);
        assert_eq!(r.skipped, 2);
    }
}
