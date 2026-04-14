//! Essence relinking — update media file references when paths change.
//!
//! When media files are moved or renamed, the path references inside AAF mobs
//! must be updated to reflect the new locations.  `AafEssenceRelinker` walks
//! through a [`Mob`] and replaces every occurrence of `old_path` with
//! `new_path` in the mob's source file references.

use crate::object_model::Mob;
use crate::AafError;
use crate::Result;

/// Relinks essence (media file) references inside AAF mobs.
///
/// # Example
///
/// ```rust
/// use oximedia_aaf::relink::AafEssenceRelinker;
/// use oximedia_aaf::object_model::{Mob, MobType};
/// use uuid::Uuid;
///
/// let mut mob = Mob::new(Uuid::new_v4(), "old/source.mxf".to_string(), MobType::Source);
/// mob.set_source_path("old/source.mxf");
///
/// let relinker = AafEssenceRelinker::new();
/// let changed = relinker.relink(&mut mob, "old/source.mxf", "new/source.mxf")
///     .expect("relink must succeed");
/// assert!(changed > 0);
/// assert_eq!(mob.source_path(), Some("new/source.mxf"));
/// ```
pub struct AafEssenceRelinker;

impl AafEssenceRelinker {
    /// Create a new relinker.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Relink all occurrences of `old_path` to `new_path` inside `mob`.
    ///
    /// Updates:
    /// - The mob's primary source path (if it matches `old_path`).
    /// - All locator paths stored in the mob's extra locators list.
    ///
    /// Returns the number of path references that were changed.
    ///
    /// # Errors
    ///
    /// Returns `AafError::InvalidFile` if `old_path` is empty.
    pub fn relink(&self, mob: &mut Mob, old_path: &str, new_path: &str) -> Result<usize> {
        if old_path.is_empty() {
            return Err(AafError::InvalidFile(
                "AafEssenceRelinker::relink: old_path must not be empty".to_string(),
            ));
        }

        let mut changed = 0usize;

        // Relink the primary source path.
        if let Some(existing) = mob.source_path() {
            if existing == old_path {
                mob.set_source_path(new_path);
                changed += 1;
            }
        }

        // Relink any additional locator paths.
        for locator in mob.locators_mut() {
            if locator.path == old_path {
                locator.path = new_path.to_string();
                changed += 1;
            }
        }

        Ok(changed)
    }

    /// Relink all mobs in a slice, returning the total number of changes.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered.
    pub fn relink_all(
        &self,
        mobs: &mut [Mob],
        old_path: &str,
        new_path: &str,
    ) -> Result<usize> {
        let mut total = 0usize;
        for mob in mobs.iter_mut() {
            total += self.relink(mob, old_path, new_path)?;
        }
        Ok(total)
    }
}

impl Default for AafEssenceRelinker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object_model::{Mob, MobLocator, MobType};
    use uuid::Uuid;

    fn make_source_mob(path: &str) -> Mob {
        let mut mob = Mob::new(Uuid::new_v4(), "test_mob".to_string(), MobType::Source);
        mob.set_source_path(path);
        mob
    }

    #[test]
    fn test_relink_changes_source_path() {
        let mut mob = make_source_mob("old/file.mxf");
        let relinker = AafEssenceRelinker::new();
        let n = relinker
            .relink(&mut mob, "old/file.mxf", "new/file.mxf")
            .expect("ok");
        assert_eq!(n, 1);
        assert_eq!(mob.source_path(), Some("new/file.mxf"));
    }

    #[test]
    fn test_relink_no_match_returns_zero() {
        let mut mob = make_source_mob("different/file.mxf");
        let relinker = AafEssenceRelinker::new();
        let n = relinker
            .relink(&mut mob, "old/file.mxf", "new/file.mxf")
            .expect("ok");
        assert_eq!(n, 0);
        assert_eq!(mob.source_path(), Some("different/file.mxf"));
    }

    #[test]
    fn test_relink_empty_old_path_returns_error() {
        let mut mob = make_source_mob("file.mxf");
        let relinker = AafEssenceRelinker::new();
        assert!(relinker.relink(&mut mob, "", "new.mxf").is_err());
    }

    #[test]
    fn test_relink_locator_path() {
        let mut mob = make_source_mob("main.mxf");
        mob.add_locator(MobLocator {
            path: "extra/file.mxf".to_string(),
        });
        let relinker = AafEssenceRelinker::new();
        let n = relinker
            .relink(&mut mob, "extra/file.mxf", "extra/new.mxf")
            .expect("ok");
        assert_eq!(n, 1);
        assert_eq!(mob.locators()[0].path, "extra/new.mxf");
    }

    #[test]
    fn test_relink_all_multiple_mobs() {
        let mut mobs = vec![
            make_source_mob("shared/media.mxf"),
            make_source_mob("shared/media.mxf"),
            make_source_mob("other/file.mxf"),
        ];
        let relinker = AafEssenceRelinker::new();
        let total = relinker
            .relink_all(&mut mobs, "shared/media.mxf", "archive/media.mxf")
            .expect("ok");
        assert_eq!(total, 2);
    }

    #[test]
    fn test_relink_both_source_and_locator() {
        let mut mob = make_source_mob("shared/media.mxf");
        mob.add_locator(MobLocator {
            path: "shared/media.mxf".to_string(),
        });
        let relinker = AafEssenceRelinker::new();
        let n = relinker
            .relink(&mut mob, "shared/media.mxf", "new/media.mxf")
            .expect("ok");
        assert_eq!(n, 2);
        assert_eq!(mob.source_path(), Some("new/media.mxf"));
        assert_eq!(mob.locators()[0].path, "new/media.mxf");
    }

    #[test]
    fn test_relink_multiple_locators() {
        let mut mob = make_source_mob("other.mxf");
        mob.add_locator(MobLocator {
            path: "target.mxf".to_string(),
        });
        mob.add_locator(MobLocator {
            path: "target.mxf".to_string(),
        });
        mob.add_locator(MobLocator {
            path: "keep.mxf".to_string(),
        });
        let relinker = AafEssenceRelinker::new();
        let n = relinker
            .relink(&mut mob, "target.mxf", "replaced.mxf")
            .expect("ok");
        assert_eq!(n, 2);
        assert_eq!(mob.locators()[0].path, "replaced.mxf");
        assert_eq!(mob.locators()[1].path, "replaced.mxf");
        assert_eq!(mob.locators()[2].path, "keep.mxf");
    }

    #[test]
    fn test_relink_all_empty_slice() {
        let mut mobs: Vec<Mob> = vec![];
        let relinker = AafEssenceRelinker::new();
        let total = relinker
            .relink_all(&mut mobs, "a.mxf", "b.mxf")
            .expect("ok");
        assert_eq!(total, 0);
    }

    #[test]
    fn test_relink_preserves_mob_without_source_path() {
        let mut mob = Mob::new(Uuid::new_v4(), "no_path".to_string(), MobType::Master);
        let relinker = AafEssenceRelinker::new();
        let n = relinker
            .relink(&mut mob, "anything.mxf", "new.mxf")
            .expect("ok");
        assert_eq!(n, 0);
        assert!(mob.source_path().is_none());
    }

    #[test]
    fn test_relink_default_constructor() {
        let relinker = AafEssenceRelinker::default();
        let mut mob = make_source_mob("file.mxf");
        let n = relinker
            .relink(&mut mob, "file.mxf", "new_file.mxf")
            .expect("ok");
        assert_eq!(n, 1);
    }

    #[test]
    fn test_relink_all_propagates_error() {
        let mut mobs = vec![make_source_mob("a.mxf")];
        let relinker = AafEssenceRelinker::new();
        let result = relinker.relink_all(&mut mobs, "", "b.mxf");
        assert!(result.is_err());
    }
}
