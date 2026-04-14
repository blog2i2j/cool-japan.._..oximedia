//! Multi-M/E linking for cascaded production workflows.
//!
//! Allows multiple M/E (Mix/Effect) rows to be linked together so that
//! transitions, program/preview selections, and cuts on a primary M/E are
//! automatically propagated to one or more follower M/E rows.
//!
//! # Cascade Modes
//!
//! - **Follow**: follower mirrors every program/preview selection made on the
//!   primary M/E.
//! - **Mirror**: follower mirrors all selections *and* transitions in lockstep.
//! - **Override**: follower receives commands from the primary unless it has a
//!   local override armed, in which case it operates independently.
//!
//! # Example
//!
//! ```rust
//! use oximedia_switcher::multi_me_link::{MeLinkManager, MeLinkConfig, CascadeMode};
//!
//! let mut manager = MeLinkManager::new(4);
//!
//! // Link M/E 1 to follow M/E 0 in mirror mode.
//! let cfg = MeLinkConfig::new(0, 1, CascadeMode::Mirror);
//! manager.add_link(cfg).expect("link should be added");
//!
//! // Propagate a program-change on M/E 0 → M/E 1 receives it too.
//! let commands = manager.propagate_program_select(0, 3).expect("propagate ok");
//! assert_eq!(commands.len(), 1);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

// ────────────────────────────────────────────────────────────────────────────
// Errors
// ────────────────────────────────────────────────────────────────────────────

/// Errors produced by the M/E link subsystem.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MeLinkError {
    /// The referenced M/E row index is out of range.
    #[error("M/E row {0} is out of range (max {1})")]
    MeRowOutOfRange(usize, usize),

    /// A link between the two rows already exists.
    #[error("A link from M/E {0} to M/E {1} already exists")]
    LinkAlreadyExists(usize, usize),

    /// The requested link was not found.
    #[error("No link from M/E {0} to M/E {1} found")]
    LinkNotFound(usize, usize),

    /// A cyclic dependency would be created.
    #[error("Adding link from M/E {0} to M/E {1} would create a cycle")]
    CyclicLink(usize, usize),

    /// A self-link is not permitted.
    #[error("An M/E row cannot be linked to itself (row {0})")]
    SelfLink(usize),

    /// The manager has not been configured for the requested M/E count.
    #[error("Manager not initialised with enough M/E rows ({0} requested, {1} available)")]
    NotEnoughMeRows(usize, usize),
}

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

/// How a follower M/E row reacts to events on its primary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CascadeMode {
    /// The follower mirrors every program/preview *selection* but runs its own
    /// transition independently.
    Follow,

    /// The follower mirrors selections *and* transitions in lockstep with the
    /// primary (tight synchronisation for complex productions).
    Mirror,

    /// The follower normally follows the primary; if a local override is armed
    /// it operates independently until the override is cleared.
    Override,
}

impl CascadeMode {
    /// Whether transitions should be propagated in this mode.
    pub fn propagates_transitions(&self) -> bool {
        matches!(self, CascadeMode::Mirror)
    }

    /// Whether program/preview selections are propagated.
    pub fn propagates_selections(&self) -> bool {
        matches!(
            self,
            CascadeMode::Follow | CascadeMode::Mirror | CascadeMode::Override
        )
    }
}

/// A single directed link from a primary M/E row to a follower M/E row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeLinkConfig {
    /// Primary M/E row (source of commands).
    pub primary_me: usize,
    /// Follower M/E row (receives propagated commands).
    pub follower_me: usize,
    /// How the follower reacts to primary events.
    pub mode: CascadeMode,
    /// Whether this link is currently active.
    pub enabled: bool,
    /// Human-readable description.
    pub label: String,
}

impl MeLinkConfig {
    /// Create a new link configuration.
    pub fn new(primary_me: usize, follower_me: usize, mode: CascadeMode) -> Self {
        Self {
            primary_me,
            follower_me,
            mode,
            enabled: true,
            label: format!("ME{primary_me}→ME{follower_me}"),
        }
    }

    /// Disable this link.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable this link.
    pub fn enable(&mut self) {
        self.enabled = true;
    }
}

/// A command that the link system wants to apply to a follower M/E row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CascadeCommand {
    /// Set program source on the specified M/E row.
    SetProgram { me_row: usize, input: usize },
    /// Set preview source on the specified M/E row.
    SetPreview { me_row: usize, input: usize },
    /// Execute a cut on the specified M/E row.
    Cut { me_row: usize },
    /// Execute an auto transition on the specified M/E row.
    AutoTransition { me_row: usize },
}

/// Optional local override state for a follower M/E row (used with
/// `CascadeMode::Override`).
#[derive(Debug, Clone, Default)]
struct OverrideState {
    /// Whether the local override is armed for this row.
    armed: bool,
}

// ────────────────────────────────────────────────────────────────────────────
// Manager
// ────────────────────────────────────────────────────────────────────────────

/// Manages all M/E link configurations and propagates switcher events across
/// linked M/E rows.
pub struct MeLinkManager {
    /// Total number of M/E rows in the switcher.
    me_count: usize,
    /// Active link configurations, keyed by (primary_me, follower_me).
    links: HashMap<(usize, usize), MeLinkConfig>,
    /// Per-row override state.
    overrides: Vec<OverrideState>,
}

impl MeLinkManager {
    /// Create a new manager for a switcher with `me_count` M/E rows.
    pub fn new(me_count: usize) -> Self {
        let overrides = (0..me_count).map(|_| OverrideState::default()).collect();
        Self {
            me_count,
            links: HashMap::new(),
            overrides,
        }
    }

    // ── Link management ──────────────────────────────────────────────────────

    /// Add a new link between two M/E rows.
    ///
    /// Returns an error if either row index is out of range, a link already
    /// exists, or adding the link would create a cycle.
    pub fn add_link(&mut self, config: MeLinkConfig) -> Result<(), MeLinkError> {
        self.validate_rows(config.primary_me, config.follower_me)?;

        if config.primary_me == config.follower_me {
            return Err(MeLinkError::SelfLink(config.primary_me));
        }

        let key = (config.primary_me, config.follower_me);
        if self.links.contains_key(&key) {
            return Err(MeLinkError::LinkAlreadyExists(
                config.primary_me,
                config.follower_me,
            ));
        }

        if self.would_create_cycle(config.primary_me, config.follower_me) {
            return Err(MeLinkError::CyclicLink(
                config.primary_me,
                config.follower_me,
            ));
        }

        self.links.insert(key, config);
        Ok(())
    }

    /// Remove an existing link.
    pub fn remove_link(
        &mut self,
        primary_me: usize,
        follower_me: usize,
    ) -> Result<(), MeLinkError> {
        self.validate_rows(primary_me, follower_me)?;
        let key = (primary_me, follower_me);
        if self.links.remove(&key).is_none() {
            return Err(MeLinkError::LinkNotFound(primary_me, follower_me));
        }
        Ok(())
    }

    /// Get a link configuration (immutable).
    pub fn get_link(&self, primary_me: usize, follower_me: usize) -> Option<&MeLinkConfig> {
        self.links.get(&(primary_me, follower_me))
    }

    /// Get a mutable link configuration.
    pub fn get_link_mut(
        &mut self,
        primary_me: usize,
        follower_me: usize,
    ) -> Option<&mut MeLinkConfig> {
        self.links.get_mut(&(primary_me, follower_me))
    }

    /// Enable a specific link.
    pub fn enable_link(
        &mut self,
        primary_me: usize,
        follower_me: usize,
    ) -> Result<(), MeLinkError> {
        self.links
            .get_mut(&(primary_me, follower_me))
            .ok_or(MeLinkError::LinkNotFound(primary_me, follower_me))
            .map(|l| l.enable())
    }

    /// Disable a specific link.
    pub fn disable_link(
        &mut self,
        primary_me: usize,
        follower_me: usize,
    ) -> Result<(), MeLinkError> {
        self.links
            .get_mut(&(primary_me, follower_me))
            .ok_or(MeLinkError::LinkNotFound(primary_me, follower_me))
            .map(|l| l.disable())
    }

    /// Return all links where the given row is the primary.
    pub fn links_from(&self, primary_me: usize) -> Vec<&MeLinkConfig> {
        self.links
            .values()
            .filter(|l| l.primary_me == primary_me)
            .collect()
    }

    /// Return all links where the given row is a follower.
    pub fn links_to(&self, follower_me: usize) -> Vec<&MeLinkConfig> {
        self.links
            .values()
            .filter(|l| l.follower_me == follower_me)
            .collect()
    }

    /// Total number of configured links (enabled + disabled).
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    // ── Override management ──────────────────────────────────────────────────

    /// Arm the local override for a follower M/E row.  While armed the row
    /// ignores cascaded commands (when its links use `CascadeMode::Override`).
    pub fn arm_override(&mut self, me_row: usize) -> Result<(), MeLinkError> {
        let state = self
            .overrides
            .get_mut(me_row)
            .ok_or(MeLinkError::MeRowOutOfRange(me_row, self.me_count))?;
        state.armed = true;
        Ok(())
    }

    /// Clear the local override for an M/E row.
    pub fn clear_override(&mut self, me_row: usize) -> Result<(), MeLinkError> {
        let state = self
            .overrides
            .get_mut(me_row)
            .ok_or(MeLinkError::MeRowOutOfRange(me_row, self.me_count))?;
        state.armed = false;
        Ok(())
    }

    /// Whether the override is currently armed for a row.
    pub fn is_override_armed(&self, me_row: usize) -> bool {
        self.overrides.get(me_row).map_or(false, |s| s.armed)
    }

    // ── Command propagation ──────────────────────────────────────────────────

    /// Propagate a program-source selection from `primary_me` to all active
    /// followers, returning the list of commands to execute.
    pub fn propagate_program_select(
        &self,
        primary_me: usize,
        input: usize,
    ) -> Result<Vec<CascadeCommand>, MeLinkError> {
        if primary_me >= self.me_count {
            return Err(MeLinkError::MeRowOutOfRange(primary_me, self.me_count));
        }
        let targets = self.collect_targets(primary_me, false);
        Ok(targets
            .into_iter()
            .map(|me_row| CascadeCommand::SetProgram { me_row, input })
            .collect())
    }

    /// Propagate a preview-source selection from `primary_me` to all active
    /// followers, returning the list of commands to execute.
    pub fn propagate_preview_select(
        &self,
        primary_me: usize,
        input: usize,
    ) -> Result<Vec<CascadeCommand>, MeLinkError> {
        if primary_me >= self.me_count {
            return Err(MeLinkError::MeRowOutOfRange(primary_me, self.me_count));
        }
        let targets = self.collect_targets(primary_me, false);
        Ok(targets
            .into_iter()
            .map(|me_row| CascadeCommand::SetPreview { me_row, input })
            .collect())
    }

    /// Propagate a cut command from `primary_me` to all active followers.
    pub fn propagate_cut(&self, primary_me: usize) -> Result<Vec<CascadeCommand>, MeLinkError> {
        if primary_me >= self.me_count {
            return Err(MeLinkError::MeRowOutOfRange(primary_me, self.me_count));
        }
        let targets = self.collect_targets(primary_me, false);
        Ok(targets
            .into_iter()
            .map(|me_row| CascadeCommand::Cut { me_row })
            .collect())
    }

    /// Propagate an auto-transition command from `primary_me` to all active
    /// followers that are in `Mirror` mode.
    pub fn propagate_auto_transition(
        &self,
        primary_me: usize,
    ) -> Result<Vec<CascadeCommand>, MeLinkError> {
        if primary_me >= self.me_count {
            return Err(MeLinkError::MeRowOutOfRange(primary_me, self.me_count));
        }
        // Auto-transition propagates only to Mirror-mode followers.
        let targets = self.collect_targets(primary_me, true);
        Ok(targets
            .into_iter()
            .map(|me_row| CascadeCommand::AutoTransition { me_row })
            .collect())
    }

    // ── Introspection ────────────────────────────────────────────────────────

    /// Return the number of configured M/E rows.
    pub fn me_count(&self) -> usize {
        self.me_count
    }

    /// Collect all enabled follower rows reachable from `primary_me` via
    /// transitive links, respecting override state.
    ///
    /// If `transitions_only` is true only links that propagate transitions
    /// (i.e. `Mirror` mode) are traversed.
    fn collect_targets(&self, primary_me: usize, transitions_only: bool) -> Vec<usize> {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut queue: Vec<usize> = vec![primary_me];
        let mut result: Vec<usize> = Vec::new();

        while let Some(current) = queue.pop() {
            for link in self.links.values() {
                if link.primary_me != current || !link.enabled {
                    continue;
                }
                if transitions_only && !link.mode.propagates_transitions() {
                    continue;
                }
                if !link.mode.propagates_selections() && !transitions_only {
                    continue;
                }
                let follower = link.follower_me;

                // Skip if override armed in Override mode.
                if link.mode == CascadeMode::Override && self.is_override_armed(follower) {
                    continue;
                }

                if visited.insert(follower) {
                    result.push(follower);
                    queue.push(follower);
                }
            }
        }

        result
    }

    /// Check whether adding a link from `from` to `to` would create a cycle
    /// (i.e. `from` is already reachable from `to`).
    fn would_create_cycle(&self, from: usize, to: usize) -> bool {
        // BFS from `to`: if we can reach `from`, adding from→to creates a cycle.
        let mut visited: HashSet<usize> = HashSet::new();
        let mut queue: Vec<usize> = vec![to];

        while let Some(current) = queue.pop() {
            if current == from {
                return true;
            }
            for link in self.links.values() {
                if link.primary_me == current && visited.insert(link.follower_me) {
                    queue.push(link.follower_me);
                }
            }
        }

        false
    }

    /// Validate that both M/E row indices are within range.
    fn validate_rows(&self, a: usize, b: usize) -> Result<(), MeLinkError> {
        if a >= self.me_count {
            return Err(MeLinkError::MeRowOutOfRange(a, self.me_count));
        }
        if b >= self.me_count {
            return Err(MeLinkError::MeRowOutOfRange(b, self.me_count));
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> MeLinkManager {
        MeLinkManager::new(4)
    }

    #[test]
    fn test_add_link_basic() {
        let mut mgr = make_manager();
        let cfg = MeLinkConfig::new(0, 1, CascadeMode::Follow);
        mgr.add_link(cfg).expect("add_link should succeed");
        assert_eq!(mgr.link_count(), 1);
    }

    #[test]
    fn test_add_link_duplicate_error() {
        let mut mgr = make_manager();
        mgr.add_link(MeLinkConfig::new(0, 1, CascadeMode::Follow))
            .expect("first add ok");
        let err = mgr
            .add_link(MeLinkConfig::new(0, 1, CascadeMode::Mirror))
            .expect_err("duplicate should fail");
        assert!(matches!(err, MeLinkError::LinkAlreadyExists(0, 1)));
    }

    #[test]
    fn test_self_link_error() {
        let mut mgr = make_manager();
        let err = mgr
            .add_link(MeLinkConfig::new(2, 2, CascadeMode::Follow))
            .expect_err("self-link should fail");
        assert!(matches!(err, MeLinkError::SelfLink(2)));
    }

    #[test]
    fn test_cycle_detection() {
        let mut mgr = make_manager();
        mgr.add_link(MeLinkConfig::new(0, 1, CascadeMode::Follow))
            .expect("0→1 ok");
        mgr.add_link(MeLinkConfig::new(1, 2, CascadeMode::Follow))
            .expect("1→2 ok");
        let err = mgr
            .add_link(MeLinkConfig::new(2, 0, CascadeMode::Follow))
            .expect_err("cycle 2→0 should be rejected");
        assert!(matches!(err, MeLinkError::CyclicLink(2, 0)));
    }

    #[test]
    fn test_remove_link() {
        let mut mgr = make_manager();
        mgr.add_link(MeLinkConfig::new(0, 1, CascadeMode::Follow))
            .expect("add ok");
        mgr.remove_link(0, 1).expect("remove ok");
        assert_eq!(mgr.link_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_link_error() {
        let mut mgr = make_manager();
        let err = mgr
            .remove_link(0, 1)
            .expect_err("remove nonexistent should fail");
        assert!(matches!(err, MeLinkError::LinkNotFound(0, 1)));
    }

    #[test]
    fn test_propagate_program_select_single_follower() {
        let mut mgr = make_manager();
        mgr.add_link(MeLinkConfig::new(0, 1, CascadeMode::Follow))
            .expect("add ok");

        let cmds = mgr.propagate_program_select(0, 5).expect("propagate ok");
        assert_eq!(cmds.len(), 1);
        assert_eq!(
            cmds[0],
            CascadeCommand::SetProgram {
                me_row: 1,
                input: 5
            }
        );
    }

    #[test]
    fn test_propagate_program_select_no_links() {
        let mgr = make_manager();
        let cmds = mgr.propagate_program_select(0, 3).expect("propagate ok");
        assert!(cmds.is_empty());
    }

    #[test]
    fn test_propagate_cut_transitive() {
        let mut mgr = make_manager();
        // 0 → 1 → 2 chain
        mgr.add_link(MeLinkConfig::new(0, 1, CascadeMode::Follow))
            .expect("0→1");
        mgr.add_link(MeLinkConfig::new(1, 2, CascadeMode::Follow))
            .expect("1→2");

        let cmds = mgr.propagate_cut(0).expect("cut ok");
        // Both 1 and 2 should receive the cut.
        assert_eq!(cmds.len(), 2);
        let rows: Vec<usize> = cmds
            .iter()
            .map(|c| match c {
                CascadeCommand::Cut { me_row } => *me_row,
                _ => panic!("expected Cut"),
            })
            .collect();
        assert!(rows.contains(&1));
        assert!(rows.contains(&2));
    }

    #[test]
    fn test_propagate_auto_transition_mirror_only() {
        let mut mgr = make_manager();
        // One Follow link and one Mirror link from the same primary.
        mgr.add_link(MeLinkConfig::new(0, 1, CascadeMode::Follow))
            .expect("0→1 follow");
        mgr.add_link(MeLinkConfig::new(0, 2, CascadeMode::Mirror))
            .expect("0→2 mirror");

        let cmds = mgr.propagate_auto_transition(0).expect("auto ok");
        // Only the Mirror follower (row 2) should receive the auto-transition.
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0], CascadeCommand::AutoTransition { me_row: 2 });
    }

    #[test]
    fn test_override_blocks_propagation() {
        let mut mgr = make_manager();
        mgr.add_link(MeLinkConfig::new(0, 1, CascadeMode::Override))
            .expect("add ok");

        // No override armed → propagates.
        let cmds = mgr.propagate_cut(0).expect("cut ok");
        assert_eq!(cmds.len(), 1);

        // Arm override on follower → propagation suppressed.
        mgr.arm_override(1).expect("arm ok");
        let cmds = mgr.propagate_cut(0).expect("cut ok");
        assert!(cmds.is_empty());

        // Clear override → propagates again.
        mgr.clear_override(1).expect("clear ok");
        let cmds = mgr.propagate_cut(0).expect("cut ok");
        assert_eq!(cmds.len(), 1);
    }

    #[test]
    fn test_enable_disable_link() {
        let mut mgr = make_manager();
        mgr.add_link(MeLinkConfig::new(0, 1, CascadeMode::Follow))
            .expect("add ok");

        // Disable → no propagation.
        mgr.disable_link(0, 1).expect("disable ok");
        let cmds = mgr.propagate_program_select(0, 2).expect("propagate ok");
        assert!(cmds.is_empty());

        // Re-enable → propagation resumes.
        mgr.enable_link(0, 1).expect("enable ok");
        let cmds = mgr.propagate_program_select(0, 2).expect("propagate ok");
        assert_eq!(cmds.len(), 1);
    }

    #[test]
    fn test_out_of_range_error() {
        let mgr = make_manager();
        let err = mgr.propagate_cut(99).expect_err("out of range should fail");
        assert!(matches!(err, MeLinkError::MeRowOutOfRange(99, 4)));
    }

    #[test]
    fn test_cascade_mode_propagates() {
        assert!(!CascadeMode::Follow.propagates_transitions());
        assert!(CascadeMode::Mirror.propagates_transitions());
        assert!(!CascadeMode::Override.propagates_transitions());

        assert!(CascadeMode::Follow.propagates_selections());
        assert!(CascadeMode::Mirror.propagates_selections());
        assert!(CascadeMode::Override.propagates_selections());
    }

    #[test]
    fn test_links_from_and_to() {
        let mut mgr = make_manager();
        mgr.add_link(MeLinkConfig::new(0, 1, CascadeMode::Follow))
            .expect("0→1");
        mgr.add_link(MeLinkConfig::new(0, 2, CascadeMode::Mirror))
            .expect("0→2");
        mgr.add_link(MeLinkConfig::new(1, 2, CascadeMode::Follow))
            .expect("1→2");

        assert_eq!(mgr.links_from(0).len(), 2);
        assert_eq!(mgr.links_from(1).len(), 1);
        assert_eq!(mgr.links_to(2).len(), 2);
        assert_eq!(mgr.links_to(0).len(), 0);
    }

    #[test]
    fn test_preview_propagation() {
        let mut mgr = make_manager();
        mgr.add_link(MeLinkConfig::new(0, 3, CascadeMode::Follow))
            .expect("add ok");

        let cmds = mgr.propagate_preview_select(0, 7).expect("propagate ok");
        assert_eq!(cmds.len(), 1);
        assert_eq!(
            cmds[0],
            CascadeCommand::SetPreview {
                me_row: 3,
                input: 7
            }
        );
    }
}
