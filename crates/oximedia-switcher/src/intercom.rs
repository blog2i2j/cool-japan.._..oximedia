//! Production intercom and IFB (Interruptible Fold-Back) communication channel
//! management for live broadcast production.
//!
//! A production intercom system provides private audio communication paths
//! between crew members: directors, producers, camera operators, floor managers,
//! talent, and technical operators.  IFB feeds allow the director to interrupt
//! the talent's earpiece with a private cue while their programme audio
//! continues at a reduced level.
//!
//! # Architecture
//!
//! ```text
//!  ┌──────────────────────────────────────────────────────────────┐
//!  │                     IntercomManager                          │
//!  │  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │
//!  │  │ Channels[] │  │ Users[]    │  │ TalkGroup routing    │  │
//!  │  └────────────┘  └────────────┘  └──────────────────────┘  │
//!  └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use oximedia_switcher::intercom::{IntercomManager, IntercomChannel, IntercomUser};
//!
//! let mut mgr = IntercomManager::new();
//!
//! // Create the "Production" party-line channel.
//! let ch_id = mgr.create_channel(IntercomChannel::partyline("Production")).expect("ok");
//!
//! // Register two users.
//! let dir = mgr.register_user(IntercomUser::new("Director")).expect("ok");
//! let cam = mgr.register_user(IntercomUser::new("Camera1")).expect("ok");
//!
//! // Assign them to the channel.
//! mgr.assign_user_to_channel(dir, ch_id).expect("ok");
//! mgr.assign_user_to_channel(cam, ch_id).expect("ok");
//!
//! // Director calls the channel.
//! mgr.activate_talk(dir, ch_id).expect("ok");
//! assert!(mgr.is_talking(dir, ch_id));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

// ────────────────────────────────────────────────────────────────────────────
// Errors
// ────────────────────────────────────────────────────────────────────────────

/// Errors from the intercom subsystem.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum IntercomError {
    /// Channel ID does not exist.
    #[error("Intercom channel {0} not found")]
    ChannelNotFound(usize),

    /// User ID does not exist.
    #[error("Intercom user {0} not found")]
    UserNotFound(usize),

    /// The user is not assigned to the channel.
    #[error("User {0} is not assigned to channel {1}")]
    UserNotOnChannel(usize, usize),

    /// The user is already assigned to the channel.
    #[error("User {0} is already assigned to channel {1}")]
    AlreadyOnChannel(usize, usize),

    /// Channel name is empty.
    #[error("Channel name must not be empty")]
    EmptyChannelName,

    /// User name is empty.
    #[error("User name must not be empty")]
    EmptyUserName,

    /// Maximum channel count exceeded.
    #[error("Cannot create more than {0} intercom channels")]
    ChannelLimitExceeded(usize),

    /// Maximum user count exceeded.
    #[error("Cannot register more than {0} intercom users")]
    UserLimitExceeded(usize),

    /// IFB target user has no dedicated IFB feed configured.
    #[error("User {0} has no IFB feed configured")]
    NoIfbFeed(usize),
}

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

/// Maximum number of intercom channels.
pub const MAX_CHANNELS: usize = 64;

/// Maximum number of registered users.
pub const MAX_USERS: usize = 256;

// ────────────────────────────────────────────────────────────────────────────
// Channel type
// ────────────────────────────────────────────────────────────────────────────

/// Determines how audio is routed within the channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelType {
    /// Party-line: all assigned users hear each other simultaneously.
    PartyLine,
    /// Point-to-point: exactly two users, duplex link.
    PointToPoint,
    /// IFB feed: unidirectional cue channel into a talent earpiece.
    /// Programme audio is ducked while the director speaks.
    Ifb,
    /// Technical channel: camera operators / RCP operators only.
    Technical,
    /// Programme relay: feed-to-talent (F2T) of programme audio.
    ProgramRelay,
}

impl ChannelType {
    /// Returns `true` if this channel type supports multiple simultaneous talkers.
    pub fn allows_multiple_talkers(&self) -> bool {
        matches!(self, ChannelType::PartyLine | ChannelType::Technical)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Talk mode
// ────────────────────────────────────────────────────────────────────────────

/// How a user activates their microphone on a channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TalkMode {
    /// Press-to-talk: microphone is active only while the button is held.
    PressToTalk,
    /// Latch-on: first press activates, second press deactivates.
    LatchOn,
    /// Hot-mic: microphone is always live (no user action required).
    HotMic,
}

// ────────────────────────────────────────────────────────────────────────────
// Channel
// ────────────────────────────────────────────────────────────────────────────

/// An intercom / IFB channel descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntercomChannel {
    /// Human-readable label (e.g., "Production", "Camera", "Director").
    pub name: String,
    /// Channel type.
    pub channel_type: ChannelType,
    /// Nominal listen volume in dBu relative to reference level (0.0 = unity).
    pub listen_gain_db: f32,
    /// IFB programme duck depth in dB (applied when a cue arrives).
    /// Relevant only for `Ifb` channel type.
    pub ifb_duck_depth_db: f32,
    /// Whether the channel is currently muted.
    pub muted: bool,
}

impl IntercomChannel {
    /// Create a new channel with a given type.
    pub fn new(name: impl Into<String>, channel_type: ChannelType) -> Self {
        Self {
            name: name.into(),
            channel_type,
            listen_gain_db: 0.0,
            ifb_duck_depth_db: -18.0,
            muted: false,
        }
    }

    /// Convenience: create a party-line channel.
    pub fn partyline(name: impl Into<String>) -> Self {
        Self::new(name, ChannelType::PartyLine)
    }

    /// Convenience: create an IFB channel.
    pub fn ifb(name: impl Into<String>) -> Self {
        Self::new(name, ChannelType::Ifb)
    }

    /// Convenience: create a point-to-point channel.
    pub fn point_to_point(name: impl Into<String>) -> Self {
        Self::new(name, ChannelType::PointToPoint)
    }

    /// Validate.
    pub fn validate(&self) -> Result<(), IntercomError> {
        if self.name.is_empty() {
            return Err(IntercomError::EmptyChannelName);
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// User
// ────────────────────────────────────────────────────────────────────────────

/// A production crew member who can send and receive intercom audio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntercomUser {
    /// Display name of the user (e.g., "Director", "Camera Op 1").
    pub name: String,
    /// Default talk mode for new channel assignments.
    pub default_talk_mode: TalkMode,
    /// Whether this user's microphone is globally disabled.
    pub mic_disabled: bool,
    /// Whether this user's earpiece is globally silenced.
    pub listen_disabled: bool,
}

impl IntercomUser {
    /// Create a user with default settings.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            default_talk_mode: TalkMode::PressToTalk,
            mic_disabled: false,
            listen_disabled: false,
        }
    }

    /// Validate.
    pub fn validate(&self) -> Result<(), IntercomError> {
        if self.name.is_empty() {
            return Err(IntercomError::EmptyUserName);
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Channel membership record
// ────────────────────────────────────────────────────────────────────────────

/// Per-user state on a specific channel.
#[derive(Debug, Clone)]
struct ChannelMembership {
    #[allow(dead_code)]
    talk_mode: TalkMode,
    /// Whether the user is currently talking (mic active) on this channel.
    talking: bool,
    /// Whether the user is currently listening to this channel.
    listening: bool,
}

impl ChannelMembership {
    fn new(talk_mode: TalkMode) -> Self {
        Self {
            talk_mode,
            talking: false,
            listening: true,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// IFB cue event
// ────────────────────────────────────────────────────────────────────────────

/// An IFB cue sent from a director to a talent.
#[derive(Debug, Clone)]
pub struct IfbCue {
    /// User ID of the sender (director/PA).
    pub sender_id: usize,
    /// User ID of the recipient (talent).
    pub recipient_id: usize,
    /// Text annotation of the cue (optional).
    pub message: String,
    /// Duration hint in milliseconds (0 = until caller releases).
    pub duration_hint_ms: u64,
}

// ────────────────────────────────────────────────────────────────────────────
// Manager
// ────────────────────────────────────────────────────────────────────────────

/// Manages intercom channels, users, and IFB cue delivery.
#[derive(Debug)]
pub struct IntercomManager {
    channels: HashMap<usize, IntercomChannel>,
    users: HashMap<usize, IntercomUser>,
    /// membership[user_id][channel_id] = ChannelMembership
    membership: HashMap<usize, HashMap<usize, ChannelMembership>>,
    /// Active IFB cues (recipient_id -> cue)
    active_ifb: HashMap<usize, IfbCue>,
    next_channel_id: usize,
    next_user_id: usize,
}

impl IntercomManager {
    /// Create a new manager with no channels or users.
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
            users: HashMap::new(),
            membership: HashMap::new(),
            active_ifb: HashMap::new(),
            next_channel_id: 0,
            next_user_id: 0,
        }
    }

    // ── Channel management ───────────────────────────────────────────────────

    /// Create a new channel and return its ID.
    pub fn create_channel(&mut self, channel: IntercomChannel) -> Result<usize, IntercomError> {
        if self.channels.len() >= MAX_CHANNELS {
            return Err(IntercomError::ChannelLimitExceeded(MAX_CHANNELS));
        }
        channel.validate()?;
        let id = self.next_channel_id;
        self.next_channel_id += 1;
        self.channels.insert(id, channel);
        Ok(id)
    }

    /// Remove a channel.  All memberships on this channel are also removed.
    pub fn remove_channel(&mut self, channel_id: usize) -> Result<(), IntercomError> {
        if !self.channels.contains_key(&channel_id) {
            return Err(IntercomError::ChannelNotFound(channel_id));
        }
        self.channels.remove(&channel_id);
        for memberships in self.membership.values_mut() {
            memberships.remove(&channel_id);
        }
        Ok(())
    }

    /// Get an immutable reference to a channel.
    pub fn channel(&self, channel_id: usize) -> Option<&IntercomChannel> {
        self.channels.get(&channel_id)
    }

    /// Mute or un-mute a channel.
    pub fn set_channel_muted(
        &mut self,
        channel_id: usize,
        muted: bool,
    ) -> Result<(), IntercomError> {
        let ch = self
            .channels
            .get_mut(&channel_id)
            .ok_or(IntercomError::ChannelNotFound(channel_id))?;
        ch.muted = muted;
        Ok(())
    }

    /// Number of configured channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    // ── User management ──────────────────────────────────────────────────────

    /// Register a new user and return their ID.
    pub fn register_user(&mut self, user: IntercomUser) -> Result<usize, IntercomError> {
        if self.users.len() >= MAX_USERS {
            return Err(IntercomError::UserLimitExceeded(MAX_USERS));
        }
        user.validate()?;
        let id = self.next_user_id;
        self.next_user_id += 1;
        self.users.insert(id, user);
        self.membership.insert(id, HashMap::new());
        Ok(id)
    }

    /// Remove a user and all their channel memberships.
    pub fn remove_user(&mut self, user_id: usize) -> Result<(), IntercomError> {
        if !self.users.contains_key(&user_id) {
            return Err(IntercomError::UserNotFound(user_id));
        }
        self.users.remove(&user_id);
        self.membership.remove(&user_id);
        self.active_ifb.remove(&user_id);
        Ok(())
    }

    /// Get an immutable reference to a user.
    pub fn user(&self, user_id: usize) -> Option<&IntercomUser> {
        self.users.get(&user_id)
    }

    /// Number of registered users.
    pub fn user_count(&self) -> usize {
        self.users.len()
    }

    // ── Membership ───────────────────────────────────────────────────────────

    /// Assign a user to a channel.
    pub fn assign_user_to_channel(
        &mut self,
        user_id: usize,
        channel_id: usize,
    ) -> Result<(), IntercomError> {
        if !self.users.contains_key(&user_id) {
            return Err(IntercomError::UserNotFound(user_id));
        }
        if !self.channels.contains_key(&channel_id) {
            return Err(IntercomError::ChannelNotFound(channel_id));
        }
        let memberships = self.membership.entry(user_id).or_default();
        if memberships.contains_key(&channel_id) {
            return Err(IntercomError::AlreadyOnChannel(user_id, channel_id));
        }
        let talk_mode = self
            .users
            .get(&user_id)
            .map(|u| u.default_talk_mode)
            .unwrap_or(TalkMode::PressToTalk);
        memberships.insert(channel_id, ChannelMembership::new(talk_mode));
        Ok(())
    }

    /// Remove a user from a channel.
    pub fn unassign_user_from_channel(
        &mut self,
        user_id: usize,
        channel_id: usize,
    ) -> Result<(), IntercomError> {
        let memberships = self
            .membership
            .get_mut(&user_id)
            .ok_or(IntercomError::UserNotFound(user_id))?;
        if memberships.remove(&channel_id).is_none() {
            return Err(IntercomError::UserNotOnChannel(user_id, channel_id));
        }
        Ok(())
    }

    /// Return the IDs of all channels a user is assigned to.
    pub fn channels_for_user(&self, user_id: usize) -> Vec<usize> {
        self.membership
            .get(&user_id)
            .map(|m| m.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Return the IDs of all users assigned to a channel.
    pub fn users_on_channel(&self, channel_id: usize) -> Vec<usize> {
        self.membership
            .iter()
            .filter_map(|(uid, memberships)| {
                if memberships.contains_key(&channel_id) {
                    Some(*uid)
                } else {
                    None
                }
            })
            .collect()
    }

    // ── Talk / listen control ────────────────────────────────────────────────

    /// Activate the user's microphone on a channel (press-to-talk or latch).
    pub fn activate_talk(
        &mut self,
        user_id: usize,
        channel_id: usize,
    ) -> Result<(), IntercomError> {
        let memberships = self
            .membership
            .get_mut(&user_id)
            .ok_or(IntercomError::UserNotFound(user_id))?;
        let entry = memberships
            .get_mut(&channel_id)
            .ok_or(IntercomError::UserNotOnChannel(user_id, channel_id))?;
        entry.talking = true;
        Ok(())
    }

    /// Deactivate the user's microphone on a channel.
    pub fn deactivate_talk(
        &mut self,
        user_id: usize,
        channel_id: usize,
    ) -> Result<(), IntercomError> {
        let memberships = self
            .membership
            .get_mut(&user_id)
            .ok_or(IntercomError::UserNotFound(user_id))?;
        let entry = memberships
            .get_mut(&channel_id)
            .ok_or(IntercomError::UserNotOnChannel(user_id, channel_id))?;
        entry.talking = false;
        Ok(())
    }

    /// Returns `true` if the user is currently talking on the channel.
    pub fn is_talking(&self, user_id: usize, channel_id: usize) -> bool {
        self.membership
            .get(&user_id)
            .and_then(|m| m.get(&channel_id))
            .map(|e| e.talking)
            .unwrap_or(false)
    }

    /// Set whether a user is listening to a channel.
    pub fn set_listen(
        &mut self,
        user_id: usize,
        channel_id: usize,
        listening: bool,
    ) -> Result<(), IntercomError> {
        let memberships = self
            .membership
            .get_mut(&user_id)
            .ok_or(IntercomError::UserNotFound(user_id))?;
        let entry = memberships
            .get_mut(&channel_id)
            .ok_or(IntercomError::UserNotOnChannel(user_id, channel_id))?;
        entry.listening = listening;
        Ok(())
    }

    /// Returns `true` if the user is listening to a channel.
    pub fn is_listening(&self, user_id: usize, channel_id: usize) -> bool {
        self.membership
            .get(&user_id)
            .and_then(|m| m.get(&channel_id))
            .map(|e| e.listening)
            .unwrap_or(false)
    }

    /// Return all users currently talking on a channel.
    pub fn active_talkers(&self, channel_id: usize) -> Vec<usize> {
        self.membership
            .iter()
            .filter_map(|(uid, memberships)| {
                memberships
                    .get(&channel_id)
                    .filter(|e| e.talking)
                    .map(|_| *uid)
            })
            .collect()
    }

    // ── IFB cue handling ─────────────────────────────────────────────────────

    /// Send an IFB cue from `sender_id` to `recipient_id`.
    ///
    /// The recipient must be assigned to at least one `Ifb`-type channel.
    /// If a cue is already active for the recipient it is replaced.
    pub fn send_ifb_cue(&mut self, cue: IfbCue) -> Result<(), IntercomError> {
        // Verify both parties exist.
        if !self.users.contains_key(&cue.sender_id) {
            return Err(IntercomError::UserNotFound(cue.sender_id));
        }
        if !self.users.contains_key(&cue.recipient_id) {
            return Err(IntercomError::UserNotFound(cue.recipient_id));
        }
        // Verify recipient is on at least one IFB channel.
        let has_ifb_channel = self
            .membership
            .get(&cue.recipient_id)
            .map(|memberships| {
                memberships.keys().any(|ch_id| {
                    self.channels
                        .get(ch_id)
                        .map(|ch| ch.channel_type == ChannelType::Ifb)
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

        if !has_ifb_channel {
            return Err(IntercomError::NoIfbFeed(cue.recipient_id));
        }

        self.active_ifb.insert(cue.recipient_id, cue);
        Ok(())
    }

    /// Clear an active IFB cue for a recipient.
    pub fn clear_ifb_cue(&mut self, recipient_id: usize) {
        self.active_ifb.remove(&recipient_id);
    }

    /// Returns `true` if `recipient_id` has an active IFB cue.
    pub fn has_active_ifb_cue(&self, recipient_id: usize) -> bool {
        self.active_ifb.contains_key(&recipient_id)
    }

    /// Get the active IFB cue for a recipient, if any.
    pub fn active_ifb_cue(&self, recipient_id: usize) -> Option<&IfbCue> {
        self.active_ifb.get(&recipient_id)
    }

    // ── Broadcast / group actions ────────────────────────────────────────────

    /// Mute all channels.
    pub fn mute_all_channels(&mut self) {
        for ch in self.channels.values_mut() {
            ch.muted = true;
        }
    }

    /// Un-mute all channels.
    pub fn unmute_all_channels(&mut self) {
        for ch in self.channels.values_mut() {
            ch.muted = false;
        }
    }

    /// Return a set of all channel IDs that currently have active talkers.
    pub fn busy_channels(&self) -> HashSet<usize> {
        let mut busy = HashSet::new();
        for memberships in self.membership.values() {
            for (ch_id, entry) in memberships {
                if entry.talking {
                    busy.insert(*ch_id);
                }
            }
        }
        busy
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (IntercomManager, usize, usize, usize) {
        let mut mgr = IntercomManager::new();
        let ch = mgr
            .create_channel(IntercomChannel::partyline("Production"))
            .expect("channel ok");
        let dir = mgr
            .register_user(IntercomUser::new("Director"))
            .expect("user ok");
        let cam = mgr
            .register_user(IntercomUser::new("Camera1"))
            .expect("user ok");
        mgr.assign_user_to_channel(dir, ch).expect("assign ok");
        mgr.assign_user_to_channel(cam, ch).expect("assign ok");
        (mgr, ch, dir, cam)
    }

    #[test]
    fn test_create_channel_and_count() {
        let mut mgr = IntercomManager::new();
        mgr.create_channel(IntercomChannel::partyline("Prod"))
            .expect("ok");
        assert_eq!(mgr.channel_count(), 1);
    }

    #[test]
    fn test_register_user_and_count() {
        let mut mgr = IntercomManager::new();
        mgr.register_user(IntercomUser::new("Director"))
            .expect("ok");
        assert_eq!(mgr.user_count(), 1);
    }

    #[test]
    fn test_assign_and_query_membership() {
        let (mgr, ch, dir, cam) = setup();
        let users = mgr.users_on_channel(ch);
        assert!(users.contains(&dir));
        assert!(users.contains(&cam));
        assert_eq!(mgr.channels_for_user(dir), vec![ch]);
    }

    #[test]
    fn test_already_on_channel_error() {
        let (mut mgr, ch, dir, _cam) = setup();
        assert!(matches!(
            mgr.assign_user_to_channel(dir, ch),
            Err(IntercomError::AlreadyOnChannel(_, _))
        ));
    }

    #[test]
    fn test_activate_and_deactivate_talk() {
        let (mut mgr, ch, dir, _) = setup();
        mgr.activate_talk(dir, ch).expect("talk ok");
        assert!(mgr.is_talking(dir, ch));
        mgr.deactivate_talk(dir, ch).expect("stop talk ok");
        assert!(!mgr.is_talking(dir, ch));
    }

    #[test]
    fn test_listen_toggle() {
        let (mut mgr, ch, dir, _) = setup();
        // Default is listening=true.
        assert!(mgr.is_listening(dir, ch));
        mgr.set_listen(dir, ch, false).expect("ok");
        assert!(!mgr.is_listening(dir, ch));
    }

    #[test]
    fn test_active_talkers() {
        let (mut mgr, ch, dir, cam) = setup();
        mgr.activate_talk(dir, ch).expect("ok");
        let talkers = mgr.active_talkers(ch);
        assert!(talkers.contains(&dir));
        assert!(!talkers.contains(&cam));
    }

    #[test]
    fn test_busy_channels() {
        let (mut mgr, ch, dir, _) = setup();
        mgr.activate_talk(dir, ch).expect("ok");
        let busy = mgr.busy_channels();
        assert!(busy.contains(&ch));
    }

    #[test]
    fn test_mute_unmute_all() {
        let (mut mgr, ch, _, _) = setup();
        mgr.mute_all_channels();
        assert!(mgr.channel(ch).expect("exists").muted);
        mgr.unmute_all_channels();
        assert!(!mgr.channel(ch).expect("exists").muted);
    }

    #[test]
    fn test_ifb_cue_delivery() {
        let mut mgr = IntercomManager::new();
        let ifb_ch = mgr
            .create_channel(IntercomChannel::ifb("Director→Talent"))
            .expect("ok");
        let director = mgr
            .register_user(IntercomUser::new("Director"))
            .expect("ok");
        let talent = mgr.register_user(IntercomUser::new("Talent")).expect("ok");
        // Assign talent to the IFB channel.
        mgr.assign_user_to_channel(talent, ifb_ch).expect("ok");

        let cue = IfbCue {
            sender_id: director,
            recipient_id: talent,
            message: "Stand by".into(),
            duration_hint_ms: 2000,
        };
        mgr.send_ifb_cue(cue).expect("cue ok");
        assert!(mgr.has_active_ifb_cue(talent));
        assert_eq!(
            mgr.active_ifb_cue(talent).expect("exists").message,
            "Stand by"
        );

        mgr.clear_ifb_cue(talent);
        assert!(!mgr.has_active_ifb_cue(talent));
    }

    #[test]
    fn test_ifb_no_ifb_channel_error() {
        let mut mgr = IntercomManager::new();
        // Create a party-line channel (NOT IFB).
        let _ch = mgr
            .create_channel(IntercomChannel::partyline("Prod"))
            .expect("ok");
        let director = mgr
            .register_user(IntercomUser::new("Director"))
            .expect("ok");
        let talent = mgr.register_user(IntercomUser::new("Talent")).expect("ok");
        // Talent is NOT on any IFB channel.

        let cue = IfbCue {
            sender_id: director,
            recipient_id: talent,
            message: "Cue".into(),
            duration_hint_ms: 1000,
        };
        assert!(matches!(
            mgr.send_ifb_cue(cue),
            Err(IntercomError::NoIfbFeed(_))
        ));
    }

    #[test]
    fn test_remove_channel_cleans_up_membership() {
        let (mut mgr, ch, dir, _) = setup();
        mgr.remove_channel(ch).expect("ok");
        // User should no longer be on any channel.
        assert!(mgr.channels_for_user(dir).is_empty());
    }

    #[test]
    fn test_remove_user_cleans_up() {
        let (mut mgr, ch, dir, _) = setup();
        mgr.remove_user(dir).expect("ok");
        assert_eq!(mgr.user_count(), 1);
        let users = mgr.users_on_channel(ch);
        assert!(!users.contains(&dir));
    }

    #[test]
    fn test_empty_channel_name_error() {
        let mut mgr = IntercomManager::new();
        assert!(matches!(
            mgr.create_channel(IntercomChannel::partyline("")),
            Err(IntercomError::EmptyChannelName)
        ));
    }

    #[test]
    fn test_channel_type_properties() {
        assert!(ChannelType::PartyLine.allows_multiple_talkers());
        assert!(!ChannelType::PointToPoint.allows_multiple_talkers());
        assert!(!ChannelType::Ifb.allows_multiple_talkers());
    }

    #[test]
    fn test_user_not_on_channel_error() {
        let (mut mgr, ch, _, _) = setup();
        let outsider = mgr
            .register_user(IntercomUser::new("Outsider"))
            .expect("ok");
        assert!(matches!(
            mgr.activate_talk(outsider, ch),
            Err(IntercomError::UserNotOnChannel(_, _))
        ));
    }
}
