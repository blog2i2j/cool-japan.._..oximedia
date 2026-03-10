//! Professional live production video switcher for OxiMedia.
//!
//! This crate provides a comprehensive video switcher implementation with:
//! - Multi-source input management (SDI, NDI, files)
//! - Program/Preview bus architecture
//! - Transitions (cut, mix, wipe, DVE)
//! - Keying (luma, chroma, linear, pattern)
//! - Upstream and downstream keyers
//! - Multi-viewer monitoring
//! - Tally system
//! - Macro recording and playback
//! - Media pool for still frames
//! - Audio follow video (AFV)
//! - Frame synchronization and genlock
//!
//! # Example
//!
//! ```rust,no_run
//! use oximedia_switcher::{Switcher, SwitcherConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a professional switcher with 2 M/E rows
//! let config = SwitcherConfig::new(2, 8, 4);
//! let mut switcher = Switcher::new(config)?;
//!
//! // Set program and preview sources
//! switcher.set_program(0, 1)?;
//! switcher.set_preview(0, 2)?;
//!
//! // Perform a cut
//! switcher.cut(0)?;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(missing_docs)]

pub mod audio_follow;
pub mod audio_follow_video;
pub mod audio_mixer;
pub mod aux_bus;
pub mod bus;
pub mod chroma;
pub mod clip_delay;
pub mod crosspoint;
pub mod downstream_key;
pub mod dve;
pub mod ftb_control;
pub mod input;
pub mod input_bank;
pub mod input_manager;
pub mod keyer;
pub mod luma;
pub mod macro_engine;
pub mod macro_exec;
pub mod macro_system;
pub mod me_bank;
pub mod media_player;
pub mod media_pool;
pub mod multiviewer;
pub mod output_routing;
pub mod pattern_generator;
pub mod preview_bus;
pub mod still_store;
pub mod super_source;
pub mod switcher_preset;
pub mod sync;
pub mod tally;
pub mod tally_protocol;
pub mod tally_state;
pub mod tally_system;
pub mod transition;
pub mod transition_engine;
pub mod transition_lib;

use audio_follow::AudioFollowManager;
use bus::BusManager;
use input::InputRouter;
use keyer::KeyerManager;
use macro_engine::MacroEngine;
use media_pool::MediaPool;
use multiviewer::Multiviewer;
use sync::FrameSynchronizer;
use tally::TallyManager;
use transition::{TransitionConfig, TransitionEngine};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Re-export commonly used types.
pub use audio_follow::{AudioFollowError, AudioFollowMode};
pub use bus::{BusError, BusType};
pub use chroma::{ChromaColor, ChromaKey, ChromaKeyError, ChromaKeyParams};
pub use dve::{DveError, DveParams, DvePosition, DveScale};
pub use input::{InputConfig, InputError, InputType};
pub use keyer::{KeyerError, KeyerType};
pub use luma::{LumaKey, LumaKeyError, LumaKeyParams};
pub use macro_engine::{Macro, MacroCommand, MacroError};
pub use media_pool::{MediaPoolError, MediaSlot};
pub use multiviewer::{MultiviewerConfig, MultiviewerError, MultiviewerLayout};
pub use sync::{FrameRate, GenlockSource, SyncError};
pub use tally::{TallyError, TallyState};
pub use transition::{TransitionError, TransitionType, WipePattern};

/// Errors that can occur with switcher operations.
#[derive(Error, Debug)]
pub enum SwitcherError {
    /// Input error
    #[error("Input error: {0}")]
    Input(#[from] InputError),

    /// Bus error
    #[error("Bus error: {0}")]
    Bus(#[from] BusError),

    /// Keyer error
    #[error("Keyer error: {0}")]
    Keyer(#[from] KeyerError),

    /// Transition error
    #[error("Transition error: {0}")]
    Transition(#[from] TransitionError),

    /// Tally error
    #[error("Tally error: {0}")]
    Tally(#[from] TallyError),

    /// Sync error
    #[error("Sync error: {0}")]
    Sync(#[from] SyncError),

    /// Audio follow error
    #[error("Audio follow error: {0}")]
    AudioFollow(#[from] AudioFollowError),

    /// Macro error
    #[error("Macro error: {0}")]
    Macro(#[from] MacroError),

    /// Media pool error
    #[error("Media pool error: {0}")]
    MediaPool(#[from] MediaPoolError),

    /// Multiviewer error
    #[error("Multiviewer error: {0}")]
    Multiviewer(#[from] MultiviewerError),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
}

/// Switcher configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitcherConfig {
    /// Number of M/E (Mix/Effect) rows
    pub me_rows: usize,
    /// Number of inputs
    pub num_inputs: usize,
    /// Number of aux outputs
    pub num_aux: usize,
    /// Number of upstream keyers per M/E
    pub upstream_keyers_per_me: usize,
    /// Number of downstream keyers
    pub downstream_keyers: usize,
    /// Frame rate
    pub frame_rate: FrameRate,
    /// Media pool capacity
    pub media_pool_capacity: usize,
    /// Maximum number of macros
    pub max_macros: usize,
}

impl SwitcherConfig {
    /// Create a new switcher configuration.
    pub fn new(me_rows: usize, num_inputs: usize, num_aux: usize) -> Self {
        Self {
            me_rows,
            num_inputs,
            num_aux,
            upstream_keyers_per_me: 4,
            downstream_keyers: 2,
            frame_rate: FrameRate::Fps25,
            media_pool_capacity: 20,
            max_macros: 100,
        }
    }

    /// Create a basic 1 M/E configuration.
    pub fn basic() -> Self {
        Self::new(1, 8, 2)
    }

    /// Create a professional 2 M/E configuration.
    pub fn professional() -> Self {
        Self::new(2, 20, 6)
    }

    /// Create a broadcast 4 M/E configuration.
    pub fn broadcast() -> Self {
        Self::new(4, 40, 10)
    }
}

impl Default for SwitcherConfig {
    fn default() -> Self {
        Self::basic()
    }
}

/// Main switcher engine.
pub struct Switcher {
    config: SwitcherConfig,
    bus_manager: BusManager,
    input_router: InputRouter,
    keyer_manager: KeyerManager,
    transition_engines: Vec<TransitionEngine>,
    tally_manager: TallyManager,
    sync: FrameSynchronizer,
    audio_follow: AudioFollowManager,
    macro_engine: MacroEngine,
    media_pool: MediaPool,
    multiviewer: Option<Multiviewer>,
}

impl Switcher {
    /// Create a new switcher.
    pub fn new(config: SwitcherConfig) -> Result<Self, SwitcherError> {
        let bus_manager = BusManager::new(config.me_rows, config.num_aux);
        let input_router = InputRouter::new(config.num_inputs);

        let num_keyers = config.me_rows * config.upstream_keyers_per_me;
        let keyer_manager = KeyerManager::new(num_keyers, config.downstream_keyers);

        let mut transition_engines = Vec::new();
        for _ in 0..config.me_rows {
            transition_engines.push(TransitionEngine::new());
        }

        let tally_manager = TallyManager::new();
        let sync = FrameSynchronizer::new(config.frame_rate, config.num_inputs);
        let audio_follow = AudioFollowManager::new(config.num_aux);
        let macro_engine = MacroEngine::new(config.max_macros);
        let media_pool = MediaPool::new(config.media_pool_capacity);

        Ok(Self {
            config,
            bus_manager,
            input_router,
            keyer_manager,
            transition_engines,
            tally_manager,
            sync,
            audio_follow,
            macro_engine,
            media_pool,
            multiviewer: None,
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &SwitcherConfig {
        &self.config
    }

    /// Get the bus manager.
    pub fn bus_manager(&self) -> &BusManager {
        &self.bus_manager
    }

    /// Get mutable bus manager.
    pub fn bus_manager_mut(&mut self) -> &mut BusManager {
        &mut self.bus_manager
    }

    /// Get the input router.
    pub fn input_router(&self) -> &InputRouter {
        &self.input_router
    }

    /// Get mutable input router.
    pub fn input_router_mut(&mut self) -> &mut InputRouter {
        &mut self.input_router
    }

    /// Get the keyer manager.
    pub fn keyer_manager(&self) -> &KeyerManager {
        &self.keyer_manager
    }

    /// Get mutable keyer manager.
    pub fn keyer_manager_mut(&mut self) -> &mut KeyerManager {
        &mut self.keyer_manager
    }

    /// Get the tally manager.
    pub fn tally_manager(&self) -> &TallyManager {
        &self.tally_manager
    }

    /// Get mutable tally manager.
    pub fn tally_manager_mut(&mut self) -> &mut TallyManager {
        &mut self.tally_manager
    }

    /// Get the frame synchronizer.
    pub fn sync(&self) -> &FrameSynchronizer {
        &self.sync
    }

    /// Get mutable frame synchronizer.
    pub fn sync_mut(&mut self) -> &mut FrameSynchronizer {
        &mut self.sync
    }

    /// Get the audio follow manager.
    pub fn audio_follow(&self) -> &AudioFollowManager {
        &self.audio_follow
    }

    /// Get mutable audio follow manager.
    pub fn audio_follow_mut(&mut self) -> &mut AudioFollowManager {
        &mut self.audio_follow
    }

    /// Get the macro engine.
    pub fn macro_engine(&self) -> &MacroEngine {
        &self.macro_engine
    }

    /// Get mutable macro engine.
    pub fn macro_engine_mut(&mut self) -> &mut MacroEngine {
        &mut self.macro_engine
    }

    /// Get the media pool.
    pub fn media_pool(&self) -> &MediaPool {
        &self.media_pool
    }

    /// Get mutable media pool.
    pub fn media_pool_mut(&mut self) -> &mut MediaPool {
        &mut self.media_pool
    }

    /// Get the multiviewer.
    pub fn multiviewer(&self) -> Option<&Multiviewer> {
        self.multiviewer.as_ref()
    }

    /// Get mutable multiviewer.
    pub fn multiviewer_mut(&mut self) -> Option<&mut Multiviewer> {
        self.multiviewer.as_mut()
    }

    /// Set the multiviewer configuration.
    pub fn set_multiviewer(&mut self, multiviewer: Multiviewer) {
        self.multiviewer = Some(multiviewer);
    }

    /// Set program source for an M/E row.
    pub fn set_program(&mut self, me_row: usize, input: usize) -> Result<(), SwitcherError> {
        self.bus_manager.set_program(me_row, input)?;
        self.update_tally();
        Ok(())
    }

    /// Set preview source for an M/E row.
    pub fn set_preview(&mut self, me_row: usize, input: usize) -> Result<(), SwitcherError> {
        self.bus_manager.set_preview(me_row, input)?;
        self.update_tally();
        Ok(())
    }

    /// Perform a cut on an M/E row.
    pub fn cut(&mut self, me_row: usize) -> Result<(), SwitcherError> {
        self.bus_manager.cut(me_row)?;
        self.update_tally();
        Ok(())
    }

    /// Perform an auto transition on an M/E row.
    pub fn auto_transition(&mut self, me_row: usize) -> Result<(), SwitcherError> {
        if me_row >= self.transition_engines.len() {
            return Err(SwitcherError::Bus(BusError::InvalidBusId(me_row)));
        }

        let program = self.bus_manager.get_program(me_row)?;
        let preview = self.bus_manager.get_preview(me_row)?;

        self.bus_manager.set_transition_active(me_row, true)?;
        self.transition_engines[me_row].start(program, preview)?;

        Ok(())
    }

    /// Get a transition engine for an M/E row.
    pub fn transition_engine(&self, me_row: usize) -> Option<&TransitionEngine> {
        self.transition_engines.get(me_row)
    }

    /// Get a mutable transition engine for an M/E row.
    pub fn transition_engine_mut(&mut self, me_row: usize) -> Option<&mut TransitionEngine> {
        self.transition_engines.get_mut(me_row)
    }

    /// Set transition configuration for an M/E row.
    pub fn set_transition_config(
        &mut self,
        me_row: usize,
        config: TransitionConfig,
    ) -> Result<(), SwitcherError> {
        if let Some(engine) = self.transition_engines.get_mut(me_row) {
            engine.set_config(config);
            Ok(())
        } else {
            Err(SwitcherError::Bus(BusError::InvalidBusId(me_row)))
        }
    }

    /// Update tally states based on current bus assignments.
    pub fn update_tally(&mut self) {
        let program = self.bus_manager.get_all_program();
        let preview = self.bus_manager.get_all_preview();

        self.tally_manager.update_from_buses(program, preview);

        // Update multiviewer tally if enabled
        if let Some(mv) = &mut self.multiviewer {
            for (&input_id, &state) in &self.tally_manager.get_all_states() {
                mv.update_tally(input_id, state);
            }
        }
    }

    /// Process one frame of the switcher.
    pub fn process_frame(&mut self) -> Result<(), SwitcherError> {
        // Advance synchronizer
        self.sync.advance_frame();

        // Process any active transitions
        let mut tally_needs_update = false;
        for (me_row, engine) in self.transition_engines.iter_mut().enumerate() {
            if engine.is_in_progress() && engine.advance()? {
                // Transition complete
                self.bus_manager.take(me_row)?;
                self.bus_manager.set_transition_active(me_row, false)?;
                tally_needs_update = true;
            }
        }

        if tally_needs_update {
            self.update_tally();
        }

        // Process macro playback
        if let Some(command) = self.macro_engine.player_mut().next_command() {
            self.execute_macro_command(command)?;
        }

        Ok(())
    }

    /// Execute a macro command.
    fn execute_macro_command(&mut self, command: MacroCommand) -> Result<(), SwitcherError> {
        match command {
            MacroCommand::SelectProgram { me_row, input } => {
                self.set_program(me_row, input)?;
            }
            MacroCommand::SelectPreview { me_row, input } => {
                self.set_preview(me_row, input)?;
            }
            MacroCommand::Cut { me_row } => {
                self.cut(me_row)?;
            }
            MacroCommand::Auto { me_row } => {
                self.auto_transition(me_row)?;
            }
            MacroCommand::SetKeyerOnAir { keyer_id, on_air } => {
                self.keyer_manager
                    .get_upstream_mut(keyer_id)?
                    .set_on_air(on_air);
            }
            MacroCommand::SetDskOnAir { dsk_id, on_air } => {
                self.keyer_manager
                    .get_downstream_mut(dsk_id)?
                    .set_on_air(on_air);
            }
            MacroCommand::SelectAux { aux_id, input } => {
                self.bus_manager.set_aux(aux_id, input)?;
            }
            MacroCommand::SetTransition {
                me_row,
                transition_type,
            } => {
                // Map the string transition type name to a TransitionConfig and
                // apply it to the correct M/E row transition engine.
                let config = match transition_type.to_ascii_lowercase().as_str() {
                    "mix" | "dissolve" => TransitionConfig::mix(30),
                    "cut" => TransitionConfig::cut(),
                    "wipe" | "wipe_horizontal" => {
                        TransitionConfig::wipe(transition::WipePattern::Horizontal, 30)
                    }
                    "wipe_vertical" => {
                        TransitionConfig::wipe(transition::WipePattern::Vertical, 30)
                    }
                    "wipe_diagonal" => {
                        TransitionConfig::wipe(transition::WipePattern::DiagonalTopLeft, 30)
                    }
                    _ => TransitionConfig::mix(30),
                };
                self.set_transition_config(me_row, config)?;
            }
            MacroCommand::LoadMediaPool { slot_id } => {
                // Mark the requested media pool slot as occupied/ready so that
                // downstream compositing can use it.  If the slot does not yet
                // exist in the pool (it may have been pre-allocated) we treat
                // the operation as a no-op because there is no filesystem path
                // available at macro-execution time.
                if self.media_pool.has_slot(slot_id) {
                    let slot = self.media_pool.get_slot_mut(slot_id)?;
                    slot.set_occupied(true);
                }
                // No error if the slot is absent — the macro may have been
                // authored before the pool was fully populated.
            }
            MacroCommand::Wait { duration_ms: _ } => {
                // Wait commands are handled by the MacroPlayer's internal
                // timing mechanism (next_command() only returns the next
                // command after the delay has elapsed).  By the time we
                // receive a Wait command here it has already been consumed
                // by the player, so nothing further is required.
            }
            MacroCommand::RunMacro { macro_id } => {
                // Trigger playback of another macro by ID.  We clone the
                // macro out of the engine first to avoid the borrow checker
                // conflict between `self.macro_engine` (borrow for
                // `get_macro`) and `self.macro_engine.player_mut()`.
                self.macro_engine.run_macro(macro_id).map_err(|e| {
                    SwitcherError::Config(format!("RunMacro {macro_id} failed: {e}"))
                })?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_switcher_config_basic() {
        let config = SwitcherConfig::basic();
        assert_eq!(config.me_rows, 1);
        assert_eq!(config.num_inputs, 8);
        assert_eq!(config.num_aux, 2);
    }

    #[test]
    fn test_switcher_config_professional() {
        let config = SwitcherConfig::professional();
        assert_eq!(config.me_rows, 2);
        assert_eq!(config.num_inputs, 20);
    }

    #[test]
    fn test_switcher_creation() {
        let config = SwitcherConfig::basic();
        let switcher = Switcher::new(config).expect("should succeed in test");

        assert_eq!(switcher.config().me_rows, 1);
        assert_eq!(switcher.bus_manager().me_rows(), 1);
    }

    #[test]
    fn test_switcher_set_program_preview() {
        let config = SwitcherConfig::basic();
        let mut switcher = Switcher::new(config).expect("should succeed in test");

        switcher.set_program(0, 1).expect("should succeed in test");
        switcher.set_preview(0, 2).expect("should succeed in test");

        assert_eq!(
            switcher
                .bus_manager()
                .get_program(0)
                .expect("should succeed in test"),
            1
        );
        assert_eq!(
            switcher
                .bus_manager()
                .get_preview(0)
                .expect("should succeed in test"),
            2
        );
    }

    #[test]
    fn test_switcher_cut() {
        let config = SwitcherConfig::basic();
        let mut switcher = Switcher::new(config).expect("should succeed in test");

        switcher.set_program(0, 1).expect("should succeed in test");
        switcher.set_preview(0, 2).expect("should succeed in test");

        switcher.cut(0).expect("should succeed in test");

        // After cut, program and preview should be swapped
        assert_eq!(
            switcher
                .bus_manager()
                .get_program(0)
                .expect("should succeed in test"),
            2
        );
        assert_eq!(
            switcher
                .bus_manager()
                .get_preview(0)
                .expect("should succeed in test"),
            1
        );
    }

    #[test]
    fn test_switcher_tally_update() {
        let config = SwitcherConfig::basic();
        let mut switcher = Switcher::new(config).expect("should succeed in test");

        switcher.set_program(0, 1).expect("should succeed in test");
        switcher.set_preview(0, 2).expect("should succeed in test");

        let tally_1 = switcher.tally_manager().get_tally(1);
        let tally_2 = switcher.tally_manager().get_tally(2);

        assert!(tally_1.is_program());
        assert!(tally_2.is_preview());
    }

    #[test]
    fn test_switcher_transition_config() {
        let config = SwitcherConfig::basic();
        let mut switcher = Switcher::new(config).expect("should succeed in test");

        let transition_config = TransitionConfig::mix(30);
        switcher
            .set_transition_config(0, transition_config)
            .expect("should succeed in test");

        let engine = switcher
            .transition_engine(0)
            .expect("should succeed in test");
        assert_eq!(engine.duration_frames(), 30);
    }

    #[test]
    fn test_switcher_auto_transition() {
        let config = SwitcherConfig::basic();
        let mut switcher = Switcher::new(config).expect("should succeed in test");

        switcher.set_program(0, 1).expect("should succeed in test");
        switcher.set_preview(0, 2).expect("should succeed in test");

        let transition_config = TransitionConfig::mix(10);
        switcher
            .set_transition_config(0, transition_config)
            .expect("should succeed in test");

        switcher.auto_transition(0).expect("should succeed in test");

        let engine = switcher
            .transition_engine(0)
            .expect("should succeed in test");
        assert!(engine.is_in_progress());
    }

    #[test]
    fn test_switcher_process_frame() {
        let config = SwitcherConfig::basic();
        let mut switcher = Switcher::new(config).expect("should succeed in test");

        assert!(switcher.process_frame().is_ok());
        assert_eq!(switcher.sync().current_frame(), 1);
    }
}
