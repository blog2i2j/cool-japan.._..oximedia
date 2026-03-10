//! Sound effects library management.
//!
//! Provides a searchable library for sound effects with tagging,
//! duration search, and category-based random selection.

#![allow(dead_code)]

/// Category of a sound effect.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SfxCategory {
    /// Ambient environmental sounds.
    Ambience,
    /// Foley / practical sound effects.
    Foley,
    /// Impact sounds (hits, crashes, etc.).
    Impact,
    /// Mechanical sounds (machines, vehicles).
    Mechanical,
    /// Human sounds (breathing, footsteps, etc.).
    Human,
    /// Nature sounds (wind, rain, birds).
    Nature,
    /// Electronic / digital sounds.
    Electronic,
    /// Musical stings, beds, and motifs.
    Musical,
}

/// A single sound effect in the library.
#[derive(Debug, Clone)]
pub struct SoundEffect {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Category of this sound effect.
    pub category: SfxCategory,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u8,
    /// Searchable tags.
    pub tags: Vec<String>,
    /// Optional BPM for musical / rhythmic SFX.
    pub bpm: Option<f32>,
}

impl SoundEffect {
    /// Create a new sound effect.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        category: SfxCategory,
        duration_secs: f64,
        sample_rate: u32,
        channels: u8,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            category,
            duration_secs,
            sample_rate,
            channels,
            tags: Vec::new(),
            bpm: None,
        }
    }

    /// Add a tag.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add multiple tags.
    #[must_use]
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    /// Set BPM.
    #[must_use]
    pub fn with_bpm(mut self, bpm: f32) -> Self {
        self.bpm = Some(bpm);
        self
    }

    /// Check if the sound effect has a given tag (case-insensitive).
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        let tag_lower = tag.to_lowercase();
        self.tags.iter().any(|t| t.to_lowercase() == tag_lower)
    }
}

/// Searchable sound effects library.
#[derive(Debug, Default)]
pub struct SoundLibrary {
    effects: Vec<SoundEffect>,
}

impl SoundLibrary {
    /// Create a new empty library.
    #[must_use]
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    /// Add a sound effect to the library.
    pub fn add_sfx(&mut self, sfx: SoundEffect) {
        self.effects.push(sfx);
    }

    /// Total number of effects in the library.
    #[must_use]
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Check if the library is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Search for sound effects that have ALL of the given tags.
    #[must_use]
    pub fn search_by_tag(&self, tags: &[&str]) -> Vec<&SoundEffect> {
        self.effects
            .iter()
            .filter(|sfx| tags.iter().all(|&tag| sfx.has_tag(tag)))
            .collect()
    }

    /// Search for sound effects within a duration range (inclusive).
    #[must_use]
    pub fn search_by_duration(&self, min_secs: f64, max_secs: f64) -> Vec<&SoundEffect> {
        self.effects
            .iter()
            .filter(|sfx| sfx.duration_secs >= min_secs && sfx.duration_secs <= max_secs)
            .collect()
    }

    /// Search by category.
    #[must_use]
    pub fn search_by_category(&self, category: &SfxCategory) -> Vec<&SoundEffect> {
        self.effects
            .iter()
            .filter(|sfx| &sfx.category == category)
            .collect()
    }

    /// Get a random sound effect from a category using a deterministic seed.
    #[must_use]
    pub fn random_from_category(&self, category: &SfxCategory, seed: u64) -> Option<&SoundEffect> {
        let candidates: Vec<&SoundEffect> = self.search_by_category(category);
        if candidates.is_empty() {
            return None;
        }
        // Simple LCG-based selection from seed for determinism
        let idx = lcg_rand(seed) as usize % candidates.len();
        Some(candidates[idx])
    }

    /// Search by sample rate.
    #[must_use]
    pub fn search_by_sample_rate(&self, sample_rate: u32) -> Vec<&SoundEffect> {
        self.effects
            .iter()
            .filter(|sfx| sfx.sample_rate == sample_rate)
            .collect()
    }

    /// Get a sound effect by ID.
    #[must_use]
    pub fn get_by_id(&self, id: &str) -> Option<&SoundEffect> {
        self.effects.iter().find(|sfx| sfx.id == id)
    }

    /// Get all effects.
    #[must_use]
    pub fn all(&self) -> &[SoundEffect] {
        &self.effects
    }
}

/// Simple LCG pseudo-random number generator for deterministic selection.
fn lcg_rand(seed: u64) -> u64 {
    const A: u64 = 6_364_136_223_846_793_005;
    const C: u64 = 1_442_695_040_888_963_407;
    seed.wrapping_mul(A).wrapping_add(C)
}

/// A layering plan for ambient sound design.
#[derive(Debug, Clone)]
pub struct SfxLayeringPlan {
    /// Layers as (sfx_id, gain_db) pairs.
    pub layers: Vec<(String, f32)>,
}

impl SfxLayeringPlan {
    /// Create an empty layering plan.
    #[must_use]
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add a layer.
    pub fn add_layer(&mut self, sfx_id: impl Into<String>, gain_db: f32) {
        self.layers.push((sfx_id.into(), gain_db));
    }

    /// Total number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl Default for SfxLayeringPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Plans layering of sound effects for ambience design.
pub struct SfxLayeringPlanner;

impl SfxLayeringPlanner {
    /// Create an ambience layering plan with `num_layers` from a given category.
    ///
    /// Uses `seed` for deterministic selection. Layers get graduated gain
    /// values from -6 dB (primary) to -18 dB (background layers).
    #[must_use]
    pub fn plan_ambience(
        library: &SoundLibrary,
        category: SfxCategory,
        num_layers: usize,
        seed: u64,
    ) -> SfxLayeringPlan {
        let mut plan = SfxLayeringPlan::new();
        let candidates = library.search_by_category(&category);

        if candidates.is_empty() || num_layers == 0 {
            return plan;
        }

        let mut current_seed = seed;
        for i in 0..num_layers {
            let idx = lcg_rand(current_seed) as usize % candidates.len();
            current_seed = lcg_rand(current_seed);

            let sfx = candidates[idx];
            // Primary layer at -6 dB, each subsequent layer drops by 3 dB, min -24 dB
            let gain_db = (-6.0 - (i as f32 * 3.0)).max(-24.0);
            plan.add_layer(sfx.id.clone(), gain_db);
        }

        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_library() -> SoundLibrary {
        let mut lib = SoundLibrary::new();

        lib.add_sfx(
            SoundEffect::new(
                "sfx-001",
                "Forest Wind",
                SfxCategory::Ambience,
                30.0,
                48000,
                2,
            )
            .with_tags(["wind", "forest", "outdoor"]),
        );
        lib.add_sfx(
            SoundEffect::new(
                "sfx-002",
                "Rain on Glass",
                SfxCategory::Nature,
                60.0,
                48000,
                2,
            )
            .with_tags(["rain", "indoor", "calm"]),
        );
        lib.add_sfx(
            SoundEffect::new("sfx-003", "Door Knock", SfxCategory::Foley, 1.5, 48000, 1)
                .with_tags(["knock", "door", "indoor"]),
        );
        lib.add_sfx(
            SoundEffect::new(
                "sfx-004",
                "Thunder Crack",
                SfxCategory::Impact,
                4.0,
                96000,
                2,
            )
            .with_tags(["thunder", "outdoor", "impact"]),
        );
        lib.add_sfx(
            SoundEffect::new(
                "sfx-005",
                "City Ambience",
                SfxCategory::Ambience,
                120.0,
                44100,
                2,
            )
            .with_tags(["city", "outdoor", "traffic"]),
        );

        lib
    }

    #[test]
    fn test_library_add_and_count() {
        let lib = make_library();
        assert_eq!(lib.len(), 5);
        assert!(!lib.is_empty());
    }

    #[test]
    fn test_search_by_single_tag() {
        let lib = make_library();
        let results = lib.search_by_tag(&["outdoor"]);
        assert_eq!(results.len(), 3); // Forest Wind, Thunder Crack, City Ambience
    }

    #[test]
    fn test_search_by_multiple_tags() {
        let lib = make_library();
        let results = lib.search_by_tag(&["outdoor", "impact"]);
        assert_eq!(results.len(), 1); // Thunder Crack only
    }

    #[test]
    fn test_search_by_duration_range() {
        let lib = make_library();
        let results = lib.search_by_duration(1.0, 10.0);
        assert_eq!(results.len(), 2); // Door Knock (1.5) and Thunder Crack (4.0)
    }

    #[test]
    fn test_search_by_duration_all() {
        let lib = make_library();
        let results = lib.search_by_duration(0.0, 9999.0);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_search_by_category() {
        let lib = make_library();
        let results = lib.search_by_category(&SfxCategory::Ambience);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_random_from_category() {
        let lib = make_library();
        let result = lib.random_from_category(&SfxCategory::Ambience, 42);
        assert!(result.is_some());
        assert_eq!(
            result.expect("result should be valid").category,
            SfxCategory::Ambience
        );
    }

    #[test]
    fn test_random_from_empty_category() {
        let lib = make_library();
        let result = lib.random_from_category(&SfxCategory::Electronic, 42);
        assert!(result.is_none());
    }

    #[test]
    fn test_get_by_id() {
        let lib = make_library();
        let sfx = lib.get_by_id("sfx-003");
        assert!(sfx.is_some());
        assert_eq!(sfx.expect("sfx should be valid").name, "Door Knock");
    }

    #[test]
    fn test_get_by_id_not_found() {
        let lib = make_library();
        assert!(lib.get_by_id("nonexistent").is_none());
    }

    #[test]
    fn test_plan_ambience_layering() {
        let lib = make_library();
        let plan = SfxLayeringPlanner::plan_ambience(&lib, SfxCategory::Ambience, 2, 123);
        assert_eq!(plan.num_layers(), 2);
        // First layer should be primary (-6 dB)
        assert!((plan.layers[0].1 - (-6.0)).abs() < 0.01);
        // Second layer lower
        assert!(plan.layers[1].1 < plan.layers[0].1);
    }

    #[test]
    fn test_plan_ambience_empty_category() {
        let lib = make_library();
        let plan = SfxLayeringPlanner::plan_ambience(&lib, SfxCategory::Musical, 3, 0);
        assert_eq!(plan.num_layers(), 0);
    }

    #[test]
    fn test_plan_ambience_zero_layers() {
        let lib = make_library();
        let plan = SfxLayeringPlanner::plan_ambience(&lib, SfxCategory::Ambience, 0, 0);
        assert_eq!(plan.num_layers(), 0);
    }

    #[test]
    fn test_sound_effect_tags() {
        let sfx = SoundEffect::new("id", "name", SfxCategory::Foley, 1.0, 48000, 1)
            .with_tag("footstep")
            .with_tag("gravel");
        assert!(sfx.has_tag("footstep"));
        assert!(sfx.has_tag("GRAVEL")); // case insensitive
        assert!(!sfx.has_tag("sand"));
    }

    #[test]
    fn test_sound_effect_with_bpm() {
        let sfx = SoundEffect::new("id", "Beat Loop", SfxCategory::Musical, 4.0, 44100, 2)
            .with_bpm(120.0);
        assert_eq!(sfx.bpm, Some(120.0));
    }
}
