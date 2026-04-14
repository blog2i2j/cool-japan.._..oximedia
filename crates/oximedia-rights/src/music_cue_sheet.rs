//! Music cue sheet management for production and broadcast.
//!
//! A music cue sheet documents every piece of music used in a film, TV
//! programme, advertisement, or other production.  Rights organisations
//! (PROs) such as ASCAP, BMI, SOCAN, and PRS require cue sheets to
//! calculate and distribute performance royalties to composers and publishers.
//!
//! # Features
//!
//! * Structured [`CueEntry`] type covering all mandatory PRO fields.
//! * [`CueSheet`] container with builder-style helpers.
//! * Generation of cue sheets from an edit timeline (ordered list of timed
//!   events).
//! * Export to a tab-separated PRO submission text format.
//! * Validation rules that mirror ASCAP/BMI requirements.

#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{Result, RightsError};

// ── UsageType ────────────────────────────────────────────────────────────────

/// How a music cue is used within the production.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsageType {
    /// Background underscore not directly tied to the action.
    BackgroundInstrumental,
    /// Background music with lyrics (source or library).
    BackgroundVocal,
    /// Music used as the main visual/auditory theme.
    Theme,
    /// An on-screen performance (singer, band, etc.).
    VisualVocal,
    /// An on-screen instrumental performance.
    VisualInstrumental,
    /// Opening or closing title sequence music.
    TitleSequence,
    /// Logo or station-identifier sting.
    Sting,
    /// Transitional music between segments.
    Transition,
    /// Music used in a commercial or advertisement within the production.
    CommercialSpot,
    /// User-defined usage category.
    Custom(String),
}

impl UsageType {
    /// Return the two-letter ASCAP/BMI usage type code where applicable.
    #[must_use]
    pub fn pro_code(&self) -> &str {
        match self {
            Self::BackgroundInstrumental => "BI",
            Self::BackgroundVocal => "BV",
            Self::Theme => "TH",
            Self::VisualVocal => "VV",
            Self::VisualInstrumental => "VI",
            Self::TitleSequence => "TS",
            Self::Sting => "ST",
            Self::Transition => "TR",
            Self::CommercialSpot => "CS",
            Self::Custom(_) => "XX",
        }
    }
}

impl std::fmt::Display for UsageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Custom(s) => write!(f, "{s}"),
            other => write!(f, "{}", other.pro_code()),
        }
    }
}

// ── Contributor ──────────────────────────────────────────────────────────────

/// A creative contributor to a musical work (composer, lyricist, publisher).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contributor {
    /// Full legal name of the contributor.
    pub name: String,
    /// Role: "Composer", "Lyricist", "Publisher", etc.
    pub role: String,
    /// Performing-rights-society affiliation (e.g. "ASCAP", "BMI", "SOCAN").
    pub pro_affiliation: Option<String>,
    /// IPI (Interested Party Information) number assigned by CISAC.
    pub ipi_number: Option<String>,
    /// Ownership share as a percentage (0.0 – 100.0).
    /// All contributors' shares for a work should sum to 100.0 on each side
    /// (writer side / publisher side).
    pub ownership_share: f64,
}

impl Contributor {
    /// Create a new contributor with a given name, role, and ownership share.
    pub fn new(
        name: impl Into<String>,
        role: impl Into<String>,
        ownership_share: f64,
    ) -> Result<Self> {
        let share = ownership_share;
        if !(0.0..=100.0).contains(&share) {
            return Err(RightsError::InvalidOperation(format!(
                "ownership_share must be in [0, 100], got {share}"
            )));
        }
        Ok(Self {
            name: name.into(),
            role: role.into(),
            pro_affiliation: None,
            ipi_number: None,
            ownership_share: share,
        })
    }

    /// Set the PRO affiliation.
    #[must_use]
    pub fn with_pro(mut self, pro: impl Into<String>) -> Self {
        self.pro_affiliation = Some(pro.into());
        self
    }

    /// Set the IPI number.
    #[must_use]
    pub fn with_ipi(mut self, ipi: impl Into<String>) -> Self {
        self.ipi_number = Some(ipi.into());
        self
    }
}

// ── CueEntry ─────────────────────────────────────────────────────────────────

/// A single music cue within a production.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CueEntry {
    /// Sequential cue number within the cue sheet (1-based).
    pub cue_number: u32,
    /// Title of the musical work as registered with the relevant PRO.
    pub title: String,
    /// ISRC of the master recording, if applicable.
    pub isrc: Option<String>,
    /// ISWC (International Standard Musical Work Code) of the composition.
    pub iswc: Option<String>,
    /// Duration of the cue in seconds.
    pub duration_secs: f64,
    /// How the cue is used in the production.
    pub usage_type: UsageType,
    /// Timecode (HH:MM:SS:FF) of the cue's in-point in the production.
    pub timecode_in: Option<String>,
    /// Timecode (HH:MM:SS:FF) of the cue's out-point in the production.
    pub timecode_out: Option<String>,
    /// Contributors (composers, lyricists, publishers).
    pub contributors: Vec<Contributor>,
    /// Library / catalogue reference if the cue is from a production library.
    pub library_reference: Option<String>,
    /// Free-form notes.
    pub notes: Option<String>,
}

impl CueEntry {
    /// Create a minimal cue entry.
    pub fn new(
        cue_number: u32,
        title: impl Into<String>,
        usage_type: UsageType,
        duration_secs: f64,
    ) -> Result<Self> {
        if duration_secs < 0.0 {
            return Err(RightsError::InvalidOperation(format!(
                "duration_secs must be non-negative, got {duration_secs}"
            )));
        }
        Ok(Self {
            cue_number,
            title: title.into(),
            isrc: None,
            iswc: None,
            duration_secs,
            usage_type,
            timecode_in: None,
            timecode_out: None,
            contributors: Vec::new(),
            library_reference: None,
            notes: None,
        })
    }

    /// Add a contributor to this cue.
    pub fn add_contributor(&mut self, contributor: Contributor) {
        self.contributors.push(contributor);
    }

    /// Return the total ownership share for a given role (e.g. "Composer").
    #[must_use]
    pub fn total_share_for_role(&self, role: &str) -> f64 {
        self.contributors
            .iter()
            .filter(|c| c.role.eq_ignore_ascii_case(role))
            .map(|c| c.ownership_share)
            .sum()
    }

    /// Format the duration as `MM:SS`.
    #[must_use]
    pub fn formatted_duration(&self) -> String {
        let total = self.duration_secs as u64;
        let mins = total / 60;
        let secs = total % 60;
        format!("{mins:02}:{secs:02}")
    }

    /// Validate this cue entry against common PRO requirements.
    pub fn validate(&self) -> Result<()> {
        if self.title.trim().is_empty() {
            return Err(RightsError::InvalidOperation(
                "cue title must not be empty".into(),
            ));
        }
        if self.contributors.is_empty() {
            return Err(RightsError::InvalidOperation(format!(
                "cue {} has no contributors",
                self.cue_number
            )));
        }
        Ok(())
    }
}

// ── ProductionType ────────────────────────────────────────────────────────────

/// The type of production to which the cue sheet belongs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProductionType {
    /// Feature film for theatrical release.
    Film,
    /// Television episode.
    Television,
    /// Video game.
    VideoGame,
    /// Advertisement / commercial.
    Advertisement,
    /// Online / streaming-only content.
    OnlineContent,
    /// Short-form web video.
    WebVideo,
    /// Podcast episode.
    Podcast,
    /// Other / unspecified.
    Other(String),
}

impl std::fmt::Display for ProductionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Film => write!(f, "Film"),
            Self::Television => write!(f, "Television"),
            Self::VideoGame => write!(f, "Video Game"),
            Self::Advertisement => write!(f, "Advertisement"),
            Self::OnlineContent => write!(f, "Online Content"),
            Self::WebVideo => write!(f, "Web Video"),
            Self::Podcast => write!(f, "Podcast"),
            Self::Other(s) => write!(f, "{s}"),
        }
    }
}

// ── CueSheet ──────────────────────────────────────────────────────────────────

/// A complete music cue sheet for a single production or episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CueSheet {
    /// Unique identifier for this cue sheet.
    pub id: String,
    /// Title of the production (film, show, etc.).
    pub production_title: String,
    /// Episode title (for series; `None` for standalone productions).
    pub episode_title: Option<String>,
    /// Episode number within a series (e.g. `"S01E03"`).
    pub episode_number: Option<String>,
    /// Type of production.
    pub production_type: ProductionType,
    /// Name of the producing company or distributor.
    pub producer: String,
    /// ISO 3166-1 alpha-2 country of production.
    pub country_of_production: String,
    /// Total duration of the production in seconds.
    pub production_duration_secs: f64,
    /// First broadcast / release date (ISO 8601, e.g. `"2024-03-15"`).
    pub air_date: Option<String>,
    /// Network or platform where the production aired / was released.
    pub network: Option<String>,
    /// Ordered list of music cues.
    pub cues: Vec<CueEntry>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl CueSheet {
    /// Create a new, empty cue sheet.
    pub fn new(
        id: impl Into<String>,
        production_title: impl Into<String>,
        production_type: ProductionType,
        producer: impl Into<String>,
        country_of_production: impl Into<String>,
        production_duration_secs: f64,
    ) -> Result<Self> {
        if production_duration_secs < 0.0 {
            return Err(RightsError::InvalidOperation(
                "production_duration_secs must be non-negative".into(),
            ));
        }
        Ok(Self {
            id: id.into(),
            production_title: production_title.into(),
            episode_title: None,
            episode_number: None,
            production_type,
            producer: producer.into(),
            country_of_production: country_of_production.into(),
            production_duration_secs,
            air_date: None,
            network: None,
            cues: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    /// Set the episode title.
    #[must_use]
    pub fn with_episode_title(mut self, title: impl Into<String>) -> Self {
        self.episode_title = Some(title.into());
        self
    }

    /// Set the episode number.
    #[must_use]
    pub fn with_episode_number(mut self, number: impl Into<String>) -> Self {
        self.episode_number = Some(number.into());
        self
    }

    /// Set the air date.
    #[must_use]
    pub fn with_air_date(mut self, date: impl Into<String>) -> Self {
        self.air_date = Some(date.into());
        self
    }

    /// Set the broadcast network / platform.
    #[must_use]
    pub fn with_network(mut self, network: impl Into<String>) -> Self {
        self.network = Some(network.into());
        self
    }

    /// Append a cue entry.  Cue numbers are reassigned automatically.
    pub fn add_cue(&mut self, mut cue: CueEntry) {
        cue.cue_number = self.cues.len() as u32 + 1;
        self.cues.push(cue);
    }

    /// Total music duration across all cues, in seconds.
    #[must_use]
    pub fn total_music_duration_secs(&self) -> f64 {
        self.cues.iter().map(|c| c.duration_secs).sum()
    }

    /// Music-to-picture ratio (0.0 – 1.0+).
    #[must_use]
    pub fn music_ratio(&self) -> f64 {
        if self.production_duration_secs == 0.0 {
            return 0.0;
        }
        self.total_music_duration_secs() / self.production_duration_secs
    }

    /// Return cues grouped by [`UsageType`] PRO code.
    #[must_use]
    pub fn cues_by_usage(&self) -> HashMap<String, Vec<&CueEntry>> {
        let mut map: HashMap<String, Vec<&CueEntry>> = HashMap::new();
        for cue in &self.cues {
            map.entry(cue.usage_type.pro_code().to_owned())
                .or_default()
                .push(cue);
        }
        map
    }

    /// Validate all cue entries in this cue sheet.
    pub fn validate(&self) -> Result<()> {
        if self.production_title.trim().is_empty() {
            return Err(RightsError::InvalidOperation(
                "production_title must not be empty".into(),
            ));
        }
        for cue in &self.cues {
            cue.validate()?;
        }
        Ok(())
    }

    /// Export the cue sheet to a tab-separated PRO submission string.
    ///
    /// The format mirrors ASCAP's standard cue-sheet text submission layout:
    /// - Line 1: header block
    /// - Subsequent lines: one row per cue
    #[must_use]
    pub fn to_pro_submission_text(&self) -> String {
        let mut out = String::new();

        // --- header ---
        out.push_str("MUSIC CUE SHEET\n");
        out.push_str(&format!("PRODUCTION:\t{}\n", self.production_title));
        if let Some(ep) = &self.episode_title {
            out.push_str(&format!("EPISODE:\t{ep}\n"));
        }
        if let Some(num) = &self.episode_number {
            out.push_str(&format!("EPISODE NO:\t{num}\n"));
        }
        out.push_str(&format!("TYPE:\t{}\n", self.production_type));
        out.push_str(&format!("PRODUCER:\t{}\n", self.producer));
        out.push_str(&format!("COUNTRY:\t{}\n", self.country_of_production));
        if let Some(date) = &self.air_date {
            out.push_str(&format!("AIR DATE:\t{date}\n"));
        }
        if let Some(net) = &self.network {
            out.push_str(&format!("NETWORK:\t{net}\n"));
        }
        let total_music = self.total_music_duration_secs();
        let tm_mins = (total_music as u64) / 60;
        let tm_secs = (total_music as u64) % 60;
        out.push_str(&format!("TOTAL MUSIC:\t{tm_mins:02}:{tm_secs:02}\n"));
        out.push('\n');

        // --- column headers ---
        out.push_str(
            "CUE#\tTITLE\tISRC\tISWC\tDURATION\tUSAGE\tTC IN\tTC OUT\tCONTRIBUTOR\tROLE\tPRO\tSHARE%\n",
        );

        // --- cue rows ---
        for cue in &self.cues {
            let base = format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t",
                cue.cue_number,
                cue.title,
                cue.isrc.as_deref().unwrap_or(""),
                cue.iswc.as_deref().unwrap_or(""),
                cue.formatted_duration(),
                cue.usage_type.pro_code(),
                cue.timecode_in.as_deref().unwrap_or(""),
                cue.timecode_out.as_deref().unwrap_or(""),
            );

            if cue.contributors.is_empty() {
                out.push_str(&base);
                out.push('\n');
            } else {
                for (i, contrib) in cue.contributors.iter().enumerate() {
                    if i == 0 {
                        out.push_str(&base);
                    } else {
                        // continuation line: repeat cue identifier cells blank
                        out.push_str(&format!(
                            "{}\t\t\t\t\t\t\t\t",
                            cue.cue_number
                        ));
                    }
                    out.push_str(&format!(
                        "{}\t{}\t{}\t{:.2}\n",
                        contrib.name,
                        contrib.role,
                        contrib.pro_affiliation.as_deref().unwrap_or(""),
                        contrib.ownership_share,
                    ));
                }
            }
        }

        out
    }
}

// ── TimelineEvent ─────────────────────────────────────────────────────────────

/// A timed music event on an edit timeline used to generate a [`CueSheet`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    /// Start position in the production (seconds from 0:00:00).
    pub start_secs: f64,
    /// Duration of this music cue (seconds).
    pub duration_secs: f64,
    /// Title of the musical work.
    pub title: String,
    /// How the cue is used.
    pub usage_type: UsageType,
    /// Optional ISRC for the recording.
    pub isrc: Option<String>,
    /// Optional ISWC for the composition.
    pub iswc: Option<String>,
    /// Contributors to include in the cue entry.
    pub contributors: Vec<Contributor>,
    /// Library reference, if any.
    pub library_reference: Option<String>,
}

impl TimelineEvent {
    /// Create a new timeline event.
    pub fn new(
        start_secs: f64,
        duration_secs: f64,
        title: impl Into<String>,
        usage_type: UsageType,
    ) -> Result<Self> {
        if start_secs < 0.0 {
            return Err(RightsError::InvalidOperation(
                "start_secs must be non-negative".into(),
            ));
        }
        if duration_secs <= 0.0 {
            return Err(RightsError::InvalidOperation(
                "duration_secs must be positive".into(),
            ));
        }
        Ok(Self {
            start_secs,
            duration_secs,
            title: title.into(),
            usage_type,
            isrc: None,
            iswc: None,
            contributors: Vec::new(),
            library_reference: None,
        })
    }

    /// Convert a seconds offset to an SMPTE-style `HH:MM:SS:00` timecode string
    /// (frame part is zeroed since we only have second-level precision).
    #[must_use]
    pub fn seconds_to_timecode(secs: f64) -> String {
        let total = secs as u64;
        let hours = total / 3600;
        let minutes = (total % 3600) / 60;
        let seconds = total % 60;
        format!("{hours:02}:{minutes:02}:{seconds:02}:00")
    }
}

// ── generate_cue_sheet ────────────────────────────────────────────────────────

/// Generate a [`CueSheet`] from an ordered list of [`TimelineEvent`]s.
///
/// The events do not need to be pre-sorted; this function sorts them by
/// `start_secs` before creating the cue entries.
pub fn generate_cue_sheet_from_timeline(
    cue_sheet_id: impl Into<String>,
    production_title: impl Into<String>,
    production_type: ProductionType,
    producer: impl Into<String>,
    country: impl Into<String>,
    production_duration_secs: f64,
    mut events: Vec<TimelineEvent>,
) -> Result<CueSheet> {
    // sort by start time
    events.sort_by(|a, b| {
        a.start_secs
            .partial_cmp(&b.start_secs)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sheet = CueSheet::new(
        cue_sheet_id,
        production_title,
        production_type,
        producer,
        country,
        production_duration_secs,
    )?;

    for event in events {
        let mut cue = CueEntry::new(
            0, // will be renumbered by add_cue
            event.title,
            event.usage_type,
            event.duration_secs,
        )?;
        cue.isrc = event.isrc;
        cue.iswc = event.iswc;
        cue.library_reference = event.library_reference;
        cue.timecode_in = Some(TimelineEvent::seconds_to_timecode(event.start_secs));
        cue.timecode_out = Some(TimelineEvent::seconds_to_timecode(
            event.start_secs + event.duration_secs,
        ));
        for contributor in event.contributors {
            cue.add_contributor(contributor);
        }
        sheet.add_cue(cue);
    }

    Ok(sheet)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_contributor() -> Contributor {
        Contributor::new("Jane Doe", "Composer", 50.0)
            .unwrap()
            .with_pro("ASCAP")
            .with_ipi("00000000123")
    }

    fn sample_cue(n: u32) -> CueEntry {
        let mut cue = CueEntry::new(n, "Test Theme", UsageType::Theme, 90.0).unwrap();
        cue.add_contributor(sample_contributor());
        cue
    }

    #[test]
    fn test_usage_type_pro_code() {
        assert_eq!(UsageType::BackgroundInstrumental.pro_code(), "BI");
        assert_eq!(UsageType::VisualVocal.pro_code(), "VV");
        assert_eq!(UsageType::Theme.pro_code(), "TH");
        assert_eq!(UsageType::Custom("X".into()).pro_code(), "XX");
    }

    #[test]
    fn test_contributor_invalid_share() {
        let result = Contributor::new("Alice", "Publisher", 110.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_contributor_valid() {
        let c = Contributor::new("Bob", "Lyricist", 25.0).unwrap();
        assert_eq!(c.name, "Bob");
        assert_eq!(c.ownership_share, 25.0);
    }

    #[test]
    fn test_cue_entry_formatted_duration() {
        let cue = CueEntry::new(1, "A", UsageType::Sting, 125.0).unwrap();
        assert_eq!(cue.formatted_duration(), "02:05");
    }

    #[test]
    fn test_cue_entry_total_share_for_role() {
        let mut cue = CueEntry::new(1, "B", UsageType::Theme, 60.0).unwrap();
        cue.add_contributor(Contributor::new("Alice", "Composer", 60.0).unwrap());
        cue.add_contributor(Contributor::new("Bob", "Composer", 40.0).unwrap());
        assert!((cue.total_share_for_role("Composer") - 100.0).abs() < f64::EPSILON);
        assert!((cue.total_share_for_role("Publisher") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cue_entry_validate_missing_contributor() {
        let cue = CueEntry::new(1, "Empty", UsageType::Sting, 5.0).unwrap();
        assert!(cue.validate().is_err());
    }

    #[test]
    fn test_cue_sheet_add_cue_renumbers() {
        let mut sheet = CueSheet::new(
            "cs1",
            "My Film",
            ProductionType::Film,
            "Studio A",
            "US",
            5400.0,
        )
        .unwrap();
        sheet.add_cue(sample_cue(99));
        sheet.add_cue(sample_cue(99));
        assert_eq!(sheet.cues[0].cue_number, 1);
        assert_eq!(sheet.cues[1].cue_number, 2);
    }

    #[test]
    fn test_cue_sheet_total_music_duration() {
        let mut sheet = CueSheet::new(
            "cs2",
            "Pilot",
            ProductionType::Television,
            "Broadcaster",
            "GB",
            1800.0,
        )
        .unwrap();
        sheet.add_cue(sample_cue(1)); // 90 s
        sheet.add_cue(sample_cue(2)); // 90 s
        assert!((sheet.total_music_duration_secs() - 180.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cue_sheet_music_ratio() {
        let mut sheet = CueSheet::new(
            "cs3",
            "Short",
            ProductionType::WebVideo,
            "Creator",
            "CA",
            300.0,
        )
        .unwrap();
        sheet.add_cue(sample_cue(1)); // 90 s out of 300 s
        let ratio = sheet.music_ratio();
        assert!((ratio - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_cue_sheet_pro_submission_text_contains_title() {
        let mut sheet = CueSheet::new(
            "cs4",
            "Big Movie",
            ProductionType::Film,
            "Big Studio",
            "US",
            7200.0,
        )
        .unwrap()
        .with_air_date("2024-06-01")
        .with_network("Netflix");
        sheet.add_cue(sample_cue(1));
        let text = sheet.to_pro_submission_text();
        assert!(text.contains("Big Movie"));
        assert!(text.contains("TH")); // theme usage code
        assert!(text.contains("Jane Doe"));
        assert!(text.contains("ASCAP"));
    }

    #[test]
    fn test_generate_cue_sheet_from_timeline_sorted() {
        let mut events = Vec::new();
        let mut e1 = TimelineEvent::new(120.0, 45.0, "Late Cue", UsageType::BackgroundInstrumental)
            .unwrap();
        e1.contributors
            .push(Contributor::new("Composer A", "Composer", 100.0).unwrap());
        let mut e2 =
            TimelineEvent::new(10.0, 30.0, "Early Cue", UsageType::Theme).unwrap();
        e2.contributors
            .push(Contributor::new("Composer B", "Composer", 100.0).unwrap());
        events.push(e1);
        events.push(e2);

        let sheet = generate_cue_sheet_from_timeline(
            "tl1",
            "Test Production",
            ProductionType::OnlineContent,
            "Producer X",
            "DE",
            600.0,
            events,
        )
        .unwrap();

        // events must be sorted by start time
        assert_eq!(sheet.cues[0].title, "Early Cue");
        assert_eq!(sheet.cues[1].title, "Late Cue");
        assert_eq!(sheet.cues[0].timecode_in.as_deref(), Some("00:00:10:00"));
    }

    #[test]
    fn test_timecode_conversion() {
        assert_eq!(
            TimelineEvent::seconds_to_timecode(3661.0),
            "01:01:01:00"
        );
        assert_eq!(TimelineEvent::seconds_to_timecode(0.0), "00:00:00:00");
    }
}
