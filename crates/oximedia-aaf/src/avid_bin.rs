//! Avid bin structure reading and writing for Avid Media Composer compatibility.
//!
//! Avid bins are collections of clip references used by Avid Media Composer.
//! A bin file (`.avb`) is a structured collection of mobs, attributes, and
//! metadata that organises media in a project.
//!
//! This module provides:
//! - [`AvidBin`] — in-memory representation of a bin
//! - [`BinItem`] — an item (clip reference) within a bin
//! - [`AvidBinReader`] — parse bin metadata from an [`AafFile`]
//! - [`AvidBinWriter`] — serialise a bin into AAF-compatible XML or EDL

use crate::{AafError, AafFile, ContentStorage, Result};
use std::collections::HashMap;
use uuid::Uuid;

// ─── BinItem ─────────────────────────────────────────────────────────────────

/// Display/viewing attributes of a bin item (column values).
#[derive(Debug, Clone)]
pub struct BinItemAttributes {
    /// User-assigned label (column value)
    pub label: Option<String>,
    /// Tape name (for source mobs that reference tape)
    pub tape_name: Option<String>,
    /// Duration in edit units
    pub duration: Option<i64>,
    /// Start timecode string
    pub start_tc: Option<String>,
    /// Whether this item is locked for editing
    pub locked: bool,
    /// Custom user metadata key-value pairs
    pub user_columns: HashMap<String, String>,
}

impl BinItemAttributes {
    /// Create new empty attributes.
    #[must_use]
    pub fn new() -> Self {
        Self {
            label: None,
            tape_name: None,
            duration: None,
            start_tc: None,
            locked: false,
            user_columns: HashMap::new(),
        }
    }

    /// Set the label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the tape name.
    pub fn with_tape_name(mut self, name: impl Into<String>) -> Self {
        self.tape_name = Some(name.into());
        self
    }

    /// Set duration.
    pub fn with_duration(mut self, dur: i64) -> Self {
        self.duration = Some(dur);
        self
    }

    /// Set start timecode string.
    pub fn with_start_tc(mut self, tc: impl Into<String>) -> Self {
        self.start_tc = Some(tc.into());
        self
    }

    /// Set a user column value.
    pub fn set_column(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.user_columns.insert(key.into(), value.into());
    }
}

impl Default for BinItemAttributes {
    fn default() -> Self {
        Self::new()
    }
}

/// A single item in an Avid bin, representing a clip or sequence reference.
#[derive(Debug, Clone)]
pub struct BinItem {
    /// Mob ID this item references
    pub mob_id: Uuid,
    /// Display name shown in the bin
    pub name: String,
    /// Item type classification
    pub item_type: BinItemType,
    /// View/column attributes
    pub attributes: BinItemAttributes,
}

impl BinItem {
    /// Create a new bin item.
    #[must_use]
    pub fn new(mob_id: Uuid, name: impl Into<String>, item_type: BinItemType) -> Self {
        Self {
            mob_id,
            name: name.into(),
            item_type,
            attributes: BinItemAttributes::new(),
        }
    }

    /// Set the attributes for this item.
    pub fn with_attributes(mut self, attrs: BinItemAttributes) -> Self {
        self.attributes = attrs;
        self
    }
}

/// The type of a bin item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinItemType {
    /// A master clip
    MasterClip,
    /// A sequence (composition mob)
    Sequence,
    /// A subclip
    SubClip,
    /// An effect
    Effect,
    /// A source mob reference (tape / file)
    SourceMob,
    /// Unknown item type
    Unknown,
}

impl std::fmt::Display for BinItemType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MasterClip => write!(f, "MasterClip"),
            Self::Sequence => write!(f, "Sequence"),
            Self::SubClip => write!(f, "SubClip"),
            Self::Effect => write!(f, "Effect"),
            Self::SourceMob => write!(f, "SourceMob"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ─── AvidBin ─────────────────────────────────────────────────────────────────

/// Sort order for bin items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinSortOrder {
    /// Ascending alphabetical
    NameAscending,
    /// Descending alphabetical
    NameDescending,
    /// Creation / insertion order (no sort)
    None,
}

/// Column definition for a bin view.
#[derive(Debug, Clone)]
pub struct BinColumn {
    /// Column identifier
    pub id: String,
    /// Display header text
    pub header: String,
    /// Column width (in abstract units)
    pub width: u32,
    /// Whether the column is visible
    pub visible: bool,
}

impl BinColumn {
    /// Create a new column.
    #[must_use]
    pub fn new(id: impl Into<String>, header: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            header: header.into(),
            width: 120,
            visible: true,
        }
    }
}

/// An Avid bin — a named collection of clips/sequences in a Media Composer project.
#[derive(Debug, Clone)]
pub struct AvidBin {
    /// Bin name (shown in the Avid project window)
    pub name: String,
    /// Unique identifier for this bin
    pub bin_id: Uuid,
    /// Ordered list of items in the bin
    items: Vec<BinItem>,
    /// Column definitions for the bin view
    columns: Vec<BinColumn>,
    /// Sort order
    pub sort_order: BinSortOrder,
}

impl AvidBin {
    /// Create a new empty bin.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bin_id: Uuid::new_v4(),
            items: Vec::new(),
            columns: Self::default_columns(),
            sort_order: BinSortOrder::None,
        }
    }

    /// Default column layout (Name, Duration, Start TC).
    fn default_columns() -> Vec<BinColumn> {
        vec![
            BinColumn::new("Name", "Name"),
            BinColumn::new("Duration", "Duration"),
            BinColumn::new("StartTC", "Start TC"),
            BinColumn::new("TapeName", "Tape Name"),
        ]
    }

    /// Add an item to the bin.
    pub fn add_item(&mut self, item: BinItem) {
        self.items.push(item);
    }

    /// Remove an item by mob_id. Returns `true` if an item was removed.
    pub fn remove_item(&mut self, mob_id: &Uuid) -> bool {
        let before = self.items.len();
        self.items.retain(|i| &i.mob_id != mob_id);
        self.items.len() < before
    }

    /// Look up an item by mob_id.
    #[must_use]
    pub fn find_item(&self, mob_id: &Uuid) -> Option<&BinItem> {
        self.items.iter().find(|i| &i.mob_id == mob_id)
    }

    /// Find items by type.
    #[must_use]
    pub fn items_of_type(&self, item_type: BinItemType) -> Vec<&BinItem> {
        self.items
            .iter()
            .filter(|i| i.item_type == item_type)
            .collect()
    }

    /// All items in bin order.
    #[must_use]
    pub fn items(&self) -> &[BinItem] {
        &self.items
    }

    /// Number of items.
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Add a column definition.
    pub fn add_column(&mut self, col: BinColumn) {
        self.columns.push(col);
    }

    /// All column definitions.
    #[must_use]
    pub fn columns(&self) -> &[BinColumn] {
        &self.columns
    }

    /// Sort items alphabetically by name.
    pub fn sort_by_name(&mut self) {
        self.items.sort_by(|a, b| a.name.cmp(&b.name));
        self.sort_order = BinSortOrder::NameAscending;
    }

    /// Sort items by name descending.
    pub fn sort_by_name_desc(&mut self) {
        self.items.sort_by(|a, b| b.name.cmp(&a.name));
        self.sort_order = BinSortOrder::NameDescending;
    }
}

impl Default for AvidBin {
    fn default() -> Self {
        Self::new("Untitled Bin")
    }
}

// ─── AvidBinReader ───────────────────────────────────────────────────────────

/// Reads Avid bin structure from an [`AafFile`].
///
/// Avid Media Composer stores bin membership information in the AAF's
/// ContentStorage and in a sidecar `.avb` file.  This reader reconstructs a
/// logical bin representation from the composition mobs present in an AAF.
pub struct AvidBinReader;

impl AvidBinReader {
    /// Create a new reader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Build a single [`AvidBin`] from all mobs in `file`.
    ///
    /// In a real Avid project each `.avb` sidecar would correspond to one bin.
    /// When only an AAF is available, all composition mobs are placed in a
    /// single synthesised bin.
    ///
    /// # Errors
    ///
    /// Returns an error if the file contains no mobs (empty project).
    pub fn read_from_file(&self, file: &AafFile) -> Result<AvidBin> {
        self.read_from_content_storage(file.content_storage(), file.name().unwrap_or("Bin"))
    }

    /// Build an [`AvidBin`] from a [`ContentStorage`] object.
    pub fn read_from_content_storage(
        &self,
        storage: &ContentStorage,
        bin_name: &str,
    ) -> Result<AvidBin> {
        let mut bin = AvidBin::new(bin_name);

        // Add composition mobs as Sequence items
        for comp_mob in storage.composition_mobs() {
            let mut attrs = BinItemAttributes::new().with_label(comp_mob.name().to_string());
            if let Some(dur) = comp_mob.duration() {
                attrs = attrs.with_duration(dur);
            }
            let item = BinItem::new(comp_mob.mob_id(), comp_mob.name(), BinItemType::Sequence)
                .with_attributes(attrs);
            bin.add_item(item);
        }

        // Add source mobs as MasterClip items
        for mob in storage.source_mobs() {
            let item = BinItem::new(mob.mob_id(), mob.name(), BinItemType::SourceMob);
            bin.add_item(item);
        }

        Ok(bin)
    }
}

impl Default for AvidBinReader {
    fn default() -> Self {
        Self::new()
    }
}

// ─── AvidBinWriter ───────────────────────────────────────────────────────────

/// Serialises an [`AvidBin`] into a portable text representation.
///
/// The output format is a simple tab-separated value file that Avid Media
/// Composer can import via its "Import Bin" function.
pub struct AvidBinWriter;

impl AvidBinWriter {
    /// Create a new writer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Write `bin` as a tab-separated text file (Avid import format).
    ///
    /// # Errors
    ///
    /// Returns an error if the bin has no items.
    pub fn write_tsv(&self, bin: &AvidBin) -> Result<String> {
        use std::fmt::Write;
        let mut out = String::new();

        // Header line: column names
        let col_names: Vec<&str> = bin.columns().iter().map(|c| c.header.as_str()).collect();
        writeln!(&mut out, "{}", col_names.join("\t"))
            .map_err(|e| AafError::WriteError(e.to_string()))?;

        for item in bin.items() {
            let name = item.name.replace('\t', " ");
            let duration = item
                .attributes
                .duration
                .map(|d| d.to_string())
                .unwrap_or_default();
            let tc = item.attributes.start_tc.clone().unwrap_or_default();
            let tape = item.attributes.tape_name.clone().unwrap_or_default();
            writeln!(&mut out, "{name}\t{duration}\t{tc}\t{tape}")
                .map_err(|e| AafError::WriteError(e.to_string()))?;
        }

        Ok(out)
    }

    /// Write `bin` as an AAF-compatible XML fragment.
    ///
    /// # Errors
    ///
    /// Returns an error only on string formatting failure (practically
    /// infallible).
    pub fn write_xml(&self, bin: &AvidBin) -> Result<String> {
        use std::fmt::Write;
        let mut xml = String::new();

        writeln!(&mut xml, r#"<?xml version="1.0" encoding="UTF-8"?>"#)
            .map_err(|e| AafError::WriteError(e.to_string()))?;
        writeln!(
            &mut xml,
            r#"<AvidBin name="{}" id="{}">"#,
            xml_escape(&bin.name),
            bin.bin_id
        )
        .map_err(|e| AafError::WriteError(e.to_string()))?;

        for item in bin.items() {
            writeln!(
                &mut xml,
                r#"  <BinItem type="{}" mobId="{}" name="{}" />"#,
                item.item_type,
                item.mob_id,
                xml_escape(&item.name)
            )
            .map_err(|e| AafError::WriteError(e.to_string()))?;
        }

        writeln!(&mut xml, "</AvidBin>").map_err(|e| AafError::WriteError(e.to_string()))?;
        Ok(xml)
    }
}

impl Default for AvidBinWriter {
    fn default() -> Self {
        Self::new()
    }
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ─── AafFile helpers for bin support ─────────────────────────────────────────

impl AafFile {
    /// Return the file name derived from the first composition mob name, or
    /// `None` if there are no composition mobs.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.composition_mobs()
            .first()
            .map(|m| m.name())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::CompositionMob;
    use crate::object_model::{Mob, MobType};

    fn make_bin() -> AvidBin {
        let mut bin = AvidBin::new("TestBin");
        bin.add_item(BinItem::new(Uuid::new_v4(), "Clip A", BinItemType::MasterClip));
        bin.add_item(BinItem::new(Uuid::new_v4(), "Sequence B", BinItemType::Sequence));
        bin
    }

    #[test]
    fn test_bin_add_and_count() {
        let bin = make_bin();
        assert_eq!(bin.item_count(), 2);
    }

    #[test]
    fn test_bin_find_item() {
        let mut bin = AvidBin::new("Bin");
        let id = Uuid::new_v4();
        bin.add_item(BinItem::new(id, "My Clip", BinItemType::MasterClip));
        assert!(bin.find_item(&id).is_some());
        assert!(bin.find_item(&Uuid::new_v4()).is_none());
    }

    #[test]
    fn test_bin_remove_item() {
        let mut bin = AvidBin::new("Bin");
        let id = Uuid::new_v4();
        bin.add_item(BinItem::new(id, "Clip", BinItemType::MasterClip));
        assert_eq!(bin.item_count(), 1);
        assert!(bin.remove_item(&id));
        assert_eq!(bin.item_count(), 0);
        // Removing again returns false
        assert!(!bin.remove_item(&id));
    }

    #[test]
    fn test_bin_items_of_type() {
        let bin = make_bin();
        let clips = bin.items_of_type(BinItemType::MasterClip);
        assert_eq!(clips.len(), 1);
        let seqs = bin.items_of_type(BinItemType::Sequence);
        assert_eq!(seqs.len(), 1);
    }

    #[test]
    fn test_bin_sort_by_name() {
        let mut bin = AvidBin::new("Bin");
        bin.add_item(BinItem::new(Uuid::new_v4(), "Zebra", BinItemType::MasterClip));
        bin.add_item(BinItem::new(Uuid::new_v4(), "Apple", BinItemType::MasterClip));
        bin.sort_by_name();
        assert_eq!(bin.items()[0].name, "Apple");
        assert_eq!(bin.items()[1].name, "Zebra");
        assert_eq!(bin.sort_order, BinSortOrder::NameAscending);
    }

    #[test]
    fn test_bin_sort_by_name_desc() {
        let mut bin = AvidBin::new("Bin");
        bin.add_item(BinItem::new(Uuid::new_v4(), "Apple", BinItemType::MasterClip));
        bin.add_item(BinItem::new(Uuid::new_v4(), "Zebra", BinItemType::MasterClip));
        bin.sort_by_name_desc();
        assert_eq!(bin.items()[0].name, "Zebra");
        assert_eq!(bin.sort_order, BinSortOrder::NameDescending);
    }

    #[test]
    fn test_bin_item_type_display() {
        assert_eq!(BinItemType::MasterClip.to_string(), "MasterClip");
        assert_eq!(BinItemType::Sequence.to_string(), "Sequence");
        assert_eq!(BinItemType::Unknown.to_string(), "Unknown");
    }

    #[test]
    fn test_bin_attributes_builder() {
        let attrs = BinItemAttributes::new()
            .with_label("Test Label")
            .with_tape_name("TAPE001")
            .with_duration(250)
            .with_start_tc("01:00:00:00");
        assert_eq!(attrs.label, Some("Test Label".to_string()));
        assert_eq!(attrs.tape_name, Some("TAPE001".to_string()));
        assert_eq!(attrs.duration, Some(250));
        assert_eq!(attrs.start_tc, Some("01:00:00:00".to_string()));
    }

    #[test]
    fn test_bin_attributes_user_columns() {
        let mut attrs = BinItemAttributes::new();
        attrs.set_column("Scene", "3A");
        attrs.set_column("Reel", "A001");
        assert_eq!(attrs.user_columns.get("Scene"), Some(&"3A".to_string()));
        assert_eq!(attrs.user_columns.len(), 2);
    }

    #[test]
    fn test_writer_tsv_output() {
        let bin = make_bin();
        let writer = AvidBinWriter::new();
        let tsv = writer.write_tsv(&bin).expect("write_tsv should succeed");
        // Header should be present
        assert!(tsv.contains("Name"));
        // Two data rows
        let lines: Vec<&str> = tsv.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 items
    }

    #[test]
    fn test_writer_xml_output() {
        let bin = make_bin();
        let writer = AvidBinWriter::new();
        let xml = writer.write_xml(&bin).expect("write_xml should succeed");
        assert!(xml.contains("<AvidBin"));
        assert!(xml.contains("</AvidBin>"));
        assert!(xml.contains("<BinItem"));
        assert!(xml.contains("MasterClip"));
        assert!(xml.contains("Sequence"));
    }

    #[test]
    fn test_writer_xml_escapes_special_chars() {
        let mut bin = AvidBin::new("Test & Bin");
        bin.add_item(BinItem::new(Uuid::new_v4(), "Clip <1>", BinItemType::MasterClip));
        let writer = AvidBinWriter::new();
        let xml = writer.write_xml(&bin).expect("write_xml should succeed");
        assert!(xml.contains("Test &amp; Bin"));
        assert!(xml.contains("Clip &lt;1&gt;"));
    }

    #[test]
    fn test_reader_from_file() {
        use crate::ContentStorage;
        use crate::composition::CompositionMob;
        let mut storage = ContentStorage::new();
        storage.add_composition_mob(CompositionMob::new(Uuid::new_v4(), "TestComp"));
        let reader = AvidBinReader::new();
        let bin = reader
            .read_from_content_storage(&storage, "MyBin")
            .expect("read should succeed");
        assert_eq!(bin.name, "MyBin");
        assert_eq!(bin.item_count(), 1);
        assert_eq!(bin.items()[0].name, "TestComp");
        assert_eq!(bin.items()[0].item_type, BinItemType::Sequence);
    }

    #[test]
    fn test_reader_includes_source_mobs() {
        use crate::ContentStorage;
        use crate::object_model::{Mob, MobType};
        let mut storage = ContentStorage::new();
        let mob = Mob::new(Uuid::new_v4(), "source.mxf".to_string(), MobType::Source);
        storage.add_mob(mob);
        let reader = AvidBinReader::new();
        let bin = reader
            .read_from_content_storage(&storage, "SourceBin")
            .expect("read should succeed");
        assert_eq!(bin.item_count(), 1);
        assert_eq!(bin.items()[0].item_type, BinItemType::SourceMob);
    }

    #[test]
    fn test_bin_default_columns() {
        let bin = AvidBin::new("Bin");
        assert!(!bin.columns().is_empty());
        assert!(bin.columns().iter().any(|c| c.header == "Name"));
    }

    #[test]
    fn test_bin_add_custom_column() {
        let mut bin = AvidBin::new("Bin");
        bin.add_column(BinColumn::new("CustomCol", "My Column"));
        assert!(bin.columns().iter().any(|c| c.id == "CustomCol"));
    }
}
