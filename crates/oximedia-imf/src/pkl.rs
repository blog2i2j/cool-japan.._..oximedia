//! Packing List (PKL) - SMPTE ST 429-8
//!
//! The PKL provides a complete inventory of all assets in an IMF package,
//! including their checksums for integrity verification.

use crate::{ImfError, ImfResult};
use chrono::{DateTime, Utc};
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::Path;
use uuid::Uuid;

/// Hash algorithm used for asset verification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HashAlgorithm {
    /// SHA-1 (160-bit)
    Sha1,
    /// MD5 (128-bit)
    Md5,
}

impl HashAlgorithm {
    /// Get the algorithm name as string
    pub fn as_str(&self) -> &str {
        match self {
            Self::Sha1 => "SHA-1",
            Self::Md5 => "MD5",
        }
    }

    /// Parse algorithm from string
    pub fn from_str(s: &str) -> ImfResult<Self> {
        match s {
            "SHA-1" | "sha1" | "SHA1" => Ok(Self::Sha1),
            "MD5" | "md5" => Ok(Self::Md5),
            _ => Err(ImfError::InvalidStructure(format!(
                "Unknown hash algorithm: {s}"
            ))),
        }
    }

    /// Calculate hash of a file
    pub fn hash_file(&self, path: &Path) -> ImfResult<String> {
        use md5::Digest as Md5Digest;
        use sha1::Sha1;
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)?;
        let mut buffer = vec![0; 8192];

        match self {
            Self::Sha1 => {
                let mut hasher = Sha1::new();
                loop {
                    let count = file.read(&mut buffer)?;
                    if count == 0 {
                        break;
                    }
                    hasher.update(&buffer[..count]);
                }
                let hash = hasher.finalize();
                Ok(hex::encode(hash))
            }
            Self::Md5 => {
                let mut hasher = md5::Md5::new();
                loop {
                    let count = file.read(&mut buffer)?;
                    if count == 0 {
                        break;
                    }
                    hasher.update(&buffer[..count]);
                }
                let hash = hasher.finalize();
                Ok(hex::encode(hash))
            }
        }
    }

    /// Verify hash of a file
    pub fn verify_file(&self, path: &Path, expected_hash: &str) -> ImfResult<bool> {
        let actual_hash = self.hash_file(path)?;
        Ok(actual_hash.eq_ignore_ascii_case(expected_hash))
    }
}

/// Asset in a packing list
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    id: Uuid,
    annotation: Option<String>,
    hash: String,
    hash_algorithm: HashAlgorithm,
    size: u64,
    asset_type: String,
    original_filename: Option<String>,
}

impl Asset {
    /// Create a new asset
    pub fn new(
        id: Uuid,
        hash: String,
        hash_algorithm: HashAlgorithm,
        size: u64,
        asset_type: String,
    ) -> Self {
        Self {
            id,
            annotation: None,
            hash,
            hash_algorithm,
            size,
            asset_type,
            original_filename: None,
        }
    }

    /// Create asset from file
    pub fn from_file(
        id: Uuid,
        path: &Path,
        hash_algorithm: HashAlgorithm,
        asset_type: String,
    ) -> ImfResult<Self> {
        let metadata = std::fs::metadata(path)?;
        let size = metadata.len();
        let hash = hash_algorithm.hash_file(path)?;

        let mut asset = Self::new(id, hash, hash_algorithm, size, asset_type);

        if let Some(filename) = path.file_name() {
            asset.original_filename = Some(filename.to_string_lossy().to_string());
        }

        Ok(asset)
    }

    /// Get the asset ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get the annotation
    pub fn annotation(&self) -> Option<&str> {
        self.annotation.as_deref()
    }

    /// Set annotation
    pub fn set_annotation(&mut self, annotation: String) {
        self.annotation = Some(annotation);
    }

    /// Get the hash
    pub fn hash(&self) -> &str {
        &self.hash
    }

    /// Get the hash algorithm
    pub fn hash_algorithm(&self) -> HashAlgorithm {
        self.hash_algorithm
    }

    /// Get the size in bytes
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get the asset type
    pub fn asset_type(&self) -> &str {
        &self.asset_type
    }

    /// Get the original filename
    pub fn original_filename(&self) -> Option<&str> {
        self.original_filename.as_deref()
    }

    /// Verify asset hash against a file
    pub fn verify(&self, path: &Path) -> ImfResult<bool> {
        // Check size first (faster)
        let metadata = std::fs::metadata(path)?;
        if metadata.len() != self.size {
            return Ok(false);
        }

        // Verify hash
        self.hash_algorithm.verify_file(path, &self.hash)
    }

    /// Set original filename
    pub fn with_original_filename(mut self, filename: String) -> Self {
        self.original_filename = Some(filename);
        self
    }
}

/// Packing List (PKL) - SMPTE ST 429-8
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackingList {
    id: Uuid,
    annotation: Option<String>,
    issue_date: DateTime<Utc>,
    issuer: Option<String>,
    creator: Option<String>,
    icon_id: Option<Uuid>,
    group_id: Option<Uuid>,
    assets: Vec<Asset>,
}

impl PackingList {
    /// Create a new packing list
    pub fn new(id: Uuid) -> Self {
        Self {
            id,
            annotation: None,
            issue_date: Utc::now(),
            issuer: None,
            creator: None,
            icon_id: None,
            group_id: None,
            assets: Vec::new(),
        }
    }

    /// Get the PKL ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get the annotation
    pub fn annotation(&self) -> Option<&str> {
        self.annotation.as_deref()
    }

    /// Set annotation
    pub fn set_annotation(&mut self, annotation: String) {
        self.annotation = Some(annotation);
    }

    /// Get the issue date
    pub fn issue_date(&self) -> DateTime<Utc> {
        self.issue_date
    }

    /// Set issue date
    pub fn set_issue_date(&mut self, date: DateTime<Utc>) {
        self.issue_date = date;
    }

    /// Get the issuer
    pub fn issuer(&self) -> Option<&str> {
        self.issuer.as_deref()
    }

    /// Set issuer
    pub fn set_issuer(&mut self, issuer: String) {
        self.issuer = Some(issuer);
    }

    /// Get the creator
    pub fn creator(&self) -> Option<&str> {
        self.creator.as_deref()
    }

    /// Set creator
    pub fn set_creator(&mut self, creator: String) {
        self.creator = Some(creator);
    }

    /// Get the icon ID
    pub fn icon_id(&self) -> Option<Uuid> {
        self.icon_id
    }

    /// Set icon ID
    pub fn set_icon_id(&mut self, id: Uuid) {
        self.icon_id = Some(id);
    }

    /// Get the group ID
    pub fn group_id(&self) -> Option<Uuid> {
        self.group_id
    }

    /// Set group ID
    pub fn set_group_id(&mut self, id: Uuid) {
        self.group_id = Some(id);
    }

    /// Get the assets
    pub fn assets(&self) -> &[Asset] {
        &self.assets
    }

    /// Add an asset
    pub fn add_asset(&mut self, asset: Asset) {
        self.assets.push(asset);
    }

    /// Find an asset by ID
    pub fn find_asset(&self, id: Uuid) -> Option<&Asset> {
        self.assets.iter().find(|a| a.id == id)
    }

    /// Remove an asset by ID
    pub fn remove_asset(&mut self, id: Uuid) -> Option<Asset> {
        if let Some(pos) = self.assets.iter().position(|a| a.id == id) {
            Some(self.assets.remove(pos))
        } else {
            None
        }
    }

    /// Get total size of all assets
    pub fn total_size(&self) -> u64 {
        self.assets.iter().map(|a| a.size).sum()
    }

    /// Parse PKL from XML
    pub fn from_xml<R: BufRead>(reader: R) -> ImfResult<Self> {
        PklParser::parse(reader)
    }

    /// Write PKL to XML
    pub fn to_xml<W: Write>(&self, writer: W) -> ImfResult<()> {
        PklWriter::write(self, writer)
    }
}

/// PKL XML parser
struct PklParser;

impl PklParser {
    #[allow(clippy::too_many_lines)]
    fn parse<R: BufRead>(reader: R) -> ImfResult<PackingList> {
        let mut xml_reader = Reader::from_reader(reader);
        xml_reader.config_mut().trim_text(true);

        let mut buf = Vec::new();
        #[allow(unused_assignments)]
        let mut current_element = String::new();
        let mut text_buffer = String::new();

        // PKL fields
        let mut id: Option<Uuid> = None;
        let mut annotation: Option<String> = None;
        let mut issue_date: Option<DateTime<Utc>> = None;
        let mut issuer: Option<String> = None;
        let mut creator: Option<String> = None;
        let mut icon_id: Option<Uuid> = None;
        let mut group_id: Option<Uuid> = None;
        let mut assets: Vec<Asset> = Vec::new();

        // State for parsing assets
        let mut in_asset = false;
        let mut current_asset_id: Option<Uuid> = None;
        let mut current_asset_annotation: Option<String> = None;
        let mut current_asset_hash: Option<String> = None;
        let mut current_asset_hash_algo: Option<HashAlgorithm> = None;
        let mut current_asset_size: Option<u64> = None;
        let mut current_asset_type: Option<String> = None;
        let mut current_asset_filename: Option<String> = None;

        loop {
            match xml_reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    current_element = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    text_buffer.clear();

                    if current_element == "Asset" {
                        in_asset = true;
                        current_asset_id = None;
                        current_asset_annotation = None;
                        current_asset_hash = None;
                        current_asset_hash_algo = None;
                        current_asset_size = None;
                        current_asset_type = None;
                        current_asset_filename = None;
                    } else if current_element == "Hash" {
                        // Get algorithm attribute
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"algorithm" {
                                let algo_str = std::str::from_utf8(&attr.value)
                                    .map_err(|e| ImfError::XmlError(format!("UTF-8 error: {e}")))?;
                                current_asset_hash_algo = Some(HashAlgorithm::from_str(algo_str)?);
                            }
                        }
                    }
                }
                Ok(Event::Text(e)) => {
                    text_buffer = String::from_utf8_lossy(e.as_ref()).to_string();
                }
                Ok(Event::End(e)) => {
                    let element_name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    if in_asset {
                        match element_name.as_str() {
                            "Id" => {
                                current_asset_id = Some(Self::parse_uuid(&text_buffer)?);
                            }
                            "Annotation" => {
                                current_asset_annotation = Some(text_buffer.clone());
                            }
                            "Hash" => {
                                current_asset_hash = Some(text_buffer.clone());
                            }
                            "Size" => {
                                current_asset_size = Some(text_buffer.parse().map_err(|_| {
                                    ImfError::InvalidStructure("Invalid size".to_string())
                                })?);
                            }
                            "Type" => {
                                current_asset_type = Some(text_buffer.clone());
                            }
                            "OriginalFileName" => {
                                current_asset_filename = Some(text_buffer.clone());
                            }
                            "Asset" => {
                                // Build asset
                                let asset_id = current_asset_id.take().ok_or_else(|| {
                                    ImfError::MissingElement("Asset Id".to_string())
                                })?;
                                let hash = current_asset_hash.take().ok_or_else(|| {
                                    ImfError::MissingElement("Asset Hash".to_string())
                                })?;
                                let hash_algo =
                                    current_asset_hash_algo.take().ok_or_else(|| {
                                        ImfError::MissingElement("Hash algorithm".to_string())
                                    })?;
                                let size = current_asset_size.take().ok_or_else(|| {
                                    ImfError::MissingElement("Asset Size".to_string())
                                })?;
                                let asset_type = current_asset_type.take().ok_or_else(|| {
                                    ImfError::MissingElement("Asset Type".to_string())
                                })?;

                                let mut asset =
                                    Asset::new(asset_id, hash, hash_algo, size, asset_type);
                                asset.annotation = current_asset_annotation.clone();
                                asset.original_filename = current_asset_filename.clone();

                                assets.push(asset);
                                in_asset = false;
                            }
                            _ => {}
                        }
                    } else {
                        // Top-level elements
                        match element_name.as_str() {
                            "Id" => id = Some(Self::parse_uuid(&text_buffer)?),
                            "Annotation" => annotation = Some(text_buffer.clone()),
                            "IssueDate" => {
                                issue_date = Some(
                                    DateTime::parse_from_rfc3339(&text_buffer)
                                        .map_err(|e| {
                                            ImfError::InvalidStructure(format!(
                                                "Invalid IssueDate: {e}"
                                            ))
                                        })?
                                        .with_timezone(&Utc),
                                );
                            }
                            "Issuer" => issuer = Some(text_buffer.clone()),
                            "Creator" => creator = Some(text_buffer.clone()),
                            "IconId" => icon_id = Some(Self::parse_uuid(&text_buffer)?),
                            "GroupId" => group_id = Some(Self::parse_uuid(&text_buffer)?),
                            _ => {}
                        }
                    }

                    text_buffer.clear();
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(ImfError::XmlError(format!("XML parse error: {e}"))),
                _ => {}
            }
            buf.clear();
        }

        // Build PKL
        let id = id.ok_or_else(|| ImfError::MissingElement("Id".to_string()))?;

        let mut pkl = PackingList::new(id);
        pkl.annotation = annotation;
        pkl.issue_date = issue_date.unwrap_or_else(Utc::now);
        pkl.issuer = issuer;
        pkl.creator = creator;
        pkl.icon_id = icon_id;
        pkl.group_id = group_id;
        pkl.assets = assets;

        Ok(pkl)
    }

    fn parse_uuid(s: &str) -> ImfResult<Uuid> {
        // Handle URN format: urn:uuid:xxxxx
        let uuid_str = s.trim().strip_prefix("urn:uuid:").unwrap_or(s);
        Uuid::parse_str(uuid_str).map_err(|e| ImfError::InvalidUuid(e.to_string()))
    }
}

/// PKL XML writer
struct PklWriter;

impl PklWriter {
    fn write<W: Write>(pkl: &PackingList, writer: W) -> ImfResult<()> {
        let mut xml_writer = Writer::new_with_indent(writer, b' ', 2);

        // XML declaration
        xml_writer
            .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        // Root element
        let mut root = BytesStart::new("PackingList");
        root.push_attribute(("xmlns", "http://www.smpte-ra.org/schemas/429-8/2007/PKL"));
        xml_writer
            .write_event(Event::Start(root))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        // Write fields
        Self::write_element(&mut xml_writer, "Id", &format!("urn:uuid:{}", pkl.id))?;

        if let Some(ref annotation) = pkl.annotation {
            Self::write_element(&mut xml_writer, "Annotation", annotation)?;
        }

        Self::write_element(&mut xml_writer, "IssueDate", &pkl.issue_date.to_rfc3339())?;

        if let Some(ref issuer) = pkl.issuer {
            Self::write_element(&mut xml_writer, "Issuer", issuer)?;
        }

        if let Some(ref creator) = pkl.creator {
            Self::write_element(&mut xml_writer, "Creator", creator)?;
        }

        if let Some(icon_id) = pkl.icon_id {
            Self::write_element(&mut xml_writer, "IconId", &format!("urn:uuid:{icon_id}"))?;
        }

        if let Some(group_id) = pkl.group_id {
            Self::write_element(&mut xml_writer, "GroupId", &format!("urn:uuid:{group_id}"))?;
        }

        // Assets
        Self::write_assets(&mut xml_writer, &pkl.assets)?;

        // Close root
        xml_writer
            .write_event(Event::End(BytesEnd::new("PackingList")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Ok(())
    }

    fn write_element<W: Write>(writer: &mut Writer<W>, name: &str, content: &str) -> ImfResult<()> {
        writer
            .write_event(Event::Start(BytesStart::new(name)))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;
        writer
            .write_event(Event::Text(BytesText::new(content)))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;
        writer
            .write_event(Event::End(BytesEnd::new(name)))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;
        Ok(())
    }

    fn write_assets<W: Write>(writer: &mut Writer<W>, assets: &[Asset]) -> ImfResult<()> {
        writer
            .write_event(Event::Start(BytesStart::new("AssetList")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        for asset in assets {
            Self::write_asset(writer, asset)?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("AssetList")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Ok(())
    }

    fn write_asset<W: Write>(writer: &mut Writer<W>, asset: &Asset) -> ImfResult<()> {
        writer
            .write_event(Event::Start(BytesStart::new("Asset")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Self::write_element(writer, "Id", &format!("urn:uuid:{}", asset.id))?;

        if let Some(ref annotation) = asset.annotation {
            Self::write_element(writer, "Annotation", annotation)?;
        }

        // Hash with algorithm attribute
        let mut hash_elem = BytesStart::new("Hash");
        hash_elem.push_attribute(("algorithm", asset.hash_algorithm.as_str()));
        writer
            .write_event(Event::Start(hash_elem))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;
        writer
            .write_event(Event::Text(BytesText::new(&asset.hash)))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;
        writer
            .write_event(Event::End(BytesEnd::new("Hash")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Self::write_element(writer, "Size", &asset.size.to_string())?;
        Self::write_element(writer, "Type", &asset.asset_type)?;

        if let Some(ref filename) = asset.original_filename {
            Self::write_element(writer, "OriginalFileName", filename)?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("Asset")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_algorithm() {
        assert_eq!(HashAlgorithm::Sha1.as_str(), "SHA-1");
        assert_eq!(HashAlgorithm::Md5.as_str(), "MD5");

        assert_eq!(
            HashAlgorithm::from_str("SHA-1").expect("test expectation failed"),
            HashAlgorithm::Sha1
        );
        assert_eq!(
            HashAlgorithm::from_str("MD5").expect("test expectation failed"),
            HashAlgorithm::Md5
        );
    }

    #[test]
    fn test_asset_creation() {
        let asset = Asset::new(
            Uuid::new_v4(),
            "abc123".to_string(),
            HashAlgorithm::Sha1,
            1024,
            "video/mxf".to_string(),
        );

        assert_eq!(asset.hash(), "abc123");
        assert_eq!(asset.size(), 1024);
        assert_eq!(asset.hash_algorithm(), HashAlgorithm::Sha1);
    }

    #[test]
    fn test_pkl_creation() {
        let mut pkl = PackingList::new(Uuid::new_v4());
        pkl.set_creator("OxiMedia".to_string());
        pkl.set_issuer("Test Studio".to_string());

        let asset = Asset::new(
            Uuid::new_v4(),
            "abc123".to_string(),
            HashAlgorithm::Sha1,
            1024,
            "video/mxf".to_string(),
        );

        pkl.add_asset(asset);

        assert_eq!(pkl.assets().len(), 1);
        assert_eq!(pkl.total_size(), 1024);
        assert_eq!(pkl.creator(), Some("OxiMedia"));
    }
}
