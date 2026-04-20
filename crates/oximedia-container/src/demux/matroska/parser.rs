//! Matroska element parser.
//!
//! This module provides parsing functions for Matroska container elements.
//! It operates on byte slices and produces structured data types.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::wildcard_imports)]

use super::ebml::{self, element_id, EbmlElement};
use super::matroska_v4;
use super::types::*;
use oximedia_core::{CodecId, OxiError, OxiResult};

// ============================================================================
// Parser State
// ============================================================================

/// Matroska element parser.
///
/// Parses EBML elements from a byte slice, maintaining position state.
pub struct MatroskaParser<'a> {
    /// Input data.
    data: &'a [u8],
    /// Current position in the data.
    position: usize,
}

impl<'a> MatroskaParser<'a> {
    /// Creates a new parser for the given data.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    /// Returns the current position.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Returns the remaining bytes.
    #[must_use]
    pub fn remaining(&self) -> &'a [u8] {
        &self.data[self.position..]
    }

    /// Returns true if at end of data.
    #[must_use]
    pub fn is_eof(&self) -> bool {
        self.position >= self.data.len()
    }

    /// Skips n bytes.
    pub fn skip(&mut self, n: usize) {
        self.position = self.position.saturating_add(n).min(self.data.len());
    }

    /// Reads the next element header.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn read_element(&mut self) -> OxiResult<EbmlElement> {
        let remaining = self.remaining();
        let (_, element) = ebml::parse_element_header(remaining).map_err(|e| OxiError::Parse {
            offset: self.position as u64,
            message: format!("Failed to parse element header: {e:?}"),
        })?;
        self.position += element.header_size;
        Ok(element)
    }

    /// Reads element data without parsing.
    ///
    /// # Errors
    ///
    /// Returns an error if there's not enough data.
    pub fn read_data(&mut self, size: usize) -> OxiResult<&'a [u8]> {
        if self.position + size > self.data.len() {
            return Err(OxiError::UnexpectedEof);
        }
        let data = &self.data[self.position..self.position + size];
        self.position += size;
        Ok(data)
    }

    /// Reads an unsigned integer element.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn read_uint(&mut self, size: usize) -> OxiResult<u64> {
        let data = self.read_data(size)?;
        let (_, value) = ebml::read_uint(data).map_err(|e| OxiError::Parse {
            offset: self.position as u64,
            message: format!("Failed to parse uint: {e:?}"),
        })?;
        Ok(value)
    }

    /// Reads a signed integer element.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn read_int(&mut self, size: usize) -> OxiResult<i64> {
        let data = self.read_data(size)?;
        let (_, value) = ebml::read_int(data).map_err(|e| OxiError::Parse {
            offset: self.position as u64,
            message: format!("Failed to parse int: {e:?}"),
        })?;
        Ok(value)
    }

    /// Reads a floating-point element.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn read_float(&mut self, size: usize) -> OxiResult<f64> {
        let data = self.read_data(size)?;
        ebml::read_float(data, size)
    }

    /// Reads a string element.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn read_string(&mut self, size: usize) -> OxiResult<String> {
        let data = self.read_data(size)?;
        ebml::read_string(data, size)
    }

    /// Reads a binary element.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn read_binary(&mut self, size: usize) -> OxiResult<Vec<u8>> {
        let data = self.read_data(size)?;
        Ok(data.to_vec())
    }
}

// ============================================================================
// EBML Header Parsing
// ============================================================================

/// Parses the EBML header.
///
/// # Errors
///
/// Returns an error if the header is invalid or incomplete.
pub fn parse_ebml_header(data: &[u8]) -> OxiResult<(EbmlHeader, usize)> {
    let mut parser = MatroskaParser::new(data);

    // Read EBML element
    let element = parser.read_element()?;
    if element.id != element_id::EBML {
        return Err(OxiError::Parse {
            offset: 0,
            message: format!("Expected EBML header, got 0x{:X}", element.id),
        });
    }

    let end_pos = parser.position() + element.size as usize;
    let mut header = EbmlHeader::default();

    while parser.position() < end_pos {
        let child = parser.read_element()?;
        let size = child.size as usize;

        match child.id {
            element_id::EBML_VERSION => {
                header.version = parser.read_uint(size)?;
            }
            element_id::EBML_READ_VERSION => {
                header.read_version = parser.read_uint(size)?;
            }
            element_id::EBML_MAX_ID_LENGTH => {
                header.max_id_length = parser.read_uint(size)?;
            }
            element_id::EBML_MAX_SIZE_LENGTH => {
                header.max_size_length = parser.read_uint(size)?;
            }
            element_id::DOC_TYPE => {
                let doc_type_str = parser.read_string(size)?;
                header.doc_type = match doc_type_str.as_str() {
                    "webm" => DocType::WebM,
                    "matroska" => DocType::Matroska,
                    _ => {
                        return Err(OxiError::Parse {
                            offset: parser.position() as u64,
                            message: format!("Unknown DocType: {doc_type_str}"),
                        });
                    }
                };
            }
            element_id::DOC_TYPE_VERSION => {
                header.doc_type_version = parser.read_uint(size)?;
            }
            element_id::DOC_TYPE_READ_VERSION => {
                header.doc_type_read_version = parser.read_uint(size)?;
            }
            _ => {
                // Skip unknown elements
                parser.skip(size);
            }
        }
    }

    Ok((header, parser.position()))
}

// ============================================================================
// Segment Info Parsing
// ============================================================================

/// Parses segment info.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_segment_info(data: &[u8], size: u64) -> OxiResult<SegmentInfo> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut info = SegmentInfo::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::TIMECODE_SCALE => {
                info.timecode_scale = parser.read_uint(elem_size)?;
            }
            element_id::DURATION => {
                info.duration = Some(parser.read_float(elem_size)?);
            }
            element_id::TITLE => {
                info.title = Some(parser.read_string(elem_size)?);
            }
            element_id::MUXING_APP => {
                info.muxing_app = Some(parser.read_string(elem_size)?);
            }
            element_id::WRITING_APP => {
                info.writing_app = Some(parser.read_string(elem_size)?);
            }
            element_id::DATE_UTC => {
                let date_data = parser.read_data(elem_size)?;
                info.date_utc = Some(ebml::read_date(date_data, elem_size)?);
            }
            element_id::SEGMENT_UID => {
                info.segment_uid = Some(parser.read_binary(elem_size)?);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    Ok(info)
}

// ============================================================================
// Track Parsing
// ============================================================================

/// Parses tracks element.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_tracks(data: &[u8], size: u64) -> OxiResult<Vec<TrackEntry>> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut tracks = Vec::new();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        if element.id == element_id::TRACK_ENTRY {
            let track_data = parser.read_data(elem_size)?;
            let track = parse_track_entry(track_data, element.size)?;
            tracks.push(track);
        } else {
            parser.skip(elem_size);
        }
    }

    Ok(tracks)
}

/// Parses a single track entry.
///
/// # Errors
///
/// Returns an error if parsing fails.
#[allow(clippy::too_many_lines)]
pub fn parse_track_entry(data: &[u8], size: u64) -> OxiResult<TrackEntry> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut track = TrackEntry::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::TRACK_NUMBER => {
                track.number = parser.read_uint(elem_size)?;
            }
            element_id::TRACK_UID => {
                track.uid = parser.read_uint(elem_size)?;
            }
            element_id::TRACK_TYPE => {
                let type_val = parser.read_uint(elem_size)?;
                track.track_type = TrackType::try_from(type_val).map_err(|()| OxiError::Parse {
                    offset: parser.position() as u64,
                    message: format!("Unknown track type: {type_val}"),
                })?;
            }
            element_id::FLAG_ENABLED => {
                track.enabled = parser.read_uint(elem_size)? != 0;
            }
            element_id::FLAG_DEFAULT => {
                track.default = parser.read_uint(elem_size)? != 0;
            }
            element_id::FLAG_FORCED => {
                track.forced = parser.read_uint(elem_size)? != 0;
            }
            element_id::FLAG_LACING => {
                track.lacing = parser.read_uint(elem_size)? != 0;
            }
            element_id::MIN_CACHE => {
                track.min_cache = parser.read_uint(elem_size)?;
            }
            element_id::MAX_CACHE => {
                track.max_cache = Some(parser.read_uint(elem_size)?);
            }
            element_id::DEFAULT_DURATION => {
                track.default_duration = Some(parser.read_uint(elem_size)?);
            }
            element_id::TRACK_TIMECODE_SCALE => {
                track.track_timecode_scale = parser.read_float(elem_size)?;
            }
            element_id::NAME => {
                track.name = Some(parser.read_string(elem_size)?);
            }
            element_id::LANGUAGE => {
                track.language = parser.read_string(elem_size)?;
            }
            element_id::LANGUAGE_IETF => {
                track.language_ietf = Some(parser.read_string(elem_size)?);
            }
            element_id::CODEC_ID => {
                track.codec_id = parser.read_string(elem_size)?;
            }
            element_id::CODEC_PRIVATE => {
                track.codec_private = Some(parser.read_binary(elem_size)?);
            }
            element_id::CODEC_NAME => {
                track.codec_name = Some(parser.read_string(elem_size)?);
            }
            element_id::CODEC_DELAY => {
                track.codec_delay = Some(parser.read_uint(elem_size)?);
            }
            element_id::SEEK_PRE_ROLL => {
                track.seek_pre_roll = Some(parser.read_uint(elem_size)?);
            }
            element_id::VIDEO => {
                let video_data = parser.read_data(elem_size)?;
                track.video = Some(parse_video_settings(video_data, element.size)?);
            }
            element_id::AUDIO => {
                let audio_data = parser.read_data(elem_size)?;
                track.audio = Some(parse_audio_settings(audio_data, element.size)?);
            }
            matroska_v4::v4_element_id::BLOCK_ADDITION_MAPPING => {
                let mapping_data = parser.read_data(elem_size)?;
                let mapping = matroska_v4::parse_block_addition_mapping(mapping_data)?;
                track.block_addition_mappings.push(mapping);
            }
            element_id::CONTENT_ENCODINGS => {
                // Skip content encodings for now (compression/encryption)
                parser.skip(elem_size);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    // Map codec ID
    track.oxi_codec = map_codec_id(&track.codec_id).ok();

    Ok(track)
}

/// Parses video settings.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_video_settings(data: &[u8], size: u64) -> OxiResult<VideoSettings> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut settings = VideoSettings::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::PIXEL_WIDTH => {
                settings.pixel_width = parser.read_uint(elem_size)? as u32;
            }
            element_id::PIXEL_HEIGHT => {
                settings.pixel_height = parser.read_uint(elem_size)? as u32;
            }
            element_id::DISPLAY_WIDTH => {
                settings.display_width = Some(parser.read_uint(elem_size)? as u32);
            }
            element_id::DISPLAY_HEIGHT => {
                settings.display_height = Some(parser.read_uint(elem_size)? as u32);
            }
            element_id::DISPLAY_UNIT => {
                settings.display_unit = parser.read_uint(elem_size)? as u8;
            }
            element_id::FLAG_INTERLACED => {
                settings.interlaced = parser.read_uint(elem_size)? != 0;
            }
            element_id::FIELD_ORDER => {
                settings.field_order = Some(parser.read_uint(elem_size)? as u8);
            }
            element_id::STEREO_MODE => {
                settings.stereo_mode = Some(parser.read_uint(elem_size)? as u8);
            }
            element_id::ALPHA_MODE => {
                settings.alpha_mode = Some(parser.read_uint(elem_size)? as u8);
            }
            element_id::PIXEL_CROP_BOTTOM => {
                settings.pixel_crop_bottom = parser.read_uint(elem_size)? as u32;
            }
            element_id::PIXEL_CROP_TOP => {
                settings.pixel_crop_top = parser.read_uint(elem_size)? as u32;
            }
            element_id::PIXEL_CROP_LEFT => {
                settings.pixel_crop_left = parser.read_uint(elem_size)? as u32;
            }
            element_id::PIXEL_CROP_RIGHT => {
                settings.pixel_crop_right = parser.read_uint(elem_size)? as u32;
            }
            element_id::COLOUR => {
                let colour_data = parser.read_data(elem_size)?;
                settings.colour = Some(parse_colour_settings(colour_data, element.size)?);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    Ok(settings)
}

/// Parses colour settings.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_colour_settings(data: &[u8], size: u64) -> OxiResult<ColourSettings> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut settings = ColourSettings::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::MATRIX_COEFFICIENTS => {
                settings.matrix_coefficients = Some(parser.read_uint(elem_size)? as u8);
            }
            element_id::BITS_PER_CHANNEL => {
                settings.bits_per_channel = Some(parser.read_uint(elem_size)? as u8);
            }
            element_id::CHROMA_SUBSAMPLING_HORZ => {
                settings.chroma_subsampling_horz = Some(parser.read_uint(elem_size)? as u8);
            }
            element_id::CHROMA_SUBSAMPLING_VERT => {
                settings.chroma_subsampling_vert = Some(parser.read_uint(elem_size)? as u8);
            }
            element_id::CB_SUBSAMPLING_HORZ => {
                settings.cb_subsampling_horz = Some(parser.read_uint(elem_size)? as u8);
            }
            element_id::CB_SUBSAMPLING_VERT => {
                settings.cb_subsampling_vert = Some(parser.read_uint(elem_size)? as u8);
            }
            element_id::CHROMA_SITING_HORZ => {
                settings.chroma_siting_horz = Some(parser.read_uint(elem_size)? as u8);
            }
            element_id::CHROMA_SITING_VERT => {
                settings.chroma_siting_vert = Some(parser.read_uint(elem_size)? as u8);
            }
            element_id::RANGE => {
                settings.range = Some(parser.read_uint(elem_size)? as u8);
            }
            element_id::TRANSFER_CHARACTERISTICS => {
                settings.transfer_characteristics = Some(parser.read_uint(elem_size)? as u8);
            }
            element_id::PRIMARIES => {
                settings.primaries = Some(parser.read_uint(elem_size)? as u8);
            }
            element_id::MAX_CLL => {
                settings.max_cll = Some(parser.read_uint(elem_size)?);
            }
            element_id::MAX_FALL => {
                settings.max_fall = Some(parser.read_uint(elem_size)?);
            }
            element_id::MASTERING_METADATA => {
                let meta_data = parser.read_data(elem_size)?;
                settings.mastering_metadata =
                    Some(parse_mastering_metadata(meta_data, element.size)?);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    Ok(settings)
}

/// Parses mastering metadata.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_mastering_metadata(data: &[u8], size: u64) -> OxiResult<MasteringMetadata> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut meta = MasteringMetadata::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::PRIMARY_R_CHROMATICITY_X => {
                meta.primary_r_chromaticity_x = Some(parser.read_float(elem_size)?);
            }
            element_id::PRIMARY_R_CHROMATICITY_Y => {
                meta.primary_r_chromaticity_y = Some(parser.read_float(elem_size)?);
            }
            element_id::PRIMARY_G_CHROMATICITY_X => {
                meta.primary_g_chromaticity_x = Some(parser.read_float(elem_size)?);
            }
            element_id::PRIMARY_G_CHROMATICITY_Y => {
                meta.primary_g_chromaticity_y = Some(parser.read_float(elem_size)?);
            }
            element_id::PRIMARY_B_CHROMATICITY_X => {
                meta.primary_b_chromaticity_x = Some(parser.read_float(elem_size)?);
            }
            element_id::PRIMARY_B_CHROMATICITY_Y => {
                meta.primary_b_chromaticity_y = Some(parser.read_float(elem_size)?);
            }
            element_id::WHITE_POINT_CHROMATICITY_X => {
                meta.white_point_chromaticity_x = Some(parser.read_float(elem_size)?);
            }
            element_id::WHITE_POINT_CHROMATICITY_Y => {
                meta.white_point_chromaticity_y = Some(parser.read_float(elem_size)?);
            }
            element_id::LUMINANCE_MAX => {
                meta.luminance_max = Some(parser.read_float(elem_size)?);
            }
            element_id::LUMINANCE_MIN => {
                meta.luminance_min = Some(parser.read_float(elem_size)?);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    Ok(meta)
}

/// Parses audio settings.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_audio_settings(data: &[u8], size: u64) -> OxiResult<AudioSettings> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut settings = AudioSettings::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::SAMPLING_FREQUENCY => {
                settings.sampling_frequency = parser.read_float(elem_size)?;
            }
            element_id::OUTPUT_SAMPLING_FREQUENCY => {
                settings.output_sampling_frequency = Some(parser.read_float(elem_size)?);
            }
            element_id::CHANNELS => {
                settings.channels = parser.read_uint(elem_size)? as u8;
            }
            element_id::BIT_DEPTH => {
                settings.bit_depth = Some(parser.read_uint(elem_size)? as u8);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    Ok(settings)
}

// ============================================================================
// Cues Parsing
// ============================================================================

/// Parses cues element.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_cues(data: &[u8], size: u64) -> OxiResult<Vec<CuePoint>> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut cues = Vec::new();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        if element.id == element_id::CUE_POINT {
            let cue_data = parser.read_data(elem_size)?;
            let cue = parse_cue_point(cue_data, element.size)?;
            cues.push(cue);
        } else {
            parser.skip(elem_size);
        }
    }

    Ok(cues)
}

/// Parses a single cue point.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_cue_point(data: &[u8], size: u64) -> OxiResult<CuePoint> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut cue = CuePoint::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::CUE_TIME => {
                cue.time = parser.read_uint(elem_size)?;
            }
            element_id::CUE_TRACK_POSITIONS => {
                let pos_data = parser.read_data(elem_size)?;
                let pos = parse_cue_track_position(pos_data, element.size)?;
                cue.track_positions.push(pos);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    Ok(cue)
}

/// Parses cue track positions.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_cue_track_position(data: &[u8], size: u64) -> OxiResult<CueTrackPosition> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut pos = CueTrackPosition::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::CUE_TRACK => {
                pos.track = parser.read_uint(elem_size)?;
            }
            element_id::CUE_CLUSTER_POSITION => {
                pos.cluster_position = parser.read_uint(elem_size)?;
            }
            element_id::CUE_RELATIVE_POSITION => {
                pos.relative_position = Some(parser.read_uint(elem_size)?);
            }
            element_id::CUE_DURATION => {
                pos.duration = Some(parser.read_uint(elem_size)?);
            }
            element_id::CUE_BLOCK_NUMBER => {
                pos.block_number = Some(parser.read_uint(elem_size)?);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    Ok(pos)
}

// ============================================================================
// Chapter Parsing
// ============================================================================

/// Parses chapters element.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_chapters(data: &[u8], size: u64) -> OxiResult<Vec<Edition>> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut editions = Vec::new();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        if element.id == element_id::EDITION_ENTRY {
            let edition_data = parser.read_data(elem_size)?;
            let edition = parse_edition(edition_data, element.size)?;
            editions.push(edition);
        } else {
            parser.skip(elem_size);
        }
    }

    Ok(editions)
}

/// Parses an edition entry.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_edition(data: &[u8], size: u64) -> OxiResult<Edition> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut edition = Edition::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::EDITION_UID => {
                edition.uid = Some(parser.read_uint(elem_size)?);
            }
            element_id::EDITION_FLAG_HIDDEN => {
                edition.hidden = parser.read_uint(elem_size)? != 0;
            }
            element_id::EDITION_FLAG_DEFAULT => {
                edition.default = parser.read_uint(elem_size)? != 0;
            }
            element_id::EDITION_FLAG_ORDERED => {
                edition.ordered = parser.read_uint(elem_size)? != 0;
            }
            element_id::CHAPTER_ATOM => {
                let chapter_data = parser.read_data(elem_size)?;
                let chapter = parse_chapter(chapter_data, element.size)?;
                edition.chapters.push(chapter);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    Ok(edition)
}

/// Parses a chapter atom.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_chapter(data: &[u8], size: u64) -> OxiResult<Chapter> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut chapter = Chapter::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::CHAPTER_UID => {
                chapter.uid = parser.read_uint(elem_size)?;
            }
            element_id::CHAPTER_STRING_UID => {
                chapter.string_uid = Some(parser.read_string(elem_size)?);
            }
            element_id::CHAPTER_TIME_START => {
                chapter.time_start = parser.read_uint(elem_size)?;
            }
            element_id::CHAPTER_TIME_END => {
                chapter.time_end = Some(parser.read_uint(elem_size)?);
            }
            element_id::CHAPTER_FLAG_HIDDEN => {
                chapter.hidden = parser.read_uint(elem_size)? != 0;
            }
            element_id::CHAPTER_FLAG_ENABLED => {
                chapter.enabled = parser.read_uint(elem_size)? != 0;
            }
            element_id::CHAPTER_DISPLAY => {
                let display_data = parser.read_data(elem_size)?;
                let display = parse_chapter_display(display_data, element.size)?;
                chapter.display.push(display);
            }
            element_id::CHAPTER_ATOM => {
                // Nested chapter
                let child_data = parser.read_data(elem_size)?;
                let child = parse_chapter(child_data, element.size)?;
                chapter.children.push(child);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    Ok(chapter)
}

/// Parses chapter display.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_chapter_display(data: &[u8], size: u64) -> OxiResult<ChapterDisplay> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut display = ChapterDisplay::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::CHAP_STRING => {
                display.string = parser.read_string(elem_size)?;
            }
            element_id::CHAP_LANGUAGE => {
                display.language = parser.read_string(elem_size)?;
            }
            element_id::CHAP_LANGUAGE_IETF => {
                display.language_ietf = Some(parser.read_string(elem_size)?);
            }
            element_id::CHAP_COUNTRY => {
                display.country = Some(parser.read_string(elem_size)?);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    Ok(display)
}

// ============================================================================
// Tags Parsing
// ============================================================================

/// Parses tags element.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_tags(data: &[u8], size: u64) -> OxiResult<Vec<Tag>> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut tags = Vec::new();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        if element.id == element_id::TAG {
            let tag_data = parser.read_data(elem_size)?;
            let tag = parse_tag(tag_data, element.size)?;
            tags.push(tag);
        } else {
            parser.skip(elem_size);
        }
    }

    Ok(tags)
}

/// Parses a single tag.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_tag(data: &[u8], size: u64) -> OxiResult<Tag> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut tag = Tag::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::TARGETS => {
                let targets_data = parser.read_data(elem_size)?;
                tag.targets = parse_tag_targets(targets_data, element.size)?;
            }
            element_id::SIMPLE_TAG => {
                let simple_tag_data = parser.read_data(elem_size)?;
                let simple_tag = parse_simple_tag(simple_tag_data, element.size)?;
                tag.simple_tags.push(simple_tag);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    Ok(tag)
}

/// Parses tag targets.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_tag_targets(data: &[u8], size: u64) -> OxiResult<TagTargets> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut targets = TagTargets::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::TARGET_TYPE_VALUE => {
                targets.target_type_value = Some(parser.read_uint(elem_size)?);
            }
            element_id::TARGET_TYPE => {
                targets.target_type = Some(parser.read_string(elem_size)?);
            }
            element_id::TAG_TRACK_UID => {
                targets.track_uid.push(parser.read_uint(elem_size)?);
            }
            element_id::TAG_EDITION_UID => {
                targets.edition_uid.push(parser.read_uint(elem_size)?);
            }
            element_id::TAG_CHAPTER_UID => {
                targets.chapter_uid.push(parser.read_uint(elem_size)?);
            }
            element_id::TAG_ATTACHMENT_UID => {
                targets.attachment_uid.push(parser.read_uint(elem_size)?);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    Ok(targets)
}

/// Parses a simple tag.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_simple_tag(data: &[u8], size: u64) -> OxiResult<SimpleTag> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut tag = SimpleTag::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::TAG_NAME => {
                tag.name = parser.read_string(elem_size)?;
            }
            element_id::TAG_LANGUAGE => {
                tag.language = parser.read_string(elem_size)?;
            }
            element_id::TAG_LANGUAGE_IETF => {
                tag.language_ietf = Some(parser.read_string(elem_size)?);
            }
            element_id::TAG_DEFAULT => {
                tag.default = parser.read_uint(elem_size)? != 0;
            }
            element_id::TAG_STRING => {
                tag.string = Some(parser.read_string(elem_size)?);
            }
            element_id::TAG_BINARY => {
                tag.binary = Some(parser.read_binary(elem_size)?);
            }
            element_id::SIMPLE_TAG => {
                // Nested tag
                let child_data = parser.read_data(elem_size)?;
                let child = parse_simple_tag(child_data, element.size)?;
                tag.children.push(child);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    Ok(tag)
}

// ============================================================================
// Seek Head Parsing
// ============================================================================

/// Parses seek head element.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_seek_head(data: &[u8], size: u64) -> OxiResult<Vec<SeekEntry>> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut entries = Vec::new();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        if element.id == element_id::SEEK {
            let seek_data = parser.read_data(elem_size)?;
            let entry = parse_seek_entry(seek_data, element.size)?;
            entries.push(entry);
        } else {
            parser.skip(elem_size);
        }
    }

    Ok(entries)
}

/// Parses a single seek entry.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_seek_entry(data: &[u8], size: u64) -> OxiResult<SeekEntry> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut entry = SeekEntry::default();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::SEEK_ID => {
                let id_data = parser.read_data(elem_size)?;
                let (_, id) = ebml::parse_element_id(id_data).map_err(|e| OxiError::Parse {
                    offset: parser.position() as u64,
                    message: format!("Failed to parse seek ID: {e:?}"),
                })?;
                entry.id = id;
            }
            element_id::SEEK_POSITION => {
                entry.position = parser.read_uint(elem_size)?;
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    Ok(entry)
}

// ============================================================================
// Block Parsing
// ============================================================================

/// Parses a block header.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_block_header(data: &[u8]) -> OxiResult<(BlockHeader, usize)> {
    if data.is_empty() {
        return Err(OxiError::UnexpectedEof);
    }

    // Parse track number (VINT)
    let (remaining, track_number) = ebml::parse_vint(data).map_err(|e| OxiError::Parse {
        offset: 0,
        message: format!("Failed to parse block track number: {e:?}"),
    })?;

    let header_consumed = data.len() - remaining.len();

    if remaining.len() < 3 {
        return Err(OxiError::UnexpectedEof);
    }

    // Parse timecode (signed 16-bit big-endian)
    let timecode = i16::from_be_bytes([remaining[0], remaining[1]]);

    // Parse flags
    let flags = remaining[2];
    let keyframe = (flags & 0x80) != 0;
    let invisible = (flags & 0x08) != 0;
    let lacing_bits = (flags >> 1) & 0x03;
    let discardable = (flags & 0x01) != 0;

    let lacing = LacingType::try_from(lacing_bits).map_err(|()| OxiError::Parse {
        offset: header_consumed as u64 + 2,
        message: format!("Invalid lacing type: {lacing_bits}"),
    })?;

    let header = BlockHeader {
        track_number,
        timecode,
        keyframe,
        invisible,
        lacing,
        discardable,
    };

    Ok((header, header_consumed + 3))
}

/// Parses a simple block.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_simple_block(data: &[u8]) -> OxiResult<Block> {
    let (header, header_size) = parse_block_header(data)?;
    let remaining = &data[header_size..];

    let frames = parse_laced_frames(remaining, header.lacing)?;

    Ok(Block {
        header,
        frames,
        duration: None,
        references: Vec::new(),
        discard_padding: None,
    })
}

/// Parses a block group.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_block_group(data: &[u8], size: u64) -> OxiResult<Block> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut block = Block::default();
    let mut block_data: Option<Vec<u8>> = None;

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            element_id::BLOCK => {
                block_data = Some(parser.read_binary(elem_size)?);
            }
            element_id::BLOCK_DURATION => {
                block.duration = Some(parser.read_uint(elem_size)?);
            }
            element_id::REFERENCE_BLOCK => {
                block.references.push(parser.read_int(elem_size)?);
            }
            element_id::DISCARD_PADDING => {
                block.discard_padding = Some(parser.read_int(elem_size)?);
            }
            _ => {
                parser.skip(elem_size);
            }
        }
    }

    if let Some(data) = block_data {
        let (header, header_size) = parse_block_header(&data)?;
        let remaining = &data[header_size..];
        let frames = parse_laced_frames(remaining, header.lacing)?;
        block.header = header;
        block.frames = frames;
    }

    Ok(block)
}

/// Parses laced frames.
///
/// # Errors
///
/// Returns an error if parsing fails.
fn parse_laced_frames(data: &[u8], lacing: LacingType) -> OxiResult<Vec<Vec<u8>>> {
    match lacing {
        LacingType::None => {
            // No lacing - entire data is a single frame
            Ok(vec![data.to_vec()])
        }
        LacingType::Xiph => parse_xiph_lacing(data),
        LacingType::Ebml => parse_ebml_lacing(data),
        LacingType::FixedSize => parse_fixed_size_lacing(data),
    }
}

/// Parses Xiph-style lacing.
fn parse_xiph_lacing(data: &[u8]) -> OxiResult<Vec<Vec<u8>>> {
    if data.is_empty() {
        return Err(OxiError::UnexpectedEof);
    }

    let frame_count = usize::from(data[0]) + 1;
    let mut pos = 1;
    let mut sizes = Vec::with_capacity(frame_count);

    // Read sizes for all frames except the last
    for _ in 0..frame_count - 1 {
        let mut size = 0usize;
        loop {
            if pos >= data.len() {
                return Err(OxiError::UnexpectedEof);
            }
            let byte = data[pos];
            pos += 1;
            size += usize::from(byte);
            if byte != 255 {
                break;
            }
        }
        sizes.push(size);
    }

    // Extract frames
    let mut frames = Vec::with_capacity(frame_count);
    for size in &sizes {
        if pos + size > data.len() {
            return Err(OxiError::UnexpectedEof);
        }
        frames.push(data[pos..pos + size].to_vec());
        pos += size;
    }

    // Last frame is the remainder
    frames.push(data[pos..].to_vec());

    Ok(frames)
}

/// Parses EBML-style lacing.
fn parse_ebml_lacing(data: &[u8]) -> OxiResult<Vec<Vec<u8>>> {
    if data.is_empty() {
        return Err(OxiError::UnexpectedEof);
    }

    let frame_count = usize::from(data[0]) + 1;
    let mut pos = 1;
    let mut sizes = Vec::with_capacity(frame_count);

    // First frame size is a regular VINT
    if frame_count > 1 {
        let (remaining, first_size) =
            ebml::parse_vint(&data[pos..]).map_err(|e| OxiError::Parse {
                offset: pos as u64,
                message: format!("Failed to parse EBML lacing size: {e:?}"),
            })?;
        pos = data.len() - remaining.len();
        sizes.push(first_size as usize);

        // Subsequent sizes are signed deltas
        let mut prev_size = first_size;
        for _ in 1..frame_count - 1 {
            let (remaining, delta) =
                ebml::parse_vint(&data[pos..]).map_err(|e| OxiError::Parse {
                    offset: pos as u64,
                    message: format!("Failed to parse EBML lacing delta: {e:?}"),
                })?;

            // Convert to signed delta
            let vint_len = data.len() - remaining.len() - pos;
            let mid = (1u64 << (7 * vint_len - 1)) - 1;
            #[allow(clippy::cast_possible_wrap)]
            let signed_delta = delta as i64 - mid as i64;

            pos = data.len() - remaining.len();

            #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
            let new_size = (prev_size as i64 + signed_delta) as u64;
            sizes.push(new_size as usize);
            prev_size = new_size;
        }
    }

    // Extract frames
    let mut frames = Vec::with_capacity(frame_count);
    for size in &sizes {
        if pos + size > data.len() {
            return Err(OxiError::UnexpectedEof);
        }
        frames.push(data[pos..pos + size].to_vec());
        pos += size;
    }

    // Last frame is the remainder
    frames.push(data[pos..].to_vec());

    Ok(frames)
}

/// Parses fixed-size lacing.
fn parse_fixed_size_lacing(data: &[u8]) -> OxiResult<Vec<Vec<u8>>> {
    if data.is_empty() {
        return Err(OxiError::UnexpectedEof);
    }

    let frame_count = usize::from(data[0]) + 1;
    let remaining_data = &data[1..];

    if remaining_data.len() % frame_count != 0 {
        return Err(OxiError::InvalidData(format!(
            "Fixed-size lacing: data length {} not divisible by frame count {}",
            remaining_data.len(),
            frame_count
        )));
    }

    let frame_size = remaining_data.len() / frame_count;
    let mut frames = Vec::with_capacity(frame_count);

    for i in 0..frame_count {
        let start = i * frame_size;
        frames.push(remaining_data[start..start + frame_size].to_vec());
    }

    Ok(frames)
}

// ============================================================================
// Codec ID Mapping
// ============================================================================

/// Maps Matroska codec ID to `OxiMedia` `CodecId`.
///
/// # Errors
///
/// Returns `PatentViolation` for patent-encumbered codecs.
/// Returns `Unsupported` for unknown codecs.
pub fn map_codec_id(mkv_codec: &str) -> OxiResult<CodecId> {
    match mkv_codec {
        // Video codecs (Green List)
        "V_VP9" => Ok(CodecId::Vp9),
        "V_VP8" => Ok(CodecId::Vp8),
        "V_AV1" => Ok(CodecId::Av1),
        "V_THEORA" => Ok(CodecId::Theora),

        // Audio codecs (Green List)
        "A_OPUS" => Ok(CodecId::Opus),
        "A_VORBIS" => Ok(CodecId::Vorbis),
        "A_FLAC" => Ok(CodecId::Flac),
        "A_PCM/INT/LIT" | "A_PCM/INT/BIG" | "A_PCM/FLOAT/IEEE" => Ok(CodecId::Pcm),

        // Subtitle formats
        "S_TEXT/WEBVTT" | "D_WEBVTT/SUBTITLES" | "D_WEBVTT/CAPTIONS" => Ok(CodecId::WebVtt),
        "S_TEXT/ASS" | "S_ASS" => Ok(CodecId::Ass),
        "S_TEXT/SSA" | "S_SSA" => Ok(CodecId::Ssa),
        "S_TEXT/UTF8" => Ok(CodecId::Srt),

        // Patent-encumbered video - reject
        "V_MPEG4/ISO/AVC" | "V_MPEG4/ISO/AVC/ES" => {
            Err(OxiError::PatentViolation("H.264/AVC".into()))
        }
        "V_MPEGH/ISO/HEVC" => Err(OxiError::PatentViolation("H.265/HEVC".into())),
        "V_MPEG4/ISO/SP" | "V_MPEG4/ISO/ASP" | "V_MPEG4/ISO/AP" | "V_MPEG4/MS/V3" => {
            Err(OxiError::PatentViolation("MPEG-4 Visual".into()))
        }
        "V_MPEG1" | "V_MPEG2" => Err(OxiError::PatentViolation("MPEG-1/2 Video".into())),
        s if s.starts_with("V_MS/VFW/FOURCC") => Err(OxiError::PatentViolation("VFW codec".into())),

        // Patent-encumbered audio - reject
        "A_AAC" | "A_AAC/MPEG2/LC" | "A_AAC/MPEG2/MAIN" | "A_AAC/MPEG2/SSR" | "A_AAC/MPEG4/LC"
        | "A_AAC/MPEG4/MAIN" | "A_AAC/MPEG4/SSR" | "A_AAC/MPEG4/LTP" | "A_AAC/MPEG4/SBR" => {
            Err(OxiError::PatentViolation("AAC".into()))
        }
        "A_AC3" | "A_AC3/BSID9" | "A_AC3/BSID10" => Err(OxiError::PatentViolation("AC-3".into())),
        "A_EAC3" => Err(OxiError::PatentViolation("E-AC-3".into())),
        "A_DTS" | "A_DTS/EXPRESS" | "A_DTS/LOSSLESS" => {
            Err(OxiError::PatentViolation("DTS".into()))
        }
        "A_MPEG/L3" | "A_MPEG/L2" | "A_MPEG/L1" => {
            Err(OxiError::PatentViolation("MPEG Audio".into()))
        }
        s if s.starts_with("A_MS/ACM") => Err(OxiError::PatentViolation("ACM codec".into())),
        "A_TRUEHD" | "A_MLP" => Err(OxiError::PatentViolation("TrueHD/MLP".into())),

        // Unknown codec
        _ => Err(OxiError::Unsupported(format!("Unknown codec: {mkv_codec}"))),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_block_header() {
        // Track number 1, timecode 0, keyframe, no lacing
        let data = [0x81, 0x00, 0x00, 0x80];
        let (header, size) = parse_block_header(&data).expect("operation should succeed");

        assert_eq!(header.track_number, 1);
        assert_eq!(header.timecode, 0);
        assert!(header.keyframe);
        assert!(!header.invisible);
        assert_eq!(header.lacing, LacingType::None);
        assert!(!header.discardable);
        assert_eq!(size, 4);
    }

    #[test]
    fn test_parse_block_header_with_lacing() {
        // Track number 1, timecode 33, Xiph lacing
        let data = [0x81, 0x00, 0x21, 0x02];
        let (header, _) = parse_block_header(&data).expect("operation should succeed");

        assert_eq!(header.track_number, 1);
        assert_eq!(header.timecode, 33);
        assert!(!header.keyframe);
        assert_eq!(header.lacing, LacingType::Xiph);
    }

    #[test]
    fn test_map_codec_id_green_list() {
        assert_eq!(
            map_codec_id("V_VP9").expect("operation should succeed"),
            CodecId::Vp9
        );
        assert_eq!(
            map_codec_id("V_VP8").expect("operation should succeed"),
            CodecId::Vp8
        );
        assert_eq!(
            map_codec_id("V_AV1").expect("operation should succeed"),
            CodecId::Av1
        );
        assert_eq!(
            map_codec_id("V_THEORA").expect("operation should succeed"),
            CodecId::Theora
        );
        assert_eq!(
            map_codec_id("A_OPUS").expect("operation should succeed"),
            CodecId::Opus
        );
        assert_eq!(
            map_codec_id("A_VORBIS").expect("operation should succeed"),
            CodecId::Vorbis
        );
        assert_eq!(
            map_codec_id("A_FLAC").expect("operation should succeed"),
            CodecId::Flac
        );
        assert_eq!(
            map_codec_id("A_PCM/INT/LIT").expect("operation should succeed"),
            CodecId::Pcm
        );
        assert_eq!(
            map_codec_id("S_TEXT/WEBVTT").expect("operation should succeed"),
            CodecId::WebVtt
        );
        assert_eq!(
            map_codec_id("S_TEXT/ASS").expect("operation should succeed"),
            CodecId::Ass
        );
    }

    #[test]
    fn test_map_codec_id_patent_violation() {
        assert!(matches!(
            map_codec_id("V_MPEG4/ISO/AVC"),
            Err(OxiError::PatentViolation(_))
        ));
        assert!(matches!(
            map_codec_id("V_MPEGH/ISO/HEVC"),
            Err(OxiError::PatentViolation(_))
        ));
        assert!(matches!(
            map_codec_id("A_AAC"),
            Err(OxiError::PatentViolation(_))
        ));
        assert!(matches!(
            map_codec_id("A_AC3"),
            Err(OxiError::PatentViolation(_))
        ));
        assert!(matches!(
            map_codec_id("A_DTS"),
            Err(OxiError::PatentViolation(_))
        ));
    }

    #[test]
    fn test_map_codec_id_unknown() {
        assert!(matches!(
            map_codec_id("V_UNKNOWN"),
            Err(OxiError::Unsupported(_))
        ));
    }

    #[test]
    fn test_parse_fixed_size_lacing() {
        // 2 frames, each 4 bytes
        let data = [0x01, 1, 2, 3, 4, 5, 6, 7, 8];
        let frames = parse_fixed_size_lacing(&data).expect("operation should succeed");

        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], vec![1, 2, 3, 4]);
        assert_eq!(frames[1], vec![5, 6, 7, 8]);
    }

    #[test]
    fn test_parse_xiph_lacing() {
        // 2 frames: first 3 bytes, second 5 bytes
        let data = [0x01, 3, 1, 2, 3, 4, 5, 6, 7, 8];
        let frames = parse_xiph_lacing(&data).expect("operation should succeed");

        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], vec![1, 2, 3]);
        assert_eq!(frames[1], vec![4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_parse_simple_block() {
        // Track 1, timecode 0, keyframe, no lacing, 4 bytes data
        let data = [0x81, 0x00, 0x00, 0x80, 1, 2, 3, 4];
        let block = parse_simple_block(&data).expect("operation should succeed");

        assert_eq!(block.header.track_number, 1);
        assert!(block.header.keyframe);
        assert_eq!(block.frames.len(), 1);
        assert_eq!(block.frames[0], vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_ebml_header_parsing() {
        // Minimal EBML header for WebM
        // Content is 23 bytes: 4+4+4+4+7 = 23
        let data = [
            0x1A, 0x45, 0xDF, 0xA3, // EBML ID
            0x97, // Size: 23 bytes (0x80 + 23 = 0x97)
            0x42, 0x86, 0x81, 0x01, // EBMLVersion: 1
            0x42, 0xF7, 0x81, 0x01, // EBMLReadVersion: 1
            0x42, 0xF2, 0x81, 0x04, // EBMLMaxIDLength: 4
            0x42, 0xF3, 0x81, 0x08, // EBMLMaxSizeLength: 8
            0x42, 0x82, 0x84, b'w', b'e', b'b', b'm', // DocType: "webm"
        ];

        let (header, consumed) = parse_ebml_header(&data).expect("operation should succeed");

        assert_eq!(header.version, 1);
        assert_eq!(header.read_version, 1);
        assert_eq!(header.max_id_length, 4);
        assert_eq!(header.max_size_length, 8);
        assert_eq!(header.doc_type, DocType::WebM);
        assert_eq!(consumed, 28); // 5 (header) + 23 (content) = 28
    }
}
