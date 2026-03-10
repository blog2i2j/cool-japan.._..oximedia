//! DNG-specific TIFF tag constants.

// ==========================================
// DNG-specific TIFF tag constants
// ==========================================

/// DNG version tag.
pub const TAG_DNG_VERSION: u16 = 50706;
/// DNG backward version tag.
pub const TAG_DNG_BACKWARD_VERSION: u16 = 50707;
/// Unique camera model name.
pub const TAG_UNIQUE_CAMERA_MODEL: u16 = 50708;
/// Color matrix 1 (maps camera color space to XYZ).
pub const TAG_COLOR_MATRIX_1: u16 = 50721;
/// Color matrix 2.
pub const TAG_COLOR_MATRIX_2: u16 = 50722;
/// Camera calibration matrix 1.
pub const TAG_CAMERA_CALIBRATION_1: u16 = 50723;
/// Analog balance ratios.
pub const TAG_ANALOG_BALANCE: u16 = 50727;
/// As-shot neutral white balance.
pub const TAG_AS_SHOT_NEUTRAL: u16 = 50728;
/// Calibration illuminant 1 (EXIF LightSource values).
pub const TAG_CALIBRATION_ILLUMINANT_1: u16 = 50778;
/// Calibration illuminant 2.
pub const TAG_CALIBRATION_ILLUMINANT_2: u16 = 50779;
/// Active area of the sensor (top, left, bottom, right).
pub const TAG_ACTIVE_AREA: u16 = 50829;
/// Forward matrix 1 (maps white-balanced camera to XYZ).
pub const TAG_FORWARD_MATRIX_1: u16 = 50964;
/// Forward matrix 2.
pub const TAG_FORWARD_MATRIX_2: u16 = 50965;
/// Opcode list 1 (applied to raw data).
pub const TAG_OPCODE_LIST_1: u16 = 51008;
/// Opcode list 2 (applied after mapping to linear reference).
pub const TAG_OPCODE_LIST_2: u16 = 51009;
/// Opcode list 3 (applied after demosaicing).
pub const TAG_OPCODE_LIST_3: u16 = 51022;
/// New raw image digest (MD5).
pub const TAG_NEW_RAW_IMAGE_DIGEST: u16 = 51111;
/// Black level per channel.
pub const TAG_BLACK_LEVEL: u16 = 50714;
/// White level per channel.
pub const TAG_WHITE_LEVEL: u16 = 50717;

// Standard TIFF tags used by DNG
pub(crate) const TAG_IMAGE_WIDTH: u16 = 256;
pub(crate) const TAG_IMAGE_LENGTH: u16 = 257;
pub(crate) const TAG_BITS_PER_SAMPLE: u16 = 258;
pub(crate) const TAG_COMPRESSION: u16 = 259;
pub(crate) const TAG_PHOTOMETRIC_INTERPRETATION: u16 = 262;
pub(crate) const TAG_STRIP_OFFSETS: u16 = 273;
pub(crate) const TAG_SAMPLES_PER_PIXEL: u16 = 277;
pub(crate) const TAG_ROWS_PER_STRIP: u16 = 278;
pub(crate) const TAG_STRIP_BYTE_COUNTS: u16 = 279;
pub(crate) const TAG_CFA_REPEAT_PATTERN_DIM: u16 = 33421;
pub(crate) const TAG_CFA_PATTERN: u16 = 33422;
pub(crate) const TAG_SOFTWARE: u16 = 305;
