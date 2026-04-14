//! AV1 Quantization parameters.
//!
//! Quantization controls the precision of transform coefficients. AV1 uses
//! adaptive quantization with separate parameters for DC and AC coefficients,
//! and supports per-plane delta quantization.
//!
//! # Quantization Parameters
//!
//! - Base Q index (0-255): Primary quantization level
//! - Delta Q values: Adjustments for DC and AC, per plane
//! - Quantization matrices (QM): Optional matrices for coefficient weighting
//! - Delta Q resolution: Precision for per-block delta Q
//!
//! # Dequantization
//!
//! The dequantizer values are derived from lookup tables based on the
//! effective Q index (base + delta).
//!
//! # Reference
//!
//! See AV1 Specification Section 5.9.12 for quantization syntax and
//! Section 7.12 for quantization semantics.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unnecessary_min_or_max)]
#![allow(clippy::unused_self)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_errors_doc)]

use super::sequence::SequenceHeader;
use crate::error::{CodecError, CodecResult};
use oximedia_io::BitReader;

// =============================================================================
// Constants
// =============================================================================

/// Maximum Q index value.
pub const MAX_Q_INDEX: u8 = 255;

/// Minimum Q index value.
pub const MIN_Q_INDEX: u8 = 0;

/// Number of Q index values.
pub const QINDEX_RANGE: usize = 256;

/// Maximum delta Q value (in absolute terms).
pub const MAX_DELTA_Q: i8 = 63;

/// Minimum delta Q value (in absolute terms).
pub const MIN_DELTA_Q: i8 = -64;

/// Delta Q bits in bitstream.
pub const DELTA_Q_BITS: u8 = 6;

/// Number of QM levels.
pub const NUM_QM_LEVELS: usize = 16;

// =============================================================================
// DC and AC Dequantizer Tables (8-bit)
// =============================================================================

/// DC dequantizer lookup table for 8-bit depth.
pub const DC_QLOOKUP: [i16; QINDEX_RANGE] = [
    4, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 23, 24, 25, 26, 26, 27,
    28, 29, 30, 31, 32, 32, 33, 34, 35, 36, 37, 38, 38, 39, 40, 41, 42, 43, 43, 44, 45, 46, 47, 48,
    48, 49, 50, 51, 52, 53, 53, 54, 55, 56, 57, 57, 58, 59, 60, 61, 62, 62, 63, 64, 65, 66, 66, 67,
    68, 69, 70, 70, 71, 72, 73, 74, 74, 75, 76, 77, 78, 78, 79, 80, 81, 81, 82, 83, 84, 85, 85, 87,
    88, 90, 92, 93, 95, 96, 98, 99, 101, 102, 104, 105, 107, 108, 110, 111, 113, 114, 116, 117,
    118, 120, 121, 123, 125, 127, 129, 131, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154,
    156, 158, 161, 164, 166, 169, 172, 174, 177, 180, 182, 185, 187, 190, 192, 195, 199, 202, 205,
    208, 211, 214, 217, 220, 223, 226, 230, 233, 237, 240, 243, 247, 250, 253, 257, 261, 265, 269,
    272, 276, 280, 284, 288, 292, 296, 300, 304, 309, 313, 317, 322, 326, 330, 335, 340, 344, 349,
    354, 359, 364, 369, 374, 379, 384, 389, 395, 400, 406, 411, 417, 423, 429, 435, 441, 447, 454,
    461, 467, 475, 482, 489, 497, 505, 513, 522, 530, 539, 549, 559, 569, 579, 590, 602, 614, 626,
    640, 654, 668, 684, 700, 717, 736, 755, 775, 796, 819, 843, 869, 896, 925, 955, 988, 1022,
    1058, 1098, 1139, 1184, 1232, 1282, 1336,
];

/// AC dequantizer lookup table for 8-bit depth.
pub const AC_QLOOKUP: [i16; QINDEX_RANGE] = [
    4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
    79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138,
    140, 142, 144, 146, 148, 150, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188,
    191, 194, 197, 200, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255, 260,
    265, 270, 275, 280, 285, 290, 295, 300, 305, 311, 317, 323, 329, 335, 341, 347, 353, 359, 366,
    373, 380, 387, 394, 401, 408, 416, 424, 432, 440, 448, 456, 465, 474, 483, 492, 501, 510, 520,
    530, 540, 550, 560, 571, 582, 593, 604, 615, 627, 639, 651, 663, 676, 689, 702, 715, 729, 743,
    757, 771, 786, 801, 816, 832, 848, 864, 881, 898, 915, 933, 951, 969, 988, 1007, 1026, 1046,
    1066, 1087, 1108, 1129, 1151, 1173, 1196, 1219, 1243, 1267, 1292, 1317, 1343, 1369, 1396, 1423,
    1451, 1479, 1508, 1537, 1567, 1597, 1628, 1660, 1692, 1725, 1759, 1793, 1828,
];

/// DC dequantizer lookup table for 10-bit depth.
pub const DC_QLOOKUP_10: [i16; QINDEX_RANGE] = [
    4, 9, 10, 13, 15, 17, 20, 22, 25, 28, 31, 34, 37, 40, 43, 47, 50, 53, 57, 60, 64, 68, 71, 75,
    78, 82, 86, 90, 93, 97, 101, 105, 109, 113, 116, 120, 124, 128, 132, 136, 140, 143, 147, 151,
    155, 159, 163, 166, 170, 174, 178, 182, 185, 189, 193, 197, 200, 204, 208, 212, 215, 219, 223,
    226, 230, 233, 237, 241, 244, 248, 251, 255, 259, 262, 266, 269, 273, 276, 280, 283, 287, 290,
    293, 297, 300, 304, 307, 310, 314, 317, 321, 324, 327, 331, 334, 337, 343, 350, 356, 362, 369,
    375, 381, 387, 394, 400, 406, 412, 418, 424, 430, 436, 442, 448, 454, 460, 466, 472, 478, 484,
    490, 499, 507, 516, 525, 533, 542, 550, 559, 567, 576, 584, 592, 601, 609, 617, 625, 634, 644,
    655, 666, 676, 687, 698, 708, 718, 729, 739, 749, 759, 770, 782, 795, 807, 819, 831, 844, 856,
    868, 880, 891, 906, 920, 933, 947, 961, 975, 988, 1001, 1015, 1030, 1045, 1061, 1076, 1090,
    1105, 1120, 1137, 1153, 1170, 1186, 1202, 1218, 1236, 1253, 1271, 1288, 1306, 1323, 1342, 1361,
    1379, 1398, 1416, 1436, 1456, 1476, 1496, 1516, 1537, 1559, 1580, 1601, 1624, 1647, 1670, 1692,
    1717, 1741, 1766, 1791, 1817, 1844, 1871, 1900, 1929, 1958, 1990, 2021, 2054, 2088, 2123, 2159,
    2197, 2236, 2276, 2319, 2363, 2410, 2458, 2508, 2561, 2616, 2675, 2737, 2802, 2871, 2944, 3020,
    3102, 3188, 3280, 3375, 3478, 3586, 3702, 3823, 3953, 4089, 4236, 4394, 4559, 4737, 4929, 5130,
    5347,
];

/// AC dequantizer lookup table for 10-bit depth.
pub const AC_QLOOKUP_10: [i16; QINDEX_RANGE] = [
    4, 9, 11, 13, 16, 18, 21, 24, 27, 30, 33, 37, 40, 44, 48, 51, 55, 59, 63, 67, 71, 75, 79, 83,
    88, 92, 96, 100, 105, 109, 114, 118, 122, 127, 131, 136, 140, 145, 149, 154, 158, 163, 168,
    172, 177, 181, 186, 190, 195, 199, 204, 208, 213, 217, 222, 226, 231, 235, 240, 244, 249, 253,
    258, 262, 267, 271, 275, 280, 284, 289, 293, 297, 302, 306, 311, 315, 319, 324, 328, 332, 337,
    341, 345, 349, 354, 358, 362, 367, 371, 375, 379, 384, 388, 392, 396, 401, 409, 417, 425, 433,
    441, 449, 458, 466, 474, 482, 490, 498, 506, 514, 523, 531, 539, 547, 555, 563, 571, 579, 588,
    596, 604, 616, 628, 640, 652, 664, 676, 688, 700, 713, 725, 737, 749, 761, 773, 785, 797, 809,
    825, 841, 857, 873, 889, 905, 922, 938, 954, 970, 986, 1002, 1018, 1038, 1058, 1078, 1098,
    1118, 1138, 1158, 1178, 1198, 1218, 1242, 1266, 1290, 1314, 1338, 1362, 1386, 1411, 1435, 1463,
    1491, 1519, 1547, 1575, 1603, 1631, 1663, 1695, 1727, 1759, 1791, 1823, 1859, 1895, 1931, 1967,
    2003, 2039, 2079, 2119, 2159, 2199, 2239, 2283, 2327, 2371, 2415, 2459, 2507, 2555, 2603, 2651,
    2703, 2755, 2807, 2859, 2915, 2971, 3027, 3083, 3143, 3203, 3263, 3327, 3391, 3455, 3523, 3591,
    3659, 3731, 3803, 3876, 3952, 4028, 4104, 4184, 4264, 4348, 4432, 4516, 4604, 4692, 4784, 4876,
    4972, 5068, 5168, 5268, 5372, 5476, 5584, 5692, 5804, 5916, 6032, 6148, 6268, 6388, 6512, 6640,
    6768, 6900, 7036, 7172, 7312,
];

/// DC dequantizer lookup table for 12-bit depth.
pub const DC_QLOOKUP_12: [i16; QINDEX_RANGE] = [
    4, 12, 18, 25, 33, 41, 50, 60, 70, 80, 91, 103, 115, 127, 140, 153, 166, 180, 194, 208, 222,
    237, 251, 266, 281, 296, 312, 327, 343, 358, 374, 390, 405, 421, 437, 453, 469, 484, 500, 516,
    532, 548, 564, 580, 596, 611, 627, 643, 659, 674, 690, 706, 721, 737, 752, 768, 783, 798, 814,
    829, 844, 859, 874, 889, 904, 919, 934, 949, 964, 978, 993, 1008, 1022, 1037, 1051, 1065, 1080,
    1094, 1108, 1122, 1136, 1151, 1165, 1179, 1192, 1206, 1220, 1234, 1248, 1261, 1275, 1288, 1302,
    1315, 1329, 1342, 1368, 1393, 1419, 1444, 1469, 1494, 1519, 1544, 1569, 1594, 1618, 1643, 1668,
    1692, 1717, 1741, 1765, 1789, 1814, 1838, 1862, 1885, 1909, 1933, 1957, 1992, 2027, 2061, 2096,
    2130, 2165, 2199, 2233, 2267, 2300, 2334, 2367, 2400, 2434, 2467, 2499, 2532, 2575, 2618, 2661,
    2704, 2746, 2788, 2830, 2872, 2913, 2954, 2995, 3036, 3076, 3127, 3177, 3226, 3275, 3324, 3373,
    3421, 3469, 3517, 3565, 3621, 3677, 3733, 3788, 3843, 3897, 3951, 4005, 4058, 4119, 4181, 4241,
    4301, 4361, 4420, 4479, 4546, 4612, 4677, 4742, 4807, 4871, 4942, 5013, 5083, 5153, 5222, 5291,
    5367, 5442, 5517, 5591, 5665, 5745, 5825, 5905, 5984, 6063, 6149, 6234, 6319, 6404, 6495, 6587,
    6678, 6769, 6867, 6966, 7064, 7163, 7269, 7376, 7483, 7599, 7715, 7832, 7958, 8085, 8214, 8352,
    8492, 8635, 8788, 8945, 9104, 9275, 9450, 9639, 9832, 10031, 10245, 10465, 10702, 10946, 11210,
    11482, 11776, 12081, 12409, 12750, 13118, 13501, 13913, 14343, 14807, 15290, 15812, 16356,
    16943, 17575, 18237, 18949, 19718, 20521, 21387,
];

/// AC dequantizer lookup table for 12-bit depth.
pub const AC_QLOOKUP_12: [i16; QINDEX_RANGE] = [
    4, 13, 19, 27, 35, 44, 54, 64, 75, 87, 99, 112, 126, 139, 154, 168, 183, 199, 214, 230, 247,
    263, 280, 297, 314, 331, 349, 366, 384, 402, 420, 438, 456, 475, 493, 511, 530, 548, 567, 586,
    604, 623, 642, 660, 679, 698, 716, 735, 753, 772, 791, 809, 828, 846, 865, 884, 902, 920, 939,
    957, 976, 994, 1012, 1030, 1049, 1067, 1085, 1103, 1121, 1139, 1157, 1175, 1193, 1211, 1229,
    1246, 1264, 1282, 1299, 1317, 1335, 1352, 1370, 1387, 1405, 1422, 1440, 1457, 1474, 1491, 1509,
    1526, 1543, 1560, 1577, 1595, 1627, 1660, 1693, 1725, 1758, 1791, 1824, 1856, 1889, 1922, 1954,
    1987, 2020, 2052, 2085, 2118, 2150, 2183, 2216, 2248, 2281, 2313, 2346, 2378, 2411, 2459, 2508,
    2556, 2605, 2653, 2701, 2750, 2798, 2847, 2895, 2943, 2992, 3040, 3088, 3137, 3185, 3234, 3298,
    3362, 3426, 3491, 3555, 3619, 3684, 3748, 3812, 3876, 3941, 4005, 4069, 4149, 4230, 4310, 4390,
    4470, 4550, 4631, 4711, 4791, 4871, 4967, 5064, 5160, 5256, 5352, 5448, 5544, 5641, 5737, 5849,
    5961, 6073, 6185, 6297, 6410, 6522, 6650, 6778, 6906, 7034, 7162, 7290, 7435, 7579, 7723, 7867,
    8011, 8155, 8315, 8475, 8635, 8795, 8956, 9132, 9308, 9484, 9660, 9836, 10028, 10220, 10412,
    10604, 10812, 11020, 11228, 11437, 11661, 11885, 12109, 12333, 12573, 12813, 13053, 13309,
    13565, 13821, 14093, 14365, 14637, 14925, 15213, 15502, 15806, 16110, 16414, 16734, 17054,
    17390, 17726, 18062, 18414, 18766, 19134, 19502, 19886, 20270, 20670, 21070, 21486, 21902,
    22334, 22766, 23214, 23662, 24126, 24590, 25070, 25551, 26047, 26559, 27071, 27599, 28143,
    28687, 29247,
];

// =============================================================================
// Structures
// =============================================================================

/// Quantization parameters as parsed from the frame header.
#[derive(Clone, Debug, Default)]
pub struct QuantizationParams {
    /// Base quantizer index (0-255).
    pub base_q_idx: u8,
    /// Delta Q for Y DC coefficients.
    pub delta_q_y_dc: i8,
    /// Delta Q for U DC coefficients.
    pub delta_q_u_dc: i8,
    /// Delta Q for U AC coefficients.
    pub delta_q_u_ac: i8,
    /// Delta Q for V DC coefficients.
    pub delta_q_v_dc: i8,
    /// Delta Q for V AC coefficients.
    pub delta_q_v_ac: i8,
    /// Use quantization matrices.
    pub using_qmatrix: bool,
    /// QM level for Y plane.
    pub qm_y: u8,
    /// QM level for U plane.
    pub qm_u: u8,
    /// QM level for V plane.
    pub qm_v: u8,
    /// Delta Q present in block level.
    pub delta_q_present: bool,
    /// Delta Q resolution (log2).
    pub delta_q_res: u8,
}

impl QuantizationParams {
    /// Create new quantization parameters with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse quantization parameters from the bitstream.
    ///
    /// # Errors
    ///
    /// Returns error if the bitstream is malformed.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn parse(reader: &mut BitReader<'_>, seq: &SequenceHeader) -> CodecResult<Self> {
        let mut qp = Self::new();

        // Base Q index
        qp.base_q_idx = reader.read_bits(8).map_err(CodecError::Core)? as u8;

        // Y DC delta
        qp.delta_q_y_dc = Self::read_delta_q(reader)?;

        // UV deltas
        let num_planes = if seq.color_config.mono_chrome { 1 } else { 3 };

        if num_planes > 1 {
            let diff_uv_delta = if seq.color_config.separate_uv_delta_q {
                reader.read_bit().map_err(CodecError::Core)? != 0
            } else {
                false
            };

            qp.delta_q_u_dc = Self::read_delta_q(reader)?;
            qp.delta_q_u_ac = Self::read_delta_q(reader)?;

            if diff_uv_delta {
                qp.delta_q_v_dc = Self::read_delta_q(reader)?;
                qp.delta_q_v_ac = Self::read_delta_q(reader)?;
            } else {
                qp.delta_q_v_dc = qp.delta_q_u_dc;
                qp.delta_q_v_ac = qp.delta_q_u_ac;
            }
        }

        // Quantization matrices
        qp.using_qmatrix = reader.read_bit().map_err(CodecError::Core)? != 0;

        if qp.using_qmatrix {
            qp.qm_y = reader.read_bits(4).map_err(CodecError::Core)? as u8;
            qp.qm_u = reader.read_bits(4).map_err(CodecError::Core)? as u8;

            if seq.color_config.separate_uv_delta_q {
                qp.qm_v = reader.read_bits(4).map_err(CodecError::Core)? as u8;
            } else {
                qp.qm_v = qp.qm_u;
            }
        }

        Ok(qp)
    }

    /// Read a delta Q value from the bitstream.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn read_delta_q(reader: &mut BitReader<'_>) -> CodecResult<i8> {
        let delta_coded = reader.read_bit().map_err(CodecError::Core)? != 0;

        if delta_coded {
            let abs_value = reader.read_bits(DELTA_Q_BITS).map_err(CodecError::Core)? as i8;
            let sign = reader.read_bit().map_err(CodecError::Core)? != 0;
            if sign {
                Ok(-abs_value)
            } else {
                Ok(abs_value)
            }
        } else {
            Ok(0)
        }
    }

    /// Get the effective Q index for Y DC.
    #[must_use]
    pub fn y_dc_qindex(&self) -> u8 {
        self.clamp_qindex(i16::from(self.base_q_idx) + i16::from(self.delta_q_y_dc))
    }

    /// Get the effective Q index for Y AC (same as base).
    #[must_use]
    pub const fn y_ac_qindex(&self) -> u8 {
        self.base_q_idx
    }

    /// Get the effective Q index for U DC.
    #[must_use]
    pub fn u_dc_qindex(&self) -> u8 {
        self.clamp_qindex(i16::from(self.base_q_idx) + i16::from(self.delta_q_u_dc))
    }

    /// Get the effective Q index for U AC.
    #[must_use]
    pub fn u_ac_qindex(&self) -> u8 {
        self.clamp_qindex(i16::from(self.base_q_idx) + i16::from(self.delta_q_u_ac))
    }

    /// Get the effective Q index for V DC.
    #[must_use]
    pub fn v_dc_qindex(&self) -> u8 {
        self.clamp_qindex(i16::from(self.base_q_idx) + i16::from(self.delta_q_v_dc))
    }

    /// Get the effective Q index for V AC.
    #[must_use]
    pub fn v_ac_qindex(&self) -> u8 {
        self.clamp_qindex(i16::from(self.base_q_idx) + i16::from(self.delta_q_v_ac))
    }

    /// Clamp a Q index to valid range.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn clamp_qindex(&self, q: i16) -> u8 {
        q.clamp(i16::from(MIN_Q_INDEX), i16::from(MAX_Q_INDEX)) as u8
    }

    /// Get the DC dequantizer for Y plane.
    #[must_use]
    pub fn get_y_dc_dequant(&self, bit_depth: u8) -> i16 {
        let qindex = self.y_dc_qindex();
        get_dc_dequant(qindex, bit_depth)
    }

    /// Get the AC dequantizer for Y plane.
    #[must_use]
    pub fn get_y_ac_dequant(&self, bit_depth: u8) -> i16 {
        let qindex = self.y_ac_qindex();
        get_ac_dequant(qindex, bit_depth)
    }

    /// Get the DC dequantizer for U plane.
    #[must_use]
    pub fn get_u_dc_dequant(&self, bit_depth: u8) -> i16 {
        let qindex = self.u_dc_qindex();
        get_dc_dequant(qindex, bit_depth)
    }

    /// Get the AC dequantizer for U plane.
    #[must_use]
    pub fn get_u_ac_dequant(&self, bit_depth: u8) -> i16 {
        let qindex = self.u_ac_qindex();
        get_ac_dequant(qindex, bit_depth)
    }

    /// Get the DC dequantizer for V plane.
    #[must_use]
    pub fn get_v_dc_dequant(&self, bit_depth: u8) -> i16 {
        let qindex = self.v_dc_qindex();
        get_dc_dequant(qindex, bit_depth)
    }

    /// Get the AC dequantizer for V plane.
    #[must_use]
    pub fn get_v_ac_dequant(&self, bit_depth: u8) -> i16 {
        let qindex = self.v_ac_qindex();
        get_ac_dequant(qindex, bit_depth)
    }

    /// Check if lossless mode is enabled.
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        self.base_q_idx == 0
            && self.delta_q_y_dc == 0
            && self.delta_q_u_dc == 0
            && self.delta_q_u_ac == 0
            && self.delta_q_v_dc == 0
            && self.delta_q_v_ac == 0
    }

    /// Check if any UV delta is non-zero.
    #[must_use]
    pub fn has_uv_delta(&self) -> bool {
        self.delta_q_u_dc != 0
            || self.delta_q_u_ac != 0
            || self.delta_q_v_dc != 0
            || self.delta_q_v_ac != 0
    }

    /// Get the QM level for a plane.
    #[must_use]
    pub const fn get_qm_level(&self, plane: usize) -> u8 {
        match plane {
            0 => self.qm_y,
            1 => self.qm_u,
            _ => self.qm_v,
        }
    }

    /// Get DC quantizer for a plane (generic method).
    #[must_use]
    pub fn get_dc_quant(&self, plane: usize, bit_depth: u8) -> i16 {
        match plane {
            0 => self.get_y_dc_dequant(bit_depth),
            1 => self.get_u_dc_dequant(bit_depth),
            2 => self.get_v_dc_dequant(bit_depth),
            _ => self.get_y_dc_dequant(bit_depth),
        }
    }

    /// Get AC quantizer for a plane (generic method).
    #[must_use]
    pub fn get_ac_quant(&self, plane: usize, bit_depth: u8) -> i16 {
        match plane {
            0 => self.get_y_ac_dequant(bit_depth),
            1 => self.get_u_ac_dequant(bit_depth),
            2 => self.get_v_ac_dequant(bit_depth),
            _ => self.get_y_ac_dequant(bit_depth),
        }
    }
}

/// Per-block delta Q state.
#[derive(Clone, Debug, Default)]
pub struct DeltaQState {
    /// Current delta Q value.
    pub delta_q: i16,
    /// Resolution (1 << delta_q_res).
    pub resolution: u8,
}

impl DeltaQState {
    /// Create a new delta Q state.
    #[must_use]
    pub const fn new(resolution: u8) -> Self {
        Self {
            delta_q: 0,
            resolution,
        }
    }

    /// Apply delta Q to base Q index.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn apply(&self, base_q_idx: u8) -> u8 {
        let q = i16::from(base_q_idx) + self.delta_q;
        q.clamp(0, i16::from(MAX_Q_INDEX)) as u8
    }

    /// Update delta Q with a delta value.
    #[allow(clippy::cast_possible_wrap)]
    pub fn update(&mut self, delta: i16) {
        self.delta_q += delta * ((1i16) << self.resolution);
        self.delta_q = self
            .delta_q
            .clamp(i16::from(MIN_DELTA_Q), i16::from(MAX_DELTA_Q));
    }

    /// Reset delta Q to zero.
    pub fn reset(&mut self) {
        self.delta_q = 0;
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Get DC dequantizer value for a given Q index and bit depth.
#[must_use]
pub fn get_dc_dequant(qindex: u8, bit_depth: u8) -> i16 {
    let table = match bit_depth {
        10 => &DC_QLOOKUP_10,
        12 => &DC_QLOOKUP_12,
        _ => &DC_QLOOKUP,
    };
    table[qindex as usize]
}

/// Get AC dequantizer value for a given Q index and bit depth.
#[must_use]
pub fn get_ac_dequant(qindex: u8, bit_depth: u8) -> i16 {
    let table = match bit_depth {
        10 => &AC_QLOOKUP_10,
        12 => &AC_QLOOKUP_12,
        _ => &AC_QLOOKUP,
    };
    table[qindex as usize]
}

/// Convert Q index to quantizer value (for display/logging).
#[must_use]
pub fn qindex_to_qp(qindex: u8) -> f32 {
    // Approximate conversion based on AV1 rate control models
    let q = f32::from(qindex);
    if q < 1.0 {
        0.0
    } else {
        (q.log2() * 6.0) + 4.0
    }
}

/// Convert QP value back to Q index (approximate).
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn qp_to_qindex(qp: f32) -> u8 {
    if qp < 4.0 {
        0
    } else {
        let q = 2.0f32.powf((qp - 4.0) / 6.0);
        (q.round() as u8).min(MAX_Q_INDEX)
    }
}

// =============================================================================
// Adaptive Quantization Matrix Selection by Content Type
// =============================================================================

/// Content type classification for adaptive QM level selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QmContentType {
    /// Flat/smooth regions — prefer low QM level.
    Flat,
    /// Natural texture — prefer medium QM level.
    Texture,
    /// High-frequency detail — prefer high QM level.
    Detail,
    /// Screen content — prefer very low QM level.
    Screen,
}

/// Result of adaptive QM level selection.
#[derive(Clone, Debug)]
pub struct AdaptiveQmSelection {
    /// Luma QM level 0-15, or `None` to disable.
    pub qm_y: Option<u8>,
    /// Cb QM level, or `None`.
    pub qm_u: Option<u8>,
    /// Cr QM level, or `None`.
    pub qm_v: Option<u8>,
    /// Content type used for selection.
    pub content_type: QmContentType,
    /// Base Q index used.
    pub base_q_idx: u8,
}

/// Select adaptive QM levels based on content type and base Q index.
///
/// QM is disabled below Q=64 (high quality). Above that threshold the QM
/// level scales with Q and is adjusted by content type.
#[must_use]
pub fn select_adaptive_qm(content_type: QmContentType, base_q_idx: u8) -> AdaptiveQmSelection {
    if base_q_idx < 64 {
        return AdaptiveQmSelection {
            qm_y: None,
            qm_u: None,
            qm_v: None,
            content_type,
            base_q_idx,
        };
    }
    let base_qm: u8 = match base_q_idx {
        64..=127 => 4 + (u16::from(base_q_idx - 64) * 4 / 63) as u8,
        128..=191 => 8 + (u16::from(base_q_idx - 128) * 4 / 63) as u8,
        _ => (12 + (u16::from(base_q_idx - 192) * 3 / 63)).min(15) as u8,
    };
    let (y_adj, uv_adj): (i8, i8) = match content_type {
        QmContentType::Flat => (-2, -1),
        QmContentType::Texture => (0, 0),
        QmContentType::Detail => (2, 1),
        QmContentType::Screen => (-3, -2),
    };
    let c = |b: u8, a: i8| Some((i16::from(b) + i16::from(a)).clamp(0, 15) as u8);
    AdaptiveQmSelection {
        qm_y: c(base_qm, y_adj),
        qm_u: c(base_qm, uv_adj),
        qm_v: c(base_qm, uv_adj),
        content_type,
        base_q_idx,
    }
}

/// Apply an `AdaptiveQmSelection` to `QuantizationParams`.
pub fn apply_adaptive_qm(qp: &mut QuantizationParams, sel: &AdaptiveQmSelection) {
    match (sel.qm_y, sel.qm_u, sel.qm_v) {
        (None, None, None) => {
            qp.using_qmatrix = false;
            qp.qm_y = 0;
            qp.qm_u = 0;
            qp.qm_v = 0;
        }
        _ => {
            qp.using_qmatrix = true;
            qp.qm_y = sel.qm_y.unwrap_or(0);
            qp.qm_u = sel.qm_u.unwrap_or(0);
            qp.qm_v = sel.qm_v.unwrap_or(0);
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_params_default() {
        let qp = QuantizationParams::default();
        assert_eq!(qp.base_q_idx, 0);
        assert_eq!(qp.delta_q_y_dc, 0);
        assert!(!qp.using_qmatrix);
    }

    #[test]
    fn test_qindex_accessors() {
        let mut qp = QuantizationParams::default();
        qp.base_q_idx = 100;
        qp.delta_q_y_dc = -10;
        qp.delta_q_u_dc = 5;
        qp.delta_q_u_ac = -5;
        qp.delta_q_v_dc = 10;
        qp.delta_q_v_ac = -10;

        assert_eq!(qp.y_dc_qindex(), 90);
        assert_eq!(qp.y_ac_qindex(), 100);
        assert_eq!(qp.u_dc_qindex(), 105);
        assert_eq!(qp.u_ac_qindex(), 95);
        assert_eq!(qp.v_dc_qindex(), 110);
        assert_eq!(qp.v_ac_qindex(), 90);
    }

    #[test]
    fn test_qindex_clamping() {
        let mut qp = QuantizationParams::default();
        qp.base_q_idx = 250;
        qp.delta_q_y_dc = 20;

        // Should clamp to 255
        assert_eq!(qp.y_dc_qindex(), 255);

        qp.base_q_idx = 10;
        qp.delta_q_y_dc = -20;

        // Should clamp to 0
        assert_eq!(qp.y_dc_qindex(), 0);
    }

    #[test]
    fn test_is_lossless() {
        let mut qp = QuantizationParams::default();
        assert!(qp.is_lossless());

        qp.base_q_idx = 1;
        assert!(!qp.is_lossless());

        qp.base_q_idx = 0;
        qp.delta_q_y_dc = 1;
        assert!(!qp.is_lossless());
    }

    #[test]
    fn test_has_uv_delta() {
        let mut qp = QuantizationParams::default();
        assert!(!qp.has_uv_delta());

        qp.delta_q_u_dc = 5;
        assert!(qp.has_uv_delta());

        qp.delta_q_u_dc = 0;
        qp.delta_q_v_ac = -3;
        assert!(qp.has_uv_delta());
    }

    #[test]
    fn test_get_qm_level() {
        let mut qp = QuantizationParams::default();
        qp.qm_y = 5;
        qp.qm_u = 7;
        qp.qm_v = 9;

        assert_eq!(qp.get_qm_level(0), 5);
        assert_eq!(qp.get_qm_level(1), 7);
        assert_eq!(qp.get_qm_level(2), 9);
    }

    #[test]
    fn test_dc_dequant_8bit() {
        // Check some known values from the lookup table
        assert_eq!(get_dc_dequant(0, 8), 4);
        assert_eq!(get_dc_dequant(255, 8), 1336);
    }

    #[test]
    fn test_ac_dequant_8bit() {
        assert_eq!(get_ac_dequant(0, 8), 4);
        assert_eq!(get_ac_dequant(255, 8), 1828);
    }

    #[test]
    fn test_dc_dequant_10bit() {
        assert_eq!(get_dc_dequant(0, 10), 4);
        assert_eq!(get_dc_dequant(255, 10), 5347);
    }

    #[test]
    fn test_ac_dequant_10bit() {
        assert_eq!(get_ac_dequant(0, 10), 4);
        assert_eq!(get_ac_dequant(255, 10), 7312);
    }

    #[test]
    fn test_dc_dequant_12bit() {
        assert_eq!(get_dc_dequant(0, 12), 4);
        assert_eq!(get_dc_dequant(255, 12), 21387);
    }

    #[test]
    fn test_ac_dequant_12bit() {
        assert_eq!(get_ac_dequant(0, 12), 4);
        assert_eq!(get_ac_dequant(255, 12), 29247);
    }

    #[test]
    fn test_dequant_methods() {
        let mut qp = QuantizationParams::default();
        qp.base_q_idx = 128;

        let y_dc = qp.get_y_dc_dequant(8);
        let y_ac = qp.get_y_ac_dequant(8);

        assert!(y_dc > 0);
        assert!(y_ac > 0);

        // DC values from the table at index 128
        assert_eq!(y_dc, DC_QLOOKUP[128]);
        assert_eq!(y_ac, AC_QLOOKUP[128]);
    }

    #[test]
    fn test_delta_q_state() {
        let mut state = DeltaQState::new(2);
        assert_eq!(state.delta_q, 0);

        state.update(5);
        assert_eq!(state.delta_q, 20); // 5 * (1 << 2)

        let q = state.apply(100);
        assert_eq!(q, 120);

        state.reset();
        assert_eq!(state.delta_q, 0);
    }

    #[test]
    fn test_delta_q_state_clamping() {
        let mut state = DeltaQState::new(0);
        state.delta_q = 50;

        let q = state.apply(220);
        assert_eq!(q, 255); // Clamped
    }

    #[test]
    fn test_qindex_to_qp_conversion() {
        // Q index 0 should give QP close to 0
        let qp_0 = qindex_to_qp(0);
        assert!(qp_0 < 1.0);

        // Higher Q index should give higher QP
        let qp_128 = qindex_to_qp(128);
        let qp_64 = qindex_to_qp(64);
        assert!(qp_128 > qp_64);
    }

    #[test]
    fn test_qp_to_qindex_conversion() {
        // Low QP should give low Q index
        let q_low = qp_to_qindex(4.0);
        assert!(q_low <= 10);

        // High QP should give higher Q index
        let q_high = qp_to_qindex(50.0);
        assert!(q_high > q_low);
    }

    #[test]
    fn test_constants() {
        assert_eq!(MAX_Q_INDEX, 255);
        assert_eq!(MIN_Q_INDEX, 0);
        assert_eq!(QINDEX_RANGE, 256);
        assert_eq!(NUM_QM_LEVELS, 16);
    }

    #[test]
    fn test_lookup_table_lengths() {
        assert_eq!(DC_QLOOKUP.len(), QINDEX_RANGE);
        assert_eq!(AC_QLOOKUP.len(), QINDEX_RANGE);
        assert_eq!(DC_QLOOKUP_10.len(), QINDEX_RANGE);
        assert_eq!(AC_QLOOKUP_10.len(), QINDEX_RANGE);
        assert_eq!(DC_QLOOKUP_12.len(), QINDEX_RANGE);
        assert_eq!(AC_QLOOKUP_12.len(), QINDEX_RANGE);
    }

    #[test]
    fn test_lookup_table_monotonic() {
        // Verify tables are monotonically increasing
        for i in 1..QINDEX_RANGE {
            assert!(DC_QLOOKUP[i] >= DC_QLOOKUP[i - 1]);
            assert!(AC_QLOOKUP[i] >= AC_QLOOKUP[i - 1]);
        }
    }

    // =========================================================================
    // Adaptive quantization matrix selection tests (Task 4)
    // =========================================================================

    #[test]
    fn test_adaptive_qm_disabled_below_q64() {
        for q in [0u8, 32, 63] {
            let sel = select_adaptive_qm(QmContentType::Texture, q);
            assert!(sel.qm_y.is_none(), "QM must be disabled for q_idx={q}");
            assert!(sel.qm_u.is_none());
            assert!(sel.qm_v.is_none());
        }
    }

    #[test]
    fn test_adaptive_qm_enabled_above_q64() {
        for q in [64u8, 128, 191, 255] {
            let sel = select_adaptive_qm(QmContentType::Texture, q);
            assert!(sel.qm_y.is_some(), "QM must be enabled for q_idx={q}");
        }
    }

    #[test]
    fn test_adaptive_qm_flat_uses_lower_level_than_detail() {
        let q = 128u8;
        let flat = select_adaptive_qm(QmContentType::Flat, q);
        let detail = select_adaptive_qm(QmContentType::Detail, q);
        let flat_y = flat.qm_y.unwrap_or(0);
        let detail_y = detail.qm_y.unwrap_or(0);
        assert!(
            flat_y < detail_y || flat_y == detail_y,
            "Flat should use QM level <= Detail: flat={flat_y}, detail={detail_y}"
        );
    }

    #[test]
    fn test_adaptive_qm_screen_content_low_level() {
        // Screen content should prefer the lowest QM level (sharpest matrix)
        let q = 128u8;
        let screen = select_adaptive_qm(QmContentType::Screen, q);
        let texture = select_adaptive_qm(QmContentType::Texture, q);
        let screen_y = screen.qm_y.unwrap_or(0);
        let texture_y = texture.qm_y.unwrap_or(0);
        assert!(
            screen_y <= texture_y,
            "Screen content must use QM level <= Texture: screen={screen_y}, texture={texture_y}"
        );
    }

    #[test]
    fn test_adaptive_qm_levels_in_valid_range() {
        for q in [64u8, 96, 128, 160, 192, 220, 255] {
            for ct in [
                QmContentType::Flat,
                QmContentType::Texture,
                QmContentType::Detail,
                QmContentType::Screen,
            ] {
                let sel = select_adaptive_qm(ct, q);
                if let Some(y) = sel.qm_y {
                    assert!(y <= 15, "qm_y={y} must be in [0,15]");
                }
                if let Some(u) = sel.qm_u {
                    assert!(u <= 15, "qm_u={u} must be in [0,15]");
                }
                if let Some(v) = sel.qm_v {
                    assert!(v <= 15, "qm_v={v} must be in [0,15]");
                }
            }
        }
    }

    #[test]
    fn test_apply_adaptive_qm_enables_matrix() {
        let mut qp = QuantizationParams::default();
        let sel = select_adaptive_qm(QmContentType::Texture, 128);
        apply_adaptive_qm(&mut qp, &sel);
        assert!(
            qp.using_qmatrix,
            "apply_adaptive_qm must set using_qmatrix=true for q>=64"
        );
    }

    #[test]
    fn test_apply_adaptive_qm_disables_for_low_q() {
        let mut qp = QuantizationParams::default();
        let sel = select_adaptive_qm(QmContentType::Flat, 32);
        apply_adaptive_qm(&mut qp, &sel);
        assert!(
            !qp.using_qmatrix,
            "apply_adaptive_qm must set using_qmatrix=false for q<64"
        );
    }

    #[test]
    fn test_adaptive_qm_content_type_preserved() {
        let ct = QmContentType::Screen;
        let sel = select_adaptive_qm(ct, 200);
        assert_eq!(
            sel.content_type, ct,
            "Content type must be preserved in selection result"
        );
        assert_eq!(sel.base_q_idx, 200);
    }

    #[test]
    fn test_adaptive_qm_monotone_with_q() {
        // Higher Q index should produce equal or higher QM level
        let ct = QmContentType::Texture;
        let low_q = select_adaptive_qm(ct, 80);
        let high_q = select_adaptive_qm(ct, 240);
        let low_y = low_q.qm_y.unwrap_or(0);
        let high_y = high_q.qm_y.unwrap_or(0);
        assert!(
            high_y >= low_y,
            "Higher Q should give >= QM level: q=80 → {low_y}, q=240 → {high_y}"
        );
    }

    // =========================================================================
    // Additional adaptive QM tests
    // =========================================================================

    #[test]
    fn test_adaptive_qm_boundary_q64_enabled() {
        let sel = select_adaptive_qm(QmContentType::Texture, 64);
        assert!(sel.qm_y.is_some(), "QM must be enabled at exactly q=64");
    }

    #[test]
    fn test_adaptive_qm_boundary_q63_disabled() {
        let sel = select_adaptive_qm(QmContentType::Texture, 63);
        assert!(sel.qm_y.is_none(), "QM must be disabled at q=63");
    }

    #[test]
    fn test_adaptive_qm_uv_levels_consistent() {
        for q in [64u8, 128, 192, 255] {
            for ct in [
                QmContentType::Flat,
                QmContentType::Texture,
                QmContentType::Detail,
                QmContentType::Screen,
            ] {
                let sel = select_adaptive_qm(ct, q);
                assert_eq!(
                    sel.qm_u, sel.qm_v,
                    "qm_u and qm_v must be equal for q={q} ct={ct:?}"
                );
            }
        }
    }

    #[test]
    fn test_adaptive_qm_detail_uses_highest_level() {
        let q = 200u8;
        let detail = select_adaptive_qm(QmContentType::Detail, q);
        let flat = select_adaptive_qm(QmContentType::Flat, q);
        let screen = select_adaptive_qm(QmContentType::Screen, q);
        let detail_y = detail.qm_y.unwrap_or(0);
        let flat_y = flat.qm_y.unwrap_or(0);
        let screen_y = screen.qm_y.unwrap_or(0);
        assert!(
            detail_y >= flat_y,
            "Detail must use >= level vs Flat: detail={detail_y}, flat={flat_y}"
        );
        assert!(
            detail_y >= screen_y,
            "Detail must use >= level vs Screen: detail={detail_y}, screen={screen_y}"
        );
    }

    #[test]
    fn test_apply_adaptive_qm_sets_levels() {
        let mut qp = QuantizationParams::default();
        let sel = select_adaptive_qm(QmContentType::Detail, 192);
        apply_adaptive_qm(&mut qp, &sel);
        assert!(qp.using_qmatrix);
        assert_eq!(qp.qm_y, sel.qm_y.unwrap_or(0));
        assert_eq!(qp.qm_u, sel.qm_u.unwrap_or(0));
        assert_eq!(qp.qm_v, sel.qm_v.unwrap_or(0));
    }

    #[test]
    fn test_adaptive_qm_q_idx_preserved_in_result() {
        for q in [64u8, 100, 200, 255] {
            let sel = select_adaptive_qm(QmContentType::Texture, q);
            assert_eq!(sel.base_q_idx, q, "base_q_idx must be preserved");
        }
    }

    #[test]
    fn test_delta_q_state_resolution_2() {
        let mut state = DeltaQState::new(2);
        state.update(3);
        assert_eq!(state.delta_q, 12);
    }

    #[test]
    fn test_delta_q_state_clamped_min() {
        let mut state = DeltaQState::new(0);
        for _ in 0..100 {
            state.update(-1);
        }
        assert_eq!(state.delta_q, i16::from(MIN_DELTA_Q));
    }

    #[test]
    fn test_qindex_to_qp_monotonic() {
        let mut prev = qindex_to_qp(1);
        for q in 2u8..=255 {
            let curr = qindex_to_qp(q);
            assert!(
                curr >= prev,
                "qindex_to_qp not monotone at q={q}: {prev} > {curr}"
            );
            prev = curr;
        }
    }

    #[test]
    fn test_get_dc_ac_dequant_12bit() {
        let dc = get_dc_dequant(128, 12);
        let ac = get_ac_dequant(128, 12);
        assert!(dc > 0);
        assert!(ac > 0);
        assert!(
            ac >= dc,
            "AC dequant should be >= DC dequant at q=128: ac={ac}, dc={dc}"
        );
    }
}
