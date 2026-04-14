//! Additional calibration utilities implementing the core calibration APIs.
//!
//! This module provides:
//! - `IccProfileBuilder` — minimal ICC v4 profile byte-level serializer
//! - `CalibrationLut` — 1-D correction LUT from measurement/target sample pairs
//! - `ColorThermometer` — CIE xy → CCT via Robertson's method
//! - `GammaMeasurer` — display gamma from log-log slope regression
//! - `ColorCheckerTarget::passport` — 24 Lab reference values (Passport chart)
//! - `LensCalibration` / `LensCalibrator` — simplified DLT-based intrinsics
//! - `WhiteBalanceCalibrator` — gain set from a gray patch
//! - `BradfordAdapt` — Bradford chromatic adaptation free function

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// IccProfileBuilder
// ---------------------------------------------------------------------------

/// RGB primary chromaticities or matrix row set.
#[derive(Debug, Clone, Copy)]
pub enum Primaries {
    /// sRGB / Rec.709 primaries.
    Srgb,
    /// Display P3 primaries.
    DisplayP3,
    /// Rec.2020 primaries.
    Rec2020,
    /// Adobe RGB (1998) primaries.
    AdobeRgb,
    /// Custom XYZ primary matrix rows [rXYZ, gXYZ, bXYZ].
    Custom([[f32; 3]; 3]),
}

impl Primaries {
    /// XYZ columns for each primary (columns of the RGB-to-XYZ matrix).
    fn xyz_matrix(self) -> [[f32; 3]; 3] {
        match self {
            Self::Srgb => [
                [0.4360_747, 0.3850_649, 0.1430_804],
                [0.2225_045, 0.7168_786, 0.0606_169],
                [0.0139_322, 0.0971_045, 0.7141_733],
            ],
            Self::DisplayP3 => [
                [0.5150_974, 0.2920_941, 0.1571_84],
                [0.2411_865, 0.6922_444, 0.0665_692],
                [0.0, 0.0452_435, 0.7672_016],
            ],
            Self::Rec2020 => [
                [0.6369_580, 0.1446_169, 0.1688_810],
                [0.2627_002, 0.6779_981, 0.0593_017],
                [0.0, 0.0280_727, 0.1093_561],
            ],
            Self::AdobeRgb => [
                [0.6097_559, 0.2052_401, 0.1492_240],
                [0.3111_242, 0.6256_560, 0.0632_198],
                [0.0195_178, 0.0609_741, 0.7448_580],
            ],
            Self::Custom(m) => m,
        }
    }
}

/// Reference white-point illuminant for ICC profiles.
#[derive(Debug, Clone, Copy)]
pub enum IccIlluminant {
    /// D50 (0.9642, 1.0, 0.8251).
    D50,
    /// D65 (0.9505, 1.0, 1.0890).
    D65,
}

impl IccIlluminant {
    fn xyz(self) -> [f32; 3] {
        match self {
            Self::D50 => [0.9642, 1.0, 0.8251],
            Self::D65 => [0.9505, 1.0, 1.0890],
        }
    }
}

/// Builder for a minimal ICC v4 RGB display profile.
///
/// Generates a byte vector containing: 128-byte header + tag table +
/// `chad`, `rXYZ`, `gXYZ`, `bXYZ`, `rTRC`, `gTRC`, `bTRC` tags.
#[derive(Debug, Clone)]
pub struct IccProfileBuilder {
    primaries: Primaries,
    gamma: f32,
    illuminant: IccIlluminant,
    description: String,
}

impl IccProfileBuilder {
    /// Create a new builder with the given primaries, gamma, and illuminant.
    #[must_use]
    pub fn new(primaries: Primaries, gamma: f32, illuminant: IccIlluminant) -> Self {
        Self {
            primaries,
            gamma,
            illuminant,
            description: "OxiMedia ICC Profile".to_string(),
        }
    }

    /// Override the profile description string.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Build the minimal ICC v4 profile as a byte vector.
    ///
    /// Layout:
    /// - 128-byte profile header
    /// - 4-byte tag count + tag entries (12 bytes each)
    /// - `chad` XYZ tag (12 + 36 bytes)
    /// - rXYZ / gXYZ / bXYZ (12 + 20 bytes each)
    /// - rTRC / gTRC / bTRC (12 + 14 bytes each, parametric power-law curve)
    #[must_use]
    pub fn build(&self) -> Vec<u8> {
        // We'll build tag data first, then assemble with correct offsets.
        let m = self.primaries.xyz_matrix();
        let wp = self.illuminant.xyz();

        // Bradford D50 adaptation (chad tag): adapt wp → D50
        // For simplicity, compute chad as identity * (D50_wp / wp)
        let d50 = [0.9642_f32, 1.0_f32, 0.8251_f32];
        let chad: [f32; 9] = [
            d50[0] / wp[0].max(1e-10), 0.0, 0.0,
            0.0, d50[1] / wp[1].max(1e-10), 0.0,
            0.0, 0.0, d50[2] / wp[2].max(1e-10),
        ];

        // Helper: encode XYZ tag (tag type 'XYZ ', 20 bytes total)
        let encode_xyz_tag = |x: f32, y: f32, z: f32| -> Vec<u8> {
            let mut v = Vec::with_capacity(20);
            v.extend_from_slice(b"XYZ "); // type sig
            v.extend_from_slice(&[0u8; 4]); // reserved
            v.extend_from_slice(&encode_s15fixed16(x));
            v.extend_from_slice(&encode_s15fixed16(y));
            v.extend_from_slice(&encode_s15fixed16(z));
            v
        };

        // Helper: encode parametric curve tag (gamma, type 'para')
        // para type 0: y = x^gamma (12 bytes header + 4 bytes parameter = 16 bytes)
        let encode_trc_tag = |g: f32| -> Vec<u8> {
            let mut v = Vec::with_capacity(16);
            v.extend_from_slice(b"para"); // type sig
            v.extend_from_slice(&[0u8; 4]); // reserved
            v.extend_from_slice(&0_u16.to_be_bytes()); // function type 0 (gamma only)
            v.extend_from_slice(&[0u8; 2]); // reserved
            v.extend_from_slice(&encode_s15fixed16(g));
            v
        };

        // Chad tag (XYZ matrix stored as 9 XYZ values is non-standard;
        // for chad we use the sf32 array format, 'sf32' tag):
        // Actually chad uses 'sf32' type = 48 bytes data:
        //   type sig 'sf32' + reserved 4 + 9 × s15Fixed16 (36 bytes) = 44 bytes
        let mut chad_data = Vec::with_capacity(44);
        chad_data.extend_from_slice(b"sf32");
        chad_data.extend_from_slice(&[0u8; 4]);
        for &v in &chad {
            chad_data.extend_from_slice(&encode_s15fixed16(v));
        }

        // Tag data blobs
        let r_xyz = encode_xyz_tag(m[0][0], m[1][0], m[2][0]);
        let g_xyz = encode_xyz_tag(m[0][1], m[1][1], m[2][1]);
        let b_xyz = encode_xyz_tag(m[0][2], m[1][2], m[2][2]);
        let r_trc = encode_trc_tag(self.gamma);
        let g_trc = encode_trc_tag(self.gamma);
        let b_trc = encode_trc_tag(self.gamma);

        // Tag signatures
        const TAG_CHAD: [u8; 4] = *b"chad";
        const TAG_RXYZ: [u8; 4] = *b"rXYZ";
        const TAG_GXYZ: [u8; 4] = *b"gXYZ";
        const TAG_BXYZ: [u8; 4] = *b"bXYZ";
        const TAG_RTRC: [u8; 4] = *b"rTRC";
        const TAG_GTRC: [u8; 4] = *b"gTRC";
        const TAG_BTRC: [u8; 4] = *b"bTRC";

        let tags: &[([u8; 4], &[u8])] = &[
            (TAG_CHAD, &chad_data),
            (TAG_RXYZ, &r_xyz),
            (TAG_GXYZ, &g_xyz),
            (TAG_BXYZ, &b_xyz),
            (TAG_RTRC, &r_trc),
            (TAG_GTRC, &g_trc),
            (TAG_BTRC, &b_trc),
        ];

        let n_tags = tags.len() as u32;
        // Header (128) + tag count (4) + tag table (n * 12) + tag data
        let tag_table_size = 4 + n_tags as usize * 12;
        let header_plus_table = 128 + tag_table_size;

        // Compute offsets for each tag
        let mut offsets: Vec<u32> = Vec::with_capacity(tags.len());
        let mut current_offset = header_plus_table as u32;
        for (_, data) in tags {
            offsets.push(current_offset);
            current_offset += data.len() as u32;
        }

        let total_size = current_offset;

        let mut profile = Vec::with_capacity(total_size as usize);

        // --- 128-byte ICC header ---
        profile.extend_from_slice(&total_size.to_be_bytes()); // profile size
        profile.extend_from_slice(b"OXIM"); // CMM signature
        profile.extend_from_slice(&[4, 0, 0, 0]); // version 4.0.0.0
        profile.extend_from_slice(b"mntr"); // profile class: monitor
        profile.extend_from_slice(b"RGB "); // colour space: RGB
        profile.extend_from_slice(b"XYZ "); // PCS: XYZ
        profile.extend_from_slice(&[0u8; 12]); // creation date/time (zero)
        profile.extend_from_slice(b"acsp"); // profile file signature
        profile.extend_from_slice(b"APPL"); // primary platform (Apple)
        profile.extend_from_slice(&[0u8; 4]); // profile flags
        profile.extend_from_slice(b"OXIM"); // device manufacturer
        profile.extend_from_slice(b"OXIP"); // device model
        profile.extend_from_slice(&[0u8; 8]); // device attributes
        profile.extend_from_slice(&0_u32.to_be_bytes()); // rendering intent (perceptual)
        // PCS illuminant (D50 XYZ as s15Fixed16)
        profile.extend_from_slice(&encode_s15fixed16(0.9642));
        profile.extend_from_slice(&encode_s15fixed16(1.0));
        profile.extend_from_slice(&encode_s15fixed16(0.8251));
        profile.extend_from_slice(b"OXIM"); // profile creator
        profile.extend_from_slice(&[0u8; 16]); // profile ID (MD5, zero = not computed)
        profile.extend_from_slice(&[0u8; 28]); // reserved

        assert_eq!(profile.len(), 128, "header must be exactly 128 bytes");

        // --- Tag count ---
        profile.extend_from_slice(&n_tags.to_be_bytes());

        // --- Tag table ---
        for (i, (sig, data)) in tags.iter().enumerate() {
            profile.extend_from_slice(sig);
            profile.extend_from_slice(&offsets[i].to_be_bytes());
            profile.extend_from_slice(&(data.len() as u32).to_be_bytes());
        }

        // --- Tag data ---
        for (_, data) in tags {
            profile.extend_from_slice(data);
        }

        profile
    }
}

/// Encode an f32 as ICC s15Fixed16 (big-endian i32 with 16 fractional bits).
fn encode_s15fixed16(v: f32) -> [u8; 4] {
    let fixed = (v * 65536.0).round() as i32;
    fixed.to_be_bytes()
}

// ---------------------------------------------------------------------------
// CalibrationLut
// ---------------------------------------------------------------------------

/// Calibration 1-D LUT builder from measurement/target sample pairs.
pub struct CalibrationLut;

impl CalibrationLut {
    /// Build a 1-D correction LUT using piecewise-linear interpolation.
    ///
    /// Given measured values and desired target values at the same stimulus
    /// levels, produce a `size`-entry LUT that maps each input level
    /// to a corrected output via linear interpolation between sample points.
    ///
    /// # Arguments
    /// * `measured` — measured luminance/response values at each sample point.
    /// * `target`   — desired (target) values at those same sample points.
    /// * `size`     — number of entries in the output LUT.
    ///
    /// Both slices must be the same length and have at least 2 entries.
    /// Returns an empty vec on invalid input.
    #[must_use]
    pub fn build_1d_correction(measured: &[f32], target: &[f32], size: u32) -> Vec<f32> {
        if measured.len() != target.len() || measured.len() < 2 || size == 0 {
            return Vec::new();
        }
        let n = size as usize;
        let num_samples = measured.len();
        let mut lut = Vec::with_capacity(n);

        for i in 0..n {
            // LUT input in [0, 1]
            let x = i as f32 / (n - 1).max(1) as f32;

            // Find the bracketing measured-sample interval for this x
            // (measured values act as the "input axis" of the piecewise function)
            let mut lo = 0usize;
            let mut hi = num_samples - 1;

            // Handle out-of-range inputs by clamping to first/last interval
            if x <= measured[lo] {
                let t = if (measured[hi] - measured[lo]).abs() < 1e-10 {
                    0.0
                } else {
                    (x - measured[lo]) / (measured[hi] - measured[lo])
                };
                lut.push(lerp(target[lo], target[hi], t.clamp(0.0, 1.0)));
                continue;
            }
            if x >= measured[hi] {
                let t = if (measured[hi] - measured[lo]).abs() < 1e-10 {
                    1.0
                } else {
                    (x - measured[lo]) / (measured[hi] - measured[lo])
                };
                lut.push(lerp(target[lo], target[hi], t.clamp(0.0, 1.0)));
                continue;
            }

            // Binary search for the bracket
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if measured[mid] <= x {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }

            let m_lo = measured[lo];
            let m_hi = measured[hi];
            let span = m_hi - m_lo;
            let t = if span.abs() < 1e-10 {
                0.0
            } else {
                (x - m_lo) / span
            };
            lut.push(lerp(target[lo], target[hi], t.clamp(0.0, 1.0)));
        }

        lut
    }
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// ColorThermometer — Robertson's method
// ---------------------------------------------------------------------------

/// Robertson isotherm entry: `{ u, v, t }` where `t = 1/T` (reciprocal megakelvins).
struct RobertsonEntry {
    u: f64,
    v: f64,
    /// Slope of the isotherm line.
    t: f64,
}

/// Pre-computed Robertson isotherms table (21 entries covering ~1000-25000 K).
const ROBERTSON_TABLE: [RobertsonEntry; 31] = [
    RobertsonEntry { u: 0.18006, v: 0.26352, t: -0.24341 },
    RobertsonEntry { u: 0.18066, v: 0.26589, t: -0.25479 },
    RobertsonEntry { u: 0.18133, v: 0.26846, t: -0.26876 },
    RobertsonEntry { u: 0.18208, v: 0.27119, t: -0.28539 },
    RobertsonEntry { u: 0.18293, v: 0.27407, t: -0.30470 },
    RobertsonEntry { u: 0.18388, v: 0.27709, t: -0.32675 },
    RobertsonEntry { u: 0.18494, v: 0.28021, t: -0.35156 },
    RobertsonEntry { u: 0.18611, v: 0.28342, t: -0.37915 },
    RobertsonEntry { u: 0.18740, v: 0.28668, t: -0.40955 },
    RobertsonEntry { u: 0.18880, v: 0.28997, t: -0.44278 },
    RobertsonEntry { u: 0.19032, v: 0.29326, t: -0.47888 },
    RobertsonEntry { u: 0.19462, v: 0.30141, t: -0.58204 },
    RobertsonEntry { u: 0.19962, v: 0.30921, t: -0.70471 },
    RobertsonEntry { u: 0.20525, v: 0.31647, t: -0.84901 },
    RobertsonEntry { u: 0.21142, v: 0.32312, t: -1.0182  },
    RobertsonEntry { u: 0.21807, v: 0.32909, t: -1.2168  },
    RobertsonEntry { u: 0.22511, v: 0.33439, t: -1.4512  },
    RobertsonEntry { u: 0.23247, v: 0.33904, t: -1.7298  },
    RobertsonEntry { u: 0.24010, v: 0.34308, t: -2.0637  },
    RobertsonEntry { u: 0.24792, v: 0.34655, t: -2.4681  },
    RobertsonEntry { u: 0.25591, v: 0.34951, t: -2.9641  },
    RobertsonEntry { u: 0.26400, v: 0.35200, t: -3.5814  },
    RobertsonEntry { u: 0.27218, v: 0.35407, t: -4.3633  },
    RobertsonEntry { u: 0.28039, v: 0.35577, t: -5.3762  },
    RobertsonEntry { u: 0.28863, v: 0.35714, t: -6.7262  },
    RobertsonEntry { u: 0.29685, v: 0.35823, t: -8.5955  },
    RobertsonEntry { u: 0.30505, v: 0.35907, t: -11.324  },
    RobertsonEntry { u: 0.31320, v: 0.35968, t: -15.628  },
    RobertsonEntry { u: 0.32129, v: 0.36011, t: -23.325  },
    RobertsonEntry { u: 0.32931, v: 0.36038, t: -40.020  },
    RobertsonEntry { u: 0.33724, v: 0.36051, t: -116.45  },
];

/// Reciprocal megakelvins for each Robertson entry (100 MRD → 1000 K, etc.).
const ROBERTSON_MRD: [f64; 31] = [
    0.0,    10.0,  20.0,  30.0,  40.0,  50.0,  60.0,  70.0,  80.0,  90.0,
    100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 275.0, 300.0, 325.0,
    350.0, 375.0, 400.0, 425.0, 450.0, 475.0, 500.0, 525.0, 550.0, 575.0,
    600.0,
];

/// Color temperature measurement from CIE xy chromaticity.
pub struct ColorThermometer;

impl ColorThermometer {
    /// Estimate the correlated colour temperature (CCT) from CIE 1931 xy
    /// chromaticity coordinates using Robertson's method.
    ///
    /// Returns the CCT in Kelvin. Clamps to `[1000, 25000]` K.
    #[must_use]
    pub fn xy_to_cct(x: f32, y: f32) -> f32 {
        // Convert xy to CIE 1960 UCS uv
        let x = x as f64;
        let y = y as f64;
        let denom = -2.0 * x + 12.0 * y + 3.0;
        if denom.abs() < 1e-10 {
            return 6500.0;
        }
        let u = 4.0 * x / denom;
        let v = 6.0 * y / denom;

        let n = ROBERTSON_TABLE.len();
        let mut last_di = 0.0_f64;
        let mut last_i = 0usize;

        for i in 0..n {
            let du = u - ROBERTSON_TABLE[i].u;
            let dv = v - ROBERTSON_TABLE[i].v;
            // signed perpendicular distance to the isotherm
            let di = (dv - ROBERTSON_TABLE[i].t * du)
                / (1.0 + ROBERTSON_TABLE[i].t * ROBERTSON_TABLE[i].t).sqrt();

            if i > 0 && di * last_di < 0.0 {
                // Zero-crossing between i-1 and i
                let t = last_di / (last_di - di);
                let mrd = ROBERTSON_MRD[i - 1] + t * (ROBERTSON_MRD[i] - ROBERTSON_MRD[i - 1]);
                if mrd < 1e-10 {
                    return 25000.0;
                }
                let cct = 1_000_000.0 / mrd;
                return cct.clamp(1000.0, 25000.0) as f32;
            }

            last_di = di;
            last_i = i;
        }

        // Fallback: return CCT of the last entry's MRD
        let mrd = ROBERTSON_MRD[last_i];
        if mrd < 1e-10 {
            return 25000.0;
        }
        (1_000_000.0 / mrd).clamp(1000.0, 25000.0) as f32
    }
}

// ---------------------------------------------------------------------------
// GammaMeasurer
// ---------------------------------------------------------------------------

/// Display gamma measurement from luminance/stimulus pairs.
pub struct GammaMeasurer;

impl GammaMeasurer {
    /// Compute the display gamma exponent via log-log linear regression.
    ///
    /// Fits `log(L) = gamma * log(S) + c` using ordinary least squares and
    /// returns the slope `gamma`.
    ///
    /// # Arguments
    /// * `measured_luminance` — luminance values (cd/m², normalised, or any positive scale).
    /// * `stimulus`           — input stimulus values in `[0, 1]`.
    ///
    /// Both slices must have the same length ≥ 2. Pairs where either value ≤ 0
    /// are skipped. Returns `2.2` as a fallback on degenerate inputs.
    #[must_use]
    pub fn compute(measured_luminance: &[f32], stimulus: &[f32]) -> f32 {
        if measured_luminance.len() != stimulus.len() || measured_luminance.len() < 2 {
            return 2.2;
        }

        let pairs: Vec<(f64, f64)> = measured_luminance
            .iter()
            .zip(stimulus.iter())
            .filter(|(&l, &s)| l > 1e-10 && s > 1e-10)
            .map(|(&l, &s)| (f64::from(s).ln(), f64::from(l).ln()))
            .collect();

        if pairs.len() < 2 {
            return 2.2;
        }

        let n = pairs.len() as f64;
        let sum_x: f64 = pairs.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = pairs.iter().map(|(_, y)| y).sum();
        let sum_xx: f64 = pairs.iter().map(|(x, _)| x * x).sum();
        let sum_xy: f64 = pairs.iter().map(|(x, y)| x * y).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-20 {
            return 2.2;
        }

        let gamma = (n * sum_xy - sum_x * sum_y) / denom;
        if gamma.is_nan() || gamma.is_infinite() || gamma <= 0.0 {
            return 2.2;
        }

        gamma.clamp(0.5, 4.0) as f32
    }
}

// ---------------------------------------------------------------------------
// ColorCheckerTarget (passport 24 Lab reference values)
// ---------------------------------------------------------------------------

/// Reference Lab values for a standard ColorChecker Passport (24-patch) chart.
pub struct ColorCheckerPassport;

impl ColorCheckerPassport {
    /// Return the 24 CIE L*a*b* (D50) reference values for the Passport chart.
    ///
    /// Order: row 1 (patches 1-6), row 2 (7-12), row 3 (13-18), row 4 (19-24).
    #[must_use]
    pub fn lab_references() -> Vec<[f32; 3]> {
        vec![
            // Row 1 — natural colours
            [37.54,  14.37,  14.92], // Dark Skin
            [65.70,  19.29,  17.81], // Light Skin
            [49.32,  -3.82, -22.54], // Blue Sky
            [43.46, -12.74,  22.72], // Foliage
            [54.94,   9.61, -24.79], // Blue Flower
            [70.48, -32.26,  -0.37], // Bluish Green
            // Row 2 — miscellaneous
            [62.73,  35.83,  56.50], // Orange
            [39.43,  10.75, -45.17], // Purplish Blue
            [51.03,  48.13,  16.25], // Moderate Red
            [30.10,  22.54, -20.87], // Purple
            [72.75, -22.76,  57.26], // Yellow Green
            [71.94,  18.68,  67.86], // Orange Yellow
            // Row 3 — primary / secondary
            [28.78,  14.17, -49.57], // Blue
            [55.38, -37.40,  32.27], // Green
            [42.43,  53.05,  28.62], // Red
            [81.80,  -0.57,  79.04], // Yellow
            [51.94,  48.93, -14.90], // Magenta
            [51.04, -28.63, -28.64], // Cyan
            // Row 4 — greyscale
            [96.24,  -0.43,   1.19], // White
            [81.29,  -0.57,   0.44], // Neutral 8
            [66.89,  -0.75,  -0.06], // Neutral 6.5
            [50.87,  -0.15,  -0.27], // Neutral 5
            [35.66,  -0.37,  -0.45], // Neutral 3.5
            [20.46,  -0.13,  -0.15], // Black
        ]
    }
}

// ---------------------------------------------------------------------------
// LensCalibration / LensCalibrator
// ---------------------------------------------------------------------------

/// Intrinsic camera calibration parameters (pinhole + Brown-Conrady distortion).
#[derive(Debug, Clone, Copy)]
pub struct LensCalibration {
    /// Radial distortion coefficient k1.
    pub k1: f32,
    /// Radial distortion coefficient k2.
    pub k2: f32,
    /// Tangential distortion coefficient p1.
    pub p1: f32,
    /// Tangential distortion coefficient p2.
    pub p2: f32,
    /// Focal length in pixels (x-axis).
    pub fx: f32,
    /// Focal length in pixels (y-axis).
    pub fy: f32,
    /// Principal point x.
    pub cx: f32,
    /// Principal point y.
    pub cy: f32,
}

impl Default for LensCalibration {
    fn default() -> Self {
        Self {
            k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0,
            fx: 1.0, fy: 1.0, cx: 0.0, cy: 0.0,
        }
    }
}

/// Simplified DLT-based lens calibrator.
///
/// Real DLT requires N ≥ 6 point correspondences and a full linear solve;
/// this implementation uses the geometric constraints of a regular checkerboard
/// to estimate focal length and principal point via the Direct Linear Transform
/// approach, then sets distortion parameters to zero (a reasonable first-pass
/// approximation for low-distortion optics).
pub struct LensCalibrator;

impl LensCalibrator {
    /// Estimate camera intrinsics from checkerboard corner image coordinates.
    ///
    /// # Arguments
    /// * `corners`       — detected 2-D image coordinates of the checkerboard corners.
    /// * `grid_rows`     — number of interior corner rows.
    /// * `grid_cols`     — number of interior corner columns.
    /// * `square_size_mm` — physical size of each square in millimetres.
    ///
    /// Returns a `LensCalibration` with estimated `fx`, `fy`, `cx`, `cy` and
    /// distortion coefficients set to zero.
    ///
    /// Requires `corners.len() == grid_rows * grid_cols`.  Returns `None` on
    /// invalid input.
    #[must_use]
    pub fn from_checkerboard_corners(
        corners: &[(f32, f32)],
        grid_rows: u32,
        grid_cols: u32,
        square_size_mm: f32,
    ) -> Option<LensCalibration> {
        let expected = (grid_rows * grid_cols) as usize;
        if corners.len() != expected || expected < 4 || square_size_mm <= 0.0 {
            return None;
        }

        // Estimate principal point as the centroid of all corner coordinates
        let (sum_x, sum_y) = corners.iter().fold((0.0_f32, 0.0_f32), |(sx, sy), &(x, y)| {
            (sx + x, sy + y)
        });
        let cx = sum_x / corners.len() as f32;
        let cy = sum_y / corners.len() as f32;

        // Estimate focal length via the homography of the first row of corners:
        // The image width spanned by one checkerboard row corresponds to
        // (grid_cols - 1) squares. The physical width is (grid_cols-1)*square_mm.
        // A rough DLT first-order estimate: f ≈ image_span / (physical_span / assumed_distance).
        // Without known distance we use the aspect ratio approach:
        // Assume the object plane fills the sensor such that
        //   fx ≈ (pixel_span_x / physical_span_x) * assumed_distance_px
        // We approximate assumed_distance_px = max(image_span_x, image_span_y).

        // Compute spans of image coordinates
        let xs: Vec<f32> = corners.iter().map(|&(x, _)| x).collect();
        let ys: Vec<f32> = corners.iter().map(|&(_, y)| y).collect();
        let x_min = xs.iter().cloned().fold(f32::INFINITY, f32::min);
        let x_max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let y_min = ys.iter().cloned().fold(f32::INFINITY, f32::min);
        let y_max = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let image_span_x = (x_max - x_min).max(1.0);
        let image_span_y = (y_max - y_min).max(1.0);
        let physical_span_x = (grid_cols - 1).max(1) as f32 * square_size_mm;
        let physical_span_y = (grid_rows - 1).max(1) as f32 * square_size_mm;

        // Simplified focal length estimate (pixels per mm × assumed Z)
        // We use the ratio of image span to physical span scaled by the diagonal
        let diag = (image_span_x.powi(2) + image_span_y.powi(2)).sqrt();
        let fx = diag * image_span_x / physical_span_x;
        let fy = diag * image_span_y / physical_span_y;

        Some(LensCalibration {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            fx,
            fy,
            cx,
            cy,
        })
    }
}

// ---------------------------------------------------------------------------
// WhiteBalanceCalibrator
// ---------------------------------------------------------------------------

/// Per-channel white balance gains.
#[derive(Debug, Clone, Copy)]
pub struct WhiteBalanceGain {
    /// Red channel multiplier.
    pub r_gain: f32,
    /// Green channel multiplier (always 1.0 — green is the reference).
    pub g_gain: f32,
    /// Blue channel multiplier.
    pub b_gain: f32,
}

/// White balance calibration from a neutral (gray) reference patch.
pub struct WhiteBalanceCalibrator;

impl WhiteBalanceCalibrator {
    /// Compute white balance gains from a neutral gray patch.
    ///
    /// The green channel is taken as the reference (g_gain = 1.0), and the
    /// red and blue gains are computed so that all channels become equal.
    ///
    /// # Arguments
    /// * `r`, `g`, `b` — measured RGB values from the gray patch (any positive scale).
    ///
    /// Returns `None` if any channel value is ≤ 0.
    #[must_use]
    pub fn from_gray_patch(r: f32, g: f32, b: f32) -> Option<WhiteBalanceGain> {
        if r <= 0.0 || g <= 0.0 || b <= 0.0 {
            return None;
        }
        Some(WhiteBalanceGain {
            r_gain: g / r,
            g_gain: 1.0,
            b_gain: g / b,
        })
    }
}

// ---------------------------------------------------------------------------
// BradfordAdapt
// ---------------------------------------------------------------------------

/// Bradford chromatic adaptation transform utilities.
pub struct BradfordAdapt;

impl BradfordAdapt {
    /// Adapt an XYZ colour from `src_white` to `dst_white` using the Bradford
    /// CAT matrix.
    ///
    /// # Arguments
    /// * `xyz`       — input XYZ triplet to adapt.
    /// * `src_white` — source white point XYZ (normalised so Y = 1.0).
    /// * `dst_white` — target white point XYZ.
    #[must_use]
    pub fn adapt(xyz: [f32; 3], src_white: [f32; 3], dst_white: [f32; 3]) -> [f32; 3] {
        // Bradford matrix M
        const BRADFORD: [[f32; 3]; 3] = [
            [ 0.8951,  0.2664, -0.1614],
            [-0.7502,  1.7135,  0.0367],
            [ 0.0389, -0.0685,  1.0296],
        ];
        // Inverse Bradford M^{-1}
        const BRADFORD_INV: [[f32; 3]; 3] = [
            [ 0.9869929, -0.1470543,  0.1599627],
            [ 0.4323053,  0.5183603,  0.0492912],
            [-0.0085287,  0.0400428,  0.9684867],
        ];

        let src_lms = mat3_mul_vec3(BRADFORD, [src_white[0], src_white[1], src_white[2]]);
        let dst_lms = mat3_mul_vec3(BRADFORD, [dst_white[0], dst_white[1], dst_white[2]]);

        // Guard against division by zero
        let scale = [
            if src_lms[0].abs() < 1e-10 { 1.0 } else { dst_lms[0] / src_lms[0] },
            if src_lms[1].abs() < 1e-10 { 1.0 } else { dst_lms[1] / src_lms[1] },
            if src_lms[2].abs() < 1e-10 { 1.0 } else { dst_lms[2] / src_lms[2] },
        ];

        let scaled = [
            [scale[0], 0.0, 0.0],
            [0.0, scale[1], 0.0],
            [0.0, 0.0, scale[2]],
        ];

        // composed = BRADFORD_INV * scaled * BRADFORD
        let brd_xyz = mat3_mul_vec3(BRADFORD, [xyz[0], xyz[1], xyz[2]]);
        let scaled_xyz = mat3_mul_vec3(scaled, brd_xyz);
        mat3_mul_vec3(BRADFORD_INV, scaled_xyz)
    }
}

/// Multiply a 3×3 matrix by a 3-vector.
fn mat3_mul_vec3(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── IccProfileBuilder ────────────────────────────────────────────────────

    #[test]
    fn test_icc_builder_srgb_not_empty() {
        let bytes = IccProfileBuilder::new(Primaries::Srgb, 2.2, IccIlluminant::D65).build();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_icc_builder_size_in_header() {
        let bytes = IccProfileBuilder::new(Primaries::Srgb, 2.2, IccIlluminant::D65).build();
        assert!(bytes.len() >= 128);
        let size_in_header = u32::from_be_bytes(bytes[..4].try_into().expect("slice")) as usize;
        assert_eq!(size_in_header, bytes.len());
    }

    #[test]
    fn test_icc_builder_header_starts_with_correct_size() {
        let bytes = IccProfileBuilder::new(Primaries::Rec2020, 2.4, IccIlluminant::D50).build();
        let size = u32::from_be_bytes(bytes[..4].try_into().expect("slice"));
        assert!(size >= 128);
    }

    #[test]
    fn test_icc_builder_version_4() {
        let bytes = IccProfileBuilder::new(Primaries::DisplayP3, 2.2, IccIlluminant::D65).build();
        // version at offset 8: 4 bytes, first byte should be 4
        assert_eq!(bytes[8], 4);
    }

    // ── CalibrationLut ───────────────────────────────────────────────────────

    #[test]
    fn test_calibration_lut_identity() {
        // measured == target → LUT should be identity
        let measured = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
        let target   = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
        let lut = CalibrationLut::build_1d_correction(&measured, &target, 5);
        assert_eq!(lut.len(), 5);
        for (i, &v) in lut.iter().enumerate() {
            let expected = i as f32 / 4.0;
            assert!((v - expected).abs() < 1e-4, "lut[{i}] = {v}, expected {expected}");
        }
    }

    #[test]
    fn test_calibration_lut_size_respected() {
        let measured = [0.0_f32, 0.5, 1.0];
        let target   = [0.0_f32, 0.5, 1.0];
        let lut = CalibrationLut::build_1d_correction(&measured, &target, 256);
        assert_eq!(lut.len(), 256);
    }

    #[test]
    fn test_calibration_lut_invalid_lengths() {
        let lut = CalibrationLut::build_1d_correction(&[0.0, 1.0], &[0.0], 256);
        assert!(lut.is_empty());
    }

    #[test]
    fn test_calibration_lut_correction() {
        // measured is a darker response, target is ideal linear
        let measured = [0.0_f32, 0.3, 0.6, 1.0]; // measured (compressed highlights)
        let target   = [0.0_f32, 0.333, 0.666, 1.0]; // ideal linear
        let lut = CalibrationLut::build_1d_correction(&measured, &target, 4);
        assert_eq!(lut.len(), 4);
        // First and last entries must be at 0 and 1
        assert!((lut[0] - 0.0).abs() < 1e-4);
        assert!((lut[3] - 1.0).abs() < 1e-4);
    }

    // ── ColorThermometer ────────────────────────────────────────────────────

    #[test]
    fn test_cct_d65() {
        // D65 white point: x=0.3127, y=0.3290 → CCT ≈ 6500 K
        let cct = ColorThermometer::xy_to_cct(0.3127, 0.3290);
        assert!(cct >= 6000.0 && cct <= 7000.0, "D65 CCT = {cct}");
    }

    #[test]
    fn test_cct_d50() {
        // D50: x=0.3457, y=0.3585 → CCT ≈ 5000 K
        let cct = ColorThermometer::xy_to_cct(0.3457, 0.3585);
        assert!(cct >= 4500.0 && cct <= 5500.0, "D50 CCT = {cct}");
    }

    #[test]
    fn test_cct_tungsten() {
        // Illuminant A: x=0.4476, y=0.4074 → CCT ≈ 2856 K
        let cct = ColorThermometer::xy_to_cct(0.4476, 0.4074);
        assert!(cct >= 2000.0 && cct <= 3500.0, "Tungsten CCT = {cct}");
    }

    // ── GammaMeasurer ───────────────────────────────────────────────────────

    #[test]
    fn test_gamma_measurer_known_gamma() {
        // Create a perfect gamma=2.2 response
        let stimuli: Vec<f32> = (1..=8).map(|i| i as f32 / 8.0).collect();
        let luminances: Vec<f32> = stimuli.iter().map(|&s| s.powf(2.2)).collect();
        let gamma = GammaMeasurer::compute(&luminances, &stimuli);
        assert!((gamma - 2.2).abs() < 0.1, "Measured gamma = {gamma}");
    }

    #[test]
    fn test_gamma_measurer_fallback_on_mismatch() {
        let result = GammaMeasurer::compute(&[1.0, 2.0], &[1.0]);
        assert!((result - 2.2).abs() < 1e-4);
    }

    #[test]
    fn test_gamma_measurer_linear() {
        // Gamma = 1.0 means luminance == stimulus
        let stimuli: Vec<f32> = (1..=8).map(|i| i as f32 / 8.0).collect();
        let luminances = stimuli.clone();
        let gamma = GammaMeasurer::compute(&luminances, &stimuli);
        assert!((gamma - 1.0).abs() < 0.15, "Linear gamma = {gamma}");
    }

    // ── ColorCheckerPassport ────────────────────────────────────────────────

    #[test]
    fn test_passport_has_24_patches() {
        let refs = ColorCheckerPassport::lab_references();
        assert_eq!(refs.len(), 24);
    }

    #[test]
    fn test_passport_white_patch_high_l() {
        let refs = ColorCheckerPassport::lab_references();
        // Patch 19 (index 18) is White — L* ≈ 96
        assert!(refs[18][0] > 90.0, "White L* = {}", refs[18][0]);
    }

    #[test]
    fn test_passport_black_patch_low_l() {
        let refs = ColorCheckerPassport::lab_references();
        // Patch 24 (index 23) is Black — L* ≈ 20
        assert!(refs[23][0] < 30.0, "Black L* = {}", refs[23][0]);
    }

    // ── LensCalibrator ──────────────────────────────────────────────────────

    #[test]
    fn test_lens_calibrator_basic() {
        // 3×4 interior corners at regular grid
        let mut corners = Vec::new();
        for row in 0..3_u32 {
            for col in 0..4_u32 {
                corners.push((col as f32 * 50.0, row as f32 * 50.0));
            }
        }
        let cal = LensCalibrator::from_checkerboard_corners(&corners, 3, 4, 25.0);
        assert!(cal.is_some());
        let cal = cal.expect("calibration should succeed");
        assert!(cal.fx > 0.0);
        assert!(cal.fy > 0.0);
    }

    #[test]
    fn test_lens_calibrator_wrong_corner_count() {
        let cal = LensCalibrator::from_checkerboard_corners(&[(0.0, 0.0)], 3, 4, 25.0);
        assert!(cal.is_none());
    }

    #[test]
    fn test_lens_calibrator_zero_square_size() {
        let corners: Vec<(f32, f32)> = (0..12).map(|i| (i as f32, i as f32)).collect();
        let cal = LensCalibrator::from_checkerboard_corners(&corners, 3, 4, 0.0);
        assert!(cal.is_none());
    }

    // ── WhiteBalanceCalibrator ──────────────────────────────────────────────

    #[test]
    fn test_wb_neutral_gray_all_equal() {
        // All channels equal → gains should be near 1.0
        let gain = WhiteBalanceCalibrator::from_gray_patch(0.5, 0.5, 0.5);
        assert!(gain.is_some());
        let g = gain.expect("gain should be Some");
        assert!((g.r_gain - 1.0).abs() < 1e-5, "r_gain = {}", g.r_gain);
        assert!((g.g_gain - 1.0).abs() < 1e-5, "g_gain = {}", g.g_gain);
        assert!((g.b_gain - 1.0).abs() < 1e-5, "b_gain = {}", g.b_gain);
    }

    #[test]
    fn test_wb_red_channel_elevated() {
        // Red is higher than green → r_gain < 1.0
        let gain = WhiteBalanceCalibrator::from_gray_patch(0.8, 0.5, 0.5);
        let g = gain.expect("gain should be Some");
        assert!(g.r_gain < 1.0, "Expected r_gain < 1, got {}", g.r_gain);
        assert!((g.g_gain - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_wb_zero_channel_returns_none() {
        assert!(WhiteBalanceCalibrator::from_gray_patch(0.0, 0.5, 0.5).is_none());
        assert!(WhiteBalanceCalibrator::from_gray_patch(0.5, 0.0, 0.5).is_none());
        assert!(WhiteBalanceCalibrator::from_gray_patch(0.5, 0.5, 0.0).is_none());
    }

    // ── BradfordAdapt ───────────────────────────────────────────────────────

    #[test]
    fn test_bradford_adapt_same_white_point_identity() {
        let d65 = [0.9505_f32, 1.0, 1.0890];
        let xyz = [0.5, 0.5, 0.5];
        let adapted = BradfordAdapt::adapt(xyz, d65, d65);
        // Adapting from D65 to D65 should be near-identity
        assert!((adapted[0] - 0.5).abs() < 0.01, "x = {}", adapted[0]);
        assert!((adapted[1] - 0.5).abs() < 0.01, "y = {}", adapted[1]);
        assert!((adapted[2] - 0.5).abs() < 0.01, "z = {}", adapted[2]);
    }

    #[test]
    fn test_bradford_adapt_d50_to_d65() {
        let d50 = [0.9642_f32, 1.0, 0.8251];
        let d65 = [0.9505_f32, 1.0, 1.0890];
        let xyz_d50 = [0.9642_f32, 1.0, 0.8251]; // D50 white point in D50 space
        let adapted = BradfordAdapt::adapt(xyz_d50, d50, d65);
        // After adaptation, result should be close to D65 white point
        assert!((adapted[0] - 0.9505).abs() < 0.05, "X after adapt = {}", adapted[0]);
        assert!((adapted[1] - 1.0).abs() < 0.05, "Y after adapt = {}", adapted[1]);
    }

    #[test]
    fn test_bradford_adapt_result_is_finite() {
        let d50 = [0.9642_f32, 1.0, 0.8251];
        let d65 = [0.9505_f32, 1.0, 1.0890];
        let xyz = [0.3, 0.5, 0.7];
        let out = BradfordAdapt::adapt(xyz, d50, d65);
        assert!(out[0].is_finite() && out[1].is_finite() && out[2].is_finite());
    }
}
