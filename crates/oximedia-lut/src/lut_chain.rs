//! LUT chain management – compose multiple LUTs into an ordered pipeline.
//!
//! Provides:
//! * [`LutChainEntry`] – a single stage (1-D or 3-D LUT) with identity detection.
//! * [`LutChain`]      – an ordered sequence of entries with pipeline application.
//! * [`LutChainValidator`] – structural validation of a chain before use.
//! * [`resample_lut3d`] – trilinear resampling of a 3-D LUT to a different size.
//! * [`resample_lut1d`] – linear resampling of a 1-D LUT to a different count.
//! * [`LutChainNormalizer`] – automatic size harmonisation across a chain.
//!
//! ## Size Validation and Resampling
//!
//! When composing LUTs from different sources they may have incompatible grid
//! sizes (e.g. one is 17³ and another is 33³).  [`LutChainNormalizer`] analyses
//! a [`LutChain`] for size mismatches and returns a new chain where every 3-D
//! entry has been resampled to a common target size.  The target can be the
//! largest size found, the smallest, or a user-specified value.

use crate::Rgb;

// ---------------------------------------------------------------------------
// LutChainEntry
// ---------------------------------------------------------------------------

/// The kind of LUT stored in a chain entry.
#[derive(Debug, Clone, PartialEq)]
pub enum LutEntryKind {
    /// 1-D per-channel curve.  Each channel has `size` values in `[0, 1]`.
    Lut1d {
        /// Number of entries per channel.
        size: usize,
        /// Interleaved R/G/B values: `[r0, g0, b0, r1, g1, b1, …]`.
        data: Vec<f64>,
    },
    /// 3-D lattice LUT with `size³` RGB entries.
    Lut3d {
        /// Number of divisions per axis.
        size: usize,
        /// Flat lattice, row-major `[r][g][b]`.
        data: Vec<Rgb>,
    },
}

/// A single stage in a [`LutChain`].
#[derive(Debug, Clone)]
pub struct LutChainEntry {
    /// Human-readable label (e.g. filename or step name).
    pub label: String,
    /// The LUT data for this stage.
    pub kind: LutEntryKind,
}

impl LutChainEntry {
    /// Create a 1-D chain entry.
    #[must_use]
    pub fn new_1d(label: impl Into<String>, size: usize, data: Vec<f64>) -> Self {
        Self {
            label: label.into(),
            kind: LutEntryKind::Lut1d { size, data },
        }
    }

    /// Create a 3-D chain entry.
    #[must_use]
    pub fn new_3d(label: impl Into<String>, size: usize, data: Vec<Rgb>) -> Self {
        Self {
            label: label.into(),
            kind: LutEntryKind::Lut3d { size, data },
        }
    }

    /// Returns `true` when this entry performs no colour transformation.
    ///
    /// A 1-D entry is identity when every value equals its normalised position.
    /// A 3-D entry is identity when every lattice point equals its lattice index.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        const EPS: f64 = 1e-6;
        match &self.kind {
            LutEntryKind::Lut1d { size, data } => {
                if *size == 0 || data.len() != size * 3 {
                    return false;
                }
                let scale = (*size - 1) as f64;
                for i in 0..*size {
                    let expected = i as f64 / scale;
                    if (data[i * 3] - expected).abs() > EPS
                        || (data[i * 3 + 1] - expected).abs() > EPS
                        || (data[i * 3 + 2] - expected).abs() > EPS
                    {
                        return false;
                    }
                }
                true
            }
            LutEntryKind::Lut3d { size, data } => {
                if *size < 2 || data.len() != size * size * size {
                    return false;
                }
                let scale = (*size - 1) as f64;
                for r in 0..*size {
                    for g in 0..*size {
                        for b in 0..*size {
                            let idx = r * size * size + g * size + b;
                            let exp = [r as f64 / scale, g as f64 / scale, b as f64 / scale];
                            if (data[idx][0] - exp[0]).abs() > EPS
                                || (data[idx][1] - exp[1]).abs() > EPS
                                || (data[idx][2] - exp[2]).abs() > EPS
                            {
                                return false;
                            }
                        }
                    }
                }
                true
            }
        }
    }

    /// Apply this entry to a single RGB pixel using trilinear interpolation for
    /// 3-D and linear interpolation for 1-D.
    #[must_use]
    pub fn apply_rgb(&self, pixel: Rgb) -> Rgb {
        match &self.kind {
            LutEntryKind::Lut1d { size, data } => apply_1d(pixel, *size, data),
            LutEntryKind::Lut3d { size, data } => apply_3d_trilinear(pixel, *size, data),
        }
    }
}

// ---------------------------------------------------------------------------
// LutChain
// ---------------------------------------------------------------------------

/// An ordered sequence of LUT stages applied left-to-right.
#[derive(Debug, Clone, Default)]
pub struct LutChain {
    entries: Vec<LutChainEntry>,
}

impl LutChain {
    /// Create an empty chain.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an entry to the end of the chain.
    pub fn push(&mut self, entry: LutChainEntry) {
        self.entries.push(entry);
    }

    /// Number of stages in the chain.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.entries.len()
    }

    /// Apply every stage in sequence to `pixel`.
    #[must_use]
    pub fn apply_rgb(&self, mut pixel: Rgb) -> Rgb {
        for entry in &self.entries {
            pixel = entry.apply_rgb(pixel);
        }
        pixel
    }

    /// Returns a reference to the entry at `index`, or `None`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&LutChainEntry> {
        self.entries.get(index)
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &LutChainEntry> {
        self.entries.iter()
    }

    /// Returns `true` when every entry in the chain is an identity.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.entries.iter().all(LutChainEntry::is_identity)
    }
}

// ---------------------------------------------------------------------------
// LutChainValidator
// ---------------------------------------------------------------------------

/// Validation error returned by [`LutChainValidator::validate`].
#[derive(Debug, Clone, PartialEq)]
pub enum ChainValidationError {
    /// Chain is empty.
    Empty,
    /// Entry at `index` has an invalid data length.
    InvalidDataLength {
        /// Position of the invalid entry in the chain.
        index: usize,
        /// Label of the invalid entry.
        label: String,
    },
    /// Entry at `index` has a size that is too small.
    SizeTooSmall {
        /// Position of the undersized entry in the chain.
        index: usize,
        /// Label of the undersized entry.
        label: String,
    },
}

impl std::fmt::Display for ChainValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "LUT chain is empty"),
            Self::InvalidDataLength { index, label } => {
                write!(f, "Entry {index} '{label}': data length mismatch")
            }
            Self::SizeTooSmall { index, label } => {
                write!(f, "Entry {index} '{label}': size must be >= 2")
            }
        }
    }
}

/// Validates a [`LutChain`] before use.
pub struct LutChainValidator;

impl LutChainValidator {
    /// Validate all entries in `chain`.
    ///
    /// Returns `Ok(())` when every entry has consistent dimensions, or the
    /// first [`ChainValidationError`] encountered.
    pub fn validate(chain: &LutChain) -> Result<(), ChainValidationError> {
        if chain.depth() == 0 {
            return Err(ChainValidationError::Empty);
        }
        for (i, entry) in chain.iter().enumerate() {
            match &entry.kind {
                LutEntryKind::Lut1d { size, data } => {
                    if *size < 2 {
                        return Err(ChainValidationError::SizeTooSmall {
                            index: i,
                            label: entry.label.clone(),
                        });
                    }
                    if data.len() != size * 3 {
                        return Err(ChainValidationError::InvalidDataLength {
                            index: i,
                            label: entry.label.clone(),
                        });
                    }
                }
                LutEntryKind::Lut3d { size, data } => {
                    if *size < 2 {
                        return Err(ChainValidationError::SizeTooSmall {
                            index: i,
                            label: entry.label.clone(),
                        });
                    }
                    if data.len() != size * size * size {
                        return Err(ChainValidationError::InvalidDataLength {
                            index: i,
                            label: entry.label.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Linear interpolation along each channel of a 1-D LUT.
fn apply_1d(pixel: Rgb, size: usize, data: &[f64]) -> Rgb {
    if size < 2 || data.len() < size * 3 {
        return pixel;
    }
    let scale = (size - 1) as f64;
    let mut out = [0.0f64; 3];
    for ch in 0..3 {
        let v = pixel[ch].clamp(0.0, 1.0) * scale;
        let lo = v.floor() as usize;
        let hi = (lo + 1).min(size - 1);
        let t = v - lo as f64;
        out[ch] = data[lo * 3 + ch] * (1.0 - t) + data[hi * 3 + ch] * t;
    }
    out
}

/// Trilinear interpolation on a 3-D LUT (size³ entries, row-major `[r][g][b]`).
fn apply_3d_trilinear(pixel: Rgb, size: usize, data: &[Rgb]) -> Rgb {
    if size < 2 || data.len() < size * size * size {
        return pixel;
    }
    let scale = (size - 1) as f64;
    let rv = pixel[0].clamp(0.0, 1.0) * scale;
    let gv = pixel[1].clamp(0.0, 1.0) * scale;
    let bv = pixel[2].clamp(0.0, 1.0) * scale;

    let r0 = rv.floor() as usize;
    let g0 = gv.floor() as usize;
    let b0 = bv.floor() as usize;
    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);

    let tr = rv - r0 as f64;
    let tg = gv - g0 as f64;
    let tb = bv - b0 as f64;

    macro_rules! idx {
        ($r:expr, $g:expr, $b:expr) => {
            $r * size * size + $g * size + $b
        };
    }

    let c000 = data[idx!(r0, g0, b0)];
    let c001 = data[idx!(r0, g0, b1)];
    let c010 = data[idx!(r0, g1, b0)];
    let c011 = data[idx!(r0, g1, b1)];
    let c100 = data[idx!(r1, g0, b0)];
    let c101 = data[idx!(r1, g0, b1)];
    let c110 = data[idx!(r1, g1, b0)];
    let c111 = data[idx!(r1, g1, b1)];

    let mut out = [0.0f64; 3];
    for ch in 0..3 {
        out[ch] = c000[ch] * (1.0 - tr) * (1.0 - tg) * (1.0 - tb)
            + c001[ch] * (1.0 - tr) * (1.0 - tg) * tb
            + c010[ch] * (1.0 - tr) * tg * (1.0 - tb)
            + c011[ch] * (1.0 - tr) * tg * tb
            + c100[ch] * tr * (1.0 - tg) * (1.0 - tb)
            + c101[ch] * tr * (1.0 - tg) * tb
            + c110[ch] * tr * tg * (1.0 - tb)
            + c111[ch] * tr * tg * tb;
    }
    out
}

// ---------------------------------------------------------------------------
// LUT size resampling (Item 1 implementation)
// ---------------------------------------------------------------------------

/// Resample a 3-D LUT to a new grid size using trilinear interpolation.
///
/// `data` must have `src_size³` entries.  The result has `dst_size³` entries.
///
/// # Errors
///
/// Returns an error string when `data.len() != src_size³`, `src_size < 2`,
/// or `dst_size < 2`.
pub fn resample_lut3d(data: &[Rgb], src_size: usize, dst_size: usize) -> Result<Vec<Rgb>, String> {
    if src_size < 2 {
        return Err(format!("src_size must be >= 2, got {src_size}"));
    }
    if dst_size < 2 {
        return Err(format!("dst_size must be >= 2, got {dst_size}"));
    }
    if data.len() != src_size * src_size * src_size {
        return Err(format!(
            "Expected {} entries for src_size={src_size}, got {}",
            src_size * src_size * src_size,
            data.len(),
        ));
    }

    let mut output = Vec::with_capacity(dst_size * dst_size * dst_size);
    let dst_scale = (dst_size - 1) as f64;

    for ri in 0..dst_size {
        for gi in 0..dst_size {
            for bi in 0..dst_size {
                let r_norm = ri as f64 / dst_scale;
                let g_norm = gi as f64 / dst_scale;
                let b_norm = bi as f64 / dst_scale;
                output.push(apply_3d_trilinear([r_norm, g_norm, b_norm], src_size, data));
            }
        }
    }

    Ok(output)
}

/// Resample a 1-D LUT (interleaved RGB) to a new entry count using linear interpolation.
///
/// `data` is `[r0, g0, b0, r1, g1, b1, …]` with length `src_size * 3`.
///
/// # Errors
///
/// Returns an error string when `data.len() != src_size * 3`, `src_size < 2`,
/// or `dst_size < 2`.
pub fn resample_lut1d(data: &[f64], src_size: usize, dst_size: usize) -> Result<Vec<f64>, String> {
    if src_size < 2 {
        return Err(format!("src_size must be >= 2, got {src_size}"));
    }
    if dst_size < 2 {
        return Err(format!("dst_size must be >= 2, got {dst_size}"));
    }
    if data.len() != src_size * 3 {
        return Err(format!(
            "Expected {} values for src_size={src_size}, got {}",
            src_size * 3,
            data.len(),
        ));
    }

    let src_scale = (src_size - 1) as f64;
    let dst_scale = (dst_size - 1) as f64;
    let mut output = Vec::with_capacity(dst_size * 3);

    for di in 0..dst_size {
        let t = di as f64 / dst_scale * src_scale;
        let lo = t.floor() as usize;
        let hi = (lo + 1).min(src_size - 1);
        let frac = t - lo as f64;
        for ch in 0..3 {
            let a = data[lo * 3 + ch];
            let b = data[hi * 3 + ch];
            output.push(a * (1.0 - frac) + b * frac);
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// LutChainNormalizer
// ---------------------------------------------------------------------------

/// Strategy for selecting the target size when normalising a chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizeNormalisationStrategy {
    /// Use the largest size found in the chain for that LUT kind.
    Largest,
    /// Use the smallest size found.
    Smallest,
    /// Use a fixed explicit size.
    Fixed(usize),
}

/// Size distribution report for a chain.
#[derive(Clone, Debug)]
pub struct ChainSizeReport {
    /// True when all 3-D entries share the same size.
    pub uniform_3d: bool,
    /// True when all 1-D entries share the same size.
    pub uniform_1d: bool,
    /// True when all entries (of the same kind) have matching sizes.
    pub uniform: bool,
    /// Minimum 3-D size (0 if none present).
    pub min_3d_size: usize,
    /// Maximum 3-D size (0 if none present).
    pub max_3d_size: usize,
    /// Minimum 1-D size (0 if none present).
    pub min_1d_size: usize,
    /// Maximum 1-D size (0 if none present).
    pub max_1d_size: usize,
}

/// Analyses and normalises the LUT sizes inside a [`LutChain`].
pub struct LutChainNormalizer;

impl LutChainNormalizer {
    /// Analyse the size distribution of all entries in `chain`.
    #[must_use]
    pub fn analyse(chain: &LutChain) -> ChainSizeReport {
        let mut min_3d = usize::MAX;
        let mut max_3d = 0_usize;
        let mut min_1d = usize::MAX;
        let mut max_1d = 0_usize;

        for entry in chain.iter() {
            match &entry.kind {
                LutEntryKind::Lut3d { size, .. } => {
                    min_3d = min_3d.min(*size);
                    max_3d = max_3d.max(*size);
                }
                LutEntryKind::Lut1d { size, .. } => {
                    min_1d = min_1d.min(*size);
                    max_1d = max_1d.max(*size);
                }
            }
        }

        let (min_3d, max_3d) = if min_3d == usize::MAX {
            (0, 0)
        } else {
            (min_3d, max_3d)
        };
        let (min_1d, max_1d) = if min_1d == usize::MAX {
            (0, 0)
        } else {
            (min_1d, max_1d)
        };

        let uniform_3d = max_3d == 0 || min_3d == max_3d;
        let uniform_1d = max_1d == 0 || min_1d == max_1d;

        ChainSizeReport {
            uniform_3d,
            uniform_1d,
            uniform: uniform_3d && uniform_1d,
            min_3d_size: min_3d,
            max_3d_size: max_3d,
            min_1d_size: min_1d,
            max_1d_size: max_1d,
        }
    }

    /// Normalise all entries to a common size chosen by `strategy_3d` / `strategy_1d`.
    ///
    /// Returns a new chain where every 3-D entry has been resampled to the
    /// target 3-D size and every 1-D entry to the target 1-D size.
    /// Entries that already match are preserved without modification.
    ///
    /// # Errors
    ///
    /// Returns an error string if any resampling step fails.
    pub fn normalise(
        chain: &LutChain,
        strategy_3d: SizeNormalisationStrategy,
        strategy_1d: SizeNormalisationStrategy,
    ) -> Result<LutChain, String> {
        let report = Self::analyse(chain);

        let target_3d = match strategy_3d {
            SizeNormalisationStrategy::Largest => report.max_3d_size,
            SizeNormalisationStrategy::Smallest => report.min_3d_size,
            SizeNormalisationStrategy::Fixed(s) => s,
        };

        let target_1d = match strategy_1d {
            SizeNormalisationStrategy::Largest => report.max_1d_size,
            SizeNormalisationStrategy::Smallest => report.min_1d_size,
            SizeNormalisationStrategy::Fixed(s) => s,
        };

        let mut new_chain = LutChain::new();

        for entry in chain.iter() {
            match &entry.kind {
                LutEntryKind::Lut3d { size, data } => {
                    if target_3d == 0 || *size == target_3d {
                        new_chain.push(entry.clone());
                    } else {
                        let resampled = resample_lut3d(data, *size, target_3d)
                            .map_err(|e| format!("Resampling '{}' failed: {e}", entry.label))?;
                        new_chain.push(LutChainEntry::new_3d(
                            entry.label.clone(),
                            target_3d,
                            resampled,
                        ));
                    }
                }
                LutEntryKind::Lut1d { size, data } => {
                    if target_1d == 0 || *size == target_1d {
                        new_chain.push(entry.clone());
                    } else {
                        let resampled = resample_lut1d(data, *size, target_1d)
                            .map_err(|e| format!("Resampling '{}' failed: {e}", entry.label))?;
                        new_chain.push(LutChainEntry::new_1d(
                            entry.label.clone(),
                            target_1d,
                            resampled,
                        ));
                    }
                }
            }
        }

        Ok(new_chain)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_1d(size: usize) -> LutChainEntry {
        let scale = (size - 1) as f64;
        let data: Vec<f64> = (0..size)
            .flat_map(|i| {
                let v = i as f64 / scale;
                [v, v, v]
            })
            .collect();
        LutChainEntry::new_1d("id1d", size, data)
    }

    fn make_identity_3d(size: usize) -> LutChainEntry {
        let scale = (size - 1) as f64;
        let mut data = Vec::with_capacity(size * size * size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    data.push([r as f64 / scale, g as f64 / scale, b as f64 / scale]);
                }
            }
        }
        LutChainEntry::new_3d("id3d", size, data)
    }

    #[test]
    fn test_identity_1d_detected() {
        let e = make_identity_1d(17);
        assert!(e.is_identity());
    }

    #[test]
    fn test_non_identity_1d() {
        let mut e = make_identity_1d(5);
        if let LutEntryKind::Lut1d { data, .. } = &mut e.kind {
            data[3] = 0.999; // shift red channel of index 1
        }
        assert!(!e.is_identity());
    }

    #[test]
    fn test_identity_3d_detected() {
        let e = make_identity_3d(5);
        assert!(e.is_identity());
    }

    #[test]
    fn test_non_identity_3d() {
        let mut e = make_identity_3d(5);
        if let LutEntryKind::Lut3d { data, .. } = &mut e.kind {
            data[1][0] += 0.1;
        }
        assert!(!e.is_identity());
    }

    #[test]
    fn test_apply_1d_identity_passthrough() {
        let e = make_identity_1d(33);
        let pixel = [0.2, 0.5, 0.8];
        let out = e.apply_rgb(pixel);
        assert!((out[0] - 0.2).abs() < 1e-4);
        assert!((out[1] - 0.5).abs() < 1e-4);
        assert!((out[2] - 0.8).abs() < 1e-4);
    }

    #[test]
    fn test_apply_3d_identity_passthrough() {
        let e = make_identity_3d(17);
        let pixel = [0.3, 0.6, 0.9];
        let out = e.apply_rgb(pixel);
        assert!((out[0] - 0.3).abs() < 1e-4);
        assert!((out[1] - 0.6).abs() < 1e-4);
        assert!((out[2] - 0.9).abs() < 1e-4);
    }

    #[test]
    fn test_chain_depth() {
        let mut chain = LutChain::new();
        assert_eq!(chain.depth(), 0);
        chain.push(make_identity_1d(17));
        assert_eq!(chain.depth(), 1);
        chain.push(make_identity_3d(17));
        assert_eq!(chain.depth(), 2);
    }

    #[test]
    fn test_chain_apply_two_identities() {
        let mut chain = LutChain::new();
        chain.push(make_identity_1d(17));
        chain.push(make_identity_3d(17));
        let pixel = [0.4, 0.5, 0.6];
        let out = chain.apply_rgb(pixel);
        assert!((out[0] - 0.4).abs() < 1e-3);
        assert!((out[1] - 0.5).abs() < 1e-3);
        assert!((out[2] - 0.6).abs() < 1e-3);
    }

    #[test]
    fn test_chain_is_identity_all() {
        let mut chain = LutChain::new();
        chain.push(make_identity_1d(9));
        chain.push(make_identity_3d(9));
        assert!(chain.is_identity());
    }

    #[test]
    fn test_chain_is_not_identity_when_one_differs() {
        let mut chain = LutChain::new();
        chain.push(make_identity_1d(9));
        let mut e = make_identity_3d(9);
        if let LutEntryKind::Lut3d { data, .. } = &mut e.kind {
            data[0][0] = 0.5;
        }
        chain.push(e);
        assert!(!chain.is_identity());
    }

    #[test]
    fn test_validator_empty_chain() {
        let chain = LutChain::new();
        assert_eq!(
            LutChainValidator::validate(&chain),
            Err(ChainValidationError::Empty)
        );
    }

    #[test]
    fn test_validator_valid_chain() {
        let mut chain = LutChain::new();
        chain.push(make_identity_3d(17));
        assert!(LutChainValidator::validate(&chain).is_ok());
    }

    #[test]
    fn test_validator_size_too_small() {
        let entry = LutChainEntry::new_3d("bad", 1, vec![[0.0, 0.0, 0.0]]);
        let mut chain = LutChain::new();
        chain.push(entry);
        assert!(matches!(
            LutChainValidator::validate(&chain),
            Err(ChainValidationError::SizeTooSmall { .. })
        ));
    }

    #[test]
    fn test_validator_data_length_mismatch_3d() {
        let entry = LutChainEntry::new_3d("bad", 3, vec![[0.0, 0.0, 0.0]; 5]);
        let mut chain = LutChain::new();
        chain.push(entry);
        assert!(matches!(
            LutChainValidator::validate(&chain),
            Err(ChainValidationError::InvalidDataLength { .. })
        ));
    }

    #[test]
    fn test_get_entry() {
        let mut chain = LutChain::new();
        chain.push(make_identity_1d(5));
        assert!(chain.get(0).is_some());
        assert!(chain.get(1).is_none());
    }

    #[test]
    fn test_entry_label() {
        let e = LutChainEntry::new_1d("my_lut", 5, vec![0.0; 15]);
        assert_eq!(e.label, "my_lut");
    }

    // -----------------------------------------------------------------------
    // Resampling tests (Item 1)
    // -----------------------------------------------------------------------

    fn identity_1d_data(size: usize) -> Vec<f64> {
        let scale = (size - 1) as f64;
        (0..size)
            .flat_map(|i| {
                let v = i as f64 / scale;
                [v, v, v]
            })
            .collect()
    }

    fn identity_3d_rgb_data(size: usize) -> Vec<Rgb> {
        let scale = (size - 1) as f64;
        let mut d = Vec::with_capacity(size * size * size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    d.push([r as f64 / scale, g as f64 / scale, b as f64 / scale]);
                }
            }
        }
        d
    }

    #[test]
    fn test_resample_3d_same_size_noop() {
        let size = 5_usize;
        let data = identity_3d_rgb_data(size);
        let out = resample_lut3d(&data, size, size).expect("resample");
        assert_eq!(out.len(), size * size * size);
        for (a, b) in data.iter().zip(out.iter()) {
            assert!((a[0] - b[0]).abs() < 1e-9);
            assert!((a[1] - b[1]).abs() < 1e-9);
            assert!((a[2] - b[2]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_resample_3d_upsample_identity_accurate() {
        let src = 3_usize;
        let dst = 9_usize;
        let data = identity_3d_rgb_data(src);
        let out = resample_lut3d(&data, src, dst).expect("resample");
        assert_eq!(out.len(), dst * dst * dst);

        let dst_scale = (dst - 1) as f64;
        for ri in 0..dst {
            for gi in 0..dst {
                for bi in 0..dst {
                    let idx = ri * dst * dst + gi * dst + bi;
                    let exp = [
                        ri as f64 / dst_scale,
                        gi as f64 / dst_scale,
                        bi as f64 / dst_scale,
                    ];
                    assert!((out[idx][0] - exp[0]).abs() < 1e-9);
                    assert!((out[idx][1] - exp[1]).abs() < 1e-9);
                    assert!((out[idx][2] - exp[2]).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn test_resample_3d_downsample_accurate() {
        let src = 9_usize;
        let dst = 3_usize;
        let data = identity_3d_rgb_data(src);
        let out = resample_lut3d(&data, src, dst).expect("resample");
        assert_eq!(out.len(), dst * dst * dst);

        let dst_scale = (dst - 1) as f64;
        for ri in 0..dst {
            for gi in 0..dst {
                for bi in 0..dst {
                    let idx = ri * dst * dst + gi * dst + bi;
                    let exp_r = ri as f64 / dst_scale;
                    assert!((out[idx][0] - exp_r).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn test_resample_3d_error_src_size() {
        assert!(resample_lut3d(&vec![[0.5f64; 3]; 8], 1, 3).is_err());
    }

    #[test]
    fn test_resample_3d_error_dst_size() {
        assert!(resample_lut3d(&vec![[0.5f64; 3]; 8], 2, 1).is_err());
    }

    #[test]
    fn test_resample_3d_error_wrong_length() {
        assert!(resample_lut3d(&vec![[0.5f64; 3]; 10], 3, 5).is_err());
    }

    #[test]
    fn test_resample_1d_same_size() {
        let data = identity_1d_data(9);
        let out = resample_lut1d(&data, 9, 9).expect("resample");
        assert_eq!(out.len(), 27);
        for (a, b) in data.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_resample_1d_upsample_identity() {
        let src = 3_usize;
        let dst = 9_usize;
        let data = identity_1d_data(src);
        let out = resample_lut1d(&data, src, dst).expect("resample");
        assert_eq!(out.len(), dst * 3);

        let dst_scale = (dst - 1) as f64;
        for i in 0..dst {
            let expected = i as f64 / dst_scale;
            for ch in 0..3 {
                assert!((out[i * 3 + ch] - expected).abs() < 1e-9, "i={i} ch={ch}");
            }
        }
    }

    #[test]
    fn test_resample_1d_errors() {
        let data = vec![0.0f64; 15];
        assert!(resample_lut1d(&data, 1, 9).is_err()); // src_size < 2
        assert!(resample_lut1d(&data, 5, 1).is_err()); // dst_size < 2
        assert!(resample_lut1d(&data, 4, 9).is_err()); // len mismatch
    }

    // -----------------------------------------------------------------------
    // LutChainNormalizer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_size_report_uniform() {
        let mut chain = LutChain::new();
        chain.push(make_identity_3d(17));
        chain.push(make_identity_3d(17));
        let r = LutChainNormalizer::analyse(&chain);
        assert!(r.uniform_3d);
        assert!(r.uniform);
        assert_eq!(r.min_3d_size, 17);
        assert_eq!(r.max_3d_size, 17);
    }

    #[test]
    fn test_size_report_non_uniform() {
        let mut chain = LutChain::new();
        chain.push(make_identity_3d(5));
        chain.push(make_identity_3d(9));
        let r = LutChainNormalizer::analyse(&chain);
        assert!(!r.uniform_3d);
        assert_eq!(r.min_3d_size, 5);
        assert_eq!(r.max_3d_size, 9);
    }

    #[test]
    fn test_normalise_fixed_size() {
        let mut chain = LutChain::new();
        chain.push(make_identity_3d(5));
        chain.push(make_identity_3d(9));
        let normalised = LutChainNormalizer::normalise(
            &chain,
            SizeNormalisationStrategy::Fixed(7),
            SizeNormalisationStrategy::Fixed(33),
        )
        .expect("normalise");
        assert_eq!(normalised.depth(), 2);
        for entry in normalised.iter() {
            if let LutEntryKind::Lut3d { size, .. } = &entry.kind {
                assert_eq!(*size, 7);
            }
        }
    }

    #[test]
    fn test_normalise_largest() {
        let mut chain = LutChain::new();
        chain.push(make_identity_3d(5));
        chain.push(make_identity_3d(9));
        let n = LutChainNormalizer::normalise(
            &chain,
            SizeNormalisationStrategy::Largest,
            SizeNormalisationStrategy::Largest,
        )
        .expect("normalise");
        for entry in n.iter() {
            if let LutEntryKind::Lut3d { size, .. } = &entry.kind {
                assert_eq!(*size, 9);
            }
        }
    }

    #[test]
    fn test_normalise_smallest() {
        let mut chain = LutChain::new();
        chain.push(make_identity_3d(5));
        chain.push(make_identity_3d(9));
        let n = LutChainNormalizer::normalise(
            &chain,
            SizeNormalisationStrategy::Smallest,
            SizeNormalisationStrategy::Smallest,
        )
        .expect("normalise");
        for entry in n.iter() {
            if let LutEntryKind::Lut3d { size, .. } = &entry.kind {
                assert_eq!(*size, 5);
            }
        }
    }

    #[test]
    fn test_normalise_1d() {
        let mut chain = LutChain::new();
        chain.push(make_identity_1d(5));
        chain.push(make_identity_1d(9));
        let n = LutChainNormalizer::normalise(
            &chain,
            SizeNormalisationStrategy::Largest,
            SizeNormalisationStrategy::Fixed(17),
        )
        .expect("normalise");
        for entry in n.iter() {
            if let LutEntryKind::Lut1d { size, .. } = &entry.kind {
                assert_eq!(*size, 17);
            }
        }
    }

    #[test]
    fn test_normalise_preserves_identity() {
        let mut chain = LutChain::new();
        chain.push(make_identity_3d(5));
        chain.push(make_identity_3d(9));
        let n = LutChainNormalizer::normalise(
            &chain,
            SizeNormalisationStrategy::Fixed(7),
            SizeNormalisationStrategy::Fixed(33),
        )
        .expect("normalise");
        let pixel = [0.4, 0.6, 0.2];
        let out = n.apply_rgb(pixel);
        assert!((out[0] - pixel[0]).abs() < 1e-6, "r={}", out[0]);
        assert!((out[1] - pixel[1]).abs() < 1e-6, "g={}", out[1]);
        assert!((out[2] - pixel[2]).abs() < 1e-6, "b={}", out[2]);
    }
}
