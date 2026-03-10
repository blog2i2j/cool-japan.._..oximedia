//! LTO tape library management.
//!
//! Models LTO tape formats (7–10), individual cartridges, and a multi-drive
//! tape library suitable for offline/nearline archive workflows.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ── Tape format ───────────────────────────────────────────────────────────────

/// LTO tape generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapeFormat {
    /// LTO-7 (6 TB native, 15 TB compressed).
    Lto7,
    /// LTO-8 (12 TB native, 30 TB compressed).
    Lto8,
    /// LTO-9 (18 TB native, 45 TB compressed).
    Lto9,
    /// LTO-10 (36 TB native, 90 TB compressed).
    Lto10,
}

impl TapeFormat {
    /// Native (uncompressed) capacity in GB.
    #[must_use]
    pub fn capacity_gb(&self) -> u64 {
        match self {
            Self::Lto7 => 6_000,
            Self::Lto8 => 12_000,
            Self::Lto9 => 18_000,
            Self::Lto10 => 36_000,
        }
    }

    /// Native sustained write speed in MB/s.
    #[must_use]
    pub fn native_speed_mbps(&self) -> f64 {
        match self {
            Self::Lto7 => 300.0,
            Self::Lto8 => 360.0,
            Self::Lto9 => 400.0,
            Self::Lto10 => 900.0,
        }
    }

    /// Compressed capacity in GB (assumes 2.5:1 compression ratio).
    #[must_use]
    pub fn compressed_capacity_gb(&self) -> u64 {
        match self {
            Self::Lto7 => 15_000,
            Self::Lto8 => 30_000,
            Self::Lto9 => 45_000,
            Self::Lto10 => 90_000,
        }
    }

    /// Generation number (7, 8, 9, or 10).
    #[must_use]
    pub fn generation(&self) -> u8 {
        match self {
            Self::Lto7 => 7,
            Self::Lto8 => 8,
            Self::Lto9 => 9,
            Self::Lto10 => 10,
        }
    }
}

// ── Tape status ───────────────────────────────────────────────────────────────

/// Operational status of a tape cartridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapeStatus {
    /// Cartridge has never been written.
    Blank,
    /// In active use; still has free space.
    Active,
    /// No usable free space remaining.
    Full,
    /// Marked for reuse; data may be erased.
    Scratch,
    /// Exceeded maximum write cycles or age limit.
    Expired,
}

impl TapeStatus {
    /// Returns `true` if the cartridge can accept new writes.
    #[must_use]
    pub fn is_writable(&self) -> bool {
        matches!(self, Self::Blank | Self::Active | Self::Scratch)
    }
}

// ── Tape cartridge ────────────────────────────────────────────────────────────

/// A physical LTO tape cartridge.
#[derive(Debug, Clone)]
pub struct TapeCartridge {
    /// Human-readable barcode label (e.g. `"LTO000001L9"`).
    pub barcode: String,
    /// LTO generation of this cartridge.
    pub format: TapeFormat,
    /// Amount of data already written, in GB.
    pub used_gb: f64,
    /// Current operational status.
    pub status: TapeStatus,
}

impl TapeCartridge {
    /// Creates a new blank cartridge.
    #[must_use]
    pub fn new(barcode: &str, format: TapeFormat) -> Self {
        Self {
            barcode: barcode.to_string(),
            format,
            used_gb: 0.0,
            status: TapeStatus::Blank,
        }
    }

    /// Returns the available (free) space in GB.
    #[must_use]
    pub fn available_gb(&self) -> f64 {
        let cap = self.format.capacity_gb() as f64;
        (cap - self.used_gb).max(0.0)
    }

    /// Returns the percentage of the cartridge that has been used (`[0.0, 100.0]`).
    #[must_use]
    pub fn utilization_pct(&self) -> f64 {
        let cap = self.format.capacity_gb() as f64;
        if cap <= 0.0 {
            return 100.0;
        }
        (self.used_gb / cap * 100.0).clamp(0.0, 100.0)
    }

    /// Estimates the write time in seconds for `data_gb` GB of data.
    ///
    /// Uses the native (uncompressed) write speed.
    #[must_use]
    pub fn write_time_s(&self, data_gb: f64) -> f64 {
        let speed_gbps = self.format.native_speed_mbps() / 1024.0;
        if speed_gbps <= 0.0 {
            return f64::MAX;
        }
        data_gb / speed_gbps
    }

    /// Marks the cartridge active and records a write of `data_gb` GB.
    ///
    /// Clamps `used_gb` at the cartridge capacity.
    pub fn record_write(&mut self, data_gb: f64) {
        self.used_gb = (self.used_gb + data_gb).min(self.format.capacity_gb() as f64);
        if self.available_gb() <= 0.0 {
            self.status = TapeStatus::Full;
        } else {
            self.status = TapeStatus::Active;
        }
    }
}

// ── Tape library ─────────────────────────────────────────────────────────────

/// A multi-drive LTO tape library (robotic or manual).
pub struct TapeLibrary {
    /// All cartridges stored in the library.
    pub cartridges: Vec<TapeCartridge>,
    /// Number of tape drives available.
    pub drives: usize,
}

impl TapeLibrary {
    /// Creates a new empty library with `drives` tape drives.
    #[must_use]
    pub fn new(drives: usize) -> Self {
        Self {
            cartridges: Vec::new(),
            drives,
        }
    }

    /// Adds a cartridge to the library.
    pub fn add_cartridge(&mut self, c: TapeCartridge) {
        self.cartridges.push(c);
    }

    /// Finds a cartridge by barcode (case-sensitive).
    #[must_use]
    pub fn find_cartridge(&self, barcode: &str) -> Option<&TapeCartridge> {
        self.cartridges.iter().find(|c| c.barcode == barcode)
    }

    /// Finds a mutable reference to a cartridge by barcode.
    #[must_use]
    pub fn find_cartridge_mut(&mut self, barcode: &str) -> Option<&mut TapeCartridge> {
        self.cartridges.iter_mut().find(|c| c.barcode == barcode)
    }

    /// Returns the total available (free) capacity across all writable cartridges in GB.
    #[must_use]
    pub fn available_capacity_gb(&self) -> f64 {
        self.cartridges
            .iter()
            .filter(|c| c.status.is_writable())
            .map(TapeCartridge::available_gb)
            .sum()
    }

    /// Returns the number of cartridges in the library.
    #[must_use]
    pub fn cartridge_count(&self) -> usize {
        self.cartridges.len()
    }

    /// Returns all cartridges with a given status.
    #[must_use]
    pub fn cartridges_by_status(&self, status: TapeStatus) -> Vec<&TapeCartridge> {
        self.cartridges
            .iter()
            .filter(|c| c.status == status)
            .collect()
    }

    /// Returns the first writable cartridge with enough free space, if any.
    #[must_use]
    pub fn find_writable(&self, required_gb: f64) -> Option<&TapeCartridge> {
        self.cartridges
            .iter()
            .filter(|c| c.status.is_writable() && c.available_gb() >= required_gb)
            .min_by(|a, b| {
                a.available_gb()
                    .partial_cmp(&b.available_gb())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_format_capacity() {
        assert_eq!(TapeFormat::Lto7.capacity_gb(), 6_000);
        assert_eq!(TapeFormat::Lto8.capacity_gb(), 12_000);
        assert_eq!(TapeFormat::Lto9.capacity_gb(), 18_000);
        assert_eq!(TapeFormat::Lto10.capacity_gb(), 36_000);
    }

    #[test]
    fn test_tape_format_compressed_capacity() {
        assert_eq!(TapeFormat::Lto9.compressed_capacity_gb(), 45_000);
        assert_eq!(TapeFormat::Lto10.compressed_capacity_gb(), 90_000);
    }

    #[test]
    fn test_tape_format_speed() {
        assert!((TapeFormat::Lto7.native_speed_mbps() - 300.0).abs() < 1e-9);
        assert!((TapeFormat::Lto10.native_speed_mbps() - 900.0).abs() < 1e-9);
    }

    #[test]
    fn test_tape_format_generation() {
        assert_eq!(TapeFormat::Lto7.generation(), 7);
        assert_eq!(TapeFormat::Lto10.generation(), 10);
    }

    #[test]
    fn test_tape_status_writable() {
        assert!(TapeStatus::Blank.is_writable());
        assert!(TapeStatus::Active.is_writable());
        assert!(TapeStatus::Scratch.is_writable());
        assert!(!TapeStatus::Full.is_writable());
        assert!(!TapeStatus::Expired.is_writable());
    }

    #[test]
    fn test_cartridge_new_is_blank() {
        let c = TapeCartridge::new("TAPE001L9", TapeFormat::Lto9);
        assert_eq!(c.status, TapeStatus::Blank);
        assert_eq!(c.used_gb, 0.0);
    }

    #[test]
    fn test_cartridge_available_gb() {
        let mut c = TapeCartridge::new("T001", TapeFormat::Lto8);
        c.used_gb = 5_000.0;
        assert!((c.available_gb() - 7_000.0).abs() < 0.01);
    }

    #[test]
    fn test_cartridge_utilization_pct() {
        let mut c = TapeCartridge::new("T001", TapeFormat::Lto8);
        c.used_gb = 6_000.0;
        assert!((c.utilization_pct() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_cartridge_write_time() {
        let c = TapeCartridge::new("T001", TapeFormat::Lto9);
        // 18 TB / (400 MB/s) = 18_000 GB / (400/1024 GB/s) ≈ 46080 s
        let t = c.write_time_s(18_000.0);
        assert!(t > 40_000.0 && t < 50_000.0);
    }

    #[test]
    fn test_cartridge_record_write_transitions_to_active() {
        let mut c = TapeCartridge::new("T001", TapeFormat::Lto9);
        c.record_write(1_000.0);
        assert_eq!(c.status, TapeStatus::Active);
        assert!((c.used_gb - 1_000.0).abs() < 0.01);
    }

    #[test]
    fn test_library_add_and_count() {
        let mut lib = TapeLibrary::new(2);
        lib.add_cartridge(TapeCartridge::new("T001", TapeFormat::Lto9));
        lib.add_cartridge(TapeCartridge::new("T002", TapeFormat::Lto9));
        assert_eq!(lib.cartridge_count(), 2);
    }

    #[test]
    fn test_library_find_cartridge() {
        let mut lib = TapeLibrary::new(1);
        lib.add_cartridge(TapeCartridge::new("FINDME", TapeFormat::Lto8));
        assert!(lib.find_cartridge("FINDME").is_some());
        assert!(lib.find_cartridge("NOTHERE").is_none());
    }

    #[test]
    fn test_library_available_capacity() {
        let mut lib = TapeLibrary::new(1);
        lib.add_cartridge(TapeCartridge::new("T001", TapeFormat::Lto9));
        lib.add_cartridge(TapeCartridge::new("T002", TapeFormat::Lto9));
        // Both blank → 2 × 18_000 GB
        assert_eq!(lib.available_capacity_gb(), 36_000.0);
    }

    #[test]
    fn test_library_find_writable() {
        let mut lib = TapeLibrary::new(1);
        let mut c = TapeCartridge::new("T001", TapeFormat::Lto9);
        c.status = TapeStatus::Full;
        lib.add_cartridge(c);
        lib.add_cartridge(TapeCartridge::new("T002", TapeFormat::Lto9));
        let w = lib.find_writable(100.0);
        assert!(w.is_some());
        assert_eq!(w.expect("test expectation failed").barcode, "T002");
    }
}
