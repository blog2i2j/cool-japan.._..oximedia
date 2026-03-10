//! Parallel processing optimization strategies for `OxiMedia`.
//!
//! Provides work splitting, chunk management, and theoretical speedup analysis
//! (Amdahl's Law) for parallel encoding and decoding workloads.

#![allow(dead_code)]

/// A contiguous range of work items for parallel processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkChunk {
    /// Unique identifier for this chunk.
    pub id: u64,
    /// First item index (inclusive).
    pub start: u64,
    /// Last item index (exclusive).
    pub end: u64,
    /// Scheduling priority (higher = more urgent).
    pub priority: u32,
}

impl WorkChunk {
    /// Returns the number of work items in this chunk.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Returns `true` if this chunk's range overlaps with `other`.
    #[must_use]
    pub fn overlaps(&self, other: &WorkChunk) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// Strategy for dividing work into chunks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkStrategy {
    /// Each chunk has exactly N items.
    Fixed(u64),
    /// Chunk size adjusts at runtime based on worker throughput.
    Dynamic,
    /// Chunk size adapts based on measured load.
    Adaptive,
}

impl ChunkStrategy {
    /// Returns the initial chunk size for the given total and worker count.
    #[must_use]
    pub fn initial_chunk_size(&self, total: u64, workers: u32) -> u64 {
        if workers == 0 {
            return total;
        }
        match self {
            Self::Fixed(n) => (*n).max(1),
            Self::Dynamic | Self::Adaptive => (total / workers as u64).max(1),
        }
    }
}

/// Splits work into chunks and merges small ones.
#[derive(Debug, Clone)]
pub struct WorkSplitter {
    /// The splitting strategy to use.
    pub strategy: ChunkStrategy,
}

impl WorkSplitter {
    /// Splits `total` items across `workers` workers, returning a list of `WorkChunk`s.
    ///
    /// Chunks are numbered starting at 0.  The last chunk absorbs any remainder.
    #[must_use]
    pub fn split(&self, total: u64, workers: u32) -> Vec<WorkChunk> {
        if total == 0 || workers == 0 {
            return vec![];
        }
        let chunk_size = self.strategy.initial_chunk_size(total, workers);
        let mut chunks = Vec::new();
        let mut start = 0u64;
        let mut id = 0u64;
        while start < total {
            let end = (start + chunk_size).min(total);
            chunks.push(WorkChunk {
                id,
                start,
                end,
                priority: 0,
            });
            start = end;
            id += 1;
        }
        chunks
    }

    /// Merges adjacent chunks that are smaller than `min_size`.
    ///
    /// Small chunks are merged with their right neighbour. Priority of the
    /// merged chunk is the maximum of the two.
    #[must_use]
    pub fn merge_small_chunks(&self, chunks: Vec<WorkChunk>, min_size: u64) -> Vec<WorkChunk> {
        if chunks.is_empty() {
            return chunks;
        }
        let mut result: Vec<WorkChunk> = Vec::new();
        for chunk in chunks {
            if let Some(last) = result.last_mut() {
                if last.size() < min_size {
                    last.end = chunk.end;
                    last.priority = last.priority.max(chunk.priority);
                    continue;
                }
            }
            result.push(chunk);
        }
        result
    }
}

/// Amdahl's Law analysis for parallel speedup estimation.
///
/// `serial_fraction` is the fraction of work that is inherently serial (0–1).
#[derive(Debug, Clone)]
pub struct Amdahl {
    /// Fraction of the workload that is serial (cannot be parallelized).
    pub serial_fraction: f64,
}

impl Amdahl {
    /// Theoretical speedup with `num_processors` parallel processors.
    ///
    /// S(n) = 1 / (serial_fraction + (1 − serial_fraction) / n)
    #[must_use]
    pub fn speedup(&self, num_processors: f64) -> f64 {
        if num_processors <= 0.0 {
            return 0.0;
        }
        let p = self.serial_fraction.max(0.0).min(1.0);
        1.0 / (p + (1.0 - p) / num_processors)
    }

    /// Parallel efficiency: speedup / num_processors.
    #[must_use]
    pub fn efficiency(&self, n: f64) -> f64 {
        if n <= 0.0 {
            return 0.0;
        }
        self.speedup(n) / n
    }

    /// The "optimal" processor count where efficiency drops below 50 %.
    ///
    /// Approximated as 2 / serial_fraction (returns f64::MAX if serial_fraction ≈ 0).
    #[must_use]
    pub fn optimal_processors(&self) -> f64 {
        let p = self.serial_fraction.max(f64::EPSILON);
        2.0 / p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_chunk_size() {
        let c = WorkChunk {
            id: 0,
            start: 10,
            end: 30,
            priority: 0,
        };
        assert_eq!(c.size(), 20);
    }

    #[test]
    fn test_work_chunk_size_zero_when_end_le_start() {
        let c = WorkChunk {
            id: 0,
            start: 5,
            end: 5,
            priority: 0,
        };
        assert_eq!(c.size(), 0);
    }

    #[test]
    fn test_work_chunk_overlaps_true() {
        let a = WorkChunk {
            id: 0,
            start: 0,
            end: 10,
            priority: 0,
        };
        let b = WorkChunk {
            id: 1,
            start: 5,
            end: 15,
            priority: 0,
        };
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_work_chunk_overlaps_false() {
        let a = WorkChunk {
            id: 0,
            start: 0,
            end: 10,
            priority: 0,
        };
        let b = WorkChunk {
            id: 1,
            start: 10,
            end: 20,
            priority: 0,
        };
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_chunk_strategy_fixed_size() {
        let s = ChunkStrategy::Fixed(100);
        assert_eq!(s.initial_chunk_size(1000, 4), 100);
    }

    #[test]
    fn test_chunk_strategy_dynamic_divides_evenly() {
        let s = ChunkStrategy::Dynamic;
        assert_eq!(s.initial_chunk_size(1000, 4), 250);
    }

    #[test]
    fn test_chunk_strategy_adaptive_minimum_one() {
        let s = ChunkStrategy::Adaptive;
        assert_eq!(s.initial_chunk_size(1, 10), 1);
    }

    #[test]
    fn test_work_splitter_covers_all_items() {
        let splitter = WorkSplitter {
            strategy: ChunkStrategy::Fixed(100),
        };
        let chunks = splitter.split(350, 4);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks.last().expect("last element should exist").end, 350);
    }

    #[test]
    fn test_work_splitter_empty_on_zero_total() {
        let splitter = WorkSplitter {
            strategy: ChunkStrategy::Dynamic,
        };
        let chunks = splitter.split(0, 4);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_work_splitter_no_overlaps() {
        let splitter = WorkSplitter {
            strategy: ChunkStrategy::Fixed(50),
        };
        let chunks = splitter.split(200, 4);
        for i in 0..chunks.len() {
            for j in (i + 1)..chunks.len() {
                assert!(!chunks[i].overlaps(&chunks[j]));
            }
        }
    }

    #[test]
    fn test_merge_small_chunks() {
        let splitter = WorkSplitter {
            strategy: ChunkStrategy::Fixed(10),
        };
        let chunks = splitter.split(100, 10);
        // Each chunk is exactly 10, so merging with min_size 5 → no merging
        let merged = splitter.merge_small_chunks(chunks.clone(), 5);
        assert_eq!(merged.len(), chunks.len());
    }

    #[test]
    fn test_merge_small_chunks_merges() {
        let chunks = vec![
            WorkChunk {
                id: 0,
                start: 0,
                end: 3,
                priority: 1,
            },
            WorkChunk {
                id: 1,
                start: 3,
                end: 10,
                priority: 2,
            },
            WorkChunk {
                id: 2,
                start: 10,
                end: 20,
                priority: 0,
            },
        ];
        let splitter = WorkSplitter {
            strategy: ChunkStrategy::Fixed(1),
        };
        let merged = splitter.merge_small_chunks(chunks, 5);
        // First chunk (size 3) < 5 → merged into second
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].start, 0);
        assert_eq!(merged[0].end, 10);
        assert_eq!(merged[0].priority, 2); // max(1,2)
    }

    #[test]
    fn test_amdahl_speedup_single_processor() {
        let a = Amdahl {
            serial_fraction: 0.1,
        };
        assert!((a.speedup(1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_amdahl_speedup_infinite_processors() {
        let a = Amdahl {
            serial_fraction: 0.1,
        };
        // S(∞) ≈ 1 / serial_fraction = 10
        let s = a.speedup(1_000_000.0);
        assert!((s - 10.0).abs() < 0.01, "got {s}");
    }

    #[test]
    fn test_amdahl_efficiency_decreases_with_more_processors() {
        let a = Amdahl {
            serial_fraction: 0.2,
        };
        let e4 = a.efficiency(4.0);
        let e8 = a.efficiency(8.0);
        assert!(e8 < e4, "efficiency should decrease with more processors");
    }

    #[test]
    fn test_amdahl_optimal_processors() {
        let a = Amdahl {
            serial_fraction: 0.1,
        };
        let opt = a.optimal_processors();
        assert!(opt > 0.0);
        assert!((opt - 20.0).abs() < 1e-9);
    }
}
