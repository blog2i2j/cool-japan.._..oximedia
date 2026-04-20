//! Parallel encoding for multiple outputs simultaneously.

use crate::{Result, TranscodeConfig, TranscodeError, TranscodeOutput};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// Configuration for parallel encoding.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Maximum number of parallel encodes.
    pub max_parallel: usize,
    /// CPU cores to use per encode.
    pub cores_per_encode: Option<usize>,
    /// Whether to use thread pools.
    pub use_thread_pool: bool,
    /// Priority for parallel jobs.
    pub priority: ParallelPriority,
}

/// Priority levels for parallel jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelPriority {
    /// Low priority (background processing).
    Low,
    /// Normal priority.
    Normal,
    /// High priority (time-sensitive).
    High,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_parallel: num_cpus(),
            cores_per_encode: None,
            use_thread_pool: true,
            priority: ParallelPriority::Normal,
        }
    }
}

impl ParallelConfig {
    /// Creates a new parallel config with automatic core detection.
    #[must_use]
    pub fn auto() -> Self {
        Self::default()
    }

    /// Creates a config with a specific number of parallel jobs.
    #[must_use]
    pub fn with_max_parallel(max: usize) -> Self {
        Self {
            max_parallel: max,
            ..Self::default()
        }
    }

    /// Sets the number of cores per encode job.
    #[must_use]
    pub fn cores_per_encode(mut self, cores: usize) -> Self {
        self.cores_per_encode = Some(cores);
        self
    }

    /// Sets the priority level.
    #[must_use]
    pub fn priority(mut self, priority: ParallelPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn validate(&self) -> Result<()> {
        if self.max_parallel == 0 {
            return Err(TranscodeError::ValidationError(
                crate::ValidationError::Unsupported(
                    "max_parallel must be greater than 0".to_string(),
                ),
            ));
        }

        if let Some(cores) = self.cores_per_encode {
            if cores == 0 {
                return Err(TranscodeError::ValidationError(
                    crate::ValidationError::Unsupported(
                        "cores_per_encode must be greater than 0".to_string(),
                    ),
                ));
            }
        }

        Ok(())
    }
}

/// Gets the number of CPU cores available.
///
/// Falls back to 4 if the system query fails.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(4) // unwrap_or is safe — this is a fallback, not unwrap()
}

/// Parallel encoder for processing multiple outputs simultaneously.
pub struct ParallelEncoder {
    config: ParallelConfig,
    jobs: Vec<TranscodeConfig>,
    results: Arc<Mutex<Vec<Result<TranscodeOutput>>>>,
}

impl ParallelEncoder {
    /// Creates a new parallel encoder with the given configuration.
    #[must_use]
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            jobs: Vec::new(),
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Adds a job to the parallel encoder.
    pub fn add_job(&mut self, job: TranscodeConfig) {
        self.jobs.push(job);
    }

    /// Adds multiple jobs at once.
    pub fn add_jobs(&mut self, jobs: Vec<TranscodeConfig>) {
        self.jobs.extend(jobs);
    }

    /// Gets the number of jobs queued.
    #[must_use]
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Executes all jobs in parallel.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid. Individual job errors
    /// are captured in the results.
    pub async fn execute_all(&mut self) -> Result<Vec<Result<TranscodeOutput>>> {
        self.config.validate()?;

        // Configure thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.max_parallel)
            .build()
            .map_err(|e| {
                TranscodeError::PipelineError(format!("Failed to create thread pool: {e}"))
            })?;

        let jobs = std::mem::take(&mut self.jobs);

        // Execute jobs in parallel and collect results directly.
        let job_results: Vec<Result<TranscodeOutput>> = pool.install(|| {
            jobs.into_par_iter()
                .map(Self::execute_job)
                .collect::<Vec<_>>()
        });

        // Store results for later retrieval.
        match self.results.lock() {
            Ok(mut guard) => {
                guard.extend(job_results.iter().cloned());
            }
            Err(poisoned) => {
                poisoned.into_inner().extend(job_results.iter().cloned());
            }
        }

        Ok(job_results)
    }

    /// Executes all jobs sequentially (for debugging).
    ///
    /// # Errors
    ///
    /// Returns an error if any job fails.
    pub async fn execute_sequential(&mut self) -> Result<Vec<TranscodeOutput>> {
        let mut outputs = Vec::new();

        for job in &self.jobs {
            let output = Self::execute_job(job.clone())?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Executes a single transcode job synchronously.
    ///
    /// Validates the job configuration and then delegates to the
    /// pipeline builder for actual transcoding. The pipeline is
    /// executed on a per-thread tokio runtime so that async I/O
    /// works within the rayon thread pool.
    #[cfg(not(target_arch = "wasm32"))]
    fn execute_job(job: TranscodeConfig) -> Result<TranscodeOutput> {
        let input = job
            .input
            .as_deref()
            .ok_or_else(|| TranscodeError::InvalidInput("No input file specified".to_string()))?;

        let output = job
            .output
            .as_deref()
            .ok_or_else(|| TranscodeError::InvalidOutput("No output file specified".to_string()))?;

        // Build a pipeline from the job config.
        let mut pipeline_builder = crate::pipeline::TranscodePipelineBuilder::new()
            .input(input)
            .output(output);

        if let Some(ref vc) = job.video_codec {
            pipeline_builder = pipeline_builder.video_codec(vc);
        }
        if let Some(ref ac) = job.audio_codec {
            pipeline_builder = pipeline_builder.audio_codec(ac);
        }
        if let Some(mode) = job.multi_pass {
            pipeline_builder = pipeline_builder.multipass(mode);
        }

        let mut pipeline = pipeline_builder.build()?;

        // Create a per-thread tokio runtime to drive the async pipeline.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                TranscodeError::PipelineError(format!("Failed to create async runtime: {e}"))
            })?;

        rt.block_on(pipeline.execute())
    }

    /// Executes a single transcode job synchronously (wasm32 stub).
    #[cfg(target_arch = "wasm32")]
    fn execute_job(_job: TranscodeConfig) -> Result<TranscodeOutput> {
        Err(TranscodeError::Unsupported(
            "Parallel job execution is not supported on wasm32".to_string(),
        ))
    }

    /// Gets the results of completed jobs.
    #[must_use]
    pub fn get_results(&self) -> Vec<Result<TranscodeOutput>> {
        match self.results.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// Clears all jobs and results.
    pub fn clear(&mut self) {
        self.jobs.clear();
        match self.results.lock() {
            Ok(mut guard) => guard.clear(),
            Err(poisoned) => poisoned.into_inner().clear(),
        }
    }
}

// ─── AV1 tile-based parallel encoding ────────────────────────────────────────

/// Configuration for AV1 tile-based parallel encoding.
///
/// AV1 supports dividing each frame into an `M × N` grid of independently
/// encodable tiles.  Encoding multiple tiles in parallel with rayon reduces
/// wall-clock time on multi-core systems while producing a bitstream that is
/// backward-compatible with single-threaded decoders.
///
/// `tile_cols` and `tile_rows` are specified as **log2** values following the
/// AV1 specification (e.g. 2 → 4 tile columns).  Use [`Av1TileConfig::auto`]
/// to pick sensible defaults for a given resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1TileConfig {
    /// Log2 number of tile columns (0–6).
    pub tile_cols_log2: u8,
    /// Log2 number of tile rows (0–6).
    pub tile_rows_log2: u8,
    /// Number of rayon worker threads (0 = auto).
    pub threads: usize,
    /// Enable row-based multi-threading within each tile.
    pub row_mt: bool,
}

impl Default for Av1TileConfig {
    fn default() -> Self {
        Self {
            tile_cols_log2: 1, // 2 columns
            tile_rows_log2: 1, // 2 rows
            threads: 0,
            row_mt: true,
        }
    }
}

impl Av1TileConfig {
    /// Creates a new AV1 tile configuration with explicit log2 tile counts.
    ///
    /// # Errors
    ///
    /// Returns an error when `tile_cols_log2` or `tile_rows_log2` exceeds 6.
    pub fn new(tile_cols_log2: u8, tile_rows_log2: u8, threads: usize) -> Result<Self> {
        if tile_cols_log2 > 6 {
            return Err(TranscodeError::ValidationError(
                crate::ValidationError::Unsupported(format!(
                    "tile_cols_log2 must be 0–6, got {tile_cols_log2}"
                )),
            ));
        }
        if tile_rows_log2 > 6 {
            return Err(TranscodeError::ValidationError(
                crate::ValidationError::Unsupported(format!(
                    "tile_rows_log2 must be 0–6, got {tile_rows_log2}"
                )),
            ));
        }
        Ok(Self {
            tile_cols_log2,
            tile_rows_log2,
            threads,
            row_mt: true,
        })
    }

    /// Automatically selects tile counts appropriate for the given resolution.
    ///
    /// | Resolution    | tile_cols_log2 | tile_rows_log2 |
    /// |---------------|----------------|----------------|
    /// | ≤ 720p        | 1 (2 cols)     | 0 (1 row)      |
    /// | 1080p         | 1 (2 cols)     | 1 (2 rows)     |
    /// | 4K (2160p)    | 2 (4 cols)     | 2 (4 rows)     |
    /// | 8K (4320p)    | 3 (8 cols)     | 2 (4 rows)     |
    #[must_use]
    pub fn auto(_width: u32, height: u32, threads: usize) -> Self {
        let (cols_log2, rows_log2) = if height <= 720 {
            (1, 0)
        } else if height <= 1080 {
            (1, 1)
        } else if height <= 2160 {
            (2, 2)
        } else {
            (3, 2)
        };
        Self {
            tile_cols_log2: cols_log2,
            tile_rows_log2: rows_log2,
            threads,
            row_mt: true,
        }
    }

    /// Returns the actual number of tile columns (2^tile_cols_log2).
    #[must_use]
    pub fn tile_cols(&self) -> u32 {
        1u32 << self.tile_cols_log2
    }

    /// Returns the actual number of tile rows (2^tile_rows_log2).
    #[must_use]
    pub fn tile_rows(&self) -> u32 {
        1u32 << self.tile_rows_log2
    }

    /// Returns the total number of tiles per frame.
    #[must_use]
    pub fn total_tiles(&self) -> u32 {
        self.tile_cols() * self.tile_rows()
    }

    /// Validates the configuration against frame dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the tile grid would produce tiles smaller than
    /// 64×64 pixels (the AV1 minimum superblock size).
    pub fn validate_for_frame(&self, width: u32, height: u32) -> Result<()> {
        const MIN_TILE_DIM: u32 = 64;
        let tile_w = width / self.tile_cols();
        let tile_h = height / self.tile_rows();
        if tile_w < MIN_TILE_DIM || tile_h < MIN_TILE_DIM {
            return Err(TranscodeError::ValidationError(
                crate::ValidationError::Unsupported(format!(
                    "Tile grid {}×{} produces tiles {}×{} which is smaller than \
                     the AV1 minimum {}×{} pixels",
                    self.tile_cols(),
                    self.tile_rows(),
                    tile_w,
                    tile_h,
                    MIN_TILE_DIM,
                    MIN_TILE_DIM
                )),
            ));
        }
        Ok(())
    }
}

/// Statistics produced by a single AV1 tile encode pass.
#[derive(Debug, Clone, Default)]
pub struct Av1TileStats {
    /// Total number of tiles encoded.
    pub tiles_encoded: u32,
    /// Total compressed bytes produced across all tiles.
    pub compressed_bytes: u64,
    /// Wall-clock time in seconds.
    pub wall_time_secs: f64,
}

impl Av1TileStats {
    /// Estimated throughput in tiles per second.
    #[must_use]
    pub fn tiles_per_second(&self) -> f64 {
        if self.wall_time_secs > 0.0 {
            f64::from(self.tiles_encoded) / self.wall_time_secs
        } else {
            0.0
        }
    }

    /// Average compressed bytes per tile.
    #[must_use]
    pub fn avg_bytes_per_tile(&self) -> u64 {
        if self.tiles_encoded == 0 {
            return 0;
        }
        self.compressed_bytes / u64::from(self.tiles_encoded)
    }
}

/// Encoder that splits video frames into tiles and encodes each tile in
/// parallel using rayon.
///
/// The encoder works with raw RGBA or YUV420 frame data (stored as a flat
/// `Vec<u8>`).  Each tile is extracted from the source buffer, passed to
/// the configurable `Av1TileEncodeOp` trait, and the resulting bitstreams
/// are assembled into a single frame bitstream.
pub struct Av1TileParallelEncoder {
    tile_config: Av1TileConfig,
    frame_width: u32,
    frame_height: u32,
    stats: Av1TileStats,
}

impl Av1TileParallelEncoder {
    /// Creates a new tile encoder for frames of `frame_width × frame_height`.
    ///
    /// # Errors
    ///
    /// Returns an error when the tile configuration is incompatible with the
    /// frame dimensions.
    pub fn new(tile_config: Av1TileConfig, frame_width: u32, frame_height: u32) -> Result<Self> {
        tile_config.validate_for_frame(frame_width, frame_height)?;
        Ok(Self {
            tile_config,
            frame_width,
            frame_height,
            stats: Av1TileStats::default(),
        })
    }

    /// Returns the tile configuration.
    #[must_use]
    pub fn tile_config(&self) -> &Av1TileConfig {
        &self.tile_config
    }

    /// Returns collected encoding statistics.
    #[must_use]
    pub fn stats(&self) -> &Av1TileStats {
        &self.stats
    }

    /// Encodes one RGBA frame using tile-based rayon parallelism.
    ///
    /// The frame data must be a row-major RGBA buffer of exactly
    /// `frame_width * frame_height * 4` bytes.
    ///
    /// # Returns
    ///
    /// An assembled tile bitstream (simple concatenation with 4-byte LE length
    /// prefix per tile, as produced by [`assemble_av1_tile_bitstream`]).
    ///
    /// # Errors
    ///
    /// Returns an error when the frame buffer is too small.
    pub fn encode_frame_rgba(&mut self, rgba: &[u8]) -> Result<Vec<u8>> {
        let expected = (self.frame_width * self.frame_height * 4) as usize;
        if rgba.len() < expected {
            return Err(TranscodeError::CodecError(format!(
                "RGBA buffer too small: got {} bytes, need {}",
                rgba.len(),
                expected
            )));
        }

        let start = std::time::Instant::now();
        let tile_cols = self.tile_config.tile_cols();
        let tile_rows = self.tile_config.tile_rows();

        let tile_w = self.frame_width / tile_cols;
        let tile_h = self.frame_height / tile_rows;

        // Build list of (tile_col, tile_row) pairs for parallel iteration.
        let coords: Vec<(u32, u32)> = (0..tile_rows)
            .flat_map(|row| (0..tile_cols).map(move |col| (col, row)))
            .collect();

        // Extract and compress each tile in parallel.
        let tile_bitstreams: Vec<(usize, Vec<u8>)> = {
            use rayon::prelude::*;

            let frame_width = self.frame_width;

            coords
                .par_iter()
                .enumerate()
                .map(|(idx, &(col, row))| {
                    let x_start = col * tile_w;
                    let y_start = row * tile_h;

                    // Extract tile RGBA into a contiguous buffer.
                    let mut tile_buf = Vec::with_capacity((tile_w * tile_h * 4) as usize);
                    for ty in 0..tile_h {
                        let src_row = y_start + ty;
                        let src_start = ((src_row * frame_width + x_start) * 4) as usize;
                        let src_end = src_start + (tile_w * 4) as usize;
                        if src_end <= rgba.len() {
                            tile_buf.extend_from_slice(&rgba[src_start..src_end]);
                        }
                    }

                    // Compress tile: simple RLE on luma bytes as a proxy for
                    // a real AV1 tile encode (full codec integration requires
                    // the AV1 encoder stack; this provides the parallelism
                    // scaffold and correct bitstream assembly).
                    let compressed = compress_tile_placeholder(&tile_buf);

                    (idx, compressed)
                })
                .collect()
        };

        let compressed_total: u64 = tile_bitstreams.iter().map(|(_, b)| b.len() as u64).sum();

        self.stats.tiles_encoded += tile_bitstreams.len() as u32;
        self.stats.compressed_bytes += compressed_total;
        self.stats.wall_time_secs += start.elapsed().as_secs_f64();

        Ok(assemble_av1_tile_bitstream(tile_bitstreams))
    }

    /// Resets the accumulated statistics.
    pub fn reset_stats(&mut self) {
        self.stats = Av1TileStats::default();
    }
}

/// Assembles per-tile bitstreams into a single frame bitstream.
///
/// Format: `[tile_count: u32 LE] ([tile_idx: u32 LE] [byte_len: u32 LE]
/// [data …]) …`
#[must_use]
pub fn assemble_av1_tile_bitstream(tiles: Vec<(usize, Vec<u8>)>) -> Vec<u8> {
    let mut out = Vec::new();
    // Write tile count header.
    out.extend_from_slice(&(tiles.len() as u32).to_le_bytes());

    // Write tiles in index order.
    let mut sorted = tiles;
    sorted.sort_by_key(|(idx, _)| *idx);

    for (idx, data) in sorted {
        out.extend_from_slice(&(idx as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&data);
    }

    out
}

/// Placeholder tile compressor: simple byte-pair run-length encoding of the
/// luma bytes (R channel in RGBA), yielding a compact but easily decodable
/// representation.  In production this would call the AV1 tile encoder from
/// `oximedia-codec`.
fn compress_tile_placeholder(rgba: &[u8]) -> Vec<u8> {
    if rgba.is_empty() {
        return Vec::new();
    }
    // Extract every 4th byte (R channel as luma proxy).
    let luma: Vec<u8> = rgba.iter().step_by(4).copied().collect();

    // RLE: pairs of (value, run_len) where run_len is 1-byte capped at 255.
    let mut out = Vec::with_capacity(luma.len());
    let mut i = 0;
    while i < luma.len() {
        let val = luma[i];
        let mut run: u8 = 1;
        while i + usize::from(run) < luma.len() && luma[i + usize::from(run)] == val && run < 255 {
            run += 1;
        }
        out.push(val);
        out.push(run);
        i += usize::from(run);
    }
    out
}

/// Builder for creating parallel encode jobs.
pub struct ParallelEncodeBuilder {
    config: ParallelConfig,
    jobs: Vec<TranscodeConfig>,
}

impl ParallelEncodeBuilder {
    /// Creates a new parallel encode builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ParallelConfig::default(),
            jobs: Vec::new(),
        }
    }

    /// Sets the maximum number of parallel jobs.
    #[must_use]
    pub fn max_parallel(mut self, max: usize) -> Self {
        self.config.max_parallel = max;
        self
    }

    /// Sets cores per encode job.
    #[must_use]
    pub fn cores_per_encode(mut self, cores: usize) -> Self {
        self.config.cores_per_encode = Some(cores);
        self
    }

    /// Sets the priority level.
    #[must_use]
    pub fn priority(mut self, priority: ParallelPriority) -> Self {
        self.config.priority = priority;
        self
    }

    /// Adds a job to the builder.
    #[must_use]
    pub fn add_job(mut self, job: TranscodeConfig) -> Self {
        self.jobs.push(job);
        self
    }

    /// Adds multiple jobs.
    #[must_use]
    pub fn add_jobs(mut self, jobs: Vec<TranscodeConfig>) -> Self {
        self.jobs.extend(jobs);
        self
    }

    /// Builds the parallel encoder.
    #[must_use]
    pub fn build(self) -> ParallelEncoder {
        let mut encoder = ParallelEncoder::new(self.config);
        encoder.add_jobs(self.jobs);
        encoder
    }
}

impl Default for ParallelEncodeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-transcode-parallel-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert!(config.max_parallel > 0);
        assert_eq!(config.priority, ParallelPriority::Normal);
        assert!(config.use_thread_pool);
    }

    #[test]
    fn test_parallel_config_validation() {
        let valid = ParallelConfig::with_max_parallel(4);
        assert!(valid.validate().is_ok());

        let invalid = ParallelConfig {
            max_parallel: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_parallel_config_cores_validation() {
        let valid = ParallelConfig::default().cores_per_encode(2);
        assert!(valid.validate().is_ok());

        let invalid = ParallelConfig::default().cores_per_encode(0);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_parallel_encoder_job_count() {
        let mut encoder = ParallelEncoder::new(ParallelConfig::default());
        assert_eq!(encoder.job_count(), 0);

        let job = TranscodeConfig {
            input: Some(tmp_str("input.mp4")),
            output: Some(tmp_str("output.mp4")),
            ..Default::default()
        };

        encoder.add_job(job);
        assert_eq!(encoder.job_count(), 1);
    }

    #[test]
    fn test_parallel_encoder_add_jobs() {
        let mut encoder = ParallelEncoder::new(ParallelConfig::default());

        let jobs = vec![
            TranscodeConfig {
                input: Some(tmp_str("input1.mp4")),
                output: Some(tmp_str("output1.mp4")),
                ..Default::default()
            },
            TranscodeConfig {
                input: Some(tmp_str("input2.mp4")),
                output: Some(tmp_str("output2.mp4")),
                ..Default::default()
            },
        ];

        encoder.add_jobs(jobs);
        assert_eq!(encoder.job_count(), 2);
    }

    #[test]
    fn test_parallel_encoder_clear() {
        let mut encoder = ParallelEncoder::new(ParallelConfig::default());

        let job = TranscodeConfig {
            input: Some(tmp_str("input.mp4")),
            output: Some(tmp_str("output.mp4")),
            ..Default::default()
        };

        encoder.add_job(job);
        assert_eq!(encoder.job_count(), 1);

        encoder.clear();
        assert_eq!(encoder.job_count(), 0);
    }

    #[test]
    fn test_parallel_builder() {
        let job = TranscodeConfig {
            input: Some(tmp_str("input.mp4")),
            output: Some(tmp_str("output.mp4")),
            ..Default::default()
        };

        let encoder = ParallelEncodeBuilder::new()
            .max_parallel(4)
            .cores_per_encode(2)
            .priority(ParallelPriority::High)
            .add_job(job)
            .build();

        assert_eq!(encoder.config.max_parallel, 4);
        assert_eq!(encoder.config.cores_per_encode, Some(2));
        assert_eq!(encoder.config.priority, ParallelPriority::High);
        assert_eq!(encoder.job_count(), 1);
    }

    #[test]
    fn test_num_cpus() {
        let cpus = num_cpus();
        assert!(cpus > 0);
        assert!(cpus <= 1024); // Reasonable upper bound
    }

    // ── Av1TileConfig tests ───────────────────────────────────────────────────

    #[test]
    fn test_av1_tile_config_default() {
        let cfg = Av1TileConfig::default();
        assert_eq!(cfg.tile_cols(), 2);
        assert_eq!(cfg.tile_rows(), 2);
        assert_eq!(cfg.total_tiles(), 4);
        assert!(cfg.row_mt);
    }

    #[test]
    fn test_av1_tile_config_new_valid() {
        let cfg = Av1TileConfig::new(2, 1, 4).expect("valid config");
        assert_eq!(cfg.tile_cols(), 4);
        assert_eq!(cfg.tile_rows(), 2);
        assert_eq!(cfg.total_tiles(), 8);
    }

    #[test]
    fn test_av1_tile_config_new_invalid_cols() {
        let result = Av1TileConfig::new(7, 1, 0);
        assert!(result.is_err(), "log2 > 6 should fail");
    }

    #[test]
    fn test_av1_tile_config_new_invalid_rows() {
        let result = Av1TileConfig::new(1, 7, 0);
        assert!(result.is_err(), "log2 > 6 should fail");
    }

    #[test]
    fn test_av1_tile_config_auto_720p() {
        let cfg = Av1TileConfig::auto(1280, 720, 4);
        assert_eq!(cfg.tile_cols_log2, 1);
        assert_eq!(cfg.tile_rows_log2, 0);
    }

    #[test]
    fn test_av1_tile_config_auto_1080p() {
        let cfg = Av1TileConfig::auto(1920, 1080, 4);
        assert_eq!(cfg.tile_cols_log2, 1);
        assert_eq!(cfg.tile_rows_log2, 1);
    }

    #[test]
    fn test_av1_tile_config_auto_4k() {
        let cfg = Av1TileConfig::auto(3840, 2160, 8);
        assert_eq!(cfg.tile_cols_log2, 2);
        assert_eq!(cfg.tile_rows_log2, 2);
    }

    #[test]
    fn test_av1_tile_config_validate_ok() {
        let cfg = Av1TileConfig::new(1, 1, 0).expect("valid");
        // 1920×1080 with 2×2 tiles → 960×540 each — above 64×64 minimum
        assert!(cfg.validate_for_frame(1920, 1080).is_ok());
    }

    #[test]
    fn test_av1_tile_config_validate_too_small() {
        let cfg = Av1TileConfig::new(3, 3, 0).expect("valid config");
        // 256×256 with 8×8 tiles → 32×32 each — below 64×64 minimum
        assert!(cfg.validate_for_frame(256, 256).is_err());
    }

    #[test]
    fn test_av1_tile_parallel_encoder_encode_frame() {
        // 512×512 RGBA frame; 2×2 tile grid (each tile 256×256)
        let cfg = Av1TileConfig::new(1, 1, 2).expect("valid");
        let mut encoder = Av1TileParallelEncoder::new(cfg, 512, 512).expect("encoder ok");

        let frame_data = vec![128u8; 512 * 512 * 4]; // grey RGBA
        let bitstream = encoder.encode_frame_rgba(&frame_data).expect("encode ok");

        // Bitstream must contain the 4-byte tile count header + at least 4 tiles.
        assert!(bitstream.len() >= 4, "bitstream should have header");
        let tile_count =
            u32::from_le_bytes([bitstream[0], bitstream[1], bitstream[2], bitstream[3]]);
        assert_eq!(tile_count, 4, "should encode 4 tiles");

        // Stats should be updated.
        assert_eq!(encoder.stats().tiles_encoded, 4);
        assert!(encoder.stats().compressed_bytes > 0);
    }

    #[test]
    fn test_av1_tile_parallel_encoder_undersized_frame() {
        let cfg = Av1TileConfig::default();
        let mut encoder = Av1TileParallelEncoder::new(cfg, 256, 256).expect("encoder ok");

        // Provide only 1 byte — way too small.
        let result = encoder.encode_frame_rgba(&[0u8]);
        assert!(result.is_err(), "undersized frame should fail");
    }

    #[test]
    fn test_av1_tile_parallel_encoder_stats_reset() {
        let cfg = Av1TileConfig::new(1, 1, 2).expect("valid");
        let mut encoder = Av1TileParallelEncoder::new(cfg, 256, 256).expect("encoder ok");

        let frame_data = vec![0u8; 256 * 256 * 4];
        encoder.encode_frame_rgba(&frame_data).expect("encode ok");
        assert!(encoder.stats().tiles_encoded > 0);

        encoder.reset_stats();
        assert_eq!(encoder.stats().tiles_encoded, 0);
        assert_eq!(encoder.stats().compressed_bytes, 0);
    }

    #[test]
    fn test_av1_tile_stats_tiles_per_second() {
        let stats = Av1TileStats {
            tiles_encoded: 100,
            compressed_bytes: 50_000,
            wall_time_secs: 2.0,
        };
        assert!((stats.tiles_per_second() - 50.0).abs() < 1e-9);
        assert_eq!(stats.avg_bytes_per_tile(), 500);
    }

    #[test]
    fn test_av1_tile_stats_zero_time() {
        let stats = Av1TileStats::default();
        assert!((stats.tiles_per_second()).abs() < 1e-9);
        assert_eq!(stats.avg_bytes_per_tile(), 0);
    }

    #[test]
    fn test_assemble_av1_tile_bitstream_order() {
        // Tiles arrive out of order; assembled bitstream must sort them.
        let tiles = vec![(1, vec![1u8, 2, 3]), (0, vec![4u8, 5, 6])];
        let bs = assemble_av1_tile_bitstream(tiles);

        // Count = 2
        let count = u32::from_le_bytes([bs[0], bs[1], bs[2], bs[3]]);
        assert_eq!(count, 2);

        // First tile entry's index should be 0.
        let idx0 = u32::from_le_bytes([bs[4], bs[5], bs[6], bs[7]]);
        assert_eq!(idx0, 0);
    }

    #[test]
    fn test_compress_tile_placeholder_empty() {
        let result = compress_tile_placeholder(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_compress_tile_placeholder_rle() {
        // 4 identical RGBA pixels (luma = 200) → single run of 4
        let rgba = vec![
            200u8, 0, 0, 255, 200, 0, 0, 255, 200, 0, 0, 255, 200, 0, 0, 255,
        ];
        let compressed = compress_tile_placeholder(&rgba);
        // Should produce [200, 4] (one run of length 4)
        assert_eq!(compressed, vec![200, 4]);
    }
}
