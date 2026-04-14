"""
Type stubs for the ``oximedia`` Python extension module.

These stubs enable IDE autocomplete and static type checking (mypy, pyright,
pylance) for the OxiMedia Python bindings.  They are maintained manually and
should be kept in sync with the Rust source in ``crates/oximedia-py/src/``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

class PixelFormat:
    """Pixel format for video frames (e.g. YUV 4:2:0)."""

    YUV420P: str
    YUV422P: str
    YUV444P: str
    GRAY8: str

    def __init__(self, format: str) -> None: ...
    def is_planar(self) -> bool: ...
    def plane_count(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class SampleFormat:
    """Audio sample format (e.g. 32-bit float, 16-bit signed integer)."""

    F32: str
    I16: str

    def __init__(self, format: str) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class ChannelLayout:
    """Audio channel layout (e.g. stereo, 5.1 surround)."""

    MONO: str
    STEREO: str
    SURROUND_5_1: str

    def __init__(self, layout: str) -> None: ...
    def channel_count(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Rational:
    """Rational number for frame rates and timebases (num/den)."""

    def __init__(self, num: int, den: int) -> None: ...
    @property
    def num(self) -> int: ...
    @property
    def den(self) -> int: ...
    def to_float(self) -> float: ...
    def __getnewargs__(self) -> Tuple[int, int]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class VideoFrame:
    """Decoded video frame containing pixel data."""

    def __init__(self, width: int, height: int, format: PixelFormat) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def format(self) -> PixelFormat: ...
    @property
    def pts(self) -> int: ...
    @pts.setter
    def pts(self, value: int) -> None: ...
    def plane_data(self, index: int) -> bytes: ...
    def plane_stride(self, index: int) -> int: ...
    def plane_count(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class AudioFrame:
    """Decoded audio frame containing PCM sample data."""

    def __init__(
        self,
        samples: bytes,
        sample_count: int,
        sample_rate: int,
        channels: int,
        format: SampleFormat,
    ) -> None: ...
    def samples(self) -> bytes: ...
    @property
    def sample_count(self) -> int: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def channels(self) -> int: ...
    @property
    def format(self) -> SampleFormat: ...
    @property
    def pts(self) -> Optional[int]: ...
    def duration_seconds(self) -> float: ...
    def to_f32(self) -> list[float]: ...
    def to_i16(self) -> list[int]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class EncoderPreset:
    """Encoder speed/quality preset."""

    ULTRAFAST: str
    FAST: str
    MEDIUM: str
    SLOW: str
    VERYSLOW: str

    def __init__(self, preset: str) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class EncoderConfig:
    """Video encoder configuration (resolution, bitrate, preset, etc.)."""

    def __init__(
        self,
        width: int,
        height: int,
        framerate: Tuple[int, int] = (30, 1),
        crf: float = 28.0,
        preset: Optional[EncoderPreset] = None,
        keyint: int = 250,
    ) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def framerate(self) -> Tuple[int, int]: ...
    @property
    def keyint(self) -> int: ...
    def __getstate__(self) -> Tuple[int, int, Tuple[int, int], int]: ...
    def __setstate__(self, state: Tuple[int, int, Tuple[int, int], int]) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Codec types
# ---------------------------------------------------------------------------

class Av1Decoder:
    """AV1 video decoder."""

    def __init__(self) -> None: ...
    def send_packet(self, data: bytes, pts: int = 0) -> None: ...
    def receive_frame(self) -> Optional[VideoFrame]: ...

class Av1Encoder:
    """AV1 video encoder."""

    def __init__(self, config: EncoderConfig) -> None: ...
    def send_frame(self, frame: VideoFrame) -> None: ...
    def receive_packet(self) -> Optional[bytes]: ...

class OpusDecoder:
    """Opus audio decoder."""

    def __init__(self) -> None: ...
    def decode(self, data: bytes) -> Optional[AudioFrame]: ...

class OpusEncoder:
    """Opus audio encoder."""

    def __init__(self, config: Any) -> None: ...
    def encode(self, frame: AudioFrame) -> Optional[bytes]: ...

# ---------------------------------------------------------------------------
# Context-manager resource wrappers
# ---------------------------------------------------------------------------

class ManagedDecoder:
    """Decoder with context-manager support (``with`` statement)."""

    def __init__(self, path: str) -> None: ...
    @property
    def path(self) -> str: ...
    @property
    def is_open(self) -> bool: ...
    def open(self) -> None: ...
    def close(self) -> None: ...
    def probe(self) -> str: ...
    def __enter__(self) -> ManagedDecoder: ...
    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool: ...
    def __repr__(self) -> str: ...

class ManagedEncoder:
    """Encoder with context-manager support (``with`` statement)."""

    def __init__(self, width: int = 1920, height: int = 1080) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def is_open(self) -> bool: ...
    @property
    def frames_encoded(self) -> int: ...
    def open(self) -> None: ...
    def close(self) -> None: ...
    def encode_frame(self) -> bytes: ...
    def __enter__(self) -> ManagedEncoder: ...
    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------

class PyQualityScore:
    """Result of a quality assessment operation."""

    @property
    def psnr(self) -> float: ...
    @property
    def ssim(self) -> float: ...
    @property
    def vmaf(self) -> Optional[float]: ...
    def __repr__(self) -> str: ...

def compute_psnr(reference: VideoFrame, distorted: VideoFrame) -> float:
    """Compute PSNR (dB) between two video frames."""
    ...

def compute_ssim(reference: VideoFrame, distorted: VideoFrame) -> float:
    """Compute SSIM score (0-1) between two video frames."""
    ...

def quality_report(reference: VideoFrame, distorted: VideoFrame) -> PyQualityScore:
    """Generate a full quality assessment report."""
    ...

# ---------------------------------------------------------------------------
# Parallel / GIL-release utilities
# ---------------------------------------------------------------------------

def compute_checksums(data: Sequence[bytes]) -> list[int]:
    """
    Compute FNV-1a 64-bit checksums for a list of byte strings.

    The GIL is released during computation so other threads can run.

    Parameters
    ----------
    data : list[bytes]
        Input byte strings.

    Returns
    -------
    list[int]
        Checksums (unsigned 64-bit integers).
    """
    ...

def compute_checksum_single(data: bytes) -> int:
    """
    Compute a single FNV-1a 64-bit checksum.

    Parameters
    ----------
    data : bytes
        Input byte string.

    Returns
    -------
    int
        64-bit FNV-1a checksum.
    """
    ...

# ---------------------------------------------------------------------------
# utils submodule
# ---------------------------------------------------------------------------

class utils:
    @staticmethod
    def duration_to_timecode(seconds: float, fps: float = 25.0) -> str:
        """Convert a duration in seconds to SMPTE timecode string (HH:MM:SS:FF)."""
        ...

    @staticmethod
    def fps_to_rational(fps: float) -> Tuple[int, int]:
        """Convert a floating-point FPS to (numerator, denominator) rational."""
        ...

    @staticmethod
    def format_duration(seconds: float, precision: int = 3) -> str:
        """Format a duration as human-readable string (e.g. "1h 2m 3.456s")."""
        ...

    @staticmethod
    def estimate_bitrate(file_size_bytes: int, duration_seconds: float) -> float:
        """Estimate average bitrate in kbps from file size and duration."""
        ...

    @staticmethod
    def calculate_frame_size(
        width: int, height: int, pixel_format: str = "yuv420p"
    ) -> int:
        """Calculate uncompressed frame size in bytes."""
        ...

# ---------------------------------------------------------------------------
# io submodule
# ---------------------------------------------------------------------------

class io:
    class MediaFileInfo:
        """High-level summary of a probed media file."""

        path: str
        duration_seconds: float
        size_bytes: int
        video_stream_count: int
        audio_stream_count: int
        container: str
        video_codec: Optional[str]
        audio_codec: Optional[str]

        def __repr__(self) -> str: ...

    @staticmethod
    def probe(path: str) -> "io.MediaFileInfo":
        """Probe a media file and return metadata."""
        ...

# ---------------------------------------------------------------------------
# test submodule
# ---------------------------------------------------------------------------

class test:
    @staticmethod
    def synthetic_video_frame(
        width: int = 1920,
        height: int = 1080,
        pts: int = 0,
        pixel_format: str = "yuv420p",
    ) -> VideoFrame:
        """Generate a synthetic video frame for testing."""
        ...

    @staticmethod
    def synthetic_audio_frame(
        sample_rate: int = 48000,
        channels: int = 2,
        duration_ms: float = 20.0,
    ) -> AudioFrame:
        """Generate a synthetic audio frame for testing."""
        ...

# ---------------------------------------------------------------------------
# logging submodule
# ---------------------------------------------------------------------------

class logging:
    @staticmethod
    def init(level: str = "info") -> None:
        """Initialise the Rust→Python logging bridge."""
        ...

    @staticmethod
    def set_level(level: str) -> None:
        """Set the active log level for the OxiMedia logger."""
        ...

    @staticmethod
    def log(level: str, message: str) -> None:
        """Emit a log message at the given level."""
        ...
