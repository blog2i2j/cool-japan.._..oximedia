"""
Type stubs for oximedia.cv2 — OpenCV-compatible image processing.

All image functions accept numpy.ndarray with shape (H, W, C) or (H, W)
and return numpy.ndarray with same shape convention.
BGR channel order (OpenCV convention).
"""
from typing import Optional, Sequence, Tuple, Union
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────
# imread flags
IMREAD_COLOR: int
IMREAD_GRAYSCALE: int
IMREAD_UNCHANGED: int

# Color conversion codes
COLOR_BGR2GRAY: int
COLOR_BGR2RGB: int
COLOR_RGB2BGR: int
COLOR_GRAY2BGR: int
COLOR_GRAY2RGB: int
COLOR_BGR2HSV: int
COLOR_RGB2HSV: int
COLOR_HSV2BGR: int
COLOR_HSV2RGB: int
COLOR_BGR2Lab: int
COLOR_RGB2Lab: int
COLOR_Lab2BGR: int
COLOR_Lab2RGB: int
COLOR_BGR2YUV: int
COLOR_YUV2BGR: int
COLOR_BGR2HLS: int
COLOR_HLS2BGR: int

# Interpolation flags
INTER_NEAREST: int
INTER_LINEAR: int
INTER_CUBIC: int
INTER_AREA: int
INTER_LANCZOS4: int

# Border types
BORDER_CONSTANT: int
BORDER_REPLICATE: int
BORDER_REFLECT: int
BORDER_WRAP: int
BORDER_DEFAULT: int

# Threshold types
THRESH_BINARY: int
THRESH_BINARY_INV: int
THRESH_TRUNC: int
THRESH_TOZERO: int
THRESH_TOZERO_INV: int
THRESH_OTSU: int
THRESH_TRIANGLE: int

# Adaptive threshold methods
ADAPTIVE_THRESH_MEAN_C: int
ADAPTIVE_THRESH_GAUSSIAN_C: int

# Morphology shapes
MORPH_RECT: int
MORPH_CROSS: int
MORPH_ELLIPSE: int

# Morphology operations
MORPH_ERODE: int
MORPH_DILATE: int
MORPH_OPEN: int
MORPH_CLOSE: int
MORPH_GRADIENT: int
MORPH_TOPHAT: int
MORPH_BLACKHAT: int

# Contour modes
RETR_EXTERNAL: int
RETR_LIST: int
RETR_CCOMP: int
RETR_TREE: int

# Contour methods
CHAIN_APPROX_NONE: int
CHAIN_APPROX_SIMPLE: int

# VideoCapture properties
CAP_PROP_POS_FRAMES: int
CAP_PROP_FRAME_WIDTH: int
CAP_PROP_FRAME_HEIGHT: int
CAP_PROP_FPS: int
CAP_PROP_FRAME_COUNT: int

# Rotation codes
ROTATE_90_CLOCKWISE: int
ROTATE_180: int
ROTATE_90_COUNTERCLOCKWISE: int

# Font types
FONT_HERSHEY_SIMPLEX: int
FONT_HERSHEY_PLAIN: int
FONT_ITALIC: int

# Line types
LINE_4: int
LINE_8: int
LINE_AA: int
FILLED: int

# Template matching methods
TM_SQDIFF: int
TM_SQDIFF_NORMED: int
TM_CCORR: int
TM_CCORR_NORMED: int
TM_CCOEFF: int
TM_CCOEFF_NORMED: int

# Data types
CV_8U: int
CV_8S: int
CV_16U: int
CV_16S: int
CV_32F: int
CV_64F: int

# ── Type aliases ───────────────────────────────────────────────────────────
Image = np.ndarray  # shape (H, W, 3) BGR uint8 or (H, W, 1) grayscale
GrayImage = np.ndarray  # shape (H, W, 1) uint8
Point = Tuple[int, int]  # (x, y)
Color = Tuple[int, int, int]  # (B, G, R)
Rect = Tuple[int, int, int, int]  # (x, y, w, h)
Contour = list[Point]

# ── Image I/O ──────────────────────────────────────────────────────────────
def imread(filename: str, flags: int = IMREAD_COLOR) -> np.ndarray: ...
def imwrite(filename: str, img: np.ndarray) -> bool: ...
def imdecode(buf: bytes, flags: int = IMREAD_COLOR) -> np.ndarray: ...
def imencode(ext: str, img: np.ndarray, params: Optional[list] = None) -> Tuple[bool, bytes]: ...

# ── Color conversion ───────────────────────────────────────────────────────
def cvtColor(src: np.ndarray, code: int) -> np.ndarray: ...

# ── Geometry ───────────────────────────────────────────────────────────────
def resize(src: np.ndarray, dsize: Tuple[int, int], interpolation: int = INTER_LINEAR) -> np.ndarray: ...
def flip(src: np.ndarray, flipCode: int) -> np.ndarray: ...
def rotate(src: np.ndarray, rotateCode: int) -> np.ndarray: ...
def warpAffine(
    src: np.ndarray, M: list, dsize: Tuple[int, int], flags: int = INTER_LINEAR
) -> np.ndarray: ...
def getRotationMatrix2D(center: Tuple[float, float], angle: float, scale: float) -> list: ...

# ── Blur / filter ──────────────────────────────────────────────────────────
def GaussianBlur(src: np.ndarray, ksize: Tuple[int, int], sigmaX: float, sigmaY: float = 0.0) -> np.ndarray: ...
def medianBlur(src: np.ndarray, ksize: int) -> np.ndarray: ...
def bilateralFilter(src: np.ndarray, d: int, sigmaColor: float, sigmaSpace: float) -> np.ndarray: ...
def filter2D(src: np.ndarray, ddepth: int, kernel: list) -> np.ndarray: ...

# ── Edge detection ─────────────────────────────────────────────────────────
def Canny(src: np.ndarray, threshold1: float, threshold2: float, apertureSize: int = 3, L2gradient: bool = False) -> np.ndarray: ...
def Sobel(src: np.ndarray, ddepth: int, dx: int, dy: int, ksize: int = 3, scale: float = 1.0, delta: float = 0.0) -> np.ndarray: ...
def Laplacian(src: np.ndarray, ddepth: int, ksize: int = 1, scale: float = 1.0, delta: float = 0.0) -> np.ndarray: ...

# ── Threshold ──────────────────────────────────────────────────────────────
def threshold(src: np.ndarray, thresh: float, maxval: float, type: int) -> Tuple[float, np.ndarray]: ...
def adaptiveThreshold(src: np.ndarray, maxValue: float, adaptiveMethod: int, thresholdType: int, blockSize: int, C: float) -> np.ndarray: ...

# ── Morphology ─────────────────────────────────────────────────────────────
def getStructuringElement(shape: int, ksize: Tuple[int, int]) -> list: ...
def erode(src: np.ndarray, kernel: list, iterations: int = 1) -> np.ndarray: ...
def dilate(src: np.ndarray, kernel: list, iterations: int = 1) -> np.ndarray: ...
def morphologyEx(src: np.ndarray, op: int, kernel: list, iterations: int = 1) -> np.ndarray: ...

# ── Contours ───────────────────────────────────────────────────────────────
def findContours(image: np.ndarray, mode: int, method: int) -> Tuple[list[Contour], list]: ...
def drawContours(image: np.ndarray, contours: list[Contour], contourIdx: int, color: Color, thickness: int = 1) -> np.ndarray: ...
def contourArea(contour: Contour) -> float: ...
def boundingRect(contour: Contour) -> Rect: ...
def arcLength(contour: Contour, closed: bool) -> float: ...
def approxPolyDP(contour: Contour, epsilon: float, closed: bool) -> Contour: ...

# ── Drawing ────────────────────────────────────────────────────────────────
def rectangle(img: np.ndarray, pt1: Point, pt2: Point, color: Color, thickness: int = 1) -> np.ndarray: ...
def circle(img: np.ndarray, center: Point, radius: int, color: Color, thickness: int = 1) -> np.ndarray: ...
def line(img: np.ndarray, pt1: Point, pt2: Point, color: Color, thickness: int = 1) -> np.ndarray: ...
def putText(img: np.ndarray, text: str, org: Point, fontFace: int, fontScale: float, color: Color, thickness: int = 1, lineType: int = LINE_8) -> np.ndarray: ...
def polylines(img: np.ndarray, pts: list[Point], isClosed: bool, color: Color, thickness: int = 1) -> np.ndarray: ...
def fillPoly(img: np.ndarray, pts: list[Point], color: Color) -> np.ndarray: ...
def ellipse(img: np.ndarray, center: Point, axes: Tuple[int, int], angle: float, startAngle: float, endAngle: float, color: Color, thickness: int = 1) -> np.ndarray: ...

# ── Features ───────────────────────────────────────────────────────────────
class KeyPoint:
    x: float
    y: float
    size: float
    angle: float
    response: float
    octave: int
    class_id: int

    def __init__(self, x: float, y: float, size: float = 1.0, angle: float = -1.0,
                 response: float = 0.0, octave: int = 0, class_id: int = -1) -> None: ...

    @property
    def pt(self) -> Tuple[float, float]: ...

class ORB:
    def __init__(self, nfeatures: int = 500, scaleFactor: float = 1.2, nlevels: int = 8,
                 edgeThreshold: int = 31, firstLevel: int = 0, WTA_K: int = 2,
                 scoreType: int = 0, patchSize: int = 31, fastThreshold: int = 20) -> None: ...
    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> list[KeyPoint]: ...
    def compute(self, image: np.ndarray, keypoints: list[KeyPoint]) -> Tuple[list[KeyPoint], np.ndarray]: ...
    def detectAndCompute(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[list[KeyPoint], np.ndarray]: ...

def ORB_create(nfeatures: int = 500, scaleFactor: float = 1.2, nlevels: int = 8) -> ORB: ...
def goodFeaturesToTrack(
    image: np.ndarray, maxCorners: int, qualityLevel: float, minDistance: float,
    mask: Optional[np.ndarray] = None, blockSize: int = 3,
    useHarrisDetector: bool = False, k: float = 0.04
) -> list[list[float]]: ...

# ── Optical flow ───────────────────────────────────────────────────────────
def calcOpticalFlowPyrLK(
    prevImg: np.ndarray, nextImg: np.ndarray,
    prevPts: list[list[float]],
    nextPts: Optional[list[list[float]]] = None,
    status: Optional[list[int]] = None,
    err: Optional[list[float]] = None,
    winSize: Tuple[int, int] = (21, 21),
    maxLevel: int = 3,
) -> Tuple[list[list[float]], list[int], list[float]]: ...

# ── Arithmetic ─────────────────────────────────────────────────────────────
def addWeighted(src1: np.ndarray, alpha: float, src2: np.ndarray, beta: float, gamma: float) -> np.ndarray: ...
def absdiff(src1: np.ndarray, src2: np.ndarray) -> np.ndarray: ...
def bitwise_and(src1: np.ndarray, src2: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray: ...
def bitwise_or(src1: np.ndarray, src2: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray: ...
def bitwise_xor(src1: np.ndarray, src2: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray: ...
def bitwise_not(src: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray: ...
def normalize(src: np.ndarray, dst: Optional[np.ndarray] = None, alpha: float = 0.0, beta: float = 255.0, norm_type: int = 32) -> np.ndarray: ...
def inRange(src: np.ndarray, lowerb: Sequence[int], upperb: Sequence[int]) -> np.ndarray: ...

# ── Histogram ──────────────────────────────────────────────────────────────
def calcHist(images: list[np.ndarray], channels: list[int], mask: Optional[np.ndarray], histSize: list[int], ranges: list[float]) -> list[float]: ...
def equalizeHist(src: np.ndarray) -> np.ndarray: ...

# ── Template matching ──────────────────────────────────────────────────────
def matchTemplate(image: np.ndarray, templ: np.ndarray, method: int) -> np.ndarray: ...
def minMaxLoc(src: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float, Tuple[int, int], Tuple[int, int]]: ...

# ── Connected components ───────────────────────────────────────────────────
def connectedComponents(image: np.ndarray, connectivity: int = 8) -> Tuple[int, np.ndarray]: ...

# ── Video ──────────────────────────────────────────────────────────────────
class VideoCapture:
    def __init__(self, filename: Union[str, int]) -> None: ...
    def isOpened(self) -> bool: ...
    def read(self) -> Tuple[bool, Optional[np.ndarray]]: ...
    def get(self, propId: int) -> float: ...
    def set(self, propId: int, value: float) -> bool: ...
    def release(self) -> None: ...
    def __enter__(self) -> "VideoCapture": ...
    def __exit__(self, *args: object) -> None: ...

class VideoWriter:
    def __init__(self, filename: str, fourcc: int, fps: float, frameSize: Tuple[int, int], isColor: bool = True) -> None: ...
    def isOpened(self) -> bool: ...
    def write(self, frame: np.ndarray) -> None: ...
    def get(self, propId: int) -> float: ...
    def release(self) -> None: ...
    def __enter__(self) -> "VideoWriter": ...
    def __exit__(self, *args: object) -> None: ...

def VideoWriter_fourcc(c1: str, c2: str, c3: str, c4: str) -> int: ...
