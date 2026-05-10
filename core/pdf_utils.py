"""
PDF utility functions shared across agents.
DPI 300 rendering + optional OpenCV auto-region segmentation.
Returns PIL.Image objects — compatible with all google-generativeai versions.
"""

from pathlib import Path
import fitz  # PyMuPDF
import io
from PIL import Image
from config import PDF_DPI, SCANNER_DPI, OPENCV_ENABLED

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def load_pdf(pdf_path: str) -> fitz.Document:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return fitz.open(str(path))


def get_page_count(pdf_path: str) -> int:
    doc = load_pdf(pdf_path)
    n = len(doc)
    doc.close()
    return n


def _page_to_pil(page: fitz.Page, dpi: int = PDF_DPI) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def render_page_as_image_part(pdf_path: str, page_number: int, dpi: int | None = None) -> Image.Image:
    """
    Render a PDF page and return a PIL Image.
    dpi defaults to PDF_DPI (200). Pass SCANNER_DPI (100) for fast classification.
    page_number is 0-indexed.
    """
    doc = load_pdf(pdf_path)
    img = _page_to_pil(doc[page_number], dpi=dpi or PDF_DPI)
    doc.close()
    return img


def render_page_fast(pdf_path: str, page_number: int) -> Image.Image:
    """Low-res render for scanner/classification — 3-4x faster than full DPI."""
    return render_page_as_image_part(pdf_path, page_number, dpi=SCANNER_DPI)


def segment_page_regions(pdf_path: str, page_number: int) -> list[Image.Image]:
    """
    Use OpenCV contour detection to split a page into sub-regions.
    Falls back to single full-page image if OpenCV unavailable or no regions found.
    """
    doc = load_pdf(pdf_path)
    pil_img = _page_to_pil(doc[page_number])
    doc.close()

    if not OPENCV_ENABLED or not _CV2_AVAILABLE:
        return [pil_img]

    # Convert PIL → OpenCV
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (h * w) * 0.05
    regions: list[Image.Image] = []
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        if rw * rh < min_area:
            continue
        crop_bgr = img[y:y+rh, x:x+rw]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        regions.append(Image.fromarray(crop_rgb))

    return regions if regions else [pil_img]


def extract_text_from_page(pdf_path: str, page_number: int) -> str:
    doc = load_pdf(pdf_path)
    text = doc[page_number].get_text("text")
    doc.close()
    return text


def extract_all_text(pdf_path: str) -> dict[int, str]:
    doc = load_pdf(pdf_path)
    result = {i: doc[i].get_text("text") for i in range(len(doc))}
    doc.close()
    return result
