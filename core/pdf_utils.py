"""
PDF utility functions shared across agents.
DPI 300 rendering + optional OpenCV auto-region segmentation.
Returns PIL.Image objects — compatible with all google-generativeai versions.
"""

import math
from pathlib import Path
import fitz  # PyMuPDF
import io
from PIL import Image, ImageDraw, ImageFont
from config import PDF_DPI, SCANNER_DPI, OPENCV_ENABLED

_LANCZOS = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) or Image.LANCZOS

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


def render_page_fast(pdf_path: str, page_number: int, dpi: int | None = None) -> Image.Image:
    """Low-res render for scanner/classification. Pass dpi= to override SCANNER_DPI."""
    return render_page_as_image_part(pdf_path, page_number, dpi=dpi or SCANNER_DPI)


def build_thumbnail_grid(
    pdf_path: str,
    dpi: int = 72,
    max_per_grid: int = 36,
    pages: list[int] | None = None,
) -> list[tuple[Image.Image, list[int]]]:
    """
    Render PDF pages as tiny labeled thumbnails arranged in a composite grid.

    Returns a list of (grid_image, actual_page_numbers) tuples — one per batch.
    Each batch covers at most max_per_grid pages. Grid is capped at 4000×4000 px
    (Gemini vision limit). Each thumbnail is labeled "P{page_num}" top-left.
    """
    doc = load_pdf(pdf_path)
    total = len(doc)
    page_list = pages if pages is not None else list(range(total))
    page_list = [p for p in page_list if 0 <= p < total]

    thumbs: list[tuple[int, Image.Image]] = []
    for p in page_list:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[p].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        thumbs.append((p, Image.open(io.BytesIO(pix.tobytes("png")))))
    doc.close()

    if not thumbs:
        return []

    border = 2
    results: list[tuple[Image.Image, list[int]]] = []

    for batch in [thumbs[i:i + max_per_grid] for i in range(0, len(thumbs), max_per_grid)]:
        n = len(batch)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        tw = max(img.width  for _, img in batch)
        th = max(img.height for _, img in batch)

        grid_w = cols * tw + (cols + 1) * border
        grid_h = rows * th + (rows + 1) * border

        scale = min(1.0, 4000 / max(grid_w, grid_h))
        if scale < 1.0:
            tw = max(1, int(tw * scale))
            th = max(1, int(th * scale))
            grid_w = cols * tw + (cols + 1) * border
            grid_h = rows * th + (rows + 1) * border

        grid  = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
        draw  = ImageDraw.Draw(grid)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        page_nums: list[int] = []
        for idx, (page_num, thumb) in enumerate(batch):
            page_nums.append(page_num)
            col = idx % cols
            row = idx // cols
            x = border + col * (tw + border)
            y = border + row * (th + border)

            grid.paste(thumb.resize((tw, th), _LANCZOS), (x, y))

            label = f"P{page_num}"
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                draw.text((x + 3 + dx, y + 3 + dy), label, fill=(0, 0, 0), font=font)
            draw.text((x + 3, y + 3), label, fill=(255, 255, 255), font=font)

        results.append((grid, page_nums))

    return results


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


def extract_tables_pdfplumber(pdf_path: str, page_number: int) -> str | None:
    """
    Extract table data as CSV text using pdfplumber.
    Returns formatted CSV if the page contains viable tables (>= 3 rows, >= 3 cols).
    Returns None for scanned/image-only pages or pages with no structured tables.
    """
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if page_number >= len(pdf.pages):
                return None
            tables = pdf.pages[page_number].extract_tables()

        viable = [
            t for t in (tables or [])
            if t and len(t) >= 3 and any(len(row) >= 3 for row in t if row)
        ]
        if not viable:
            return None

        lines = []
        for tbl in viable:
            for row in tbl:
                cells = [str(c).strip() if c is not None else "" for c in row]
                lines.append(",".join(cells))
            lines.append("")
        return "\n".join(lines).strip()
    except Exception:
        return None


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
