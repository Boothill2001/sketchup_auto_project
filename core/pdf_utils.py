"""
PDF utility functions shared across agents.
DPI 300 rendering + optional OpenCV auto-region segmentation.
Returns PIL.Image objects — compatible with all google-generativeai versions.
"""

import math
import re
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


# ==========================================================================
# FIX 1 — Deterministic RC dimension extraction from schedule pages
# ==========================================================================

def extract_text_blocks_with_bbox(pdf_path: str, page_number: int) -> list[dict]:
    """
    Extract text blocks with bounding boxes from a PDF page using PyMuPDF.
    Returns list of {text, x0, y0, x1, y1} dicts sorted top-to-bottom, left-to-right.
    Useful for deterministic parsing of table cells and RC dimensions.
    """
    doc = load_pdf(pdf_path)
    page = doc[page_number]
    blocks = page.get_text("dict")["blocks"]
    doc.close()

    text_blocks = []
    for block in blocks:
        if block["type"] == 0:  # text block
            for line in block.get("lines", []):
                line_text = ""
                x0 = line["bbox"][0]
                y0 = line["bbox"][1]
                x1 = line["bbox"][2]
                y1 = line["bbox"][3]
                for span in line.get("spans", []):
                    line_text += span["text"]
                if line_text.strip():
                    text_blocks.append({
                        "text": line_text.strip(),
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                    })

    # Sort by y then x (top-to-bottom, left-to-right reading order)
    text_blocks.sort(key=lambda b: (b["y0"], b["x0"]))
    return text_blocks


def _group_blocks_into_rows(blocks: list[dict], y_tolerance: float = 15) -> list[list[dict]]:
    """Group text blocks into rows by Y-coordinate proximity."""
    if not blocks:
        return []
    rows = []
    current_row = [blocks[0]]
    current_y = blocks[0]["y0"]
    for b in blocks[1:]:
        if abs(b["y0"] - current_y) < y_tolerance:
            current_row.append(b)
        else:
            rows.append(current_row)
            current_row = [b]
            current_y = b["y0"]
    if current_row:
        rows.append(current_row)
    return rows


def parse_rc_dimensions_from_schedule(pdf_path: str, schedule_pages: list[int]) -> dict[str, dict]:
    """
    Deterministic extraction of RC member dimensions from schedule pages.
    Uses PyMuPDF text blocks with bounding boxes to match:
      - Column schedule: mark -> width_mm x depth_mm (or diameter_mm for circular)
      - Beam schedule: mark -> width_mm x depth_mm
      - Wall schedule: mark -> thickness_mm
      - Slab schedule: mark -> thickness_mm

    Returns dict: { "C1": {"width_mm": 500, "depth_mm": 500}, ... }
    """
    results: dict[str, dict] = {}

    # Regex for dimension patterns (e.g. "500x500", "300x600", "300×600")
    dim_pattern = re.compile(r"(\d{3,4})\s*[xX×]\s*(\d{3,4})")

    # Mark patterns
    rc_col_mark = re.compile(
        r"\b((?:C\s*O?\s*L?\s*_?\s*[A-D]?|C|COL|RC\s*PILE|RC\s*COL)\s*\d+[A-Za-z]?)\b",
        re.IGNORECASE,
    )
    rc_beam_mark = re.compile(
        r"\b((?:[BG]\s*|BM)\s*\d+[A-Za-z]?)\b",
        re.IGNORECASE,
    )
    rc_wall_mark = re.compile(
        r"\b((?:W\s*|WL)\s*\d+[A-Za-z]?)\b",
        re.IGNORECASE,
    )
    rc_slab_mark = re.compile(
        r"\b((?:S\s*|SL|PT)\s*\d+[A-Za-z]?)\b",
        re.IGNORECASE,
    )

    for page_num in schedule_pages:
        try:
            blocks = extract_text_blocks_with_bbox(pdf_path, page_num)
        except Exception:
            continue

        if not blocks:
            continue

        rows = _group_blocks_into_rows(blocks, y_tolerance=15)

        for row in rows:
            row_texts = [b["text"] for b in row]
            row_str = " ".join(row_texts)

            # Try column marks first
            mark_match = rc_col_mark.search(row_str)
            member_type = "column"
            if not mark_match:
                mark_match = rc_wall_mark.search(row_str)
                member_type = "wall"
            if not mark_match:
                mark_match = rc_beam_mark.search(row_str)
                member_type = "beam"
            if not mark_match:
                mark_match = rc_slab_mark.search(row_str)
                member_type = "slab"

            if not mark_match:
                continue

            mark = mark_match.group(1).strip().replace(" ", "").upper()

            # Skip if already found
            if mark in results:
                continue

            # Look for dimension pattern (WxH) in the row
            dim_match = dim_pattern.search(row_str)
            if dim_match:
                w = int(dim_match.group(1))
                d = int(dim_match.group(2))
                results[mark] = {"width_mm": w, "depth_mm": d}
                continue

            # For walls/slabs: look for single thickness number
            if member_type in ("wall", "slab"):
                for text in row_texts:
                    num_match = re.search(r"(\d{3,4})", text.strip())
                    if num_match:
                        val = int(num_match.group(1))
                        if 100 <= val <= 600:  # plausible RC thickness
                            results[mark] = {"thickness_mm": val}
                            break
            else:
                # For columns/beams without "x" pattern
                # Look for numbers in cells after the mark — first 2 substantive numbers
                nums = []
                for text in row_texts:
                    found = re.findall(r"\b(\d{3,4})\b", text.strip())
                    nums.extend(int(n) for n in found if 200 <= int(n) <= 2000)
                if len(nums) >= 2:
                    results[mark] = {"width_mm": nums[0], "depth_mm": nums[1]}
                elif len(nums) == 1:
                    # Single number — could be circular column (diameter)
                    results[mark] = {"width_mm": nums[0], "depth_mm": nums[0]}

    return results