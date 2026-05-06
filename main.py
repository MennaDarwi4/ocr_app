"""
OCR App — FastAPI Backend
Serves the HTML UI and exposes /api/ocr endpoint.
"""

import io
import os
import time
import tempfile

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ocr_engine import load_image, to_grayscale, binarize, otsu_threshold
from ocr_engine import find_row_spans, find_col_spans, merge_spans
from ocr_engine import extract_features, compute_template_features, match_character

app = FastAPI(title="OCR From Scratch", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-compute templates once at startup
TEMPLATES = compute_template_features()


def run_ocr_on_bytes(data: bytes, filename: str, threshold: int = None,
                     row_gap: int = 3, col_gap: int = 5):
    """Run full OCR pipeline on raw image bytes. Returns result dict."""
    suffix = os.path.splitext(filename)[1].lower() or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()

        pixels, width, height = load_image(tmp_path)
        gray = to_grayscale(pixels)
        auto_threshold = otsu_threshold(gray)
        used_threshold = threshold if threshold is not None else auto_threshold
        bin_img = binarize(gray, width, height, used_threshold)

        row_spans = find_row_spans(bin_img, height, width)
        row_spans = merge_spans(row_spans, row_gap)

        lines = []
        char_count = 0

        for y0, y1 in row_spans:
            line_h = y1 - y0 + 1
            if line_h < 4:
                continue
            col_spans = find_col_spans(bin_img, y0, y1, width)
            col_spans = merge_spans(col_spans, col_gap)

            chars = []
            prev_x1 = -1
            prev_w = 1
            for x0, x1 in col_spans:
                char_w = x1 - x0 + 1
                if char_w < 2:
                    continue
                if prev_x1 >= 0:
                    gap = x0 - prev_x1
                    if gap > max(prev_w, char_w) * 0.8:
                        chars.append(" ")
                feat = extract_features(bin_img, x0, x1, y0, y1)
                if feat is not None:
                    ch = match_character(feat, TEMPLATES)
                    chars.append(ch)
                    char_count += 1
                prev_x1 = x1
                prev_w = char_w

            line_text = "".join(chars).strip()
            if line_text:
                lines.append(line_text)

        elapsed = time.perf_counter() - t0
        text = "\n".join(lines)

        return {
            "success": True,
            "text": text,
            "lines": len(lines),
            "characters": char_count,
            "words": len(text.split()) if text else 0,
            "width": width,
            "height": height,
            "threshold_used": used_threshold,
            "threshold_auto": auto_threshold,
            "elapsed_ms": round(elapsed * 1000, 1),
        }
    finally:
        os.unlink(tmp_path)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r") as f:
        return f.read()


@app.post("/api/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    threshold: int = Form(None),
    row_gap: int = Form(3),
    col_gap: int = Form(5),
):
    allowed = {".png", ".bmp", ".jpg", ".jpeg"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

    try:
        result = run_ocr_on_bytes(
            data, file.filename,
            threshold=threshold,
            row_gap=row_gap,
            col_gap=col_gap,
        )
        return JSONResponse(result)
    except AssertionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}
