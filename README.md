# Pixel·OCR — From Scratch

A complete OCR web app built with **FastAPI** + pure Python OCR engine.  
**No OpenCV. No NumPy. No image processing libraries.**

## Project Structure

```
ocr_app/
├── main.py            ← FastAPI server + /api/ocr endpoint
├── ocr_engine.py      ← Full OCR pipeline (pure Python)
├── requirements.txt
└── templates/
    └── index.html     ← HTML/CSS/JS frontend
```

## Setup & Run

```bash
# 1. Install dependencies (standard library only for OCR, FastAPI for server)
pip install -r requirements.txt

# 2. Start the server
uvicorn main:app --reload --port 8000

# 3. Open in browser
open http://localhost:8000
```

## API

### POST /api/ocr

Upload an image and get extracted text back.

**Form fields:**
| Field       | Type    | Default | Description                        |
|-------------|---------|---------|-----------------------------------|
| `file`      | file    | —       | PNG, BMP, or JPG image            |
| `threshold` | int     | auto    | Binarization threshold (0–255)    |
| `row_gap`   | int     | 3       | Max pixel gap between text rows   |
| `col_gap`   | int     | 5       | Max pixel gap between characters  |

**Response:**
```json
{
  "success": true,
  "text": "Hello World",
  "lines": 1,
  "words": 2,
  "characters": 10,
  "width": 640,
  "height": 480,
  "threshold_used": 128,
  "threshold_auto": 128,
  "elapsed_ms": 312.4
}
```

## OCR Pipeline

1. **Raw byte decoding** — PNG chunk parsing with zlib inflate, BMP header parsing
2. **Grayscale** — ITU-R BT.601: `0.299·R + 0.587·G + 0.114·B`
3. **Otsu binarization** — maximize inter-class pixel variance
4. **Row segmentation** — find horizontal ink-filled bands
5. **Column segmentation** — isolate individual character bounding boxes
6. **Feature extraction** — 26-feature vector per glyph
7. **Template matching** — weighted Euclidean distance to 80+ reference bitmaps

## Tips for Best Results

- Use **dark text on white/light background**
- Works best on **printed or typed text** (not handwriting)
- Try adjusting the threshold slider if detection is poor
- PNG format gives the cleanest results
