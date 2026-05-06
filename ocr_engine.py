"""
OCR From Scratch — Pure Python, Zero External Libraries
========================================================
No OpenCV, no numpy, no Pillow, no image processing libraries.
Only Python standard library is used.

Pipeline:
  1. Read image file bytes manually (supports PNG and BMP)
  2. Decode pixels by hand (zlib inflate for PNG, raw rows for BMP)
  3. Grayscale conversion via luminance formula
  4. Binarization (Otsu's method — computed from scratch)
  5. Row & column segmentation to find text lines and characters
  6. Feature extraction per glyph (zone densities, run counts, projections)
  7. Template matching against rendered reference glyphs (drawn pixel-by-pixel)
  8. Output the extracted text
"""

import sys
import zlib
import struct
import math
import os


# ---------------------------------------------------------------------------
# 1. Image decoding — PNG and BMP, from scratch
# ---------------------------------------------------------------------------

def read_file(path):
    with open(path, "rb") as f:
        return f.read()


def decode_png(data):
    """Parse a PNG file and return (pixels, width, height).
    pixels is a flat list of (R,G,B) tuples, row-major.
    """
    assert data[:8] == b'\x89PNG\r\n\x1a\n', "Not a valid PNG file"
    pos = 8
    width = height = 0
    bit_depth = color_type = 0
    idat_chunks = []

    while pos < len(data):
        length = struct.unpack('>I', data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8]
        chunk_data = data[pos+8:pos+8+length]
        pos += 12 + length  # length + type + data + crc

        if chunk_type == b'IHDR':
            width  = struct.unpack('>I', chunk_data[0:4])[0]
            height = struct.unpack('>I', chunk_data[4:8])[0]
            bit_depth  = chunk_data[8]
            color_type = chunk_data[9]
            # color_type: 0=gray, 2=RGB, 3=indexed, 4=gray+alpha, 6=RGBA
        elif chunk_type == b'IDAT':
            idat_chunks.append(chunk_data)
        elif chunk_type == b'IEND':
            break

    assert bit_depth == 8, f"Only 8-bit PNG supported (got {bit_depth}-bit)"

    raw = zlib.decompress(b''.join(idat_chunks))

    channels = {0: 1, 2: 3, 3: 3, 4: 2, 6: 4}[color_type]
    stride = width * channels

    # PNG filter reconstruction
    pixels = []
    prev_row = [0] * stride

    raw_pos = 0
    for y in range(height):
        filter_type = raw[raw_pos]
        raw_pos += 1
        row_raw = list(raw[raw_pos:raw_pos + stride])
        raw_pos += stride

        row = _reconstruct_png_filter(filter_type, row_raw, prev_row, channels)
        prev_row = row

        for x in range(width):
            base = x * channels
            if color_type == 0:       # grayscale
                g = row[base]
                pixels.append((g, g, g))
            elif color_type == 2:     # RGB
                pixels.append((row[base], row[base+1], row[base+2]))
            elif color_type == 3:     # indexed — treat as gray
                g = row[base]
                pixels.append((g, g, g))
            elif color_type == 4:     # grayscale + alpha
                g = row[base]
                pixels.append((g, g, g))
            elif color_type == 6:     # RGBA
                pixels.append((row[base], row[base+1], row[base+2]))

    return pixels, width, height


def _paeth(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    elif pb <= pc:
        return b
    return c


def _reconstruct_png_filter(filter_type, raw, prev, bpp):
    n = len(raw)
    out = [0] * n
    for i in range(n):
        x   = raw[i]
        a   = out[i - bpp] if i >= bpp else 0
        b   = prev[i]
        c   = prev[i - bpp] if i >= bpp else 0
        if filter_type == 0:
            out[i] = x
        elif filter_type == 1:
            out[i] = (x + a) & 0xFF
        elif filter_type == 2:
            out[i] = (x + b) & 0xFF
        elif filter_type == 3:
            out[i] = (x + (a + b) // 2) & 0xFF
        elif filter_type == 4:
            out[i] = (x + _paeth(a, b, c)) & 0xFF
    return out


def decode_bmp(data):
    """Parse a BMP file and return (pixels, width, height)."""
    assert data[:2] == b'BM', "Not a valid BMP file"
    pixel_offset = struct.unpack('<I', data[10:14])[0]
    width        = struct.unpack('<i', data[18:22])[0]
    height       = struct.unpack('<i', data[22:26])[0]
    bpp          = struct.unpack('<H', data[28:30])[0]
    assert bpp in (24, 32), f"Only 24/32-bit BMP supported (got {bpp})"
    channels = bpp // 8
    row_size = (width * channels + 3) & ~3  # rows padded to 4 bytes
    flipped = height > 0  # positive height = bottom-up
    height = abs(height)

    pixels = []
    rows = []
    for y in range(height):
        off = pixel_offset + y * row_size
        row = data[off:off + width * channels]
        row_pixels = []
        for x in range(width):
            b = row[x * channels]
            g = row[x * channels + 1]
            r = row[x * channels + 2]
            row_pixels.append((r, g, b))
        rows.append(row_pixels)

    if flipped:
        rows = rows[::-1]
    for row in rows:
        pixels.extend(row)

    return pixels, width, height


def load_image(path):
    data = read_file(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.png' or data[:4] == b'\x89PNG':
        return decode_png(data)
    elif ext in ('.bmp',) or data[:2] == b'BM':
        return decode_bmp(data)
    else:
        # Try PNG first, then BMP
        try:
            return decode_png(data)
        except Exception:
            return decode_bmp(data)


# ---------------------------------------------------------------------------
# 2. Grayscale & Binarization
# ---------------------------------------------------------------------------

def to_grayscale(pixels):
    """Convert RGB pixel list to grayscale using ITU-R BT.601 luminance."""
    return [int(0.299 * r + 0.587 * g + 0.114 * b) for (r, g, b) in pixels]


def otsu_threshold(gray):
    """Compute Otsu's optimal binarization threshold from scratch."""
    hist = [0] * 256
    n = len(gray)
    for v in gray:
        hist[v] += 1

    total_sum = sum(i * hist[i] for i in range(256))
    sum_b = 0
    w_b = 0
    max_var = 0.0
    best_t = 128

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = n - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / w_b
        mean_f = (total_sum - sum_b) / w_f
        var = w_b * w_f * (mean_b - mean_f) ** 2
        if var > max_var:
            max_var = var
            best_t = t

    return best_t


def binarize(gray, width, height, threshold=None):
    """Return a 2D list of 1 (ink) / 0 (background)."""
    if threshold is None:
        threshold = otsu_threshold(gray)
    bin_img = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(1 if gray[y * width + x] < threshold else 0)
        bin_img.append(row)
    return bin_img


# ---------------------------------------------------------------------------
# 3. Segmentation — rows then columns
# ---------------------------------------------------------------------------

def find_row_spans(bin_img, height, width):
    """Find vertical spans (y_start, y_end) that contain ink pixels."""
    spans = []
    in_span = False
    start = 0
    for y in range(height):
        has_ink = any(bin_img[y][x] for x in range(width))
        if has_ink and not in_span:
            in_span = True
            start = y
        elif not has_ink and in_span:
            spans.append((start, y - 1))
            in_span = False
    if in_span:
        spans.append((start, height - 1))
    return spans


def find_col_spans(bin_img, y0, y1, width):
    """Find horizontal spans (x_start, x_end) with ink pixels in row band."""
    spans = []
    in_span = False
    start = 0
    for x in range(width):
        has_ink = any(bin_img[y][x] for y in range(y0, y1 + 1))
        if has_ink and not in_span:
            in_span = True
            start = x
        elif not has_ink and in_span:
            spans.append((start, x - 1))
            in_span = False
    if in_span:
        spans.append((start, width - 1))
    return spans


def merge_spans(spans, max_gap):
    """Merge adjacent spans that are within max_gap pixels of each other."""
    if not spans:
        return spans
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s - merged[-1][1] <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# ---------------------------------------------------------------------------
# 4. Glyph feature extraction — zone-based, font-agnostic
# ---------------------------------------------------------------------------

def _bitmap_to_2d(bitmap):
    """Convert BITMAP_FONT string list to a 2D list of 0/1 values."""
    return [[1 if c == 'X' else 0 for c in row] for row in bitmap]


def _tight_crop_2d(pixels_2d, height, width):
    """Return tightly cropped 2D pixel array and (new_h, new_w), or (None,0,0)."""
    top = 0
    while top < height and not any(pixels_2d[top][x] for x in range(width)):
        top += 1
    bottom = height - 1
    while bottom >= 0 and not any(pixels_2d[bottom][x] for x in range(width)):
        bottom -= 1
    if top > bottom:
        return None, 0, 0
    left = 0
    while left < width and not any(pixels_2d[y][left] for y in range(top, bottom + 1)):
        left += 1
    right = width - 1
    while right >= 0 and not any(pixels_2d[y][right] for y in range(top, bottom + 1)):
        right -= 1
    cropped = [pixels_2d[y][left:right + 1] for y in range(top, bottom + 1)]
    return cropped, bottom - top + 1, right - left + 1


def _zone_features(pixels_2d, height, width):
    """
    Extract a 24-dimensional feature vector from a 2D binary character image.
    All values are normalised to [0, 1] (aspect ratio capped at 3).

    Features:
      [0]       Aspect ratio  W/H (capped at 3)
      [1]       Overall ink density
      [2..13]   4-row × 3-col zone ink densities  (12 values)
      [14..16]  Horizontal crossing ratio at row 25%, 50%, 75%   (3 values)
      [17..19]  Vertical crossing ratio at col 25%, 50%, 75%     (3 values)
      [20..21]  Top-half / bottom-half ink density  (2 values)
      [22..23]  Left-half / right-half ink density  (2 values)
    """
    if height == 0 or width == 0:
        return None

    feats = []

    # 1. Aspect ratio
    feats.append(min(width / height, 3.0))

    # 2. Overall ink density
    total_ink = sum(pixels_2d[y][x] for y in range(height) for x in range(width))
    feats.append(total_ink / (height * width))

    # 3. Zone densities  (4 row-bands × 3 col-bands)
    ZONE_ROWS = 4
    ZONE_COLS = 3
    for zr in range(ZONE_ROWS):
        y0 = int(zr * height / ZONE_ROWS)
        y1 = max(y0 + 1, int((zr + 1) * height / ZONE_ROWS))
        y1 = min(y1, height)
        for zc in range(ZONE_COLS):
            x0 = int(zc * width / ZONE_COLS)
            x1 = max(x0 + 1, int((zc + 1) * width / ZONE_COLS))
            x1 = min(x1, width)
            area = (y1 - y0) * (x1 - x0)
            ink = sum(pixels_2d[y][x] for y in range(y0, y1) for x in range(x0, x1))
            feats.append(ink / area if area > 0 else 0.0)

    # 4. Horizontal crossing ratio at three heights
    for frac in (0.25, 0.5, 0.75):
        row = min(int(frac * height), height - 1)
        crossings = sum(
            1 for x in range(1, width)
            if pixels_2d[row][x] != pixels_2d[row][x - 1]
        )
        feats.append(crossings / max(width - 1, 1))

    # 5. Vertical crossing ratio at three column positions
    for frac in (0.25, 0.5, 0.75):
        col = min(int(frac * width), width - 1)
        crossings = sum(
            1 for y in range(1, height)
            if pixels_2d[y][col] != pixels_2d[y - 1][col]
        )
        feats.append(crossings / max(height - 1, 1))

    # 6. Top-half vs bottom-half ink density
    mid_y = height // 2
    top_ink = sum(pixels_2d[y][x] for y in range(mid_y) for x in range(width))
    bot_ink = sum(pixels_2d[y][x] for y in range(mid_y, height) for x in range(width))
    feats.append(top_ink / max(mid_y * width, 1))
    feats.append(bot_ink / max((height - mid_y) * width, 1))

    # 7. Left-half vs right-half ink density
    mid_x = width // 2
    left_ink = sum(pixels_2d[y][x] for y in range(height) for x in range(mid_x))
    right_ink = sum(pixels_2d[y][x] for y in range(height) for x in range(mid_x, width))
    feats.append(left_ink / max(height * mid_x, 1))
    feats.append(right_ink / max(height * (width - mid_x), 1))

    return feats  # length 24


def extract_features(bin_img, x0, x1, y0, y1):
    """
    Tightly crop the character within the given bounding box, then return
    (zone_feature_vector, ink_pixel_count) or None if the glyph is empty.
    """
    # Tight vertical crop
    char_y0 = y0
    while char_y0 <= y1 and not any(bin_img[char_y0][x] for x in range(x0, x1 + 1)):
        char_y0 += 1
    char_y1 = y1
    while char_y1 >= char_y0 and not any(bin_img[char_y1][x] for x in range(x0, x1 + 1)):
        char_y1 -= 1
    if char_y0 > char_y1:
        return None

    # Tight horizontal crop
    char_x0 = x0
    while char_x0 <= x1 and not any(bin_img[y][char_x0] for y in range(char_y0, char_y1 + 1)):
        char_x0 += 1
    char_x1 = x1
    while char_x1 >= char_x0 and not any(bin_img[y][char_x1] for y in range(char_y0, char_y1 + 1)):
        char_x1 -= 1

    H = char_y1 - char_y0 + 1
    W = char_x1 - char_x0 + 1
    if H <= 0 or W <= 0:
        return None

    pixels_2d = [
        [bin_img[y][x] for x in range(char_x0, char_x1 + 1)]
        for y in range(char_y0, char_y1 + 1)
    ]
    total_ink = sum(pixels_2d[y][x] for y in range(H) for x in range(W))
    feats = _zone_features(pixels_2d, H, W)
    if feats is None:
        return None
    return feats, total_ink


# ---------------------------------------------------------------------------
# 5. Reference glyph templates — drawn pixel-by-pixel, no libraries
# ---------------------------------------------------------------------------

# Each character template is a list of strings ('X' = ink, '.' = background).
# These are hand-crafted 5x7 pixel bitmaps for ASCII characters.

BITMAP_FONT = {
    '0': [
        ".XXX.",
        "X...X",
        "X..XX",
        "X.X.X",
        "XX..X",
        "X...X",
        ".XXX.",
    ],
    '1': [
        "..X..",
        ".XX..",
        "..X..",
        "..X..",
        "..X..",
        "..X..",
        ".XXX.",
    ],
    '2': [
        ".XXX.",
        "X...X",
        "....X",
        "..XX.",
        ".X...",
        "X....",
        "XXXXX",
    ],
    '3': [
        "XXXXX",
        "....X",
        "...X.",
        "..XX.",
        "....X",
        "X...X",
        ".XXX.",
    ],
    '4': [
        "...X.",
        "..XX.",
        ".X.X.",
        "X..X.",
        "XXXXX",
        "...X.",
        "...X.",
    ],
    '5': [
        "XXXXX",
        "X....",
        "XXXX.",
        "....X",
        "....X",
        "X...X",
        ".XXX.",
    ],
    '6': [
        ".XXX.",
        "X....",
        "X....",
        "XXXX.",
        "X...X",
        "X...X",
        ".XXX.",
    ],
    '7': [
        "XXXXX",
        "....X",
        "...X.",
        "..X..",
        ".X...",
        ".X...",
        ".X...",
    ],
    '8': [
        ".XXX.",
        "X...X",
        "X...X",
        ".XXX.",
        "X...X",
        "X...X",
        ".XXX.",
    ],
    '9': [
        ".XXX.",
        "X...X",
        "X...X",
        ".XXXX",
        "....X",
        "....X",
        ".XXX.",
    ],
    'A': [
        ".XXX.",
        "X...X",
        "X...X",
        "XXXXX",
        "X...X",
        "X...X",
        "X...X",
    ],
    'B': [
        "XXXX.",
        "X...X",
        "X...X",
        "XXXX.",
        "X...X",
        "X...X",
        "XXXX.",
    ],
    'C': [
        ".XXX.",
        "X...X",
        "X....",
        "X....",
        "X....",
        "X...X",
        ".XXX.",
    ],
    'D': [
        "XXXX.",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        "XXXX.",
    ],
    'E': [
        "XXXXX",
        "X....",
        "X....",
        "XXXX.",
        "X....",
        "X....",
        "XXXXX",
    ],
    'F': [
        "XXXXX",
        "X....",
        "X....",
        "XXXX.",
        "X....",
        "X....",
        "X....",
    ],
    'G': [
        ".XXX.",
        "X...X",
        "X....",
        "X.XXX",
        "X...X",
        "X...X",
        ".XXX.",
    ],
    'H': [
        "X...X",
        "X...X",
        "X...X",
        "XXXXX",
        "X...X",
        "X...X",
        "X...X",
    ],
    'I': [
        ".XXX.",
        "..X..",
        "..X..",
        "..X..",
        "..X..",
        "..X..",
        ".XXX.",
    ],
    'J': [
        "..XXX",
        "....X",
        "....X",
        "....X",
        "X...X",
        "X...X",
        ".XXX.",
    ],
    'K': [
        "X...X",
        "X..X.",
        "X.X..",
        "XX...",
        "X.X..",
        "X..X.",
        "X...X",
    ],
    'L': [
        "X....",
        "X....",
        "X....",
        "X....",
        "X....",
        "X....",
        "XXXXX",
    ],
    'M': [
        "X...X",
        "XX.XX",
        "X.X.X",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
    ],
    'N': [
        "X...X",
        "XX..X",
        "X.X.X",
        "X..XX",
        "X...X",
        "X...X",
        "X...X",
    ],
    'O': [
        ".XXX.",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        ".XXX.",
    ],
    'P': [
        "XXXX.",
        "X...X",
        "X...X",
        "XXXX.",
        "X....",
        "X....",
        "X....",
    ],
    'Q': [
        ".XXX.",
        "X...X",
        "X...X",
        "X...X",
        "X.X.X",
        "X..XX",
        ".XXXX",
    ],
    'R': [
        "XXXX.",
        "X...X",
        "X...X",
        "XXXX.",
        "X.X..",
        "X..X.",
        "X...X",
    ],
    'S': [
        ".XXX.",
        "X...X",
        "X....",
        ".XXX.",
        "....X",
        "X...X",
        ".XXX.",
    ],
    'T': [
        "XXXXX",
        "..X..",
        "..X..",
        "..X..",
        "..X..",
        "..X..",
        "..X..",
    ],
    'U': [
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        ".XXX.",
    ],
    'V': [
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        ".X.X.",
        ".X.X.",
        "..X..",
    ],
    'W': [
        "X...X",
        "X...X",
        "X...X",
        "X.X.X",
        "X.X.X",
        "XX.XX",
        "X...X",
    ],
    'X': [
        "X...X",
        "X...X",
        ".X.X.",
        "..X..",
        ".X.X.",
        "X...X",
        "X...X",
    ],
    'Y': [
        "X...X",
        "X...X",
        ".X.X.",
        "..X..",
        "..X..",
        "..X..",
        "..X..",
    ],
    'Z': [
        "XXXXX",
        "....X",
        "...X.",
        "..X..",
        ".X...",
        "X....",
        "XXXXX",
    ],
    'a': [
        ".....",
        ".....",
        ".XXX.",
        "....X",
        ".XXXX",
        "X...X",
        ".XXXX",
    ],
    'b': [
        "X....",
        "X....",
        "XXXX.",
        "X...X",
        "X...X",
        "X...X",
        "XXXX.",
    ],
    'c': [
        ".....",
        ".....",
        ".XXX.",
        "X....",
        "X....",
        "X...X",
        ".XXX.",
    ],
    'd': [
        "....X",
        "....X",
        ".XXXX",
        "X...X",
        "X...X",
        "X...X",
        ".XXXX",
    ],
    'e': [
        ".....",
        ".....",
        ".XXX.",
        "X...X",
        "XXXXX",
        "X....",
        ".XXX.",
    ],
    'f': [
        "..XX.",
        ".X...",
        ".X...",
        "XXXX.",
        ".X...",
        ".X...",
        ".X...",
    ],
    'g': [
        ".....",
        ".XXXX",
        "X...X",
        "X...X",
        ".XXXX",
        "....X",
        ".XXX.",
    ],
    'h': [
        "X....",
        "X....",
        "XXXX.",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
    ],
    'i': [
        "..X..",
        ".....",
        ".XX..",
        "..X..",
        "..X..",
        "..X..",
        ".XXX.",
    ],
    'j': [
        "...X.",
        ".....",
        "..XX.",
        "...X.",
        "...X.",
        "X..X.",
        ".XX..",
    ],
    'k': [
        "X....",
        "X....",
        "X..X.",
        "X.X..",
        "XX...",
        "X.X..",
        "X..X.",
    ],
    'l': [
        ".XX..",
        "..X..",
        "..X..",
        "..X..",
        "..X..",
        "..X..",
        ".XXX.",
    ],
    'm': [
        ".....",
        ".....",
        "XX.X.",
        "X.X.X",
        "X.X.X",
        "X.X.X",
        "X...X",
    ],
    'n': [
        ".....",
        ".....",
        "XXXX.",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
    ],
    'o': [
        ".....",
        ".....",
        ".XXX.",
        "X...X",
        "X...X",
        "X...X",
        ".XXX.",
    ],
    'p': [
        ".....",
        "XXXX.",
        "X...X",
        "X...X",
        "XXXX.",
        "X....",
        "X....",
    ],
    'q': [
        ".....",
        ".XXXX",
        "X...X",
        "X...X",
        ".XXXX",
        "....X",
        "....X",
    ],
    'r': [
        ".....",
        ".....",
        "X.XX.",
        "XX...",
        "X....",
        "X....",
        "X....",
    ],
    's': [
        ".....",
        ".....",
        ".XXX.",
        "X....",
        ".XXX.",
        "....X",
        ".XXX.",
    ],
    't': [
        ".X...",
        ".X...",
        "XXXX.",
        ".X...",
        ".X...",
        ".X...",
        "..XX.",
    ],
    'u': [
        ".....",
        ".....",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        ".XXX.",
    ],
    'v': [
        ".....",
        ".....",
        "X...X",
        "X...X",
        ".X.X.",
        ".X.X.",
        "..X..",
    ],
    'w': [
        ".....",
        ".....",
        "X...X",
        "X.X.X",
        "X.X.X",
        "XX.XX",
        "X...X",
    ],
    'x': [
        ".....",
        ".....",
        "X...X",
        ".X.X.",
        "..X..",
        ".X.X.",
        "X...X",
    ],
    'y': [
        ".....",
        "X...X",
        "X...X",
        ".XXXX",
        "....X",
        "X...X",
        ".XXX.",
    ],
    'z': [
        ".....",
        ".....",
        "XXXXX",
        "...X.",
        "..X..",
        ".X...",
        "XXXXX",
    ],
    ' ': [
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
    ],
    '.': [
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        "..X..",
        "..X..",
    ],
    ',': [
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        "..X..",
        ".X...",
    ],
    '!': [
        "..X..",
        "..X..",
        "..X..",
        "..X..",
        ".....",
        "..X..",
        "..X..",
    ],
    '?': [
        ".XXX.",
        "X...X",
        "....X",
        "..XX.",
        "..X..",
        ".....",
        "..X..",
    ],
    ':': [
        ".....",
        "..X..",
        "..X..",
        ".....",
        "..X..",
        "..X..",
        ".....",
    ],
    ';': [
        ".....",
        "..X..",
        "..X..",
        ".....",
        "..X..",
        "..X..",
        ".X...",
    ],
    '-': [
        ".....",
        ".....",
        ".....",
        "XXXXX",
        ".....",
        ".....",
        ".....",
    ],
    '_': [
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        "XXXXX",
    ],
    "'": [
        "..X..",
        "..X..",
        ".X...",
        ".....",
        ".....",
        ".....",
        ".....",
    ],
    '"': [
        ".X.X.",
        ".X.X.",
        "X.X..",
        ".....",
        ".....",
        ".....",
        ".....",
    ],
    '(': [
        "...X.",
        "..X..",
        ".X...",
        ".X...",
        ".X...",
        "..X..",
        "...X.",
    ],
    ')': [
        ".X...",
        "..X..",
        "...X.",
        "...X.",
        "...X.",
        "..X..",
        ".X...",
    ],
    '/': [
        "....X",
        "....X",
        "...X.",
        "..X..",
        ".X...",
        "X....",
        "X....",
    ],
    '+': [
        ".....",
        "..X..",
        "..X..",
        "XXXXX",
        "..X..",
        "..X..",
        ".....",
    ],
    '=': [
        ".....",
        ".....",
        "XXXXX",
        ".....",
        "XXXXX",
        ".....",
        ".....",
    ],
    '@': [
        ".XXX.",
        "X...X",
        "X.XX.",
        "X.X.X",
        "X.XXX",
        "X....",
        ".XXXX",
    ],
    '#': [
        ".X.X.",
        ".X.X.",
        "XXXXX",
        ".X.X.",
        "XXXXX",
        ".X.X.",
        ".X.X.",
    ],
    '$': [
        "..X..",
        ".XXXX",
        "X.X..",
        ".XXX.",
        "..X.X",
        "XXXX.",
        "..X..",
    ],
    '%': [
        "XX...",
        "XX..X",
        "...X.",
        "..X..",
        ".X...",
        "X..XX",
        "...XX",
    ],
    '&': [
        ".XX..",
        "X..X.",
        "X.X..",
        ".XX..",
        "X.X.X",
        "X..X.",
        ".XX.X",
    ],
    '*': [
        ".....",
        "X.X.X",
        ".XXX.",
        "XXXXX",
        ".XXX.",
        "X.X.X",
        ".....",
    ],
    '<': [
        "...X.",
        "..X..",
        ".X...",
        "X....",
        ".X...",
        "..X..",
        "...X.",
    ],
    '>': [
        ".X...",
        "..X..",
        "...X.",
        "....X",
        "...X.",
        "..X..",
        ".X...",
    ],
    '[': [
        ".XXX.",
        ".X...",
        ".X...",
        ".X...",
        ".X...",
        ".X...",
        ".XXX.",
    ],
    ']': [
        ".XXX.",
        "...X.",
        "...X.",
        "...X.",
        "...X.",
        "...X.",
        ".XXX.",
    ],
}


def bitmap_to_flat(bitmap):
    """Convert bitmap string list to flat binary list (row-major). Kept for reference."""
    return [1 if c == 'X' else 0 for row in bitmap for c in row]


def compute_template_features():
    """
    Return pre-computed zone-feature vectors derived from Arial-rendered glyphs.
    These match real-world document fonts far better than hand-crafted 5x7 bitmaps.
    Falls back to BITMAP_FONT zone features if the rendered table is unavailable.
    """
    try:
        from font_templates import RENDERED_TEMPLATES
        return RENDERED_TEMPLATES
    except ImportError:
        pass
    # Fallback: compute from BITMAP_FONT
    templates = {}
    for ch, bitmap in BITMAP_FONT.items():
        if ch == ' ':
            continue
        pixels_2d = _bitmap_to_2d(bitmap)
        H = len(pixels_2d)
        W = len(pixels_2d[0]) if H > 0 else 0
        cropped, new_h, new_w = _tight_crop_2d(pixels_2d, H, W)
        if cropped is None:
            continue
        feats = _zone_features(cropped, new_h, new_w)
        if feats is not None:
            templates[ch] = feats
    return templates


# Minimum real ink pixels required before we attempt matching (noise gate).
_MIN_INK = 3
_MAX_DIST = 2.0


def match_character(feat_tuple, templates):
    """
    Zone-feature nearest-neighbour matching using Euclidean distance.
    feat_tuple = (feature_vector, ink_count) as returned by extract_features().
    Returns the best-matching character, or '?' if confidence is too low.
    """
    feats, ink = feat_tuple

    if ink < _MIN_INK:
        return '?'

    best_dist = float('inf')
    best_ch = '?'

    for ch, tfeats in templates.items():
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(feats, tfeats)))
        # Break ties by preferring lowercase (more common in body text)
        if dist < best_dist or (abs(dist - best_dist) < 1e-6 and ch.islower()):
            best_dist = dist
            best_ch = ch

    return best_ch if best_dist <= _MAX_DIST else '?'


# ---------------------------------------------------------------------------
# 6. Full OCR pipeline
# ---------------------------------------------------------------------------

def ocr(image_path, threshold=None, row_merge_gap=3, col_merge_gap=5):
    print(f"\n{'='*55}")
    print(f"  OCR From Scratch — Pure Python")
    print(f"  Image: {image_path}")
    print(f"{'='*55}")

    # Load image
    print("\n[1/5] Loading image...")
    pixels, width, height = load_image(image_path)
    print(f"  Dimensions: {width} x {height} px  ({width * height:,} pixels)")

    # Grayscale
    print("[2/5] Converting to grayscale...")
    gray = to_grayscale(pixels)

    # Binarize
    print("[3/5] Binarizing (Otsu's method)...")
    bin_img = binarize(gray, width, height, threshold)

    # Segment
    print("[4/5] Segmenting text regions...")
    row_spans = find_row_spans(bin_img, height, width)
    row_spans = merge_spans(row_spans, row_merge_gap)
    print(f"  Found {len(row_spans)} text line(s)")

    # Build templates
    print("[5/5] Matching characters...")
    templates = compute_template_features()

    lines = []
    total_chars = 0

    for line_idx, (y0, y1) in enumerate(row_spans):
        line_h = y1 - y0 + 1
        if line_h < 4:
            continue

        col_spans = find_col_spans(bin_img, y0, y1, width)
        col_spans = merge_spans(col_spans, col_merge_gap)

        chars = []
        prev_x1 = -1
        prev_w = 1

        for x0, x1 in col_spans:
            char_w = x1 - x0 + 1
            if char_w < 2:
                continue

            # Insert space if gap is large relative to char width
            if prev_x1 >= 0:
                gap = x0 - prev_x1
                if gap > max(prev_w, char_w) * 0.8:
                    chars.append(' ')

            feat = extract_features(bin_img, x0, x1, y0, y1)
            if feat is not None and feat[1] >= _MIN_INK:  # feat = (pixels, ink)
                ch = match_character(feat, templates)
                chars.append(ch)
                total_chars += 1

            prev_x1 = x1
            prev_w = char_w

        line_text = ''.join(chars).strip()
        if line_text:
            lines.append(line_text)

    result = '\n'.join(lines)

    print(f"\n{'='*55}")
    print("  EXTRACTED TEXT:")
    print(f"{'='*55}")
    print(result if result else "(no text detected)")
    print(f"{'='*55}")
    print(f"  Lines: {len(lines)}  |  Characters matched: {total_chars}")
    print(f"{'='*55}\n")

    return result


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr_from_scratch.py <image_path> [threshold]")
        print()
        print("  image_path  — path to a PNG or BMP image file")
        print("  threshold   — optional binarization threshold (0-255)")
        print("                default: auto (Otsu's method)")
        print()
        print("Examples:")
        print("  python ocr_from_scratch.py scan.png")
        print("  python ocr_from_scratch.py document.bmp 120")
        sys.exit(1)

    image_path = sys.argv[1]
    threshold = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    ocr(image_path, threshold=threshold)


if __name__ == '__main__':
    main()
