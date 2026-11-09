"""
Генерує іконку PC Benchmark (блискавка на синьому колі) у форматі .ico
без сторонніх бібліотек: растеризація з 4×4 згладжуванням, PNG-кадри всередині ICO.

    python tools/make_icon.py assets/pcbench.ico
"""
import struct
import sys
import zlib

ACCENT = (79, 140, 255)
BOLT = [(0.585, 0.13), (0.31, 0.55), (0.475, 0.55), (0.415, 0.87), (0.69, 0.45), (0.525, 0.45), (0.585, 0.13)]
SIZES = (16, 24, 32, 48, 64, 128, 256)


def inside_polygon(x, y, poly):
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def render(size, ss=4):
    rows = []
    for py in range(size):
        row = bytearray([0])  # тип фільтра PNG
        for px in range(size):
            circle = bolt = 0
            for sy in range(ss):
                for sx in range(ss):
                    x = (px + (sx + 0.5) / ss) / size
                    y = (py + (sy + 0.5) / ss) / size
                    if (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.47 ** 2:
                        circle += 1
                        if inside_polygon(x, y, BOLT):
                            bolt += 1
            n = ss * ss
            a = circle / n
            w = bolt / circle if circle else 0.0
            r, g, b = (round(c + (255 - c) * w) for c in ACCENT)
            row += bytes((r, g, b, round(255 * a)))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def make_ico(path):
    images = [render(s) for s in SIZES]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for s, png in zip(SIZES, images):
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(png), offset)
        offset += len(png)
    with open(path, "wb") as f:
        f.write(header + entries + b"".join(images))


if __name__ == "__main__":
    make_ico(sys.argv[1] if len(sys.argv) > 1 else "pcbench.ico")
