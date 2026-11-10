"""forge.sprite - hand-authored pixel art.

A ``Sprite`` is a small grid of palette indices (0 = transparent). Write it as
rows of characters, or draw into it with ``rect`` / ``line`` / ``ellipse`` /
``put``; then mirror, outline, shade and export at any integer scale.
"""
import numpy as np
from PIL import Image

from .post import PALETTES, _rgba


class Sprite:
    def __init__(self, width, height, palette):
        """palette: list of colours ('#rrggbb'); index 0 is always transparent,
        so palette[0] becomes index 1."""
        if isinstance(palette, str):
            palette = PALETTES[palette]
        self.palette = list(palette)
        self.px = np.zeros((int(height), int(width)), dtype=np.int16)

    # ---- construction
    @classmethod
    def from_rows(cls, rows, key):
        """rows: equal-length strings; key: {char: colour}. '.' or ' ' = transparent.

            Sprite.from_rows(["..hh..", ".hvvh.", ".bbbb."], {"h": "#c8ccd4", "v": "#35c4d6", "b": "#2b3445"})
        """
        rows = [r for r in rows]
        w = max(len(r) for r in rows)
        chars = list(key)
        sp = cls(w, len(rows), [key[c] for c in chars])
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch in key:
                    sp.px[y, x] = chars.index(ch) + 1
                elif ch not in ". ":
                    raise ValueError(f"row {y}: character '{ch}' is not in the key")
        return sp

    @property
    def width(self):
        return self.px.shape[1]

    @property
    def height(self):
        return self.px.shape[0]

    def color(self, c):
        """Index for a colour, adding it to the palette if new."""
        if isinstance(c, int):
            return c
        if c not in self.palette:
            self.palette.append(c)
        return self.palette.index(c) + 1

    # ---- drawing (all coordinates are pixel indices, y down)
    def put(self, x, y, c):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.px[y, x] = self.color(c)
        return self

    def rect(self, x0, y0, x1, y1, c, fill=True):
        """Inclusive rectangle."""
        i = self.color(c)
        x0, x1, y0, y1 = max(min(x0, x1), 0), max(x0, x1), max(min(y0, y1), 0), max(y0, y1)
        if fill:
            self.px[y0:y1 + 1, x0:x1 + 1] = i
        else:
            self.px[y0, x0:x1 + 1] = self.px[min(y1, self.height - 1), x0:x1 + 1] = i
            self.px[y0:y1 + 1, x0] = self.px[y0:y1 + 1, min(x1, self.width - 1)] = i
        return self

    def line(self, x0, y0, x1, y1, c):
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for k in range(n + 1):
            self.put(round(x0 + (x1 - x0) * k / n), round(y0 + (y1 - y0) * k / n), c)
        return self

    def ellipse(self, cx, cy, rx, ry, c, fill=True):
        """Centre may be fractional (e.g. 7.5) for even-sized shapes."""
        i = self.color(c)
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        d = ((xx - cx) / max(rx, 0.5)) ** 2 + ((yy - cy) / max(ry, 0.5)) ** 2
        mask = d <= 1.0
        if not fill:
            inner = ((xx - cx) / max(rx - 1, 0.5)) ** 2 + ((yy - cy) / max(ry - 1, 0.5)) ** 2 <= 1.0
            mask &= ~inner
        self.px[mask] = i
        return self

    def replace(self, old, new):
        self.px[self.px == self.color(old)] = self.color(new)
        return self

    # ---- whole-sprite operations
    def mirror_right(self):
        """Copy the left half onto the right half (symmetric characters)."""
        w = self.width
        self.px[:, w - w // 2:] = self.px[:, : w // 2][:, ::-1]
        return self

    def flipped(self):
        s = Sprite(self.width, self.height, self.palette)
        s.px = self.px[:, ::-1].copy()
        return s

    def outline(self, c="#0b0d12"):
        """Grow the canvas by 1px and add an outline around the silhouette."""
        i = self.color(c)
        p = np.pad(self.px, 1)
        solid = p > 0
        near = np.zeros_like(solid)
        near[1:, :] |= solid[:-1, :]
        near[:-1, :] |= solid[1:, :]
        near[:, 1:] |= solid[:, :-1]
        near[:, :-1] |= solid[:, 1:]
        p[near & ~solid] = i
        self.px = p
        return self

    def shade(self, base, shadow, light=None):
        """Auto-shade one colour: its bottom/right rim becomes ``shadow`` and,
        if given, its top/left rim becomes ``light``."""
        b = self.color(base)
        m = self.px == b
        below = np.zeros_like(m)
        below[:-1, :] = m[1:, :]
        right = np.zeros_like(m)
        right[:, :-1] = m[:, 1:]
        above = np.zeros_like(m)
        above[1:, :] = m[:-1, :]
        left = np.zeros_like(m)
        left[:, 1:] = m[:, :-1]
        if light is not None:
            self.px[m & (~above | ~left)] = self.color(light)
        self.px[m & (~below | ~right)] = self.color(shadow)
        return self

    # ---- export
    def to_image(self, scale=1):
        table = np.array([(0, 0, 0, 0)] + [_rgba(c) for c in self.palette], dtype=np.uint8)
        img = Image.fromarray(table[self.px], "RGBA")
        if scale != 1:
            img = img.resize((img.width * int(scale), img.height * int(scale)), Image.NEAREST)
        return img

    def save(self, path, scale=8):
        self.to_image(scale).save(path)
        return path
