"""forge.paint - anti-aliased 2D drawing for flat / vector / poster styles.

``Canvas`` draws with Pillow on a supersampled buffer and downsamples on
export, so edges are smooth at any output size up to 4K and beyond.
Coordinates are in OUTPUT pixels (floats are fine); y grows downward.
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .post import _rgba
from .noise import noise2d


class Canvas:
    def __init__(self, width, height, bg=None, ss=None):
        """bg None = transparent. ss = supersampling factor (auto: 4 for small
        images down to 2 at 4K, keeping memory reasonable)."""
        self.width, self.height = int(width), int(height)
        self.ss = ss or (4 if width * height <= 1.5e6 else 3 if width * height <= 4e6 else 2)
        self.img = Image.new("RGBA", (self.width * self.ss, self.height * self.ss),
                             (0, 0, 0, 0) if bg is None else _rgba(bg))

    # ---- helpers
    def _s(self, v):
        return v * self.ss

    def _pts(self, pts):
        return [(x * self.ss, y * self.ss) for x, y in pts]

    def _layer(self):
        layer = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
        return layer, ImageDraw.Draw(layer)

    def _merge(self, layer, opacity=1.0, blur=0.0):
        if blur:
            layer = layer.filter(ImageFilter.GaussianBlur(blur * self.ss))
        if opacity < 1.0:
            layer.putalpha(layer.getchannel("A").point(lambda v: int(v * opacity)))
        self.img.alpha_composite(layer)
        return self

    # ---- shapes: every shape takes fill, outline, width, opacity, blur
    def polygon(self, pts, fill=None, outline=None, width=1.0, opacity=1.0, blur=0.0):
        layer, d = self._layer()
        P = self._pts(pts)
        if fill is not None:
            d.polygon(P, fill=_rgba(fill))
        if outline is not None:
            d.line(P + [P[0]], fill=_rgba(outline), width=max(1, round(width * self.ss)), joint="curve")
        return self._merge(layer, opacity, blur)

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1.0, radius=0.0, opacity=1.0, blur=0.0):
        layer, d = self._layer()
        d.rounded_rectangle([self._s(x0), self._s(y0), self._s(x1), self._s(y1)], radius=self._s(radius),
                            fill=None if fill is None else _rgba(fill),
                            outline=None if outline is None else _rgba(outline),
                            width=max(1, round(width * self.ss)))
        return self._merge(layer, opacity, blur)

    def ellipse(self, cx, cy, rx, ry=None, fill=None, outline=None, width=1.0, opacity=1.0, blur=0.0):
        ry = rx if ry is None else ry
        layer, d = self._layer()
        d.ellipse([self._s(cx - rx), self._s(cy - ry), self._s(cx + rx), self._s(cy + ry)],
                  fill=None if fill is None else _rgba(fill),
                  outline=None if outline is None else _rgba(outline), width=max(1, round(width * self.ss)))
        return self._merge(layer, opacity, blur)

    def line(self, pts, color, width=2.0, opacity=1.0, blur=0.0, round_caps=True):
        layer, d = self._layer()
        P = self._pts(pts)
        w = max(1, round(width * self.ss))
        d.line(P, fill=_rgba(color), width=w, joint="curve")
        if round_caps:
            for x, y in (P[0], P[-1]):
                d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=_rgba(color))
        return self._merge(layer, opacity, blur)

    def curve(self, p0, p1, p2, p3, color, width=2.0, steps=48, **kw):
        """Cubic Bezier stroke."""
        pts = []
        for i in range(steps + 1):
            t = i / steps
            a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
            pts.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
        return self.line(pts, color, width, **kw)

    def text(self, x, y, s, size=32, fill="#ffffff", anchor="la", font=None, opacity=1.0):
        layer, d = self._layer()
        try:
            f = ImageFont.truetype(font or "DejaVuSans-Bold.ttf", round(size * self.ss))
        except OSError:
            f = ImageFont.load_default()
        d.text((self._s(x), self._s(y)), s, font=f, fill=_rgba(fill), anchor=anchor)
        return self._merge(layer, opacity)

    # ---- fills
    def gradient(self, stops, angle=90.0, box=None, opacity=1.0, mask=None):
        """Linear gradient. stops = [(0.0, colour), (1.0, colour), ...];
        angle 90 = top to bottom, 0 = left to right. ``box`` = (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = box or (0, 0, self.width, self.height)
        X0, Y0, X1, Y1 = (round(v * self.ss) for v in (x0, y0, x1, y1))
        w, h = X1 - X0, Y1 - Y0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = math.radians(angle)
        t = (xx / max(w - 1, 1) - 0.5) * math.cos(a) + (yy / max(h - 1, 1) - 0.5) * math.sin(a) + 0.5
        return self._paste_ramp(np.clip(t, 0, 1), stops, (X0, Y0), opacity, mask)

    def radial(self, cx, cy, radius, stops, opacity=1.0, mask=None):
        """Radial gradient centred on (cx, cy); stop 0 = centre, 1 = radius."""
        X0, Y0 = round((cx - radius) * self.ss), round((cy - radius) * self.ss)
        n = round(2 * radius * self.ss)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        t = np.hypot(xx - n / 2, yy - n / 2) / (n / 2)
        return self._paste_ramp(np.clip(t, 0, 1), stops, (X0, Y0), opacity, mask)

    def _paste_ramp(self, t, stops, origin, opacity, mask):
        pos = np.array([s[0] for s in stops], dtype=np.float32)
        cols = np.array([_rgba(s[1]) for s in stops], dtype=np.float32)
        out = np.stack([np.interp(t, pos, cols[:, c]) for c in range(4)], axis=2)
        out[..., 3] *= opacity
        tile = Image.fromarray(out.astype(np.uint8), "RGBA")
        layer = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
        layer.paste(tile, origin)
        if mask is not None:  # mask: another Canvas whose alpha limits the fill
            layer.putalpha(Image.fromarray(
                (np.asarray(layer.getchannel("A"), dtype=np.float32) *
                 np.asarray(mask.img.getchannel("A"), dtype=np.float32) / 255).astype(np.uint8)))
        self.img.alpha_composite(layer)
        return self

    def grain(self, amount=0.04, scale=1.5, seed=0):
        """Add subtle paper/film texture to whatever is already drawn."""
        a = np.asarray(self.img).astype(np.float32)
        n = (noise2d(a.shape[:2], scale * self.ss, seed) - 0.5) * 2 * amount * 255
        a[..., :3] = np.clip(a[..., :3] + n[..., None], 0, 255)
        self.img = Image.fromarray(a.astype(np.uint8), "RGBA")
        return self

    def paste(self, image, x=0, y=0, scale=1.0):
        """Composite another PIL image (e.g. a render) with its top-left at (x, y)."""
        im = image.convert("RGBA")
        im = im.resize((round(im.width * scale * self.ss), round(im.height * scale * self.ss)), Image.LANCZOS)
        self.img.alpha_composite(im, (round(x * self.ss), round(y * self.ss)))
        return self

    # ---- export
    def to_image(self):
        return self.img.resize((self.width, self.height), Image.LANCZOS) if self.ss > 1 else self.img.copy()

    def save(self, path):
        self.to_image().save(path)
        return path
