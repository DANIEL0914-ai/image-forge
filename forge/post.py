"""forge.post - turn render buffers into finished images, and restyle images.

develop()   Frame -> PIL RGBA (cavity shading, bloom, tone mapping, grain)
sheet()     lay images side by side (turnarounds, sprite sheets, contact sheets)
pixelate()  any image -> pixel art (downsample, palette, outline)
upscale()   crisp nearest-neighbour enlargement for pixel art
on_background() / checker()  flatten transparency for previews
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

F = np.float32


def _srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _aces(x):
    x = x * 0.6  # so exposure 1.0 keeps mid-greys where they were
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def _cavity(frame, r):
    """Screen-space curvature: >0 on convex edges, <0 in creases."""
    n, a = frame.normal, frame.alpha
    z = np.where(np.isfinite(frame.depth), frame.depth, 1e6)
    cur = np.zeros(a.shape, dtype=F)
    lim = 14 * r * frame.pixel
    # horizontal: right neighbour normal.x minus left neighbour normal.x
    v = (a[:, 2 * r:] >= 1) & (a[:, :-2 * r] >= 1) & (np.abs(z[:, 2 * r:] - z[:, :-2 * r]) < lim)
    cur[:, r:-r] += np.where(v, n[0][:, 2 * r:] - n[0][:, :-2 * r], 0)
    # vertical: image rows grow downward, camera up is toward smaller rows
    v = (a[2 * r:, :] >= 1) & (a[:-2 * r, :] >= 1) & (np.abs(z[2 * r:, :] - z[:-2 * r, :]) < lim)
    cur[r:-r, :] += np.where(v, n[1][:-2 * r, :] - n[1][2 * r:, :], 0)
    return cur


def develop(frame, exposure=1.0, cavity=0.5, bloom=0.5, bloom_threshold=1.1, bloom_size=0.012,
            tonemap="aces", saturation=1.0, contrast=1.0, grain=0.0, seed=0):
    """Finish a render. Returns a PIL RGBA image (transparent where nothing was hit).

    cavity     edge highlight / crease darkening from screen-space curvature (0 = off)
    bloom      glow around emissives and hot highlights (0 = off)
    bloom_size glow radius as a fraction of image height
    tonemap    'aces' (filmic) or 'none' (clip)
    """
    rgb = frame.rgb * exposure
    a = frame.alpha
    H, W = a.shape
    if cavity:
        r1 = max(1, round(H / 1100))
        c = _cavity(frame, r1) + 0.6 * _cavity(frame, r1 * 3)
        c = ndi.gaussian_filter(c, r1 * 0.6)
        gain = 1.0 + cavity * (0.55 * np.clip(c, 0, 1.2) - 0.75 * np.clip(-c, 0, 1.0))
        rgb = rgb * np.maximum(gain, 0.15)[..., None]
    glow = None
    if bloom:
        k = max(1, H // 540)
        small = rgb[: H // k * k, : W // k * k].reshape(H // k, k, W // k, k, 3).mean(axis=(1, 3))
        lum = small.max(axis=2)
        hot = small * (np.clip(lum - bloom_threshold, 0, None) / np.maximum(lum, 1e-6))[..., None]
        sig = max(1.0, bloom_size * H / k)
        g = sum(ndi.gaussian_filter(hot, (sig * m, sig * m, 0)) * w
                for m, w in ((1.0, 0.55), (3.0, 0.30), (7.0, 0.15)))
        g = np.ascontiguousarray(g, dtype=F)
        glow = np.stack([np.asarray(Image.fromarray(g[..., i], "F").resize((W, H), Image.BILINEAR))
                         for i in range(3)], axis=2) * bloom
    out_a = a.copy()
    if glow is not None:
        rgb = rgb + glow
        out_a = np.clip(a + (1 - a) * np.clip(glow.max(axis=2) * 1.6, 0, 1), 0, 1)
    col = rgb / np.maximum(out_a, 1e-4)[..., None]
    col = _aces(col) if tonemap == "aces" else np.clip(col, 0, 1)
    col = _srgb(col)
    if contrast != 1.0:
        col = np.clip((col - 0.5) * contrast + 0.5, 0, 1)
    if saturation != 1.0:
        g = col @ np.array([0.2126, 0.7152, 0.0722], dtype=F)
        col = np.clip(g[..., None] + (col - g[..., None]) * saturation, 0, 1)
    if grain:
        rng = np.random.default_rng(seed)
        col = np.clip(col + rng.normal(0, grain, (H, W, 1)).astype(F), 0, 1)
    out = np.dstack([col, out_a])
    out[out_a <= 0] = 0
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")


# ------------------------------------------------------------------ layout
def sheet(images, cols=None, gap=0, bg=None, pad=0):
    """Arrange same-size images in a grid. ``bg`` None keeps transparency."""
    cols = cols or len(images)
    rows = -(-len(images) // cols)
    w, h = images[0].size
    out = Image.new("RGBA", (pad * 2 + cols * w + (cols - 1) * gap, pad * 2 + rows * h + (rows - 1) * gap),
                    (0, 0, 0, 0) if bg is None else _rgba(bg))
    for i, im in enumerate(images):
        im = im.convert("RGBA")
        out.alpha_composite(im, (pad + (i % cols) * (w + gap), pad + (i // cols) * (h + gap)))
    return out


def _rgba(c):
    if isinstance(c, str):
        s = c.lstrip("#")
        c = tuple(int(s[i:i + 2], 16) for i in range(0, len(s), 2))
    c = tuple(int(v * 255) if isinstance(v, float) else int(v) for v in c)
    return c if len(c) == 4 else c + (255,)


def on_background(img, color="#3a3d42"):
    """Flatten transparency onto a solid colour (for previews / JPEG export)."""
    bgd = Image.new("RGBA", img.size, _rgba(color))
    bgd.alpha_composite(img.convert("RGBA"))
    return bgd.convert("RGB")


def checker(img, cell=16, light=205, dark=160):
    """Flatten onto a transparency checkerboard so alpha edges can be inspected."""
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    c = np.where(((xx // cell) + (yy // cell)) % 2 == 0, light, dark).astype(np.uint8)
    bgd = Image.fromarray(np.dstack([c, c, c, np.full_like(c, 255)]), "RGBA")
    bgd.alpha_composite(img.convert("RGBA"))
    return bgd.convert("RGB")


def crop_to_content(img, pad=0):
    """Trim fully transparent margins."""
    box = img.getchannel("A").getbbox()
    if box is None:
        return img
    l, t, r, b = box
    return img.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, img.width), min(b + pad, img.height)))


# --------------------------------------------------------------- pixel art
PALETTES = {
    "pico8": ["#000000", "#1d2b53", "#7e2553", "#008751", "#ab5236", "#5f574f", "#c2c3c7", "#fff1e8",
              "#ff004d", "#ffa300", "#ffec27", "#00e436", "#29adff", "#83769c", "#ff77a8", "#ffccaa"],
    "gameboy": ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"],
    "mono4": ["#111111", "#555555", "#aaaaaa", "#f4f4f4"],
    "steel16": ["#0b0d12", "#1a1f2b", "#2b3445", "#3f4c61", "#586a82", "#7a8ea6", "#a3b5c8", "#d6e0ea",
                "#f6f9fc", "#12343f", "#1d6b78", "#35c4d6", "#a8f4ff", "#5a2a14", "#c8601c", "#ffb347"],
}


def _pal_array(palette):
    if isinstance(palette, str):
        palette = PALETTES[palette]
    return np.array([_rgba(c)[:3] for c in palette], dtype=np.float32)


def quantize(img, palette=None, colors=16, dither=False):
    """Reduce an RGBA image to a fixed palette (name, list of colours) or to
    ``colors`` automatically chosen colours. Alpha is kept as-is."""
    rgba = np.asarray(img.convert("RGBA")).copy()
    opaque = rgba[..., 3] > 0
    if not opaque.any():
        return img
    if palette is None:
        flat = Image.fromarray(rgba[opaque][None, :, :3], "RGB")
        q = flat.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.Dither.NONE)
        pal = np.array(q.getpalette()[: colors * 3], dtype=np.float32).reshape(-1, 3)
        pal = pal[: len(q.getcolors())]
    else:
        pal = _pal_array(palette)
    src = rgba[..., :3].astype(np.float32)
    if dither:  # ordered 4x4 Bayer
        b = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32) / 16 - 0.47
        h, w = src.shape[:2]
        src = src + np.tile(b, (h // 4 + 1, w // 4 + 1))[:h, :w, None] * (255.0 / max(len(pal) ** (1 / 3), 2))
    wgt = np.array([0.30, 0.59, 0.11], dtype=np.float32)
    d = ((src[:, :, None, :] - pal[None, None, :, :]) ** 2 * wgt).sum(axis=3)
    rgba[..., :3] = pal[np.argmin(d, axis=2)].astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def outline(img, color="#0b0d12", inner=False):
    """1-pixel outline around the opaque silhouette (outside it unless inner=True)."""
    rgba = np.asarray(img.convert("RGBA")).copy()
    solid = rgba[..., 3] > 0
    grow = ndi.binary_dilation(solid, structure=ndi.generate_binary_structure(2, 1))
    edge = (solid & ~ndi.binary_erosion(solid, border_value=0)) if inner else (grow & ~solid)
    rgba[edge] = _rgba(color)
    return Image.fromarray(rgba, "RGBA")


def pixelate(img, height, palette=None, colors=16, outline_color="#0b0d12", dither=False,
             alpha_cut=0.45, crop=True, punch=1.15):
    """Turn any RGBA image (e.g. a 3D render) into a sprite ``height`` pixels tall.

    Returns the small image - enlarge it with ``upscale`` for display.
    """
    img = img.convert("RGBA")
    if crop:
        img = crop_to_content(img)
    w = max(1, round(img.width * height / img.height))
    a = np.asarray(img).astype(np.float32) / 255.0
    pm = np.dstack([a[..., :3] * a[..., 3:4], a[..., 3]])
    small = np.asarray(Image.fromarray((pm * 255).astype(np.uint8), "RGBA").resize((w, height), Image.BOX)).astype(np.float32) / 255
    al = small[..., 3]
    col = small[..., :3] / np.maximum(al, 1e-3)[..., None]
    col = np.clip((col - 0.5) * punch + 0.5, 0, 1)
    out = np.dstack([col, (al >= alpha_cut).astype(np.float32)])
    out[al < alpha_cut] = 0
    res = quantize(Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA"), palette, colors, dither)
    if outline_color:
        padded = Image.new("RGBA", (res.width + 2, res.height + 2), (0, 0, 0, 0))
        padded.paste(res, (1, 1))
        res = outline(padded, outline_color)
    return res


def upscale(img, factor):
    """Nearest-neighbour enlargement (keeps pixel art crisp)."""
    return img.resize((img.width * int(factor), img.height * int(factor)), Image.NEAREST)
