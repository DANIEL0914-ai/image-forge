"""forge.noise - small, fast value noise for procedural surface detail."""
import numpy as np

F = np.float32
_A, _B, _C, _D = (np.uint32(v) for v in (374761393, 668265263, 2246822519, 1274126177))
_S13, _S16 = np.uint32(13), np.uint32(16)


def _hash(ix, iy, iz):
    h = ix * _A + iy * _B + iz * _C
    h = (h ^ (h >> _S13)) * _D
    h = h ^ (h >> _S16)
    return h.astype(F) * F(1.0 / 4294967296.0)


def vnoise(x, y, z):
    """3-D value noise in 0..1 (inputs are float arrays in 'cells')."""
    fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
    ix = fx.astype(np.int32).view(np.uint32)
    iy = fy.astype(np.int32).view(np.uint32)
    iz = fz.astype(np.int32).view(np.uint32)
    tx, ty, tz = x - fx, y - fy, z - fz
    ux, uy, uz = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty), tz * tz * (3 - 2 * tz)
    one = np.uint32(1)
    c000, c100 = _hash(ix, iy, iz), _hash(ix + one, iy, iz)
    c010, c110 = _hash(ix, iy + one, iz), _hash(ix + one, iy + one, iz)
    c001, c101 = _hash(ix, iy, iz + one), _hash(ix + one, iy, iz + one)
    c011, c111 = _hash(ix, iy + one, iz + one), _hash(ix + one, iy + one, iz + one)
    a = c000 + (c100 - c000) * ux
    b = c010 + (c110 - c010) * ux
    c = c001 + (c101 - c001) * ux
    d = c011 + (c111 - c011) * ux
    e = a + (b - a) * uy
    f = c + (d - c) * uy
    return (e + (f - e) * uz).astype(F, copy=False)


def fbm(x, y, z, octaves=3):
    """Fractal noise in roughly 0..1."""
    total, amp, norm = 0.0, 1.0, 0.0
    for _ in range(octaves):
        total = total + vnoise(x, y, z) * amp
        norm += amp
        amp *= 0.5
        x, y, z = x * 2.03 + 11.3, y * 2.03 + 5.7, z * 2.03 + 23.1
    return total * (1.0 / norm)


def noise2d(shape, scale, seed=0, octaves=4):
    """(H, W) fractal noise image in 0..1; ``scale`` = feature size in pixels."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(F)
    return fbm(xx / scale, yy / scale, np.full_like(xx, seed * 17.31), octaves)
