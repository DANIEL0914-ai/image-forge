"""forge.render - CPU sphere-tracing renderer for SDF scenes.

Designed for stills from a 64 px sprite up to 4K: a coarse pre-pass finds where
the model is, full-resolution rays start right in front of the surface, only
edge pixels are supersampled, and the work is split across CPU cores.

Shading is physically-based-ish: GGX speculars, a procedural studio environment
for reflections, ambient occlusion, soft shadows, clear coat, emissives.
Output is a ``Frame`` of linear HDR buffers - turn it into a picture with
``forge.post.develop``.
"""
import math
import os
import time
import numpy as np
from multiprocessing import get_context

from .sdf import F, BIG, _len
from .noise import fbm

_G = {}  # render state, inherited by forked workers
RGSS = [(-0.125, -0.375), (0.375, -0.125), (-0.375, 0.125), (0.125, 0.375)]

# Lights and environment are given in CAMERA space (x right, y up, z toward the
# viewer), so every view of a turnaround is lit the same way.
STUDIO_LIGHTS = [
    dict(dir=(-0.50, 0.62, 0.60), color=(1.00, 0.96, 0.90), power=1.50, shadow=True),  # key
    dict(dir=(0.80, 0.10, 0.55), color=(0.72, 0.84, 1.00), power=0.28),               # fill
    dict(dir=(0.62, 0.45, -0.64), color=(0.80, 0.92, 1.00), power=1.25),              # rim right
    dict(dir=(-0.72, 0.25, -0.64), color=(1.00, 0.90, 0.80), power=0.65),             # rim left
]
STUDIO_ENV = dict(
    sky=(0.36, 0.40, 0.47), horizon=(0.20, 0.21, 0.23), ground=(0.060, 0.057, 0.055), ambient=1.0,
    boxes=[
        dict(dir=(-0.50, 0.55, 0.67), size=0.32, color=(1.00, 0.97, 0.92), power=3.2),
        dict(dir=(0.86, 0.18, 0.48), size=0.28, color=(0.78, 0.88, 1.00), power=1.1),
        dict(dir=(0.00, 0.97, 0.12), size=0.50, color=(1.00, 1.00, 1.00), power=1.6),
        dict(dir=(0.74, 0.36, -0.56), size=0.26, color=(0.88, 0.95, 1.00), power=3.2),
        dict(dir=(-0.80, 0.25, -0.54), size=0.26, color=(1.00, 0.92, 0.84), power=2.0),
    ],
)


def _take(a, idx):
    return a[idx] if isinstance(a, np.ndarray) else a


def _norm3(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


# -------------------------------------------------------------------- camera
class Camera:
    """Orbit camera. ``yaw`` 0 looks at the model's front (+Z side), 90 at its
    left (+X) side, 180 at its back. Orthographic by default (``height`` =
    world units covered vertically); give ``fov`` (degrees) for perspective."""

    def __init__(self, target=(0, 1, 0), yaw=0.0, pitch=0.0, dist=8.0, height=2.2, fov=None):
        self.target = tuple(float(v) for v in target)
        self.yaw, self.pitch, self.dist = float(yaw), float(pitch), float(dist)
        self.height, self.fov = float(height), fov

    def basis(self):
        a, b = math.radians(self.yaw), math.radians(self.pitch)
        back = np.array([math.sin(a) * math.cos(b), math.sin(b), math.cos(a) * math.cos(b)])
        fwd = -back
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return right, up, fwd

    def eye(self):
        _, _, fwd = self.basis()
        return np.array(self.target) - fwd * self.dist

    def pixel_size(self, H):
        """(constant, per-unit-distance) world size of one pixel."""
        if self.fov is None:
            return self.height / H, 0.0
        return 0.0, 2 * math.tan(math.radians(self.fov) / 2) / H

    def rays(self, px, py, W, H):
        r, u, f = (tuple(float(c) for c in v) for v in self.basis())
        e = tuple(float(c) for c in self.eye())
        s0, s1 = self.pixel_size(H)
        if self.fov is None:
            a = (px - W * 0.5) * s0
            b = (H * 0.5 - py) * s0
            o = tuple((e[k] + r[k] * a + u[k] * b).astype(F, copy=False) for k in range(3))
            return o, f
        a = (px - W * 0.5) * s1
        b = (H * 0.5 - py) * s1
        d = [f[k] + r[k] * a + u[k] * b for k in range(3)]
        inv = 1.0 / _len(*d)
        o = tuple(np.full(px.shape, e[k], dtype=F) for k in range(3))
        return o, tuple((c * inv).astype(F, copy=False) for c in d)


def _ray_sphere(o, d, c, R):
    ox, oy, oz = o[0] - c[0], o[1] - c[1], o[2] - c[2]
    b = ox * d[0] + oy * d[1] + oz * d[2]
    disc = b * b - (ox * ox + oy * oy + oz * oz - R * R)
    sq = np.sqrt(np.maximum(disc, 0.0))
    t0 = np.maximum(-b - sq, 0.0).astype(F, copy=False)
    t1 = np.where(disc > 0, -b + sq, -1.0).astype(F, copy=False)
    return t0, t1


# ------------------------------------------------------------------ marching
def _march_coarse(scene, o, d, t_in, t_out, delta, m, eps, max_steps=160):
    """Returns (t_safe, dmin): where nearby full-res rays may safely start, and
    the closest approach of each coarse ray (to skip empty regions)."""
    n = t_in.shape[0]
    t = t_in.copy()
    dmin = np.full(n, BIG, dtype=F)
    tsafe = t_in.copy()
    frozen = np.zeros(n, dtype=bool)
    alive = np.nonzero(t_out > t_in)[0]
    d2 = delta * delta
    for _ in range(max_steps):
        if alive.size == 0:
            break
        ta = t[alive]
        dist = scene.eval(tuple(_take(o[k], alive) + _take(d[k], alive) * ta for k in range(3)))
        dmin[alive] = np.minimum(dmin[alive], dist)
        fr = frozen[alive]
        far = (dist >= m) & ~fr
        idx = alive[far]
        tsafe[idx] = ta[far] + np.sqrt(dist[far] * dist[far] - d2)
        frozen[alive[(dist < m) & ~fr]] = True
        ta = ta + np.maximum(dist, eps)
        t[alive] = ta
        alive = alive[(dist >= eps) & (ta < t_out[alive])]
    dmin[alive] = 0.0  # ran out of steps: keep the neighbourhood active
    return tsafe, dmin


def _march(scene, o, d, t0, t1, eps0, eps1, max_steps):
    n = t0.shape[0]
    t = t0.astype(F, copy=True)
    hit = np.zeros(n, dtype=bool)
    last = np.full(n, BIG, dtype=F)
    alive = np.nonzero(t1 > t0)[0]
    for _ in range(max_steps):
        if alive.size == 0:
            break
        ta = t[alive]
        dist = scene.eval(tuple(_take(o[k], alive) + _take(d[k], alive) * ta for k in range(3)))
        hn = dist < (eps0 + eps1 * ta)
        hit[alive[hn]] = True
        ta2 = np.where(hn, ta, ta + dist)
        t[alive] = ta2
        last[alive] = dist
        alive = alive[(~hn) & (ta2 < t1[alive])]
    if alive.size:  # grazing rays that ran out of steps
        hit[alive[last[alive] < 4 * (eps0 + eps1 * t[alive])]] = True
    return t, hit


def _normal(scene, P, h):
    x, y, z = P
    d1 = scene.eval((x + h, y - h, z - h))
    d2 = scene.eval((x - h, y - h, z + h))
    d3 = scene.eval((x - h, y + h, z - h))
    d4 = scene.eval((x + h, y + h, z + h))
    nx, ny, nz = d1 - d2 - d3 + d4, -d1 - d2 + d3 + d4, -d1 + d2 - d3 + d4
    inv = 1.0 / np.maximum(_len(nx, ny, nz), 1e-12)
    return nx * inv, ny * inv, nz * inv


# ------------------------------------------------------------------- shading
def _hex_edge(u, v):
    ax, ay = np.mod(u, 1.0) - 0.5, np.mod(v, 1.7320508) - 0.8660254
    bx, by = np.mod(u - 0.5, 1.0) - 0.5, np.mod(v - 0.8660254, 1.7320508) - 0.8660254
    use_a = (ax * ax + ay * ay) < (bx * bx + by * by)
    gx, gy = np.abs(np.where(use_a, ax, bx)), np.abs(np.where(use_a, ay, by))
    return 0.5 - np.maximum(gx, gx * 0.5 + gy * 0.8660254)


def _detail(mat, P, N, rough, alb):
    x, y, z = P
    nx, ny, nz = N
    if mat.rough_var or mat.albedo_var:
        s = mat.var_scale
        n1 = fbm(x * s, y * s, z * s, 3)
        if mat.rough_var:
            rough = np.clip(rough + mat.rough_var * (n1 - 0.5) * 1.6, 0.03, 1.0)
        if mat.albedo_var:
            s2 = s * 2.7
            f = 1.0 + mat.albedo_var * (fbm(x * s2 + 31.0, y * s2 + 17.0, z * s2 + 5.0, 3) - 0.5) * 1.6
            alb = [a * f for a in alb]
    if mat.bump:
        s, e = mat.bump_scale, 0.35
        xs, ys, zs = x * s, y * s, z * s
        g0 = fbm(xs, ys, zs, 2)
        gx, gy, gz = fbm(xs + e, ys, zs, 2) - g0, fbm(xs, ys + e, zs, 2) - g0, fbm(xs, ys, zs + e, 2) - g0
        dot = gx * nx + gy * ny + gz * nz
        k = mat.bump * 4.0
        nx, ny, nz = nx - k * (gx - dot * nx), ny - k * (gy - dot * ny), nz - k * (gz - dot * nz)
    if mat.pattern == "ribbed":
        ph = y * (2 * math.pi / mat.pattern_scale)
        c = np.cos(ph) * mat.pattern_strength
        nx, ny, nz = nx + c * ny * nx, ny - c * (1.0 - ny * ny), nz + c * ny * nz
        f = 0.82 + 0.18 * np.sin(ph)
        alb = [a * f for a in alb]
    elif mat.pattern == "hex":
        anx, any_, anz = np.abs(nx), np.abs(ny), np.abs(nz)
        use_x = (anx >= any_) & (anx >= anz)
        use_y = (~use_x) & (any_ >= anz)
        inv = 1.0 / mat.pattern_scale
        e = _hex_edge(np.where(use_x, z, x) * inv, np.where(use_y, z, y) * inv)
        gap = 1.0 - np.clip(e * (1.0 / 0.10), 0.0, 1.0)
        f = 1.0 - mat.pattern_strength * gap * 0.85
        alb = [a * f for a in alb]
        rough = np.clip(rough + gap * 0.25, 0.03, 1.0)
    if mat.bump or mat.pattern == "ribbed":
        inv = 1.0 / np.maximum(_len(nx, ny, nz), 1e-9)
        nx, ny, nz = nx * inv, ny * inv, nz * inv
    return (nx, ny, nz), rough, alb


def _ao(scene, P, N):
    x, y, z = P
    nx, ny, nz = N
    occ = 0.0
    for h, w in ((0.008, 0.30), (0.022, 0.25), (0.05, 0.20), (0.10, 0.15), (0.19, 0.10)):
        d = scene.eval((x + nx * h, y + ny * h, z + nz * h), margin=max(h * 1.05, 0.02))
        occ = occ + w * np.clip((h - d) * (1.0 / h), 0.0, 1.0)
    return np.clip(1.0 - 1.3 * occ, 0.0, 1.0)


def _shadow(scene, P, N, L, bias, k=16.0, steps=30):
    x, y, z = P
    lx, ly, lz = L
    ox, oy, oz = x + N[0] * bias, y + N[1] * bias, z + N[2] * bias
    _, t1 = _ray_sphere((ox, oy, oz), L, _G["bc"], _G["br"])
    n = x.shape[0]
    res = np.ones(n, dtype=F)
    t = np.full(n, 0.012, dtype=F)
    alive = np.nonzero((N[0] * lx + N[1] * ly + N[2] * lz > 0) & (t1 > t))[0]
    for _ in range(steps):
        if alive.size == 0:
            break
        ta = t[alive]
        mg = min(max(0.03, float(ta.max()) / k), 0.2)
        dist = scene.eval((ox[alive] + lx * ta, oy[alive] + ly * ta, oz[alive] + lz * ta), margin=mg)
        r = np.minimum(res[alive], k * dist / ta)
        res[alive] = r
        ta = ta + np.clip(dist, 0.008, 0.25)
        t[alive] = ta
        alive = alive[(r > 0.01) & (ta < t1[alive])]
    res = np.clip(res, 0.0, 1.0)
    return res * res * (3.0 - 2.0 * res)


def _smooth(a, b, x):
    t = np.clip((x - a) * (1.0 / max(b - a, 1e-6)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _env_spec(env, dx, dy, dz, rough):
    """Blurry studio reflection for direction (camera space)."""
    t = 0.5 + 0.5 * dy
    w = 0.04 + 0.42 * rough
    tt = np.clip((t - (0.5 - w)) / (2 * w), 0.0, 1.0)
    g = tt * tt * (3.0 - 2.0 * tt)
    up = np.clip((t - 0.5) * 2.0, 0.0, 1.0)
    out = []
    for c in range(3):
        upper = env["horizon"][c] + (env["sky"][c] - env["horizon"][c]) * up
        out.append(env["ground"][c] + (upper - env["ground"][c]) * g)
    for bx in env["boxes"]:
        bd = bx["_dir"]
        cosang = dx * bd[0] + dy * bd[1] + dz * bd[2]
        ang = np.arccos(np.clip(cosang, -1.0, 1.0))
        soft = 0.03 + 0.95 * rough
        inten = 1.0 - np.clip((ang - (bx["size"] - soft * 0.4)) / (soft * 1.4), 0.0, 1.0)
        inten = inten * inten * (3.0 - 2.0 * inten)
        inten = inten * (bx["power"] * (bx["size"] / (bx["size"] + soft)) ** 2)
        for c in range(3):
            out[c] = out[c] + inten * bx["color"][c]
    return out


def _shade(P, N, pid, V, t):
    S = _G
    scene, T, env = S["scene"], S["mtab"], S["env"]
    mid = S["part_mat"][pid]
    x, y, z = P
    alb = [T["r"][mid], T["g"][mid], T["b"][mid]]
    metal, rough = T["metal"][mid], T["rough"][mid]
    coat, coat_rough, spec0 = T["coat"][mid], T["coat_rough"][mid], T["spec"][mid]
    emis = [T["er"][mid], T["eg"][mid], T["eb"][mid]]
    Ng = N  # geometric normal for AO / shadows
    nx, ny, nz = (a.copy() for a in N)
    for m in np.unique(mid):
        mat = S["mats"][m]
        if not mat.procedural:
            continue
        k = np.nonzero(mid == m)[0]
        (ax, ay, az), rk, ak = _detail(mat, (x[k], y[k], z[k]), (nx[k], ny[k], nz[k]),
                                       rough[k], [a[k] for a in alb])
        nx[k], ny[k], nz[k], rough[k] = ax, ay, az, rk
        for c in range(3):
            alb[c][k] = ak[c]
    vx, vy, vz = V
    ndv = np.clip(nx * vx + ny * vy + nz * vz, 1e-3, 1.0)
    ao = _ao(scene, P, Ng) if S["ao"] else np.ones_like(x)
    if S["ao"] != 1.0 and S["ao"]:
        ao = ao ** S["ao"]
    f0 = [spec0 * (1 - metal) + alb[c] * metal for c in range(3)]
    dif = [alb[c] * (1 - metal) for c in range(3)]
    a2 = np.maximum(rough * rough, 2e-3) ** 2
    kv = (rough + 1.0) ** 2 * 0.125
    ca2 = np.maximum(coat_rough * coat_rough, 2e-3) ** 2
    out = [emis[0].copy(), emis[1].copy(), emis[2].copy()]
    pix = S["pix0"] + S["pix1"] * t
    for L in S["lights"]:
        lx, ly, lz = L["_dir"]
        ndl = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
        lit = ndl
        if L.get("shadow") and S["shadows"]:
            lit = ndl * _shadow(scene, P, Ng, (lx, ly, lz), 2.0 * pix + 0.002)
        hx, hy, hz = lx + vx, ly + vy, lz + vz
        inv = 1.0 / np.maximum(_len(hx, hy, hz), 1e-9)
        hx, hy, hz = hx * inv, hy * inv, hz * inv
        ndh = np.clip(nx * hx + ny * hy + nz * hz, 0.0, 1.0)
        fres = (1.0 - np.clip(vx * hx + vy * hy + vz * hz, 0.0, 1.0)) ** 5
        den = ndh * ndh * (a2 - 1.0) + 1.0
        vis = 0.25 / ((ndl * (1 - kv) + kv) * (ndv * (1 - kv) + kv))
        sp = np.minimum(a2 / (den * den) * vis, 60.0)
        dc = ndh * ndh * (ca2 - 1.0) + 1.0
        spc = np.minimum(ca2 / (dc * dc) * 0.25, 60.0) * coat * (0.04 + 0.96 * fres)
        for c in range(3):
            e = L["color"][c] * L["power"]
            out[c] = out[c] + (dif[c] + sp * (f0[c] + (1 - f0[c]) * fres) + spc) * lit * e
    # ambient + environment reflections (camera space)
    R_, U_, B_ = S["cam_r"], S["cam_u"], S["cam_b"]
    hemi = 0.5 + 0.5 * (nx * U_[0] + ny * U_[1] + nz * U_[2])
    rxw, ryw, rzw = 2 * ndv * nx - vx, 2 * ndv * ny - vy, 2 * ndv * nz - vz
    rc = (rxw * R_[0] + ryw * R_[1] + rzw * R_[2],
          rxw * U_[0] + ryw * U_[1] + rzw * U_[2],
          rxw * B_[0] + ryw * B_[1] + rzw * B_[2])
    er = _env_spec(env, rc[0], rc[1], rc[2], rough)
    so = np.clip((ndv + ao) ** np.exp2(-16.0 * rough - 1.0) - 1.0 + ao, 0.0, 1.0)
    # Karis' analytic environment BRDF
    r0, r1, r2, r3 = rough * -1.0 + 1.0, rough * -0.0275 + 0.0425, rough * -0.572 + 1.04, rough * 0.022 - 0.04
    a004 = np.minimum(r0 * r0, np.exp2(-9.28 * ndv)) * r0 + r1
    A, B = -1.04 * a004 + r2, 1.04 * a004 + r3
    fc = 0.04 + 0.96 * (1.0 - ndv) ** 5
    has_coat = bool(np.any(coat > 0))
    ec = _env_spec(env, rc[0], rc[1], rc[2], coat_rough) if has_coat else None
    sky_avg = [0.5 * (env["sky"][c] + env["horizon"][c]) for c in range(3)]
    for c in range(3):
        irr = (env["ground"][c] + (sky_avg[c] - env["ground"][c]) * hemi) * env["ambient"]
        out[c] = out[c] + dif[c] * irr * ao + er[c] * (f0[c] * A + B) * so
        if has_coat:
            out[c] = out[c] + ec[c] * fc * coat * so
    return out


# ---------------------------------------------------------------------- jobs
def _geom(px, py, t0):
    S = _G
    scene, cam, W, H = S["scene"], S["cam"], S["W"], S["H"]
    o, d = cam.rays(px, py, W, H)
    _, t1 = _ray_sphere(o, d, S["bc"], S["br"])
    t, hit = _march(scene, o, d, t0, t1, 0.4 * S["pix0"], 0.4 * S["pix1"], S["max_steps"])
    n = px.shape[0]
    nrm = [np.zeros(n, dtype=F) for _ in range(3)]
    pid = np.zeros(n, dtype=np.int16)
    idx = np.nonzero(hit)[0]
    if idx.size:
        th = t[idx]
        P = tuple(_take(o[k], idx) + _take(d[k], idx) * th for k in range(3))
        h = np.maximum(0.7 * (S["pix0"] + S["pix1"] * th), 2e-4) if S["pix1"] else max(0.7 * S["pix0"], 2e-4)
        N = _normal(scene, P, h)
        _, ids = scene.eval(P, with_id=True)
        for k in range(3):
            nrm[k][idx] = N[k]
        pid[idx] = np.maximum(ids, 0)
    return t, hit, nrm[0], nrm[1], nrm[2], pid


def _shade_job(px, py, t, nx, ny, nz, pid):
    S = _G
    o, d = S["cam"].rays(px, py, S["W"], S["H"])
    P = tuple((o[k] + d[k] * t).astype(F, copy=False) for k in range(3))
    V = tuple(-d[k] for k in range(3))
    return _shade(P, (nx, ny, nz), pid, V, t)


def _full(px, py, t0):
    t, hit, nx, ny, nz, pid = _geom(px, py, t0)
    n = px.shape[0]
    rgb = [np.zeros(n, dtype=F) for _ in range(3)]
    idx = np.nonzero(hit)[0]
    if idx.size:
        c = _shade_job(px[idx], py[idx], t[idx], nx[idx], ny[idx], nz[idx], pid[idx])
        for k in range(3):
            rgb[k][idx] = c[k]
    return hit, rgb[0], rgb[1], rgb[2]


def _job(args):
    kind = args[0]
    return {"geom": _geom, "shade": _shade_job, "full": _full}[kind](*args[1:])


# --------------------------------------------------------------------- frame
class Frame:
    """Linear HDR render buffers. rgb is premultiplied by alpha."""

    def __init__(self, rgb, alpha, depth, normal, part, pixel, seconds):
        self.rgb, self.alpha, self.depth = rgb, alpha, depth
        self.normal, self.part = normal, part       # camera-space normal (3,H,W), part index
        self.pixel, self.seconds = pixel, seconds   # world size of a pixel, render time
        self.height, self.width = alpha.shape


def _edges(hit, depth, nrm, pid, pix):
    e = np.zeros(hit.shape, dtype=bool)
    for ax in (0, 1):
        a = [slice(None), slice(None)]
        b = [slice(None), slice(None)]
        a[ax], b[ax] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        both = hit[a] & hit[b]
        dn = (nrm[0][a] * nrm[0][b] + nrm[1][a] * nrm[1][b] + nrm[2][a] * nrm[2][b]) < 0.94
        ed = (hit[a] != hit[b]) | (both & ((pid[a] != pid[b]) | dn | (np.abs(depth[a] - depth[b]) > 6 * pix)))
        e[a] |= ed
        e[b] |= ed
    return e


def render(scene, camera, width, height, lights=None, env=None, aa=4, ao=1.0, shadows=True,
           workers=None, chunk=100_000, coarse=4, max_steps=80, verbose=False):
    """Render ``scene`` and return a ``Frame``.

    aa        1 = none, 4 = supersample edge pixels 4x (default)
    ao        ambient occlusion strength (0 disables), shadows: soft key shadow
    lights    list of dicts (dir, color, power, shadow) in camera space
    env       studio environment dict (see STUDIO_ENV)
    workers   processes (default: all cores)
    """
    t_start = time.time()
    W, H = int(width), int(height)
    scene._build()
    lights = [dict(L) for L in (STUDIO_LIGHTS if lights is None else lights)]
    env = dict(STUDIO_ENV if env is None else env)
    env["boxes"] = [dict(b) for b in env["boxes"]]
    cr, cu, cf = camera.basis()
    cb = -cf
    for L in lights:
        dcam = _norm3(L["dir"])
        L["_dir"] = tuple(float(v) for v in (cr * dcam[0] + cu * dcam[1] + cb * dcam[2]))
    for b in env["boxes"]:
        b["_dir"] = tuple(float(v) for v in _norm3(b["dir"]))
    mats = scene.mats
    tab = lambda f: np.array([f(m) for m in mats], dtype=F)
    mtab = dict(r=tab(lambda m: m.color[0]), g=tab(lambda m: m.color[1]), b=tab(lambda m: m.color[2]),
                metal=tab(lambda m: m.metallic), rough=tab(lambda m: m.rough), coat=tab(lambda m: m.coat),
                coat_rough=tab(lambda m: m.coat_rough), spec=tab(lambda m: m.spec),
                er=tab(lambda m: m.emissive[0]), eg=tab(lambda m: m.emissive[1]), eb=tab(lambda m: m.emissive[2]))
    bc, br = scene.bound()
    pix0, pix1 = camera.pixel_size(H)
    _G.clear()
    _G.update(scene=scene, cam=camera, W=W, H=H, bc=tuple(float(v) for v in bc), br=float(br) * 1.02,
              pix0=pix0, pix1=pix1, max_steps=max_steps, mats=mats, mtab=mtab, part_mat=scene.part_mat,
              lights=lights, env=env, ao=float(ao), shadows=bool(shadows),
              cam_r=tuple(map(float, cr)), cam_u=tuple(map(float, cu)), cam_b=tuple(map(float, cb)))

    # 1. coarse pass: where is the model, and how far can rays skip ahead?
    s = max(1, int(coarse))
    cw, ch = -(-W // s), -(-H // s)
    gy, gx = np.mgrid[0:ch, 0:cw]
    cpx, cpy = (gx.ravel() * s + s * 0.5).astype(F), (gy.ravel() * s + s * 0.5).astype(F)
    o, d = camera.rays(cpx, cpy, W, H)
    t_in, t_out = _ray_sphere(o, d, _G["bc"], _G["br"])
    pix_far = pix0 + pix1 * float(max(t_out.max(), 0.0))
    delta = (s * 0.5 + 0.5) * 1.4143 * pix_far
    m = 2.0 * delta
    tsafe, dmin = _march_coarse(scene, o, d, t_in, t_out, delta, m, 0.4 * pix_far)
    act = (dmin < m + 2 * pix_far).reshape(ch, cw)
    act = np.repeat(np.repeat(act, s, axis=0), s, axis=1)[:H, :W]
    ys, xs = np.nonzero(act)
    t0_all = tsafe.reshape(ch, cw)[ys // s, xs // s]
    px_all, py_all = (xs + 0.5).astype(F), (ys + 0.5).astype(F)
    n_act = xs.shape[0]
    if verbose:
        print(f"[forge] {W}x{H}: coarse {time.time() - t_start:.1f}s, {n_act:,} active rays", flush=True)

    workers = workers or os.cpu_count() or 1
    pool = get_context("fork").Pool(workers) if workers > 1 and n_act > 20_000 else None

    def run(jobs):
        return pool.map(_job, jobs, chunksize=1) if pool else [_job(j) for j in jobs]

    def chunks(n):
        return [slice(i, min(i + chunk, n)) for i in range(0, n, chunk)]

    try:
        # 2. geometry for every active pixel
        hit = np.zeros((H, W), dtype=bool)
        depth = np.zeros((H, W), dtype=F)
        nrm = np.zeros((3, H, W), dtype=F)
        pid = np.zeros((H, W), dtype=np.int16)
        sl = chunks(n_act)
        for c, (t, h, nx, ny, nz, pi) in zip(sl, run([("geom", px_all[c], py_all[c], t0_all[c]) for c in sl])):
            yy, xx = ys[c], xs[c]
            hit[yy, xx], depth[yy, xx], pid[yy, xx] = h, t, pi
            nrm[0, yy, xx], nrm[1, yy, xx], nrm[2, yy, xx] = nx, ny, nz
        if verbose:
            print(f"[forge] geometry {time.time() - t_start:.1f}s, {int(hit.sum()):,} surface pixels", flush=True)

        # 3. shade interior pixels once, supersample edge pixels
        edge = _edges(hit, depth, nrm, pid, pix_far) if aa > 1 else np.zeros((H, W), dtype=bool)
        rgb = np.zeros((H, W, 3), dtype=F)
        alpha = hit.astype(F)
        iy, ix = np.nonzero(hit & ~edge)
        sl = chunks(iy.shape[0])
        jobs = [("shade", (ix[c] + 0.5).astype(F), (iy[c] + 0.5).astype(F), depth[iy[c], ix[c]],
                 nrm[0, iy[c], ix[c]], nrm[1, iy[c], ix[c]], nrm[2, iy[c], ix[c]], pid[iy[c], ix[c]]) for c in sl]
        for c, col in zip(sl, run(jobs)):
            rgb[iy[c], ix[c], 0], rgb[iy[c], ix[c], 1], rgb[iy[c], ix[c], 2] = col
        if verbose:
            print(f"[forge] shading {time.time() - t_start:.1f}s, {int(edge.sum()):,} edge pixels x{aa}", flush=True)
        ey, ex = np.nonzero(edge)
        if ey.size:
            offs = RGSS if aa >= 4 else [(-0.25, -0.25), (0.25, 0.25)]
            et0 = tsafe.reshape(ch, cw)[ey // s, ex // s]
            sl = chunks(ey.shape[0])
            jobs = [("full", (ex[c] + 0.5 + ox).astype(F), (ey[c] + 0.5 + oy).astype(F), et0[c])
                    for ox, oy in offs for c in sl]
            acc = np.zeros((ey.shape[0], 4), dtype=F)
            for j, (h, r, g, b) in enumerate(run(jobs)):
                c = sl[j % len(sl)]
                acc[c, 0] += r
                acc[c, 1] += g
                acc[c, 2] += b
                acc[c, 3] += h
            acc /= len(offs)
            rgb[ey, ex] = acc[:, :3]
            alpha[ey, ex] = acc[:, 3]
    finally:
        if pool:
            pool.close()
            pool.join()
    ncam = np.stack([nrm[0] * cr[0] + nrm[1] * cr[1] + nrm[2] * cr[2],
                     nrm[0] * cu[0] + nrm[1] * cu[1] + nrm[2] * cu[2],
                     nrm[0] * cb[0] + nrm[1] * cb[1] + nrm[2] * cb[2]]).astype(F)
    depth[~hit] = np.inf
    secs = time.time() - t_start
    if verbose:
        print(f"[forge] done in {secs:.1f}s", flush=True)
    return Frame(rgb, alpha, depth, ncam, np.where(hit, pid, -1), pix_far, secs)


def turnaround(scene, views=(("front", 0), ("side", 90), ("back", 180)), width=3840, height=2160,
               target=None, view_height=None, pitch=0.0, fov=None, **kw):
    """Render several orbit views as equal-width panels. Returns a list of Frames
    (compose them with ``forge.post.sheet`` after ``develop``).

    ``views`` is a list of (label, yaw_degrees). ``width`` is the total sheet width.
    """
    c, r = scene.bound()
    target = tuple(c) if target is None else target
    view_height = 2.0 * r * 1.04 if view_height is None else view_height
    pw = int(width) // len(views)
    frames = []
    for label, yaw in views:
        cam = Camera(target=target, yaw=yaw, pitch=pitch, dist=4.0 * r + 2.0, height=view_height, fov=fov)
        if kw.get("verbose"):
            print(f"[forge] view '{label}' (yaw {yaw})", flush=True)
        frames.append(render(scene, cam, pw, int(height), **kw))
    return frames
