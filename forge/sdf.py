"""forge.sdf - model 3D shapes as signed distance fields (SDFs).

Everything is vectorised numpy. A point set is a tuple ``(x, y, z)`` of 1-D
float32 arrays; ``node.eval(p)`` returns the signed distance for every point
(negative inside). Units are metres by convention, +Y is up, and a character
faces +Z.

Build shapes from the lowercase factories (``sphere``, ``capsule``, ``box`` ...),
combine them with ``union`` / ``subtract`` / ``intersect`` (or ``|``, ``-``, ``&``)
and register them in a ``Scene`` as named parts with a material.
"""
import math
import numpy as np

F = np.float32
BIG = 1e9


def _len(x, y, z):
    return np.sqrt(x * x + y * y + z * z)


def _t3(v):
    return (float(v[0]), float(v[1]), float(v[2]))


# ----------------------------------------------------------------- rotations
def rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def ry(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def align_y(direction, ref=(0, 0, 1)):
    """Rotation that takes local +Y onto ``direction`` (handy for limb-aligned boxes)."""
    y = np.asarray(direction, dtype=np.float64)
    y = y / np.linalg.norm(y)
    r = np.asarray(ref, dtype=np.float64)
    if abs(np.dot(y, r)) > 0.98:
        r = np.array([1.0, 0, 0])
    x = np.cross(y, r)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def lerp(a, b, t):
    """Point between two 3-vectors (t=0 -> a, t=1 -> b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return tuple(float(v) for v in a + (b - a) * t)


# ---------------------------------------------------------------- smooth ops
def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) * (1.0 / k)
    return np.minimum(a, b) - h * h * (k * 0.25)


def smax(a, b, k):
    if k <= 0:
        return np.maximum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) * (1.0 / k)
    return np.maximum(a, b) + h * h * (k * 0.25)


# --------------------------------------------------------------------- nodes
class Node:
    def eval(self, p):
        raise NotImplementedError

    def bound(self):
        """(center ndarray(3), radius) of a sphere containing the shape."""
        raise NotImplementedError

    # --- transforms
    def move(self, dx=0.0, dy=0.0, dz=0.0):
        return Transform(self, None, (dx, dy, dz))

    def rotate(self, R, about=(0, 0, 0)):
        about = np.asarray(about, dtype=np.float64)
        return Transform(self, R, about - R @ about)

    def scale(self, sx, sy=None, sz=None, about=(0, 0, 0)):
        sy = sx if sy is None else sy
        sz = sx if sz is None else sz
        return Scale(self, (sx, sy, sz), about)

    def mirror(self):
        """Symmetric copy across the x=0 plane."""
        return MirrorX(self)

    # --- surface modifiers
    def round(self, r):
        return Round(self, r)

    def shell(self, thickness):
        return Shell(self, thickness)

    def grooves(self, normal, offsets, width=0.004, depth=0.004):
        """Carve thin panel lines: slabs perpendicular to ``normal`` at ``offsets``."""
        return Groove(self, normal, offsets, width, depth)

    def repeat(self, step, n):
        """n copies spaced by vector ``step`` (first copy is the original)."""
        return Repeat(self, step, n)

    # --- booleans
    def __or__(self, other):
        return Union([self, other])

    def __sub__(self, other):
        return Subtract(self, other)

    def __and__(self, other):
        return Intersect([self, other])


class Sphere(Node):
    def __init__(self, c, r):
        self.c, self.r = _t3(c), float(r)

    def eval(self, p):
        x, y, z = p
        cx, cy, cz = self.c
        return _len(x - cx, y - cy, z - cz) - self.r

    def bound(self):
        return np.array(self.c), self.r


class Ellipsoid(Node):
    """Centred at the origin (use the ``ellipsoid`` factory for placement)."""

    def __init__(self, radii):
        self.r = _t3(radii)

    def eval(self, p):
        x, y, z = p
        a, b, c = self.r
        k0 = _len(x * (1 / a), y * (1 / b), z * (1 / c))
        k1 = _len(x * (1 / (a * a)), y * (1 / (b * b)), z * (1 / (c * c)))
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)

    def bound(self):
        return np.zeros(3), max(self.r)


class Capsule(Node):
    """Line segment a-b with radius r1 at a and r2 at b (a tapered limb)."""

    def __init__(self, a, b, r1, r2=None):
        self.a, self.b = _t3(a), _t3(b)
        self.r1 = float(r1)
        self.r2 = self.r1 if r2 is None else float(r2)
        self.ba = (self.b[0] - self.a[0], self.b[1] - self.a[1], self.b[2] - self.a[2])
        self.inv = 1.0 / max(sum(v * v for v in self.ba), 1e-12)

    def eval(self, p):
        x, y, z = p
        ax, ay, az = self.a
        bx, by, bz = self.ba
        px, py, pz = x - ax, y - ay, z - az
        h = np.clip((px * bx + py * by + pz * bz) * self.inv, 0.0, 1.0)
        d = _len(px - bx * h, py - by * h, pz - bz * h)
        if self.r1 == self.r2:
            return d - self.r1
        return d - (self.r1 + (self.r2 - self.r1) * h)

    def bound(self):
        a, b = np.array(self.a), np.array(self.b)
        return (a + b) / 2, float(np.linalg.norm(b - a)) / 2 + max(self.r1, self.r2)


class Box(Node):
    """Origin-centred box, ``half`` = half extents, ``r`` = edge rounding."""

    def __init__(self, half, r=0.0):
        self.r = float(r)
        self.half = _t3(half)
        self.h = tuple(max(v - self.r, 1e-5) for v in self.half)

    def eval(self, p):
        x, y, z = p
        hx, hy, hz = self.h
        qx, qy, qz = np.abs(x) - hx, np.abs(y) - hy, np.abs(z) - hz
        outside = _len(np.maximum(qx, 0.0), np.maximum(qy, 0.0), np.maximum(qz, 0.0))
        inside = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0.0)
        return outside + inside - self.r

    def bound(self):
        return np.zeros(3), float(np.linalg.norm(self.half))


class Cylinder(Node):
    """Capped cylinder / cone frustum from a to b (radius r at a, r2 at b)."""

    def __init__(self, a, b, r, r2=None, round=0.0):
        self.a, self.b = _t3(a), _t3(b)
        self.r1 = float(r)
        self.r2 = self.r1 if r2 is None else float(r2)
        self.rnd = float(round)
        ba = np.array(self.b) - np.array(self.a)
        self.L = float(np.linalg.norm(ba))
        self.n = _t3(ba / max(self.L, 1e-9))
        self.cos = self.L / math.hypot(self.L, self.r2 - self.r1)

    def eval(self, p):
        x, y, z = p
        ax, ay, az = self.a
        nx, ny, nz = self.n
        px, py, pz = x - ax, y - ay, z - az
        yy = px * nx + py * ny + pz * nz
        rad = np.sqrt(np.maximum(px * px + py * py + pz * pz - yy * yy, 0.0))
        if self.r1 == self.r2:
            dx = rad - (self.r1 - self.rnd)
        else:
            rr = self.r1 + (self.r2 - self.r1) * np.clip(yy * (1.0 / self.L), 0.0, 1.0)
            dx = (rad - (rr - self.rnd)) * self.cos
        dy = np.abs(yy - self.L * 0.5) - (self.L * 0.5 - self.rnd)
        mx, my = np.maximum(dx, 0.0), np.maximum(dy, 0.0)
        return np.minimum(np.maximum(dx, dy), 0.0) + np.sqrt(mx * mx + my * my) - self.rnd

    def bound(self):
        a, b = np.array(self.a), np.array(self.b)
        return (a + b) / 2, math.hypot(self.L / 2, max(self.r1, self.r2))


class Torus(Node):
    def __init__(self, c, axis, R, r):
        self.c = _t3(c)
        ax = np.asarray(axis, dtype=np.float64)
        self.n = _t3(ax / np.linalg.norm(ax))
        self.R, self.r = float(R), float(r)

    def eval(self, p):
        x, y, z = p
        cx, cy, cz = self.c
        nx, ny, nz = self.n
        px, py, pz = x - cx, y - cy, z - cz
        yy = px * nx + py * ny + pz * nz
        rad = np.sqrt(np.maximum(px * px + py * py + pz * pz - yy * yy, 0.0)) - self.R
        return np.sqrt(rad * rad + yy * yy) - self.r

    def bound(self):
        return np.array(self.c), self.R + self.r


class Plane(Node):
    """Half-space: negative behind the plane (opposite the normal).

    Unbounded, so only use it as the 2nd+ argument of intersect / subtract.
    """

    def __init__(self, point, normal):
        n = np.asarray(normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        self.n = _t3(n)
        self.h = float(np.dot(n, np.asarray(point, dtype=np.float64)))

    def eval(self, p):
        x, y, z = p
        nx, ny, nz = self.n
        return x * nx + y * ny + z * nz - self.h

    def bound(self):
        return np.zeros(3), float("inf")


# ----------------------------------------------------------------- operators
def _enclose(bounds, pad=0.0):
    bounds = [(c, r) for c, r in bounds if math.isfinite(r)]
    if not bounds:
        return np.zeros(3), float("inf")
    lo = np.min([c - r for c, r in bounds], axis=0)
    hi = np.max([c + r for c, r in bounds], axis=0)
    c = (lo + hi) / 2
    return c, max(float(np.linalg.norm(ci - c)) + ri for ci, ri in bounds) + pad


class Union(Node):
    def __init__(self, nodes, k=0.0):
        self.nodes, self.k = list(nodes), float(k)

    def eval(self, p):
        d = self.nodes[0].eval(p)
        for n in self.nodes[1:]:
            d = smin(d, n.eval(p), self.k)
        return d

    def bound(self):
        return _enclose([n.bound() for n in self.nodes], self.k * 0.25)


class Subtract(Node):
    def __init__(self, a, b, k=0.0):
        self.a, self.b, self.k = a, b, float(k)

    def eval(self, p):
        return smax(self.a.eval(p), -self.b.eval(p), self.k)

    def bound(self):
        return self.a.bound()


class Intersect(Node):
    def __init__(self, nodes, k=0.0):
        self.nodes, self.k = list(nodes), float(k)

    def eval(self, p):
        d = self.nodes[0].eval(p)
        for n in self.nodes[1:]:
            d = smax(d, n.eval(p), self.k)
        return d

    def bound(self):
        c, r = min((n.bound() for n in self.nodes), key=lambda b: b[1])
        return c, r + self.k * 0.25


class Transform(Node):
    """world = R @ local + t."""

    def __init__(self, node, R=None, t=(0, 0, 0)):
        self.node = node
        self.R = None if R is None else np.asarray(R, dtype=np.float64)
        self.t = _t3(t)
        if self.R is not None:
            self.m = [[float(self.R[i, j]) for j in range(3)] for i in range(3)]

    def eval(self, p):
        x, y, z = p
        tx, ty, tz = self.t
        qx, qy, qz = x - tx, y - ty, z - tz
        if self.R is None:
            return self.node.eval((qx, qy, qz))
        m = self.m
        return self.node.eval((
            m[0][0] * qx + m[1][0] * qy + m[2][0] * qz,
            m[0][1] * qx + m[1][1] * qy + m[2][1] * qz,
            m[0][2] * qx + m[1][2] * qy + m[2][2] * qz,
        ))

    def bound(self):
        c, r = self.node.bound()
        c = c if self.R is None else self.R @ c
        return c + np.array(self.t), r


class Scale(Node):
    def __init__(self, node, s, about=(0, 0, 0)):
        self.node, self.s, self.o = node, _t3(s), _t3(about)
        self.k = min(self.s)

    def eval(self, p):
        x, y, z = p
        sx, sy, sz = self.s
        ox, oy, oz = self.o
        q = (ox + (x - ox) * (1 / sx), oy + (y - oy) * (1 / sy), oz + (z - oz) * (1 / sz))
        return self.node.eval(q) * self.k

    def bound(self):
        c, r = self.node.bound()
        o, s = np.array(self.o), np.array(self.s)
        return o + (c - o) * s, r * max(self.s)


class MirrorX(Node):
    def __init__(self, node):
        self.node = node

    def eval(self, p):
        x, y, z = p
        return self.node.eval((np.abs(x), y, z))

    def bound(self):
        c, r = self.node.bound()
        return np.array([0.0, c[1], c[2]]), r + abs(float(c[0]))


class Round(Node):
    def __init__(self, node, r):
        self.node, self.r = node, float(r)

    def eval(self, p):
        return self.node.eval(p) - self.r

    def bound(self):
        c, r = self.node.bound()
        return c, r + self.r


class Shell(Node):
    def __init__(self, node, t):
        self.node, self.t = node, float(t)

    def eval(self, p):
        return np.abs(self.node.eval(p)) - self.t * 0.5

    def bound(self):
        c, r = self.node.bound()
        return c, r + self.t * 0.5


class Groove(Node):
    def __init__(self, node, normal, offsets, width, depth):
        n = np.asarray(normal, dtype=np.float64)
        self.node, self.n = node, _t3(n / np.linalg.norm(n))
        self.offsets = [float(o) for o in np.atleast_1d(offsets)]
        self.w, self.depth = float(width) * 0.5, float(depth)

    def eval(self, p):
        x, y, z = p
        nx, ny, nz = self.n
        d = self.node.eval(p)
        s = x * nx + y * ny + z * nz
        g = None
        for o in self.offsets:
            gi = np.abs(s - o) - self.w
            g = gi if g is None else np.minimum(g, gi)
        return np.maximum(d, np.minimum(-g, d + self.depth))

    def bound(self):
        return self.node.bound()


class Repeat(Node):
    def __init__(self, node, step, n):
        self.node, self.step, self.n = node, _t3(step), int(n)
        self.L2 = sum(v * v for v in self.step)
        c, _ = node.bound()
        self.s0 = float(np.dot(c, np.array(self.step))) / self.L2

    def eval(self, p):
        x, y, z = p
        sx, sy, sz = self.step
        i = np.clip(np.rint((x * sx + y * sy + z * sz) * (1 / self.L2) - self.s0), 0, self.n - 1)
        return self.node.eval((x - i * sx, y - i * sy, z - i * sz))

    def bound(self):
        c, r = self.node.bound()
        st = np.array(self.step) * (self.n - 1) / 2
        return c + st, r + float(np.linalg.norm(st))


# ----------------------------------------------------------------- factories
def sphere(c, r):
    return Sphere(c, r)


def ellipsoid(c, radii, rot=None):
    return Transform(Ellipsoid(radii), rot, c)


def capsule(a, b, r, r2=None):
    return Capsule(a, b, r, r2)


def box(c, half, round=0.0, rot=None):
    return Transform(Box(half, round), rot, c)


def cylinder(a, b, r, r2=None, round=0.0):
    return Cylinder(a, b, r, r2, round)


def torus(c, axis, R, r):
    return Torus(c, axis, R, r)


def plane(point, normal):
    return Plane(point, normal)


def union(*nodes, k=0.0):
    return Union(nodes, k)


def subtract(a, *bs, k=0.0):
    for b in bs:
        a = Subtract(a, b, k)
    return a


def intersect(*nodes, k=0.0):
    return Intersect(nodes, k)


# ------------------------------------------------------------ scene/material
def to_linear(color):
    """'#rrggbb' or an sRGB 0-1 tuple -> linear RGB tuple."""
    if isinstance(color, str):
        s = color.lstrip("#")
        color = tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(float(c) ** 2.2 for c in color)


class Material:
    """Surface description used by the renderer.

    color      sRGB hex or tuple          metallic   0 dielectric .. 1 metal
    rough      0 mirror .. 1 chalk        emit       glow strength (0 = none)
    coat       0..1 glossy clear coat     spec       dielectric reflectance (0.04)
    bump / bump_scale       fine surface relief (strength, features per metre)
    rough_var / albedo_var  low-frequency smudges and tone variation (0..1)
    pattern    None | 'ribbed' | 'hex'    pattern_scale = feature size in metres
    """

    def __init__(self, name, color, metallic=0.0, rough=0.5, emit=0.0, emit_color=None,
                 coat=0.0, coat_rough=0.12, spec=0.04, bump=0.0, bump_scale=120.0,
                 rough_var=0.0, albedo_var=0.0, var_scale=8.0,
                 pattern=None, pattern_scale=0.006, pattern_strength=0.5):
        self.name = name
        self.color = to_linear(color)
        self.metallic, self.rough = float(metallic), float(rough)
        self.emit, self._own_emit_color = float(emit), emit_color is None
        ec = self.color if emit_color is None else to_linear(emit_color)
        self.emissive = tuple(c * float(emit) for c in ec)
        self.coat, self.coat_rough, self.spec = float(coat), float(coat_rough), float(spec)
        self.bump, self.bump_scale = float(bump), float(bump_scale)
        self.rough_var, self.albedo_var, self.var_scale = float(rough_var), float(albedo_var), float(var_scale)
        self.pattern, self.pattern_scale = pattern, float(pattern_scale)
        self.pattern_strength = float(pattern_strength)

    def recolor(self, color):
        """Change the base colour (and the glow colour, if it followed the base)."""
        self.color = to_linear(color)
        if self._own_emit_color:
            self.emissive = tuple(c * self.emit for c in self.color)
        return self

    @property
    def procedural(self):
        return bool(self.bump or self.rough_var or self.albedo_var or self.pattern)


class Part:
    def __init__(self, name, node, material, mirror, group):
        self.name, self.node, self.material = name, node, material
        self.mirror, self.group = bool(mirror), group


class Scene:
    """A list of named parts (SDF node + material).

    ``mirror=True`` builds the part on the +x side and reflects it to -x for
    free. ``group`` names parts that sit close together (e.g. 'head', 'arm');
    the renderer culls whole groups at once, which is what keeps big scenes fast.
    """

    def __init__(self, margin=0.03):
        self.parts, self.materials, self.margin = [], {}, float(margin)
        self._groups = None

    def material(self, name, color, **kw):
        self.materials[name] = Material(name, color, **kw)
        self._groups = None
        return self.materials[name]

    def add(self, name, node, material, mirror=False, group=None):
        if material not in self.materials:
            raise KeyError(f"unknown material '{material}' - call scene.material() first")
        self.parts.append(Part(name, node, material, mirror, group))
        self._groups = None
        return self

    def remove(self, *names):
        """Drop parts by name (a trailing * matches a prefix, e.g. 'pack*')."""
        def hit(n):
            return any(n == k or (k.endswith("*") and n.startswith(k[:-1])) for k in names)
        self.parts = [p for p in self.parts if not hit(p.name)]
        self._groups = None
        return self

    def recolor(self, **colors):
        """scene.recolor(plate="#8a2f2f", glow="#ffb000")"""
        for name, color in colors.items():
            self.materials[name].recolor(color)
        self._groups = None
        return self

    def part_names(self):
        return [p.name for p in self.parts]

    def _build(self):
        names = list(self.materials)
        self.mats = [self.materials[n] for n in names]
        groups = {}
        for i, p in enumerate(self.parts):
            p.index, p.mat_id = i, names.index(p.material)
            c, r = p.node.bound()
            if not math.isfinite(r):
                raise ValueError(f"part '{p.name}' is unbounded (a bare plane?)")
            p.bc, p.br = _t3(c), float(r)
            key = (p.group if p.group else f"_{i}", p.mirror)
            groups.setdefault(key, []).append(p)
        self._groups = []
        for (_, mirror), parts in groups.items():
            c, r = _enclose([(np.array(p.bc), p.br) for p in parts])
            self._groups.append((mirror, _t3(c), float(r), parts))
        self.part_mat = np.array([p.mat_id for p in self.parts], dtype=np.int16)

    def bound(self):
        if self._groups is None:
            self._build()
        bs = []
        for mirror, c, r, _ in self._groups:
            c = np.array(c)
            bs.append((np.array([0.0, c[1], c[2]]), r + abs(c[0])) if mirror else (c, r))
        return _enclose(bs)

    def eval(self, p, with_id=False, margin=None):
        """Distance to the scene. Exact near surfaces (within ``margin`` of a
        part's bounding sphere), a conservative lower bound elsewhere."""
        if self._groups is None:
            self._build()
        m = self.margin if margin is None else float(margin)
        x, y, z = p
        d = np.full(x.shape[0], BIG, dtype=F)
        ids = np.full(x.shape[0], -1, dtype=np.int16) if with_id else None
        ax = None
        for mirror, (gx, gy, gz), gr, parts in self._groups:
            if mirror:
                if ax is None:
                    ax = np.abs(x)
                xx = ax
            else:
                xx = x
            gdb = _len(xx - gx, y - gy, z - gz) - gr
            near_g = gdb < m
            gsel = np.nonzero(near_g & (gdb < d))[0]
            d = np.minimum(d, np.where(near_g, BIG, gdb))
            if gsel.size == 0:
                continue
            sx, sy, sz = xx[gsel], y[gsel], z[gsel]
            dsub = d[gsel]
            isub = ids[gsel] if with_id else None
            for part in parts:
                bx, by, bz = part.bc
                db = _len(sx - bx, sy - by, sz - bz) - part.br
                near = db < m
                k = np.nonzero(near & (db < dsub))[0]
                dsub = np.minimum(dsub, np.where(near, BIG, db))
                if k.size:
                    dp = part.node.eval((sx[k], sy[k], sz[k])).astype(F, copy=False)
                    cur = dsub[k]
                    better = dp < cur
                    dsub[k] = np.where(better, dp, cur)
                    if with_id:
                        isub[k[better]] = part.index
            d[gsel] = dsub
            if with_id:
                ids[gsel] = isub
        return (d, ids) if with_id else d
