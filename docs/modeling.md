# Modelling well with SDFs

Read this before building a character, creature, vehicle or any model with more
than a few parts. It is the distilled experience of building the example soldier.

## Contents
1. The build order
2. Human proportions (metres)
3. Part recipes
4. Material presets
5. What makes a model look like a toy - and the fixes
6. Lighting and exposure
7. Performance rules
8. Debugging checklist

## 1. The build order

1. **Skeleton points first.** Define joints as vectors (`S` shoulder, `E` elbow,
   `W` wrist, `Hp` hip, `K` knee, `A` ankle) and derive everything from them
   with `lerp(a, b, t)`. Changing a pose then means moving a few points.
2. **Base body** from capsules and ellipsoids in a dark "undersuit" material,
   smooth-unioned (`k=0.02-0.07`). Preview. Fix proportions now.
3. **Big armour / shell shapes** on top, slightly larger than the body
   (5-12 mm). Preview from front, side and back.
4. **Secondary forms**: layered plates, cut lines (`.grooves`), trim pieces.
5. **Small readable details**: joint pivots, hoses, lights, pouches, one or two
   asymmetric items (antenna, holster) - symmetry everywhere looks artificial.
6. **Materials and colour** last, then exposure.

Build symmetric parts once on +x with `mirror=True`.

## 2. Human proportions (metres, standing, ~1.88 m in helmet and boots)

| Landmark | y | half-width x / radius |
|---|---|---|
| Top of helmet | 1.88 | helmet radii ~ (0.104, 0.122, 0.128) |
| Eye line / visor | 1.77 | |
| Chin | 1.62 | |
| Shoulder joint | 1.47 | x = 0.22-0.23 (pauldrons reach 0.36) |
| Chest centre | 1.36 | 0.176 wide, 0.11 deep |
| Waist | 1.15 | 0.13 wide, 0.09 deep |
| Belt / hips | 1.02 / 0.98 | 0.16-0.17 wide |
| Crotch, hip joint | 0.90, 0.93 | hip joint x = 0.10 |
| Knee | 0.52 | x = 0.125, radius 0.06 |
| Ankle | 0.10 | x = 0.14, radius 0.045 |
| Foot | 0-0.09 | 0.30 long, heel z = -0.09, toe z = +0.19 |
| Elbow (A-pose 22 deg) | 1.19 | x = 0.34 |
| Wrist | 0.945 | x = 0.43 |
| Fingertips | 0.77 | x = 0.49 |

Limb radii: upper arm 0.054 -> 0.046, forearm 0.046 -> 0.036, thigh 0.088 ->
0.060, shin 0.058 -> 0.044, neck 0.055. Armour adds 5-12 mm.

For turnarounds use an **A-pose** (arms ~22 degrees out) so the torso is visible
from the side and the arms do not merge with the body. Framing that fits this
figure: `target=(0, 0.98, 0), view_height=2.05`.

Scale the table for other builds: a heroic figure widens the shoulders 10%, a
child or stylised figure enlarges the head. Robots and creatures follow the same
method with different numbers.

## 3. Part recipes

**Plate that follows the body** - intersect an enlarged body shape with a box
that marks the plate's outline, instead of sticking a flat box on a curved
surface:

```python
belly = ellipsoid((0, 1.165, 0.012), (0.128, 0.140, 0.098))
plate = intersect(belly, box((0, 1.17, 0.075), (0.086, 0.021, 0.06), 0.008))
```

**Raised panel / painted stripe on a shell** - inflate the shell slightly and
intersect with a box: `intersect(shell.round(0.004), box(...))`.

**Hollow shoulder / knee shell** - `subtract(intersect(ellipsoid(c, r, rot),
plane(cut_point, down)), ellipsoid(c, r_minus_12mm, rot))`. The cut edge then
shows thickness instead of a flat slab.

**Half shells** - `intersect(shape, plane(point, normal))` keeps the side
*opposite* the normal. A bare `plane` is unbounded: never add one as a part.

**Recessed window / visor** - `subtract(helmet, cut)` for the frame and
`intersect(helmet.round(-0.005), cut)` for the glass sitting 5 mm inside it.

**Panel lines** - `.grooves(normal, [offsets], width, depth)`. Offsets are
distances along the normal from the origin: for an axis-aligned normal use the
coordinate (`(0,1,0), [1.43]` cuts at y = 1.43); for a limb direction `d` use
`float(np.dot(d, point_on_limb))`.

**Repeated details** (fingers, vents, vertebrae) - `.repeat(step_vector, n)`.

**Limb-aligned boxes** - `box(center, half, round, align_y(limb_direction))`;
local +Y then runs along the limb.

**Hoses and cables** - chains of `capsule`s unioned with a small `k`, in a
`pattern="ribbed"` material.

**Glowing elements** - tiny geometry (3-8 mm thick) in an `emit` material, sitting
just proud of the surface. A few small lights read better than one big one.

## 4. Material presets

```python
sc.material("paint",  "#737c86", rough=0.44, coat=0.35, coat_rough=0.20,   # painted armour
            rough_var=0.22, albedo_var=0.09, bump=0.02, bump_scale=260)
sc.material("gunmetal", "#3b4049", metallic=0.70, rough=0.48, rough_var=0.22, albedo_var=0.10)
sc.material("steel",  "#8f959c", metallic=1.0, rough=0.28, rough_var=0.20)  # bare bolts, nozzles
sc.material("suit",   "#22252a", rough=0.62, pattern="hex", pattern_scale=0.0075, pattern_strength=0.65)
sc.material("rubber", "#0d0e10", rough=0.80, pattern="ribbed", pattern_scale=0.011)  # hoses, joints
sc.material("fabric", "#3d4239", rough=0.92, bump=0.10, bump_scale=420, albedo_var=0.15, var_scale=30)
sc.material("glass",  "#05080c", rough=0.17, spec=0.05, emit=0.10, emit_color="#2a9fc0")
sc.material("glow",   "#3fdcff", emit=4.5)
sc.material("gold",   "#d4a84a", metallic=1.0, rough=0.25, rough_var=0.15)
sc.material("skin",   "#c58c6a", rough=0.55, spec=0.03, albedo_var=0.05)     # stylised only
```

`rough_var` / `albedo_var` (smudges, tone variation) are what stop surfaces
looking like CG plastic - put a little on everything.

## 5. What makes a model look like a toy - and the fixes

| Symptom | Cause | Fix |
|---|---|---|
| Big surfaces blown out to white | Light colours + default exposure | Use mid-tone colours (#6-#9 range), or `develop(exposure=0.7)` |
| Reads as a familiar franchise character | Iconic colour scheme / silhouette | Change the palette and the helmet / shoulder shapes; design your own |
| Shiny black balloons | Glossy dark ellipsoids as "armour" | Segment into plates, use rough >= 0.45, lighter grey |
| Two bright blobs on glass that look like eyes | Mirror-smooth surface reflecting the softboxes | rough 0.15-0.25 on glass, add a faint `emit` tint |
| Fingers or small parts look like combs | `ribbed` pattern on tiny geometry | Use a plain material on anything under ~4 cm |
| Limbs are featureless tubes | Single capsule per segment | Add a raised panel, a groove, a joint pivot, a strap |
| Head too big, body stubby | Helmet radii above ~0.13 | Keep the figure 7-7.5 heads tall |
| Spikes or blades where a plate should be | Plane cut through a thin capsule | Use a full band, or cut more than half away |
| Dark parts vanish | Albedo below ~#1a with no pattern | Lift to #20-#28, add `pattern` or `rough_var` |
| Invisible on a white (or dark) page | No mid-tones at the silhouette | Check both backgrounds; add trim of the opposite value |

## 6. Lighting and exposure

`render.STUDIO_LIGHTS` (key, fill, two rims) and `render.STUDIO_ENV` (sky
gradient + softboxes for reflections) are tuned so a mid-grey material comes out
mid-grey at `exposure=1.0`. Copy and edit them for other moods:

```python
from forge import render as R
lights = [dict(L) for L in R.STUDIO_LIGHTS]
lights[0]["power"] = 2.0                 # harder key
lights[0]["dir"] = (-0.8, 0.4, 0.4)      # more side light (camera space: x right, y up, z to viewer)
frame = R.render(sc, cam, W, H, lights=lights)
```

Directions are in camera space, so the lighting follows the camera around a
turnaround. Lower `env["ambient"]` or the fill power for drama; raise the rims to
separate a dark subject from a dark background.

## 7. Performance rules

- Tag every part with a `group` (head, torso, arm, forearm, thigh, shin, foot...)
  and keep each group spatially compact. Mirrored and unmirrored parts need
  different group names (`"head"` vs `"head_m"`).
- Split long limbs into separate groups (thigh / shin) - bounding spheres of long
  thin things are mostly empty space.
- Keep smooth-union `k` under ~0.07 and `.scale()` factors near 1; both make
  distances conservative and rays slower.
- Preview with `--scale 0.25`-style factors; only the final needs full size.

## 8. Debugging checklist

- **Nothing renders**: is the subject inside the view? Check `target` and
  `view_height`; remember `plane` parts are rejected, and `intersect` of shapes
  that do not overlap is empty.
- **A part is missing**: it may be buried inside another part - parts simply
  overlap, nearest surface wins. Inflate it or move it outward 3-5 mm.
- **Speckles or holes on thin parts** at low resolution: features thinner than
  ~1.5 pixels alias; they clean up at final size.
- **Banding or dents near joins**: large `k` blends bulge; reduce `k`.
- **Wrong side kept by a cut**: flip the plane normal.
