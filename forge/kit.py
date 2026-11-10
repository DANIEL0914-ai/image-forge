"""forge.kit - shortcuts that keep scene scripts short: material presets,
one-image previews, and a ready-made character to customise."""
import numpy as np

from . import post
from .render import Camera, render, turnaround
from .sdf import (Scene, align_y, box, capsule, cylinder, ellipsoid, intersect, lerp,
                  plane, rx, ry, rz, sphere, subtract, torus, union)

PRESETS = {
    "paint": dict(color="#737c86", rough=0.44, coat=0.35, coat_rough=0.20, rough_var=0.22, albedo_var=0.09,
                  bump=0.02, bump_scale=260),
    "gunmetal": dict(color="#3b4049", metallic=0.70, rough=0.48, rough_var=0.22, albedo_var=0.10),
    "steel": dict(color="#8f959c", metallic=1.0, rough=0.28, rough_var=0.20),
    "gold": dict(color="#d4a84a", metallic=1.0, rough=0.25, rough_var=0.15),
    "suit": dict(color="#22252a", rough=0.62, pattern="hex", pattern_scale=0.0075, pattern_strength=0.65),
    "rubber": dict(color="#0d0e10", rough=0.80, pattern="ribbed", pattern_scale=0.011, pattern_strength=0.55),
    "matte": dict(color="#1b1d21", rough=0.62, bump=0.06, bump_scale=300),
    "fabric": dict(color="#3d4239", rough=0.92, bump=0.10, bump_scale=420, albedo_var=0.15, var_scale=30),
    "glass": dict(color="#05080c", rough=0.17, spec=0.05, emit=0.10, emit_color="#2a9fc0"),
    "glow": dict(color="#3fdcff", emit=4.5),
    "skin": dict(color="#c58c6a", rough=0.55, spec=0.03, albedo_var=0.05),
}


def materials(sc, *names, **recolored):
    """Add presets to a scene: materials(sc, "steel", "glow", hull=("paint", "#a33"))
    registers 'steel', 'glow' and a paint-like material called 'hull' in red."""
    for n in names:
        kw = dict(PRESETS[n])
        sc.material(n, kw.pop("color"), **kw)
    for name, (preset, color) in recolored.items():
        kw = dict(PRESETS[preset])
        kw.pop("color")
        sc.material(name, color, **kw)
    return sc


def preview(scene, out="preview.png", yaws=(0, -90, 180), closeups=(), size=420, target=None,
            view_height=None, bg="#5a5e66"):
    """Draft-quality contact sheet in ONE small image: orbit views plus optional
    close-ups [(target_xyz, height_m, yaw), ...]. Look at this instead of
    separate renders - it is the cheapest way to check a model."""
    c, r = scene.bound()
    target = tuple(c) if target is None else target
    view_height = 2.0 * r * 1.04 if view_height is None else view_height
    pw = int(size * 0.62)
    frames = turnaround(scene, [(str(y), y) for y in yaws], width=pw * len(yaws), height=size,
                        target=target, view_height=view_height, aa=2)
    imgs = [post.develop(f) for f in frames]
    for t, h, yaw in closeups:
        f = render(scene, Camera(target=t, yaw=yaw, dist=4.0 * r + 2.0, height=h), pw, size, aa=2)
        imgs.append(post.develop(f))
    post.on_background(post.sheet(imgs), bg).save(out)
    return out


def final(scene, out, yaws=(0,), width=2048, height=2048, target=None, view_height=None, **develop):
    """Full-quality render of one or more orbit views side by side, transparent PNG."""
    c, r = scene.bound()
    frames = turnaround(scene, [(str(y), y) for y in yaws], width=width, height=height,
                        target=tuple(c) if target is None else target,
                        view_height=2.0 * r * 1.04 if view_height is None else view_height)
    post.sheet([post.develop(f, **develop) for f in frames]).save(out)
    return out


def V(*a):
    return np.array(a, dtype=float)


def soldier(sc=None, **colors):
    """Armoured SF trooper (~85 parts, A-pose, 1.88 m). Returns the Scene.

    Materials: plate, plate2, trim, steel, suit, rubber, glove, fabric, visor,
    glow, accent - recolour with keyword args: soldier(plate="#8a2f2f").
    Then scene.remove("antenna", "pack*") / scene.add(...) to customise.
    """
    sc = sc or Scene()
    # ------------------------------------------------------------ materials
    sc.material("plate", "#737c86", rough=0.44, coat=0.35, coat_rough=0.20,
                rough_var=0.22, albedo_var=0.09, bump=0.02, bump_scale=260)
    sc.material("plate2", "#a9afb6", rough=0.42, coat=0.40, coat_rough=0.18,
                rough_var=0.22, albedo_var=0.08, bump=0.02, bump_scale=260)
    sc.material("trim", "#3b4049", metallic=0.70, rough=0.48, rough_var=0.22, albedo_var=0.10)
    sc.material("steel", "#8f959c", metallic=1.0, rough=0.28, rough_var=0.20)
    sc.material("suit", "#22252a", rough=0.62, pattern="hex", pattern_scale=0.0075, pattern_strength=0.65)
    sc.material("rubber", "#0d0e10", rough=0.80, pattern="ribbed", pattern_scale=0.011, pattern_strength=0.55)
    sc.material("glove", "#1b1d21", rough=0.62, bump=0.06, bump_scale=300)
    sc.material("fabric", "#3d4239", rough=0.92, bump=0.10, bump_scale=420, albedo_var=0.15, var_scale=30)
    sc.material("visor", "#05080c", rough=0.17, spec=0.05, emit=0.10, emit_color="#2a9fc0")
    sc.material("glow", "#3fdcff", emit=4.5)
    sc.material("accent", "#d9571c", rough=0.42, coat=0.4, albedo_var=0.08)

    # ----------------------------------------------------------------- head
    dome = ellipsoid((0, 1.765, -0.006), (0.104, 0.122, 0.128))
    nape = ellipsoid((0, 1.690, -0.060), (0.090, 0.066, 0.068))
    helmet = union(dome, nape, k=0.045)
    visor_cut = box((0.050, 1.768, 0.100), (0.072, 0.027, 0.100), 0.016, rz(6)).mirror()
    shell = subtract(helmet, visor_cut, k=0.004)
    shell = shell.grooves((0, 1, 0), [1.722], 0.004, 0.004).grooves((0, 0, 1), [-0.030], 0.004, 0.004)
    sc.add("helmet", shell, "plate", group="head")
    sc.add("visor", intersect(helmet.round(-0.005), visor_cut), "visor", group="head")
    sc.add("crest", intersect(helmet.round(0.005), box((0, 1.86, -0.030), (0.020, 0.075, 0.140), 0.004)),
           "trim", group="head")
    sc.add("brow", intersect(helmet.round(0.004), box((0.050, 1.800, 0.100), (0.076, 0.007, 0.100), 0.003, rz(6)).mirror()),
           "trim", group="head")
    mask = union(ellipsoid((0, 1.688, 0.034), (0.080, 0.066, 0.092)),
                 box((0, 1.668, 0.105), (0.030, 0.034, 0.022), 0.012, rx(12)), k=0.02)
    sc.add("mask", mask.grooves((1, 0, 0), [-0.016, 0.0, 0.016], 0.005, 0.006), "trim", group="head")
    sc.add("cheek", box((0.062, 1.690, 0.070), (0.012, 0.030, 0.036), 0.008, ry(28)), "plate", mirror=True, group="head_m")
    sc.add("filter", cylinder((0.060, 1.655, 0.062), (0.092, 1.640, 0.082), 0.020, round=0.006),
           "steel", mirror=True, group="head_m")
    ear = cylinder((0.088, 1.752, -0.020), (0.120, 1.752, -0.020), 0.036, round=0.008)
    sc.add("ear", ear, "trim", mirror=True, group="head_m")
    sc.add("ear_ring", torus((0.121, 1.752, -0.020), (1, 0, 0), 0.021, 0.004), "steel", mirror=True, group="head_m")
    sc.add("ear_glow", cylinder((0.118, 1.752, -0.020), (0.1235, 1.752, -0.020), 0.011, round=0.002),
           "glow", mirror=True, group="head_m")
    sc.add("antenna", union(capsule((-0.112, 1.780, -0.040), (-0.117, 1.930, -0.052), 0.0030),
                            cylinder((-0.111, 1.765, -0.040), (-0.112, 1.805, -0.041), 0.008, round=0.003)),
           "trim", group="head")
    sc.add("rear_vent", box((0, 1.712, -0.120), (0.040, 0.022, 0.018), 0.006).grooves((1, 0, 0), [-0.02, 0.0, 0.02], 0.006, 0.010),
           "trim", group="head")
    sc.add("neck", capsule((0, 1.515, -0.015), (0, 1.650, -0.008), 0.057), "rubber", group="head")
    sc.add("collar", cylinder((0, 1.528, -0.012), (0, 1.590, -0.012), 0.086, 0.070, round=0.013), "trim", group="head")

    # ---------------------------------------------------------------- torso
    body = union(ellipsoid((0, 1.360, -0.005), (0.176, 0.165, 0.110)),
                 ellipsoid((0, 1.150, 0.000), (0.130, 0.140, 0.090)),
                 ellipsoid((0, 0.985, -0.005), (0.156, 0.110, 0.104)),
                 capsule((0, 1.500, -0.020), (0.200, 1.475, -0.010), 0.054).mirror(), k=0.07)
    sc.add("body", body, "suit", group="torso")
    chest = intersect(ellipsoid((0, 1.378, 0.020), (0.172, 0.122, 0.122)),
                      box((0, 1.385, 0.062), (0.170, 0.108, 0.072), 0.045),
                      plane((0, 0, 0.000), (0, 0, -1)), k=0.02)
    chest = union(chest, box((0, 1.478, 0.028), (0.100, 0.030, 0.068), 0.026), k=0.02)
    chest = chest.grooves((0, 1, 0), [1.432], 0.005, 0.005).grooves((1, 0, 0), [-0.070, 0.070], 0.005, 0.005)
    sc.add("chest", chest, "plate", group="torso")
    sc.add("chest_mark", intersect(chest.round(0.0012), box((0.115, 1.405, 0.10), (0.022, 0.009, 0.08), 0.002)),
           "accent", group="torso")
    sc.add("sternum", box((0, 1.372, 0.118), (0.027, 0.062, 0.020), 0.008), "trim", group="torso")
    sc.add("reactor", cylinder((0, 1.392, 0.128), (0, 1.392, 0.1415), 0.017, round=0.003), "glow", group="torso")
    sc.add("reactor_ring", torus((0, 1.392, 0.1385), (0, 0, 1), 0.0215, 0.0048), "steel", group="torso")
    flank = intersect(ellipsoid((0, 1.285, 0.000), (0.188, 0.135, 0.120)), box((0.158, 1.262, 0.000), (0.034, 0.070, 0.078), 0.010))
    sc.add("flank", flank.grooves((0, 1, 0), [1.235, 1.285], 0.006, 0.008), "trim", mirror=True, group="torso_m")
    back = intersect(ellipsoid((0, 1.375, -0.045), (0.170, 0.150, 0.094)), plane((0, 0, -0.03), (0, 0, 1)))
    sc.add("backplate", back.grooves((0, 1, 0), [1.30], 0.005, 0.005), "plate", group="torso")
    belly = ellipsoid((0, 1.165, 0.012), (0.128, 0.140, 0.098))
    for i, (y, w) in enumerate([(1.224, 0.094), (1.172, 0.086), (1.120, 0.078)]):
        sc.add(f"abs{i}", intersect(belly, box((0, y, 0.075), (w, 0.0215, 0.060), 0.008)), "plate", group="torso")
    sc.add("spine", box((0, 1.085, -0.090), (0.030, 0.017, 0.014), 0.006).repeat((0, 0.043, 0), 4),
           "trim", group="torso")
    belt = cylinder((0, 0.998, 0), (0, 1.050, 0), 1.0, round=0.012).scale(0.170, 1.0, 0.120)
    sc.add("belt", belt, "trim", group="hips")
    sc.add("buckle", box((0, 1.024, 0.120), (0.040, 0.030, 0.012), 0.008), "plate", group="hips")
    sc.add("pouch_front", box((0.105, 1.015, 0.098), (0.030, 0.040, 0.022), 0.010, ry(38)),
           "fabric", mirror=True, group="hips_m")
    sc.add("pouch_back", box((0.110, 1.015, -0.098), (0.034, 0.042, 0.022), 0.010, ry(-40)),
           "fabric", mirror=True, group="hips_m")
    sc.add("groin", box((0, 0.925, 0.088), (0.044, 0.052, 0.026), 0.018, rx(-12)), "trim", group="hips")
    sc.add("seat", ellipsoid((0, 0.955, -0.072), (0.116, 0.070, 0.048)).grooves((1, 0, 0), [0.0], 0.006, 0.008), "trim", group="hips")
    sc.add("tasset", box((0.170, 0.935, 0.004), (0.022, 0.085, 0.070), 0.018, rz(-8)).grooves((0, 1, 0), [0.94], 0.005, 0.005),
           "plate", mirror=True, group="hips_m")

    # ----------------------------------------------------------------- pack
    pack = box((0, 1.365, -0.165), (0.114, 0.145, 0.052), 0.030)
    sc.add("pack", pack.grooves((0, 1, 0), [1.30, 1.42], 0.006, 0.006), "trim", group="pack")
    sc.add("pack_top", box((0, 1.462, -0.186), (0.088, 0.048, 0.040), 0.020), "plate2", group="pack")
    sc.add("pack_core", cylinder((0, 1.345, -0.205), (0, 1.345, -0.226), 0.034, round=0.006), "steel", group="pack")
    sc.add("pack_core_glow", cylinder((0, 1.345, -0.222), (0, 1.345, -0.2295), 0.020, round=0.002), "glow", group="pack")
    sc.add("pack_strip", box((0.070, 1.365, -0.2165), (0.007, 0.062, 0.004), 0.003), "glow", mirror=True, group="pack_m")
    sc.add("nozzle", cylinder((0.060, 1.262, -0.172), (0.060, 1.196, -0.172), 0.026, 0.034, round=0.004),
           "steel", mirror=True, group="pack_m")
    hose = union(capsule((0.088, 1.470, -0.150), (0.104, 1.585, -0.105), 0.013),
                 capsule((0.104, 1.585, -0.105), (0.088, 1.668, -0.066), 0.013), k=0.02)
    sc.add("hose", hose, "rubber", mirror=True, group="pack_m")

    # ------------------------------------------------------------------ arm
    S, E, W = V(0.228, 1.470, -0.005), V(0.340, 1.190, -0.015), V(0.432, 0.945, 0.020)
    up_dir, fo_dir = (E - S) / np.linalg.norm(E - S), (W - E) / np.linalg.norm(W - E)
    out = V(0.93, 0.37, 0.0)  # points away from the body, perpendicular to the arm
    sc.add("arm", union(sphere(S, 0.068), capsule(S, E, 0.054, 0.046), capsule(E, W, 0.046, 0.036), k=0.02),
           "suit", mirror=True, group="arm")
    pc, prot = V(0.262, 1.492, -0.005), rz(-30)
    pcut = plane(pc + prot @ V(0, -0.022, 0), prot @ V(0, -1, 0))
    pauldron = subtract(intersect(ellipsoid(pc, (0.112, 0.074, 0.104), prot), pcut),
                        ellipsoid(pc, (0.100, 0.062, 0.092), prot))
    sc.add("pauldron", pauldron.grooves(prot @ V(1, 0, 0), [float(np.dot(prot @ V(1, 0, 0), pc)) + 0.030], 0.005, 0.004),
           "plate2", mirror=True, group="arm")
    stripe = intersect(ellipsoid(pc, (0.1135, 0.0755, 0.1055), prot), pcut,
                       box(pc + prot @ V(0.070, 0, 0), (0.014, 0.2, 0.2), 0.0, prot))
    sc.add("pauldron_stripe", subtract(stripe, ellipsoid(pc, (0.106, 0.068, 0.098), prot)), "accent", mirror=True, group="arm")
    c2 = pc + prot @ V(0.014, -0.040, 0)
    lame = subtract(intersect(ellipsoid(c2, (0.098, 0.062, 0.092), prot), plane(c2 + prot @ V(0, -0.020, 0), prot @ V(0, -1, 0))),
                    ellipsoid(c2, (0.088, 0.052, 0.082), prot))
    sc.add("pauldron_lame", lame, "plate", mirror=True, group="arm")
    sc.add("elbow_pivot", cylinder(E + V(0, 0, 0.040), E + V(0, 0, 0.056), 0.021, round=0.004), "steel", mirror=True, group="arm")
    bicep = capsule(lerp(S, E, 0.46), lerp(S, E, 0.84), 0.060, 0.055)
    sc.add("bicep", bicep.grooves(up_dir, [float(np.dot(up_dir, lerp(S, E, 0.66)))], 0.005, 0.005), "plate", mirror=True, group="arm")
    sc.add("elbow", sphere(E, 0.051), "rubber", mirror=True, group="arm")
    sc.add("elbow_guard", ellipsoid(E + V(0.014, 0.0, -0.032), (0.040, 0.052, 0.040)), "trim", mirror=True, group="arm")
    gauntlet = capsule(lerp(E, W, 0.20), lerp(E, W, 0.90), 0.055, 0.044)
    gs = [float(np.dot(fo_dir, lerp(E, W, t))) for t in (0.34, 0.80)]
    sc.add("gauntlet", gauntlet.grooves(fo_dir, gs, 0.005, 0.005), "plate", mirror=True, group="forearm")
    frot = align_y(fo_dir)
    gmid = V(*lerp(E, W, 0.56))
    sc.add("gauntlet_unit", box(gmid + out * 0.046, (0.014, 0.062, 0.030), 0.008, frot), "trim", mirror=True, group="forearm")
    sc.add("gauntlet_glow", box(gmid + out * 0.0605, (0.003, 0.036, 0.007), 0.002, frot), "glow", mirror=True, group="forearm")
    sc.add("cuff", cylinder(lerp(E, W, 0.90), lerp(E, W, 1.02), 0.042, round=0.006), "trim", mirror=True, group="forearm")
    hd = V(0.25, -0.95, 0.18)
    hd /= np.linalg.norm(hd)
    hrot = align_y(hd)
    zdir = hrot @ V(0, 0, 1)
    palm = box(W + hd * 0.052, (0.017, 0.046, 0.040), 0.014, hrot)
    f0 = W + hd * 0.094 - zdir * 0.030
    fd1 = hd + V(-0.30, 0, 0)
    fd1 /= np.linalg.norm(fd1)
    fd2 = hd + V(-0.85, 0, 0)
    fd2 /= np.linalg.norm(fd2)
    f1 = f0 + fd1 * 0.045
    finger = union(capsule(f0, f1, 0.0100, 0.0092), capsule(f1, f1 + fd2 * 0.040, 0.0092, 0.0082))
    t0 = W + hd * 0.030 + zdir * 0.038
    td = hd + V(-0.25, 0, 0.75)
    td /= np.linalg.norm(td)
    thumb = capsule(t0, t0 + td * 0.062, 0.0125, 0.0100)
    sc.add("hand", union(palm, finger.repeat(zdir * 0.020, 4), thumb, k=0.006), "glove", mirror=True, group="forearm")
    sc.add("hand_plate", box(W + hd * 0.050 + out * 0.0185, (0.006, 0.034, 0.032), 0.005, hrot), "plate", mirror=True, group="forearm")

    # ------------------------------------------------------------------ leg
    Hp, K, A = V(0.100, 0.930, 0.000), V(0.125, 0.520, 0.020), V(0.140, 0.100, -0.010)
    th_dir, sh_dir = (K - Hp) / np.linalg.norm(K - Hp), (A - K) / np.linalg.norm(A - K)
    sc.add("leg", union(sphere(V(0.100, 0.940, 0.0), 0.090), capsule(Hp, K, 0.088, 0.060),
                        capsule(K, A, 0.058, 0.044), k=0.03), "suit", mirror=True, group="thigh")
    tn = V(-0.45, 0.0, -0.89)
    tn /= np.linalg.norm(tn)
    thigh = intersect(capsule(lerp(Hp, K, 0.15), lerp(Hp, K, 0.82), 0.096, 0.070), plane(V(*lerp(Hp, K, 0.5)) + tn * 0.022, tn))
    gs = [float(np.dot(th_dir, lerp(Hp, K, 0.62)))]
    sc.add("thigh_plate", thigh.grooves(th_dir, gs, 0.006, 0.006), "plate", mirror=True, group="thigh")
    panel = intersect(capsule(lerp(Hp, K, 0.15), lerp(Hp, K, 0.82), 0.096, 0.070).round(0.005),
                      box(V(*lerp(Hp, K, 0.40)) + V(0.014, 0, 0.085), (0.034, 0.080, 0.030), 0.008, align_y(th_dir)))
    sc.add("thigh_panel", panel, "plate2", mirror=True, group="thigh")
    sc.add("thigh_glow", box(V(*lerp(Hp, K, 0.38)) + V(0.0885, 0, 0.012), (0.004, 0.050, 0.006), 0.003, align_y(th_dir)),
           "glow", mirror=True, group="thigh")
    sc.add("knee", sphere(K, 0.064), "rubber", mirror=True, group="shin")
    sc.add("knee_pivot", cylinder(K + V(0.052, 0, 0), K + V(0.071, 0, 0), 0.022, round=0.004), "steel", mirror=True, group="shin")
    sc.add("knee_trim", ellipsoid(K + V(0, -0.012, 0.034), (0.066, 0.052, 0.046)), "trim", mirror=True, group="shin")
    sc.add("knee_cap", ellipsoid(K + V(0, 0.006, 0.046), (0.055, 0.078, 0.045)), "plate2", mirror=True, group="shin")
    greave = union(capsule(lerp(K, A, 0.17), lerp(K, A, 0.86), 0.068, 0.055),
                   ellipsoid(V(*lerp(K, A, 0.36)) + V(0, 0, -0.034), (0.057, 0.115, 0.058)), k=0.03)
    sc.add("greave", greave.grooves((0, 1, 0), [0.205, 0.315], 0.005, 0.005), "plate", mirror=True, group="shin")
    sc.add("shin_ridge", box(V(*lerp(K, A, 0.50)) + V(0, 0, 0.060), (0.012, 0.130, 0.013), 0.006, align_y(sh_dir)),
           "trim", mirror=True, group="shin")
    toe_out = ry(8)
    pivot = (0.140, 0.0, -0.010)
    upper = intersect(union(ellipsoid((0.145, 0.075, 0.055), (0.054, 0.058, 0.125)),
                            capsule((0.140, 0.070, -0.015), (0.140, 0.165, -0.015), 0.058), k=0.035),
                      plane((0, 0.024, 0), (0, -1, 0)))
    sc.add("boot", upper.rotate(toe_out, pivot), "trim", mirror=True, group="foot")
    sole = union(box((0.145, 0.016, 0.050), (0.060, 0.016, 0.142), 0.012),
                 box((0.142, 0.022, -0.050), (0.056, 0.022, 0.040), 0.012))
    sc.add("sole", sole.rotate(toe_out, pivot), "rubber", mirror=True, group="foot")
    toe = intersect(ellipsoid((0.146, 0.062, 0.138), (0.051, 0.046, 0.056)), plane((0, 0.030, 0), (0, -1, 0)))
    sc.add("toe_cap", toe.rotate(toe_out, pivot), "plate", mirror=True, group="foot")
    sc.add("ankle_cuff", cylinder((0.140, 0.132, -0.013), (0.140, 0.188, -0.013), 0.066, round=0.012),
           "plate", mirror=True, group="foot")

    # ---------------------------------------- asymmetric gear (right thigh)
    Hr, Kr = Hp * V(-1, 1, 1), K * V(-1, 1, 1)
    axis = (Kr - Hr) / np.linalg.norm(Kr - Hr)
    sc.add("holster", box((-0.222, 0.735, 0.000), (0.030, 0.088, 0.048), 0.012, rz(3)), "fabric", group="holster")
    sc.add("sidearm", union(box((-0.232, 0.838, -0.012), (0.015, 0.042, 0.021), 0.007, rx(14)),
                            box((-0.232, 0.815, 0.016), (0.014, 0.016, 0.040), 0.006)), "trim", group="holster")
    sc.add("straps", union(torus(lerp(Hr, Kr, 0.30), axis, 0.093, 0.008), torus(lerp(Hr, Kr, 0.52), axis, 0.085, 0.008)),
           "rubber", group="holster")
    return sc.recolor(**colors) if colors else sc
