"""forge.humanoid - base humanoid: a neutral mannequin to dress up.

    from forge import kit, sdf
    from forge.humanoid import JOINTS, TARGET, VIEW_HEIGHT
    sc = kit.humanoid(body="#8a9099")        # Scene, A-pose, 1.80 m, faces +Z
    sc.add("cuff", sdf.sphere(JOINTS["wrist"], 0.045), "joint", mirror=True)
    kit.preview(sc, "p.png", target=TARGET, view_height=VIEW_HEIGHT)

Parts: head, ear, neck, torso, shoulder, upper_arm, elbow, forearm, wrist, hand,
thumb, thigh, knee, shin, ankle, foot.  Materials: body, joint.
JOINTS holds the skeleton points (+x side, metres) for placing new parts.
Reference render: assets/humanoid_base_4k.png (python -m forge.humanoid out.png).
"""
import numpy as np

from .sdf import Scene, align_y, box, capsule, ellipsoid, lerp, sphere, subtract, union

TARGET, VIEW_HEIGHT = (0, 0.90, 0), 1.95


def V(*a):
    return np.array(a, dtype=float)


JOINTS = dict(
    shoulder=V(0.205, 1.470, -0.010), elbow=V(0.325, 1.190, -0.012), wrist=V(0.415, 0.945, 0.008),
    fingertip=V(0.475, 0.770, 0.022), hip=V(0.100, 0.930, 0.000), knee=V(0.115, 0.520, 0.006),
    ankle=V(0.125, 0.090, -0.012),
)


def humanoid(sc=None, body="#8a9099", joint="#4d535b", joints=True):
    """Neutral A-pose mannequin, 1.80 m, ~7.5 heads tall. Returns the Scene."""
    sc = sc or Scene()
    sc.material("body", body, rough=0.50, coat=0.25, coat_rough=0.25, rough_var=0.18, albedo_var=0.06)
    sc.material("joint", joint, rough=0.55, rough_var=0.20, albedo_var=0.08)
    S, E, W, F = (JOINTS[k] for k in ("shoulder", "elbow", "wrist", "fingertip"))
    Hp, K, A = (JOINTS[k] for k in ("hip", "knee", "ankle"))

    # ----------------------------------------------------------------- head
    skull = ellipsoid((0, 1.705, -0.012), (0.076, 0.095, 0.094))
    jaw = ellipsoid((0, 1.642, 0.020), (0.060, 0.068, 0.072))
    head = union(skull, jaw, k=0.04)
    head = union(head, capsule((0, 1.682, 0.086), (0, 1.656, 0.100), 0.012, 0.014), k=0.018)
    socket = ellipsoid((0.031, 1.694, 0.090), (0.021, 0.012, 0.020)).mirror()
    sc.add("head", subtract(head, socket, k=0.02), "body", group="head")
    sc.add("ear", ellipsoid((0.073, 1.682, -0.012), (0.009, 0.024, 0.014)), "body", mirror=True, group="head_m")
    sc.add("neck", capsule((0, 1.490, -0.016), (0, 1.625, -0.006), 0.050), "body", group="head")

    # ---------------------------------------------------------------- torso
    torso = union(ellipsoid((0, 1.360, -0.005), (0.165, 0.160, 0.105)),
                  ellipsoid((0, 1.150, 0.000), (0.126, 0.140, 0.088)),
                  ellipsoid((0, 0.985, -0.005), (0.152, 0.112, 0.100)),
                  ellipsoid((0.072, 0.955, -0.045), (0.075, 0.080, 0.060)).mirror(),
                  capsule((0, 1.500, -0.020), (0.195, 1.478, -0.010), 0.052).mirror(), k=0.06)
    sc.add("torso", torso, "body", group="torso")

    # ------------------------------------------------------------------ arm
    sc.add("shoulder", sphere(S, 0.064), "body", mirror=True, group="arm_m")
    sc.add("upper_arm", capsule(lerp(S, E, 0.08), lerp(S, E, 0.94), 0.054, 0.045), "body", mirror=True, group="arm_m")
    sc.add("forearm", union(capsule(lerp(E, W, 0.06), lerp(E, W, 0.95), 0.046, 0.033),
                            ellipsoid(lerp(E, W, 0.28), (0.046, 0.085, 0.050), align_y(W - E)), k=0.03),
           "body", mirror=True, group="forearm_m")
    d = (F - W) / np.linalg.norm(F - W)
    R = align_y(d)
    palm = box(lerp(W, F, 0.30), (0.017, 0.046, 0.040), 0.014, R)
    fingers = box(lerp(W, F, 0.74), (0.013, 0.045, 0.036), 0.011, R)
    zc = float(lerp(W, F, 0.74)[2])
    fingers = fingers.grooves((0, 0, 1), [zc - 0.018, zc, zc + 0.018], 0.004, 0.012)
    sc.add("hand", union(palm, fingers, k=0.012), "body", mirror=True, group="hand_m")
    t0 = lerp(W, F, 0.18) + V(-0.004, 0, 0.036)
    sc.add("thumb", capsule(t0, t0 + V(0.010, -0.062, 0.022), 0.013, 0.010), "body", mirror=True, group="hand_m")

    # ------------------------------------------------------------------ leg
    sc.add("thigh", capsule(lerp(Hp, K, 0.04), lerp(Hp, K, 0.95), 0.088, 0.060), "body", mirror=True, group="thigh_m")
    sc.add("shin", union(capsule(lerp(K, A, 0.05), lerp(K, A, 0.97), 0.056, 0.040),
                         ellipsoid(lerp(K, A, 0.30) + V(0, 0, -0.022), (0.052, 0.120, 0.056)), k=0.04),
           "body", mirror=True, group="shin_m")
    foot = union(ellipsoid((0.130, 0.040, 0.060), (0.048, 0.040, 0.130)),
                 ellipsoid((0.127, 0.048, -0.050), (0.042, 0.048, 0.052)), k=0.03)
    foot = subtract(foot, box((0.13, -0.05, 0.03), (0.2, 0.05, 0.3)))
    sc.add("foot", foot, "body", mirror=True, group="foot_m")

    # --------------------------------------------------------------- joints
    if joints:
        sc.add("elbow", sphere(E, 0.050), "joint", mirror=True, group="arm_m")
        sc.add("wrist", sphere(W, 0.036), "joint", mirror=True, group="forearm_m")
        sc.add("knee", sphere(K, 0.064), "joint", mirror=True, group="thigh_m")
        sc.add("ankle", sphere(A, 0.045), "joint", mirror=True, group="shin_m")
    return sc


if __name__ == "__main__":
    import sys
    from . import kit
    out = sys.argv[1] if len(sys.argv) > 1 else "humanoid_base_4k.png"
    kit.final(humanoid(), out, yaws=(0, 180, -90), width=3840, height=2160, target=TARGET, view_height=VIEW_HEIGHT)
