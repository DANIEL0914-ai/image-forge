# image-forge

Rendering engine for the image-forge Claude skill: pixel sprites, 2D art and SDF 3D renders from code (16 px to 4K).

- `forge/` - the Python package (needs numpy, pillow, scipy)
- `selftest.py` - quick check of all three modes (writes test.png)
- `examples/soldier_turnaround.py` - 4K character turnaround
- `kit.humanoid()` (`forge/humanoid.py`) - neutral unarmoured mannequin base (A-pose, 1.80 m, 16 parts); reference render in `assets/humanoid_base_4k.png`
- `docs/modeling.md` - how to model characters and props well
