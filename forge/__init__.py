"""forge - make images with code, from 16px sprites to 4K renders.

    from forge import sdf, render, post      # 3D: model with SDFs, raymarch, develop
    from forge.sprite import Sprite          # hand-authored pixel art
    from forge.paint import Canvas           # anti-aliased 2D / vector style
"""
from . import kit, noise, paint, post, render, sdf, sprite
from .paint import Canvas
from .render import Camera, render as render_frame, turnaround
from .sdf import Scene
from .sprite import Sprite

__all__ = ["sdf", "render", "post", "sprite", "paint", "noise", "kit",
           "Scene", "Camera", "render_frame", "turnaround", "Sprite", "Canvas"]
