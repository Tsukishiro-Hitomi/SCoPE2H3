"""Runtime SCoPE overlay for the DiffSynth MiniMax H3 Ref2VA implementation."""

from .adapter import install_scope_h3, scope_model_fn, set_scope_camera
from .camera import build_target_plucker
from .encoding import H3SightlineCoordinatePE

__all__ = ["H3SightlineCoordinatePE", "build_target_plucker", "install_scope_h3", "scope_model_fn", "set_scope_camera"]
