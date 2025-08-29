"""Integration modules for Isaac Lab environment."""

from .space_converters import ObservationConverter, ActionConverter, ObservationSpaceConfig

# Import wrapper only when Isaac Lab is available
try:
    from .isaac_lab_wrapper import IsaacLabDiffusionWrapper
    __all__ = [
        "ObservationConverter",
        "ActionConverter",
        "ObservationSpaceConfig",
        "IsaacLabDiffusionWrapper",
    ]
except ImportError:
    # Isaac Lab not available - only export converters
    __all__ = [
        "ObservationConverter",
        "ActionConverter",
        "ObservationSpaceConfig",
    ]