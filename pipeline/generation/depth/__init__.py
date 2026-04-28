# Estimation de profondeur (Depth Map)
from .depth_estimation import DepthEstimator

__all__ = [
    "DepthEstimator",
    "DepthAnythingEstimator",
    "get_depth_estimator",
]

try:
    from .depth_anything_estimator import DepthAnythingEstimator, get_depth_estimator
except ImportError:
    DepthAnythingEstimator = None  # type: ignore[misc, assignment]

    def get_depth_estimator(use_depth_anything: bool | None = None):
        """Fallback : toujours MiDaS si depth_anything_estimator absent."""
        return DepthEstimator()