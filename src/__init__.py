"""
IMP302 Classical Image Processing Pipeline for License Plate Restoration,
Detection, Segmentation, and Pattern Recognition.
"""

from . import restoration
from . import detection
from . import segmentation
from . import recognition
from . import pipeline

__all__ = ["restoration", "detection", "segmentation", "recognition", "pipeline"]

