"""YOLO-based fire/smoke detector and its offline mock."""

from .detector import YOLODetector
from .mock_detector import MockYOLODetector

__all__ = ["YOLODetector", "MockYOLODetector"]

