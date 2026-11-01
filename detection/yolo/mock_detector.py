"""
Offline stand-in for YOLODetector, so the pipeline can be run and tested
without the YOLO weights or the ultralytics package.

Implements the same interface as YOLODetector:
  detect(frame) -> Tuple[bool, annotated_frame]
"""
import random
import time
from typing import Any, Tuple

import cv2
import numpy as np


class MockYOLODetector:
  """Drop-in replacement for YOLODetector that returns random detections."""

  def __init__(self, simulate_latency: bool = True, fire_rate: float = 0.4) -> None:
    """
    fire_rate: probability that any given frame is flagged as fire/smoke.
    """
    self.simulate_latency = simulate_latency
    self.fire_rate = fire_rate
    self.classes = ['fire', 'smoke']

  def detect(self, frame: Any) -> Tuple[bool, Any]:
    """
    Matches YOLODetector.detect():
      Returns (is_fire_detected: bool, annotated_frame: ndarray).
    """
    if self.simulate_latency:
      time.sleep(random.uniform(0.05, 0.15))

    is_detected = random.random() < self.fire_rate

    if not is_detected:
      return False, frame

    # Draw a fake bounding box on the frame so the annotated output is visible
    annotated = frame.copy() if isinstance(frame, np.ndarray) else frame
    if isinstance(annotated, np.ndarray) and annotated.ndim >= 2:
      h, w = annotated.shape[:2]
      x1 = random.randint(10, max(11, w // 3))
      y1 = random.randint(10, max(11, h // 3))
      x2 = x1 + random.randint(50, max(51, w // 4))
      y2 = y1 + random.randint(50, max(51, h // 4))

      label = random.choice(self.classes)
      confidence = round(random.uniform(0.65, 0.98), 2)

      color = (0, 0, 255) if label == 'fire' else (128, 128, 128)
      cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
      cv2.putText(
        annotated, f"{label} {confidence:.2f}",
        (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
      )

    return True, annotated

