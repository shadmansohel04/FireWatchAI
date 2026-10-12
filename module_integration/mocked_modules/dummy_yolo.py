import time
import random
from typing import Any

class DummyYOLO:
  def __init__(self, simulate_latency: bool = True) -> None:
    self.simulate_latency = simulate_latency
    self.classes = ['fire', 'smoke', None]

  # Input: frame (Any) - The input frame to be processed, not used in this mock implementation.
  def predict(self, frame: Any) -> list[dict[str, Any]]:
    # Simulate processing latency if enabled
    if self.simulate_latency:
      time.sleep(random.uniform(0.05, 0.15))

    detected_class = random.choices(self.classes, weights=[0.2, 0.2, 0.6], k=1)[0]

    if not detected_class:
      return []

    x_min = random.randint(10, 200)
    y_min = random.randint(10, 200)
    x_max = x_min + random.randint(50, 150)
    y_max = y_min + random.randint(50, 150)
    
    confidence = round(random.uniform(0.65, 0.98), 2)

    return [{
      "label": detected_class,
      "bbox": [x_min, y_min, x_max, y_max],
      "confidence": confidence
    }]