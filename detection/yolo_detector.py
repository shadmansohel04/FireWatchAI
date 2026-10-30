from pathlib import Path
from typing import Any, Tuple
from ultralytics import YOLO

class YOLODetector:
  def __init__(self, weights_path: str = "best.pt", conf: float = 0.45) -> None:
    path = Path(weights_path)
    if not path.is_absolute():
      path = Path(__file__).parent / path
    if not path.exists():
      raise FileNotFoundError(f"Weights file not found: {path}")

    self.model = YOLO(str(path))
    self.conf = conf

  def detect(self, frame: Any) -> Tuple[bool, Any]:
    result = self.model.predict(frame, conf=self.conf, verbose=False)[0]
    return len(result.boxes) > 0, result.plot()