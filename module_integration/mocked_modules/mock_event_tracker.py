class DummyEventTracker:
  def __init__(self, persistence_threshold: int = 5) -> None:
    # Number of consecutive or near-consecutive frames required to confirm an event
    self.persistence_threshold = persistence_threshold
    self.label_counters = {"fire": 0, "smoke": 0}

def update(self, yolo_predictions: list[dict]) -> dict | None:
    detected_labels = {pred["label"] for pred in yolo_predictions if pred.get("label")}

    for label in self.label_counters.keys():
      if label in detected_labels:
        self.label_counters[label] += 1
      else:
        self.label_counters[label] = max(0, self.label_counters[label] - 1)

    confirmed_labels = [
      label for label, count in self.label_counters.items() 
      if count >= self.persistence_threshold
    ]

    if confirmed_labels:
      return {
        "status": "PERSISTENT_EVENT_DETECTED",
        "confirmed_labels": confirmed_labels,
        "latest_predictions": yolo_predictions
      }
    
    return None