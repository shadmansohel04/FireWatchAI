class YOLO_Pipeline:
  def __init__(self, weights_path="best.pt"):
    self.weights_path = weights_path
    print(f"Loaded YOLO model from {weights_path}")

  def detect(self, frame):
    """
    Processes the frame and returns a boolean indicating if fire/smoke was found,
    along with bounding box data.
    """
    # Run inference here
    is_fire_detected = False # Replace with actual detection logic
    
    return is_fire_detected