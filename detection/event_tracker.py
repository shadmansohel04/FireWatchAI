import time
from collections import deque

class EventTracker:
  def __init__(self, window_size=10, threshold=7, cooldown_seconds=5):
    self.window_size = window_size
    self.threshold = threshold
    self.cooldown_seconds = cooldown_seconds
    
    self.detection_window = deque(maxlen=self.window_size)
    self.current_count = 0
    self.last_alert_time = 0

  def update(self, is_detected):
    """
    Updates the sliding window and maintains an O(1) running count.
    Returns a tuple: (should_trigger_alert, timestamp)
    """
    new_val = 1 if is_detected else 0
    
    # If the window is full, subtract the oldest value before it gets overwritten
    if len(self.detection_window) == self.window_size:
      self.current_count -= self.detection_window[0]
      
    self.detection_window.append(new_val)
    self.current_count += new_val
    
    current_time = time.time()
    if self.current_count >= self.threshold:
      if (current_time - self.last_alert_time) > self.cooldown_seconds:
        self.last_alert_time = current_time
        self.detection_window.clear()
        self.current_count = 0
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        return True, timestamp
            
    return False, None