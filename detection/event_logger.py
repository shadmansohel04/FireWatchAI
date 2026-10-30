import os
import json
import threading
from datetime import datetime
from typing import Any

import cv2

LOG_PATH = "event_log.json"
FRAMES_DIR = "event_frames"

# One lock so overlapping alert threads can't hand out the same id
# or write the JSON file at the same time.
_log_lock = threading.Lock()


def _load_events() -> list:
  if not os.path.exists(LOG_PATH):
    return []
  try:
    with open(LOG_PATH, "r", encoding="utf-8") as f:
      return json.load(f).get("events", [])
  except (json.JSONDecodeError, OSError, AttributeError):
    # Corrupt log: keep a backup instead of overwriting it.
    backup = f"{LOG_PATH}.corrupt.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
      os.replace(LOG_PATH, backup)
      print(f"[LOGGER] Could not parse {LOG_PATH}; moved to {backup}")
    except OSError:
      pass
    return []


def log_event_handler(frame_copy: Any, timestamp: str) -> None:
  """
  Asynchronous handler that saves the alert frame to disk and appends an
  indexed entry to the JSON log (creating the file if it doesn't exist).
  """
  try:
    with _log_lock:
      os.makedirs(FRAMES_DIR, exist_ok=True)

      events = _load_events()
      event_id = max((e.get("id", 0) for e in events), default=0) + 1

      frame_path = os.path.join(FRAMES_DIR, f"event_{event_id:05d}.jpg")
      if not cv2.imwrite(frame_path, frame_copy):
        raise IOError(f"cv2.imwrite failed for {frame_path}")

      events.append({
        "id": event_id,
        "timestamp": timestamp,
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        "frame_path": frame_path,
      })

      # Write to a temp file then swap, so a crash can't leave a half-written log.
      tmp_path = f"{LOG_PATH}.tmp"
      with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"events": events}, f, indent=2)
      os.replace(tmp_path, LOG_PATH)

    print(f"[LOGGER] Event #{event_id} saved -> {frame_path}")
  except Exception as e:
    print(f"[LOGGER] Failed to log event: {e}")


def spawn_logger_thread(frame_copy: Any, timestamp: str) -> None:
  log_thread = threading.Thread(
    target=log_event_handler,
    args=(frame_copy, timestamp),
    daemon=True
  )
  log_thread.start()