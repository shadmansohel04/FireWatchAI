import os
import atexit
import threading
from typing import Any, Optional, Sequence

from detection.vision_llm import ReportWorker, VisionLLMClient
from detection.vision_llm.mock_client import MockAnthropicClient

REPORTS_DIR = os.getenv("FIREWATCH_REPORTS_DIR", "incident_reports")

_worker: Optional[ReportWorker] = None
_worker_lock = threading.Lock()


def get_llm_worker() -> ReportWorker:
  """Starts the single background Vision LLM worker on first use."""
  global _worker
  with _worker_lock:
    if _worker is None:
      api = MockAnthropicClient(latency_s=(1.0, 2.0)) if os.getenv("FIREWATCH_LLM_MOCK") == "1" else None
      _worker = ReportWorker(client=VisionLLMClient(client=api), reports_dir=REPORTS_DIR)
      _worker.start()
      atexit.register(shutdown_llm_worker)
    return _worker


def spawn_llm_thread(frame_copy: Any, timestamp: str, detections: Optional[Sequence[dict]] = None) -> None:
  """
  Hands a verified event to the Vision LLM without blocking the edge pipeline.
  The frame is queued for the background worker, which submits it to the
  Vision LLM and writes the incident report to REPORTS_DIR.

  detections (optional): YOLO boxes for the event, in the format
  [{"label": "fire", "bbox": [x1, y1, x2, y2], "confidence": 0.87}, ...].
  When given, they are added to the prompt and a crop of the region is sent too.
  """
  print(f"\n[ALERT - {timestamp}] FIRE FOUND!")
  get_llm_worker().submit(frame_copy, timestamp, detections)
  print("-> Queued frame for Vision LLM analysis...\n")


def shutdown_llm_worker(timeout: float = 30.0) -> None:
  """Gives queued reports up to `timeout` seconds to finish (runs automatically at exit)."""
  if _worker is None or _worker.pending() == 0:
    return
  print(f"[LLM] Waiting up to {timeout:.0f}s for {_worker.pending()} pending report(s)... (Ctrl+C to skip)")
  try:
    if not _worker.stop(timeout):
      print("[LLM] Pending reports were not finished.")
  except KeyboardInterrupt:
    print("[LLM] Skipped pending reports.")
