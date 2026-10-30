"""
Background worker for Vision LLM reports (the "queue -> background worker"
part of the reporting workflow). The detection loop only ever calls submit(),
which never blocks, so slow or failed API calls can't delay detection or the
local alerts.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Optional, Sequence

from .client import VisionLLMClient
from .report import build_report, format_report_text, save_report

_STOP = object()


def process_event(
  client: VisionLLMClient,
  frame: Any,
  timestamp: Optional[str],
  detections: Optional[Sequence[dict]],
  reports_dir: Optional[str],
) -> dict:
  """
  Runs one event through the Vision LLM and returns the incident report.
  Never raises: any failure produces a report with status "unavailable".
  """
  try:
    result = client.analyze(frame, timestamp, detections)
    report = build_report(timestamp, detections, result=result)
  except Exception as e:
    report = build_report(timestamp, detections, error=str(e))

  if reports_dir:
    try:
      report["report_path"] = str(save_report(report, reports_dir))
    except OSError as e:
      print(f"[LLM] Could not save report {report['report_id']}: {e}")
  return report


class ReportWorker:
  def __init__(
    self,
    client: Optional[VisionLLMClient] = None,
    reports_dir: Optional[str] = "incident_reports",
    max_pending: int = 4,
    on_report: Optional[Callable[[dict], None]] = None,
  ) -> None:
    """
    on_report: optional callback run (on the worker thread) with each finished
    report, e.g. to attach it to the event log.
    """
    self.client = client or VisionLLMClient()
    self.reports_dir = reports_dir
    self.on_report = on_report
    self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
    self._thread: Optional[threading.Thread] = None

  def start(self) -> None:
    if self._thread is None or not self._thread.is_alive():
      self._thread = threading.Thread(target=self._run, name="vision-llm-worker", daemon=True)
      self._thread.start()

  def submit(self, frame: Any, timestamp: Optional[str], detections: Optional[Sequence[dict]] = None) -> None:
    """Queues an event without blocking. If the queue is full the oldest pending event is dropped."""
    job = (frame, timestamp, detections)
    while True:
      try:
        self._queue.put_nowait(job)
        return
      except queue.Full:
        try:
          dropped = self._queue.get_nowait()
          self._queue.task_done()
          print(f"[LLM] Queue full; dropped pending event from {dropped[1]}")
        except queue.Empty:
          pass

  def pending(self) -> int:
    """Events queued or currently being analyzed."""
    return self._queue.unfinished_tasks

  def wait_until_idle(self, timeout: Optional[float] = None) -> bool:
    """Waits for queued events to finish. Returns False if the timeout ran out first."""
    deadline = None if timeout is None else time.monotonic() + timeout
    with self._queue.all_tasks_done:
      while self._queue.unfinished_tasks:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
          return False
        self._queue.all_tasks_done.wait(remaining)
    return True

  def stop(self, timeout: Optional[float] = None) -> bool:
    """
    Finishes queued events (up to timeout), then stops the thread. Returns False
    if events were still pending; the daemon thread then ends with the program.
    """
    if not self.wait_until_idle(timeout):
      return False
    if self._thread is not None and self._thread.is_alive():
      self._queue.put(_STOP)
      self._thread.join(timeout=1.0)
    return True

  def _run(self) -> None:
    while True:
      job = self._queue.get()
      try:
        if job is _STOP:
          return
        frame, timestamp, detections = job
        report = process_event(self.client, frame, timestamp, detections, self.reports_dir)
        self._announce(report)
        if self.on_report is not None:
          self.on_report(report)
      except Exception as e:  # keep the worker alive no matter what
        print(f"[LLM] Report worker error: {e}")
      finally:
        self._queue.task_done()

  @staticmethod
  def _announce(report: dict) -> None:
    where = f" -> {report['report_path']}" if "report_path" in report else ""
    if report["status"] == "complete":
      print(f"\n[LLM] Report {report['report_id']}{where}")
      print(format_report_text(report) + "\n")
    else:
      print(f"[LLM] Analysis unavailable for event {report['event_timestamp']}: {report['error']}{where}")
