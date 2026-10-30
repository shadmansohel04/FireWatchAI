"""
Incident reports: combines the local detection event with the Vision LLM
analysis into one structured record (JSON) plus a short readable version.
If the LLM was unavailable the report is still produced, marked "unavailable".
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from .client import AnalysisResult
from .schema import NOT_OBSERVABLE, not_observable_fields

STATUS_COMPLETE = "complete"
STATUS_UNAVAILABLE = "unavailable"


def build_report(
  timestamp: Optional[str],
  detections: Optional[Sequence[dict]] = None,
  result: Optional[AnalysisResult] = None,
  error: Optional[str] = None,
) -> dict:
  detections = list(detections or [])
  confidences = [d["confidence"] for d in detections if isinstance(d.get("confidence"), (int, float))]
  now = datetime.now()

  report = {
    "report_id": f"INC-{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}",
    "event_timestamp": timestamp,
    "generated_at": now.isoformat(timespec="seconds"),
    "status": STATUS_COMPLETE if result is not None else STATUS_UNAVAILABLE,
    "error": None if result is not None else (error or "unknown error"),
    "local_detection": {
      "classes": sorted({d.get("label") for d in detections if d.get("label")}),
      "max_confidence": max(confidences) if confidences else None,
      "detections": detections,
    },
    "llm": None,
    "analysis": None,
  }
  if result is not None:
    report["llm"] = {
      "model": result.model,
      "latency_s": result.latency_s,
      "input_tokens": result.input_tokens,
      "output_tokens": result.output_tokens,
      "attempts": result.attempts,
    }
    report["analysis"] = result.analysis
  return report


def _words(value: str) -> str:
  """Enum value -> readable words ("dark_grey" -> "dark grey")."""
  return "not observable" if value == NOT_OBSERVABLE else value.replace("_", " ")


def _free_text(value: str) -> str:
  return "not observable" if value == NOT_OBSERVABLE else value


def format_report_text(report: dict) -> str:
  """Short human-readable version of a report for monitoring personnel."""
  local = report["local_detection"]
  classes = ", ".join(local["classes"]) or "fire/smoke (details not provided)"
  if local["max_confidence"] is not None:
    classes += f" (max confidence {local['max_confidence']:.2f})"

  lines = [
    f"FIREWATCH-AI INCIDENT REPORT  {report['report_id']}",
    f"Event time:      {report['event_timestamp'] or 'unknown'}",
    f"Local detection: {classes}",
  ]

  analysis = report["analysis"]
  if report["status"] != STATUS_COMPLETE or analysis is None:
    lines += [
      "Vision analysis: UNAVAILABLE",
      f"Reason:          {report['error']}",
      "Local detection and alerts are unaffected. Check the camera feed directly.",
    ]
    return "\n".join(lines)

  fire, smoke = analysis["fire"], analysis["smoke"]
  llm = report["llm"]
  lines += [
    f"Recommendation:  {_words(analysis['recommended_status']).upper()}",
    f"LLM confidence:  {analysis['assessment_confidence']}  ({llm['model']}, {llm['latency_s']} s)",
    "",
    f"Summary: {analysis['incident_summary']}",
    "",
    f"Fire visible:  {analysis['fire_visible']}  | size: {_words(fire['size'])}, "
    f"flame colour: {_words(fire['flame_colour'])}",
    f"Smoke visible: {analysis['smoke_visible']}  | density: {_words(smoke['density'])}, "
    f"colour: {_words(smoke['colour'])}, spread: {_words(smoke['spread'])}, "
    f"direction: {_words(smoke['direction'])}",
    f"Location in frame: {_free_text(analysis['location_in_frame'])}",
    f"Environment:       {_words(analysis['environment'])}",
    f"Evidence:          {analysis['visible_evidence']}",
  ]
  if not_observable_fields(analysis):
    lines.append(f"Not observable:    {analysis['not_observable_reasons']}")
  return "\n".join(lines)


def save_report(report: dict, reports_dir: str) -> Path:
  """Writes <report_id>.json and <report_id>.txt; returns the JSON path."""
  os.makedirs(reports_dir, exist_ok=True)
  json_path = Path(reports_dir) / f"{report['report_id']}.json"
  json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
  json_path.with_suffix(".txt").write_text(format_report_text(report) + "\n", encoding="utf-8")
  return json_path
