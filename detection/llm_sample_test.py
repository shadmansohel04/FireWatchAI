"""
Independent test harness for the Vision LLM component (no YOLO, no camera).

Runs sample fire/smoke images, or frames sampled from a video, through the
Vision LLM and writes one incident report per image plus summary.csv, so the
responses can be reviewed for completeness, formatting consistency, and whether
the descriptions match what is visible (fill in the manual_review column).

Examples:
  python detection/llm_sample_test.py --images samples/
  python detection/llm_sample_test.py --video detection/test_video/indoorFire.mp4 --interval 5
  python detection/llm_sample_test.py --images samples/ --repeats 3   # answer consistency
  python detection/llm_sample_test.py --images samples/ --mock        # offline, no API key
"""
import os
import csv
import argparse
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2

from vision_llm import LLMConfig, VisionLLMClient, process_event
from vision_llm.mock_client import MockAnthropicClient
from vision_llm.schema import flatten_analysis, not_observable_fields

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANALYSIS_COLUMNS = [
  "fire_visible", "smoke_visible", "fire.size", "fire.flame_colour", "smoke.density",
  "smoke.colour", "smoke.spread", "smoke.direction", "environment",
  "assessment_confidence", "recommended_status",
]


def collect_images(paths: list) -> list:
  samples = []
  for raw in paths:
    path = Path(raw)
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS) if path.is_dir() else [path]
    for file in files:
      frame = cv2.imread(str(file))
      if frame is None:
        print(f"Skipping unreadable image: {file}")
      else:
        samples.append((str(file), frame))
  return samples


def sample_video(path: str, interval_s: float, max_frames: int) -> list:
  cap = cv2.VideoCapture(path)
  if not cap.isOpened():
    print(f"Could not open video: {path}")
    return []
  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  step = max(1, int(round(fps * interval_s)))
  samples, index = [], 0
  while len(samples) < max_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
      break
    samples.append((f"{Path(path).name}@{index / fps:.1f}s", frame))
    index += step
  cap.release()
  return samples


def summarize(rows: list, repeats: int) -> None:
  complete = [r for r in rows if r["status"] == "complete"]
  print(f"\nRuns: {len(rows)}   complete: {len(complete)}   unavailable: {len(rows) - len(complete)}")
  if not complete:
    return

  latencies = [float(r["latency_s"]) for r in complete]
  print(f"Latency (s): mean {statistics.mean(latencies):.2f}, median {statistics.median(latencies):.2f}, max {max(latencies):.2f}")

  print("\n'not_observable' rate per field:")
  for column in ANALYSIS_COLUMNS:
    rate = sum(r[column] == "not_observable" for r in complete) / len(complete)
    if rate:
      print(f"  {column:<22} {rate:.0%}")

  if repeats > 1:
    by_image = defaultdict(list)
    for r in complete:
      by_image[r["image"]].append(r)
    groups = [g for g in by_image.values() if len(g) > 1]
    if groups:
      print(f"\nAgreement across {repeats} repeats (share of images with identical answers):")
      for column in ANALYSIS_COLUMNS:
        agreed = sum(len({r[column] for r in g}) == 1 for g in groups) / len(groups)
        print(f"  {column:<22} {agreed:.0%}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Independent Vision LLM component test")
  parser.add_argument("--images", nargs="*", default=[], help="Image files and/or folders")
  parser.add_argument("--video", help="Video file to sample frames from")
  parser.add_argument("--interval", type=float, default=5.0, help="Seconds between sampled video frames")
  parser.add_argument("--max-frames", type=int, default=10, help="Max frames to sample from the video")
  parser.add_argument("--repeats", type=int, default=1, help="Analyze each image N times to check consistency")
  parser.add_argument("--out", default="llm_test_results", help="Output folder")
  parser.add_argument("--mock", action="store_true", help="Use the offline mock instead of the real API")
  args = parser.parse_args()

  samples = collect_images(args.images)
  if args.video:
    samples += sample_video(args.video, args.interval, args.max_frames)
  if not samples:
    parser.error("no samples: pass --images and/or --video")

  config = LLMConfig.from_env()
  client = VisionLLMClient(config, client=MockAnthropicClient() if args.mock else None)
  run_dir = os.path.join(args.out, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
  print(f"Model: {'mock' if args.mock else config.model} (effort {config.effort})   samples: {len(samples)}   repeats: {args.repeats}")
  print(f"Writing results to {run_dir}\n")

  rows = []
  for name, frame in samples:
    for repeat in range(1, args.repeats + 1):
      report = process_event(client, frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), None, run_dir)
      row = {"image": name, "repeat": repeat, "status": report["status"], "error": report["error"] or "",
             "model": "", "latency_s": ""}
      row.update({column: "" for column in ANALYSIS_COLUMNS})
      row.update({"not_observable_count": "", "incident_summary": "",
                  "report_file": report.get("report_path", ""), "manual_review": ""})
      if report["analysis"] is not None:
        flat = flatten_analysis(report["analysis"])
        row.update({column: flat[column] for column in ANALYSIS_COLUMNS})
        row.update({
          "model": report["llm"]["model"],
          "latency_s": report["llm"]["latency_s"],
          "not_observable_count": len(not_observable_fields(report["analysis"])),
          "incident_summary": report["analysis"]["incident_summary"],
        })
      rows.append(row)
      outcome = row["recommended_status"] or f"UNAVAILABLE ({row['error']})"
      print(f"[{len(rows)}] {name} (run {repeat}): {outcome}")

  os.makedirs(run_dir, exist_ok=True)
  csv_path = os.path.join(run_dir, "summary.csv")
  with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  summarize(rows, args.repeats)
  print(f"\nSummary CSV: {csv_path}")


if __name__ == "__main__":
  main()
