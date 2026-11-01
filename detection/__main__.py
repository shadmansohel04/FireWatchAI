"""
FireWatch-AI Vision Pipeline entry point.

Usage:
  python -m detection                             # live webcam
  python -m detection --source test_video.mp4     # video file
  python -m detection --mock                      # use mock YOLO (no weights needed)
"""
import os
import argparse

import cv2
from dotenv import load_dotenv

# Load environment variables from .env file in project root or current directory
load_dotenv()

from detection.tracking import EventTracker, spawn_logger_thread
from detection.llm_handler import spawn_llm_thread


def main() -> None:
  parser = argparse.ArgumentParser(description="FireWatch-AI Vision Pipeline")
  parser.add_argument(
    '--source',
    type=str,
    default='0',
    help='Video source: "0" for live webcam, or a file path (e.g., "test_video.mp4")'
  )
  default_weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo", "best.pt")
  parser.add_argument(
    '--weights',
    type=str,
    default=default_weights,
    help='Path to the trained YOLO weights'
  )
  parser.add_argument(
    '--mock',
    action='store_true',
    help='Use the mock YOLO detector (no weights or ultralytics needed)'
  )
  args = parser.parse_args()

  source = int(args.source) if args.source.isdigit() else args.source
  cap = cv2.VideoCapture(source)

  if not cap.isOpened():
    print(f"Error: Could not open video source {source}")
    return

  tracker = EventTracker(window_size=10, threshold=7, cooldown_seconds=5)

  if args.mock:
    from detection.yolo import MockYOLODetector
    detector = MockYOLODetector()
    print("Using mock YOLO detector.")
  else:
    from detection.yolo import YOLODetector
    detector = YOLODetector(weights_path=args.weights)

  print("System initialized. Press 'q' or Esc in the video window (or Ctrl+C) to quit.")

  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        print("Video stream ended or interrupted.")
        break

      is_fire_detected, annotated_frame = detector.detect(frame)

      trigger_alert, timestamp = tracker.update(is_fire_detected)

      if trigger_alert and timestamp:
        frame_to_analyze = annotated_frame.copy()
        spawn_llm_thread(frame_to_analyze, timestamp)
        spawn_logger_thread(frame_to_analyze, timestamp)

      cv2.imshow("FireWatch-AI Camera Stream", annotated_frame)

      key = cv2.waitKey(1) & 0xFF
      if key in (ord('q'), ord('Q'), 27):
        break
  except KeyboardInterrupt:
    print("Interrupted by user.")
  finally:
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
  main()

