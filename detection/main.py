import os
import cv2
import argparse
from event_tracker import EventTracker
from llm_handler import spawn_llm_thread
from yolo_detector import YOLODetector


def main() -> None:
  parser = argparse.ArgumentParser(description="FireWatch-AI Vision Pipeline")
  parser.add_argument(
    '--source',
    type=str,
    default='0',
    help='Video source: "0" for live webcam, or a file path (e.g., "test_video.mp4")'
  )
  default_weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
  parser.add_argument(
    '--weights',
    type=str,
    default=default_weights,
    help='Path to the trained YOLO weights'
  )
  args = parser.parse_args()

  source = int(args.source) if args.source.isdigit() else args.source
  cap = cv2.VideoCapture(source)

  if not cap.isOpened():
    print(f"Error: Could not open video source {source}")
    return

  tracker = EventTracker(window_size=10, threshold=7, cooldown_seconds=5)
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