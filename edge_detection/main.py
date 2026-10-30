import cv2
import argparse
from event_tracker import modu
from llm_handler import spawn_llm_thread

# Import the YOLO wrapper
from best import YOLO_Pipeline

def main():
  parser = argparse.ArgumentParser(description="FireWatch-AI Vision Pipeline")
  parser.add_argument(
    '--source', 
    type=str, 
    default='0', 
    help='Video source: "0" for live webcam, or a file path (e.g., "test_video.mp4")'
  )
  args = parser.parse_args()

  # Determine if source is a webcam index or a video file path
  source = int(args.source) if args.source.isdigit() else args.source
  cap = cv2.VideoCapture(source)

  if not cap.isOpened():
    print(f"Error: Could not open video source {source}")
    return

  # Initialize modular components
  tracker = EventTracker(window_size=10, threshold=7, cooldown_seconds=5)
  model = YOLO_Pipeline("best.pt")

  print("System initialized. Press 'q' to quit.")

  while True:
    ret, frame = cap.read()
    if not ret:
      print("Video stream ended or interrupted.")
      break

    # 1. Edge Detection (YOLO)
    is_fire_detected = model.detect(frame) 

    # 2. Temporal Filtering & Tracking
    trigger_alert, timestamp = tracker.update(is_fire_detected)

    # 3. Vision LLM API Routing
    if trigger_alert:
      frame_to_analyze = frame.copy()
      spawn_llm_thread(frame_to_analyze, timestamp)

    # 4. Local Display (Hardware Pipeline Debugging)
    cv2.imshow("Camera Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()