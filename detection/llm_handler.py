import time
import threading
from typing import Any

def mock_llm_handler(frame_copy: Any, timestamp: str) -> None:
  """
  Asynchronous handler to process verified events without blocking the main edge pipeline.
  Currently prints alerts; designed to format structured outputs and submit representative 
  frames to the Vision LLM in Phase 4.
  """
  print(f"\n[ALERT - {timestamp}] FIRE FOUND!")
  print("-> Spawning API call thread for further interpretation...\n")
  
  time.sleep(2)
  

def spawn_llm_thread(frame_copy: Any, timestamp: str) -> None:
  api_thread = threading.Thread(
    target=mock_llm_handler, 
    args=(frame_copy, timestamp),
    daemon=True
  )
  api_thread.start()