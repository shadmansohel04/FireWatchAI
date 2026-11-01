# FireWatch-AI

AI-based fire and smoke detection system: GT07 capstone project (COE/ELE 70A/B), supervised by Ghassem Tofighi.

A camera feed is checked frame by frame by a YOLO model that runs locally. When fire or smoke persists across several frames, the system raises an alert, logs the event, and sends the frame to a Vision LLM, which writes a short incident report. Detection and alerts work without internet. The LLM report is an extra layer on top.

The system currently runs on a laptop. Deployment on a Raspberry Pi 5 with LEDs and a buzzer is planned.

## How it works

1. **Video input**: webcam or a video file (OpenCV).
2. **Detection**: YOLO model (`detection/yolo/best.pt`, trained from `yolo11n`) finds fire and smoke.
3. **Event tracking**: an alert fires only if fire/smoke is seen in 7 of the last 10 frames, with a 5 s cooldown between alerts.
4. **On an alert**:
   - the frame is saved to `event_frames/` and logged in `event_log.json`
   - the frame is sent to the Vision LLM (Claude) in the background, and a report is written to `incident_reports/`

## Team

| Role | Student | Responsibility |
|---|---|---|
| A | Azaan Cheema | Hardware: Raspberry Pi, camera, LEDs/buzzer, enclosure |
| B | Jennifer Lay | YOLO model: datasets, training, detection |
| C | Safwan Haque | Vision LLM and incident reports |
| D | Shadman Sohel | System integration, logging, evaluation |

## Project structure

```
detection/
  __main__.py          # pipeline entry point (python -m detection)
  llm_handler.py       # sends events to the Vision LLM
  yolo/                # YOLO detection
    detector.py        # YOLO model wrapper
    mock_detector.py   # fake detector for testing without the model
    best.pt            # trained YOLO weights
  tracking/            # event tracking & logging
    event_tracker.py   # sliding-window event confirmation
    event_logger.py    # saves event frames + JSON log
  vision_llm/          # Vision LLM + report code (see its README)
  tests/               # unit tests
    test_video/        # test videos
tools/
  llm_sample_test.py   # tests the Vision LLM on sample images/videos
```

## Setup

```bash
pip install -r requirements.txt
```

For LLM incident reports, set an Anthropic API key. Without one, detection and alerts still work and the reports are marked "unavailable".

```bash
export ANTHROPIC_API_KEY="sk-ant-..."         # macOS / Linux
$env:ANTHROPIC_API_KEY = "sk-ant-..."         # Windows PowerShell
```

To try the pipeline without a key, set `FIREWATCH_LLM_MOCK=1` to use a fake LLM response.

## Running

```bash
python -m detection                                # webcam
python -m detection --source path/to/video.mp4     # video file
python -m detection --mock                         # mock YOLO (no weights needed)
```

Press `q` or `Esc` to quit.

## Tests

```bash
python -m pytest detection/tests/ -v
```

Test video links are in `detection/tests/test_video/videos.md`.
