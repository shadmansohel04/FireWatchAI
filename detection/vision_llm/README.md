# Vision LLM & Reporting Component (Safwan Haque)

When the YOLO + event tracker pipeline confirms a fire/smoke event, this component
sends the event frame to a Vision LLM (Claude), gets back a fixed-format description
of what is visible, and writes a structured incident report. It runs on a background
thread, so detection and local alerts never wait for it. If the LLM can't be reached,
the report is still written and marked "unavailable".

```
event frame + metadata -> queue -> background worker -> Claude (structured JSON)
                                                         |
                                   valid? --no--> retry once -> report "unavailable" + error
                                     | yes
                          parse observations -> incident report (.json + .txt)
```

## Files

| File | Purpose |
|---|---|
| `prompts.py` | System prompt (rules + field guide) and the per-event prompt |
| `schema.py` | Output schema, local validation, `not_observable` handling |
| `client.py` | Image encoding/cropping, API request, response checks, error handling |
| `report.py` | Builds the incident report (JSON) and a readable text version |
| `worker.py` | Queue and background worker thread |
| `mock_client.py` | Offline stand-in for the API (no key needed) |
| `../llm_handler.py` | Entry point used by `main.py` (`spawn_llm_thread`) |
| `../llm_sample_test.py` | Independent testing on sample images or videos |
| `../tests/test_vision_llm.py` | Offline unit tests |

## Setup

```bash
pip install -r requirements.txt
```

Set an Anthropic API key (from https://platform.claude.com):

```powershell
# Windows PowerShell (current session)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

```bash
# macOS / Linux / Raspberry Pi
export ANTHROPIC_API_KEY="sk-ant-..."
```

Optional settings (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `FIREWATCH_LLM_MODEL` | `claude-opus-5-5` | Claude model, e.g. `claude-haiku-5-5` for cheaper and faster runs |
| `FIREWATCH_LLM_EFFORT` | `medium` | `low` / `medium` / `high`: speed vs. thoroughness |
| `FIREWATCH_LLM_TIMEOUT` | `60` | Seconds per API request |
| `FIREWATCH_LLM_MOCK` | off | `1` = use the offline mock instead of the API |
| `FIREWATCH_REPORTS_DIR` | `incident_reports` | Where reports are written |

## How it connects to the rest of the system

`main.py` already calls the handler when the event tracker fires. Nothing else has to change:

```python
spawn_llm_thread(frame_copy, timestamp)
```

It returns immediately. The report appears in `incident_reports/` a few seconds later.
YOLO boxes can optionally be passed in the same format as `mock_yolo.py`. They are added
to the prompt, and a close-up crop of the region is sent along with the full frame:

```python
spawn_llm_thread(frame_copy, timestamp, detections=[
  {"label": "fire", "bbox": [x1, y1, x2, y2], "confidence": 0.87},
])
```

Teammates without an API key can run the full pipeline with `FIREWATCH_LLM_MOCK=1`.
On exit, queued reports get up to 30 s to finish.

## Output

Each event produces `INC-<date>-<time>-<id>.json` and a matching `.txt`. The `analysis` part
of the JSON follows `ANALYSIS_SCHEMA`:

| Field | Values |
|---|---|
| `fire_visible`, `smoke_visible` | yes / no / uncertain |
| `fire.size` | small / medium / large / not_observable |
| `fire.flame_colour` | yellow_orange / red / blue_white / mixed / not_observable |
| `smoke.density` | light / moderate / dense / not_observable |
| `smoke.colour` | white / light_grey / dark_grey / black / yellow_brown / mixed / not_observable |
| `smoke.spread` | localized / moderate / widespread / not_observable |
| `smoke.direction` | rising_vertically / drifting_left / drifting_right / toward_camera / away_from_camera / spreading_along_ceiling / not_observable |
| `environment` | indoor / outdoor / not_observable |
| `assessment_confidence` | low / medium / high |
| `recommended_status` | no_action / continue_monitoring / verify_on_site / immediate_response |
| `visible_evidence`, `location_in_frame`, `not_observable_reasons`, `incident_summary` | short text |

The report also records the event timestamp, the local YOLO detections, the model that
answered, latency, and token usage. These feed into the system evaluation.

## Design notes

- **No unsupported claims.** Every visual characteristic (size, colour, density, spread, direction,
  location) has a `not_observable` option, and fire/smoke presence can be `uncertain`. The prompt
  says that this is a correct answer and asks the model to explain it in `not_observable_reasons`.
  A single frame can't show motion, so direction is only reported when the plume's shape shows it.
- **The detector is treated as a hint.** The model is told YOLO can be wrong. If it sees no
  fire or smoke, it says so and suggests `no_action` (a probable false positive).
- **Guaranteed format.** Claude's structured outputs (`output_config.format`) constrain the
  reply to the schema. The reply is still validated locally before it is accepted, and an
  invalid reply is retried once.
- **Fault handling.** Missing key, no internet, timeouts, rate limits, refusals, and bad
  replies all end in an "unavailable" report with the reason. The worker thread never crashes,
  and the detection loop never blocks. Server-side refusal fallback (`fallbacks: "default"`) is
  enabled for models that support it.
- **Privacy.** The prompt forbids describing people's appearance or identity.

## Testing

Offline unit tests (no key needed):

```bash
python -m unittest discover -s detection/tests -v
```

Independent component test on sample images (real API). This writes per-image reports and a
`summary.csv` with a blank `manual_review` column for checking each answer against the image:

```bash
python detection/llm_sample_test.py --images path/to/fire_smoke_images/
python detection/llm_sample_test.py --video detection/test_video/indoorFire.mp4 --interval 5
python detection/llm_sample_test.py --images path/to/images/ --repeats 3   # answer consistency
python detection/llm_sample_test.py --images path/to/images/ --mock        # dry run, no key
```

Cost: with the default model a 640x480 frame costs roughly 2-4 cents per analysis.
`claude-haiku-5-5` is much cheaper if you're running many test images.
