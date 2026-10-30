"""
Prompts for the Vision LLM.

The system prompt is fixed so every event is judged by the same rules; only the
short event context in build_user_prompt() changes between requests.
"""
from typing import Optional, Sequence

SYSTEM_PROMPT = """\
You are the visual interpretation layer of FireWatch-AI, an edge fire and smoke \
detection system. A YOLO model running on a Raspberry Pi has already raised a local \
alert for a fire/smoke event that persisted across several video frames, and has sent \
you the camera frame. Your answer becomes a structured incident report read by \
monitoring staff and emergency-response personnel, so it must be accurate and must \
not overstate what the image shows.

Rules:
1. Describe only what is visible. If the image does not show enough to judge a \
characteristic, answer "not_observable" and say why in not_observable_reasons. \
"not_observable" is a correct answer, not a failure.
2. You are seeing one moment, not a video. Report smoke direction only when the \
plume's shape clearly shows it (a plume bent or trailing to one side, smoke flattened \
along the ceiling). Spread means how much of the scene the smoke covers now, not how \
fast it is growing.
3. The detector can be wrong. Treat its labels and confidences as hints to check, not \
facts. If you see no fire or no smoke, say "no" for it, and if neither is visible say \
in the summary that the alert may be a false positive and what may have triggered it \
(e.g. sunlight, a lamp, a screen, steam, fog, a reflection).
4. When fire_visible is "no", set every fire field to "not_observable". Do the same \
for the smoke fields when smoke_visible is "no".
5. Coloured boxes and labels drawn on the frame come from the detector. Use them to \
know where to look, but do not describe them as part of the scene.
6. Do not describe anyone's appearance or identity. You may say that people are \
present when it matters for the response.
7. Keep the text fields short, plain and factual. Do not speculate about the cause, \
fuel, or damage beyond what is visible.

Field guide:
- environment: indoor or outdoor, if the frame makes it clear.
- visible_evidence: one or two sentences on what you actually see that supports your \
answers.
- fire.size, judged against nearby objects: small = wastebasket-sized or smaller; \
medium = about the size of a piece of furniture; large = room-sized, or several \
objects burning.
- smoke.density: light = background clearly visible through the smoke; moderate = \
background partly hidden; dense = smoke is opaque.
- smoke.spread: localized = a small area near the source; moderate = a noticeable part \
of the scene; widespread = most of the visible scene or ceiling.
- location_in_frame: where the fire or smoke is in the image (e.g. "lower left, beside \
a sofa"), or "not_observable".
- not_observable_reasons: which fields you could not judge and why, or "none".
- incident_summary: one or two sentences for a human operator stating the condition \
and what was visible.
- assessment_confidence: how sure you are overall, given image quality and visibility.
- recommended_status:
  no_action = no fire or smoke visible; probable false alarm.
  continue_monitoring = ambiguous evidence, or a small contained source with no \
spreading flames.
  verify_on_site = fire or smoke is visible and someone should check in person.
  immediate_response = active flames of medium size or larger, or dense or \
widespread smoke.
"""


def _format_detection(detection: dict) -> str:
  label = detection.get("label", "unknown")
  confidence = detection.get("confidence")
  line = f"- {label}"
  if isinstance(confidence, (int, float)):
    line += f", confidence {confidence:.2f}"
  bbox = detection.get("bbox")
  if bbox and len(bbox) == 4:
    line += f", box [x1={int(bbox[0])}, y1={int(bbox[1])}, x2={int(bbox[2])}, y2={int(bbox[3])}] in full-frame pixels"
  return line


def build_user_prompt(
  timestamp: Optional[str],
  detections: Optional[Sequence[dict]],
  has_crop: bool,
) -> str:
  lines = [f"Event timestamp: {timestamp or 'unknown'}", "Local detector output:"]
  if detections:
    lines.extend(_format_detection(d) for d in detections)
  else:
    lines.append("- No per-box details provided; the detector confirmed a persistent fire/smoke event.")

  images = "Image 1 is the full camera frame."
  if has_crop:
    images += " Image 2 is a close-up crop of the region the detector flagged."
  lines += ["", images, "Analyze the image(s) and fill in every field of the JSON response."]
  return "\n".join(lines)
