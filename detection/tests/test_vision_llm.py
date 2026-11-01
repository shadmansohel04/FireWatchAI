"""
Offline tests for the Vision LLM component. No API key or network needed.

  python -m pytest detection/tests/ -v
  python -m unittest discover -s detection/tests -v
"""
import copy
import json
import time
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from detection.vision_llm import LLMConfig, ReportWorker, VisionLLMClient, VisionLLMError, process_event
from detection.vision_llm.client import FALLBACK_BETA, crop_detections, encode_image
from detection.vision_llm.mock_client import MOCK_ANALYSIS, MockAnthropicClient, make_response
from detection.vision_llm.report import format_report_text
from detection.vision_llm.schema import ANALYSIS_SCHEMA, InvalidResponseError, parse_analysis, validate_analysis

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
DETECTIONS = [{"label": "fire", "bbox": [100, 120, 180, 220], "confidence": 0.87}]


def make_client(*responses, **config) -> VisionLLMClient:
  return VisionLLMClient(LLMConfig(**config), client=MockAnthropicClient(responses or None))


class SchemaTests(unittest.TestCase):
  def test_mock_analysis_is_valid(self):
    self.assertEqual(validate_analysis(MOCK_ANALYSIS), [])

  def test_all_not_observable_is_valid(self):
    analysis = copy.deepcopy(MOCK_ANALYSIS)
    analysis["fire"] = {"size": "not_observable", "flame_colour": "not_observable"}
    analysis["smoke"] = {key: "not_observable" for key in analysis["smoke"]}
    self.assertEqual(validate_analysis(analysis), [])

  def test_rejects_unknown_value_missing_and_extra_fields(self):
    analysis = copy.deepcopy(MOCK_ANALYSIS)
    analysis["smoke"]["density"] = "very thick"
    del analysis["incident_summary"]
    analysis["extra"] = "x"
    errors = validate_analysis(analysis)
    self.assertTrue(any("smoke.density" in e for e in errors))
    self.assertTrue(any("incident_summary: missing" in e for e in errors))
    self.assertTrue(any("extra: unexpected" in e for e in errors))

  def test_parse_accepts_code_fence_and_rejects_non_json(self):
    fenced = "```json\n" + json.dumps(MOCK_ANALYSIS) + "\n```"
    self.assertEqual(parse_analysis(fenced), MOCK_ANALYSIS)
    with self.assertRaises(InvalidResponseError):
      parse_analysis("The image shows a fire.")

  def test_schema_is_strict_for_structured_outputs(self):
    def objects(schema):
      if schema["type"] == "object":
        yield schema
        for child in schema["properties"].values():
          yield from objects(child)
    for obj in objects(ANALYSIS_SCHEMA):
      self.assertIs(obj["additionalProperties"], False)
      self.assertEqual(set(obj["required"]), set(obj["properties"]))


class ImageTests(unittest.TestCase):
  def test_crop_is_padded_and_clamped(self):
    crop = crop_detections(FRAME, [{"bbox": [0, 0, 100, 100]}], padding=0.25)
    self.assertEqual(crop.shape[:2], (125, 125))

  def test_crop_skipped_without_boxes_or_when_almost_full_frame(self):
    self.assertIsNone(crop_detections(FRAME, None, 0.25))
    self.assertIsNone(crop_detections(FRAME, [{"label": "fire"}], 0.25))
    self.assertIsNone(crop_detections(FRAME, [{"bbox": [10, 10, 630, 470]}], 0.25))

  def test_large_frames_are_downscaled(self):
    import base64
    import cv2
    big = np.zeros((3000, 4000, 3), dtype=np.uint8)
    block = encode_image(big, max_edge=2576, quality=80)
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(block["source"]["data"]), np.uint8), cv2.IMREAD_COLOR)
    self.assertEqual(max(decoded.shape[:2]), 2576)
    self.assertEqual(block["source"]["media_type"], "image/jpeg")


class ClientTests(unittest.TestCase):
  def test_request_shape(self):
    client = make_client()
    client.analyze(FRAME, "2026-10-30 12:00:00", DETECTIONS)
    request = client._client.requests[0]

    self.assertEqual(request["model"], "claude-opus-5-5")
    self.assertEqual(request["output_config"]["format"], {"type": "json_schema", "schema": ANALYSIS_SCHEMA})
    self.assertEqual(request["output_config"]["effort"], "medium")
    self.assertEqual(request["fallbacks"], "default")
    self.assertEqual(request["betas"], [FALLBACK_BETA])
    self.assertNotIn("thinking", request)

    content = request["messages"][0]["content"]
    self.assertEqual([b["type"] for b in content].count("image"), 2)  # full frame + crop
    prompt = content[-1]["text"]
    self.assertIn("fire, confidence 0.87", prompt)
    self.assertIn("2026-10-30 12:00:00", prompt)

  def test_single_image_without_detections_and_no_fallback_for_haiku(self):
    client = make_client(model="claude-haiku-5-5")
    client.analyze(FRAME, "t")
    request = client._client.requests[0]
    self.assertEqual([b["type"] for b in request["messages"][0]["content"]].count("image"), 1)
    self.assertNotIn("fallbacks", request)

  def test_valid_response_is_returned(self):
    result = make_client().analyze(FRAME, "t")
    self.assertEqual(result.analysis, MOCK_ANALYSIS)
    self.assertEqual(result.model, "mock-vision-llm")
    self.assertEqual(result.attempts, 1)

  def test_invalid_reply_is_retried_once(self):
    bad = copy.deepcopy(MOCK_ANALYSIS)
    bad["smoke"]["colour"] = "purple"
    result = make_client(bad, MOCK_ANALYSIS).analyze(FRAME, "t")
    self.assertEqual(result.attempts, 2)

  def test_gives_up_after_retries(self):
    with self.assertRaisesRegex(VisionLLMError, "invalid response"):
      make_client("not json").analyze(FRAME, "t")

  def test_refusal_and_api_errors_raise_vision_llm_error(self):
    with self.assertRaisesRegex(VisionLLMError, "declined"):
      make_client(make_response("", stop_reason="refusal", category="other")).analyze(FRAME, "t")
    with self.assertRaisesRegex(VisionLLMError, "ConnectionError"):
      make_client(ConnectionError("offline")).analyze(FRAME, "t")

  def test_bad_image_input(self):
    with self.assertRaises(VisionLLMError):
      make_client().analyze("does/not/exist.jpg")


class ReportTests(unittest.TestCase):
  def test_complete_report_saved_as_json_and_text(self):
    with tempfile.TemporaryDirectory() as tmp:
      report = process_event(make_client(), FRAME, "2026-10-30 12:00:00", DETECTIONS, tmp)
      self.assertEqual(report["status"], "complete")
      self.assertEqual(report["local_detection"]["classes"], ["fire"])
      self.assertEqual(report["local_detection"]["max_confidence"], 0.87)

      saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
      self.assertEqual(saved["analysis"], MOCK_ANALYSIS)
      text = Path(report["report_path"]).with_suffix(".txt").read_text(encoding="utf-8")
      self.assertIn("VERIFY ON SITE", text)
      self.assertIn("direction: not observable", text)

  def test_failure_produces_unavailable_report(self):
    report = process_event(make_client(TimeoutError("slow")), FRAME, "t", None, None)
    self.assertEqual(report["status"], "unavailable")
    self.assertIsNone(report["analysis"])
    self.assertIn("UNAVAILABLE", format_report_text(report))


class WorkerTests(unittest.TestCase):
  def test_worker_survives_failures_and_reports_every_event(self):
    reports = []
    client = make_client(ConnectionError("offline"), MOCK_ANALYSIS)
    worker = ReportWorker(client, reports_dir=None, on_report=reports.append)
    worker.start()
    worker.submit(FRAME, "first")
    worker.submit(FRAME, "second")
    self.assertTrue(worker.stop(timeout=5))
    self.assertEqual([r["status"] for r in reports], ["unavailable", "complete"])

  def test_submit_never_blocks_and_drops_oldest_when_full(self):
    release = threading.Event()

    class SlowClient:
      def analyze(self, frame, timestamp, detections):
        release.wait(5)
        raise VisionLLMError("stopped")

    reports = []
    worker = ReportWorker(SlowClient(), reports_dir=None, max_pending=2, on_report=reports.append)
    worker.start()
    worker.submit(FRAME, "e1")
    while worker._queue.qsize():  # wait until e1 is being processed
      time.sleep(0.01)
    for name in ("e2", "e3", "e4"):  # queue holds 2, so e2 is dropped
      worker.submit(FRAME, name)
    release.set()
    self.assertTrue(worker.wait_until_idle(timeout=5))
    self.assertEqual([r["event_timestamp"] for r in reports], ["e1", "e3", "e4"])


if __name__ == "__main__":
  unittest.main()
