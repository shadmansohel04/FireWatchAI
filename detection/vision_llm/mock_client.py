"""
Offline stand-in for anthropic.Anthropic, so the pipeline can be run and tested
without an API key or internet connection (FIREWATCH_LLM_MOCK=1).
Mirrors the parts of the SDK response that VisionLLMClient reads.
"""
import json
import random
import time
from types import SimpleNamespace
from typing import Any, Optional, Sequence

MOCK_ANALYSIS = {
  "environment": "indoor",
  "visible_evidence": "MOCK RESPONSE: no image was analyzed; this is a fixed placeholder.",
  "fire_visible": "yes",
  "smoke_visible": "yes",
  "fire": {"size": "small", "flame_colour": "yellow_orange"},
  "smoke": {"density": "moderate", "colour": "dark_grey", "spread": "localized", "direction": "not_observable"},
  "location_in_frame": "not_observable",
  "not_observable_reasons": "Mock mode: direction and location are not evaluated.",
  "incident_summary": "MOCK RESPONSE: placeholder report generated without calling the Vision LLM.",
  "assessment_confidence": "low",
  "recommended_status": "verify_on_site",
}


def make_response(text: str, stop_reason: str = "end_turn", category: Optional[str] = None) -> SimpleNamespace:
  return SimpleNamespace(
    model="mock-vision-llm",
    stop_reason=stop_reason,
    stop_details=SimpleNamespace(category=category) if stop_reason == "refusal" else None,
    content=[SimpleNamespace(type="text", text=text)],
    usage=SimpleNamespace(input_tokens=0, output_tokens=0),
  )


class MockAnthropicClient:
  def __init__(self, responses: Optional[Sequence[Any]] = None, latency_s: tuple = (0.0, 0.0)) -> None:
    """
    responses: replies returned in order (the last one repeats). Each is a dict
    (sent as JSON), a raw string, a response object from make_response(), or an
    exception to raise. Defaults to MOCK_ANALYSIS.
    """
    self.responses = list(responses or [MOCK_ANALYSIS])
    self.latency_s = latency_s
    self.requests: list = []
    self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

  def _create(self, **request: Any) -> SimpleNamespace:
    self.requests.append(request)
    if self.latency_s[1] > 0:
      time.sleep(random.uniform(*self.latency_s))

    index = min(len(self.requests), len(self.responses)) - 1
    reply = self.responses[index]
    if isinstance(reply, BaseException):
      raise reply
    if isinstance(reply, dict):
      return make_response(json.dumps(reply))
    if isinstance(reply, str):
      return make_response(reply)
    return reply
