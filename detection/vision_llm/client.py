"""
Vision LLM client: encodes an event frame (plus an optional crop of the detected
region), sends it to Claude with the FireWatch prompt and output schema, and
returns a validated analysis.
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import cv2
import numpy as np

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schema import ANALYSIS_SCHEMA, InvalidResponseError, parse_analysis

# Server-side refusal fallback: if the model's safety classifiers decline a request,
# the API re-runs it on Anthropic's recommended fallback model in the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}

ImageInput = Union[np.ndarray, str, Path]


class VisionLLMError(Exception):
  """The Vision LLM could not produce a usable analysis for this event."""


@dataclass(frozen=True)
class LLMConfig:
  model: str = "claude-opus-5-5"
  effort: str = "medium"
  max_tokens: int = 16000
  timeout_seconds: float = 60.0
  max_retries: int = 2               # SDK retries for timeouts, 429s and 5xx errors
  invalid_response_retries: int = 1  # extra attempts when a reply fails validation
  max_image_edge: int = 2576         # Claude's maximum long-edge resolution
  jpeg_quality: int = 90
  crop_padding: float = 0.25         # context kept around the detected boxes, per side

  @classmethod
  def from_env(cls) -> "LLMConfig":
    return cls(
      model=os.getenv("FIREWATCH_LLM_MODEL", cls.model),
      effort=os.getenv("FIREWATCH_LLM_EFFORT", cls.effort),
      timeout_seconds=float(os.getenv("FIREWATCH_LLM_TIMEOUT", cls.timeout_seconds)),
    )


@dataclass
class AnalysisResult:
  analysis: dict
  model: str          # model that produced the answer (differs from config if a fallback ran)
  latency_s: float
  input_tokens: int
  output_tokens: int
  attempts: int


def load_image(image: ImageInput) -> np.ndarray:
  if isinstance(image, (str, Path)):
    frame = cv2.imread(str(image))
    if frame is None:
      raise VisionLLMError(f"Could not read image: {image}")
    return frame
  if isinstance(image, np.ndarray) and image.ndim in (2, 3) and image.size > 0:
    return image
  raise VisionLLMError(f"Unsupported image input: {type(image).__name__}")


def crop_detections(frame: np.ndarray, detections: Optional[Sequence[dict]], padding: float) -> Optional[np.ndarray]:
  """
  Crops the union of the detection boxes ([x1, y1, x2, y2] pixels) with some
  padding for context. Returns None when there are no usable boxes or when the
  crop would cover most of the frame anyway.
  """
  boxes = [d["bbox"] for d in detections or [] if d.get("bbox") is not None and len(d["bbox"]) == 4]
  if not boxes:
    return None

  height, width = frame.shape[:2]
  x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
  x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
  pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
  x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
  x2, y2 = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))

  if x2 - x1 < 8 or y2 - y1 < 8:
    return None
  if (x2 - x1) * (y2 - y1) > 0.8 * width * height:
    return None
  return frame[y1:y2, x1:x2]


def encode_image(frame: np.ndarray, max_edge: int, quality: int) -> dict:
  """Returns a base64 JPEG image content block, downscaled only if over max_edge."""
  height, width = frame.shape[:2]
  scale = max_edge / max(height, width)
  if scale < 1:
    frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

  ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
  if not ok:
    raise VisionLLMError("JPEG encoding failed")
  return {
    "type": "image",
    "source": {
      "type": "base64",
      "media_type": "image/jpeg",
      "data": base64.standard_b64encode(buffer.tobytes()).decode("ascii"),
    },
  }


def _describe_api_error(error: Exception) -> str:
  try:
    import anthropic
  except ImportError:
    return str(error)

  if isinstance(error, TypeError) and "authentication" in str(error).lower():
    return "no Anthropic API key found (set ANTHROPIC_API_KEY)"
  if isinstance(error, anthropic.AuthenticationError):
    return "API key rejected (check ANTHROPIC_API_KEY)"
  if isinstance(error, anthropic.RateLimitError):
    return "rate limited by the API"
  if isinstance(error, anthropic.APITimeoutError):
    return "API request timed out"
  if isinstance(error, anthropic.APIConnectionError):
    return "could not reach the API (no internet connection?)"
  if isinstance(error, anthropic.APIStatusError):
    return f"API error {error.status_code}: {error.message}"
  return f"{type(error).__name__}: {error}"


class VisionLLMClient:
  def __init__(self, config: Optional[LLMConfig] = None, client: Any = None) -> None:
    """
    client: an anthropic.Anthropic instance (created on first use if omitted),
    or a stand-in such as MockAnthropicClient for offline runs and tests.
    """
    self.config = config or LLMConfig.from_env()
    self._client = client

  def _api(self) -> Any:
    if self._client is None:
      try:
        import anthropic
      except ImportError as e:
        raise VisionLLMError("anthropic package is not installed (pip install anthropic)") from e
      self._client = anthropic.Anthropic(
        timeout=self.config.timeout_seconds,
        max_retries=self.config.max_retries,
      )
    return self._client

  def build_request(
    self,
    frame: np.ndarray,
    timestamp: Optional[str] = None,
    detections: Optional[Sequence[dict]] = None,
  ) -> dict:
    cfg = self.config
    content = [
      {"type": "text", "text": "Image 1: full camera frame."},
      encode_image(frame, cfg.max_image_edge, cfg.jpeg_quality),
    ]
    crop = crop_detections(frame, detections, cfg.crop_padding)
    if crop is not None:
      content += [
        {"type": "text", "text": "Image 2: close-up of the detected region."},
        encode_image(crop, cfg.max_image_edge, cfg.jpeg_quality),
      ]
    content.append({"type": "text", "text": build_user_prompt(timestamp, detections, crop is not None)})

    request = {
      "model": cfg.model,
      "max_tokens": cfg.max_tokens,
      "system": SYSTEM_PROMPT,
      "messages": [{"role": "user", "content": content}],
      "output_config": {
        "effort": cfg.effort,
        "format": {"type": "json_schema", "schema": ANALYSIS_SCHEMA},
      },
    }
    if cfg.model in FALLBACK_MODELS:
      request["betas"] = [FALLBACK_BETA]
      request["fallbacks"] = "default"
    return request

  def analyze(
    self,
    image: ImageInput,
    timestamp: Optional[str] = None,
    detections: Optional[Sequence[dict]] = None,
  ) -> AnalysisResult:
    """
    Sends one event to the Vision LLM. Blocks until it answers, so call it from a
    background thread (see ReportWorker). Raises VisionLLMError on any failure.
    """
    frame = load_image(image)
    request = self.build_request(frame, timestamp, detections)
    start = time.monotonic()
    input_tokens = output_tokens = 0
    last_problem = "no attempts made"

    for attempt in range(1, self.config.invalid_response_retries + 2):
      try:
        response = self._api().beta.messages.create(**request)
      except VisionLLMError:
        raise
      except Exception as e:  # the SDK has already retried transient failures
        raise VisionLLMError(_describe_api_error(e)) from e

      usage = getattr(response, "usage", None)
      input_tokens += getattr(usage, "input_tokens", 0) or 0
      output_tokens += getattr(usage, "output_tokens", 0) or 0

      if response.stop_reason == "refusal":
        category = getattr(getattr(response, "stop_details", None), "category", None)
        raise VisionLLMError(f"model declined the request (category: {category})")

      text = "".join(block.text for block in response.content if block.type == "text")
      try:
        analysis = parse_analysis(text)
      except InvalidResponseError as e:
        last_problem = "reply was cut off (max_tokens)" if response.stop_reason == "max_tokens" else str(e)
        print(f"[LLM] Invalid response on attempt {attempt}: {last_problem}")
        continue

      return AnalysisResult(
        analysis=analysis,
        model=getattr(response, "model", self.config.model),
        latency_s=round(time.monotonic() - start, 2),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        attempts=attempt,
      )

    raise VisionLLMError(f"invalid response: {last_problem}")
