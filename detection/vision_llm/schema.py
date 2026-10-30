"""
Structured output schema for the Vision LLM.

Every visual characteristic is a fixed set of values plus "not_observable", so
the model has an explicit way to say a characteristic can't be judged from the
image instead of guessing. The same schema is sent to the API (structured
outputs) and checked again locally before a response is accepted.
"""
import json
from typing import Any

NOT_OBSERVABLE = "not_observable"

PRESENCE = ["yes", "no", "uncertain"]
FIRE_SIZE = ["small", "medium", "large", NOT_OBSERVABLE]
FLAME_COLOUR = ["yellow_orange", "red", "blue_white", "mixed", NOT_OBSERVABLE]
SMOKE_DENSITY = ["light", "moderate", "dense", NOT_OBSERVABLE]
SMOKE_COLOUR = ["white", "light_grey", "dark_grey", "black", "yellow_brown", "mixed", NOT_OBSERVABLE]
SMOKE_SPREAD = ["localized", "moderate", "widespread", NOT_OBSERVABLE]
SMOKE_DIRECTION = [
  "rising_vertically", "drifting_left", "drifting_right", "toward_camera",
  "away_from_camera", "spreading_along_ceiling", NOT_OBSERVABLE,
]
ENVIRONMENT = ["indoor", "outdoor", NOT_OBSERVABLE]
CONFIDENCE = ["low", "medium", "high"]
RECOMMENDED_STATUS = ["no_action", "continue_monitoring", "verify_on_site", "immediate_response"]


def _enum(values: list) -> dict:
  return {"type": "string", "enum": values}


def _text() -> dict:
  return {"type": "string"}


def _object(properties: dict) -> dict:
  return {
    "type": "object",
    "properties": properties,
    "required": list(properties),
    "additionalProperties": False,
  }


# Field order matters: the evidence is written before the conclusions drawn from it.
ANALYSIS_SCHEMA = _object({
  "environment": _enum(ENVIRONMENT),
  "visible_evidence": _text(),
  "fire_visible": _enum(PRESENCE),
  "smoke_visible": _enum(PRESENCE),
  "fire": _object({
    "size": _enum(FIRE_SIZE),
    "flame_colour": _enum(FLAME_COLOUR),
  }),
  "smoke": _object({
    "density": _enum(SMOKE_DENSITY),
    "colour": _enum(SMOKE_COLOUR),
    "spread": _enum(SMOKE_SPREAD),
    "direction": _enum(SMOKE_DIRECTION),
  }),
  "location_in_frame": _text(),
  "not_observable_reasons": _text(),
  "incident_summary": _text(),
  "assessment_confidence": _enum(CONFIDENCE),
  "recommended_status": _enum(RECOMMENDED_STATUS),
})


class InvalidResponseError(ValueError):
  """The model's reply could not be parsed or does not match ANALYSIS_SCHEMA."""


def _check(value: Any, schema: dict, path: str, errors: list) -> None:
  where = path or "<root>"
  if schema["type"] == "object":
    if not isinstance(value, dict):
      errors.append(f"{where}: expected an object")
      return
    for key in schema["required"]:
      if key not in value:
        errors.append(f"{path}{key}: missing")
    for key, item in value.items():
      if key not in schema["properties"]:
        errors.append(f"{path}{key}: unexpected field")
      else:
        _check(item, schema["properties"][key], f"{path}{key}.", errors)
  elif schema["type"] == "string":
    where = where.rstrip(".")
    if not isinstance(value, str):
      errors.append(f"{where}: expected a string")
    elif "enum" in schema and value not in schema["enum"]:
      errors.append(f"{where}: {value!r} is not one of {schema['enum']}")


def validate_analysis(analysis: Any) -> list:
  """Returns a list of problems; an empty list means the analysis is valid."""
  errors: list = []
  _check(analysis, ANALYSIS_SCHEMA, "", errors)
  return errors


def parse_analysis(text: str) -> dict:
  """Parses the model's JSON reply and validates it against ANALYSIS_SCHEMA."""
  cleaned = text.strip()
  # Structured outputs return bare JSON, but tolerate a ```json fence anyway.
  if cleaned.startswith("```"):
    cleaned = cleaned.strip("`")
    cleaned = cleaned[cleaned.find("{"):]
  try:
    analysis = json.loads(cleaned)
  except json.JSONDecodeError as e:
    raise InvalidResponseError(f"reply is not valid JSON ({e.msg})") from e

  errors = validate_analysis(analysis)
  if errors:
    raise InvalidResponseError("; ".join(errors))
  return analysis


def flatten_analysis(analysis: dict) -> dict:
  """{"smoke": {"density": "light"}} -> {"smoke.density": "light"} (for CSVs and comparisons)."""
  flat = {}
  for key, value in analysis.items():
    if isinstance(value, dict):
      for sub_key, sub_value in value.items():
        flat[f"{key}.{sub_key}"] = sub_value
    else:
      flat[key] = value
  return flat


def not_observable_fields(analysis: dict) -> list:
  return [key for key, value in flatten_analysis(analysis).items() if value == NOT_OBSERVABLE]
