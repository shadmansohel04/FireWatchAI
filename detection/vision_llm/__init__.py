"""
FireWatch-AI Vision LLM and reporting component (Student C).

Takes frames of verified fire/smoke events, asks a Vision LLM (Claude) for the
visible characteristics in a fixed JSON schema, and turns the answer into an
incident report. See README.md in this folder.
"""
from .client import AnalysisResult, LLMConfig, VisionLLMClient, VisionLLMError
from .report import build_report, format_report_text, save_report
from .schema import ANALYSIS_SCHEMA, NOT_OBSERVABLE, parse_analysis, validate_analysis
from .worker import ReportWorker, process_event

__all__ = [
  "ANALYSIS_SCHEMA", "NOT_OBSERVABLE", "AnalysisResult", "LLMConfig", "ReportWorker",
  "VisionLLMClient", "VisionLLMError", "build_report", "format_report_text",
  "parse_analysis", "process_event", "save_report", "validate_analysis",
]
