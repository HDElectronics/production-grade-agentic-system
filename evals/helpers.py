"""Helper functions for the evaluation process."""

import json
import os
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from langfuse.api.resources.commons.types.trace_with_details import TraceWithDetails

from src.system.logs import logger
from evals.schemas import ScoreSchema

# Metrics where 1.0 means the problem is fully present
LOWER_IS_BETTER = {"hallucination", "toxicity"}


def _tool_call_arguments(previous: Optional[dict]) -> Optional[str]:
    """Return the arguments of the first tool call on the message before a tool result."""
    if not previous:
        return None
    calls = previous.get("tool_calls") or previous.get("additional_kwargs", {}).get("tool_calls") or []
    if not calls:
        return None
    first = calls[0]
    return str(first.get("args") or first.get("function", {}).get("arguments"))


def format_messages(messages: list[dict]) -> str:
    """Format a list of messages for evaluation.

    Accepts LangChain-serialized messages (with "type") and OpenAI-style ones (with "role").

    Args:
        messages: List of message dictionaries.

    Returns:
        String representation of formatted messages.
    """
    formatted_messages = []
    for idx, message in enumerate(messages):
        kind = message.get("type") or message.get("role")
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        if kind == "tool":
            args = _tool_call_arguments(messages[idx - 1] if idx > 0 else None)
            snippet = f"{content[:100]}..." if len(content) > 100 else content
            formatted_messages.append(f"tool {message.get('name')} input: {args} output: {snippet}")
        elif content:
            formatted_messages.append(f"{kind}: {content}")
    return "\n".join(formatted_messages)


def get_input_output(trace: TraceWithDetails) -> Tuple[Optional[str], Optional[str]]:
    """Extract and format input and output messages from a trace.

    Args:
        trace: The trace to extract messages from.

    Returns:
        Tuple of (formatted_input, formatted_output). (None, None) if the trace has no messages.
    """
    if not isinstance(trace.output, dict):
        return None, None
    messages = trace.output.get("messages") or []
    if not messages:
        return None, None
    return format_messages(messages[:-1]), format_messages([messages[-1]])


def initialize_report(model_name: str) -> Dict[str, Any]:
    """Initialize report data structure.

    Args:
        model_name: Name of the model being evaluated.

    Returns:
        Dict containing initialized report structure.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "total_traces": 0,
        "successful_traces": 0,
        "failed_traces": 0,
        "skipped_traces": 0,
        "duration_seconds": 0,
        "metrics_summary": {},
        "successful_traces_details": [],
        "failed_traces_details": [],
    }


def initialize_metrics_summary(report: Dict[str, Any], metrics: List[Dict[str, str]]) -> None:
    """Initialize metrics summary in the report.

    Args:
        report: The report dictionary.
        metrics: List of metric definitions.
    """
    for metric in metrics:
        report["metrics_summary"][metric["name"]] = {
            "success_count": 0,
            "failure_count": 0,
            "avg_score": 0.0,
            # For hallucination and toxicity a score of 1 is the worst outcome
            "higher_is_better": metric["name"] not in LOWER_IS_BETTER,
        }


def update_success_metrics(
    report: Dict[str, Any], trace_id: str, metric_name: str, score: ScoreSchema, trace_results: Dict[str, Any]
) -> None:
    """Update metrics for a successful evaluation.

    Args:
        report: The report dictionary.
        trace_id: ID of the trace being evaluated.
        metric_name: Name of the metric.
        score: The score object.
        trace_results: Dictionary to store trace results.
    """
    trace_results[trace_id]["metrics_succeeded"] += 1
    trace_results[trace_id]["metrics_results"][metric_name] = {
        "success": True,
        "score": score.score,
        "reasoning": score.reasoning,
    }
    report["metrics_summary"][metric_name]["success_count"] += 1
    report["metrics_summary"][metric_name]["avg_score"] += score.score


def update_failure_metrics(
    report: Dict[str, Any], trace_id: str, metric_name: str, trace_results: Dict[str, Any]
) -> None:
    """Update metrics for a failed evaluation.

    Args:
        report: The report dictionary.
        trace_id: ID of the trace being evaluated.
        metric_name: Name of the metric.
        trace_results: Dictionary to store trace results.
    """
    trace_results[trace_id]["metrics_results"][metric_name] = {"success": False}
    report["metrics_summary"][metric_name]["failure_count"] += 1


def process_trace_results(
    report: Dict[str, Any], trace_id: str, trace_results: Dict[str, Any], metrics_count: int
) -> None:
    """Process results for a single trace.

    Args:
        report: The report dictionary.
        trace_id: ID of the trace being evaluated.
        trace_results: Dictionary to store trace results.
        metrics_count: Total number of metrics.
    """
    if trace_results[trace_id]["metrics_succeeded"] == metrics_count:
        trace_results[trace_id]["success"] = True
        report["successful_traces"] += 1
        report["successful_traces_details"].append(
            {"trace_id": trace_id, "metrics_results": trace_results[trace_id]["metrics_results"]}
        )
    else:
        report["failed_traces"] += 1
        report["failed_traces_details"].append(
            {
                "trace_id": trace_id,
                "metrics_evaluated": trace_results[trace_id]["metrics_evaluated"],
                "metrics_succeeded": trace_results[trace_id]["metrics_succeeded"],
                "metrics_results": trace_results[trace_id]["metrics_results"],
            }
        )


def calculate_avg_scores(report: Dict[str, Any]) -> None:
    """Calculate average scores for each metric.

    Args:
        report: The report dictionary.
    """
    for _, data in report["metrics_summary"].items():
        if data["success_count"] > 0:
            data["avg_score"] = round(data["avg_score"] / data["success_count"], 2)


def generate_report(report: Dict[str, Any]) -> str:
    """Generate a JSON report file with evaluation results.

    Args:
        report: The report dictionary.

    Returns:
        str: Path to the generated report file.
    """
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"evaluation_report_{timestamp}.json")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Add the report path to the report data for reference
    report["generate_report_path"] = report_path

    logger.info("Evaluation report generated", report_path=report_path)
    return report_path
