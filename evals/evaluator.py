"""Evaluator for evals."""

import asyncio
import os
import sys
import time
from datetime import (
    UTC,
    datetime,
    timedelta,
)

import openai
from langfuse import Langfuse
from langfuse.api.resources.commons.types.trace_with_details import TraceWithDetails
from tqdm import tqdm

# Fix import path for app module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config.settings import settings
from src.system.logs import logger
from evals.helpers import (
    calculate_avg_scores,
    generate_report,
    get_input_output,
    initialize_metrics_summary,
    initialize_report,
    process_trace_results,
    update_failure_metrics,
    update_success_metrics,
)
from evals.metrics import metrics
from evals.schemas import ScoreSchema

PAGE_SIZE = 100
JUDGE_RETRIES = 3


class Evaluator:
    """Evaluates model outputs using predefined metrics.

    This class handles fetching traces from Langfuse, evaluating them against
    metrics, and uploading scores back to Langfuse.

    Attributes:
        client: OpenAI client for API calls.
        langfuse: Langfuse client for trace management.
    """

    def __init__(self):
        """Initialize Evaluator with OpenAI and Langfuse clients."""
        self.client = openai.AsyncOpenAI(api_key=settings.EVALUATION_API_KEY, base_url=settings.EVALUATION_BASE_URL)
        self.langfuse = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        # Bounds concurrent judge calls (EVALUATION_CONCURRENCY) instead of sleeping after every trace
        self._judge_slots = asyncio.Semaphore(max(1, settings.EVALUATION_CONCURRENCY))
        # Initialize report data structure
        self.report = initialize_report(settings.EVALUATION_LLM)
        initialize_metrics_summary(self.report, metrics)

    async def run(self, generate_report_file=True):
        """Main execution function that fetches and evaluates traces.

        Retrieves traces from Langfuse, evaluates each one against all metrics
        (several traces at a time), and uploads the scores back to Langfuse.

        Args:
            generate_report_file: Whether to generate a JSON report after evaluation. Defaults to True.
        """
        start_time = time.time()
        traces = self.__fetch_traces()  # Fetch traces from Langfuse
        self.report["total_traces"] = len(traces)

        trace_results = {}
        progress = tqdm(total=len(traces), desc="Evaluating traces")

        async def evaluate(trace: TraceWithDetails) -> None:
            await self._evaluate_trace(trace, trace_results)
            progress.update(1)

        await asyncio.gather(*(evaluate(trace) for trace in traces))
        progress.close()

        # Calculate evaluation duration and average scores
        self.report["duration_seconds"] = round(time.time() - start_time, 2)
        calculate_avg_scores(self.report)

        # Generate a report file if required
        if generate_report_file:
            generate_report(self.report)

        # Log the evaluation summary
        logger.info(
            "evaluation_completed",
            total_traces=self.report["total_traces"],
            successful_traces=self.report["successful_traces"],
            failed_traces=self.report["failed_traces"],
            skipped_traces=self.report["skipped_traces"],
            duration_seconds=self.report["duration_seconds"],
        )

    async def _evaluate_trace(self, trace: TraceWithDetails, trace_results: dict) -> None:
        """Score one trace against every metric and record the results.

        Args:
            trace: The trace to evaluate.
            trace_results: Shared per-trace results, filled in place.
        """
        trace_id = trace.id
        input, output = get_input_output(trace)  # Extract input and output from the trace
        if not input or not output:
            # Nothing to judge (no chat output on this trace): skip it rather than count it as failed
            self.report["skipped_traces"] += 1
            logger.info("trace_skipped_no_messages", trace_id=trace_id)
            return

        trace_results[trace_id] = {
            "success": False,
            "metrics_evaluated": 0,
            "metrics_succeeded": 0,
            "metrics_results": {},
        }
        scores = await asyncio.gather(*(self._run_metric_evaluation(metric, input, output) for metric in metrics))
        for metric, score in zip(metrics, scores, strict=True):
            if score:
                self._push_to_langfuse(trace, score, metric)  # Push the score to Langfuse
                update_success_metrics(self.report, trace_id, metric["name"], score, trace_results)
            else:
                update_failure_metrics(self.report, trace_id, metric["name"], trace_results)
            trace_results[trace_id]["metrics_evaluated"] += 1

        # Process results for the current trace
        process_trace_results(self.report, trace_id, trace_results, len(metrics))

    def _push_to_langfuse(self, trace: TraceWithDetails, score: ScoreSchema, metric: dict):
        """Push evaluation score to Langfuse.

        Args:
            trace: The trace to score.
            score: The evaluation score.
            metric: The metric used for evaluation.
        """
        self.langfuse.create_score(
            trace_id=trace.id,
            name=metric["name"],
            data_type="NUMERIC",
            value=score.score,
            comment=score.reasoning,
        )

    async def _run_metric_evaluation(self, metric: dict, input: str, output: str) -> ScoreSchema | None:
        """Evaluate a single trace against a specific metric.

        Args:
            metric: The metric definition to use for evaluation.
            input: The input to evaluate.
            output: The output to evaluate.

        Returns:
            ScoreSchema with evaluation results or None if evaluation failed.
        """
        metric_name = metric["name"]
        score = await self._call_openai(metric["prompt"], input, output)
        if score:
            logger.info("metric_evaluation_completed", metric=metric_name, score=score.score)
        else:
            logger.error("metric_evaluation_failed", metric=metric_name)
        return score

    async def _call_openai(self, metric_system_prompt: str, input: str, output: str) -> ScoreSchema | None:
        """Call OpenAI API to evaluate a trace.

        Args:
            metric_system_prompt: System prompt defining the evaluation metric.
            input: Formatted input messages.
            output: Formatted output message.

        Returns:
            ScoreSchema with evaluation results or None if API call failed.
        """
        for attempt in range(1, JUDGE_RETRIES + 1):
            try:
                async with self._judge_slots:
                    response = await self.client.beta.chat.completions.parse(
                        model=settings.EVALUATION_LLM,
                        messages=[
                            {"role": "system", "content": metric_system_prompt},
                            {"role": "user", "content": f"Input: {input}\nGeneration: {output}"},
                        ],
                        response_format=ScoreSchema,
                    )
                return response.choices[0].message.parsed
            except Exception as e:
                logger.error(
                    "judge_call_failed", attempt=attempt, error=str(e), sleep_time=settings.EVALUATION_SLEEP_TIME
                )
                if attempt < JUDGE_RETRIES:
                    await asyncio.sleep(settings.EVALUATION_SLEEP_TIME)  # non-blocking backoff
        return None

    def __fetch_traces(self) -> list[TraceWithDetails]:
        """Fetch unscored traces from the past 24 hours, page by page.

        Stops after EVALUATION_MAX_TRACES unscored traces.

        Returns:
            List of traces that haven't been scored yet.
        """
        since = datetime.now(UTC) - timedelta(hours=24)
        unscored: list[TraceWithDetails] = []
        page = 1
        try:
            while len(unscored) < settings.EVALUATION_MAX_TRACES:
                result = self.langfuse.api.trace.list(
                    from_timestamp=since, order_by="timestamp.asc", limit=PAGE_SIZE, page=page
                )
                unscored.extend(trace for trace in result.data if not trace.scores)
                if page >= (result.meta.total_pages or 1) or not result.data:
                    break
                page += 1
        except Exception as e:
            logger.error("fetch_traces_failed", error=str(e), page=page)
        return unscored[: settings.EVALUATION_MAX_TRACES]
