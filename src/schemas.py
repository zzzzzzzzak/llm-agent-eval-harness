"""Typed data contracts for evaluation tasks, agent traces, scores, and regression gates."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Dict, List, Literal, Optional


TaskType = Literal["code_synthesis", "tool_trajectory", "grounded_qa", "structured_json"]


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


@dataclass
class EvalTask:
    task_id: str
    category: TaskType
    prompt: str
    reference: Dict[str, Any]
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EvalTask":
        return cls(
            task_id=raw["task_id"],
            category=raw["category"],
            prompt=raw["prompt"],
            reference=raw["reference"],
            context=raw.get("context"),
            metadata=raw.get("metadata", {}),
        )


@dataclass
class AgentResponse:
    task_id: str
    output_text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None


@dataclass
class TaskEvalResult:
    task_id: str
    category: TaskType
    passed: bool
    score: float
    details: Dict[str, Any]
    latency_ms: float
    total_tokens: int


@dataclass
class RegressionGateSpec:
    min_overall_score: float = 0.75
    max_score_regression: float = 0.02
    significance_alpha: float = 0.05
    max_p95_latency_ms: float = 1500.0
    min_category_scores: Dict[str, float] = field(default_factory=dict)


def compute_dataset_fingerprint(tasks: List[EvalTask]) -> str:
    """Compute deterministic SHA-256 hash over task definitions for run reproducibility."""
    canonical = json.dumps([asdict(t) for t in tasks], sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
