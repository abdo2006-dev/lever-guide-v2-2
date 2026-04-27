"""
Lightweight retrieval layer for the optional Analysis Copilot.

Indexes compact analysis artifacts per request_id. It intentionally stores
summaries and analysis outputs, not the raw dataframe.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from app.schemas import AnalysisBundle, CopilotCitation
from app.utils.preprocess import infer_column_kind


@dataclass
class Artifact:
    artifact_id: str
    title: str
    kind: str
    text: str
    metadata: dict[str, Any]


@dataclass
class Chunk:
    chunk_id: str
    artifact_id: str
    title: str
    kind: str
    text: str
    metadata: dict[str, Any]


@dataclass
class SessionIndex:
    analysis_id: str
    chunks: list[Chunk]
    matrix: Any
    created_at: float


VECTOR_SIZE = int(os.environ.get("RAG_VECTOR_SIZE", "4096"))
MAX_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "7000"))
INDEX_TTL_SECONDS = int(os.environ.get("RAG_INDEX_TTL_SECONDS", str(6 * 60 * 60)))

_vectorizer = HashingVectorizer(
    n_features=VECTOR_SIZE,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
    stop_words="english",
)
_indexes: dict[str, SessionIndex] = {}


def _compact_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _column_profile(df: pd.DataFrame, roles: dict[str, str]) -> str:
    lines = [
        f"Dataset rows in analysis: {len(df)}",
        f"Dataset columns: {len(df.columns)}",
    ]
    for col in df.columns:
        s = df[col]
        role = roles.get(col, "ignore")
        kind = infer_column_kind(s)
        base = f"- {col}: role={role}, type={kind}, missing={int(s.isna().sum())}, unique={int(s.nunique())}"
        if kind == "numeric":
            nums = pd.to_numeric(s, errors="coerce")
            if not nums.isna().all():
                base += (
                    f", mean={_compact_float(nums.mean())}, std={_compact_float(nums.std())}, "
                    f"p25={_compact_float(nums.quantile(0.25))}, p75={_compact_float(nums.quantile(0.75))}"
                )
        else:
            top = ", ".join(f"{k} ({v})" for k, v in s.value_counts().head(5).items())
            if top:
                base += f", top values={top}"
        lines.append(base)
    return "\n".join(lines)


def _build_artifacts(bundle: AnalysisBundle, df: pd.DataFrame, roles: dict[str, str]) -> list[Artifact]:
    artifacts: list[Artifact] = [
        Artifact(
            artifact_id="dataset_profile",
            title="Dataset Profile",
            kind="dataset",
            text=_column_profile(df, roles),
            metadata={"dataset_name": bundle.dataset_name, "target": bundle.target},
        ),
        Artifact(
            artifact_id="executive_summary",
            title="Executive Summary",
            kind="summary",
            text="\n".join([
                bundle.executive.headline,
                bundle.executive.sub_headline,
                "Key findings:",
                *[f"- {b}" for b in bundle.executive.bullets],
                "Cautions:",
                *[f"- {c}" for c in bundle.executive.cautions],
                f"Methodology: {bundle.executive.methodology_note}",
                f"Disclaimer: {bundle.executive.disclaimer}",
            ]),
            metadata={"target": bundle.target},
        ),
        Artifact(
            artifact_id="dag_validation",
            title="DAG Validation",
            kind="dag",
            text="\n".join([
                f"Valid DAG: {bundle.dag_validation.valid}",
                *[f"Warning: {w}" for w in bundle.dag_validation.warnings],
                *[f"Error: {e}" for e in bundle.dag_validation.errors],
            ]),
            metadata={"valid": bundle.dag_validation.valid},
        ),
    ]

    model_lines = []
    for result in bundle.predictive:
        metric = result.metrics
        model_lines.append(
            f"{result.display_name}: winner={result.is_winner}, R2={_compact_float(metric.r2)}, "
            f"adjusted_R2={_compact_float(metric.adj_r2)}, RMSE={_compact_float(metric.rmse)}, "
            f"MAE={_compact_float(metric.mae)}, train_rows={metric.n_train}, test_rows={metric.n_test}"
        )
        top_importances = ", ".join(
            f"{imp.feature} ({_compact_float(imp.importance_norm)})"
            for imp in result.importances[:10]
        )
        if top_importances:
            model_lines.append(f"Top importances for {result.display_name}: {top_importances}")
    artifacts.append(Artifact(
        artifact_id="model_metrics",
        title="Predictive Model Metrics",
        kind="model",
        text="\n".join(model_lines),
        metadata={"best_model": bundle.best_model},
    ))

    causal_lines = []
    for effect in bundle.causal:
        causal_lines.append(
            f"{effect.feature}: beta_per_sd={_compact_float(effect.effect_per_std)}, "
            f"raw_effect={_compact_float(effect.effect_raw)}, p={_compact_float(effect.p_value)}, "
            f"ci=[{_compact_float(effect.conf_int_lo)}, {_compact_float(effect.conf_int_hi)}], "
            f"evidence={effect.evidence_strength}, adjusted_for={', '.join(effect.adjusted_for) or 'none'}, "
            f"warning={effect.warning or 'none'}"
        )
    artifacts.append(Artifact(
        artifact_id="causal_findings",
        title="Causal Findings",
        kind="causal",
        text="\n".join(causal_lines) or "No causal effects were computed.",
        metadata={"target": bundle.target},
    ))

    intervention_lines = []
    for iv in bundle.interventions:
        intervention_lines.append(
            f"Rank {iv.rank} {iv.feature}: direction={iv.direction}, suggested_value={_compact_float(iv.suggested_value)}, "
            f"expected_change={_compact_float(iv.expected_kpi_change)} ({_compact_float(iv.expected_kpi_change_pct)}%), "
            f"evidence={iv.evidence_type}/{iv.evidence_strength}. Rationale: {iv.rationale} "
            f"Tradeoff: {iv.tradeoff} Caveat: {iv.caveat}"
        )
    artifacts.append(Artifact(
        artifact_id="interventions",
        title="Intervention Recommendations",
        kind="intervention",
        text="\n".join(intervention_lines) or "No interventions were generated.",
        metadata={"target": bundle.target},
    ))

    corr_lines = [
        f"{c.feature_a} vs {c.feature_b}: correlation={_compact_float(c.correlation)}"
        for c in bundle.correlations[:30]
    ]
    artifacts.append(Artifact(
        artifact_id="eda_correlations",
        title="EDA Correlations",
        kind="eda",
        text="\n".join(corr_lines) or "No numeric correlations were available.",
        metadata={"target": bundle.target},
    ))
    return artifacts


def _chunk_artifact(artifact: Artifact, chunk_chars: int = 1200, overlap: int = 160) -> list[Chunk]:
    text = " ".join(artifact.text.split())
    if len(text) <= chunk_chars:
        return [Chunk(
            chunk_id=f"{artifact.artifact_id}:0",
            artifact_id=artifact.artifact_id,
            title=artifact.title,
            kind=artifact.kind,
            text=text,
            metadata=artifact.metadata,
        )]

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(Chunk(
            chunk_id=f"{artifact.artifact_id}:{idx}",
            artifact_id=artifact.artifact_id,
            title=artifact.title,
            kind=artifact.kind,
            text=text[start:end],
            metadata=artifact.metadata,
        ))
        idx += 1
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def index_analysis_session(bundle: AnalysisBundle, df: pd.DataFrame, roles: dict[str, str]) -> None:
    _prune_old_indexes()
    artifacts = _build_artifacts(bundle, df, roles)
    chunks = [chunk for artifact in artifacts for chunk in _chunk_artifact(artifact)]
    if not chunks:
        return
    matrix = _vectorizer.transform([chunk.text for chunk in chunks])
    _indexes[bundle.request_id] = SessionIndex(
        analysis_id=bundle.request_id,
        chunks=chunks,
        matrix=matrix,
        created_at=time.time(),
    )


def retrieve(analysis_id: str, question: str, top_k: int = 5) -> list[CopilotCitation]:
    index = _indexes.get(analysis_id)
    if index is None:
        raise KeyError(analysis_id)
    if not question.strip():
        return []

    query_vector = _vectorizer.transform([question])
    scores = (index.matrix @ query_vector.T).toarray().ravel()
    if len(scores) == 0:
        return []

    top_indices = np.argsort(scores)[::-1][:top_k]
    citations: list[CopilotCitation] = []
    for i in top_indices:
        score = float(scores[i])
        if score <= 0:
            continue
        chunk = index.chunks[int(i)]
        citations.append(CopilotCitation(
            artifact_id=chunk.artifact_id,
            title=chunk.title,
            kind=chunk.kind,
            snippet=chunk.text[:700],
            score=round(score, 4),
            metadata=chunk.metadata,
        ))
    return citations


async def answer_with_groq(question: str, citations: list[CopilotCitation]) -> tuple[str, str | None, bool]:
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    api_key = os.environ.get("GROQ_API_KEY")
    api_base = os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1").rstrip("/")

    context_parts = []
    total_chars = 0
    for i, citation in enumerate(citations, 1):
        block = f"[{i}] {citation.title} ({citation.artifact_id})\n{citation.snippet}"
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(block)
        total_chars += len(block)
    context = "\n\n".join(context_parts)

    if not citations:
        return (
            "I could not find relevant analysis artifacts for that question. Try asking about model metrics, causal effects, interventions, or the executive summary.",
            model,
            False,
        )

    if not api_key:
        return (
            "Groq is not configured, so I can only show retrieved context. The most relevant artifacts are: "
            + "; ".join(f"{c.title} ({c.artifact_id})" for c in citations[:3])
            + ". Set GROQ_API_KEY to enable generated answers.",
            model,
            False,
        )

    system_prompt = (
        "You are LeverGuide's Analysis Copilot. Answer only from the provided analysis artifacts. "
        "Do not invent causal claims, data values, model results, or recommendations. "
        "If the retrieved context does not answer the question, say what is missing. "
        "Keep the answer concise and mention caveats when causal or intervention claims are involved."
    )
    user_prompt = f"Question:\n{question}\n\nRetrieved analysis artifacts:\n{context}"

    async with httpx.AsyncClient(timeout=float(os.environ.get("GROQ_TIMEOUT_SECONDS", "30"))) as client:
        response = await client.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_completion_tokens": 700,
            },
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload["choices"][0]["message"]["content"]
        return answer, model, True


def has_index(analysis_id: str) -> bool:
    return analysis_id in _indexes


def _prune_old_indexes() -> None:
    cutoff = time.time() - INDEX_TTL_SECONDS
    stale = [analysis_id for analysis_id, index in _indexes.items() if index.created_at < cutoff]
    for analysis_id in stale:
        _indexes.pop(analysis_id, None)
