"""
Qdrant-backed retrieval layer for the optional Analysis Copilot.

The corpus stores compact analysis artifacts per request_id. It intentionally
indexes summaries and analysis outputs, not the raw dataframe.
"""
from __future__ import annotations

import atexit
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
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


VECTOR_SIZE = int(os.environ.get("RAG_VECTOR_SIZE", "4096"))
MAX_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "7000"))
INDEX_TTL_SECONDS = int(os.environ.get("RAG_INDEX_TTL_SECONDS", str(6 * 60 * 60)))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "analysis_copilot")

_vectorizer = HashingVectorizer(
    n_features=VECTOR_SIZE,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
    stop_words="english",
)
_qdrant_client: QdrantClient | None = None
_qdrant_config: tuple[Any, ...] | None = None


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


def _vectorize_texts(texts: list[str]) -> np.ndarray:
    return _vectorizer.transform(texts).toarray().astype(np.float32)


def _qdrant_runtime_config() -> tuple[Any, ...]:
    return (
        os.environ.get("QDRANT_URL", "").strip(),
        os.environ.get("QDRANT_API_KEY", "").strip(),
        os.environ.get("QDRANT_PATH", "./.qdrant").strip(),
        os.environ.get("QDRANT_TIMEOUT_SECONDS", "10").strip(),
        QDRANT_COLLECTION,
    )


def _ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(QDRANT_COLLECTION):
        return
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client, _qdrant_config

    config = _qdrant_runtime_config()
    if _qdrant_client is not None and _qdrant_config == config:
        return _qdrant_client

    close_retrieval_store()

    qdrant_url, qdrant_api_key, qdrant_path, timeout_raw, _ = config
    timeout = int(timeout_raw)
    if qdrant_url:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key or None,
            timeout=timeout,
        )
    else:
        resolved_path = os.path.abspath(qdrant_path)
        os.makedirs(resolved_path, exist_ok=True)
        client = QdrantClient(
            path=resolved_path,
            timeout=timeout,
            force_disable_check_same_thread=True,
        )

    _ensure_collection(client)
    _qdrant_client = client
    _qdrant_config = config
    return client


def _analysis_filter(analysis_id: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="analysis_id",
                match=models.MatchValue(value=analysis_id),
            )
        ]
    )


def _point_id(analysis_id: str, chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{analysis_id}:{chunk_id}"))


def index_analysis_session(bundle: AnalysisBundle, df: pd.DataFrame, roles: dict[str, str]) -> None:
    _prune_old_indexes()
    artifacts = _build_artifacts(bundle, df, roles)
    chunks = [chunk for artifact in artifacts for chunk in _chunk_artifact(artifact)]
    if not chunks:
        return

    client = _get_qdrant_client()
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=_analysis_filter(bundle.request_id),
        wait=True,
    )

    created_at = time.time()
    vectors = _vectorize_texts([chunk.text for chunk in chunks])
    points = [
        models.PointStruct(
            id=_point_id(bundle.request_id, chunk.chunk_id),
            vector=vectors[idx].tolist(),
            payload={
                "analysis_id": bundle.request_id,
                "artifact_id": chunk.artifact_id,
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "kind": chunk.kind,
                "text": chunk.text,
                "created_at": created_at,
                "metadata": chunk.metadata,
            },
        )
        for idx, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)


def retrieve(analysis_id: str, question: str, top_k: int = 5) -> list[CopilotCitation]:
    if not question.strip():
        return []
    if not has_index(analysis_id):
        raise KeyError(analysis_id)

    client = _get_qdrant_client()
    query_vector = _vectorize_texts([question])[0].tolist()
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=_analysis_filter(analysis_id),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    citations: list[CopilotCitation] = []
    for point in response.points:
        payload = point.payload or {}
        citations.append(CopilotCitation(
            artifact_id=str(payload.get("artifact_id", "unknown")),
            title=str(payload.get("title", "Unknown Artifact")),
            kind=str(payload.get("kind", "summary")),
            snippet=str(payload.get("text", ""))[:700],
            score=round(float(point.score or 0.0), 4),
            metadata=dict(payload.get("metadata") or {}),
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
    client = _get_qdrant_client()
    result = client.count(
        collection_name=QDRANT_COLLECTION,
        count_filter=_analysis_filter(analysis_id),
        exact=False,
    )
    return int(result.count) > 0


def close_retrieval_store() -> None:
    global _qdrant_client, _qdrant_config
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
        except Exception:
            pass
    _qdrant_client = None
    _qdrant_config = None


def _prune_old_indexes() -> None:
    if INDEX_TTL_SECONDS <= 0:
        return
    cutoff = time.time() - INDEX_TTL_SECONDS
    client = _get_qdrant_client()
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key="created_at",
                    range=models.Range(lte=cutoff),
                )
            ]
        ),
        wait=True,
    )


atexit.register(close_retrieval_store)
