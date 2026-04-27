"""
DAG utilities: parsing, validation, adjustment-set reasoning.
Uses NetworkX for graph operations.
"""
from __future__ import annotations
from typing import Optional
import networkx as nx
from app.schemas import DagEdge, DagValidationResult


def build_dag(edges: list[DagEdge]) -> nx.DiGraph:
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e.source, e.target)
    return G


def validate_dag(
    edges: list[DagEdge],
    columns: list[str],
    target: str,
    controllable: list[str],
) -> DagValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    G = build_dag(edges)
    known = set(columns)

    if target not in known:
        errors.append(f"Target column '{target}' is not present in the dataset.")

    for cause in controllable:
        if cause not in known:
            errors.append(f"Controllable column '{cause}' is not present in the dataset.")

    # Unknown columns and malformed/self-loop edges
    for i, e in enumerate(edges):
        if e.source == e.target:
            errors.append(f"DAG edge {i} is malformed: source and target are both '{e.source}'.")
        for node in (e.source, e.target):
            if node not in known:
                errors.append(f"DAG references unknown column: '{node}'")

    # Cycle check
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            cycle_strs = [" → ".join(c + [c[0]]) for c in cycles[:3]]
            errors.append(f"DAG contains cycles: {'; '.join(cycle_strs)}")
    except Exception as exc:
        errors.append(f"DAG could not be validated: {exc}")

    # Target must not be a source of other nodes (it's the outcome)
    if target in G:
        out_edges = list(G.out_edges(target))
        if out_edges:
            warnings.append(
                f"Target '{target}' has outgoing edges — it should be a terminal node."
            )

    return DagValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
    )


def parents_of(node: str, G: nx.DiGraph) -> set[str]:
    if not G.has_node(node):
        return set()
    return set(G.predecessors(node))


def ancestors_of(node: str, G: nx.DiGraph) -> set[str]:
    if not G.has_node(node):
        return set()
    try:
        return nx.ancestors(G, node)
    except Exception:
        return set()


def descendants_of(node: str, G: nx.DiGraph) -> set[str]:
    if not G.has_node(node):
        return set()
    try:
        return nx.descendants(G, node)
    except Exception:
        return set()


def adjustment_set(
    cause: str,
    outcome: str,
    G: nx.DiGraph,
    confounders: list[str],
    mediators: list[str],
    context: list[str],
) -> set[str]:
    """
    Build the adjustment set for estimating cause → outcome effect.
    Strategy (pragmatic back-door approximation):
      - Include: observed confounders + DAG parents of cause + context variables
      - Exclude: mediators (would block the causal path)
      - Exclude: descendants of cause (collider / post-treatment bias)
      - Exclude: outcome itself
      - Exclude: cause itself
    """
    if not G.has_node(cause):
        return {c for c in [*confounders, *context] if c not in {cause, outcome, *mediators}}

    desc = descendants_of(cause, G)
    dag_parents = parents_of(cause, G)

    adj: set[str] = set()
    adj.update(confounders)
    adj.update(dag_parents)
    adj.update(context)

    # Remove what must not be adjusted for
    for m in mediators:
        adj.discard(m)
    for d in desc:
        adj.discard(d)
    adj.discard(outcome)
    adj.discard(cause)

    return adj


def auto_dag(
    controllable: list[str],
    confounders: list[str],
    context: list[str],
    target: str,
) -> list[DagEdge]:
    """
    Generate a sensible default DAG when the user hasn't specified one:
      confounders → controllables (confounders cause process variables)
      controllables → target
      context → target
      confounders → target
    """
    edges: list[DagEdge] = []
    for cf in confounders:
        for c in controllable:
            edges.append(DagEdge(source=cf, target=c))
        edges.append(DagEdge(source=cf, target=target))
    for c in controllable:
        edges.append(DagEdge(source=c, target=target))
    for cx in context:
        edges.append(DagEdge(source=cx, target=target))
    return edges
