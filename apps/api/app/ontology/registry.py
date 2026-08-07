"""
Ontology lookup and consistency checking.

`resolve_ontology` matches an uploaded dataset against the curated ontologies we
ship. It matches conservatively: a dataset that merely looks similar gets no
ontology, and therefore no domain claims.
"""
from __future__ import annotations

from typing import Optional

import networkx as nx

from app.ontology.injection_molding import ONTOLOGY as INJECTION_MOULDING
from app.ontology.schema import DatasetOntology

ONTOLOGIES: tuple[DatasetOntology, ...] = (INJECTION_MOULDING,)

# A curated ontology only applies when nearly all of its variables are present.
# Below this the dataset is a different dataset and gets the generic path.
MATCH_THRESHOLD = 0.9


def resolve_ontology(columns: list[str], target: str) -> Optional[DatasetOntology]:
    """Return the curated ontology for this dataset, or None."""
    present = set(columns)
    for onto in ONTOLOGIES:
        if onto.target != target:
            continue
        declared = set(onto.names)
        overlap = len(declared & present) / len(declared)
        if overlap >= MATCH_THRESHOLD:
            return onto
    return None


def get_ontology(dataset_id: str) -> Optional[DatasetOntology]:
    for onto in ONTOLOGIES:
        if onto.dataset_id == dataset_id:
            return onto
    return None


# ── Consistency checking ──────────────────────────────────────────────────────

def build_graph(onto: DatasetOntology) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(onto.names)
    G.add_edges_from(onto.edges)
    return G


def validate_ontology(onto: DatasetOntology) -> list[str]:
    """
    Check the declared graph and adjustment sets against each other.

    Returns a list of problems; empty means consistent. These are invariants of
    the *declaration*, not of any dataset, so they are checked in the test suite
    rather than at request time.
    """
    problems: list[str] = []
    known = set(onto.names)
    G = build_graph(onto)

    # Graph well-formedness.
    for src, dst in onto.edges:
        if src not in known:
            problems.append(f"edge source '{src}' is not a declared variable")
        if dst not in known:
            problems.append(f"edge target '{dst}' is not a declared variable")
    if not nx.is_directed_acyclic_graph(G):
        cycles = [" -> ".join(c + [c[0]]) for c in list(nx.simple_cycles(G))[:3]]
        problems.append(f"declared graph contains cycles: {'; '.join(cycles)}")

    # The outcome must be terminal.
    if onto.target in G and list(G.successors(onto.target)):
        problems.append(f"outcome '{onto.target}' has outgoing edges")

    mediators = set(onto.mediators)

    for cause, adjusters in onto.adjustment_sets.items():
        spec = onto.spec(cause)
        if spec is None:
            problems.append(f"adjustment set declared for unknown variable '{cause}'")
            continue
        if not spec.controllable:
            problems.append(
                f"adjustment set declared for '{cause}', which is not controllable"
            )
        if len(set(adjusters)) != len(adjusters):
            problems.append(f"adjustment set for '{cause}' has duplicate entries")

        descendants = nx.descendants(G, cause) if cause in G else set()
        for adj in adjusters:
            if adj not in known:
                problems.append(f"adjustment set for '{cause}' names unknown '{adj}'")
            if adj == cause:
                problems.append(f"adjustment set for '{cause}' contains the cause itself")
            if adj == onto.target:
                problems.append(f"adjustment set for '{cause}' contains the outcome")
            if adj in mediators:
                problems.append(
                    f"adjustment set for '{cause}' contains mediator '{adj}' — "
                    "conditioning on it would turn a total effect into a direct effect"
                )
            if adj in descendants:
                problems.append(
                    f"adjustment set for '{cause}' contains '{adj}', a descendant of "
                    "the cause — post-treatment conditioning"
                )

    for rel in onto.derived_relationships:
        for name in (rel.target, *rel.inputs):
            if name not in known:
                problems.append(f"derived relationship names unknown variable '{name}'")
        if rel.ratio_lo >= rel.ratio_hi:
            problems.append(f"derived relationship for '{rel.target}' has an empty ratio band")

    return problems


def adjustment_set_for(onto: DatasetOntology, cause: str) -> Optional[list[str]]:
    """The declared total-effect adjustment set for `cause`, if one exists."""
    declared = onto.adjustment_sets.get(cause)
    return sorted(declared) if declared is not None else None
