"""
Serialise a curated ontology to the JSON the frontend consumes.

Python is authoritative. `apps/web/lib/generated/demo-ontology.json` is generated
from it by `scripts/export_ontology.py`, and `tests/test_ontology.py` fails if the
committed JSON no longer matches this module — that test *is* the synchronisation
mechanism between the two languages.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from app.ontology.injection_molding import SOURCE_DEVIATIONS
from app.ontology.schema import DatasetOntology, VariableSpec

# Repo-relative path of the generated file, from the repository root.
GENERATED_JSON_PATH = "apps/web/lib/generated/demo-ontology.json"


def _variable_payload(spec: VariableSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "label": spec.label,
        "unit": spec.unit,
        "description": spec.description,
        "causal_role": spec.causal_role,
        "column_role": spec.column_role,
        "controllable": spec.controllable,
        "derived": spec.derived,
        "mediator": spec.mediator,
        "valid_range": list(spec.valid_range) if spec.valid_range else None,
        "observed_range": list(spec.observed_range) if spec.observed_range else None,
        "categories": list(spec.categories) if spec.categories else None,
        "intervention_eligibility": spec.intervention_eligibility,
        "evidence_status": spec.evidence_status,
        "role_source": spec.role_source,
        "notes": spec.notes,
    }


def ontology_payload(onto: DatasetOntology) -> dict[str, Any]:
    """The full JSON-serialisable form of an ontology."""
    return {
        "_generated": (
            "Generated from apps/api/app/ontology/ by apps/api/scripts/export_ontology.py. "
            "Do not edit by hand — edit the Python source and regenerate."
        ),
        "dataset_id": onto.dataset_id,
        "version": onto.version,
        "title": onto.title,
        "target": onto.target,
        "provenance": onto.provenance,
        "graph_assumption": onto.graph_assumption,
        "variables": [_variable_payload(v) for v in onto.variables],
        "column_roles": onto.column_roles(),
        "adjustment_sets": {k: sorted(v) for k, v in sorted(onto.adjustment_sets.items())},
        "source_deviations": dict(sorted(SOURCE_DEVIATIONS.items())),
        "edges": [{"source": s, "target": t} for s, t in onto.edges],
        "derived_relationships": [asdict(r) for r in onto.derived_relationships],
    }


def ontology_json(onto: DatasetOntology) -> str:
    """Canonical serialisation — stable ordering, trailing newline."""
    return json.dumps(ontology_payload(onto), indent=2, ensure_ascii=True) + "\n"
