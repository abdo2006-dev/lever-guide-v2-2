"""Declared dataset ontologies — the single source of truth for curated demos."""
from app.ontology.export import GENERATED_JSON_PATH, ontology_json, ontology_payload
from app.ontology.injection_molding import ONTOLOGY as INJECTION_MOULDING_ONTOLOGY
from app.ontology.injection_molding import ONTOLOGY_VERSION, SOURCE_DEVIATIONS
from app.ontology.registry import (
    ONTOLOGIES,
    adjustment_set_for,
    build_graph,
    get_ontology,
    resolve_ontology,
    validate_ontology,
)
from app.ontology.schema import (
    CAUSAL_ROLE_TO_COLUMN_ROLE,
    DatasetOntology,
    DerivedRelationship,
    VariableSpec,
)

__all__ = [
    "CAUSAL_ROLE_TO_COLUMN_ROLE",
    "GENERATED_JSON_PATH",
    "INJECTION_MOULDING_ONTOLOGY",
    "ONTOLOGIES",
    "ONTOLOGY_VERSION",
    "SOURCE_DEVIATIONS",
    "DatasetOntology",
    "DerivedRelationship",
    "VariableSpec",
    "adjustment_set_for",
    "build_graph",
    "get_ontology",
    "ontology_json",
    "ontology_payload",
    "resolve_ontology",
    "validate_ontology",
]
