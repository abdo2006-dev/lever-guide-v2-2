"""
Dataset ontology types.

An ontology is a *declared* description of a curated dataset: what each variable
means, what causal role it plays, whether it can be intervened on, and which
variables must be adjusted for when estimating a lever's total effect.

Nothing here is learned from data. An ontology is a domain assumption that the
application is transparent about, not a discovered structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# ── Vocabulary ────────────────────────────────────────────────────────────────

# Rich causal vocabulary. This is finer-grained than the app's ColumnRole
# because the distinctions matter scientifically even when the app collapses
# them (e.g. an operator covariate and a fixed part property are both adjusted
# for, but only one of them is a design property).
CausalRole = Literal[
    "process_lever",           # real-time setpoint an operator can change
    "planning_lever",          # scheduling decision, not a per-interval setpoint
    "confounder",              # causes both levers and the outcome
    "mediator",                # transmits a lever's effect; never a total-effect adjuster
    "context",                 # condition on, never intervene
    "operator_covariate",      # adjusted for, never a target
    "batch_covariate",         # material batch property, adjusted for
    "identifier",              # key / timestamp
    "outcome",                 # the KPI
    "post_treatment_outcome",  # downstream of the levers; predictor use is leakage
]

# Whether — and how honestly — this variable can carry an intervention today.
InterventionEligibility = Literal[
    "eligible",               # can be simulated and ranked
    "derived_constrained",    # determined by other columns; cannot be moved independently
    "mediated_unsupported",   # acts only through a mediator whose propagation is not implemented
    "preliminary",            # representable, but the mechanism is only partly modelled
    "not_eligible",           # not an intervention target at all
]

# How a variable's causal role was decided when sources disagree.
RoleSource = Literal["paper_taxonomy", "challenge_ontology", "measured", "derived_from_both"]


# The app's coarse ColumnRole vocabulary (mirrors app.schemas.ColumnRole).
# The mapping below is the single documented bridge between the two.
CAUSAL_ROLE_TO_COLUMN_ROLE: dict[str, str] = {
    "process_lever": "controllable",
    "planning_lever": "planning_lever",
    "confounder": "confounder",
    "mediator": "mediator",
    "context": "context",
    "operator_covariate": "context",
    "batch_covariate": "confounder",
    "identifier": "identifier",
    "outcome": "outcome",
    # Post-treatment outcomes are excluded from the analysis entirely: using one
    # as a predictor is target leakage (e.g. cycle_time_s mechanically contains
    # cooling_time_s).
    "post_treatment_outcome": "ignore",
}


@dataclass(frozen=True)
class VariableSpec:
    """One variable in a curated dataset."""

    name: str
    label: str
    causal_role: CausalRole
    unit: Optional[str] = None
    description: str = ""

    # Can an operator or planner set this value directly?
    controllable: bool = False
    # Is this value determined by other columns rather than set independently?
    derived: bool = False
    # Does this variable sit on a causal path between a lever and the outcome?
    mediator: bool = False

    # Declared physical / process bounds. Ontology or engineering limits, not data.
    valid_range: Optional[tuple[float, float]] = None
    # Range actually seen in the shipped dataset. Extrapolating past this is
    # not the same kind of claim as interpolating inside it.
    observed_range: Optional[tuple[float, float]] = None
    # For categoricals.
    categories: Optional[tuple[str, ...]] = None

    intervention_eligibility: InterventionEligibility = "not_eligible"
    # Set when the evidence for this variable is known to be specification-dependent.
    evidence_status: Optional[Literal["conflicting", "inconclusive"]] = None
    role_source: RoleSource = "paper_taxonomy"
    notes: str = ""

    @property
    def column_role(self) -> str:
        """The coarse role the application's schemas use."""
        return CAUSAL_ROLE_TO_COLUMN_ROLE[self.causal_role]


@dataclass(frozen=True)
class DerivedRelationship:
    """
    A documented identity that constrains one column given others.

    `target` is approximately `product(inputs) * k` with `k` inside
    `[ratio_lo, ratio_hi]`. A counterfactual that moves `target` without moving
    `inputs` is physically impossible whenever the implied `k` leaves that band.
    """

    target: str
    inputs: tuple[str, ...]
    ratio_lo: float
    ratio_hi: float
    description: str
    # Share of rows allowed to violate the band before the whole intervention is
    # rejected rather than merely warned about.
    max_violation_share: float = 0.05


@dataclass(frozen=True)
class DatasetOntology:
    """A declared description of one curated dataset."""

    dataset_id: str
    version: str
    title: str
    target: str
    # Human-readable statement of where the roles and adjustment sets come from.
    provenance: str
    # Why the causal graph below is an assumption rather than a finding.
    graph_assumption: str

    variables: tuple[VariableSpec, ...]
    # Per-lever total-effect adjustment sets, declared from the domain graph.
    adjustment_sets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Declared causal edges. Used to check the adjustment sets and to replace the
    # role-template graph, which cannot express lever -> lever structure.
    edges: tuple[tuple[str, str], ...] = ()
    derived_relationships: tuple[DerivedRelationship, ...] = ()

    # ── lookups ───────────────────────────────────────────────────────────────

    def spec(self, name: str) -> Optional[VariableSpec]:
        return self._by_name.get(name)

    @property
    def _by_name(self) -> dict[str, VariableSpec]:
        return {v.name: v for v in self.variables}

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    def column_roles(self) -> dict[str, str]:
        """Coarse ColumnRole per column — what the API and UI consume."""
        return {v.name: v.column_role for v in self.variables}

    def by_causal_role(self, *roles: str) -> list[str]:
        return [v.name for v in self.variables if v.causal_role in roles]

    @property
    def mediators(self) -> list[str]:
        return [v.name for v in self.variables if v.mediator]

    def derived_for(self, name: str) -> Optional[DerivedRelationship]:
        for rel in self.derived_relationships:
            if rel.target == name:
                return rel
        return None
