"""
Curated ontology for the injection-moulding demo dataset.

This module is the **single source of truth** for the demo's causal roles,
adjustment sets, physical bounds and derived-variable constraints. The frontend
copy (`apps/web/lib/generated/demo-ontology.json`) is generated from this file;
see `app/ontology/export.py` and `tests/test_ontology.py::test_generated_json_is_in_sync`.

Provenance
----------
* Causal roles and per-lever adjustment sets: `datathon-CUB-2026/src/utils.py:23-113`,
  which states that the roles match §2.1 of the source paper.
* Labels, units and declared ranges: `datathon-CUB-2026/data/injection_molding_ontology.json`.
* Physical lever bounds: `datathon-CUB-2026/src/intervention_helpers.py:32-43`.
* Observed ranges: measured on the 5,000-row demo CSV shipped with this app.

Where sources disagree, the paper taxonomy in `src/utils.py` wins and the
disagreement is recorded in the variable's `notes` rather than resolved silently.
The challenge ontology JSON's coarse `role` field is a labelling convenience, not
an identification argument; it calls `dryer_dewpoint_c` a mediator and
`maintenance_days_since_last` a confounder, and the paper taxonomy calls both
levers.

Deviations from the source adjustment sets are listed in `SOURCE_DEVIATIONS` and
surfaced on the affected estimates. They are not silent.
"""
from __future__ import annotations

from app.ontology.schema import (
    DatasetOntology,
    DerivedRelationship,
    VariableSpec,
)

ONTOLOGY_VERSION = "injection-molding-2026.1"
DATASET_ID = "injection_molding_demo"
TARGET = "scrap_rate_pct"

_AMBIENT = ("ambient_humidity_pct", "ambient_temperature_c")


# ── Variables ─────────────────────────────────────────────────────────────────

VARIABLES: tuple[VariableSpec, ...] = (
    # ── Outcome ──────────────────────────────────────────────────────────────
    VariableSpec(
        name="scrap_rate_pct",
        label="Scrap Rate",
        unit="percent",
        causal_role="outcome",
        description="Share of parts scrapped in the 30-minute interval.",
        valid_range=(0.0, 10.0),
        observed_range=(0.931, 8.463),
        notes="Pass/fail threshold in the source study is 3.2 %; 78 % of intervals fail it.",
    ),
    # ── Process levers ───────────────────────────────────────────────────────
    VariableSpec(
        name="cooling_time_s",
        label="Cooling Time",
        unit="seconds",
        causal_role="process_lever",
        description="In-mould cooling time per cycle.",
        controllable=True,
        valid_range=(5.0, 40.0),
        observed_range=(7.19, 27.29),
        intervention_eligibility="eligible",
        notes=(
            "The dataset's headline lever. Raw correlation with scrap is positive "
            "because operators extend cooling in response to observed mould "
            "temperature; the adjusted estimate reverses sign. mold_temperature_c "
            "is therefore a required adjuster, not an optional one."
        ),
    ),
    VariableSpec(
        name="mold_temperature_c",
        label="Mould Temperature",
        unit="degC",
        causal_role="process_lever",
        description="Mould surface temperature setpoint.",
        controllable=True,
        valid_range=(40.0, 110.0),
        observed_range=(35.0, 95.0),
        intervention_eligibility="eligible",
        notes=(
            "Also the confounder that opens the back-door path for cooling time. "
            "Observed minimum (35 degC) sits below the declared process floor "
            "(40 degC)."
        ),
    ),
    VariableSpec(
        name="barrel_temperature_c",
        label="Barrel Temperature",
        unit="degC",
        causal_role="process_lever",
        description="Melt temperature setpoint in the barrel.",
        controllable=True,
        valid_range=(180.0, 310.0),
        observed_range=(196.58, 281.16),
        intervention_eligibility="eligible",
    ),
    VariableSpec(
        name="injection_pressure_bar",
        label="Injection Pressure",
        unit="bar",
        causal_role="process_lever",
        description="Peak pressure during the filling phase.",
        controllable=True,
        valid_range=(600.0, 1800.0),
        observed_range=(689.34, 1665.77),
        intervention_eligibility="eligible",
        notes=(
            "The source study conditions its pressure recommendation on "
            "tool_wear_index >= 0.45. Conditional interventions are not "
            "representable in this API yet, so the unconditional estimate is "
            "shown and the condition is not applied."
        ),
    ),
    VariableSpec(
        name="hold_pressure_bar",
        label="Hold Pressure",
        unit="bar",
        causal_role="process_lever",
        description="Packing pressure applied after the filling phase.",
        controllable=True,
        valid_range=(300.0, 1200.0),
        observed_range=(390.85, 1045.92),
        intervention_eligibility="eligible",
        evidence_status="conflicting",
        notes=(
            "Evidence is specification-dependent and is not settled. The source "
            "analysis estimates a small negative coefficient whose 95 % interval "
            "crosses zero and excludes the lever from its action package on those "
            "grounds; this application's earlier specification produced a "
            "positive, nominally significant coefficient. Changing the adjustment "
            "set changes the conclusion, so no direction is asserted here. "
            "Resolving it requires the estimator work deferred to a later phase."
        ),
    ),
    VariableSpec(
        name="screw_speed_rpm",
        label="Screw Speed",
        unit="rpm",
        causal_role="process_lever",
        description="Screw rotation speed during plasticising.",
        controllable=True,
        valid_range=(20.0, 150.0),
        observed_range=(37.4, 108.15),
        intervention_eligibility="eligible",
    ),
    VariableSpec(
        name="dryer_dewpoint_c",
        label="Dryer Dew Point",
        unit="degC",
        causal_role="process_lever",
        description="Dew point of the resin dryer's air circuit.",
        controllable=True,
        valid_range=(-50.0, -20.0),
        observed_range=(-48.0, -24.63),
        intervention_eligibility="mediated_unsupported",
        notes=(
            "A lever in the source taxonomy, and the entry point to the moisture "
            "pathway. Its effect reaches scrap only through resin_moisture_pct, so "
            "a what-if that moves the dew point while holding measured moisture "
            "fixed understates it roughly threefold. Mediator propagation is not "
            "implemented in this phase, so simulations for this lever are marked "
            "unsupported rather than reported as intervention estimates. The "
            "adjusted total-effect estimate is unaffected and is shown. "
            "The challenge ontology JSON labels this variable a mediator; the "
            "paper taxonomy labels it a controllable lever, and the paper "
            "taxonomy is used here."
        ),
    ),
    VariableSpec(
        name="shot_size_g",
        label="Shot Size",
        unit="grams",
        causal_role="process_lever",
        description="Mass of material injected per cycle.",
        controllable=True,
        derived=True,
        valid_range=(50.0, 2000.0),
        observed_range=(24.21, 1734.84),
        intervention_eligibility="derived_constrained",
        notes=(
            "Nominally a setpoint, but mechanically determined by the tooling: "
            "shot size must cover cavity_count x part_weight_g plus runner and "
            "cushion. Measured ratio to that product is 0.972-1.214 across the "
            "demo (correlation 0.9989). It therefore cannot be moved "
            "independently while cavity count and part weight are held fixed, and "
            "any simulation that does so is a short shot for most rows."
        ),
    ),
    # ── Planning lever ───────────────────────────────────────────────────────
    VariableSpec(
        name="maintenance_days_since_last",
        label="Days Since Last Maintenance",
        unit="days",
        causal_role="planning_lever",
        description="Interval since the machine's last maintenance event.",
        controllable=True,
        valid_range=(1.0, 60.0),
        observed_range=(0.0, 57.0),
        intervention_eligibility="preliminary",
        notes=(
            "A scheduling decision, not a per-interval setpoint: it is set by "
            "maintenance policy over days, not by an operator at the machine. Its "
            "effect runs through calibration_drift_index, whose propagation is not "
            "implemented in this phase, so simulations are preliminary. The "
            "adjusted total-effect estimate is shown. The challenge ontology JSON "
            "labels this a confounder; the paper taxonomy labels it a planning "
            "lever."
        ),
    ),
    # ── Mediators ────────────────────────────────────────────────────────────
    VariableSpec(
        name="resin_moisture_pct",
        label="Resin Moisture",
        unit="percent",
        causal_role="mediator",
        description="Moisture content of the resin entering the barrel.",
        mediator=True,
        valid_range=(0.005, 0.3),
        observed_range=(0.005, 0.3),
        notes=(
            "Carries the dryer dew-point effect to scrap via splay defects. "
            "Conditioning on it turns a total effect into a direct effect, so it "
            "must never appear in a total-effect adjustment set. This application "
            "previously labelled it a confounder and adjusted for it everywhere."
        ),
    ),
    VariableSpec(
        name="calibration_drift_index",
        label="Calibration Drift Index",
        unit="index",
        causal_role="mediator",
        description="Accumulated drift of the machine's calibration.",
        mediator=True,
        valid_range=(0.0, 1.0),
        observed_range=(0.0, 0.772),
        notes="Carries the maintenance-interval effect to scrap.",
    ),
    VariableSpec(
        name="tool_wear_index",
        label="Tool Wear Index",
        unit="index",
        causal_role="mediator",
        description="Accumulated wear state of the mould/tool.",
        mediator=True,
        valid_range=(0.0, 1.0),
        observed_range=(0.0, 0.731),
        notes=(
            "Mediator in the paper taxonomy, and the conditioning variable for the "
            "source study's pressure rule. The source nonetheless adjusts for it "
            "when estimating pressure effects, treating it there as a prior wear "
            "state; see SOURCE_DEVIATIONS."
        ),
    ),
    # ── Confounders ──────────────────────────────────────────────────────────
    VariableSpec(
        name="ambient_humidity_pct",
        label="Ambient Relative Humidity",
        unit="percent",
        causal_role="confounder",
        description="Shop-floor relative humidity during the interval.",
        valid_range=(30.0, 88.0),
        observed_range=(39.06, 88.0),
        notes="Environmental, not a lever: it is not controllable at interval granularity.",
    ),
    VariableSpec(
        name="ambient_temperature_c",
        label="Ambient Temperature",
        unit="degC",
        causal_role="confounder",
        description="Shop-floor temperature during the interval.",
        valid_range=(17.0, 34.0),
        observed_range=(17.0, 34.0),
    ),
    VariableSpec(
        name="operator_shift",
        label="Operator Shift",
        causal_role="confounder",
        description="Which shift was running the machine.",
        categories=("A_Day", "B_Evening", "C_Night"),
        notes=(
            "A named confounder in the source study, absorbed there as a fixed "
            "effect. This application encodes it ordinally, which is a known "
            "limitation deferred to a later phase."
        ),
    ),
    VariableSpec(
        name="resin_batch_quality_index",
        label="Resin Batch Quality Index",
        unit="index",
        causal_role="batch_covariate",
        description="Quality index of the resin lot in use.",
        valid_range=(0.75, 1.05),
        observed_range=(0.835, 1.05),
    ),
    # ── Context ──────────────────────────────────────────────────────────────
    VariableSpec(
        name="cavity_count",
        label="Cavity Count",
        unit="count",
        causal_role="context",
        description="Number of cavities in the mould.",
        valid_range=(1.0, 8.0),
        observed_range=(1.0, 8.0),
        notes="A property of the mould. Changing it means changing tooling, not a setpoint.",
    ),
    VariableSpec(
        name="part_weight_g",
        label="Part Weight",
        unit="grams",
        causal_role="context",
        description="Nominal weight of one moulded part.",
        valid_range=(15.0, 230.0),
        observed_range=(22.75, 192.02),
        notes=(
            "A fixed design property of the part, and an adjuster in three of the "
            "source study's adjustment sets. It cannot be a mediator of a process "
            "setpoint, which is how this application previously classified it — "
            "with the effect that it was dropped from every adjustment set."
        ),
    ),
    VariableSpec(
        name="clamp_force_kn",
        label="Clamp Force",
        unit="kN",
        causal_role="context",
        description="Clamping force applied to hold the mould closed.",
        valid_range=(500.0, 4400.0),
        observed_range=(1698.71, 4400.0),
        notes=(
            "Explicitly not a controllable lever in the source study: clamp force "
            "follows from tonnage requirement, which follows from mould and part "
            "geometry. This application previously offered it as a lever and "
            "ranked it. The source also clips it at 4400 kN as sensor noise; the "
            "observed maximum here is exactly 4400."
        ),
    ),
    VariableSpec(
        name="product_variant",
        label="Product Variant",
        causal_role="context",
        description="Part family being produced.",
        categories=(
            "V_CONNECTOR_F", "V_COVER_H", "V_FILTER_FRAME_G", "V_HANDLE_E",
            "V_HOUSING_A", "V_HOUSING_B", "V_NOZZLE_C", "V_PUMP_CAP_D",
        ),
    ),
    VariableSpec(
        name="operator_experience_level",
        label="Operator Experience Level",
        unit="1-7 scale",
        causal_role="operator_covariate",
        description="Experience band of the operator on shift.",
        valid_range=(1.0, 7.0),
        observed_range=(1.0, 7.0),
        notes=(
            "Adjusted for, never a target. The source study warns that its "
            "positive coefficient reflects assignment bias — experienced operators "
            "are put on harder jobs — and explicitly does not recommend "
            "reassignment."
        ),
    ),
    # ── Identifiers ──────────────────────────────────────────────────────────
    VariableSpec(
        name="timestamp", label="Timestamp", unit="ISO-8601",
        causal_role="identifier",
        description="Start of the 30-minute production interval.",
    ),
    VariableSpec(
        name="plant_id", label="Plant", causal_role="identifier",
        description="Production site.",
        categories=("DE_OBERSONT", "DE_WINNENDEN", "RO_CURTEA", "VN_QUANGNAM"),
    ),
    VariableSpec(
        name="machine_id", label="Injection Moulding Machine", causal_role="identifier",
        description="Machine identifier.",
        notes=(
            "Absorbed as a fixed effect in the source study. This application "
            "drops it; mean scrap ranges 3.82-5.53 % between machines, so the "
            "panel structure is currently unmodelled. Deferred to a later phase."
        ),
    ),
    VariableSpec(
        name="mold_id", label="Mould", causal_role="identifier",
        description="Mould/tool identifier.",
    ),
    VariableSpec(
        name="resin_lot_id", label="Resin Lot", causal_role="identifier",
        description="Resin lot identifier.",
    ),
    # ── Post-treatment outcomes (never predictors) ───────────────────────────
    VariableSpec(
        name="cycle_time_s",
        label="Cycle Time",
        unit="seconds",
        causal_role="post_treatment_outcome",
        description="Total cycle time for the interval.",
        valid_range=(16.0, 95.0),
        observed_range=(16.0, 95.0),
        notes=(
            "Mechanically subsumes cooling_time_s, so using it as a predictor is "
            "target leakage for any cooling-time question. It is also the "
            "denominator of the study's throughput trade-off. This application "
            "previously labelled it a mediator, which happened to exclude it from "
            "the feature matrix — but only incidentally."
        ),
    ),
    VariableSpec(
        name="scrap_count", label="Scrap Count", unit="count",
        causal_role="post_treatment_outcome",
        description="Number of scrapped parts in the interval.",
        valid_range=(0.0, 100.0), observed_range=(0.0, 33.0),
        notes="Numerator of the target. Using it as a predictor is circular.",
    ),
    VariableSpec(
        name="parts_produced", label="Parts Produced", unit="count",
        causal_role="post_treatment_outcome",
        description="Parts produced in the interval.",
        valid_range=(10.0, 800.0), observed_range=(34.0, 458.0),
        notes="Denominator of the target.",
    ),
    VariableSpec(
        name="energy_kwh_interval", label="Energy Use per Interval", unit="kWh",
        causal_role="post_treatment_outcome",
        description="Energy consumed during the interval.",
        valid_range=(5.0, 28.0), observed_range=(9.45, 28.0),
    ),
    VariableSpec(
        name="defect_type", label="Dominant Defect Type",
        causal_role="post_treatment_outcome",
        description="Dominant defect class observed in the interval.",
        categories=(
            "burn_mark", "dimensional_deviation", "flash", "none", "short_shot",
            "sink_mark", "splay_moisture", "warpage",
        ),
        notes=(
            "Downstream of both the process settings and the same latent defect "
            "intensity that drives scrap. Never conditioned on."
        ),
    ),
    VariableSpec(
        name="pass_fail_flag", label="Pass / Fail Flag", unit="0/1",
        causal_role="post_treatment_outcome",
        description="Whether the interval passed the 3.2 % scrap threshold.",
        valid_range=(0.0, 1.0), observed_range=(0.0, 1.0),
        notes="A deterministic function of the target.",
    ),
)


# ── Adjustment sets ───────────────────────────────────────────────────────────
# Ported from datathon-CUB-2026/src/utils.py:80-113. Each set is intended to
# block every back-door path from the lever to scrap while containing no
# descendant of the lever.
#
# Levers with no entry here (shot_size_g) have no declared set in the source; the
# graph-derived fallback is used and labelled as such in the API response.

ADJUSTMENT_SETS: dict[str, tuple[str, ...]] = {
    "cooling_time_s": (
        "mold_temperature_c", "part_weight_g", "shot_size_g",
        "ambient_humidity_pct", "ambient_temperature_c",
        "maintenance_days_since_last",
    ),
    "mold_temperature_c": (
        "barrel_temperature_c", "part_weight_g", *_AMBIENT,
    ),
    "barrel_temperature_c": (
        "injection_pressure_bar", "mold_temperature_c",
        "resin_batch_quality_index", *_AMBIENT,
    ),
    "injection_pressure_bar": (
        "clamp_force_kn", "barrel_temperature_c", "resin_batch_quality_index",
        "part_weight_g", "hold_pressure_bar", *_AMBIENT,
    ),
    "hold_pressure_bar": (
        "injection_pressure_bar", *_AMBIENT,
    ),
    "dryer_dewpoint_c": _AMBIENT,
    "maintenance_days_since_last": _AMBIENT,
    "screw_speed_rpm": (
        "shot_size_g", "barrel_temperature_c", *_AMBIENT,
    ),
}

# Where the declared sets above depart from the source, and why. Surfaced on the
# affected estimate rather than left in a comment.
SOURCE_DEVIATIONS: dict[str, str] = {
    "mold_temperature_c": (
        "cooling_time_s was dropped from the source's adjustment set. The graph "
        "used here encodes mold_temperature_c -> cooling_time_s (operators extend "
        "cooling in response to observed mould temperature), which makes cooling "
        "time a descendant of this lever; adjusting for it would be "
        "post-treatment conditioning. The source declares both directions across "
        "its two sets, which no single acyclic graph can satisfy."
    ),
    "injection_pressure_bar": (
        "tool_wear_index was dropped from the source's adjustment set. It is a "
        "mediator in the same source's taxonomy, and this application enforces "
        "that no mediator enters a total-effect adjustment set. The source treats "
        "it here as a prior wear state instead; deciding between the two readings "
        "needs temporal ordering that 30-minute intervals cannot supply."
    ),
    "hold_pressure_bar": (
        "tool_wear_index was dropped from the source's adjustment set, for the "
        "same reason as injection_pressure_bar."
    ),
}


# ── Declared causal graph ─────────────────────────────────────────────────────
# A domain-informed assumption, not a discovered structure. It exists so that the
# adjustment sets above can be checked, and so the demo stops running on the
# role-template graph, which cannot express lever -> lever edges at all.

EDGES: tuple[tuple[str, str], ...] = (
    # Environment
    ("ambient_humidity_pct", "dryer_dewpoint_c"),
    ("ambient_humidity_pct", "resin_moisture_pct"),
    ("ambient_humidity_pct", "calibration_drift_index"),
    ("ambient_humidity_pct", "cooling_time_s"),
    ("ambient_humidity_pct", "scrap_rate_pct"),
    ("ambient_temperature_c", "dryer_dewpoint_c"),
    ("ambient_temperature_c", "resin_moisture_pct"),
    ("ambient_temperature_c", "calibration_drift_index"),
    ("ambient_temperature_c", "cooling_time_s"),
    ("ambient_temperature_c", "mold_temperature_c"),
    ("ambient_temperature_c", "scrap_rate_pct"),
    ("operator_shift", "calibration_drift_index"),
    ("operator_shift", "cooling_time_s"),
    ("operator_shift", "scrap_rate_pct"),
    ("operator_experience_level", "cooling_time_s"),
    ("operator_experience_level", "scrap_rate_pct"),
    ("resin_batch_quality_index", "resin_moisture_pct"),
    ("resin_batch_quality_index", "scrap_rate_pct"),
    # Design / tooling context
    ("product_variant", "cavity_count"),
    ("product_variant", "part_weight_g"),
    ("product_variant", "scrap_rate_pct"),
    ("cavity_count", "shot_size_g"),
    ("cavity_count", "clamp_force_kn"),
    ("cavity_count", "scrap_rate_pct"),
    ("part_weight_g", "shot_size_g"),
    ("part_weight_g", "clamp_force_kn"),
    ("part_weight_g", "cooling_time_s"),
    ("part_weight_g", "mold_temperature_c"),
    ("part_weight_g", "injection_pressure_bar"),
    ("part_weight_g", "scrap_rate_pct"),
    ("clamp_force_kn", "tool_wear_index"),
    ("clamp_force_kn", "scrap_rate_pct"),
    # Planning
    ("maintenance_days_since_last", "calibration_drift_index"),
    ("maintenance_days_since_last", "tool_wear_index"),
    ("maintenance_days_since_last", "cooling_time_s"),
    ("maintenance_days_since_last", "scrap_rate_pct"),
    # Levers. The only lever -> lever edges are the ones a mechanism supports.
    ("mold_temperature_c", "cooling_time_s"),   # the reactive-compensation edge
    ("shot_size_g", "cooling_time_s"),
    ("shot_size_g", "screw_speed_rpm"),
    ("dryer_dewpoint_c", "resin_moisture_pct"),  # acts only through moisture
    ("barrel_temperature_c", "scrap_rate_pct"),
    ("mold_temperature_c", "scrap_rate_pct"),
    ("injection_pressure_bar", "tool_wear_index"),
    ("injection_pressure_bar", "scrap_rate_pct"),
    ("hold_pressure_bar", "scrap_rate_pct"),
    ("screw_speed_rpm", "scrap_rate_pct"),
    ("shot_size_g", "scrap_rate_pct"),
    ("cooling_time_s", "scrap_rate_pct"),
    ("cooling_time_s", "cycle_time_s"),          # mechanical subsumption
    # Mediators
    ("resin_moisture_pct", "scrap_rate_pct"),
    ("calibration_drift_index", "scrap_rate_pct"),
    ("tool_wear_index", "scrap_rate_pct"),
)


DERIVED_RELATIONSHIPS: tuple[DerivedRelationship, ...] = (
    DerivedRelationship(
        target="shot_size_g",
        inputs=("cavity_count", "part_weight_g"),
        # Observed band is 0.972-1.214; widened by ~2 % on each side so the check
        # flags physical impossibility rather than ordinary sampling noise.
        ratio_lo=0.95,
        ratio_hi=1.24,
        description=(
            "Shot size must cover cavity_count x part_weight_g plus runner and "
            "cushion. Measured ratio across the demo is 0.972-1.214 (correlation "
            "0.9989). A shot below the cavity requirement is a short shot, which "
            "is itself one of the dataset's defect classes."
        ),
        max_violation_share=0.05,
    ),
)


ONTOLOGY = DatasetOntology(
    dataset_id=DATASET_ID,
    version=ONTOLOGY_VERSION,
    title="Injection Moulding — curated demo",
    target=TARGET,
    provenance=(
        "Causal roles and adjustment sets from datathon-CUB-2026/src/utils.py "
        "(stated to match paper section 2.1); labels, units and declared ranges "
        "from data/injection_molding_ontology.json; physical lever bounds from "
        "src/intervention_helpers.py; observed ranges measured on the 5,000-row "
        "demo CSV shipped with this application."
    ),
    graph_assumption=(
        "This graph is a domain-informed causal assumption taken from the source "
        "study's ontology and adjustment sets. It was not discovered from data "
        "and is not proven. Every effect estimate below is valid only if this "
        "graph is correct and there is no important unmeasured confounding."
    ),
    variables=VARIABLES,
    adjustment_sets=ADJUSTMENT_SETS,
    edges=EDGES,
    derived_relationships=DERIVED_RELATIONSHIPS,
)
