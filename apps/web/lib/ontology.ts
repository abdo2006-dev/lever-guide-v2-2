/**
 * Typed access to the curated demo ontology.
 *
 * The JSON in `lib/generated/` is produced from `apps/api/app/ontology/` by
 * `apps/api/scripts/export_ontology.py`. Python is the single source of truth for
 * causal roles; this file only gives the generated data a type. A backend test
 * (`tests/test_ontology.py::test_generated_json_is_in_sync`) fails if the two
 * ever disagree, so there is no second place to edit a role.
 */
import raw from "./generated/demo-ontology.json";

export type CausalRole =
  | "process_lever"
  | "planning_lever"
  | "confounder"
  | "mediator"
  | "context"
  | "operator_covariate"
  | "batch_covariate"
  | "identifier"
  | "outcome"
  | "post_treatment_outcome";

export type InterventionEligibility =
  | "eligible"
  | "derived_constrained"
  | "mediated_unsupported"
  | "preliminary"
  | "not_eligible";

export interface OntologyVariable {
  name: string;
  label: string;
  unit: string | null;
  description: string;
  causal_role: CausalRole;
  column_role: string;
  controllable: boolean;
  derived: boolean;
  mediator: boolean;
  valid_range: [number, number] | null;
  observed_range: [number, number] | null;
  categories: string[] | null;
  intervention_eligibility: InterventionEligibility;
  evidence_status: "conflicting" | "inconclusive" | null;
  role_source: string;
  notes: string;
}

export interface DatasetOntology {
  dataset_id: string;
  version: string;
  title: string;
  target: string;
  provenance: string;
  graph_assumption: string;
  variables: OntologyVariable[];
  column_roles: Record<string, string>;
  adjustment_sets: Record<string, string[]>;
  source_deviations: Record<string, string>;
  edges: { source: string; target: string }[];
  derived_relationships: {
    target: string;
    inputs: string[];
    ratio_lo: number;
    ratio_hi: number;
    description: string;
    max_violation_share: number;
  }[];
}

export const DEMO_ONTOLOGY = raw as unknown as DatasetOntology;

const BY_NAME = new Map(DEMO_ONTOLOGY.variables.map((v) => [v.name, v]));

/** Ontology entry for a column, or undefined when the dataset is not the demo. */
export function variableSpec(name: string): OntologyVariable | undefined {
  return BY_NAME.get(name);
}

/** Human-readable label with unit, falling back to the raw column name. */
export function variableLabel(name: string): string {
  const spec = BY_NAME.get(name);
  if (!spec) return name;
  return spec.unit ? `${spec.label} (${spec.unit})` : spec.label;
}
