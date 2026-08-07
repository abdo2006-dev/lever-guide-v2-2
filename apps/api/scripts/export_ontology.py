#!/usr/bin/env python
"""
Regenerate the frontend's copy of the demo ontology.

    cd apps/api && ./.venv/bin/python scripts/export_ontology.py

The Python ontology is authoritative; this writes the derived JSON the frontend
imports. `tests/test_ontology.py::test_generated_json_is_in_sync` fails if the
committed JSON drifts from the Python source, so regenerating is not optional
after an ontology change.
"""
from __future__ import annotations

import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
sys.path.insert(0, str(API_ROOT))

from app.ontology import (  # noqa: E402
    GENERATED_JSON_PATH,
    INJECTION_MOULDING_ONTOLOGY,
    ontology_json,
    validate_ontology,
)


def main() -> int:
    problems = validate_ontology(INJECTION_MOULDING_ONTOLOGY)
    if problems:
        print("Ontology is inconsistent; refusing to export:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    out = REPO_ROOT / GENERATED_JSON_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ontology_json(INJECTION_MOULDING_ONTOLOGY), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
