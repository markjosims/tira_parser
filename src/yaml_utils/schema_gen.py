"""
Generates schemas/<Kind>.json files from msgspec type definitions.

Prototype: only the Rules kind is migrated to this approach so far. The
generated schema is the same source of truth used to decode/validate Rule
data at runtime (`src.yaml_utils.models.resolve_rule`), so the two can no
longer drift apart the way a hand-authored schema.json can from its
NamedTuple counterpart.

Run with: `uv run python -m src.yaml_utils.schema_gen`
"""

import json
from pathlib import Path

import msgspec

from src.constants import SCHEMA_DIR
from src.models import Rule, RulesFile


def generate_rules_schema() -> dict:
    return msgspec.json.schema(RulesFile)


def main() -> None:
    schema = generate_rules_schema()
    out_path = Path(SCHEMA_DIR) / "Rules.json"
    out_path.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
