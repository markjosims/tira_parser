# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**parC** (Paradigm Compiler) is a toolkit for building and applying morphological analyzers (finite-state-transducer-based parsers) from linguistic fieldwork data. The grammar model is entirely config-driven (YAML validated against JSON Schemas) and the Python side is written in a functional style — plain functions over immutable `NamedTuple`/dict data, no class hierarchies for the grammar/compilation logic.

## Commands

- Install deps: `uv sync` (`uv.lock` is canonical; `pyproject.toml` pins `pynini==2.1.6` and `requires-python = ">=3.8, <3.11"` — the checked-in `.venv` is 3.10, don't assume a newer interpreter works).
- Run the app: `uv run parC` (the `parC` console script → `src.api:run_app`), or directly `uv run uvicorn src.api:app --reload --port 8000`. `src/launcher.py` is stale/non-functional (broken import, checks a literal string instead of `YAML_DIR`) — don't use it as an entry point.
- Run tests: `YAML_DIR=yaml/spanish-example uv run pytest`. You must set `YAML_DIR` explicitly — `pyproject.toml` has a `[pytest]` `env_files = [".test.env"]` section intended to set `YAML_DIR=yaml/spanish-example` automatically, but the `pytest-env` plugin it depends on isn't installed, so it's a no-op. Without the explicit env var, `parC.env`'s `YAML_DIR=yaml/tira-example` wins by default and tests fail (they're written against the `spanish-example` dataset, e.g. rule name `diphthongization`, pattern `word_final_coda`).
- Run a single test: `YAML_DIR=yaml/spanish-example uv run pytest tests/transduction_test.py::test_suffix`.
- Which YAML dataset loads is controlled by `YAML_DIR` (`src/constants.py::get_yaml_dir`, falls back to `parC.env`, then to `yaml/spanish-example`). Two example datasets ship in-repo: `yaml/spanish-example` and `yaml/tira-example`.
- Logging: `PARC_LOG_LEVEL` (default `INFO`) and `TIRA_LOG_OUTPUT` (`stdout`/`stderr`) control `loguru` output (`src/__init__.py`).
- There's no dedicated CLI for bulk schema validation — validation happens automatically whenever config is read, via `get_yaml_data_safe`/`get_yaml_kind` (`src/yaml_utils/yaml_server.py`), which validate each file against `schemas/<Kind>.json` on load.

## Architecture

### Data lifecycle: YAML → validated dict → typed data → compiled FST → API → frontend

1. **Reading + validation** — `src/yaml_utils/yaml_server.py` reads YAML files under `YAML_DIR`, validates each against a hand-authored JSON Schema in `schemas/<Kind>.json` (`src/yaml_utils/schema_validation.py::validate_yaml`), and returns plain dicts. `CONFIG_KINDS` enumerates the active kinds; `FeatureCombinations`, `MorphemeSequence`, and `MorphemeSet` have schemas in `schemas/` but are commented out of `CONFIG_KINDS` as buggy/unimplemented — don't assume they work end-to-end (the morpheme-sequence UI is correspondingly commented out in `frontend/index.html`).
2. **Typed resolution** — dicts for rules/markers are resolved into `NamedTuple` domain types in `src/yaml_utils/models.py` (`Rule = SimpleRule | StringMapRule | RuleSequence`, `Marker = SingleStringMarker | StringTupleMarker | UnorderedMarker | PrincipalPartMarker`) via `resolve_rule`/`resolve_marker`. These pick a variant by trying each `NamedTuple` constructor in turn and catching failures, not via an explicit discriminator field for rules — field names can and do drift from the JSON Schema (e.g. the schema's `rule_sequence` vs. the `RuleSequence` NamedTuple's `rules` field), so check both sides when changing a rule/marker shape.
3. **FST compilation**, in dependency order:
   - `src/grammar/acceptor_compilation.py` — builds the `pynini.SymbolTable` from inventory phones/tags + feature values, special FSAs (sigma, phone, flag, boundary...), a token map for the pattern-string DSL, and compiles pattern strings (`fsa()`, `word_fsa()`) via a hand-written recursive-descent parser over the operators in `ReservedSymbolMixin` (`src/fst_utils.py`).
   - `src/grammar/transducer_compilation.py` — compiles `Rule`s into `pynini.cdrewrite` FSTs and `Marker`s into prefix/suffix/suppletion/replace/rule/string-map FSTs, built on `acceptor_compilation`'s `fsa`/symbol table.
   - `src/grammar/marker_resolution.py` — given a paradigm + feature-value combo, resolves which markers apply (multi-feature markers first, then regular feature markers for remaining features, then global/principal-part markers), including resolving `principal_part` markers into a string-map marker via the lexicon (`src/lexicon.py`).
   - `src/grammar/paradigm_compilation.py` — builds per-paradigm `inflect`/`parse`/`search_lexicon`/`search_left_factor` FSTs by applying resolved markers to every root × feature-combo, and exposes the public `inflect`/`parse`/`search`/`inflect_stages` functions consumed by the API.
4. `src/api.py` — FastAPI app exposing `grammar-stats`/`inflection-meta`/`roots`/`lexical-features`/`patterns`/`rules`/`test-pattern`/`test-rule`/`inflect`/`parse`/`search`, and mounts `frontend/` as static files at `/`.
5. `frontend/` — plain ES-module JS, no build step. `api.js` is the only file that calls `fetch`; `hub.js`/`inflect.js`/`parse.js`/`tests.js` drive the tab-based UI defined in `index.html`.

### Caching — two independent layers

- **In-memory**: `@observed_cache([dirs...])` (`src/yaml_utils/cache.py`) wraps `functools.lru_cache` and additionally invalidates whenever the max mtime across the given source directories advances; args/kwargs are coerced to hashable equivalents (`list`→`tuple`, `dict`→`frozendict`) before hitting the cache. Used throughout `acceptor_compilation.py`/`transducer_compilation.py`/`paradigm_compilation.py`.
- **On-disk**: the symbol table and per-paradigm FSTs also persist under `<YAML_DIR>/.cache/` (`symbol_table.syms`, `Paradigm/{name}.{fst_kind}.fst`), validated against source-file mtimes the same way (`is_syms_cache_valid`/`is_fst_cache_valid`).
- `tests/cache_invalidation_test.py` exercises both invalidation paths by writing to real YAML files under `YAML_DIR` and restoring them via fixtures — if a test run is interrupted, check `git status` under `yaml/` for leftover mutations.

### Config directory taxonomy

Per-language config lives under `yaml/<language>-example/<ParDir>/<Kind>/*.yaml`, e.g. `yaml/spanish-example/Phonology/Rules/vowel_alternations.yaml`. `CONFIG_KIND_TO_PARDIR` (`src/yaml_utils/schema_validation.py`) maps each kind to its parent dir: `Inventory`/`Patterns`/`Rules` → `Phonology`; `FeatureDefinitions`/`FeatureMarkers`/`ContingentFeatureMarkers` → `Exponence`; `Paradigm` → `Morphotactics`; `PartOfSpeech`/`Wordlists` → `Lexicon`.

Lexicon word lists (`Lexicon/Wordlists/*.csv`/`.xlsx`) are the one config surface *not* validated against a JSON Schema — `src/lexicon.py` reads them directly via pandas, cross-referencing `lexical_features`/`principal_parts` declared in the corresponding `PartOfSpeech` YAML, and auto-creates an empty CSV with the right columns if the wordlist is missing.

### Pattern-string DSL

Inventory classes, Patterns, and morpheme/rule contexts all share one regex-like DSL: `<ClassName>` references an inventory class or named pattern, `|` is disjunction, `{A B}` is union of literal tokens, `*`/`+`/`?` are closures, `^` negates inside `{}`. Reserved operators/symbols live in `ReservedSymbolMixin` (`src/fst_utils.py`). See `doc/grammar_modules.rst` for the linguistic rationale behind the phonology/exponence/morphotactics module split.

### `src/search/` — in-progress replacement for the paradigm-compilation search path

`src/search/` (`beam_search.py`, `beam_search_jit.py`, `edit_graph.py`, `edit_modeling.py`) implements a numba-jitted beam search over the compiled FSTs, intended to eventually replace the search path in `paradigm_compilation.py`. It is not yet wired into `src/api.py` — the live fuzzy-search path today is `build_search_lexicon_and_leftfactor`/`search` in `src/grammar/paradigm_compilation.py`. `beam_search_jit_stale.py` is a superseded variant of `beam_search_jit.py` — check before using either.
