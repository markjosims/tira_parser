"""
# models.py
Structs for grammar data objects. Each YAML file type has a
struct defining the schema for the entire file, which in turn
is comprised of structs for data objects defined by the file
(e.g. the PatternFile struct has Pattern as a sub-struct).
"""

from __future__ import annotations

from typing import Annotated, Literal, NamedTuple, get_args

import msgspec



"""
Shared models
"""

ObjectId = Annotated[
    str,
    msgspec.Meta(
        pattern=r"^<[^>]+>$",
        description="A unique identifier for an object, circumfixed with angle brackets.",
    ),
]

ObjectRef = Annotated[
    str,
    msgspec.Meta(
        pattern=r"^\$.+$",
        description="A reference to an object defined in another file, prefixed with a dollar sign.",
    ),
]

NonObjectRef = Annotated[
    str,
    msgspec.Meta(
        pattern=r"^[^\$].*$",
        description="A string that is not an object reference (does not start with a dollar sign).",
    ),
]

"""
## Phonology modules

Contains the following sublcasses:
- Inventory
- Patterns
- Rules
"""

_reserved_symbols = r".+*?{}[]()<>"
_reserved_symbols_escaped = r"\.\+\*\?\{\}\[\]\(\)<>"


# tags must be circumfixed with square brackets
# no reserved symbols allowed inside square brackets
Tag = Annotated[
    str, msgspec.Meta(pattern=r"^\[[^" + _reserved_symbols_escaped + r"]+\]$")
]


# phones cannot contain any of the reserved symbols, viz: .+*?{}[]()<>
Phone = Annotated[
    str, msgspec.Meta(pattern=r"^[^" + _reserved_symbols_escaped + r"]+$")
]


class PhonesNode(
    msgspec.Struct, kw_only=True, frozen=True, tag_field="kind", tag="phones"
):
    """
    A single inventory node containing an array of phones.
    Phones are strings of one or more characters excluding
    the reserved symbols: .+*?{}[]()<>.
    """

    id: ObjectId
    name: str | None = None
    data: tuple[Phone, ...]


class TagsNode(msgspec.Struct, kw_only=True, frozen=True, tag_field="kind", tag="tags"):
    """
    A single inventory node containing an array of tags
    where each tag is a string circumfixed with [square brackets].
    No other reserved symbols are allowed inside the square brackets.
    """

    id: ObjectId
    name: str | None = None
    data: tuple[Tag, ...]


class NestedNode(
    msgspec.Struct, kw_only=True, frozen=True, tag_field="kind", tag="nested"
):
    """
    A single inventory node containing an array of nested
    inventory nodes.
    """

    id: ObjectId
    name: str | None = None
    data: tuple[InventoryNode, ...]


InventoryNode = PhonesNode | TagsNode | NestedNode


class InventoryFile(msgspec.Struct, kw_only=True, tag_field="kind", tag="Inventory"):
    """
    A file containing an inventory of phones, tags, and nested nodes.
    """

    data: tuple[InventoryNode, ...]
    source_path: str | None = None


class Pattern(msgspec.Struct, kw_only=True, frozen=True):
    """
    A regular expression pattern and, optionally, 'include' and 'exclude'
    strings for unit testing.

    The pattern string must conform to the grammar of parC-flavored Regex,
    handled by `src.grammar.acceptor_compilation.py` and must only contain
    phones and tags described in the Inventory, or reserved symbols.

    No string validation is handled here, and is instead left to
    `acceptor_compilation.py`.
    """

    pattern: str
    test_includes: tuple[str] | None = None
    test_excludes: tuple[str] | None = None
    name: str | None = None


class PatternFile(msgspec.Struct, kw_only=True, tag_field="kind", tag="Pattern"):
    """
    A file containing a list of patterns.
    """

    data: tuple[Pattern, ...]
    source_path: str | None = None


class Token(NamedTuple):
    value: str
    kind: Literal[
        "phone",
        "tag",
        "class_ref",
        "pattern_ref",
        "bow_eow",
        "edit_flag",
        "special_ref",
        "unary_operator",
        "pipe_operator",
        "caret_operator",
        "boundary",
        "left_delimiter",
        "right_delimiter",
    ]

    def __len__(self) -> int:
        return len(self.value)


"""
## Rules modules
"""


class SimpleRule(
    msgspec.Struct, kw_only=True, frozen=True, tag_field="kind", tag="simple"
):
    """
    A context-sensitive rewrite rule.
    """

    name: str
    input_pattern: str | None
    output_pattern: str | None
    description: str = ""
    left_context: str = ""
    right_context: str = ""


class StringMapRule(
    msgspec.Struct, kw_only=True, frozen=True, tag_field="kind", tag="string_map"
):
    """
    A rule for mapping strings.
    """

    name: str
    string_map: tuple[tuple[str, str], ...]
    description: str = ""
    left_context: str = ""
    right_context: str = ""


class RuleSequence(
    msgspec.Struct, kw_only=True, frozen=True, tag_field="kind", tag="rule_sequence"
):
    """
    A sequence of rules to be applied.
    Here just stored as a list of strings indicating rule names,
    which are resolved to rule data up in `fst_compilation.compile_rule`
    """

    name: str
    rules: tuple[ObjectRef, ...]
    description: str = ""


# A mapping of rule names to their corresponding rule objects.
# `kind` (declared via tag_field/tag above) discriminates the union on decode
Rule = SimpleRule | StringMapRule | RuleSequence


class RulesFile(msgspec.Struct, kw_only=True, tag_field="kind", tag="Rules"):
    rules: tuple[Rule, ...]
    source_path: str | None = None


"""
## Exponence modules

Contains the following submodules:
- FeatureDefinitions
- FeatureMarkers
- MultiFeatureMarkers
"""


class Feature(msgspec.Struct, kw_only=True, frozen=True):
    """
    A feature with a name and a list of possible values.
    """

    name: str
    values: tuple[str, ...]


class FeatureDefinitionsFile(
    msgspec.Struct, kw_only=True, tag_field="kind", tag="FeatureDefinitions"
):
    """
    A file containing a list of feature definitions.
    """

    data: tuple[Feature, ...]
    source_path: str | None = None


class PrefixMarker(
    msgspec.Struct, kw_only=True, frozen=True, tag_field="kind", tag="prefix"
):
    """
    A marker for a prefix operation.
    """

    value: str
    stage: str | None = None


class SuffixMarker(
    msgspec.Struct, kw_only=True, frozen=True, tag_field="kind", tag="suffix"
):
    """
    A marker for a suffix operation.
    """

    value: str
    stage: str | None = None


class SuppletionMarker(
    msgspec.Struct, kw_only=True, frozen=True, tag_field="kind", tag="suppletion"
):
    """
    A marker for a suppletion operation.
    """

    value: str
    stage: str | None = None


class RuleMarker(
    msgspec.Struct, kw_only=True, frozen=True, tag_field="kind", tag="rule"
):
    """
    A marker that applies a contextual rule.
    """

    value: str
    stage: str | None = None


class ReplaceMarker(
    msgspec.Struct, kw_only=True, frozen=True, tag_field="kind", tag="replace"
):
    """
    A marker that applies a context-insensitive A->B replace rule.
    """

    value: tuple[str, str]
    stage: str | None = None


class PrincipalPartMarker(
    msgspec.Struct, kw_only=True, frozen=True, tag_field="kind", tag="principal_part"
):
    """
    A marker that selects a principal part for the given lexeme.
    """

    value: str


Marker = (
    PrefixMarker
    | SuffixMarker
    | SuppletionMarker
    | PrincipalPartMarker
    | RuleMarker
    | ReplaceMarker
)


class FeatureMarker(msgspec.Struct, kw_only=True, frozen=True):
    """
    A feature value paired with a tuple of markers which expone the feature value.
    """

    feature_value: str
    markers: tuple[Marker, ...]


class MultiFeatureMarker(msgspec.Struct, kw_only=True, frozen=True):
    """
    A collection of feature, value pairs exponed by a single tuple of Markers.
    """

    feature_values: dict[str, str]
    markers: tuple[Marker, ...]


class FeatureMarkerFile(
    msgspec.Struct, kw_only=True, tag_field="kind", tag="FeatureMarkers"
):
    """
    A file containing a list of feature markers.
    """

    data: tuple[FeatureMarker, ...]
    feature: str
    source_path: str | None = None


class MultiFeatureMarkerFile(
    msgspec.Struct, kw_only=True, tag_field="kind", tag="MultiFeatureMarkers"
):
    """
    A file containing a list of multi-feature markers.
    """

    data: tuple[MultiFeatureMarker, ...]
    source_path: str | None = None


"""
## Morphotactics modules

Contains the following submodules:
- FeatureCombinations
- Paradigm
"""

FeatureCombination = Annotated[
    dict[str, tuple[str] | Literal["*", "undefined"]],
    msgspec.Meta(
        description="Object mapping feature values to an array of strings indicating "
        + "possible values that feature may take on in the given combination, or a wildcard "
        + '"*" to indicate the feature may take on any value, or "undefined" to indicate '
        + "the feature must be undefined in this combination."
    ),
]


class FeatureCombinationsFile(
    msgspec.Struct,
    kw_only=True,
    frozen=True,
    tag_field="kind",
    tag="FeatureCombinations",
):
    """
    A file specifying a set of licit feature vectors for a given
    feature set.
    """

    part_of_speech: ObjectRef
    data: tuple[FeatureCombination, ...]


class ParadigmFilter(msgspec.Struct, kw_only=True, frozen=True):
    """
    A filter the selects lexical roots based on whether they possess
    certain lexical features or whether they match a given regex pattern.
    """

    lexical_features: dict[str, str] | None = None
    pattern: str | None = None


class ParadigmFile(msgspec.Struct, kw_only=True, tag_field="kind", tag="Paradigm"):
    """
    A file defining a paradigm (or partial paradigm) for a given part of speech.
    The `part_of_speech` and `feature_markers` fields are obligatory. `part_of_speech`
    must contain a reference to a PartOfSpeech config file, and `feature_markers` is a
    dictionary which must map every inflectional feature for the given part of speech to
    either (1) a reference to a `FeatureMarkers` config, (2) a str indicating a fixed
    feature value or (3) `None`, indicating that the feature is only exponed by
    `MultiFeatureMarkers` configs.

    Optional attributes include:
    - `stage_order`:    An array of names that defines the order staged operations are
                        applied in.
    - `global_markers`: An array of `Marker` objects that apply to all paradigm cells
                        (regardless of feature values)
    - `feature_value_combinations`: An object reference to a `FeatureValueCombinations`
                                    config that constrains what feature vectors are licit
                                    for the given paradigm.
    - `multifeature_markers`:       An array of object references pointing to a set of
                                    `MultiFeatureMarker` configs which expone inflection
                                    features for the current paradigm.
    """

    part_of_speech: ObjectRef
    filter: ParadigmFilter | None = None
    feature_markers: dict[str, str | None] | None = None
    stage_order: tuple[str, ...] | None = None
    global_markers: tuple[Marker, ...] | None = None
    feature_value_combinations: ObjectRef | None = None
    multifeature_markers: tuple[ObjectRef, ...] | None = None

    source_path: str | None = None


"""
## Lexicon modules

Contains the following submodules:
- WordList (no struct defined here, just a CSV file)
- PartOfSpeech
"""


class PartOfSpeechFile(
    msgspec.Struct, kw_only=True, tag_field="kind", tag="PartOfSpeech"
):
    """
    A file defining a part of speech and its associated inflectional features.
    The `name` and `inflectional_features` fields are obligatory.
    Contains the following optional attributes:
    - `lexical_features`:   An array of strings indicating the lexical features
                            associated with the part of speech.
    - `principal_parts`:    Column names in the lexicon CSV that specify alternate stems
                            for a root (e.g., present_stem, past_stem)
    """

    name: str
    inflectional_features: tuple[str, ...]
    lexical_features: tuple[str, ...] | None = None
    principal_parts: tuple[str, ...] | None = None
    source_path: str | None = None


"""
## Global union over grammar files
"""

GrammarFile = (
    InventoryFile
    | PatternFile
    | PartOfSpeechFile
    | RulesFile
    | FeatureDefinitionsFile
    | FeatureMarkerFile
    | MultiFeatureMarkerFile
    | FeatureCombinationsFile
    | ParadigmFile
)

ConfigKindType = Literal[
    # TODO: FeatureCombinations, MorphemeSet and MorphemeSequence are buggy
    # so they are commented out for now
    "ContingentFeatureMarkers",
    "FeatureCombinations",
    "FeatureDefinitions",
    "FeatureMarkers",
    "Inventory",
    # "MorphemeSet",
    "Paradigm",
    "PartOfSpeech",
    "Patterns",
    "Rules",
]
CONFIG_KINDS: tuple[str, ...] = get_args(ConfigKindType)

CONFIG_KIND_TO_STRUCT: dict[str, msgspec.Struct] = {
# TODO
}


CONFIG_KIND_TO_PARDIR = {
    "ContingentFeatureMarkers": "Exponence",
    "FeatureDefinitions": "Exponence",
    "FeatureMarkers": "Exponence",
    "Inventory": "Phonology",
    "Rules": "Phonology",
    "Patterns": "Phonology",
    "Paradigm": "Morphotactics",
    "FeatureCombinations": "Morphotactics",
    "PartOfSpeech": "Lexicon",
    "Wordlists": "Lexicon",
}