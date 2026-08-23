from __future__ import annotations
from typing import Annotated
import msgspec

Ref = Annotated[str, msgspec.Meta(pattern=r'^<[^>]+>$')]
Tag = Annotated[str, msgspec.Meta(pattern=r'^\[[^\]]+\]$')]

class InventoryNode(msgspec.Struct, kw_only=True, frozen=True):
    ref: Ref
    name: str | None = None
    phones: tuple[str, ...] = ()
    tags: tuple[Tag, ...] = ()
    children: tuple[InventoryNode, ...] = ()

good = {'ref': '<C>', 'tags': ['[TBU]']}
print(msgspec.convert(good, type=InventoryNode))

bad = {'ref': 'C'}  # missing angle brackets
try:
    msgspec.convert(bad, type=InventoryNode)
except Exception as e:
    print('rejected bad ref:', type(e).__name__, e)

bad_tag = {'ref': '<C>', 'tags': ['TBU']}  # missing brackets
try:
    msgspec.convert(bad_tag, type=InventoryNode)
except Exception as e:
    print('rejected bad tag:', type(e).__name__, e)

import json
print(json.dumps(msgspec.json.schema(InventoryNode)['$defs']['InventoryNode']['properties']['ref'], indent=2))
breakpoint()