"""
WIP: Script that loads lexical data from .csv files and compiles FSTs mapping
stems to glosses and to principal parts.
"""

import pynini
import pandas as pd
from constants import VERB_ROOTS_PATH, ROOT2FV_FST_PATH, ROOT2GLOSS_FST_PATH

def get_root2gloss_fst() -> pynini.Fst:
    verbs_df = pd.read_csv(VERB_ROOTS_PATH)
    root2gloss_strs = [list(t) for t in zip(
        verbs_df['verb_root'].tolist(),
        verbs_df['root_sense'].tolist())
    ]
    root2gloss = pynini.string_map(root2gloss_strs)
    return root2gloss

def get_root2fv_fst() -> pynini.Fst:
    verbs_df = pd.read_csv(VERB_ROOTS_PATH)
    root2fv_strs = [list(t) for t in zip(
        verbs_df['verb_root'].tolist(),
        verbs_df['root_fv'].tolist())
    ]
    root2fv = pynini.string_map(root2fv_strs)
    return root2fv

def main() -> int:
    root2gloss = get_root2gloss_fst()
    root2gloss.write(ROOT2GLOSS_FST_PATH)

    root2fv = get_root2fv_fst()
    root2fv.write(ROOT2FV_FST_PATH)

    return 0

if __name__ == '__main__':
    main()