"""
TODO: Load lexical data from Excel spreadsheets and save as `.csv` files
containing data for a lexeme's root, principal parts/inflection class, and gloss.
"""

import pandas as pd
import os
from dataset_builder import DATA_DIR, EXCEL_VERBS_PATH, EXCEL_SHEET_NAME

ROOTS_OUTPATH = os.path.join(DATA_DIR, "verb_roots.csv")
ROOTS_INPATH = os.path.join(DATA_DIR, "verb_roots_new.csv")

def get_roots_from_excel() -> int:
    """
    Creates a `.csv` file from Excel data containing a single row for each unique verb root
    along with all senses and inflection classes associated with that root.
    """
    df = pd.read_excel(EXCEL_VERBS_PATH, sheet_name=EXCEL_SHEET_NAME)
    get_translation_and_class = lambda row: f"{row['Translation']} ({row['Inflection class']})"
    df['sense']=df.apply(get_translation_and_class, axis=1)
    group_by_root = df.groupby(by='Root')
    root2unique_senses = group_by_root['sense'].apply(set).apply("; ".join)
    root2unique_senses.to_csv(ROOTS_OUTPATH, index_label='root')

    return 0

if __name__ == '__main__':
    get_roots_from_excel()