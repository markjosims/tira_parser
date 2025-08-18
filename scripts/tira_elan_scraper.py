"""
Reads .eaf files from `TIRA_RECORDINGS_GDRIVE` and makes a list stored in `meta/tira_elan_raw.csv`
with file basenames, timestamps, Elan tier names and values.
"""

import os
from glob import glob
from pympi import Eaf
from tqdm import tqdm
import pandas as pd

TIRA_RECORDINGS_GDRIVE = os.environ.get("TIRA_RECORDINGS_GDRIVE")
DATA_DIR = os.environ.get(
    'TIRA_MORPH_DATA_DIR',
    os.path.join(os.getcwd(), 'data')
)
LIST_PATH = os.path.join(
    DATA_DIR,
    'tira_elan_raw.csv'
)


def main() -> int:
    rows = []

    eaf_paths = glob(os.path.join(TIRA_RECORDINGS_GDRIVE, "**", "*.eaf"))
    for eaf_path in tqdm(eaf_paths):
        eaf_basename = os.path.basename(eaf_path)
        wav_basename = eaf_basename.replace('.eaf', '.wav')
        eaf = Eaf(eaf_path)
        for tier in eaf.get_tier_names():
            for annotation in eaf.get_annotation_data_for_tier(tier):
                start, end, val = annotation[:3]
                rows.append({
                    'audio_basename': wav_basename,
                    'start': start,
                    'end': end,
                    'duration': end-start,
                    'text': val,
                    'eaf_basename': eaf_basename,
                    'tier': tier,
                })
    df = pd.DataFrame(rows)
    df.to_csv(LIST_PATH, index=False, encoding='utf8')
    print(f"Data saved to {LIST_PATH}")
    return 0

if __name__ == "__main__":
    main()