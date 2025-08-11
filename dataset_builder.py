from typing import *
from tira_elan_scraper import LIST_PATH
import os
import pandas as pd
import string
import json
import unicodedata
import wordfreq

AUDIO_DIR = os.environ.get("TIRA_ELICITATION_WAVS")
TIRA_ASR_CLIPS_DIR = os.environ.get("TIRA_ASR_CLIPS")
TIRA_ASR_PYARROW_DIR = os.environ.get("TIRA_ASR_PYARROW")
VERSION = "0.1.0"

README_HEADER = string.Template(
"""
# tira_morph
Dataset of unique Tira sentences for purposes of training morphological segmentation.
Uses same textnorm steps as `tira_asr`. Contains $num_sentences unique sentences for
a total of $num_words words ($num_word_unique unique words) averaging
$mean_sentence_len words per sentence. Of these, $num_analyses sentences have
morphological decompositions, for $num_word_analyzed unique analyzed words and
$num_morphs unique morphemes.
"""
)
PREPROCESSING_STEPS = []

# -------------- #
# string helpers #
# -------------- #

TONE_DIACS = ['grave', 'macrn', 'acute', 'circm', 'caron',]

COMBINING = {
    'grave': "\u0300",
    'macrn': "\u0304",
    'acute': "\u0301",
    'circm': "\u0302",
    'caron': "\u030C",
    'tilde': "\u0303",
    'bridge': "\u032A",
}

def strip_punct(f):
    def g(s):
        return f(s.strip(string.punctuation))
    return g


def remove_punct(text: str, keep: Optional[Union[str, Sequence[str]]] = None) -> str:
    for p in string.punctuation:
        if keep and p in keep:
            continue
        text = text.replace(p, '')
    return text

def unicode_normalize(
        text: str,
        unicode_format: Literal['NFC', 'NFKC', 'NFD', 'NFKD'] = 'NFKD',
    ) -> str:
    """
    wraps unicodedata.normalize with default format set to NFKD
    """
    return unicodedata.normalize(unicode_format, text)

def unicode_description(char: str):
    unicode_name = unicodedata.name(char, 'No unicode name found')
    unicode_point = str(hex(ord(char)))
    return {
        'unicode_name': unicode_name,
        'unicode_point': unicode_point,
    }

def has_diac(text: str, tone_only: bool = False) -> str:
    text = unicode_normalize(text)
    for diac_name, diac in COMBINING.items():
        if tone_only and diac_name not in TONE_DIACS:
            continue
        if diac in text:
            return True
    return False

@strip_punct
def has_unicode(s):
    return unidecode(s) != s
@strip_punct
def is_en_word(w: str, expect_ascii: bool = True, threshold=1e-9) -> bool:
    """
    Returns True if a word is detected as English, False otherwise.
    Default behavior is to return True if word is recognized by `wordfreq`.
    Frequency must be greater than `threshold` value to prevent ultra
    low-frequency words from slipping through. If `expect_ascii=True`,
    return False if non-ascii unicode characters are detected, regardless
    of `wordfreq` search.
    """
    if expect_ascii and has_unicode(w):
        return False
    return wordfreq.word_frequency(w, 'en')>threshold

def max_ord_in_str(text: str) -> int:
    return max(ord(c) for c in text)

def make_replacements(text: str, reps: Dict[str, str]) -> str:
    """
    Makes all replacements specified by `reps`, a dict whose keys are intabs
    and values are outtabs to replace them.
    Avoids transitivity by first replacing intabs to a unique char not found in the original string.
    """
    max_ord_str = max_ord_in_str(text)
    max_ord_reps = max_ord_in_str(''.join([*reps.keys(), *reps.values()]))
    max_ord = max(max_ord_str, max_ord_reps)
    intab2unique = {
        k: chr(max_ord+i+1) for i, k in enumerate(reps.keys())
    }
    unique2outtab = {
        intab2unique[k]: v for k, v in reps.items()
    }

    # sort intabs so that longest sequences come first
    intabs = sorted(reps.keys(), key=len, reverse=True)

    for intab in intabs:
        sentinel = intab2unique[intab]
        text = text.replace(intab, sentinel)
    for sentinel, outtab in unique2outtab.items():
        text = text.replace(sentinel, outtab)
    return text

# --------------------------- #
# text normalization pipeline #
# --------------------------- #

def perform_textnorm(
        df: pd.DataFrame,
        preproc_steps: List[str],
        norm_col: str = 'text',
        keep_punct: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:

    # drop ungrammatical rows
    ungrammatical_mask = df[norm_col].str.contains('*', regex=False)
    df = df[~ungrammatical_mask]
    ungrammatical_str = f"- removed {int(ungrammatical_mask.sum())} ungrammatical rows"
    print(ungrammatical_str)
    preproc_steps.append(ungrammatical_str)

    # basic string normalization
    print("String normalization...")
    df[norm_col] = df[norm_col].apply(unicode_normalize)
    df[norm_col] = df[norm_col].str.lower()
    df[norm_col] = df[norm_col].apply(lambda s: remove_punct(s, keep=keep_punct))
    nfkd_str = f"- applied NFKD unicode normalization to text, set to lowercase and removed punctuation"
    print(nfkd_str)
    preproc_steps.append(nfkd_str)

    # skip all toneless entries
    print("Dropping rows with no tone diacritics")
    has_tone_mask = df[norm_col].apply(lambda s: has_diac(s, tone_only=True))
    prev_len = len(df)
    df = df[has_tone_mask]
    toneless_row_num = prev_len-len(df)
    toneless_str = f"- removed {toneless_row_num} rows with no tone marked, {len(df)} rows remaining, {get_df_duration(df)}"
    print(toneless_str)
    preproc_steps.append(toneless_str)


    # remove all rows with English words
    print("Removing rows with English")
    en_words = set()
    tira_words = set()
    def detect_en_words(sentence):
        has_en_word = False
        for word in sentence.split():
            if is_en_word(word) and (len(word)>1) or word in ['downstep']:
                en_words.add(word)
                has_en_word=True
            else:
                tira_words.add(word)
        return has_en_word

    has_en_mask = df[norm_col].apply(detect_en_words)

    # save detected words for manual verification
    en_words_path = os.path.join(TIRA_ASR_CLIPS_DIR, "english_words.txt")
    tira_words_path = os.path.join(TIRA_ASR_CLIPS_DIR, "tira_words.txt")
    with open(en_words_path, 'w', encoding='utf8') as f:
        f.writelines(['\n'.join(en_words)])
    with open(tira_words_path, 'w', encoding='utf8') as f:
        f.writelines(['\n'.join(tira_words)])

    prev_len = len(df)
    df = df[~has_en_mask]
    en_row_num = prev_len-len(df)
    no_en_str = f"- removed {en_row_num} rows with English words, {len(df)} rows remaining, {get_df_duration(df)}"
    unique_words_str = "- saved all detected English words to $TIRA_ASR_CLIPS/english_words "+\
        "and Tira words to $TIRA_ASR_CLIPS/tira_words.txt"
    print(no_en_str)
    print(unique_words_str)
    preproc_steps.append(no_en_str)
    preproc_steps.append(unique_words_str)

    # remove tone words (e.g. HLL, LHL,...)
    print("Removing tone words from transcriptions...")
    is_tone_word = lambda s: all(c in 'hml' for c in s)
    has_tone_word = lambda s: any(is_tone_word(w) for w in s.split())
    remove_tone_word = lambda s: ' '.join(word for word in s.split() if not is_tone_word(word))
    has_tone_word_mask = df[norm_col].apply(has_tone_word)
    df[norm_col] = df[norm_col].apply(remove_tone_word)
    remove_tone_word_str = f"- removed tone words (e.g. HLL, LHL, LLHH) from transcription, {int(has_tone_word_mask.sum())} rows affected"
    print(remove_tone_word_str)
    preproc_steps.append(remove_tone_word_str)

    # normalize IPA charset
    print("Normalizing IPA character set...")
    char_rep_json_path = os.path.join(TIRA_ASR_CLIPS_DIR, 'char_replacements.json')
    # # Uncomment to overwrite `char_rep_json`
    # unique_chars = set()
    # df[norm_col].apply(unique_chars.update)
    # rep_dict = {
    #     char: {
    #         'target': char,
    #         'comment': '',
    #         **unicode_description(char)
    #     } for char in unique_chars
    # }
    # with open(char_rep_json_path, 'w', encoding='utf8') as f:
    #     json.dump(rep_dict, f, ensure_ascii=True, indent=2)
    with open(char_rep_json_path, encoding='utf8') as f:
        rep_dict = json.load(f)
    rep_dict = {k: v['target'] for k, v in rep_dict.items()}
    normalize_ipa = lambda s: make_replacements(s, rep_dict)
    # apply twice since some diacritics may interfere with replacing digraphs
    df[norm_col]=df[norm_col].apply(normalize_ipa)
    df[norm_col]=df[norm_col].apply(normalize_ipa)

    print("Checking only expected chars are found in dataset...")
    expected_chars_basename = 'tira_asr_unique_chars.json'
    expected_chars_path = os.path.join('meta', expected_chars_basename)
    with open(expected_chars_path, encoding='utf8') as f:
        expected_ipa_chars = json.load(f)
    if keep_punct:
        expected_ipa_chars.extend([p for p in keep_punct])
    unexpected_chars = set()
    def find_unexpected_chars(sentence):
        found_unexpected_char = False
        for c in sentence:
            if c not in expected_ipa_chars:
                unexpected_chars.add(c)
                found_unexpected_char=True
        return found_unexpected_char

    unexpected_chars_mask = df[norm_col].apply(find_unexpected_chars)
    if (unexpected_chars_mask).sum()>0:
        raise ValueError(f"Found unexpected chars after normalizing IPA. Inspect {norm_col} col in dataframe.")
    expected_char_str = "- Checked that only expected IPA chars are found in dataset, "+\
        f"as defined by JSON file {expected_chars_basename}"
    print(expected_char_str)
    preproc_steps.append(expected_char_str)

    return df, preproc_steps

def main() -> int:
    df = pd.read_csv(LIST_PATH)
    print(len(df))

    # only interested in 'IPA Transcription', no other tiers
    print("Dropping non-transcription annotations...")
    ipa_mask = df['tier'] == 'IPA Transcription'
    df=df[ipa_mask]
    df=df.drop(columns=['tier'])
    print(len(df))

    # drop na rows
    df=df.dropna()
    nan_str = f"- {len(df)} non-NaN transcriptions in dataset"
    print(nan_str)
    PREPROCESSING_STEPS.append(nan_str)
    
    df, _ = perform_textnorm(df, PREPROCESSING_STEPS)

    readme_header_str = README_HEADER.substitute(
        num_records=len(df),
    )
    readme_out = os.path.join(TIRA_ASR_CLIPS_DIR, 'README.md')
    with open(readme_out, 'w', encoding='utf8') as f:
        f.write(readme_header_str+'\n')
        f.write('\n'.join(PREPROCESSING_STEPS))

    transcriptions_path = os.path.join(TIRA_ASR_CLIPS_DIR, 'transcriptions.csv')
    df.to_csv(transcriptions_path, index_label='index')

if __name__ == '__main__':
    main()