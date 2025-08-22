
# tira_morph
Dataset of unique Tira sentences for purposes of training morphological segmentation.
Uses same textnorm steps as `tira_asr`. Contains 14285 unique sentences for
a total of 38750 words (12078 unique words) averaging
2.712635631781589 words per sentence. Of these, None sentences have
morphological decompositions, for None unique analyzed words and
None unique morphemes.

- 29007 non-NaN transcriptions in dataset
- 13148 non-duplicate transcriptions in dataset
- added 1929 rows from excel data
- removed 72 ungrammatical rows
- applied NFKD unicode normalization to text, set to lowercase and removed punctuation
- removed 439 rows with no tone marked, 14566 rows remaining
- removed 281 rows with English words, 14285 rows remaining
- saved all detected English words to $TIRA_ASR_CLIPS/english_words and Tira words to $TIRA_ASR_CLIPS/tira_words.txt
- removed tone words (e.g. HLL, LHL, LLHH) from transcription, 16 rows affected
- Checked that only expected IPA chars are found in dataset, as defined by JSON file tira_asr_unique_chars.json